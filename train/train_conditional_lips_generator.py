import torch

import options
from utils import train_utils, files_utils, image_utils
from custom_types import *
import constants
from dataloader import prcoess_face_forensics
from options import OptionsLipsGenerator
from models import models_utils, lips_detection_model
import lpips


class TrainLipsGenerator:

    def init_dataloader(self):
        # self.opt.data_dir = constants.MNT_ROOT + 'video_frames/obama/'
        ds = prcoess_face_forensics.LipsConditionedDS(self.opt)

        split_path = f"{self.opt.data_dir}/split.pkl"
        split = files_utils.load_pickle(split_path)

        if split is None:
            split = torch.rand(len(ds)).argsort()
            split_val, split_train = split[:int(len(split) * .1)], split[int(len(split) * .1):]
            files_utils.save_pickle({'split_val': split_val, 'split_train': split_train}, split_path)
        else:
            split_val, split_train = split['split_val'], split['split_train']
        ds_train, ds_val = Subset(ds, split_train), Subset(ds, split_val)
        dl_train = DataLoader(ds_train, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
                              batch_size=8 if DEBUG else self.opt.batch_size)
        dl_val = DataLoader(ds_val, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
                              batch_size=self.opt.batch_size)
        return ds, dl_train, dl_val

    def between_epochs(self, epoch, scores):
        if scores['loss'] < self.best_score:
            self.best_score = scores['loss']
            self.model.save()
        if epoch > self.offset and (epoch - self.offset) % self.opt.lr_decay_every == 0:
            self.scheduler.step()

    def prepare_data(self, data: Union[T, TS], unsqueeze=False) -> TS:
        if type(data) is T:
            item = data.to(self.device)
            if unsqueeze:
                return item.unsqueeze(0)
            else:
                return item
        out = map(lambda x: x.to(self.device), data)
        if unsqueeze:
            out = map(lambda x: x.unsqueeze(0), out)
        return tuple(out)

    def get_masked_loss(self, predict, gt, mask, mask_sum, invert_mask=False, key: Optional[str] = None):
        if invert_mask:
            b, c, h, w = mask.shape
            mask_sum = h * w - mask_sum
            mask = 1 - mask
        loss = nnf.mse_loss(predict, gt, reduction='none') + self.loss_fn_vgg(predict, gt)
        loss = loss * mask
        loss = torch.einsum('bchw->b', loss) / (3 * mask_sum)
        loss = loss.mean()
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss

    def get_lips_detection_loss(self, predict: T, landmarks: T, key: Optional[str] = None):
        predict = nnf.interpolate(predict, 240)
        predict_landmarks = self.lips_detection(predict)
        key_points = ((14, 18), (12, 16))  # height, width
        loss_all = []
        for pair in key_points:
            dist_predict = (predict_landmarks[:, pair[0]] - predict_landmarks[:, pair[1]]).norm(2, dim=1)
            dist_gt = (landmarks[:, pair[0]] - landmarks[:, pair[1]]).norm(2, dim=1)
            loss_all.append(nnf.mse_loss(dist_predict, dist_gt))
        loss = sum(loss_all)
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss

    def train_iter(self, epoch, data, is_train: bool):
        images, landmarks, gt_images, mask, mask_sum = self.prepare_data(data)
        out = self.model(images, landmarks)
        loss = self.get_masked_loss(out, gt_images, mask, mask_sum, key='image_loss')
        if self.opt.reg_lips > 0:
            out_predict = out * mask + (1 - mask) * gt_images
            loss += self.opt.reg_lips * self.get_lips_detection_loss(out_predict, landmarks, 'lips_loss')
        if self.opt.unpaired:
            landmarks_flip = landmarks.__reversed__()
            out_flip = self.model(images, landmarks_flip)
            out_predict_flip = out_flip * mask + (1 - mask) * gt_images
            loss += self.opt.unpaired * self.get_lips_detection_loss(out_predict_flip, landmarks, 'unpaired_loss')
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.warm_up_scheduler.step()
        self.logger.stash_iter('loss', loss)

    @models_utils.torch_no_grad
    def view(self):
        self.model.eval()
        # self.lips_detection = train_utils.model_lc(options.OptionsLipsDetection(tag='vit', device=self.device).load())[0].eval()
        self.dataset.is_train = False
        for data in self.val_loader:
            images, landmarks, gt_images, mask, mask_sum = self.prepare_data(data)
            b = gt_images.shape[0]
            mask = mask.bool().float()
            mask = 1 - nnf.max_pool2d(1 - mask, (3, 3), (1, 1), padding=1)
            images_in = images[0].unsqueeze(0).repeat(b, 1, 1, 1)
            bg_in = gt_images[0].unsqueeze(0).repeat(b, 1, 1, 1)
            out = self.model(images_in, landmarks)
            out_predict = out * mask + (1 - mask) * bg_in
            # predict_landmarks = self.lips_detection(nnf.interpolate(out_predict, 240))
            # predict_landmarks = (predict_landmarks.cpu().numpy() + 1) * 128
            out_predict = out_predict.permute(0, 2, 3, 1).cpu().numpy()
            out_predict = ((out_predict + 1) * 127.5).astype(np.uint8)
            gt_images = gt_images.permute(0, 2, 3, 1).cpu().numpy()
            gt_images = ((gt_images + 1) * 127.5).astype(np.uint8)
            # all_images = torch.cat((gt_images, out_predict), dim=-1).cpu()
            for i in range(out_predict.shape[0]):
                # predict_image = prcoess_face_forensics.draw_lips(out_predict[i], predict_landmarks[i])
                files_utils.imshow(np.concatenate((gt_images[i], out_predict[i]), axis=1))

    def finalize_video(self, sequence, driven_dir, name):
        from moviepy import editor
        metadata = files_utils.load_pickle(f"{driven_dir}/metadata")
        sample_rate, audio = files_utils.load_wav(f"{driven_dir}/audio")
        audio_out_len = float(audio.shape[0] * len(sequence)) / metadata['frames_count']
        files_utils.save_wav(audio[:int(audio_out_len)], sample_rate, f'{self.opt.cp_folder}/vid2vid/tmp.wav')
        image_utils.gif_group(sequence, f'{self.opt.cp_folder}/vid2vid/tmp', metadata['fps'])
        video_clip = editor.VideoFileClip(f'{self.opt.cp_folder}/vid2vid/tmp.mp4')
        audio_clip = editor.AudioFileClip(f'{self.opt.cp_folder}/vid2vid/tmp.wav')
        audio_clip = editor.CompositeAudioClip([audio_clip])
        video_clip.audio = audio_clip
        video_clip.write_videofile(f'{self.opt.cp_folder}/vid2vid/{name}.mp4')
        video_clip.close()
        files_utils.delete_single(f'{self.opt.cp_folder}/vid2vid/tmp.mp4')
        files_utils.delete_single(f'{self.opt.cp_folder}/vid2vid/tmp.wav')

    def prep_process_infer(self, path_base, path_driving):
        data = self.dataset.prep_infer(path_base, path_driving)
        data = self.prepare_data(data, True)
        return data

    @models_utils.torch_no_grad
    def vid2vid(self, base_folder, driving_folder):
        self.model.eval()
        self.dataset.is_train = False
        name = f"{base_folder.split('/')[-2]}_{driving_folder.split('/')[-2]}"
        images_base = files_utils.collect(base_folder, '.npy')
        images_driving = files_utils.collect(driving_folder, '.npy')
        images_driving = [path for path in images_driving if 'image_' in path[1] or 'crop_' in path[1]]
        images_base = [path for path in images_base if 'image_' in path[1] or 'crop_' in path[1]]
        out = []
        out_align = []
        # self.load()
        vid_len = min(len(images_base), len(images_driving))
        self.logger.start(vid_len)
        for path_base, path_driving in zip(images_base, images_driving):
            image_in, landmarks, image_full, driving_image, mask = self.prep_process_infer(path_base, path_driving)
            out_base = self.model(image_in, landmarks)
            out_predict = out_base * mask + (1 - mask) * image_full
            # w = self.e4e.encode(out_image)
            # out_image = self.stylegan(w)
            driving_lips_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
            driven_landmarks = (landmarks[0].cpu().numpy() + 1) * 128
            driving_lips = prcoess_face_forensics.draw_lips(driving_lips_image, driven_landmarks)
            driving_lips = torch.from_numpy(driving_lips).to(self.device, dtype=torch.float32)
            driving_lips = driving_lips.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
            image = [nnf.interpolate(image, (256, 256)) for image in (image_full, driving_image, driving_lips,
                                                                      out_predict)]
            image = torch.cat(image, dim=3)[0]
            image = files_utils.image_to_display(image)
            out.append(image)
            self.logger.reset_iter()
        self.logger.stop()
        self.finalize_video(out, driving_folder, name)

    def train_epoch(self, epoch, loader, is_train):
        self.model.train(is_train)
        self.logger.start(len(loader))
        for data in loader:
            self.train_iter(epoch, data, is_train)
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=True).to(self.device)
        if self.opt.reg_lips > 0:
            self.lips_detection = train_utils.model_lc(options.OptionsLipsDetection(tag='vit').load())[0].eval()
        for epoch in range(self.opt.epochs):
            self.train_epoch(epoch, self.data_loader, True)
            with torch.no_grad():
                log = self.train_epoch(epoch, self.val_loader, False)
            self.between_epochs(epoch, log)

    @property
    def device(self):
        return self.opt.device

    def __init__(self, opt: OptionsLipsGenerator):
        self.model, self.opt = train_utils.model_lc(opt)
        # if self.opt.pretrained_path:
        #     files_utils.load_model(self.model, self.opt.pretrained_path, self.device, True)
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-4, opt.warm_up)
        self.dataset, self.data_loader, self.val_loader = self.init_dataloader()
        self.offset = opt.warm_up // len(self.data_loader)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000
        self.loss_fn_vgg: Optional[lpips.LPIPS] = None
        self.lips_detection: Optional[lips_detection_model.LipsDetectionModel] = None
        # self.model.vid2vid(f'{constants.DATA_ROOT}/putin_a/', f'{constants.DATA_ROOT}obama/', 'putin_obama')


def main():
    opt = OptionsLipsGenerator(tag='all_ge').load()
    model = TrainLipsGenerator(opt)
    # model.view()
    model.vid2vid(f'{constants.DATA_ROOT}/smith/', f'{constants.DATA_ROOT}obama/')


if __name__ == '__main__':
    main()
