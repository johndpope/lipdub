from torchvision.transforms import GaussianBlur
import options
from utils import train_utils, files_utils, image_utils
from custom_types import *
import constants
from dataloader import prcoess_face_forensics
from options import OptionsLipsGenerator
from models import models_utils, lips_detection_model
import lpips
import cv2


class TrainLipsGenerator:

    @staticmethod
    def get_blended_mask(mask):
        mask = mask.bool().float()
        mask = 1 - nnf.max_pool2d(1 - mask, (9, 9), (1, 1), padding=4)
        kernel_size = 15
        blur = GaussianBlur(kernel_size, sigma=3.)
        mask = blur(mask)
        return mask

    def save_model(self):
        save = self.model.module.save if self.data_parallel else self.model.save
        save()

    def forward(self, images, landmarks, person_id, ref_images, return_id=False):
        if self.opt.reverse_input:
            ref_images, images = images, ref_images
        if self.opt.train_visual_encoder or self.opt.concat_ref:
            return self.model(images, landmarks, ref_images, return_id)
        return self.model(images, landmarks, person_id, return_id)

    def init_dataloader(self, opt):
        # self.opt.data_dir = constants.MNT_ROOT + 'video_frames/obama/'
        ds = prcoess_face_forensics.LipsConditionedDS(opt)
        split_path = f"{opt.data_dir}/split.pkl"
        split = files_utils.load_pickle(split_path)

        if split is None:
            split = torch.rand(len(ds)).argsort()
            split_val, split_train = split[:int(len(split) * .1)], split[int(len(split) * .1):]
            files_utils.save_pickle({'split_val': split_val, 'split_train': split_train}, split_path)
        else:
            split_val, split_train = split['split_val'], split['split_train']
        ds_train, ds_val = Subset(ds, split_train), Subset(ds, split_val)
        dl_train = DataLoader(ds_train, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
                              batch_size=8 if DEBUG else opt.batch_size)
        dl_val = DataLoader(ds_val, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
                            batch_size=opt.batch_size)
        return ds, dl_train, dl_val

    def between_epochs(self, epoch, scores):
        if scores['loss'] < self.best_score:
            self.best_score = scores['loss']
            self.save_model()
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

    def get_masked_loss(self, predict, gt, mask, mask_sum, invert_mask=False, key: Optional[str] = None,
                        mse=True, lpips=True):
        if invert_mask:
            b, c, h, w = mask.shape
            mask_sum = h * w - mask_sum
            mask = 1 - mask
        loss = 0
        if mse:
            loss += nnf.mse_loss(predict, gt, reduction='none')
        if lpips:
            loss += self.loss_fn_vgg(predict, gt)
        loss = loss * mask
        loss = torch.einsum('bchw->b', loss) / (3 * mask_sum)
        loss = loss.mean()
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss

    def get_landmarks(self, predict):
        predict = nnf.interpolate(predict, 240)
        landmarks = self.lips_detection(predict)
        return landmarks

    def get_lips_detection_loss(self, predict_landmarks: T, landmarks: T, key: Optional[str] = None):
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

    def get_lips_center_loss(self, predict_landmarks: T, lines: T, key: Optional[str] = None):
        key_points = ([3, 9], [0, 6])  # height, width
        directions = lines[:, :, 1] - lines[:, :, 0]
        directions_norm = directions.norm(2, 2)
        loss_all = []
        for i, pair in enumerate(key_points):
            points = predict_landmarks[:, pair]
            origin = lines[:, i, 0]
            direction = directions[:, i]
            direction_norm = directions_norm[:, i]
            diff_ = points - origin[:, None]
            project = diff_[:, :, 1] * direction[:, None, 0] - diff_[:, :, 0] * direction[:, None, 1]
            project = project / direction_norm[:, None]
            loss_ = (project ** 2).mean()
            loss_all.append(loss_ ** 2)
        loss = sum(loss_all)
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss

    @staticmethod
    def blend(source, target, mask):
        im_src = files_utils.image_to_display(source)
        im_dst = files_utils.image_to_display(target)
        im_mask = mask.bool().cpu()[0]
        y_pos, x_pos = torch.where(im_mask)
        center = int(x_pos.max() + x_pos.min()) // 2, int(y_pos.max() + y_pos.min()) // 2
        im_mask = im_mask.numpy().astype(np.uint8) * 255
        im_mask = np.expand_dims(im_mask, 2)
        # center = (im_src.shape[1] // 2, im_src.shape[0] // 2)

        im_clone = cv2.seamlessClone(im_src, im_dst, im_mask, center, cv2.NORMAL_CLONE)

        return im_clone

    def constructive_loss(self, group_a, group_b, temperature=.07, key: Optional[str] = None):
        fe = torch.cat((group_a, group_b))
        labels = torch.arange(group_a.shape[0])
        labels = torch.cat((labels + group_a.shape[0] - 1, labels)).to(self.device)
        similarity_matrix = torch.matmul(fe, fe.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        logits = similarity_matrix / temperature
        loss = nnf.cross_entropy(logits, labels)
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss

    def train_iter(self, epoch, data, is_train: bool):
        person_id, images, ref_images, landmarks, gt_images, mask, mask_sum, lines = self.prepare_data(data)
        out, ref_id = self.forward(images, landmarks, person_id, ref_images, True)
        loss = self.get_masked_loss(out, gt_images, mask, mask_sum, key='image_loss')
        if self.opt.reg_lips > 0:
            out_predict = out * mask + (1 - mask) * gt_images
            predict_landmarks = self.get_landmarks(out_predict)
            loss += self.opt.reg_lips * self.get_lips_detection_loss(predict_landmarks, landmarks, 'lips_open')
            if self.opt.reg_lips_center > 0:
                loss += self.opt.reg_lips_center * self.get_lips_center_loss(predict_landmarks, lines, 'lips_center')
        if self.opt.reg_constructive > 0:
            base_id = self.model.encode_images(gt_images)
            loss += self.opt.reg_constructive * self.constructive_loss(base_id, ref_id, key='constructive_loss')

            pass
        if self.opt.unpaired:
            landmarks_flip = landmarks.__reversed__()
            person_id_flip = person_id.__reversed__()
            out_flip = self.forward(images, landmarks_flip, person_id_flip, ref_images)
            out_predict_flip = out_flip * mask + (1 - mask) * gt_images
            loss += self.opt.unpaired * self.get_lips_detection_loss(out_predict_flip, landmarks, 'unpaired_loss')
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.warm_up_scheduler:
                self.warm_up_scheduler.step()
        self.logger.stash_iter('loss', loss)

    @models_utils.torch_no_grad
    def view(self):
        self.model.eval()
        # self.lips_detection = train_utils.model_lc(options.OptionsLipsDetection(tag='vit', device=self.device).load())[0].eval()
        self.dataset.is_train = False
        for data in self.val_loader:
            person_id, images, ref_images, landmarks, gt_images, mask_raw, mask_sum, lines = self.prepare_data(data)
            b = gt_images.shape[0]
            mask = self.get_blended_mask(mask_raw)

            images_in = images[0].unsqueeze(0).repeat(b, 1, 1, 1)
            mask = mask[0].unsqueeze(0).repeat(b, 1, 1, 1)
            bg_in = gt_images[0].unsqueeze(0).repeat(b, 1, 1, 1)
            landmarks = landmarks[0].unsqueeze(0).repeat(b, 1, 1)
            out = self.forward(images_in, landmarks, person_id, ref_images)
            out_predict = out * mask + (1 - mask) * bg_in
            # predict_landmarks = self.lips_detection(nnf.interpolate(out_predict, 240))
            # predict_landmarks = (predict_landmarks.cpu().numpy() + 1) * 128
            out_predict = out_predict.permute(0, 2, 3, 1).cpu().numpy()
            out_predict = ((out_predict + 1) * 127.5).astype(np.uint8)
            gt_images = gt_images.permute(0, 2, 3, 1).cpu().numpy()
            gt_images = ((gt_images + 1) * 127.5).astype(np.uint8)
            # all_images = torch.cat((gt_images, out_predict), dim=-1).cpu()
            for i in range(out_predict.shape[0]):
                blended = self.blend(out[i], bg_in[i], mask_raw[i])
                # predict_image = prcoess_face_forensics.draw_lips(out_predict[i], predict_landmarks[i])
                files_utils.imshow(np.concatenate((gt_images[i], out_predict[i]), axis=1))

    @models_utils.torch_no_grad
    def view_grid(self, num_lips, num_rows):
        seed = 10
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model.eval()
        self.dataset.is_train = False
        counter = 0
        stash = [[] for _ in range(6)]
        for data in self.val_loader:
            for i, item in enumerate(self.prepare_data(data)):
                stash[i].append(item)
            counter += stash[0][-1].shape[0]
            if  counter >= num_lips and counter >= num_rows:
                break
        for i, item in enumerate(stash):
            stash[i] = torch.cat(item, dim=0)
        person_id, images, landmarks, gt_images, mask, _ = stash
        person_id, images, mask, gt_images = person_id[:num_rows], images[:num_rows], mask[:num_rows], gt_images[:num_rows]
        landmarks = landmarks[:num_lips]
        mask = self.get_blended_mask(mask)
        all_images = []
        num_lips_real = 0
        for i in range(num_lips):
            if i not in (0, 2, 3, 4, 10, 14, 15, 18):
                continue
            num_lips_real += 1
            b = gt_images.shape[0]
            all_images.append(prcoess_face_forensics.draw_lips(256, landmarks[i], 4))
            landmarks_in = landmarks[i].unsqueeze(0).repeat(b, 1, 1)
            out = self.model(images, landmarks_in, person_id)
            out_predict = out * mask + (1 - mask) * gt_images
            out_predict = out_predict.permute(0, 2, 3, 1).cpu().numpy()
            out_predict = ((out_predict + 1) * 127.5).astype(np.uint8)
            for j in range(out_predict.shape[0]):
                if j in (3, 5, 6):
                    all_images.append(out_predict[j])
        rows_cur = len(all_images) // num_lips_real
        out = image_utils.simple_grid(rows_cur, num_lips_real, 5, lambda r, c: all_images[c * rows_cur + r])
        files_utils.save_image(out, f"{constants.CACHE_ROOT}for_slides/ddpm/grid_naive_tmp")

    @models_utils.torch_no_grad
    def view_grid_id(self, num_id, num_rows):
        seed = 10
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model.eval()
        self.dataset.is_train = False
        counter = 0
        stash = [[] for _ in range(6)]
        for data in self.val_loader:
            for i, item in enumerate(self.prepare_data(data)):
                stash[i].append(item)
            counter += stash[0][-1].shape[0]
            if counter >= num_id + num_rows:
                break
        for i, item in enumerate(stash):
            stash[i] = torch.cat(item, dim=0)
        person_id, images, landmarks, gt_images, mask, _ = stash
        gt_images_id = gt_images[num_rows:]
        person_id, images, mask, gt_images = person_id[num_rows:], images[:num_rows], mask[:num_rows], gt_images[ :num_rows]
        landmarks = landmarks[:num_rows]
        mask = self.get_blended_mask(mask)
        all_images = [files_utils.image_to_display(gt_images[i]) for i in range(num_rows) if i in (1, 2, 9)]
        all_images = [np.ones_like(all_images[0]) * 255] + all_images
        # gt_images_ = gt_images.permute(0, 2, 3, 1).cpu().numpy()
        # gt_images = ((gt_images + 1) * 127.5).astype(np.uint8)
        num_id_real = 0
        for i in range(num_id):
            if i not in (0, 2, 3, 6, 7, 12, 15):
                continue
            num_id_real += 1
            b = gt_images.shape[0]
            all_images.append(files_utils.image_to_display(gt_images_id[i]))
            person_id_ind = person_id[i].unsqueeze(0).repeat(b)
            out = self.model(images, landmarks, person_id_ind)
            out_predict = out * mask + (1 - mask) * gt_images
            out_predict = out_predict.permute(0, 2, 3, 1).cpu().numpy()
            out_predict = ((out_predict + 1) * 127.5).astype(np.uint8)
            for j in range(out_predict.shape[0]):
                if j in (1, 2, 9):
                    all_images.append(out_predict[j])
        rows_cur = len(all_images) // (num_id_real + 1)
        out = image_utils.simple_grid(rows_cur, num_id_real + 1, lambda r, c: 10 if r == 0 or c == 0 else 5, lambda r, c: all_images[c * rows_cur + r])
        # files_utils.imshow(out)
        files_utils.save_image(out, f"{constants.CACHE_ROOT}for_slides/ddpm/grid_id_tmp")

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

    def prep_process_infer(self, path_base, path_driving, i):
        data = self.dataset.prep_infer(path_base, path_driving, i)
        data = self.prepare_data(data, True)
        return data

    def blend_infer(self, out_base, mask, image_full):
        return out_base * mask + (1 - mask) * image_full, image_full

    def get_driving_lips(self, landmarks):
        driving_lips_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        driven_landmarks = (landmarks[0].cpu().numpy() + 1) * 128
        driving_lips = prcoess_face_forensics.draw_lips(driving_lips_image, driven_landmarks)
        driving_lips = torch.from_numpy(driving_lips).to(self.device, dtype=torch.float32)
        driving_lips = driving_lips.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
        return driving_lips

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
        for i in  range(vid_len):
            image_in, image_ref, landmarks, image_full, driving_image, mask = self.prep_process_infer(images_base, images_driving, i)
            out_base = self.forward(image_in, landmarks, None, image_ref)
            # blend = self.blend(out_base[0], image_full[0], mask[0])
            out_predict, image_full = self.blend_infer(out_base, mask, image_full)
            driving_lips = self.get_driving_lips(landmarks)
            image = [nnf.interpolate(image, (256, 256)) for image in (image_full, driving_image, driving_lips,
                                                                      out_predict)]
            image = torch.cat(image, dim=3)[0]
            image = files_utils.image_to_display(image)
            out.append(image)
            self.logger.reset_iter()
        self.logger.stop()
        self.finalize_video(out, driving_folder, name)

    def train_epoch(self, epoch, loader, is_train):
        self.set_train(is_train)
        self.logger.start(len(loader))
        for data in loader:
            self.train_iter(epoch, data, is_train)
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=True).to(self.device)
        if self.opt.reg_lips > 0:
            self.lips_detection = train_utils.model_lc(options.OptionsLipsDetection(tag='vit',
                                                                                    device=self.device).load())[0].eval()
            if self.data_parallel:
                self.lips_detection = nn.DataParallel(self.lips_detection, device_ids=[i for i in range(torch.cuda.device_count())])
                self.loss_fn_vgg = nn.DataParallel(self.loss_fn_vgg, device_ids=[i for i in range(torch.cuda.device_count())])
        for epoch in range(self.opt.epochs):
            self.train_epoch(epoch, self.data_loader, True)
            with torch.no_grad():
                log = self.train_epoch(epoch, self.val_loader, False)
            self.between_epochs(epoch, log)

    @property
    def device(self):
        return self.opt.device

    def set_train(self, is_train: bool):
        self.model.train(is_train)

    def get_optimizer(self):
        optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .9)
        warm_up_scheduler = train_utils.LinearWarmupScheduler(optimizer, 1e-4, self.opt.warm_up)
        return optimizer, scheduler, warm_up_scheduler

    def __init__(self, opt: OptionsLipsGenerator):
        self.dataset, self.data_loader, self.val_loader = self.init_dataloader(opt)
        # opt.num_ids = 975
        self.model, opt = train_utils.model_lc(opt)
        self.data_parallel = torch.cuda.device_count() > 1 and opt.device != CPU
        if self.data_parallel:
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(torch.cuda.device_count())])
        self.opt: OptionsLipsGenerator = opt
        self.optimizer, self.scheduler, self.warm_up_scheduler = self.get_optimizer()
        self.offset = opt.warm_up // len(self.data_loader)
        self.logger = train_utils.Logger()
        self.best_score = 10000
        self.loss_fn_vgg: Optional[lpips.LPIPS] = None
        self.lips_detection: Optional[lips_detection_model.LipsDetectionModel] = None
        # self.model.vid2vid(f'{constants.DATA_ROOT}/putin_a/', f'{constants.DATA_ROOT}obama/', 'putin_obama')


class InferenceTraining(TrainLipsGenerator):

    def save_model(self):
        files_utils.save_model(self.z_new.clone().detach().cpu(),  f"{self.opt.cp_folder}/z_id.pt")

    def load(self):
        z = files_utils.load_model(None, f"{self.opt.cp_folder}/z_id.pt", self.device, True)
        if z is None:
            z = self.model.get_new_z_id()
        return z

    def forward(self, images, landmarks, person_id):
        return self.model.inference_forward(images, landmarks, self.z_new)

    def set_train(self, is_train):
        self.model.eval()

    def get_optimizer(self):
        return None, None, None

    def __init__(self, opt):
        super(InferenceTraining, self).__init__(opt)
        files_utils.load_model(self.model, self.opt.pretrained_path, self.device, True)
        self.dataset.is_train = False
        self.z_new: T = self.load()
        self.z_new.requires_grad = True
        self.optimizer = Optimizer([self.z_new], lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)


class InferenceTrainingFull(TrainLipsGenerator):

    def save_model(self):
        super(InferenceTrainingFull, self).save_model()
        torch.save(self.z_new.clone().detach().cpu(), f"{self.opt.cp_folder}/z_id.pt")

    def load(self):
        if files_utils.is_file(f"{self.opt.cp_folder}/z_id.pt"):
            z = torch.load(f"{self.opt.cp_folder}/z_id.pt").to(self.device)
        else:
            z = self.model.get_new_z_id()
        return z

    def forward(self, images, landmarks, person_id):
        return self.model.inference_forward(images, landmarks, self.z_new)

    def set_train(self, is_train):
        self.model.train(is_train)
        self.model.encoder.eval()

    def train(self):
        files_utils.load_model(self.model, self.opt.pretrained_path, self.device, True)
        super(InferenceTrainingFull, self).train()

    def get_optimizer(self):
        return None, None, None

    def __init__(self, opt):
        super(InferenceTrainingFull, self).__init__(opt)
        self.z_new: T = self.load()
        self.z_new.requires_grad = True
        self.optimizer = Optimizer(self.model.generator.parameters(), lr=1e-5)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-4, self.opt.warm_up)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)


def main():
    opt = OptionsLipsGenerator(tag='all').load()
    opt.device = CPU
    model = TrainLipsGenerator(opt)
    # model.train()
    # model.view()
    # model.view_grid(20, 12)
    model.vid2vid(f'{constants.DATA_ROOT}/office_michael/', f'{constants.DATA_ROOT}office_jim/')
    # model.vid2vid(f'{constants.DATA_ROOT}/obama/', f'{constants.DATA_ROOT}smith/')


def main_inference():
    opt = OptionsLipsGenerator(tag='smith_id_token',
                               data_dir=constants.MNT_ROOT + 'video_frames/smith/',
                               pretrained_path=f'{constants.CHECKPOINTS_ROOT}conditional_lips_generator_all_id_token/model.pt',
                               epochs=20).load()
    model = InferenceTraining(opt)
    # model.train()
    # model.vid2vid(f'{constants.DATA_ROOT}/obama/', f'{constants.DATA_ROOT}101_purpledino_front_comp_v019/')


if __name__ == '__main__':
    main()
