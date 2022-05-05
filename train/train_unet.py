import lpips
from utils import train_utils, files_utils, image_utils
from models import models_utils, stylegan_wrapper
from custom_types import *
import constants
from dataloader import disentanglement_ds
from options import OptionsVisemeUnet, OptionsVisemeClassifier
import align_faces
from e4e import E4E
import PIL

class TrainVisemeDisentanglementUnet:

    def load(self):
        files_utils.load_model(self.stylegan, f'{constants.CHECKPOINTS_ROOT}/pti/obama_a', self.device)
        return

    def init_dataloader(self):
        ds = disentanglement_ds.DualVisemeDS(self.opt)
        dl = DataLoader(ds, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
                        batch_size=4 if DEBUG else self.opt.batch_size)
        return ds, dl

    def between_epochs(self, epoch, scores):
        if scores['loss'] < self.best_score:
            self.best_score = scores['loss']
            self.model.save()
        if epoch > self.offset and (epoch - self.offset) % self.opt.lr_decay_every == 0:
            self.scheduler.step()

    def get_loss(self, predict, gt):
        loss = (nnf.mse_loss(predict, gt) + self.loss_fn_vgg(predict, gt).mean()) / 2
        return loss

    def prepare_data(self, data: Union[T, TS]) -> TS:
        if type(data) is T:
            return data.to(self.device)
        return tuple(map(lambda x: x.to(self.device) if type(x) is T else x, data))

    def get_viseme_classification_loss(self, image, label):
        b, c, h, w = image.shape
        start = w // 4
        image = image[:, :, h // 2:, start: start + w // 2]
        image = nnf.interpolate(image, (128, 128), mode='bilinear', align_corners=True)
        predict = self.viseme_classifier(image)
        loss = nnf.cross_entropy(predict, label)
        self.logger.stash_iter('loss_classifier', loss)
        return loss

    @models_utils.torch_no_grad
    def view(self):
        items = torch.randint(len(self.dataset), (10,)).tolist()
        self.model.eval()
        self.dataset.is_train = False
        for i in items:
            seq = self.dataset[i]
            seq = self.prepare_data(seq)[:4]
            image_in, out_gt, driven_image, mask = [item.unsqueeze(0) for item in seq]
            mask = mask.expand(1, 3, 512, 512) * 2 - 1
            # viseme_vec = self.viseme_classifier.encode(driven_image)
            out = self.model(image_in, driven_image)
            reconstructed = out['reconstructed']

            out_image, lips_predict, mask_predict = self.post_process_result(out)
            # for image, name in zip((image_in, driven_image, out_gt, mask, out_image, lips_predict, mask_predict, reconstructed),
            #                        ('image_in', 'driven_image', 'out_gt', 'mask', 'out_image', 'lips_predict', 'mask_predict', 'reconstructed')):
            #     image = nnf.interpolate(image, (256, 256))
            #
            #     files_utils.save_image(image, f'{constants.CACHE_ROOT}/for_slides/unet_triple/{i:05d}_{name}')
            # elif self.opt.train_mask_decoder:
            #     out, mask_predict = out
            # image = [nnf.interpolate(image, (256, 256)) for image in (image_in, driven_image, out_image, out_gt, lips_predict)]
            image = [nnf.interpolate(image, (256, 256)) for image in
                     (image_in, driven_image, lips_predict, mask_predict, out_image)]
            image = torch.cat(image, dim=3)[0]
            # image = files_utils.image_to_display(image)
            files_utils.save_image(image, f'{constants.CACHE_ROOT}/for_slides/unet_seq/{i:05d}')

    def prep_from_file(self, path):
        image = files_utils.load_np(''.join(path))
        return image

    def prep_process_infer(self, path_base, path_driving):
        image_base = self.prep_from_file(path_base)
        driven_image_full = self.prep_from_file(path_driving)
        driven_image_crop = driven_image_full[512:, 256: 768]
        image_base = self.dataset.transform(PIL.Image.fromarray(image_base), train=False)[0]
        driven_image_full = self.dataset.transform(PIL.Image.fromarray(driven_image_full), train=False)[0]
        driven_image_crop = self.dataset.transform_viseme_inference(PIL.Image.fromarray(driven_image_crop)) * 2 - 1
        return [item.unsqueeze(0).to(self.device) for item in (image_base, driven_image_full, driven_image_crop)]

    def post_process_result(self, out):
        out_image = out['reconstructed']
        if self.opt.train_lips_decoder:
            mask_predict = out['mask']
            lips_predict = out['lips']
            k = 15
            mask_predict = nnf.max_pool2d(mask_predict, (k * 2 + 1, k * 2 + 1), (1, 1), padding=k)
            out_image = out_image * (1 - mask_predict) + mask_predict * lips_predict
        else:
            lips_predict = mask_predict = None
        mask_predict = mask_predict.expand(1, 3, *mask_predict.shape[2:]) * 2 - 1
        return out_image, lips_predict, mask_predict

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


    @models_utils.torch_no_grad
    def vid2vid(self, base_folder, driving_folder, name):

        # self.model.eval()
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
            image_in, driven_image_full, driven_image_crop = self.prep_process_infer(path_base, path_driving)
            out_base = self.model(image_in, driven_image_crop)
            out_image, lips_predict, mask_predict = self.post_process_result(out_base)
            # w = self.e4e.encode(out_image)
            # out_image = self.stylegan(w)
            image = [nnf.interpolate(image, (256, 256)) for image in (image_in, driven_image_full, out_image)]
            image = torch.cat(image, dim=3)[0]
            image = files_utils.image_to_display(image)
            out.append(image)
            self.logger.reset_iter()
        self.logger.stop()

        self.finalize_video(out, driving_folder, name)
        # image_utils.gif_group(out, f'{self.opt.cp_folder}/vid2vid/{name}', 30)
        # video_clip = editor.VideoFileClip(f'{self.opt.cp_folder}/vid2vid/tmp.mp4')
        # audio_clip = editor.AudioFileClip(f'{constants.DATA_ROOT}/raw_videos/101_purpledino_front_comp_v019.wav')
        # audio_clip = editor.CompositeAudioClip([audio_clip])
        # video_clip.audio = audio_clip
        # video_clip.write_videofile(f'{self.opt.cp_folder}/vid2vid/obama_sound.mp4')
        # video_clip.close()
        # align = align_faces.FaceAlign()
        # out_align = align.uncrop_video(f"{constants.DATA_ROOT}/raw_videos/obama_062814.mp4", base_folder, out_align)
        # image_utils.gif_group(out_align, f'{self.opt.cp_folder}/vid2vid/tmp', 24)
        # video_clip = editor.VideoFileClip(f'{self.opt.cp_folder}/vid2vid/tmp.mp4')
        # audio_clip = editor.AudioFileClip(f'{constants.DATA_ROOT}/raw_videos/101_purpledino_front_comp_v019.wav')
        # audio_clip = editor.CompositeAudioClip([audio_clip])
        # video_clip.audio = audio_clip
        # video_clip.write_videofile(f'{self.opt.cp_folder}/vid2vid/obama_pti_align.mp4')
        # video_clip.close()

    def get_masked_loss(self, predict, gt, mask, mask_sum, invert_mask=False):
        if invert_mask:
            b, c, h, w = mask.shape
            mask_sum = h * w - mask_sum
            mask = 1 - mask
        loss = nnf.mse_loss(predict, gt, reduction='none') + self.loss_fn_vgg(predict, gt)
        loss = loss * mask
        loss = torch.einsum('bchw->b', loss) / (3 * mask_sum)
        loss = loss.mean()

        return loss

    def train_iter(self, data):
        image_in, out_gt, driven_image, mask, mask_sum, viseme_select = self.prepare_data(data)
        self.optimizer.zero_grad()
        # viseme_vec = self.viseme_classifier.encode(driven_image)
        out = self.model(image_in, driven_image)
        predict_bg = out['reconstructed']
        if self.opt.train_mask_decoder:
            mask_predict = out['mask']
            loss_mask = nnf.binary_cross_entropy(mask_predict, mask)
            self.logger.stash_iter('loss_mask', loss_mask)
        else:
            loss_mask = 0
        # b, c, h, w = out.shape
        # start = w // 4
        if self.opt.train_lips_decoder:
            lips_predict = out['lips']
            loss_lips = self.get_masked_loss(lips_predict, out_gt, mask, mask_sum)
            loss_image = self.get_masked_loss(predict_bg, out_gt, mask, mask_sum, True)
        else:
            loss_lips = self.get_masked_loss(out, out_gt, mask, mask_sum)
            loss_image = self.get_loss(predict_bg, out_gt)
        self.logger.stash_iter('loss_lips', loss_lips)
        self.logger.stash_iter('loss_image', loss_image)
        # if self.opt.classification_loss > 0:
        #     loss_classifier = self.get_viseme_classification_loss(out, viseme_select)
        #     loss = loss_image + self.opt.classification_loss * loss_classifier
        # else:
        #     loss = loss_image
        loss = 2 * loss_lips + loss_mask + loss_image
        self.logger.stash_iter('loss', loss)
        loss.backward()
        self.optimizer.step()
        self.warm_up_scheduler.step()

    def direction_transfer(self, base_folder, driving_folder, name):
        viseme_ds = disentanglement_ds.VisemeWDS()
        images_base = files_utils.collect(base_folder, '.npy')
        images_driving = files_utils.collect(driving_folder, '.npy')
        images_driving = [path for path in images_driving if 'image_' in path[1] or 'crop_' in path[1]]
        images_base = [path for path in images_base if 'image_' in path[1] or 'crop_' in path[1]]
        out = []
        out_align = []
        vid_len = min(len(images_driving), len(images_driving))
        self.logger.start(vid_len)
        out = []
        for path_base, path_driving in zip(images_base, images_driving):
            image_base, driven_image_full, _ = self.prep_process_infer(path_base, path_driving)
            w_base = self.e4e.encode(image_base)
            w_driven = self.e4e.encode(driven_image_full)
            w_base_viseme_dir = viseme_ds.find_viseme_dir(w_base)
            w_driven_viseme_dir = viseme_ds.find_viseme_dir(w_driven)
            out_a = self.stylegan(w_base - w_base_viseme_dir)
            out_b = self.stylegan(w_base - w_base_viseme_dir + w_driven_viseme_dir)
            image = [nnf.interpolate(image, (256, 256)) for image in (image_base, driven_image_full, out_a, out_b)]
            image = torch.cat(image, dim=3)[0]
            image = files_utils.image_to_display(image)
            out.append(image)
            self.logger.reset_iter()
            # files_utils.imshow(out_a)
            # files_utils.imshow(out_b)
        # image_utils.gif_group(out, f'{self.opt.cp_folder}/vid2vid/tmp', 24)
        self.finalize_video(out, driving_folder, name)
        self.logger.stop()


    def train_epoch(self):
        self.model.train(True)
        self.logger.start(len(self.data_loader))
        for data in self.data_loader:
            self.train_iter(data)
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=True).to(self.device)
        print("training")
        for epoch in range(self.opt.epochs):
            log = self.train_epoch()
            self.between_epochs(epoch, log)

    @property
    def device(self):
        return self.opt.device

    def __init__(self, opt: OptionsVisemeUnet):
        opt_cls = OptionsVisemeClassifier().load()
        opt.conditional_dim = opt_cls.hidden_dim
        self.model, self.opt = train_utils.model_lc(opt)
        self.opt: OptionsVisemeUnet = self.opt
        self.stylegan = stylegan_wrapper.StyleGanWrapper(opt).to(self.device).eval()
        self.e4e = E4E()
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-3, opt.warm_up)
        self.dataset, self.data_loader = self.init_dataloader()
        self.offset = opt.warm_up // len(self.data_loader)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000
        self.loss_fn_vgg: Optional[lpips.LPIPS] = None


def main():
    opt = OptionsVisemeUnet(tag='triple_decoder_instant').load()
    model = TrainVisemeDisentanglementUnet(opt)
    model.view()
    # model.train()
    # model.view()
    # model.direction_transfer(f'{constants.DATA_ROOT}/processed/',
    #                          f'{constants.DATA_ROOT}101_purpledino_front_comp_v019/', 'obama_dino')
    # model.vid2vid(f'{constants.DATA_ROOT}/putin_a/', f'{constants.DATA_ROOT}obama/', 'putin_obama')


if __name__ == '__main__':
    main()
