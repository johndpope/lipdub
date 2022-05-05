import lpips
from utils import train_utils, files_utils, image_utils
from models import stylegan_wrapper, models_utils
from custom_types import *
import constants
from dataloader import disentanglement_ds
from options import OptionsDisentanglementViseme
from e4e import E4E
import align_faces


class TrainVisemeDisentanglement:

    def init_dataloader(self):
        ds = disentanglement_ds.VisemeWDS()
        dl = DataLoader(ds, num_workers=0 if DEBUG else 4, shuffle=not constants.DEBUG, drop_last=True,
                        batch_size=4 if DEBUG else self.opt.batch_size)
        return ds, dl

    def between_epochs(self, epoch, scores):
        if scores['loss'] < self.best_score:
            self.best_score = scores['loss']
            self.model.save()
        if epoch > self.offset and (epoch - self.offset) % self.opt.lr_decay_every == 0:
            self.scheduler.step()

    @models_utils.torch_no_grad
    def view(self):
        items = torch.randint(len(self.dataset), (10,)).tolist()
        for i in items:
            seq = self.dataset[i]
            seq = self.prepare_data(seq)
            seq = [item.unsqueeze(0) for item in seq]
            w_base, w_drive, out_gt, base_image_in, driven_image_in = self.prep_process_iter(*seq, True)
            out = self.model(w_base, w_drive, self.stylegan)
            image = [nnf.interpolate(image, (256, 256)) for image in (base_image_in, driven_image_in, out)]
            image = torch.cat(image, dim=3)[0]
            image = files_utils.image_to_display(image)
            files_utils.imshow(image)

    def prep_process_infer(self, path_base, path_driving):
        image_base = self.e4e.prepare_image(''.join(path_base), is_np=True).to(self.device)
        image_driving = files_utils.load_np(''.join(path_driving))
        image_driving = torch.from_numpy(image_driving).float() / 127.5 - 1
        image_driving = image_driving.cuda().unsqueeze(0).permute(0, 3, 1, 2)
        # w_drive = image_driving[:, :, 512:, 256: 768]
        # w_drive = nnf.interpolate(w_drive, (128, 128))
        w_base = self.e4e.encode(image_base)
        w_drive =  self.e4e.encode(image_driving)
        return w_base, w_drive, image_base, image_driving

    def load(self):
        files_utils.load_model(self.stylegan, f'{constants.CHECKPOINTS_ROOT}/pti/obama_a', self.device)
        return


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
    def vid2vid_ds(self):
        w_base_seq, w_driven_seq = self.dataset.get_seq()
        sequence = []
        self.logger.start(w_base_seq.shape[0])
        for w_base, w_driven in zip(w_base_seq, w_driven_seq):
            w_base, w_driven = w_base.unsqueeze(0).to(self.device), w_driven.unsqueeze(0).to(self.device)
            w_base, w_drive, _, base_image_in, driven_image_in = self.prep_process_iter(w_base, w_driven, return_images=True)
            out = self.model(w_base, w_driven, self.stylegan)
            image = [nnf.interpolate(image, (256, 256)).cpu() for image in (base_image_in, driven_image_in, out)]
            image = torch.cat(image, dim=3)[0]
            image = files_utils.image_to_display(image)
            sequence.append(image)
            self.logger.reset_iter()
        image_utils.gif_group(sequence, f'{self.opt.cp_folder}/vid2vid/tmp', 24)
        self.logger.stop()

    @models_utils.torch_no_grad
    def vid2vid(self, base_folder, driving_folder):
        self.model.eval()
        self.stylegan.eval()
        name = f"{base_folder.split('/')[-2]}_{driving_folder.split('/')[-2]}"
        images_base = files_utils.collect(base_folder, '.npy')
        images_driving = files_utils.collect(driving_folder, '.npy')
        images_driving = [path for path in images_driving if 'image_' in path[1] or 'crop_' in path[1]]
        images_base = [path for path in images_base if 'image_' in path[1] or 'crop_' in path[1]]
        out = []
        out_align = []
        vid_len = min(len(images_driving), len(images_base))
        self.logger.start(vid_len)
        for path_base, path_driving in zip(images_base, images_driving):
            w_base, w_drive, base_image_in, driven_image_in = self.prep_process_infer(path_base, path_driving)
            out_ = self.model(w_base, w_drive, self.stylegan, True).cpu()
            # image = [nnf.interpolate(image, (256, 256)) for image in (base_image_in, driven_image_in, out_)]
            # image = torch.cat(image, dim=3)[0]
            # image = files_utils.image_to_display(out_)
            out.append(out_)
            self.logger.reset_iter()
        self.logger.stop()
        self.logger.start(vid_len)
        self.load()
        for i, (path_base, path_driving) in enumerate(zip(images_base, images_driving)):
            w_base, w_drive, base_image_in, driven_image_in = self.prep_process_infer(path_base, path_driving)
            out_ = self.model(w_base, w_drive, self.stylegan, True)
            out_align.append(files_utils.image_to_display(out_))
            image = [nnf.interpolate(image, (256, 256)).cpu() for image in (base_image_in, driven_image_in, out[i], out_, )]
            image = torch.cat(image, dim=3)[0]
            image = files_utils.image_to_display(image)
            out[i] = image
            self.logger.reset_iter()
        self.logger.stop()
        self.finalize_video(out, driving_folder, name)
        align = align_faces.FaceAlign()
        out_align = align.uncrop_video(f"{constants.DATA_ROOT}/raw_videos/obama_062814.mp4", base_folder, out_align)
        self.finalize_video(out_align, driving_folder, name + '_align')


    def get_loss(self, predict, gt):
        loss = nnf.mse_loss(predict, gt)
        loss += self.loss_fn_vgg(predict, gt).mean()
        return loss

    def prepare_data(self, data: Union[T, TS]) -> TS:
        if type(data) is T:
            return data.to(self.device)
        return tuple(map(lambda x: x.to(self.device), data))

    @models_utils.torch_no_grad
    def prep_process_iter(self, base_vec_in, driven_vec, base_vec_out: TN = None, return_images=False):
        base_image_in = self.stylegan(base_vec_in)
        driven_image_in = self.stylegan(driven_vec)
        if base_vec_out is not None:
            out_gt = self.stylegan(base_vec_out)
        else:
            out_gt = None
        w_base = self.e4e.encode(base_image_in)
        w_drive = self.e4e.encode(driven_image_in)
        if return_images:
            return w_base, w_drive, out_gt, base_image_in, driven_image_in
        return w_base, w_drive, out_gt

    def train_iter(self, data):
        base_vec_in, driven_vec, base_vec_out = self.prepare_data(data)
        w_base, w_drive, gt_image = self.prep_process_iter(base_vec_in, driven_vec, base_vec_out)
        self.optimizer.zero_grad()
        out = self.model(w_base, w_drive, self.stylegan)
        loss = self.get_loss(out, gt_image)
        loss.backward()
        self.optimizer.step()
        self.warm_up_scheduler.step()
        self.logger.stash_iter('loss', loss)

    def train_epoch(self):
        self.model.train(self.base_training)
        self.logger.start(len(self.data_loader))
        for data in self.data_loader:
            self.train_iter(data)
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device)
        print("training")
        for epoch in range(self.opt.epochs):
            log = self.train_epoch()
            self.between_epochs(epoch, log)

    @property
    def device(self):
        return self.opt.device

    def __init__(self, opt: OptionsDisentanglementViseme):
        self.model, self.opt = train_utils.model_lc(opt)
        self.base_training = True
        self.stylegan = stylegan_wrapper.StyleGanWrapper(opt).to(self.device).eval()
        self.e4e = E4E()
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-4, opt.warm_up)
        self.dataset, self.data_loader = self.init_dataloader()
        self.offset = opt.warm_up // len(self.data_loader)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000
        self.loss_fn_vgg: Optional[lpips.LPIPS] = None


def main():
    opt = OptionsDisentanglementViseme().load()
    model = TrainVisemeDisentanglement(opt)
    # model.train()
    # model.view()
    #
    model.vid2vid_ds()
    # model.vid2vid(f'{constants.DATA_ROOT}/processed/', f'{constants.DATA_ROOT}101_purpledino_front_comp_v019/')


if __name__ == '__main__':
    main()
