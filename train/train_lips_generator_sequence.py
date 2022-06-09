import torch

from utils import files_utils, train_utils
from custom_types import *
import constants
from dataloader import prcoess_face_forensics
from options import OptionsLipsGeneratorSeq
from train.train_conditional_lips_generator import TrainLipsGenerator
from models import models_utils, sync_net
import align_faces


class TrainLipsGeneratorSeq(TrainLipsGenerator):

    def init_dataloader(self, opt):
        ds = prcoess_face_forensics.LipsSeqDS(opt)
        split_path = f"{opt.data_dir}/split.pkl"
        split = files_utils.load_pickle(split_path)
        if split is None:
            split_train, split_val = ds.get_split(.05)
            files_utils.save_pickle({'split_val': split_val, 'split_train': split_train}, split_path)
        else:
            split_val, split_train = split['split_val'], split['split_train']
        ds_train, ds_val = Subset(ds, split_train), Subset(ds, split_val)
        dl_train = DataLoader(ds_train, num_workers=0 if DEBUG else 12, shuffle=not constants.DEBUG, drop_last=True,
                              batch_size=8 if DEBUG else opt.batch_size)
        dl_val = DataLoader(ds_val, num_workers=0 if DEBUG else 8, shuffle=True, drop_last=True,
                            batch_size=opt.batch_size)
        return ds, dl_train, dl_val

    def seq2batch(self, *items: T):
        out = []
        for item in items:
            shape = item.shape
            if len(shape) == 2:
                item = item.flatten()
            else:
                item = item.reshape(shape[0] * shape[1], *shape[2:])
            out.append(item)
        return out

    def pad(self, *x):
        return [nnf.pad(item, (64, 64, 128, 0)) for item in x]

    def get_discriminator_loss(self, out, key:Optional[str] = None):
        return 0

    @models_utils.torch_no_grad
    def view(self):
        self.model.eval()
        self.dataset.is_train = False
        for data in self.val_loader:
            _, images, ref_images, landmarks, gt_images, mask, mask_sum, lines = self.prepare_data(data)
            b = images.shape[0]
            out = self.model(images, landmarks, ref_images)
            data = self.seq2batch(out['out_mid'], out['result'], gt_images, mask)
            out_mid, result, gt_images, mask = data
            gt_images = gt_images[:, :, 128:, 64: -64]
            out_predict = result * mask + (1 - mask) * gt_images
            out_predict_mid = out_mid * mask + (1 - mask) * gt_images
            for i in range(0, b * self.opt.image_seq, self.opt.image_seq):
                to_view = torch.cat((out_predict_mid[i], out_predict[i], gt_images[i]), dim=2)
                files_utils.imshow(to_view)

    def blend_infer(self, out_base, mask, image_full):
        out_base = out_base["result"][:, self.opt.image_seq // 2]
        image_full = image_full[:, self.opt.image_seq // 2]
        mask = mask[:, self.opt.image_seq // 2]
        mask, out_base = self.pad(mask, out_base)
        return super(TrainLipsGeneratorSeq, self).blend_infer(out_base, mask, image_full)

    def get_driving_lips(self, landmarks):
        return super(TrainLipsGeneratorSeq, self).get_driving_lips(landmarks[:, self.opt.lips_seq // 2])

    def forward(self, images, landmarks, person_id, ref_images, return_id=False):
        return self.model(images, landmarks, ref_images)

    def train_iter(self, epoch, data, is_train: bool):
        _, images, ref_images, landmarks, gt_images, mask_seq, mask_sum, lines = self.prepare_data(data)
        offset_ = (self.opt.lips_seq - self.opt.image_seq) // 2
        landmarks_trimmed = landmarks[:, offset_: -offset_]
        out = self.model(images, landmarks, ref_images)
        data = self.seq2batch(out['out_mid'], out['result'], landmarks_trimmed, gt_images, mask_seq, mask_sum, lines)
        out_mid, result, landmarks, gt_images, mask, mask_sum, lines = data
        loss = self.get_masked_loss(result, gt_images[:, :, 128:, 64: -64], mask, mask_sum, key='image_loss')
        loss += self.get_masked_loss(out_mid, gt_images[:, :, 128:, 64: -64], mask, mask_sum, key='image_loss_mid',
                                     mse=True, lpips=False)
        mask, result = self.pad(mask, result)
        out_predict_result = result * mask + (1 - mask) * gt_images
        if self.opt.discriminator:
            loss += self.opt.discriminator_lambda * self.get_discriminator_loss(out_predict_result[:, :, 128:, 64: -64])
        if self.opt.reg_lips > 0:

            predict_landmarks_result = self.get_landmarks(out_predict_result)
            loss += self.opt.reg_lips * self.get_lips_detection_open_loss(predict_landmarks_result, landmarks, 'lips_open')
            if self.opt.reg_lips_center > 0:
                loss += self.opt.reg_lips_center * self.get_lips_center_loss(predict_landmarks_result, lines, 'lips_center')
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.warm_up_scheduler:
                self.warm_up_scheduler.step()
        self.logger.stash_iter('loss', loss)

    def get_sync_loss(self, images, audio, key: Optional[str] = None):
        images = (images + 1) / 2
        images = nnf.interpolate(images, 96, mode='bilinear')[:, :, 96 // 2:, :]
        images = images.view(-1, 15, 96 // 2, 96)
        audio_embedding, face_embedding = self.sync_net(audio, images)
        loss = (1 - torch.einsum('bd,bd->b', audio_embedding, face_embedding)).mean()
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss


    def infer_train(self, base_folder: str, driving_folder: str, starting_time: int = -1, ending_time: int = -1,
                    num_epochs: int = 5):
        infer_ds = prcoess_face_forensics.LipsSeqDSInfer(self.dataset, base_folder, driving_folder, starting_time, ending_time)
        dataloader = DataLoader(infer_ds, batch_size=10, drop_last=True, shuffle=True, num_workers=0 if DEBUG else 8)
        dl_train = DataLoader(self.dataset, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
                              batch_size=10)
        for epoch in range(num_epochs):
            self.logger.start(len(dataloader))
            for i, (data_a, data_b) in enumerate(zip(dataloader, dl_train)):
                image_in, image_ref, landmarks, image_full, mask_seq, audio = self.prepare_data(data_a, False)
                offset_ = (self.opt.lips_seq - self.opt.image_seq) // 2
                landmarks_trimmed = landmarks[:, offset_: -offset_]
                out = self.forward(image_in, landmarks, None, image_ref)
                data = self.seq2batch(out['out_mid'], out['result'], landmarks_trimmed, image_full, mask_seq)
                out_mid, result, landmarks, gt_images, mask = data
                mask, result = self.pad(mask, result)
                out_predict_result = result * mask + (1 - mask) * gt_images
                predict_landmarks_result = self.get_landmarks(out_predict_result)
                # loss = self.get_sync_loss(out_predict_result, audio, "sync")
                loss = 10 * self.opt.reg_lips * self.get_lips_detection_open_loss(predict_landmarks_result, landmarks,
                                                                         'lips_open_test')
                loss += 10 * self.get_lips_detection_loss(predict_landmarks_result, landmarks,
                                                                        'lips_all')
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.train_iter(epoch, data_b, True)
                self.logger.reset_iter()
            # exporting landmarks
            self.logger.stop()
            dataloader = DataLoader(infer_ds, batch_size=10, drop_last=False, shuffle=False,
                                    num_workers=0 if DEBUG else 8)
        self.model.eval()
        landmarks_all = []
        name = f"{base_folder.split('/')[-2]}_{driving_folder.split('/')[-2]}"
        with torch.no_grad():
            for i, data_a in enumerate(dataloader):
                image_in, image_ref, landmarks, image_full, mask_seq, audio = self.prepare_data(data_a, False)
                out_base = self.forward(image_in, landmarks, None, image_ref)
                out_predict, image_full = self.blend_infer(out_base, mask_seq, image_full)
                images = [files_utils.image_to_display(image) for image in out_predict]
                landmarks_all += [self.dataset.aligner.get_landmarks(image) for image in images]
        landmarks_all = np.concatenate(landmarks_all)
        files_utils.save_np(landmarks_all, f"{self.opt.cp_folder}/test_lm/{name}")



    @models_utils.torch_no_grad
    def infer(self, base_folder: str, driving_folder: str, starting_time: int = -1, ending_time: int = -1,
              export_aligned=True, align_driving=True):
        infer_ds = prcoess_face_forensics.LipsSeqDSInfer(self.dataset, base_folder, driving_folder,
                                                         starting_time, ending_time, align_driving=align_driving)
        dataloader = DataLoader(infer_ds, batch_size=20, drop_last=False, shuffle=False, num_workers=0 if DEBUG else 8)
        name = f"{base_folder.split('/')[-2]}_{driving_folder.split('/')[-2]}_nosync"
        self.logger.start(len(dataloader))
        out = []
        out_align = []
        for i, data in enumerate(dataloader):
            image_in, image_ref, landmarks, image_full, mask, _ = self.prepare_data(data, False)
            out_base = self.forward(image_in, landmarks, None, image_ref)
            # blend = self.blend(out_base[0], image_full[0], mask[0])
            out_predict, image_full = self.blend_infer(out_base, mask, image_full)
            driving_lips = self.get_driving_lips(landmarks)
            images = [nnf.interpolate(image, (256, 256)) for image in (image_full, driving_lips, out_predict)]
            images = torch.cat(images, dim=3).cpu()
            images = [files_utils.image_to_display(image) for image in images]
            if export_aligned:
                out_align += [files_utils.image_to_display(image) for image in out_predict.cpu()]
            out += images
            self.logger.reset_iter()
        self.logger.stop()
        align = align_faces.FaceAlign()
        self.finalize_video(out, driving_folder, name)
        if export_aligned:
            out_align = align.uncrop_video(base_folder, out_align, starting_time)
            self.finalize_video(out_align, driving_folder, name + '_align')

    @property
    def sync_net(self):
        if self.sync_net_ is None:
            self.sync_net_ = sync_net.SyncNet().eval().to(self.device)
        return self.sync_net_

    def __init__(self, opt: OptionsLipsGeneratorSeq):
        super(TrainLipsGeneratorSeq, self).__init__(opt)
        self.opt: OptionsLipsGeneratorSeq = opt
        self.sync_net_: Optional[sync_net.SyncNet] = None


class TrainLipsGeneratorSeqDisc(TrainLipsGeneratorSeq):

    def save_model(self):
        super(TrainLipsGeneratorSeqDisc, self).save_model()
        save = self.discriminator.module.save if self.data_parallel else self.discriminator.save
        save(suffix='discriminator')

    def between_epochs(self, epoch, scores):
        super(TrainLipsGeneratorSeqDisc, self).between_epochs(epoch, scores)
        if epoch > self.offset and (epoch - self.offset) % self.opt.lr_decay_every == 0:
            self.scheduler_d.step()

    def get_discriminator_loss(self, result, key: Optional[str] = None):
        b, c, h, w = result.shape
        result = result.reshape(-1, self.opt.image_seq, c, h, w)
        loss = self.run_discriminator_loss(result, False)
        # loss = - self.discriminator(result).mean()
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss

    def gradient_penalty(self, fake, real) -> T:
        fake_data = fake.data
        real_data = real.data
        alpha = torch.rand(1, device=self.device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2
        return gradient_penalty.mean()

    def run_discriminator_loss(self, images, is_fake):
        out = self.discriminator(images)
        labels = torch.zeros(images.shape[0]) if is_fake else torch.ones(images.shape[0])
        labels = labels.to(out.device)
        loss = nnf.binary_cross_entropy_with_logits(out.squeeze(), labels)
        return loss


    def disc_iter(self, epoch, data, last_iter: bool):
        _, images, ref_images, landmarks, gt_images, mask_seq = self.prepare_data(data[:-2])
        with torch.no_grad():
            gt_images = gt_images[:, :, :, 128:, 64: -64]
            out = self.model(images, landmarks, ref_images)["result"] * mask_seq + (1 - mask_seq) * gt_images

        d_fake = self.run_discriminator_loss(out, True)
        d_real = self.run_discriminator_loss(gt_images, False)
        loss = d_fake + d_real
        # penalty = self.gradient_penalty(out, gt_images)
        # loss = loss + self.opt.penalty_lambda * penalty
        self.optimizer_d.zero_grad()
        loss.backward()
        self.optimizer_d.step()
        if last_iter:
            self.logger.stash_iter('d_r', d_real, 'd_f', d_fake)
        self.warm_up_scheduler_d.step()

    def train_epoch(self, epoch, loader, is_train):
        self.set_train(is_train)
        self.discriminator.train(True)
        self.logger.start(len(loader) // self.opt.discriminator_iters)
        for i, data in enumerate(loader):
            if not is_train or (i + 1) % self.opt.discriminator_iters == 0:
                self.train_iter(epoch, data, is_train)
                self.logger.reset_iter()
            else:
                self.disc_iter(epoch, data, (i + 2) % self.opt.discriminator_iters == 0)
        return self.logger.stop()

    def __init__(self, opt: OptionsLipsGeneratorSeq):
        super(TrainLipsGeneratorSeqDisc, self).__init__(opt)
        self.discriminator, _ = train_utils.model_lc(opt, suffix='discriminator',
                                                  override_model='seq_lips_discriminator')
        if self.data_parallel:
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=[i for i in range(torch.cuda.device_count())])
        self.optimizer_d, self.scheduler_d, self.warm_up_scheduler_d = self.get_optimizer(self.discriminator)


class InferenceTrainingSeq(TrainLipsGeneratorSeq):

    def set_train(self, is_train):
        self.model.train(is_train)
        self.model.visual_encoder.eval()
        self.model.lips_encoder.eval()

    def fine_tune(self, tag: str):
        opt = OptionsLipsGeneratorSeq(tag=tag, device=self.device).load()
        model, _ = train_utils.model_lc(opt)
        self.model.load_state_dict(model.state_dict())
        super(InferenceTrainingSeq, self).train()

    def get_optimizer(self, _):
        return None, None, None

    def __init__(self, opt: OptionsLipsGeneratorSeq):
        super(InferenceTrainingSeq, self).__init__(opt)
        self.optimizer = Optimizer(self.model.generator_parameters, lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)


class InferenceTrainingSeqFaceFormer(InferenceTrainingSeq):

    def init_dataloader(self, opt: OptionsLipsGeneratorSeq):
        ds = prcoess_face_forensics.LipsSeqDSDual(opt, self.landmarks_root)
        split_path = f"{opt.data_dir}/split_faceformer.pkl"
        split = files_utils.load_pickle(split_path)
        if split is None:
            split_train, split_val = ds.get_split(.05)
            files_utils.save_pickle({'split_val': split_val, 'split_train': split_train}, split_path)
        else:
            split_val, split_train = split['split_val'], split['split_train']
        ds_train, ds_val = Subset(ds, split_train), Subset(ds, split_val)
        dl_train = DataLoader(ds_train, num_workers=0 if DEBUG else 12, shuffle=not constants.DEBUG, drop_last=True,
                              batch_size=8 if DEBUG else opt.batch_size)
        dl_val = DataLoader(ds_val, num_workers=0 if DEBUG else 8, shuffle=True, drop_last=True,
                            batch_size=opt.batch_size)
        return ds, dl_train, dl_val

    def __init__(self, opt: OptionsLipsGeneratorSeq, landmarks_root: str):
        self.landmarks_root = landmarks_root
        super(InferenceTrainingSeqFaceFormer, self).__init__(opt)
        self.optimizer = Optimizer(self.model.generator_parameters + self.model.lips_encoder_parameters, lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)



def main():
    opt = OptionsLipsGeneratorSeq(tag='chin').load()
    opt.device = CUDA(0)
    model = TrainLipsGeneratorSeq(opt)
    model.train()
    # model.view()
    # model.vid2vid(f'{constants.MNT_ROOT}/processed_infer/Eilish/',
    #               f'{constants.MNT_ROOT}/processed_infer/Eilish_French052522/', start_base=26)


def inference_train():

    info_dict = {'Eilish': {'name_base': 'Eilish', 'name_driving': 'Eilish_French052522', 'tag': 'Eilish'},
                 'Johnson': {'name_base': 'Johnson', 'name_driving': 'Dwayne_Spanish_FaceFormer', 'tag': 'Johnson'},
                 'Bourdain': {'name_base': 'BourdainT', 'name_driving': 'Bourdain_Italian_faceformer', 'tag': 'Bourdain'}}
    info = info_dict['Eilish']
    opt = OptionsLipsGeneratorSeq(tag=info['tag'],
                                  data_dir=constants.MNT_ROOT + f'video_frames/{info["name_base"]}/',
                                  epochs=2,
                                  batch_size=10).load()
    opt.device = CUDA(0)
    model = InferenceTrainingSeq(opt)
    # model.fine_tune('all_nose')
    model.infer_train(f'{constants.MNT_ROOT}/processed_infer/{info["name_base"]}/',
                      f'{constants.MNT_ROOT}/processed_infer/{info["name_driving"]}/',
                      starting_time=10, ending_time=-1, num_epochs=10)

    model.infer(f'{constants.MNT_ROOT}/processed_infer/{info["name_base"]}/',
                f'{constants.MNT_ROOT}/processed_infer/{info["name_driving"]}/',
                starting_time=10, ending_time=-1, export_aligned=True)


if __name__ == '__main__':
    main()
