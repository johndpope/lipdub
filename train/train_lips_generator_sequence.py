from utils import files_utils, train_utils
from custom_types import *
import constants
from dataloader import prcoess_face_forensics
from options import OptionsLipsGeneratorSeq
from train.train_conditional_lips_generator import TrainLipsGenerator
from models import models_utils


class TrainLipsGeneratorSeq(TrainLipsGenerator):

    def init_dataloader(self, opt):
        ds = prcoess_face_forensics.LipsSeqDS(opt)
        split_path = f"{opt.data_dir}/split.pkl"
        split = files_utils.load_pickle(split_path)
        if split is None:
            split_train, split_val = ds.get_split(.02)
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
        _, images, ref_images, landmarks, gt_images, mask, mask_sum, lines = self.prepare_data(data)
        offset_ = (self.opt.lips_seq - self.opt.image_seq) // 2
        landmarks_trimmed = landmarks[:, offset_: -offset_]
        out = self.model(images, landmarks, ref_images)
        data = self.seq2batch(out['out_mid'], out['result'], landmarks_trimmed, gt_images, mask, mask_sum, lines)
        out_mid, result, landmarks, gt_images, mask, mask_sum, lines = data
        loss = self.get_masked_loss(result, gt_images[:, :, 128:, 64: -64], mask, mask_sum, key='image_loss')
        loss += self.get_masked_loss(out_mid, gt_images[:, :, 128:, 64: -64], mask, mask_sum, key='image_loss_mid',
                                     mse=True, lpips=False)
        if self.opt.reg_lips > 0:
            mask, result = self.pad(mask, result)
            out_predict_result = result * mask + (1 - mask) * gt_images
            predict_landmarks_result = self.get_landmarks(out_predict_result)
            loss += self.opt.reg_lips * self.get_lips_detection_loss(predict_landmarks_result, landmarks, 'lips_open')
            if self.opt.reg_lips_center > 0:
                loss += self.opt.reg_lips_center * self.get_lips_center_loss(predict_landmarks_result, lines, 'lips_center')
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.warm_up_scheduler:
                self.warm_up_scheduler.step()
        self.logger.stash_iter('loss', loss)

    def __init__(self, opt: OptionsLipsGeneratorSeq):
        super(TrainLipsGeneratorSeq, self).__init__(opt)
        self.dataset, self.data_loader, self.val_loader = self.init_dataloader(opt)
        self.opt: OptionsLipsGeneratorSeq = opt


class InferenceTrainingSeq(TrainLipsGeneratorSeq):

    def set_train(self, is_train):
        self.model.train(is_train)
        self.model.visual_encoder.eval()
        self.model.lips_encoder.eval()

    def inference_train(self, tag: str):
        opt = OptionsLipsGeneratorSeq(tag=tag, device=self.device).load()
        model, _ = train_utils.model_lc(opt)
        self.model.load_state_dict(model.state_dict())
        super(InferenceTrainingSeq, self).train()

    def get_optimizer(self):
        return None, None, None

    def __init__(self, opt: OptionsLipsGeneratorSeq):
        super(TrainLipsGeneratorSeq, self).__init__(opt)
        self.optimizer = Optimizer(self.model.generator_parameters, lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)


def main():
    opt = OptionsLipsGeneratorSeq(tag='all').load()
    opt.device = CPU
    model = TrainLipsGeneratorSeq(opt)
    # model.train()
    # model.view()
    model.vid2vid(f'{constants.DATA_ROOT}/office_michael/', f'{constants.DATA_ROOT}office_jim/')


def inference_train():
    opt = OptionsLipsGeneratorSeq(tag='smith',
                                  data_dir=constants.MNT_ROOT + 'video_frames/smith/',
                                  epochs=5,
                                  batch_size=10).load()
    opt.device = CUDA(0)
    model = InferenceTrainingSeq(opt)
    model.vid2vid(f'{constants.DATA_ROOT}/smith/', f'{constants.DATA_ROOT}/obama/')
    # model.inference_train('all')


if __name__ == '__main__':
    main()
