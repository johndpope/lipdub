from utils import files_utils
from custom_types import *
import constants
from dataloader import prcoess_face_forensics
from options import OptionsLipsGeneratorSeq
from train.train_conditional_lips_generator import TrainLipsGenerator



class TrainLipsGeneratorSeq(TrainLipsGenerator):

    def init_dataloader(self, opt):
        # self.opt.data_dir = constants.MNT_ROOT + 'video_frames/obama/'
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
        dl_val = DataLoader(ds_val, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
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

    def train_iter(self, epoch, data, is_train: bool):
        _, images, ref_images, landmarks, gt_images, mask, mask_sum, lines = self.prepare_data(data)
        offset_ = (self.opt.lips_seq - self.opt.image_seq) // 2
        landmarks_trimmed = landmarks[:, offset_: -offset_]
        out = self.model(images, landmarks, ref_images)
        data = self.seq2batch(out['out_mid'], out['result'], landmarks_trimmed, gt_images, mask, mask_sum, lines)
        out_mid, result, landmarks, gt_images, mask, mask_sum, lines = data
        loss = self.get_masked_loss(result, gt_images[:, :, 128:, 64: -64], mask, mask_sum, key='image_loss')
        if self.opt.reg_lips > 0:
            mask, result, out_mid  = self.pad(mask, result, out_mid)
            out_predict_result = result * mask + (1 - mask) * gt_images
            out_predict_mid = out_mid * mask + (1 - mask) * gt_images
            predict_landmarks_mid = self.get_landmarks(out_predict_mid)
            predict_landmarks_result = self.get_landmarks(out_predict_result)
            loss_open = self.get_lips_detection_loss(predict_landmarks_mid, landmarks, 'lips_open_mid')
            loss_open += self.get_lips_detection_loss(predict_landmarks_result, landmarks, 'lips_open')
            loss += self.opt.reg_lips * loss_open / 2.
            if self.opt.reg_lips_center > 0:
                loss_lines = self.get_lips_center_loss(predict_landmarks_mid, lines, 'lips_center_mid')
                loss_lines += self.get_lips_center_loss(predict_landmarks_result, lines, 'lips_center')
                loss += self.opt.reg_lips_center * loss_lines
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


def main():
    opt = OptionsLipsGeneratorSeq(tag='all').load()
    model = TrainLipsGeneratorSeq(opt)
    model.train()



if __name__ == '__main__':
    main()
