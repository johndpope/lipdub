from custom_types import *
from train import train_disentanglement
from options import OptionsDisentanglementViseme
from utils import files_utils
import constants


class VisemePtiDS(Dataset):

    def __getitem__(self, item):
        image = files_utils.load_np(''.join(self.paths[item]))
        image = torch.from_numpy(image).float()
        image = image / 127.5 - 1
        image = image.permute(2, 0, 1)
        return image

    def __len__(self):
        return len(self.paths)

    def __init__(self, root: str, max_num_frames):
        paths = files_utils.collect(root, '.npy')
        self.paths = [path for path in paths if 'crop_' in path[1] or 'image_' in path[1]]
        if max_num_frames < len(self.paths):
            self.paths = self.path[:max_num_frames]


class PtiWithDisentanglement(train_disentanglement.TrainVisemeDisentanglement):

    def init_dataloader(self):
        ds = VisemePtiDS(self.video_root, self.max_num_frames)
        dl = DataLoader(ds, num_workers=0 if DEBUG else 4, shuffle=not constants.DEBUG, drop_last=True,
                        batch_size=4 if DEBUG else self.opt.batch_size)
        return ds, dl

    def between_epochs(self, epoch, scores):
        if scores['loss'] < self.best_score:
            self.best_score = scores['loss']
            files_utils.save_model(self.stylegan, f'{self.opt.cp_folder}/pti/{self.video_name}')

    def train_iter(self, data):
        images = self.prepare_data(data)
        with torch.no_grad():
            w_drive = w_base = self.e4e.encode(images)
        out = self.model.pti_forward(w_base, w_drive, self.stylegan)
        loss = self.get_loss(out, images)
        loss.backward()
        self.optimizer.step()
        self.logger.stash_iter('loss', loss)

    def __init__(self, opt: OptionsDisentanglementViseme, video_root: str, video_name: str, max_num_frames: int):
        self.video_name = video_name
        self.max_num_frames = max_num_frames
        self.video_root = video_root
        super(PtiWithDisentanglement, self).__init__(opt)
        self.base_training = False
        self.stylegan.train(True)
        self.model.eval()
        self.optimizer = Optimizer(self.stylegan.parameters(),  betas=(.9, 0.999), lr=3e-5)


def main():
    trainer = PtiWithDisentanglement(OptionsDisentanglementViseme().load(), f'{constants.DATA_ROOT}/processed/',
                                      'obama', 1000)
    trainer.train()


if __name__ == '__main__':
    main()
