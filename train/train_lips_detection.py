from utils import train_utils, files_utils
from custom_types import *
import constants
from dataloader import prcoess_face_forensics
from options import OptionsLipsDetection
from models import models_utils


class TrainLipsDetection:

    def init_dataloader(self):
        ds = prcoess_face_forensics.LipsLandmarksDS(self.opt.data_dir)
        split_path = f"{self.opt.cp_folder}/split.pkl"
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

    def prepare_data(self, data: Union[T, TS]) -> TS:
        if type(data) is T:
            return data.to(self.device)
        return tuple(map(lambda x: x.to(self.device), data))

    def train_iter(self, data, is_train: bool):
        images, landmarks = self.prepare_data(data)
        out = self.model(images)
        loss = nnf.mse_loss(out, landmarks)
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.warm_up_scheduler.step()
        self.logger.stash_iter('loss', loss)

    @models_utils.torch_no_grad
    def view(self):
        self.model.eval()
        for data in self.val_loader:
            images, landmarks = self.prepare_data(data)
            out = self.model(images)
            out = (out.cpu().numpy() + 1) * 120
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            images = ((images + 1) * 127.5).astype(np.uint8)
            landmarks = (landmarks.cpu().numpy() + 1) * 120
            for i in range(images.shape[0]):
                image_a = prcoess_face_forensics.draw_lips(images[i], out[i])
                image_b = prcoess_face_forensics.draw_lips(images[i], landmarks[i])
                image = np.concatenate((image_a, image_b), axis=1)
                files_utils.imshow(image)
            break

    def train_epoch(self, loader, is_train):
        self.model.train(is_train)
        self.logger.start(len(loader))
        for data in loader:
            self.train_iter(data, is_train)
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        for epoch in range(self.opt.epochs):
            self.train_epoch(self.data_loader, True)
            with torch.no_grad():
                log = self.train_epoch(self.val_loader, False)
            self.between_epochs(epoch, log)

    @property
    def device(self):
        return self.opt.device

    def __init__(self, opt: OptionsLipsDetection):
        self.model, self.opt = train_utils.model_lc(opt)
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-4, opt.warm_up)
        self.dataset, self.data_loader, self.val_loader = self.init_dataloader()
        self.offset = opt.warm_up // len(self.data_loader)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000


def main():
    opt = OptionsLipsDetection().load()
    model = TrainLipsDetection(opt)
    model.view()


if __name__ == '__main__':
    main()
