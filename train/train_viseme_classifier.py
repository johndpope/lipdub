from utils import train_utils
from custom_types import *
import constants
from dataloader import disentanglement_ds
from options import OptionsVisemeClassifier


class TrainVisemeClassifier:

    def init_dataloader(self):
        ds = disentanglement_ds.VisemeDS()
        dl = DataLoader(ds, num_workers=0 if DEBUG else 8, shuffle=not constants.DEBUG, drop_last=True,
                        batch_size=8 if DEBUG else self.opt.batch_size)
        return ds, dl

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

    def train_iter(self, data):
        images, labels = self.prepare_data(data)
        self.optimizer.zero_grad()
        out = self.model(images)
        loss =  nnf.cross_entropy(out, labels)
        loss.backward()
        self.optimizer.step()
        self.warm_up_scheduler.step()
        self.logger.stash_iter('loss', loss)

    def train_epoch(self):
        self.model.train(True)
        self.logger.start(len(self.data_loader))
        for data in self.data_loader:
            self.train_iter(data)
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        for epoch in range(self.opt.epochs):
            log = self.train_epoch()
            self.between_epochs(epoch, log)

    @property
    def device(self):
        return self.opt.device

    def __init__(self, opt: OptionsVisemeClassifier):
        self.model, self.opt = train_utils.model_lc(opt)
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-4, opt.warm_up)
        self.dataset, self.data_loader = self.init_dataloader()
        self.offset = opt.warm_up // len(self.data_loader)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000


def main():
    opt = OptionsVisemeClassifier().load()
    model = TrainVisemeClassifier(opt)
    model.train()


if __name__ == '__main__':
    main()
