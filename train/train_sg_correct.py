import lpips
from utils import train_utils, files_utils, image_utils
from models import stylegan_wrapper, models_utils
from custom_types import *
from options import OptionsSC


class TrainSC:

    def between_epochs(self, epoch, scores):
        if scores['loss_256'] + scores['loss_512'] < self.best_score:
            self.best_score = scores['loss_256'] + scores['loss_512']
            self.model.save()
        if (epoch + 1) % self.opt.lr_decay_every == 0:
            self.scheduler.step()

    def get_loss(self, predict, gt):
        gt = nnf.interpolate(gt, predict.shape[2], mode='bilinear', align_corners=True)
        loss = nnf.l1_loss(predict, gt)
        loss += self.loss_fn_vgg(predict, gt).mean()
        return loss

    def prepare_data(self, data: TS) -> TS:
        return tuple(map(lambda x: x.to(self.device), data))

    @models_utils.torch_no_grad
    def view(self):
        self.model.eval()
        z = torch.randn(self.opt.batch_size, 512, device=self.device).detach()
        out_256_base, out_512_base, out_1024 = self.stylegan.forward_z(z, out_res=(256, 512, 1024))
        out_256, out_512 = self.model(out_256_base, out_512_base)
        out_1024 = nnf.interpolate(out_1024, 256, mode='bilinear', align_corners=True)
        out_512 = nnf.interpolate(out_512, 256, mode='bilinear', align_corners=True)
        out_512_base = nnf.interpolate(out_512_base, 256, mode='bilinear', align_corners=True)
        out_512 = nnf.interpolate(out_512, 256, mode='bilinear', align_corners=True)
        all_out = torch.cat((out_256, out_512), dim=2)
        all_base = torch.cat((out_256_base, out_512_base), dim=2)
        all_gt = torch.cat((out_1024, out_1024), dim=2)
        all = torch.cat((all_gt, all_out, all_base), dim=3)
        for i in range(self.opt.batch_size):
            files_utils.imshow(all[i])

    def train_iter(self, data):
        z = data.to(self.device)
        self.optimizer.zero_grad()
        out_256, out_512, out_1024 = self.stylegan.forward_z(z, out_res=(256, 512, 1024))
        out_256, out_512 = self.model(out_256, out_512)
        loss_256 = self.get_loss(out_256, out_1024)
        loss_512 = self.get_loss(out_512, out_1024)
        (loss_256 + loss_512).backward()
        self.optimizer.step()
        self.logger.stash_iter('loss_256', loss_256, 'loss_512', loss_512)

    def train_epoch(self):
        self.model.train(True)
        self.logger.start(100)
        data = torch.randn(100, self.opt.batch_size, 512).detach()
        for i in range(100):
            self.train_iter(data[i])
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(self.device).eval()
        print("training")
        for epoch in range(self.opt.epochs):
            log = self.train_epoch()
            self.between_epochs(epoch, log)

    @property
    def device(self):
        return self.opt.device

    def __init__(self, opt: OptionsSC):
        self.model, self.opt = train_utils.model_lc(opt)
        self.stylegan = stylegan_wrapper.StyleGanWrapper(opt).to(self.device).eval()
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000
        self.loss_fn_vgg: Optional[lpips.LPIPS] = None


def main():
    TrainSC(OptionsSC().load()).view()


if __name__ == '__main__':
    main()
