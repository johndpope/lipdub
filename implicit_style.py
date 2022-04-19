import torch

import draw
import face_detection
from models import siren
from utils import train_utils, files_utils
from custom_types import *
from style_fusion_simple import StyleFusionSimple, tensor2im
import constants


class ImplicitStyle(nn.Module):

    def forward(self, t: T):
        z = self.net(t)
        z = z + self.fixed_z
        return z

    def __init__(self, start_s):
        super(ImplicitStyle, self).__init__()
        self.net = siren.Siren(1, start_s.shape[1], 512, 5)
        self.fixed_z = nn.Parameter(start_s.clone().detach().cpu())


def init_dataloader():
    ds = face_detection.VidDs('assets/old/obama_cropped.mp4', True)
    print("init dl")
    dl = DataLoader(ds, num_workers=0 if DEBUG else 4, shuffle=not constants.DEBUG, drop_last=True, batch_size=8)
    return ds, dl


class Train:

    def between_epochs(self, epoch, scores):
        if (epoch + 1) % 100 == 0:
            self.scheduler.step()
        if scores['loss'] < self.best_score:
            self.best_score = scores['loss']
            files_utils.save_model(self.model, './assets/checkpoints/obama_seq.pt')

        # if epoch % self.opt.plot_every == 0:
        #     self.plot(epoch)

    def get_loss(self, predict, gt):
        if predict.shape[2] < gt.shape[2]:
            gt = nnf.interpolate(gt, predict.shape[2], mode='bicubic', align_corners=True)
        elif predict.shape[2] > gt.shape[2]:
            predict = nnf.interpolate(predict, gt.shape[2], mode='bicubic', align_corners=True)
        loss = nnf.mse_loss(predict, gt) + nnf.l1_loss(predict, gt)
        return loss

    def vis(self):
        files_utils.load_model(self.model, './assets/checkpoints/obama_seq.pt', self.device, verbose=True)
        self.model.eval()
        with torch.no_grad():
            for data in self.data_loader:
                out_image, gt_image = self.get_images(data)
                # data = self.ds[0]
                # data = data[0], data[1].unsqueeze(0)
                # out_z = self.model.fixed_z
                # s = draw.split_s(out_z)
                # out_image = self.stylegan.s_to_image(s)
                for i in range(out_image.shape[0]):
                    files_utils.imshow(tensor2im(out_image[i]))
                    files_utils.imshow(tensor2im(gt_image[i]))
                break

    def save_s(self):
        t = self.ds.t.to(self.device).permute(1, 0)
        self.model.eval()
        out_s = self.model(t).detach().cpu()
        files_utils.save_pickle(out_s, './assets/checkpoints/obama_seq_s')

    def get_images(self, data):
        image, t = data
        out_z = self.model(t.to(self.device))
        s = draw.split_s(out_z)
        out_image = self.stylegan.s_to_image(s)
        return out_image, image


    def train_single(self):

        def train_iter(data):
            self.optimizer.zero_grad()
            image, t = data
            out_z = self.model.fixed_z
            s = draw.split_s(out_z)
            out_image = self.stylegan.s_to_image(s)
            loss = self.get_loss(out_image, image.to(self.device))
            loss.backward()
            self.optimizer.step()
            self.logger.stash_iter('loss', loss)

        self.logger.start(100000)
        for i in range(100000):
            data = self.ds[0]
            data = data[0].unsqueeze(0), data[1].unsqueeze(0)
            train_iter(data)
            self.logger.reset_iter()
            if (i + 1) % 1000 == 0:
                files_utils.save_model(self.model, './assets/checkpoints/obama_single.pt')

    def train_iter(self, data):
        self.optimizer.zero_grad()
        out_image, image = self.get_images(data)
        loss = self.get_loss(out_image, image.to(self.device))
        loss.backward()
        self.optimizer.step()
        self.logger.stash_iter('loss', loss)

        # out_s = torch.rand(5, 9088, device=self.device)
        # s = draw.split_s(out_s)

    def train_epoch(self):
        self.logger.start(len(self.data_loader))
        for data in self.data_loader:
            self.train_iter(data)
            self.logger.reset_iter()
        return self.logger.stop()

    def train(self):
        print("training")
        for epoch in range(1000):
            log = self.train_epoch()
            self.between_epochs(epoch, log)

    def __init__(self):
        self.device  = CUDA(0)

        print("init style gan")
        self.stylegan = StyleFusionSimple("ffhq", "weights/ffhq/stylegan2-ffhq-config-f.pt",
                                          "weights/ffhq_weights.json")
        print("init ds")
        with torch.no_grad():
            s = self.stylegan.z_to_s(torch.zeros(1, 512, device=self.device))
            s = draw.cat_s(s)
        self.model = ImplicitStyle(s).to(self.device)
        files_utils.load_model(self.model, './assets/checkpoints/obama_seq.pt', self.device, verbose=True)
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-5)
        self.ds, self.data_loader = init_dataloader()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000


def main():
    Train().save_s()


if __name__ == '__main__':
    main()
