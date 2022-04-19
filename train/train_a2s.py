import lpips
from utils import train_utils, files_utils, image_utils
from models import stylegan_wrapper, models_utils
from custom_types import *
import constants
from dataloader import single_video_ds
from options import Options


class TrainA2S:

    @models_utils.torch_no_grad
    def view(self):
        self.model.eval()
        self.stylegan.eval()
        self.dataset.to_linear()
        self.dataset.set_sound('/home/ahertz/projects/StyleFusion-main/assets/cache//obama_b/obama_b')
        self.dataset.set_sound_offset(1000)
        seq_len = min(len(self.dataset), 600)
        self.logger.start(seq_len)
        seq = []
        seq_gt = []
        for i in range(seq_len):
            data = self.dataset[i]
            data = [item.unsqueeze(0) if type(item) is T else torch.tensor([item]) for item in data]
            audio_input, images_input, image_gt, mask, select = self.prepare_data(data)
            out = self.model(audio_input, images_input, select, self.stylegan)
            out = image_gt * (1 - mask[:, None]) + out * mask[:, None]
            self.logger.reset_iter()
            seq.append(files_utils.image_to_display(out[0]))
            seq_gt.append(files_utils.image_to_display(image_gt[0]))
            # for j in range(out.shape[0]):
            # files_utils.save_image(out[0], f"{self.opt.cp_folder}/predict/{i:03d}")
            # files_utils.save_image(image_gt[0], f"{self.opt.cp_folder}/gt/{i:03d}")
            # image = torch.cat((out[0], image_gt[0]), dim=2)
            # files_utils.imshow(image)
        self.logger.stop()
        image_utils.gif_group(seq, f"{self.opt.cp_folder}/predict/seq_obama_b", 29.7)
        # image_utils.gif_group(seq_gt, f"{self.opt.cp_folder}/predict/gt", 29.7)

    def init_dataloader(self):
        ds = single_video_ds.SingleVideoDS(self.opt)
        dl = DataLoader(ds, num_workers=0 if DEBUG else 4, shuffle=not constants.DEBUG, drop_last=True,
                        batch_size=4 if DEBUG else self.opt.batch_size)
        return ds, dl

    def between_epochs(self, epoch, scores):
        if scores['loss'] < self.best_score:
            self.best_score = scores['loss']
            self.model.save()
            if self.opt.stylegan_ft:
                files_utils.save_model(self.stylegan, f'{self.opt.cp_folder}/stylegan.pt')
        if epoch > self.offset and (epoch - self.offset) % self.opt.lr_decay_every == 0:
            self.scheduler.step()

    def get_loss(self, predict, gt, mask):
        predict = predict * mask[:, None]
        gt = gt * mask[:, None]
        loss = nnf.l1_loss(predict, gt, reduction='none')
        loss = torch.einsum('bchw->b', loss) / (3 * torch.einsum('bhw->b', mask))
        loss = loss.mean()
        loss += self.loss_fn_vgg(predict, gt).mean()
        return loss
        # if predict.shape[2] < gt.shape[2]:
        #     gt = nnf.interpolate(gt, predict.shape[2], mode='bicubic', align_corners=True)
        # elif predict.shape[2] > gt.shape[2]:
        #     predict = nnf.interpolate(predict, gt.shape[2], mode='bicubic', align_corners=True)
        # loss = nnf.mse_loss(predict, gt) + nnf.l1_loss(predict, gt)
        # return loss

    def prepare_data(self, data: TS) -> TS:
        return tuple(map(lambda x: x.to(self.device), data))

    def train_iter(self, data):
        audio_input, images_input, image_gt, mask, select = self.prepare_data(data)
        self.optimizer.zero_grad()
        if self.opt.stylegan_ft:
            self.optimizer_st.zero_grad()
        out = self.model(audio_input, images_input, select, self.stylegan)
        loss = self.get_loss(out, image_gt, mask)
        loss.backward()
        self.optimizer.step()
        self.warm_up_scheduler.step()
        if self.opt.stylegan_ft:
            self.optimizer_st.step()
            self.warm_up_scheduler_st.step()
        self.logger.stash_iter('loss', loss)

    def train_epoch(self):
        self.model.train(True)
        self.stylegan.train(self.opt.stylegan_ft)
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

    def __init__(self, opt: Options):
        self.model, self.opt = train_utils.model_lc(opt)
        self.stylegan = stylegan_wrapper.StyleGanWrapper(opt).to(self.device).eval()
        if self.opt.stylegan_ft:
            files_utils.load_model(self.stylegan, f'{self.opt.cp_folder}/stylegan.pt', self.device, verbose=True)
            self.optimizer_st = Optimizer(self.stylegan.parameters(), lr=1e-8)
            self.warm_up_scheduler_st = train_utils.LinearWarmupScheduler(self.optimizer_st, 1e-3, opt.warm_up)
        else:
            self.optimizer_st = self.warm_up_scheduler_st = None
        self.optimizer = Optimizer(self.model.parameters(), lr=1e-7)
        self.warm_up_scheduler = train_utils.LinearWarmupScheduler(self.optimizer, 1e-4, opt.warm_up)
        self.dataset, self.data_loader = self.init_dataloader()
        self.offset = opt.warm_up // len(self.data_loader)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .9)
        self.logger = train_utils.Logger()
        self.best_score = 10000
        self.loss_fn_vgg: Optional[lpips.LPIPS] = None



def main():
    TrainA2S(Options().load()).view()


if __name__ == '__main__':
    main()
