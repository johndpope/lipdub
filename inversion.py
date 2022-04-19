import lpips
from custom_types import *
import constants
from utils import files_utils, train_utils
from models import stylegan_wrapper
from options import OptionsA2S, OptionsSC


class SingleVideoDS(Dataset):

    def load_image(self, item: int):
        path = f"{self.root}/image_{self.frame_numbers[item]:08d}"
        image = files_utils.load_np(path)
        image = image.astype(np.float32) / 127.5 - 1
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

    def __getitem__(self, item: int):
        image = self.load_image(item)
        return image

    def __len__(self):
        return len(self.frame_numbers)

    @property
    def name(self):
        return self.opt.video_name

    @property
    def frames_per_item(self):
        return self.opt.frames_per_item

    def __init__(self, opt: OptionsA2S):
        self.opt = opt
        self.root = f"{constants.CACHE_ROOT}/{self.name}/"
        self.metadata = files_utils.load_pickle(f"{self.root}/metadata")
        paths = files_utils.collect(self.root, '.npy', prefix=f'image_')
        self.frame_numbers = list(map(lambda x: int(x[1].split('_')[-1]), paths))


class InversionInTime:

    def get_predict(self, w):
        out = self.model(w)
        out_correct = self.s2s.forward256(out)
        return out_correct

    def invert_single(self):
        num_iters = 100000
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').to(CUDA(0)).eval()
        target = self.ds[0].unsqueeze(0).to(CUDA(0))
        w = self.w_base.clone().to(CUDA(0))
        w.requires_grad = True
        optimizer = torch.optim.SGD([w], lr=1e-2)
        self.logger.start(num_iters)
        for i in range(num_iters):
            optimizer.zero_grad()
            out = self.get_predict(w)
            loss = self.loss_fn_vgg(out, target) + nnf.l1_loss(out, target)
            loss.backward()
            optimizer.step()
            self.logger.reset_iter('loss', loss)
            if (i + 1) % 1000 == 0:
                image = torch.cat((target, out), dim=3)
                files_utils.imshow(image)

    def __init__(self):
        opt = OptionsA2S(video_name='me_lips_crop', stylegan_size=256)
        self.model = stylegan_wrapper.StyleGanWrapper(opt).to(CUDA(0)).eval()
        self.s2s, _ = train_utils.model_lc(OptionsSC())
        self.s2s = self.s2s.eval()
        self.ds = SingleVideoDS(opt)
        w_base = files_utils.load_np(f"{constants.CACHE_ROOT}/ffhq_w")
        self.w_base = torch.from_numpy(w_base).unsqueeze(0)
        self.loss_fn_vgg: Optional[lpips.LPIPS] = None
        self.logger = train_utils.Logger()


if __name__ == '__main__':
    InversionInTime().invert_single()
