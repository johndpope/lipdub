from custom_types import *
from options import BaseOptions, OptionsSC
import constants
from encoder4editing.e4e_models.stylegan2.model import Generator


class StyleGanWrapper(nn.Module):

    def forward_z(self, z, out_res=()):
        return self.generator([z], input_is_latent=False, randomize_noise=False, out_res=out_res)

    def forward(self, w):
        return self.generator([w], input_is_latent=True, randomize_noise=False)

    def reset_ckp(self, device):
        ckpt = torch.load(constants.stylegan_weights, map_location=device)
        self.generator.load_state_dict(ckpt['g_ema'], strict=False)

    def __init__(self, opt: BaseOptions):
        super(StyleGanWrapper, self).__init__()
        self.generator = Generator(opt.stylegan_size, 512, 8, channel_multiplier=2)
        self.reset_ckp(CPU)


def tensor2im(var: T):
    # var shape: (3, H, W)
    if var.dim() == 4:
        var = var[0]
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))


def get_model():
    return StyleGanWrapper(OptionsSC()).cuda()


if __name__ == '__main__':
    from PIL import Image
    from utils import files_utils
    model = StyleGanWrapper(OptionsSC()).cuda()
    x = torch.randn(1, 512).cuda()
    w = model.generator.style(x).unsqueeze(1).repeat(1, 18, 1)
    # x = torch.randn(1, 18, 512).cuda()
    out = model(w)
    files_utils.imshow(out)
    print(out.shape)
