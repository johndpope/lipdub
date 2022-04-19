from custom_types import *
from options import OptionsA2S
from models import transformers, models_utils


class VisionTransformer(nn.Module):

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = torch.cat((x, self.const_embedding.repeat(x.shape[0], 1, 1)), dim=1)
        x = self.transformer(x)
        x = x[:, -1]
        return x

    def __init__(self, input_resolution: int, patch_size: int, hidden_dim: int, layers: int, heads: int,
                 input_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(patch_size, patch_size),
                               stride=(patch_size, patch_size), bias=False)

        scale = hidden_dim ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2, hidden_dim))
        self.const_embedding = nn.Parameter(scale * torch.randn(1, 1, hidden_dim))
        self.ln_pre = nn.LayerNorm(hidden_dim)
        self.transformer = transformers.Transformer(hidden_dim, heads, layers)


class Audio2Style(models_utils.Model):

    def forward(self, audio, images, select, style2image):
        b, s, c, h, w = images.shape
        images = images.reshape(b * s, c, h, w)
        vec_images = self.image2vec(images).reshape(b , s, -1)
        vec_audio = self.audio2vec(audio)
        vec_audio = self.audio_encoder(vec_audio)
        style = self.style_decoder(vec_images, vec_audio)
        select = select.unsqueeze(-1).unsqueeze(-1).expand(b, 1, style.shape[-1])
        vec = style.gather(1, select)[:, 0]
        vec = self.h_to_w(vec).reshape(b, 18, 512)
        out = style2image(vec)
        out = self.rgb(out)
        return out

    def __init__(self, opt: OptionsA2S):
        super(Audio2Style, self).__init__()
        self.image2vec = VisionTransformer(opt.image_input_resolution, 16, opt.image_h, opt.num_layers_image,
                                           opt.num_heads_image, opt.in_channels)
        self.audio2vec = nn.Linear(opt.audio_multiplier * 13, opt.audio_h)
        self.h_to_w = nn.Linear(opt.image_h, 18 * 512)
        self.audio_encoder = transformers.Transformer(opt.audio_h, opt.num_heads_audio, opt.num_layers_audio)
        self.style_decoder = transformers.CombTransformer(opt.image_h, opt.num_heads_image, opt.num_layers_image, opt.audio_h)
        self.rgb = nn.Conv2d(3, 3, (3, 3), (1, 1), 1)


# def callback(stylegan):
#
#     def func_(vec):
#         b, s, h = vec.shape
#         select = select.unsqueeze(-1).unsqueeze(-1).expand(b, 1, h)
#         vec = vec.gather(1, select)[:, 0]
#         vec = vec.reshape(b, 18, -1)
#         out = stylegan(vec)
#         return out
#
#     return func_


if __name__ == '__main__':
    from models import stylegan_wrapper
    images = torch.rand(5, 30, 3, 64, 64).cuda()
    audio = torch.rand(5, 50, 13 * 4).cuda()
    select = torch.randint(30, (5,)).cuda()
    model = Audio2Style(OptionsA2S()).cuda()
    stylegan = stylegan_wrapper.StyleGanWrapper(OptionsA2S()).cuda()
    out = model(images, audio, select, stylegan)
    print(out.shape)
