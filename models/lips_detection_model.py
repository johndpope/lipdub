import torch

from custom_types import *
import options
from models import models_utils, transformers
from torchvision import models
from models.diffusion_factory import get_conditional_unet

def get_resent():
    model = models.resnet50(True)
    return nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3
    )


class LipsDetectionModel(models_utils.Model):

    def encode(self, x):
        x = self.encoder(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        x = self.project_codes(x)
        x = x + self.embedding_encoder
        return x

    def forward(self, x):
        codes = self.encode(x)
        queries = self.embedding_decoder.unsqueeze(0).repeat(codes.shape[0], 1, 1)
        out = self.decoder(queries, codes)
        out = self.project(out)
        return out

    def __init__(self, opt: options.OptionsLipsDetection):
        super(LipsDetectionModel, self).__init__()
        self.encoder = get_resent()
        scale = opt.hidden_dim ** -0.5
        self.embedding_decoder = nn.Parameter(torch.randn(20, opt.hidden_dim) * scale)
        self.embedding_encoder = nn.Parameter((torch.randn(15 ** 2, opt.hidden_dim) * scale).unsqueeze(0))
        self.decoder = transformers.CombTransformer(opt.hidden_dim, opt.num_layers, opt.num_heads, opt.hidden_dim)
        self.project = nn.Linear(opt.hidden_dim, 2)
        self.project_codes = nn.Linear(1024, opt.hidden_dim)



class LipsEncoder(nn.Module):

    def forward(self, lips):
        lips = self.project_viseme(lips)
        b, _, h = lips.shape
        lips = torch.cat(( torch.zeros(b, 1, h, device=lips.device), lips), dim=1)
        lips = lips + self.embedding
        out = self.transformer(lips)
        return out[:, 0]

    def __init__(self, opt):
        super(LipsEncoder, self).__init__()
        scale = 512 ** -0.5
        self.project_viseme = nn.Linear(2, 512)
        self.embedding = nn.Parameter((torch.randn(21, 512) * scale).unsqueeze(0))
        self.transformer = transformers.Transformer(512, opt.num_heads, opt.num_layers)


class ConditionalLipsGenerator(models_utils.Model):

    def forward(self, image, lips):
        cond = self.encoder(lips)
        out = self.generator(image, None, cond).pred
        return out

    def __init__(self, opt):
        super(ConditionalLipsGenerator, self).__init__()
        self.generator = get_conditional_unet()
        self.encoder = LipsEncoder(opt)



if __name__ == '__main__':
    model = ConditionalLipsGenerator(options.OptionsLipsDetection())
    x = torch.rand(5, 3, 256, 256)
    y = torch.rand(5, 20, 2)
    out = model(x, y)
    print(out.shape)