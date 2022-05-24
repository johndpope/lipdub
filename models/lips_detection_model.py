import math
from custom_types import *
import options
from models import models_utils, transformers
from torchvision import models
from models.diffusion_factory import get_conditional_unet, get_visual_encoder


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

    def forward(self, lips, z_id: Optional[T] = None):
        lips = self.project_viseme(lips)
        b, _, h = lips.shape
        if z_id is None:
            lips = torch.cat((torch.zeros(b, 1, h, device=lips.device), lips), dim=1)
            lips = lips + self.embedding
        else:
            lips = lips + self.embedding
            lips = torch.cat((z_id.unsqueeze(1), lips), dim=1)
        out = self.transformer(lips)
        return out[:, 0]

    def __init__(self, opt):
        super(LipsEncoder, self).__init__()
        scale = 512 ** -0.5
        self.project_viseme = nn.Linear(2, 512)
        self.embedding = nn.Parameter((torch.randn(21 - int(opt.z_token), 512) * scale).unsqueeze(0))
        self.transformer = transformers.Transformer(512, opt.num_heads, opt.num_layers)


class LipsEncoderSeq(nn.Module):

    def forward(self, lips):
        b, s, p, d = lips.shape
        lips = lips.reshape(b * s, p, d)
        base_encoding = self.lips_encoder(lips).reshape(b, s, -1)
        base_encoding = torch.cat((torch.zeros(b, 1, 512, device=lips.device), base_encoding), dim=1)
        base_encoding = base_encoding + self.embedding
        out = self.transformer(base_encoding)
        return out[:, 0]

    def __init__(self, opt: options.OptionsLipsGeneratorSeq):
        super(LipsEncoderSeq, self).__init__()
        opt.z_token = False
        scale = 512 ** -0.5
        self.lips_encoder = LipsEncoder(opt)
        self.embedding = nn.Parameter((torch.randn(opt.lips_seq + 1, 512) * scale).unsqueeze(0))
        self.transformer = transformers.Transformer(512, opt.num_heads, opt.num_layers)


class ConditionalLipsGenerator(models_utils.Model):

    @models_utils.torch_no_grad
    def get_new_z_id(self):
        z = self.z_id.weight.data.mean(0)
        z = z.clone().detach().unsqueeze(0)
        return z

    def forward_after_token(self, image, lips, z_id, return_id=False):
        if self.z_token or self.visual_encoder:
            cond_lips = self.encoder(lips, z_id)
            out = self.generator(image, None, cond_lips).pred
        else:
            cond_lips = self.encoder(lips)
            out = self.generator(image, z_id, cond_lips).pred
        if return_id:
            return out, z_id
        return out

    def inference_forward(self, image, lips, z_id):
        b = lips.shape[0]
        if z_id.shape[0] != b:
            z_id = z_id.repeat(b, 1)
        return self.forward_after_token(image, lips, z_id)

    def encode_images(self, images: T):
        out = self.visual_encoder(images)
        if self.opt.reg_constructive:
            out = nnf.normalize(out, 2, 1)
        return out

    def forward(self, image, lips, items: Optional[T] = None, return_id=False):
        if self.z_id is not None:
            if items is None:
                z_id = self.get_new_z_id().repeat(image.shape[0], 1)
            else:
                z_id = self.z_id(items)
        elif self.visual_encoder is not None:
            z_id = self.encode_images(items)
        else:
            z_id = None
        if self.opt.concat_ref:
            image = torch.cat((image, items), dim=1)
        return self.forward_after_token(image, lips, z_id, return_id=return_id)

    def __init__(self, opt: options.OptionsLipsGenerator):
        super(ConditionalLipsGenerator, self).__init__()
        self.opt = opt
        if opt.train_visual_encoder:
            self.visual_encoder = get_visual_encoder(opt)
            self.z_id = None
        else:
            self.visual_encoder = None
            if opt.z_token > 0:
                self.z_id = nn.Embedding(opt.num_ids, 512)
                torch.nn.init.normal_(
                    self.z_id.weight.data,
                    0.0,
                    1. / math.sqrt(512),
                )
            else:
                self.z_id = None
        self.z_token = opt.z_token
        self.generator = get_conditional_unet(opt)
        self.encoder = LipsEncoder(opt)


class LipsGeneratorSeq(models_utils.Model):

    def encode_images(self, images: T):
        out = self.visual_encoder(images)
        if self.opt.reg_constructive:
            out = nnf.normalize(out, 2, 1)
        return out

    def forward(self, images, lips, ref):
        b, s, c, h, w = images.shape
        images = images.reshape(b, s * c, w, h)
        cond_lips = self.lips_encoder(lips)
        cond_visual = self.encode_images(ref)
        out_a = self.generator_a(images, None, cond_visual).pred
        out_b = self.generator_b(out_a, None, cond_lips).pred
        out_a = out_a.reshape(b, s, c, w, h)
        out_b = out_b.reshape(b, s, c, w, h)
        return {"cond_visual": cond_visual, "out_mid": out_a, "result": out_b}

    @property
    def generator_parameters(self):
        return list(self.generator_a.parameters()) + list(self.generator_b.parameters())

    def __init__(self, opt: options.OptionsLipsGeneratorSeq):
        super(LipsGeneratorSeq, self).__init__()
        self.opt = opt
        self.visual_encoder = get_visual_encoder(opt)
        self.lips_encoder = LipsEncoderSeq(opt)
        self.generator_a = get_conditional_unet(opt)
        self.generator_b = get_conditional_unet(opt)


if __name__ == '__main__':
    model = LipsGeneratorSeq(options.OptionsLipsGeneratorSeq())
    lips = torch.rand(5, 11, 20, 2)
    images = torch.rand(5, 5, 3, 128, 128)
    ref = torch.rand(5, 3, 128, 128)
    out = model(images, lips, ref)
    # model = ConditionalLipsGenerator(options.OptionsLipsGenerator())
    # x = torch.rand(5, 3, 256, 256)
    # y = torch.rand(5, 20, 2)
    # items = torch.arange(5)
    # cond_image = x
    # out = model(x, y, cond_image)
    # print(out.shape)
