from models import models_utils
from custom_types import *
from options import OptionsVisemeUnet


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding=padding,
        bias=bias,
        groups=groups)


def up_conv2x2(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=(2, 2),
            stride=(2, 2))
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(1, 1),
        groups=groups,
        stride=(1, 1))


class UpConvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, residual=True, batch_norm=True, transpose=True, concat=True):
        super(UpConvD, self).__init__()
        self.concat = concat
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv2 = []
        self.up_conv = up_conv2x2(in_channels, out_channels, transpose=transpose)
        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        self.bn = []
        for _ in range(blocks):
            if batch_norm:
                self.bn.append(nn.BatchNorm2d(out_channels))
            else:
                self.bn.append(nn.InstanceNorm2d(out_channels))
        self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, from_up, from_down):
        return self.forward(from_up, from_down)

    def forward(self, from_up, from_down):
        from_up = self.up_conv(from_up)
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up
        x1 = nnf.relu(self.conv1(x1))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = nnf.relu(x2)
            x1 = x2
        return x2


class DownConvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, residual=True, batch_norm=True):
        super(DownConvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.bn = None
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        self.bn = []
        for _ in range(blocks):
            if batch_norm:
                self.bn.append(nn.BatchNorm2d(out_channels))
            else:
                self.bn.append(nn.InstanceNorm2d(out_channels))
        self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = nnf.relu(self.conv1(x))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = nnf.relu(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool


class UnetDecoder(nn.Module):

    def forward(self, x, encoder_outs=None):
        x = self.project_cond(x)
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool)
        if self.conv_final is not None:
            x = self.conv_final(x)
        x = self.last_act(x)
        return x

    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, conditional_dim=0, last_act=torch.tanh):
        super(UnetDecoder, self).__init__()
        self.last_act = last_act
        self.conv_final = None
        self.up_convs = []
        outs = in_channels
        if conditional_dim > 0:
            self.project_cond = nn.Conv2d(in_channels + conditional_dim, in_channels, (1, 1))
        else:
            self.project_cond = lambda x: x
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat)
            self.up_convs.append(up_conv)
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpConvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)


class UnetEncoder(nn.Module):

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs

    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True):
        super(UnetEncoder, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)


class UnetEncoderDecoder(models_utils.Model):

    def forward(self, synthesized, condition: TN = None):
        image_code, before_pool = self.encoder(synthesized)
        out = {}
        if self.viseme_encoder is not None and condition is not None:
            condition, _ = self.encoder(condition)
            condition = condition.view(*condition.shape[:2], -1).mean(-1)
        if condition is not None:
            condition = condition.unsqueeze(2).unsqueeze(2)
            condition = condition.expand(*condition.shape[:2], *image_code.shape[2:])
            image_code = torch.cat((image_code, condition), dim=1)
        reconstructed = self.decoder(image_code, before_pool)
        out['reconstructed'] = reconstructed
        if self.mask_decoder is not None:
            mask = self.mask_decoder(image_code)
            out['mask'] = mask
        if self.lips_decoder is not None:
            lips = self.lips_decoder(image_code)
            out['lips'] = lips
        return out

    def __init__(self, opt: OptionsVisemeUnet):
        super(UnetEncoderDecoder, self).__init__()
        bath_norm = (opt.norm_type=='batch')
        if opt.train_viseme_encoder:
            self.viseme_encoder = UnetEncoder(in_channels=3, depth=5, blocks=opt.blocks,
                                   start_filters=32, residual=True, batch_norm=bath_norm)
            conditional_dim = 32 * 2 ** (5 - 1)
        else:
            conditional_dim = opt.conditional_dim
            self.viseme_encoder = None
        self.encoder = UnetEncoder(in_channels=3, depth=opt.depth, blocks=opt.blocks,
                                   start_filters=32, residual=True, batch_norm=bath_norm)
        self.decoder = UnetDecoder(in_channels=32 * 2 ** (opt.depth - 1),
                                   out_channels=3, depth=opt.depth, blocks=opt.blocks, residual=True,
                                   batch_norm=bath_norm, transpose=True, concat=True,
                                   conditional_dim=conditional_dim)
        if opt.train_mask_decoder:
            self.mask_decoder = UnetDecoder(in_channels=32 * 2 ** (opt.depth - 1),
                                       out_channels=1, depth=opt.depth, blocks=opt.blocks, residual=True,
                                       batch_norm=bath_norm, transpose=True, concat=False,
                                       conditional_dim=conditional_dim, last_act=torch.sigmoid)
        else:
            self.mask_decoder = None

        if opt.train_lips_decoder:
            self.lips_decoder = UnetDecoder(in_channels=32 * 2 ** (opt.depth - 1),
                                            out_channels=3, depth=opt.depth, blocks=opt.blocks, residual=True,
                                            batch_norm=bath_norm, transpose=True, concat=False,
                                            conditional_dim=conditional_dim)
        else:
            self.lips_decoder = None



if __name__ == '__main__':
    model = UnetEncoderDecoder(OptionsVisemeUnet())
    image = torch.rand(5, 3, 512, 512)
    image_cond = torch.rand(5, 3, 128, 128)
    cond = torch.randn(5, 512)
    out = model(image, image_cond)
