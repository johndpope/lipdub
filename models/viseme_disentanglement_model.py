from custom_types import *
from models import models_utils, transformers, stylegan_wrapper
import math
import options


class VisemeClassifier(models_utils.Model):

    def encode(self, image):
        x = self.vit(image)
        return x[:, 0]

    def forward(self, image):
        x = self.encode(image)
        x = self.ln(x)
        return x

    def __init__(self,opt: options.OptionsVisemeClassifier):
        super(VisemeClassifier, self).__init__()
        self.vit = transformers.VisionTransformer(128, 16, opt.hidden_dim, opt.num_layers, opt.num_heads, 3, extra_token=True)
        self.ln = nn.Linear(opt.hidden_dim, 16)


class VisemeDisentanglement(models_utils.Model):

    def pti_forward(self, w_base, w_drive, generator):
        with torch.no_grad():
            w_base = w_base + self.positional_embeddings_base
            w_drive = w_drive + self.positional_embeddings_drive
            out = self.style_decoder(w_base, w_drive)
        image = generator(out)
        return image

    def forward(self, w_base, w_drive, generator: Optional[stylegan_wrapper.StyleGanWrapper] = None, inference: bool = False):
        w_base = w_base + self.positional_embeddings_base
        w_drive = w_drive + self.positional_embeddings_drive
        w_new = out = self.style_decoder(w_base, w_drive)
        if generator is None:
            return out
        if inference:
            w_new = w_base.clone()
            w_new[:, :6] = out[:, :6]
        image = generator(w_new)
        return image

    def __init__(self, opt: options.OptionsDisentanglementViseme):
        super(VisemeDisentanglement, self).__init__()
        positional_embeddings_base = torch.zeros(1, 18, 512)
        positional_embeddings_drive = torch.zeros(1, 18, 512)
        self.positional_embeddings_base = nn.Parameter(positional_embeddings_base)
        torch.nn.init.normal_(
            self.positional_embeddings_base.data,
            0.0,
            1. / math.sqrt(512),
        )
        self.positional_embeddings_drive = nn.Parameter(positional_embeddings_drive)
        torch.nn.init.normal_(
            self.positional_embeddings_drive.data,
            0.0,
            1. / math.sqrt(512),
        )
        self.style_decoder = transformers.CombTransformer(512, opt.num_heads, opt.num_layers,  512)