import torch

from custom_types import *
from models import models_utils, transformers
import math
import options


class VisemeDisentanglement(models_utils.Model):

    def pti_forward(self, w_base, w_drive, generator):
        with torch.no_grad():
            w_base = w_base + self.positional_embeddings_base
            w_drive = w_drive + self.positional_embeddings_drive
            out = self.style_decoder(w_base, w_drive)
        image = generator(out)
        return image

    def forward(self, w_base, w_drive, generator):
        w_base = w_base + self.positional_embeddings_base
        w_drive = w_drive + self.positional_embeddings_drive
        out = self.style_decoder(w_base, w_drive)
        image = generator(out)
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