from __future__ import annotations
import abc
import os
import pickle
import constants as const
from custom_types import *


class BaseOptions(abc.ABC):

    def load(self):
        device = self.device
        if os.path.isfile(self.save_path):
            print(f'loading opitons from {self.save_path}')
            with open(self.save_path, 'rb') as f:
                options = pickle.load(f)
            options = backward_compatibility(options)
            options.device = device
            return options
        return self

    def save(self):
        if os.path.isdir(self.cp_folder):
            # self.already_saved = True
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @property
    def info(self) -> str:
        return f'{self.model_name}_{self.tag}'

    @property
    def cp_folder(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}'

    @property
    def save_path(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}/options.pkl'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    def __init__(self, **kwargs):
        self.device = CUDA(0)
        self.stylegan_size = 1024
        self.tag = 'obama_a_sanity'
        self.epochs = 2700
        self.batch_size = 8
        self.lr_decay = .5
        self.lr_decay_every = 500
        self.warm_up = 2000
        self.fill_args(kwargs)


class OptionsDisentanglementViseme(BaseOptions):

    @property
    def model_name(self) -> str:
        return 'disentanglement_viseme'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        super(OptionsDisentanglementViseme, self).__init__(**kwargs)
        self.tag = 'DisentanglementE4E'
        self.num_heads = 8
        self.num_layers = 6
        self.fill_args(kwargs)

class OptionsA2S(BaseOptions):

    @property
    def model_name(self) -> str:
        return 'audio2style'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        super(OptionsA2S, self).__init__(**kwargs)
        self.video_name = 'obama_a'
        self.audio_h = 256
        self.image_h = 256
        self.audio_multiplier = 4
        self.num_heads_image = 4
        self.num_layers_image = 4
        self.num_heads_audio = 4
        self.num_layers_audio = 4
        self.image_input_resolution = 64
        self.frames_per_item = 1
        self.audio_per_item = 30
        self.batch_size = 64
        self.stylegan_ft = True
        self.in_channels = 1
        self.fill_args(kwargs)


class OptionsSC(BaseOptions):

    @property
    def model_name(self) -> str:
        return 'style_correct'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        super(OptionsSC, self).__init__(**kwargs)
        self.tag = '256_512'


def backward_compatibility(opt: BaseOptions) -> BaseOptions:
    defaults = {}
    for key, item in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, item)
    return opt