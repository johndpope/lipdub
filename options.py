from __future__ import annotations
import abc
import os
import pickle

import constants
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
        self.tag = 'disentanglement_fixed'
        self.num_heads = 8
        self.num_layers = 6
        self.lambda_identity_reg = .5
        self.fill_args(kwargs)


class OptionsVisemeClassifier(BaseOptions):

    @property
    def model_name(self) -> str:
        return 'viseme_classifier'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        super(OptionsVisemeClassifier, self).__init__(**kwargs)
        self.tag = 'base'
        self.num_heads = 8
        self.num_layers = 6
        self.hidden_dim = 512
        self.batch_size = 128
        self.fill_args(kwargs)


class OptionsLipsDetection(BaseOptions):

    @property
    def model_name(self) -> str:
        return 'lips_detection'

    def __init__(self, **kwargs):
        super(OptionsLipsDetection, self).__init__(**kwargs)
        self.tag = 'vit'
        self.batch_size = 128
        self.patch_size = 15
        self.hidden_dim = 768
        self.num_layers = 8
        self.num_heads = 8
        self.data_dir = f'{constants.FaceForensicsRoot}processed_frames/'
        self.fill_args(kwargs)


class OptionsLipsGenerator(BaseOptions):

    @property
    def model_name(self) -> str:
        return 'conditional_lips_generator'

    def __init__(self, **kwargs):
        super(OptionsLipsGenerator, self).__init__(**kwargs)
        self.tag = 'all_encoder_light_cat'
        self.data_dir = f'{constants.FaceForensicsRoot}processed_frames/'
        self.batch_size = 32
        self.num_layers = 8
        self.num_heads = 8
        self.reg_lips = 10
        self.reg_constructive = 0
        self.unpaired = 0
        self.num_ids = -1
        self.train_visual_encoder = False
        self.color_jitter = False
        self.z_token = False
        self.is_light = True
        self.draw_lips_lines = True
        self.reg_lips_center = 0
        self.concat_ref = True
        self.pretrained_tag = f''
        self.reverse_input = False
        self.res = 256
        self.fill_args(kwargs)


class OptionsLipsGeneratorSeq(OptionsLipsGenerator):

    @property
    def model_name(self) -> str:
        return 'seq_lips_generator'

    def __init__(self, **kwargs):
        super(OptionsLipsGeneratorSeq, self).__init__(**kwargs)
        self.image_seq = 5
        self.lips_seq = 11
        self.res = 128
        self.is_light = False
        self.lr_decay_every = 1
        self.data_dir = f'{constants.FaceForensicsRoot}processed_frames_all/'
        self.fill_args(kwargs)


class OptionsVisemeUnet(BaseOptions):

    @property
    def model_name(self) -> str:
        return 'unet'


    def __init__(self, **kwargs):
        super(OptionsVisemeUnet, self).__init__(**kwargs)
        self.tag = 'triple_decoder'
        self.blocks = 3
        self.depth = 5
        self.res = 512
        self.batch_size = 10
        self.conditional_dim = 512
        self.classification_loss = 0
        self.train_viseme_encoder = True
        self.train_mask_decoder = True
        self.train_lips_decoder = True
        self.norm_type = 'instant'
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
    defaults = {'train_viseme_encoder': False, 'train_mask_decoder': False,
                'train_lips_decoder': False, 'norm_type': 'batch',
                'data_dir': f'{constants.FaceForensicsRoot}processed_frames/',
                'reg_lips': 1., 'unpaired': 0, 'pretrained_path': '',
                'num_ids': -1, 'color_jitter': True,  'z_token': False,
                'is_light': False, 'train_visual_encoder': False, 'draw_lips_lines': False,
                'reg_lips_center': 0, 'reg_constructive': 0, 'reverse_input': False, 'concat_ref': False,
                'res': 256
                }
    for key, item in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, item)
    return opt