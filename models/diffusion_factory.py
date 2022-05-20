import options
from custom_types import *
from models.diffusion_choises import *
from models.diffusion_config import BaseConfig
from models.diffusion_auto_encoder import BeatGANsAutoencConfig, BeatGANsEncoderConfig
from models.diffusion_unet import BeatGANsUNetModel
from models.diffusion_blocks import ScaleAt


class TrainConfig(BaseConfig):
    # random seed
    seed: int = 0
    train_mode: TrainMode = TrainMode.diffusion
    train_cond0_prob: float = 0
    train_pred_xstart_detach: bool = True
    train_interpolate_prob: float = 0
    train_interpolate_img: bool = False
    manipulate_mode: ManipulateMode = ManipulateMode.celebahq_all
    manipulate_cls: str = None
    manipulate_shots: int = None
    manipulate_loss: ManipulateLossType = ManipulateLossType.bce
    manipulate_znormalize: bool = False
    manipulate_seed: int = 0
    accum_batches: int = 1
    autoenc_mid_attn: bool = True
    batch_size: int = 16
    batch_size_eval: int = None
    beatgans_gen_type: GenerativeType = GenerativeType.ddim
    beatgans_loss_type: LossType = LossType.mse
    beatgans_model_mean_type: ModelMeanType = ModelMeanType.eps
    beatgans_model_var_type: ModelVarType = ModelVarType.fixed_large
    beatgans_rescale_timesteps: bool = False
    latent_infer_path: str = None
    latent_znormalize: bool = False
    latent_gen_type: GenerativeType = GenerativeType.ddim
    latent_loss_type: LossType = LossType.mse
    latent_model_mean_type: ModelMeanType = ModelMeanType.eps
    latent_model_var_type: ModelVarType = ModelVarType.fixed_large
    latent_rescale_timesteps: bool = False
    latent_T_eval: int = 1_000
    latent_clip_sample: bool = False
    latent_beta_scheduler: str = 'linear'
    beta_scheduler: str = 'linear'
    data_name: str = ''
    data_val_name: str = None
    diffusion_type: str = None
    dropout: float = 0.1
    ema_decay: float = 0.9999
    eval_num_images: int = 5_000
    eval_every_samples: int = 200_000
    eval_ema_every_samples: int = 200_000
    fid_use_torch: bool = True
    fp16: bool = False
    grad_clip: float = 1
    img_size: int = 64
    lr: float = 0.0001
    optimizer: OptimizerType = OptimizerType.adam
    weight_decay: float = 0
    model_conf = None
    model_name = None
    model_type: ModelType = None
    net_attn: Tuple[int] = None
    net_beatgans_attn_head: int = 1
    # not necessarily the same as the the number of style channels
    net_beatgans_embed_channels: int = 512
    net_resblock_updown: bool = True
    net_enc_use_time: bool = False
    net_enc_pool: str = 'adaptivenonzero'
    net_beatgans_gradient_checkpoint: bool = False
    net_beatgans_resnet_two_cond: bool = False
    net_beatgans_resnet_use_zero_module: bool = True
    net_beatgans_resnet_scale_at: ScaleAt = ScaleAt.after_norm
    net_beatgans_resnet_cond_channels: int = None
    net_ch_mult: Tuple[int] = None
    net_ch: int = 64
    net_enc_attn: Tuple[int] = None
    net_enc_k: int = None
    # number of resblocks for the encoder (half-unet)
    net_enc_num_res_blocks: int = 2
    net_enc_channel_mult: Tuple[int] = None
    net_enc_grad_checkpoint: bool = False
    net_autoenc_stochastic: bool = False
    net_latent_activation: Activation = Activation.silu
    net_latent_channel_mult: Tuple[int] = (1, 2, 4)
    net_latent_condition_bias: float = 0
    net_latent_dropout: float = 0
    net_latent_layers: int = None
    net_latent_net_last_act: Activation = Activation.none
    net_latent_net_type = None
    net_latent_num_hid_channels: int = 1024
    net_latent_num_time_layers: int = 2
    net_latent_skip_layers: Tuple[int] = None
    net_latent_time_emb_channels: int = 64
    net_latent_use_norm: bool = False
    net_latent_time_last_act: bool = False
    net_num_res_blocks: int = 2
    # number of resblocks for the UNET
    net_num_input_res_blocks: int = None
    net_enc_num_cls: int = None
    num_workers: int = 4
    parallel: bool = False
    postfix: str = ''
    sample_size: int = 64
    sample_every_samples: int = 20_000
    save_every_samples: int = 100_000
    style_ch: int = 512
    T_eval: int = 1_000
    T_sampler: str = 'uniform'
    T: int = 1_000
    total_samples: int = 10_000_000
    warmup: int = 0
    pretrain = None
    continue_from = None
    eval_programs: Tuple[str] = None
    # if present load the checkpoint from this path instead
    eval_path: str = None
    base_dir: str = 'checkpoints'
    use_cache_dataset: bool = False
    # to be overridden
    name: str = ''
    in_channels = 3
    model_out_channels = 3


    def make_model_conf(self):
        if self.model_name in [
                ModelName.beatgans_autoenc,
        ]:
            cls = BeatGANsAutoencConfig
            # supports both autoenc and vaeddpm
            if self.model_name == ModelName.beatgans_autoenc:
                self.model_type = ModelType.autoencoder
            else:
                raise NotImplementedError()

            latent_net_conf = None
            self.model_conf = cls(
                attention_resolutions=self.net_attn,
                channel_mult=self.net_ch_mult,
                conv_resample=True,
                dims=2,
                dropout=self.dropout,
                embed_channels=self.net_beatgans_embed_channels,
                enc_out_channels=self.style_ch,
                enc_pool=self.net_enc_pool,
                enc_num_res_block=self.net_enc_num_res_blocks,
                enc_channel_mult=self.net_enc_channel_mult,
                enc_grad_checkpoint=self.net_enc_grad_checkpoint,
                enc_attn_resolutions=self.net_enc_attn,
                image_size=self.img_size,
                in_channels=self.in_channels,
                model_channels=self.net_ch,
                num_classes=None,
                num_head_channels=-1,
                num_heads_upsample=-1,
                num_heads=self.net_beatgans_attn_head,
                num_res_blocks=self.net_num_res_blocks,
                num_input_res_blocks=self.net_num_input_res_blocks,
                out_channels=self.model_out_channels,
                resblock_updown=self.net_resblock_updown,
                use_checkpoint=self.net_beatgans_gradient_checkpoint,
                use_new_attention_order=False,
                resnet_two_cond=self.net_beatgans_resnet_two_cond,
                resnet_use_zero_module=self.
                net_beatgans_resnet_use_zero_module,
                resnet_cond_channels=self.net_beatgans_resnet_cond_channels,
            )
        else:
            raise NotImplementedError(self.model_name)

        return self.model_conf


def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 32
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf

def ffhq128_autoenc_base():
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.img_size = 128
    conf.net_ch = 128
    # final resolution = 8x8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()

    return conf


def ffhq128_autoenc(opt):
    conf = autoenc_base()
    conf.data_name = 'ffhqlmdb256'
    conf.img_size = 128
    if opt.is_light:
        conf.net_ch = 64
        conf.net_num_res_blocks = 1
        conf.net_enc_num_res_blocks = 1
        conf.net_ch_mult = (1, 2, 4, 4, 8)
        conf.net_enc_channel_mult = (1, 2, 4, 4, 8, 8)
    else:
        conf.net_ch = 128
        # final resolution = 8x8
        conf.net_ch_mult = (1, 1, 2, 3, 4)
        # final resolution = 4x4
        conf.net_enc_channel_mult = (1, 1, 2, 3, 4, 4)
    conf.in_channels = opt.image_seq * 3
    conf.model_out_channels = opt.image_seq * 3
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq256_autoenc(opt: options.OptionsLipsGenerator):
    conf = ffhq128_autoenc_base()
    conf.img_size = 256
    if opt.is_light:
        conf.net_ch = 32
        conf.net_num_res_blocks = 1
        conf.net_enc_num_res_blocks = 1
        conf.net_ch_mult = (1, 2, 2, 4, 4, 8)
        conf.net_enc_channel_mult = (1, 2, 2, 4, 4, 8, 8)
    else:
        conf.net_ch = 128
        conf.net_ch_mult = (1, 1, 2, 2, 4, 4)
        conf.net_enc_channel_mult = (1, 1, 2, 2, 4, 4, 4)
    if opt.concat_ref:
        conf.in_channels = 6
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 200_000_000
    conf.batch_size = 64
    conf.make_model_conf()
    conf.name = 'ffhq256_autoenc'
    return conf


def get_conditional_unet(opt):
    if opt.res == 256:
        conf = ffhq256_autoenc(opt)
    else:
        conf = ffhq128_autoenc(opt)
    model = BeatGANsUNetModel(conf.model_conf)
    return model


def get_visual_encoder(opt):
    if opt.res == 256:
        conf = ffhq256_autoenc(opt).model_conf
    else:
        conf = ffhq128_autoenc(opt).model_conf
    return BeatGANsEncoderConfig(
        image_size=conf.image_size,
        in_channels=3,
        model_channels=conf.model_channels,
        out_hid_channels=conf.enc_out_channels,
        out_channels=conf.enc_out_channels,
        num_res_blocks=conf.enc_num_res_block,
        attention_resolutions=(conf.enc_attn_resolutions
                               or conf.attention_resolutions),
        dropout=conf.dropout,
        channel_mult=conf.enc_channel_mult or conf.channel_mult,
        use_time_condition=False,
        conv_resample=conf.conv_resample,
        dims=conf.dims,
        use_checkpoint=conf.use_checkpoint or conf.enc_grad_checkpoint,
        num_heads=conf.num_heads,
        num_head_channels=conf.num_head_channels,
        resblock_updown=conf.resblock_updown,
        use_new_attention_order=conf.use_new_attention_order,
        pool=conf.enc_pool,
    ).make_model()



if __name__ == '__main__':
    conf = ffhq256_autoenc()
    model = BeatGANsUNetModel(conf.model_conf)
    # model = conf.model_conf.make_model()
    x_start = torch.rand(5, 3, 256, 256)
    x = torch.rand(5, 3, 256, 256)
    t = torch.arange(5) + 10
    cond = torch.rand(5, 512)
    out = model(x, t, cond)
    print(out.shape)
