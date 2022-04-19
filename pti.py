import torch.optim.optimizer

from models import models_utils
from custom_types import *
from models import stylegan_wrapper
from utils import files_utils, image_utils, train_utils
import lpips
import options
import constants


class PtiDS(Dataset):

    def __getitem__(self, item):
        image = files_utils.load_np(''.join(self.paths[item]))
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1)
        w = self.w_plus[item]
        return w, image

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def interpolate_paths(paths, w_plus):

        def get_frame_number(frame):
            return int(frame[1].split('_')[1])

        num_frames = get_frame_number(paths[-1]) + 1
        out_path = []
        out_w = torch.zeros(num_frames, *w_plus.shape[1:])
        cur_frame = 0
        for i in range(num_frames):
            cur_frame_number = get_frame_number(paths[cur_frame])
            out_path.append(paths[cur_frame])
            if cur_frame_number == i:
                out_w[i] = w_plus[cur_frame]
                cur_frame += 1
            else:
                prev_frame_number = get_frame_number(paths[cur_frame - 1])
                alpha = float(i - prev_frame_number) / (cur_frame_number - prev_frame_number)
                out_w[i] = w_plus[cur_frame - 1] + alpha * (w_plus[cur_frame] - w_plus[cur_frame - 1])
        return out_path, out_w


    def __init__(self, folder, trim_start=0, num_frames=2500, with_interpolation=False, prefix='crop'):
        paths = files_utils.collect(folder, '.npy')
        paths = [path for path in paths if prefix in path[1]]
        w_plus = files_utils.load_pickle(f'{folder}/e4e_w_plus')[:, 0]
        if with_interpolation:
            paths, w_plus = self.interpolate_paths(paths, w_plus)
        if trim_start > 0:
            paths = paths[trim_start:]
            w_plus = w_plus[trim_start:]
        if len(paths) > num_frames > 0:
            paths = paths[:num_frames]
            w_plus = w_plus[:num_frames]
        self.paths = paths
        self.trim_start = trim_start
        self.w_plus = w_plus

# ## Architecture
# lpips_type = 'alex'
# first_inv_type = 'e4e'
#
# ## Locality regularization
# latent_ball_num_of_samples = 1
# locality_regularization_interval = 1
# use_locality_regularization = True
# regularizer_l2_lambda = 0.1
# regularizer_lpips_lambda = 0.1
# regularizer_alpha = 30
#
# ## Loss
# pt_l2_lambda = 1
# pt_lpips_lambda = 1
#
# ## Steps
# max_pti_steps = 50
# first_inv_steps = 50
#
# ## Optimization
# pti_learning_rate = 3e-5
# first_inv_lr = 5e-3
# stitching_tuning_lr = 3e-4
# pti_adam_beta1 = 0.9
# lr_rampdown_length = 0.25
# lr_rampup_length = 0.05
# use_lr_ramp = False
# def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch):
#     loss = 0.0
#
#     if hyperparameters.pt_l2_lambda > 0:
#         l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
#         loss += l2_loss_val * hyperparameters.pt_l2_lambda
#     if hyperparameters.pt_lpips_lambda > 0:
#         loss_lpips = self.lpips_loss(generated_images, real_images)
#         loss_lpips = torch.squeeze(loss_lpips)
#
#         loss += loss_lpips * hyperparameters.pt_lpips_lambda
#
#     if use_ball_holder and hyperparameters.use_locality_regularization:
#         ball_holder_loss_val = self.space_regularizer.space_regularizer_loss(new_G, w_batch, log_name,
#                                                                              use_wandb=self.use_wandb)
#         loss += ball_holder_loss_val
#
#     return loss, l2_loss_val, loss_lpips
class PTI:

    @property
    def original_model(self) -> stylegan_wrapper.StyleGanWrapper:
        if self.original_model_ is None:
            self.original_model_ = stylegan_wrapper.StyleGanWrapper(self.opt).to(self.opt.device).eval()
        return self.original_model_

    def get_morphed_w_code(self, sample, w):
        interpolation_direction = sample - w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2, dim=[1, 2])
        direction_to_move = 30 * interpolation_direction / interpolation_direction_norm[: , None, None]
        w_moved = w + direction_to_move
        return w_moved

    def space_regularizer_loss(self, w):
        z_samples = torch.randn(1, 512).to(self.device)
        w_samples = self.original_model.generator.style(z_samples)
        territory_indicator_ws = self.get_morphed_w_code(w_samples, w)
        out = self.model(territory_indicator_ws)
        with torch.no_grad():
            target = self.original_model(territory_indicator_ws)
        loss = self.lpips_loss(out, target).mean() + nnf.mse_loss(out, target)
        return loss

    def iter(self, w_plus, target) -> TS:
        out = self.model(w_plus)
        if out.shape[2] != target.shape[2] or out.shape[3] != target.shape[3]:
            out = nnf.interpolate(out, size=target.shape[2:], mode='bilinear', align_corners=True)
        loss_lpips = self.lpips_loss(out, target).mean()
        loss_l2 = nnf.mse_loss(out, target)
        # loss_ball_holder = self.space_regularizer_loss(w_plus)
        self.logger.stash_iter('l2', loss_l2, 'lpips', loss_lpips)
        return loss_l2 + loss_lpips, out

    def simple_transform(self, image):
        image = image.astype(np.float32)
        image = image / 127.5 - 1
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)

    def load_w_plus_image(self, name):
        path = f'{constants.CHECKPOINTS_ROOT}pti/{name}'
        w_plus = files_utils.load_pickle(path + '_e4e').to(self.device)
        image = files_utils.load_image(path)
        image = self.simple_transform(image)
        return w_plus, image

    def load_ws(self, name: str, model_name):
        w_s_path = files_utils.collect(f'./assets/cache/obama_images_stich/', '.pkl')
        self.model = files_utils.load_model(self.model, f'{constants.CHECKPOINTS_ROOT}pti/{model_name}', self.device).eval()
        w_plus = [files_utils.load_pickle(''.join(path)) for path in w_s_path]
        w_plus = torch.stack(w_plus).to(self.device).detach()
        return w_plus, w_s_path

    @staticmethod
    def interpolate(w_plus: T, num_mid: int):
        w_many = []
        cur_w = w_plus[0]
        alpha = torch.linspace(0, 1, num_mid + 1, device=w_plus.device)[:-1]
        for i in range(1, w_plus.shape[0] + 1):
            next_w = w_plus[i % w_plus.shape[0]]
            seq = cur_w + (next_w - cur_w) * alpha[:, None, None]
            w_many.append(seq)
            cur_w = next_w
        w_many = torch.cat(w_many).unsqueeze(1)
        return w_many

    @staticmethod
    def add_pose(w: T) -> T:
        w_pose = torch.load("./encoder4editing/editings/interfacegan_directions/pose.pt").to(w.device)
        alpha = torch.linspace(-6, 6, w.shape[0], device=w.device)
        w_pose = w_pose[None] * alpha[:, None, None, None]
        return w + w_pose

    @models_utils.torch_no_grad
    def infer_seq(self, name, model_name, num_mid, with_pose:bool = False):
        w_pluses, w_s_path = self.load_ws(name, model_name)
        images = []
        w_pluses = self.interpolate(w_pluses, num_mid)
        if with_pose:
            w_pluses = self.add_pose(w_pluses)
        root = w_s_path[0][0]
        for i in range(w_pluses.shape[0]):
            w_plus = w_pluses[i]
            out = self.model(w_plus)
            image = files_utils.image_to_display(out)
            images.append(image)
        image_utils.gif_group(images, f'{root}/pti_{name}_mouth{"_pose" if with_pose else ""}', 20.)

    @models_utils.torch_no_grad
    def infer(self, name, folder):
        model_name = folder.split('/')[-1]
        w_pluses, w_s_path = self.load_ws(name, model_name)
        w_plus_base = files_utils.load_pickle(f'{folder}/e4e_w_plus')[int(name.split('_')[1]) -290].to(self.device)
        # image_base = self.model(w_plus_base.unsqueeze(0))
        # files_utils.imshow(image_base)
        # return
        for i in range(w_pluses.shape[0]):
            path = w_s_path[i]
            w_plus = w_pluses[i]
            out_a = self.model(w_plus)
            w_plus[0, -10:] = w_plus_base[-10:]
            out_b = self.model(w_plus)
            files_utils.imshow(torch.cat((out_a, out_b), dim=3))
            # files_utils.save_image(out, f'{path[0]}/pti_{path[1]}')

    @staticmethod
    def init_alphas(w_target, w_anchors):
        # with torch.no_grad():
        #     diff = (w_target[:, None] - w_anchors[None, :]) ** 2
        #     diff = diff.sum(-1).sum(-1)
        #     select = diff.argmin(-1)
        #     alpha = torch.zeros(w_target.shape[0], w_anchors.shape[0], device=w_target.device)
        #     for i in range(select.shape[0]):
        #         alpha[i, select] = np.log(w_anchors.shape[0] - 1)
        # alpha = alpha.clone()
        alpha = torch.randn(w_target.shape[0],  *w_anchors.shape, device=w_target.device)
        alpha.requires_grad = True
        return alpha

    def mirror_descent(self, name, folder):
        num_iters = 1000
        th = 0.01
        model_name = folder.split('/')[-1]
        ds = PtiDS(folder)
        w_target, w_s_path = self.load_ws(name, model_name)
        w_target = w_target[:, 0]
        base_images = self.model(w_target).detach()
        w_anchors = ds.w_plus.to(self.device).detach()
        alpha = self.init_alphas(w_target, w_anchors)
        optimizer = torch.optim.Adam([alpha], lr=.1)
        self.logger.start(num_iters)
        for i in range(num_iters):
            optimizer.zero_grad()
            alpha_p = torch.softmax(alpha, dim=1)
            w_new = torch.einsum('band,and->bnd', alpha_p, w_anchors)
            # out_images = self.model(w_new)
            loss = nnf.mse_loss(w_new, w_target) + nnf.l1_loss(w_new, w_target)
            # loss = nnf.mse_loss(out_images, base_images) + self.lpips_loss(out_images, base_images).mean()
            loss.backward()
            optimizer.step()
            self.logger.reset_iter('loss', loss)
            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    out_images = self.model(w_new)
                    select = torch.randint(base_images.shape[0], size=(1,)).item()
                    image = torch.cat((base_images[select], out_images[select]), dim=-1).detach().cpu()
                    files_utils.imshow(image)
            # if loss < th:
            #     break
        # files_utils.save_pickle(alpha.detach().clone(), f'{folder}/alphas_{name}')

    @models_utils.torch_no_grad
    def output_inversion(self, model_name, folder_source, folder_target, with_interpolation=True):
        seq_name = folder_target.split('/')
        if seq_name[-1] == '':
            seq_name = seq_name[-2]
        else:
            seq_name = seq_name[-1]
        ds_source = PtiDS(folder_source)
        ds_target = PtiDS(folder_target, with_interpolation=with_interpolation, prefix='crop')
        images = []
        self.model = files_utils.load_model(self.model, f'{constants.CHECKPOINTS_ROOT}pti/{model_name}',
                                            self.device).eval()
        for i in range(len(ds_target)):
            w_source = ds_source[i][0]
            w_target = ds_target[i][0]
            w_target[-8:] = w_source[-8:]
            out = self.model(w_target.unsqueeze(0).to(self.device)).detach().cpu()[0]
            out = out.clamp(-1, 1)
            out = (out.permute(1, 2, 0) + 1) * 127.5
            out = out.numpy().astype(np.uint8)
            images.append(out)

            # if (i + 1) % 40 == 0:
            #     files_utils.save_image(images[-1], f'{constants.CHECKPOINTS_ROOT}pti/seq_genie/orig_{i:03d}')


        image_utils.gif_group(images, f'{constants.CHECKPOINTS_ROOT}pti/{seq_name}/{seq_name}', 24)


    def train_seq(self, folder):
        ds = PtiDS(folder)
        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0 if DEBUG else 4)
        num_iters = 5000
        th = 0.01
        name = folder.split('/')[-1]
        optimizer = Optimizer(self.model.parameters(), betas=(.9, 0.999), lr=3e-5)
        self.logger.start(num_iters)
        for i in range(num_iters):
            counter = total_loss = 0
            for j, (w_plus, target) in enumerate(loader):
                w_plus, target = w_plus.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss, out = self.iter(w_plus, target)
                total_loss += loss
                counter += 1
                loss.backward()
                optimizer.step()
            self.logger.reset_iter()
            total_loss = float(total_loss) / counter
            if total_loss < th:
                break
            files_utils.save_model(self.model, f'{constants.CHECKPOINTS_ROOT}pti/{name}')
        self.logger.stop()

    def train(self, name: str):
        w_plus, target = self.load_w_plus_image(name)
        num_iters = 10000
        th = 0.01
        optimizer = Optimizer(self.model.parameters(),  betas=(.9, 0.999), lr=3e-5)
        self.logger.start(num_iters)
        files_utils.imshow(target)
        for i in range(num_iters):
            optimizer.zero_grad()
            loss, out = self.iter(w_plus, target)
            loss.backward()
            optimizer.step()
            self.logger.reset_iter()
            if (i + 1) % 100 == 0:
                files_utils.imshow(out)
            if loss < th:
                break
        files_utils.save_model(self.model, f'{constants.CHECKPOINTS_ROOT}pti/{name}')

    def __init__(self):
        self.opt = options.OptionsA2S()
        self.original_model_ = None
        self.device = self.opt.device
        self.model = stylegan_wrapper.StyleGanWrapper(self.opt).to(self.opt.device)
        self.model.train()
        self.logger = train_utils.Logger()
        self.lpips_loss = lpips.LPIPS().to(self.device)


def main():
    from moviepy import editor
    # pti = PTI()
    # pti.output_inversion('processed', '/home/ahertz/projects/StyleFusion-main/assets/processed',
    #                      f'/home/ahertz/projects/StyleFusion-main/assets/101_beigefox_front_comp_v017/', False)


    # seq_name = '101_purpledino_front_comp_v019_obama'
    # audio = '101_purpledino_front_comp_v019'
    video_clip = editor.VideoFileClip(f'{constants.DATA_ROOT}preview/0002_101_beigefox_front_comp_v017.mp4')
    audio_clip = editor.AudioFileClip( f'/home/ahertz/projects/StyleFusion-main/assets/101_beigefox_front_comp_v017/101_beigefox_front_comp_v017.wav')
    # audio_clip = editor.AudioFileClip(f'{constants.DATA_ROOT}raw_videos/{audio}.wav')
    audio_clip = editor.CompositeAudioClip([audio_clip])
    video_clip.audio = audio_clip
    video_clip.write_videofile(f'{constants.DATA_ROOT}preview/0002_101_beigefox_front_comp_v017_sound.mp4')

    return




    # pti.infer_seq('processed', 20, with_pose=True)




if __name__ == '__main__':
    main()
