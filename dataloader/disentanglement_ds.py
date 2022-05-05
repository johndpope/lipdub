import PIL.Image
import torch

from models import models_utils
from custom_types import *
import constants
from utils import files_utils
import torchvision
from torchvision.transforms import functional as tsf
import options
import random


class BatchTransform:

    @staticmethod
    def init_transform():
        color_jitter = torchvision.transforms.ColorJitter(hue=0.2, brightness=.8, saturation=.8, contrast=.8)
        color_jitter = torchvision.transforms.RandomApply([color_jitter], .8)
        kernel_size = int(.05 * 128)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(.1, 2.))
        blur = torchvision.transforms.RandomApply([blur], .3)
        perspective_aug = torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=.5, fill=(255, 255, 255),
                                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(128),
            torchvision.transforms.RandomHorizontalFlip(.5),
            perspective_aug,
            color_jitter,
            blur,
            torchvision.transforms.ToTensor()
        ])
        return transform

    def __call__(self, *images, train=True):
        images = [self.resize(image) for image in images]
        # if torch.rand((1,)).item() < .5 and train:
        #     images = [tsf.hflip(image) for image in images]
        if torch.rand((1,)).item() < .8 and train:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.contrast, self.color_jitter.saturation, self.color_jitter.hue)
            out = []
            for img in images:
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness_factor is not None:
                        img = tsf.adjust_brightness(img, brightness_factor)
                    elif fn_id == 1 and contrast_factor is not None:
                        img = tsf.adjust_contrast(img, contrast_factor)
                    elif fn_id == 2 and saturation_factor is not None:
                        img = tsf.adjust_saturation(img, saturation_factor)
                    elif fn_id == 3 and hue_factor is not None:
                        img = tsf.adjust_hue(img, hue_factor)
                out.append(img)
            images = out
        # if train:
        # [self.blur(image) for image in images]
        images = [self.to_tensor(image) * 2 - 1 for image in images]
        return images

    def __init__(self, opt: options.OptionsVisemeUnet):
        self.resize = torchvision.transforms.Resize(opt.res)
        self.to_tensor = torchvision.transforms.ToTensor()
        self.color_jitter = torchvision.transforms.ColorJitter(hue=0.2, brightness=.8, saturation=.8, contrast=.8)
        kernel_size = int(.05 * 128)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(.1, 2.))
        self.blur = torchvision.transforms.RandomApply([blur], .3)


class VisemeWDS(Dataset):

    def __len__(self):
        return self.w_base.shape[0]

    def get_random_viseme_dir(self):
        if np.random.random((1,)) < .2:
            select_b = select_a = 12
        else:
            select_a = torch.randint(16, (1,)).item()
            select_b = torch.randint(16, (1,)).item()
        magnitude = torch.rand((1,)).item() * 3
        alpha = torch.rand((1,)).item()

        viseme_dir = magnitude * (self.dirs[select_a] * alpha + self.dirs[select_b] * (1 - alpha))
        return viseme_dir, alpha, select_a, select_b

    def get_random_pose(self):
        alpha = -6 + torch.rand((1,)).item() * 12
        return self.w_pose * alpha

    def get_random_viseme(self, item):
        item = item % len(self.w_base)
        base_vec = self.w_base[item].clone()
        pose = self.get_random_pose()
        viseme_dir, alpha, select_a, select_b = self.get_random_viseme_dir()
        base_vec[:10] += viseme_dir[:10]
        base_vec += pose
        return base_vec, select_a, select_b, alpha

    def get_pair(self, item):
        item = item % len(self.w_base)
        base_vec = self.w_base[item].clone()
        base_vec_b = self.w_base[item].clone()
        driven_vec = self.w_base[torch.randint(len(self.w_base), (1,)).item()].clone()
        driven_vec_b = driven_vec.clone()
        pose = self.get_random_pose()
        pose_b = self.get_random_pose()
        viseme_dir, alpha, select_a_a, select_a_b = self.get_random_viseme_dir()
        viseme_dir_b, alpha_b, select_b_a, select_b_b = self.get_random_viseme_dir()
        base_vec[:10] += viseme_dir[:10]
        driven_vec[:10] += viseme_dir[:10]
        driven_vec += pose_b
        base_vec += pose
        base_vec_b[:10] += viseme_dir_b[:10]
        driven_vec_b[:10] += viseme_dir_b[:10]
        base_vec_b += pose
        driven_vec_b += pose_b
        return base_vec, base_vec_b, driven_vec, driven_vec_b, (select_a_a, select_a_b), alpha, (select_b_a, select_b_b), alpha_b


    # def __getitem__(self, item):
    #     base_vec = self.w_base[item]
    #     driven_vec = self.w_base[torch.randint(len(self), (1,)).item()].clone()
    #     pose_a = self.get_random_pose()
    #     viseme_a = self.get_random_viseme_dir()[0]
    #     pose_b = self.get_random_pose() / 2.
    #     viseme_b = self.get_random_viseme_dir()[0]
    #     driven_vec[:10] += viseme_b[:10]
    #     driven_vec += pose_b
    #     base_vec_in, base_vec_out = base_vec.clone(), base_vec.clone()
    #     base_vec_in[:10] += viseme_a[:10]
    #     base_vec_out[:10] += viseme_b[:10]
    #     base_vec_in += pose_a
    #     base_vec_out += pose_a
    #     return base_vec_in, driven_vec, base_vec_out

    def load_driven(self, item: int):
        driven_image = files_utils.load_image(''.join(self.driven_images_paths[item]))
        driven_image = PIL.Image.fromarray(driven_image)
        driven_image = self.viseme_aug(driven_image)
        # driven_image = torch.from_numpy(driven_image).float().permute(2, 0, 1)
        driven_image = driven_image * 2 - 1
        select, alpha = self.driven_info[item]
        viseme_dir = self.dirs[int(select)] * alpha
        return driven_image, viseme_dir


    @staticmethod
    def merge(w_id, w_pose, w_viseme):
        w_merge = w_id.clone()
        w_merge[:10] += w_viseme[:10]
        w_merge += w_pose
        return w_merge

    def get_single_seq(self, num_interval, interval_len):
        id_base = self.w_base[torch.randint(len(self), (1,)).item()].clone()
        pose_base_ = [self.get_random_pose() for _ in range(num_interval // 4 + 1)]
        pose_base = [pose_base_[0].unsqueeze(0)]
        alpha = torch.linspace(0, 1, 5)[1:]
        for i in range(0, len(pose_base_) - 1):
            delta = pose_base_[i + 1] - pose_base_[i]
            seq = pose_base_[i] + alpha[:, None, None] * delta[None]
            pose_base.append(seq)
        pose_base = torch.cat(pose_base)
        viseme_base = [self.get_random_viseme_dir()[0] for _ in range(num_interval)]
        key_frames = [self.merge(id_base, pose_base[i], viseme_base[i]) for i in range(num_interval)]
        out = [key_frames[0].unsqueeze(0)]
        alpha = torch.linspace(0, 1, interval_len + 1)[1:]
        for i in range(0, num_interval - 1):
            delta = key_frames[i + 1] - key_frames[i]
            seq = key_frames[i] + alpha[:, None, None] * delta[None]
            out.append(seq)
        return torch.cat(out)

    def get_seq(self):
        return self.get_single_seq(20, 30), self.get_single_seq(20, 30)

    def __getitem__(self, item):
        base_vec = self.w_base[item]
        driven_vec = self.w_base[torch.randint(len(self), (1,)).item()].clone()
        pose_a = self.get_random_pose()
        viseme_a = self.get_random_viseme_dir()[0]
        pose_b = self.get_random_pose() / 2.
        viseme_b = self.get_random_viseme_dir()[0]
        driven_vec = self.merge(driven_vec, pose_b, viseme_b)
        base_vec_in = self.merge(base_vec, viseme_a, pose_a)
        base_vec_out = self.merge(base_vec, viseme_b, pose_a)
        return base_vec_in, driven_vec, base_vec_out

    @staticmethod
    def init_transform(train: bool = True):
        color_jitter = torchvision.transforms.ColorJitter(hue=0.2, brightness=.8, saturation=.8, contrast=.8)
        color_jitter = torchvision.transforms.RandomApply([color_jitter], .8)
        kernel_size = int(.05 * 128)
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(.1, 2.))
        blur = torchvision.transforms.RandomApply([blur], .3)
        perspective_aug = torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=.5,
                                                                   interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        if train:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(128),
                torchvision.transforms.RandomHorizontalFlip(.5),
                perspective_aug,
                color_jitter,
                blur,
                torchvision.transforms.ToTensor()
            ])

        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(128),
                torchvision.transforms.ToTensor()
            ])
        return transform

    @staticmethod
    def gram_s(vecs):
        vecs = vecs.view(vecs.shape[0], -1)
        vecs = nnf.normalize(vecs, 2, -1)
        out = [vecs[0]]
        for i in range(1, vecs.shape[0]):
            cur_vec = vecs[i]
            for j in range(len(out)):
                proj = (cur_vec * out[j]).sum()
                cur_vec = cur_vec - proj * out[j]
                cur_vec = cur_vec / cur_vec.norm(2)
            out.append(cur_vec)
        return torch.stack(out)

    def find_viseme_dir(self, w):
        device = w.device
        w_proj = w[0].cpu()
        w_proj = (w_proj - self.w_base[-1]).flatten()
        w_proj_norm = w_proj.norm(2)
        all_projections = torch.einsum('d,bd->b', w_proj, self.dirs_ortho)
        viseme_dir = (all_projections[:, None] * self.dirs_ortho).sum(0)
        # w_proj = w_proj / w_proj_norm
        all_dirs = self.dirs.view(16, -1)
        all_dirs_norm = all_dirs / all_dirs.norm(2, dim=1)[:, None]
        projection = torch.einsum('d,bd->b', w_proj, all_dirs_norm)
        max_projection_dir = projection.argmax()
        viseme_dir = projection[max_projection_dir, None] * all_dirs_norm[max_projection_dir]
        return viseme_dir.view(1, 18, 512).to(device)

    def __init__(self):
        self.w_pose = torch.load(f"{constants.PROJECT_ROOT}/encoder4editing/editings/interfacegan_directions/pose.pt").to(CPU)[0]
        valid = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_valid_v3')
        w_plus = []
        for i in range(17):
            w_ = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}_v3')[valid]
            w_plus.append(w_)
        w_plus_v1 = [files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}')[valid] for i in
                     range(17)]
        dirs_all = [w_plus_v1[i] - w_plus_v1[-1] for i in range(17)]
        dirs = torch.stack([dirs_all[i].mean(0) for i in range(17)], dim=0)

        dirs_all = [w_plus[i] - w_plus[-1][:200] for i in range(16)]
        self.dirs = torch.stack([dirs_all[i].mean(0) for i in range(16)], dim=0)
        self.base_vec = dirs[-1]
        self.w_base = w_plus[-1]
        self.dirs[12] = dirs[12]
        self.dirs[4] = dirs[4]
        self.dirs[:2] /= 2
        self.dirs_ortho = self.gram_s(self.dirs)
        self.driven_images_paths = files_utils.collect(f"{constants.MNT_ROOT}/viseme_ds/", '.png')
        self.driven_info = files_utils.load_pickle(f'{constants.MNT_ROOT}/viseme_ds/viseme_info')
        self.viseme_aug = self.init_transform()


class VisemeDS(Dataset):

    def __len__(self):
        return len(self.driven_images_paths)

    def __getitem__(self, item):
        driven_image = files_utils.load_image(''.join(self.driven_images_paths[item]))
        driven_image = PIL.Image.fromarray(driven_image)
        driven_image = self.viseme_aug(driven_image)
        driven_image = driven_image * 2 - 1
        label, _ = self.driven_info[item]
        return driven_image, int(label)

    def __init__(self):
        self.driven_images_paths = files_utils.collect(f"{constants.MNT_ROOT}/viseme_ds/", '.png')
        self.driven_info = files_utils.load_pickle(f'{constants.MNT_ROOT}/viseme_ds/viseme_info')
        self.viseme_aug = VisemeWDS.init_transform()


class DualVisemeDS(Dataset):

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        path = ''.join(self.images_paths[item])

        flip = torch.rand((1,)).item() < .5
        flip_inplace = torch.rand((1,)).item() < .5
        if flip:
            image_a = files_utils.load_image(path.replace('_a', '_c'))
            image_b = files_utils.load_image(path.replace('_a', '_d'))
            if flip_inplace:
                image_driven = files_utils.load_image(path)
                mask = files_utils.load_np(path.replace('_a.png', '_c_lips.npy'))
            else:
                image_driven = files_utils.load_image(path.replace('_a', '_b'))
                mask = files_utils.load_np(path.replace('_a.png', '_d_lips.npy'))
        else:
            image_a = files_utils.load_image(path)
            image_b = files_utils.load_image(path.replace('_a', '_b'))
            if flip_inplace:
                image_driven = files_utils.load_image(path.replace('_a', '_c'))
                mask = files_utils.load_np(path.replace('_a.png', '_a_lips.npy'))
            else:
                image_driven = files_utils.load_image(path.replace('_a', '_d'))
                mask = files_utils.load_np(path.replace('_a.png', '_b_lips.npy'))


        image_a = PIL.Image.fromarray(image_a)
        image_b = PIL.Image.fromarray(image_b)
        image_a, image_b = self.transform(image_a, image_b, train=self.is_train)
        image_driven = image_driven[512:, 256: 768]
        image_driven = PIL.Image.fromarray(image_driven)
        if self.is_train:
            image_driven = self.transform_viseme(image_driven)
        else:
            image_driven = self.transform_viseme_inference(image_driven)
        image_driven = image_driven * 2 - 1
        mask = torch.from_numpy(mask).float().view(1, 1, *mask.shape)
        mask = nnf.interpolate(mask, image_a.shape[1:])[0]
        # k = 15
        # mask = nnf.max_pool2d(mask, (k * 2 + 1, k * 2 + 1), (1, 1), padding=k)[0]
        label, alpha, label_b, alpha_b = self.labels[item]
        if flip_inplace:
            image_a, image_b = image_b, image_a
        else:
            label = label_b

        return image_a, image_b, image_driven, mask, mask.sum(), int(label)


    def __init__(self, opt):
        images_paths = files_utils.collect(f"{constants.MNT_ROOT}/viseme_pairs_ds/", '.png')
        self.is_train = True
        self.images_paths = [path for path in images_paths if '_a' in path[1]]
        self.labels = files_utils.load_pickle(f'{constants.MNT_ROOT}/viseme_pairs_ds/viseme_info')
        self.transform = BatchTransform(opt)
        self.transform_viseme = VisemeWDS.init_transform()
        self.transform_viseme_inference = VisemeWDS.init_transform(False)


@models_utils.torch_no_grad
def generate_viseme_ds():
    from models import stylegan_wrapper
    num_items = 10000
    out_root = f"{constants.MNT_ROOT}/viseme_ds/"
    model = stylegan_wrapper.get_model()
    ds = VisemeWDS()
    # select_all = torch.zeros(num_items, 2)
    logger = train_utils.Logger().start(num_items)
    select_all = files_utils.load_pickle(f'{out_root}/viseme_info')
    for i in range(num_items):
        w, select, alpha = ds.get_random_viseme(i)
        image = model(w.unsqueeze(0).cuda()).clip(-1, 1)
        select_all[i, 0] = select
        select_all[i, 1] = alpha
        image = image[:, :, 512:, 256: 768]
        # files_utils.save_image(image, f'{out_root}/{i:06d}')
        logger.reset_iter()
        if i > 500:
            break
    files_utils.save_pickle(select_all, f'{out_root}/viseme_info')
    logger.stop()


@models_utils.torch_no_grad
def generate_viseme_pairs():
    from models import stylegan_wrapper
    num_items = 10000
    out_root = f"{constants.MNT_ROOT}/viseme_pairs_ds/"
    model = stylegan_wrapper.get_model()
    ds = VisemeWDS()
    select_all = torch.zeros(num_items, 4)
    logger = train_utils.Logger().start(num_items)
    # select_all = files_utils.load_pickle(f'{out_root}/viseme_info')
    for i in range(num_items):
        base_vec, sec_vec, driven_a, driven_b, select, alpha, select_b, alpha_b = ds.get_pair(i)
        w = torch.stack((base_vec, sec_vec, driven_a, driven_b))
        image_a, image_b, driven_a, driven_b = model(w.cuda()).clip(-1, 1)
        select_all[i, 0] = select
        select_all[i, 1] = alpha
        select_all[i, 2] = select_b
        select_all[i, 3] = alpha_b
        # files_utils.imshow(image_a)
        # files_utils.imshow(image_b)
        files_utils.save_image(image_a, f'{out_root}/{i:06d}_a')
        files_utils.save_image(image_b, f'{out_root}/{i:06d}_b')
        files_utils.save_image(driven_a, f'{out_root}/{i:06d}_c')
        files_utils.save_image(driven_b, f'{out_root}/{i:06d}_d')
        logger.reset_iter()
        if (i + 1 % 500) == 0:
            files_utils.save_pickle(select_all, f'{out_root}/viseme_info')

    files_utils.save_pickle(select_all, f'{out_root}/viseme_info')
    logger.stop()



if __name__ == '__main__':
    from utils import train_utils
    print("start")
    ds = DualVisemeDS(options.OptionsVisemeUnet())
    for i in range(10):
        image_in, out_gt, driven, mask, mask_sum, viseme_select = ds[i]
    # generate_viseme_pairs()
    # generate_viseme_ds()

