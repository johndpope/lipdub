from models import models_utils
from custom_types import *
import constants
from utils import files_utils


class VisemeWDS(Dataset):

    def __len__(self):
        return self.w_base.shape[0]

    def get_random_viseme_dir(self):
        alpha = torch.rand((1,)).item() * 3
        select = torch.randint(16, (1,)).item()
        viseme_dir = self.dirs[select] * alpha
        return viseme_dir, alpha, select

    def get_random_pose(self):
        alpha = -6 + torch.rand((1,)).item() * 12
        return self.w_pose * alpha

    def get_random_viseme(self, item):
        item = item % len(self)
        base_vec = self.w_base[item].clone()
        pose = self.get_random_pose()
        viseme_dir, alpha, select = self.get_random_viseme_dir()
        base_vec[:10] += viseme_dir[:10]
        base_vec += pose
        return base_vec, select, alpha

    def __getitem__(self, item):
        base_vec = self.w_base[item]
        driven_vec = self.w_base[torch.randint(len(self), (1,)).item()].clone()
        pose_a = self.get_random_pose()
        viseme_a = self.get_random_viseme_dir()[0]
        pose_b = self.get_random_pose() / 2.
        viseme_b = self.get_random_viseme_dir()[0]
        driven_vec[:10] += viseme_b[:10]
        driven_vec += pose_b
        base_vec_in, base_vec_out = base_vec.clone(), base_vec.clone()
        base_vec_in[:10] += viseme_a[:10]
        base_vec_out[:10] += viseme_b[:10]
        base_vec_in += pose_a
        base_vec_out += pose_a
        return base_vec_in, driven_vec, base_vec_out

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
        self.w_base = w_plus[-1]
        self.dirs[12] = dirs[12]
        self.dirs[4] = dirs[4]


@models_utils.torch_no_grad
def generate_viseme_ds():
    num_items = 10000
    out_root = f"{constants.MNT_ROOT}/viseme_ds/"
    model = stylegan_wrapper.get_model()
    ds = VisemeWDS()
    select_all = torch.zeros(num_items, 2)
    logger = train_utils.Logger().start(num_items)
    for i in range(num_items):
        w, select, alpha = ds.get_random_viseme(i)
        image = model(w.unsqueeze(0).cuda()).clip(0, 1)
        select_all[i, 0] = select
        select_all[i, 1] = alpha
        image = image[:, :, 512:, 256: 768]
        files_utils.save_image(image, f'{out_root}/{i:06d}')
        logger.reset_iter()
    files_utils.save_pickle(select_all, f'{out_root}/viseme_info')
    logger.stop()



if __name__ == '__main__':
    from models import stylegan_wrapper
    from utils import train_utils
    generate_viseme_ds()
