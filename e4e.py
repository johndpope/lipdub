from custom_types import *
import constants
from argparse import Namespace
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# from encoder4editing.e4e_utils.common import tensor2im
from encoder4editing.e4e_models.psp import pSp
from models import models_utils
from utils import files_utils, image_utils, train_utils


class E4E:

    def invert_single(self, image, is_np=False):
        transformed_image = self.prepare_image(image, is_np)
        with torch.no_grad():
            images, latents = self.run_on_batch(transformed_image)
        return transformed_image[0], images[0], latents.detach().cpu()

    def prepare_image(self, image, is_np=False):
        if type(image) is str:
            if is_np:
                input_image = files_utils.load_np(image)
            else:
                input_image = files_utils.load_image(image)
        else:
            input_image = image
        input_image = Image.fromarray(input_image)
        transformed_image = self.transform(input_image)
        return transformed_image.unsqueeze(0)

    def display_alongside_source_image(self, result_image, source_image):
        res = np.concatenate([np.array(source_image.resize(self.resize_dims)),
                              np.array(result_image.resize(self.resize_dims))], axis=1)
        return Image.fromarray(res)

    def run_on_batch(self, inputs):
        images, latents = self.net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        return images, latents

    def run_alignment(self, image_path):
        from encoder4editing.e4e_utils.alignment import align_face
        aligned_image = align_face(filepath=image_path, predictor=self.predictor)
        return aligned_image

    @property
    def predictor(self):
        if self.predictor_ is None:
            import dlib
            self.predictor_ = dlib.shape_predictor("./weights/ffhq/shape_predictor_68_face_landmarks.dat")
        return self.predictor_

    def __call__(self, image, is_np=False):
        return self.invert_single(image, is_np)

    def encode(self, image):
        if image.shape[2] != self.resize_dims[0]:
            image = nnf.interpolate(image, self.resize_dims)
        latents = self.net.encode(image)
        return latents

    def decode(self, w_plus):
        return self.net(w_plus, input_code=False, resize=False, return_latents=True, only_decode=True)[0]

    def __init__(self):
        model_path = f'{constants.PROJECT_ROOT}/weights/ffhq/e4e_ffhq_encode.pt'
        self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.resize_dims = (256, 256)
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts= Namespace(**opts)
        self.net = pSp(opts)
        self.net.eval()
        self.net.cuda()
        self.predictor_ = None


class NpLoader(Dataset):


    def __init__(self, root: str, start = 0, end =-1):
        self.paths = files_utils.collect(root, '.npy', prefix='image_')
        if end < 0 :
            end = len(self.paths)
        end = min(end, len(self.paths))


def main(root: str, verbose=True):
    # name = '6014826820595920814'
    name = root.split('/')[-2]
    # paths = files_utils.collect('./assets/tmp/', '.png')
    face_contours = files_utils.load_np(f'{root}face_contours')
    paths = files_utils.collect(root, '.npy', prefix='image_')
    if verbose:
        logger = train_utils.Logger().start(len(paths))
    all_w = []
    all_images = []
    all_images_source = []
    for image_path in paths:
        input_image = files_utils.load_np(''.join(image_path))
        # contour = face_contours[int(image_path[1].split('_')[1])]
        # contour_mask = image_utils.contour2mask(contour, input_image)
        # mask = input_image.sum(-1) < 50
        # mask = np.logical_and(mask, ~contour_mask)
        # input_image[mask] = 126
        all_images_source.append(input_image)
        input_image = Image.fromarray(input_image)
        # image_path = ''.join(image_path)
        # input_image = run_alignment(image_path)
        # input_image.resize(resize_dims)
        transformed_image = transform(input_image)
        with torch.no_grad():
            images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
            result_image = images[0]
            result_image = result_image.permute(1, 2, 0)
            result_image = (result_image + 1) * 255. / 2.
            result_image = result_image.clip(0, 255)
            result_image = result_image.detach().cpu().numpy().astype(np.uint8)
            all_images.append(result_image)
            all_w.append(latents.detach().cpu())
        if verbose:
            logger.reset_iter()
    image_utils.gif_group(all_images, f"./assets/tmp/{name}_e4e", 20)
    image_utils.gif_group(all_images_source, f"./assets/tmp/{name}_source", 20)
    files_utils.save_np(torch.cat(all_w).cpu().detach().numpy(), f'{root}/w_plus')
        # out = display_alongside_source_image(tensor2im(result_image), input_image)
        # files_utils.imshow(out)
    if verbose:
        logger.stop()



        # files_utils.imshow(images[0])
        # files_utils.save_image(images[0], image + '_e4e')
        # files_utils.save_pickle(latents.detach().cpu(), image + '_e4e')


def invert_all():
    all_dirs = files_utils.collect(constants.LRS2_PROCESS, '.pkl', prefix='metadata')
    logger = train_utils.Logger().start(len(all_dirs))
    counter = 20
    for root in all_dirs:
        main(root[0], False)
        logger.reset_iter()
        counter -= 1
        if counter == 0:
            break
    logger.stop()


@models_utils.torch_no_grad
def invert_folder(folder, is_np: bool = True, export_video=False):
    e4e = E4E()
    latents = []
    images = []
    if is_np:
        paths = files_utils.collect(folder, '.npy')
    else:
        paths = files_utils.collect(folder, '.png')
    paths = [path for path in paths if 'crop' in path[1] or 'stich' in path[1]]
    logger = train_utils.Logger().start(len(paths))
    print(len(paths))
    for path in paths:
        if 'e4e' in path[1]:
            continue
        input_image, image, latent = e4e.invert_single(''.join(path), is_np)
        if export_video:
            images.append(files_utils.image_to_display(image))
        latents.append(latent.detach().cpu())
        logger.reset_iter()
    logger.stop()
    latents = torch.stack(latents)
    files_utils.save_pickle(latents, f'{folder}/e4e_w_plus')
    if export_video:
        name = folder.split('/')[-2]
        image_utils.gif_group(images, f'{constants.DATA_ROOT}/preview/{name}', 24)
        # files_utils.save_image(image, f'{path[0]}/e4e_{path[1]}')
        # files_utils.save_pickle(latent, f'{path[0]}/e4e_{path[1]}')


@models_utils.torch_no_grad
def invert_seq(folder):
    paths = files_utils.collect(folder, '.npy', prefix='image')
    w = []
    e4e = E4E()
    for path in paths:
        input_image, image, latent = e4e.invert_single(''.join(path), True)
        w.append(latent.cpu())
    w = torch.cat(w)
    files_utils.save_pickle(w, f'{folder}/e4e_w_plus')



if __name__ == '__main__':
    invert_folder(f'/home/ahertz/projects/StyleFusion-main/assets/dubbing/0002_101_beigefox_front_comp_v017/', export_video=True)
    # invert_seq('/home/ahertz/projects/StyleFusion-main/assets/cache/obama_a/')
    # input_image, _, latent = invert_single('/home/ahertz/projects/StyleFusion-main/assets/cache/obama_a/image_00002404', True)
    # files_utils.save_pickle(latent.detach().cpu(), f'{constants.CHECKPOINTS_ROOT}pti/image_00002404_e4e')
    # files_utils.save_image(input_image, f'{constants.CHECKPOINTS_ROOT}pti/image_00002404')
    # pose_dir = torch.load("./encoder4editing/editings/interfacegan_directions/pose.pt")
    # exit(0)
    # invert_folder(f"./assets/cache/stylesdf_images_stich/")
    # for i in range(7):
    #     invert_single(f"./assets/cache/tmp/stich_face_{i:02d}")



    # invert_single('./assets/cache/tmp/invert_face')
    # for i in range(0, 1):
    #     main(f'{constants.CACHE_ROOT}/exp_{i:02d}/', True)

    # inert_all()

