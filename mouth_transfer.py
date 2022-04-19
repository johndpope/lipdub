import imageio
from custom_types import *
from models import models_utils
from utils import files_utils, image_utils, train_utils
import mediapipe as mp
import constants
import cv2
import process_video
import e4e


FACEMESH_LIPS = V([(291, 409), (409, 270), (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40), (40, 185), (185, 61), (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                   (17, 314), (314, 405), (405, 321), (321, 375),
                   (375, 291), ], dtype=np.int32)


FACEMESH_FACE_OVAL = V([(10, 338), (338, 297), (297, 332), (332, 284),
                                (284, 251), (251, 389), (389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162), (162, 21), (21, 54),
                                (54, 103), (103, 67), (67, 109), (109, 10)], dtype=np.int32)



def find_transform(source, target, img_target, *sources):

    source = np.roll(source, 1, -1).reshape(-1, 1, 2)
    target = np.roll(target, 1, -1).reshape(-1, 1, 2)

    img_target = img_target[0].permute(1, 2, 0).numpy()
    mat_h, mask = cv2.findHomography(source, target)
    # matchesMask = mask.ravel().tolist()
    out = []
    for source in sources:
        img_source = source[0].permute(1, 2, 0).numpy()
        im_dst = cv2.warpPerspective(img_source,  mat_h, (img_target.shape[1], img_target.shape[0]))
        im_dst = torch.from_numpy(im_dst)
        if im_dst.dim() == 2:
            im_dst = im_dst.unsqueeze(0).unsqueeze(0)
        else:
            im_dst = im_dst.permute(2, 0, 1).unsqueeze(0)
        out.append(im_dst)
        # files_utils.imshow(im_dst)
        # files_utils.imshow(img_target)
    return out



def get_masked_mean_sdt(x, mask):
    eps = 1e-5
    mask_pixels = mask.permute(0, 2, 3, 1).flatten()
    x_pixels = x.permute(0, 2, 3, 1).view(-1, 3)[mask_pixels.bool()]
    mean_x = x_pixels.mean(0)
    std_x = x_pixels.std(0) + eps
    return mean_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) , std_x.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)


def masked_ada_in(x, y, mask_x, mask_y):
    mean_x, std_x = get_masked_mean_sdt(x, mask_x)
    mean_y, std_y = get_masked_mean_sdt(y, mask_y)
    out = (x - mean_x) / std_x * std_y + mean_y
    return out


def get_lips(image, landmarks, select):
    # landmark_arr = landmarks_to_arr(landmarks, image)
    contour = landmarks[:, 0]
    points = landmarks[select][:, 0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.fillPoly(mask, [np.roll(points, 1, -1).astype(np.int32)], True,  255)
    mask = torch.from_numpy(mask)
    mask = mask.view(1, 1, *mask.shape).float()
    return mask, points


def landmarks_to_arr(landmark: List[Dict[str, float]], image: ARRAY) -> ARRAY:
    h_pix, w_pix, _ = image.shape
    landmark_arr = map(lambda x: [x.y, x.x], landmark)
    landmark_arr = V(list(landmark_arr))
    landmark_arr[:, 0] *= h_pix
    landmark_arr[:, 1] *= w_pix
    return landmark_arr


class TargetMouth:
    cache = None

    def get_geometric_loss(self, predict: T):
        loss = nnf.l1_loss(predict * self.lips_mask64_bin,
                           self.depth, reduction='none')
        loss = loss.sum() / (self.lips_mask64_bin.sum() * 3)
        return loss

    def get_pixel_loss_reversed(self, rgb_images, rgb_target):
        mask = 1 - self.lips_mask_raw
        loss = nnf.l1_loss(rgb_images, rgb_target, reduction='none') * mask
        loss = loss.sum() / (mask.sum() * 3)
        return loss

    def get_pixel_loss(self, predict):
        loss = nnf.l1_loss(predict * self.lips_mask_raw,
                           self.rgb_images, reduction='none')
        loss = loss.sum() / (self.lips_mask_raw.sum() * 3)
        return loss

    @staticmethod
    def get_offsets(mask: T):
        inds = torch.where(mask[0, 0].bool())
        top, bottom = inds[0].min(), inds[0].max()
        left, right = inds[1].min(), inds[1].max()
        return left.item(), right.item(), top.item(), bottom.item()

    @staticmethod
    def crop(image: T, mask, offsets, offset_other):
        left, right, top, bottom = offsets
        h, w = TargetMouth.to_hw(offsets)
        h_other, w_other = TargetMouth.to_hw(offset_other)
        scale = w_other / w
        h_new = scale * h
        mask = mask.float()
        if h_new < h_other:
            diff = (h_other - h_new) / (2 * scale)
            diff = int(diff)
            top -= diff
            bottom += diff
            mask = nnf.max_pool2d(mask, (diff * 2 + 1, diff * 2 + 1), (1, 1), padding=diff)
        crop = image[:, :, top: bottom, left: right]
        mask = mask[:, :, top: bottom, left: right]
        image_alpha = torch.cat((crop, mask), dim=1)
        return image_alpha, (left, right, top, bottom)

    @staticmethod
    def paste(image, lips, offset):
        _, _, h, w = lips.shape
        left, right, top, bottom = offset
        lips, mask = lips[:, :3], lips[:, 3:]
        center_x, center_y = int((left + right) / 2.), int((top + bottom) / 2.)
        replace = image[:, :, center_y - h // 2: center_y - h // 2 + h, center_x - w // 2: center_x - w // 2 + w]
        replace = lips * mask + (1 - mask) * replace
        image[:, :, center_y - h // 2: center_y - h // 2 + h, center_x - w // 2: center_x - w // 2 + w] = replace
        return image

    def init_image(self, path: str, is_np: bool = False):
        if type(path) is str:
            if is_np:
                image = files_utils.load_np(path)
            else:
                image = files_utils.load_image(path)
        else:
            image = path
        lips_mask, face_mask, face_contour = self.get_lips_mask(image)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.
        offset = self.get_offsets(lips_mask)
        return lips_mask, face_mask, image, offset, face_contour

    def get_lips_image(self, res=128):
        mask = self.lips_mask.float()
        image_bg = self.base_image * .3 + .7
        image = image_bg * (1-mask) + mask * self.base_image
        if res < image.shape[3]:
            image = nnf.interpolate(image, res, mode='bilinear')
        image = (image[0] * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        return image

    @staticmethod
    def to_hw(offset):
        left, right, top, bottom = offset
        w = right - left
        h = bottom - top
        return h, w

    def resize(self, crop, source_offset, target_offset):
        h, w = self.to_hw(source_offset)
        h_, w_ = self.to_hw(target_offset)
        crop = nnf.interpolate(crop, scale_factor=float(w_) / w)
        return crop

    def transfer_lips(self, other, is_np=False, res=-1):
        lips_mask, face_mask, image, offset, face_contour = self.init_image(other, is_np)
        base_image_ada = self.base_image
        # base_image_ada = masked_ada_in(self.base_image, image, self.face_mask, face_mask)

        transformed = find_transform(self.face_contour, face_contour, image, base_image_ada, self.lips_mask)
        offset_transformed = self.get_offsets(transformed[1])
        # return
        lips_crop, offset_ = self.crop(*transformed, offset_transformed, offset)
        # lips_crop = self.resize(lips_crop, offset_, offset)
        image_new = self.paste(image, lips_crop, offset)
        if 0 < res < image_new.shape[-1]:
            image_new = nnf.interpolate(image_new, res, mode='bilinear')
        # image_new = image
        image_new = (image_new[0].permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        # files_utils.imshow(image_new)
        return image_new
        # files_utils.imshow(lips_mask)

    def get_lips_mask(self, image):
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).multi_face_landmarks
            if results is None:
                results = TargetMouth.cache
                print('error')
                self.error = True
            else:
                TargetMouth.cache = results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).multi_face_landmarks[0].landmark
        landmark_arr = landmarks_to_arr(results, image)
        lips, _ = get_lips(image, landmark_arr, FACEMESH_LIPS)
        face, face_contour = get_lips(image, landmark_arr, FACEMESH_FACE_OVAL)
        _, _, h, w = lips.shape
        k = int(max(5, (min(h, w) * .031)))
        lips = nnf.max_pool2d(lips, (k * 2 + 1, k * 2 + 1), (1, 1), padding=k)
        # input, kernel_size, stride = None, padding = 0, dilation = 1,
        # ceil_mode = False, return_indices = False
        return lips, face, face_contour

    def __init__(self, item: Union[str, int, ARRAY], is_np: bool = False):
        self.device = CUDA(0)
        self.error = False
        if type(item) is int:
            image_path = f"{constants.DATA_ROOT}/exp_mid/{item:02d}"
        else:
            image_path = item
        self.lips_mask, self.face_mask, self.base_image, self.offset, self.face_contour = self.init_image(image_path, is_np)


def transfer_viseme(is_step_a:bool=True):
    folder_source = f'{constants.DATA_ROOT}/processed/'
    folder_database = f'{constants.DATA_ROOT}/processed/'
    viseme_database = f"{constants.DATA_ROOT}/raw_videos/obama_062814_viseme_vec"
    viseme_target = f"{constants.DATA_ROOT}/raw_videos/aladdin_trim_viseme_vec"
    out_dir = f"{constants.DATA_ROOT}/genie2obama"
    target_idx = files_utils.load_pickle(viseme_target)['is_center']
    viseme_select = process_video.align_visemes(viseme_database, viseme_target)

    mouth_all = files_utils.collect(folder_database, '.npy', prefix='crop')
    mouth_seq = [mouth_all[item] for item in viseme_select]
    video_seq = files_utils.collect(folder_source, '.npy', prefix='crop')[:len(mouth_seq)]
    logger = train_utils.Logger().start(len(video_seq))
    for i, (path_a, path_b) in enumerate(zip(video_seq, mouth_seq)):
        if target_idx[i] or not is_step_a:
            mouth = TargetMouth(''.join(path_b), True)
            image_ = mouth.transfer_lips(''.join(path_a), True)
            files_utils.save_np(image_, f'{out_dir}/crop_{i:03d}')
        logger.reset_iter()
    logger.stop()



def viseme2id():

    @models_utils.torch_no_grad
    def get_base_image():
        if j <= len(base_images):
            image = files_utils.load_image(''.join(image_paths[j]))
            _, _, w_plus = e4e_net(image)
            w = w_plus[:, :1, :].expand(1, 18, 512)
            out = e4e_net.decode(w.cuda())[0]
            out = (out.cpu().detach() + 1) * 127.5
            out = out.clip(0, 255).permute(1, 2, 0).numpy().astype(np.uint8)
            base_images.append(out)
        return base_images[j]




    base_images = []
    e4e_net = e4e.E4E()
    image_paths = files_utils.collect(f'{constants.CACHE_ROOT}/stylesdf_images/', '.png')
    viseme_root = f'{constants.CACHE_ROOT}/viseme/'
    logger = train_utils.Logger()
    valid = torch.ones(len(image_paths), dtype=torch.bool)
    with torch.no_grad():
        for i in range(17):
            all_w = torch.zeros(len(image_paths), 18, 512)
            mouth = TargetMouth(f'{viseme_root}/{i:02d}.png', False)
            logger.start(len(image_paths))
            for j in range(len(image_paths)):
                image_ = get_base_image()
                image_ = mouth.transfer_lips(image_, False)
                image_256, image_e4e, w_plus = e4e_net(image_)
                if mouth.error:
                    mouth.error = False
                    valid[j] = False
                all_w[j] = w_plus
                logger.reset_iter()
            logger.stop()
            files_utils.save_pickle(all_w, f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}_v3')
            files_utils.save_pickle(valid, f'{constants.CACHE_ROOT}/viseme/viseme_valid_v3')
            # return
            # if DEBUG:
            #     files_utils.imshow(torch.cat((image_256.cpu(), image_e4e.cpu()), dim=2))


def transfer_video():
    all_vid = files_utils.collect(constants.LRS2_ROOT, '.mp4')
    vid = vid_path = fps = num_frames = None
    for vid_path in all_vid:
        vid = imageio.get_reader(''.join(vid_path), 'ffmpeg')
        fps = vid._meta['fps']
        num_frames = vid.count_frames()
        if float(num_frames) / fps > 5:
            break
    vid_name = f'{vid_path[0].split("/")[-2]}_{vid_path[1]}'
    out_root = f'/home/ahertz/projects/StyleFusion-main/assets/lrs2_obama/{vid_name}/'
    process_video.split_audio(''.join(vid_path), f'{out_root}{vid_name}')
    folder_image = f'/home/ahertz/projects/StyleFusion-main/assets/processed/'
    paths_images = files_utils.collect(folder_image, '.npy')
    paths_images = [path for path in paths_images if 'crop' in path[1] or 'image' in path[1]]
    counter = 0

    print(vid_name)
    # group_a = paths[:120]
    # group_b = paths[120:]
    images = []
    for i in range(num_frames):
        path_a = paths_images[i]
        mouth_image = vid.get_data(i)
        counter += 1
        # if counter % 20 != 0:
        #     continue
        mouth = TargetMouth(mouth_image)
        image_ = mouth.transfer_lips(''.join(path_a), True)
        # if mouth.error:
        #     files_utils.imshow(image_)
        # images.append(image_)
        images.append(image_)
        files_utils.save_np(image_, f'{out_root}/crop_{counter:03d}')



        # files_utils.save_image(image_, f'/home/ahertz/projects/StyleFusion-main/assets/slides_0404/lips_origin_{counter:02d}')

    image_utils.gif_group(images, f'{constants.CHECKPOINTS_ROOT}pti/{vid_name}/stich', fps)


def video2image(name_mouth):
    image_paths = files_utils.collect(f'{constants.CACHE_ROOT}/stylesdf_images/', '.png')
    folder_mouth = f'/home/ahertz/projects/StyleFusion-main/assets/{name_mouth}/'
    paths_mouth = files_utils.collect(folder_mouth, '.npy')
    paths_mouth = [path for path in paths_mouth if 'crop' in path[1] or 'image' in path[1]]
    logger = train_utils.Logger()
    for image_path in image_paths:
        out_name = f'{constants.MNT_ROOT}/{image_path[1]}_{name_mouth}'
        if files_utils.is_dir(out_name):
            continue
        logger.start(len(paths_mouth))
        for i, mouth_path in enumerate(paths_mouth):
            mouth = TargetMouth(''.join(mouth_path), True)
            image_ = mouth.transfer_lips(''.join(image_path), False)
            files_utils.save_np(image_,
                                f'/{out_name}/stich_{i:03d}')
            logger.reset_iter()
        logger.stop()
        return


def main():
    folder_mouth = f'/home/ahertz/projects/StyleFusion-main/assets/101_beigefox_front_comp_v017/'
    folder_image = f'/home/ahertz/projects/StyleFusion-main/assets/processed/'
    paths_images = files_utils.collect(folder_image, '.npy')
    paths_images = [path for path in paths_images if 'crop' in path[1] or 'image' in path[1]]
    paths_mouth = files_utils.collect(folder_mouth, '.npy')
    paths_mouth = [path for path in paths_mouth if 'crop' in path[1] or 'image' in path[1]]
    counter = 0
    # group_a = paths[:120]
    # group_b = paths[120:]
    images = []
    for path_a, path_b in zip(paths_images, paths_mouth):

        counter += 1
        # if counter % 20 != 0:
        #     continue
        mouth = TargetMouth(''.join(path_b), True)
        image_ = mouth.transfer_lips(''.join(path_a), True)
        # if mouth.error:
        #     files_utils.imshow(image_)
        # images.append(image_)
        images.append(image_)
        files_utils.save_np(image_, f'/home/ahertz/projects/StyleFusion-main/assets/101_beigefox_front_comp_v017/crop_{counter:03d}')
        # files_utils.imshow(image_)
        # image = mouth.get_lips_image()

        # files_utils.save_image(image_, f'/home/ahertz/projects/StyleFusion-main/assets/slides_0404/lips_origin_{counter:02d}')

    image_utils.gif_group(images, f'{constants.CHECKPOINTS_ROOT}pti/101_beigefox_front_comp_v017/stich', 24)
        # if counter == 51:
        #     break
    # w_plus = files_utils.load_pickle(f'{folder}/e4e_w_plus')[:, 0]
    # image_path = f'{constants.CHECKPOINTS_ROOT}pti/image_00002404'
    # for i in range(7):
    #     mouth = TargetMouth(i)
    #     image = mouth.transfer_lips(image_path)
    #     files_utils.save_image(image, f"./assets/cache/obama_images_stich/image_00002404_{i:d}")

    # all_images = files_utils.collect(f"{constants.CACHE_ROOT}/stylesdf_images/", ".png")
    # for j, path in enumerate(all_images):
    #     if j < 5:
    #         continue
    #     for i in range(7):
    #         mouth = TargetMouth(i)
    #         image = mouth.transfer_lips(''.join(path))
    #         files_utils.save_image(image, f"./assets/cache/stylesdf_images_stich/{j:03d}_{i:d}")
    #     if j == 10:
    #         break


if __name__ == '__main__':
    viseme2id()