
import cv2
import numpy as np

import constants
from utils import files_utils, image_utils
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def read_exr(filename):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    swap = img[:, :, 0].copy()
    img[:, :, 0] = img[:, :, 2]
    img[:, :, 2] = swap
    img = img * 65535
    img[img > 65535] = 65535
    img = img / 257.
    img = np.uint8(img)
    # img = img / img.max() * 255.
    # img = img.astype(np.uint8)
    return img

def write_exr(file_path, image):
    cv2.imwrite(file_path, np.float32(image), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])



def pad_all_images(all_images):

    def pad_image(image_):
        h, w, _ = image_.shape
        if h < max_h or w < max_w:
            out_ = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            start_h, start_w = (max_h - h) // 2, (max_w - w) // 2
            out_[start_h: start_h + h, start_w: start_w + w] = image_
        else:
            out_ = image_
        return out_
    max_w = max_h = 0
    out = []
    for image in all_images:
        h, w, _ = image.shape
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    for image in all_images:
        out.append(pad_image(image))
    return out


def save_mid():
    path = files_utils.collect(
        r'C:\Users\hertz\Downloads\mnt\r\projects\facesimile\assets\char\GEN_WOMAN\lighting\images\lte_main_slapcomp\v001\exr/',
        '.exr')
    all_images = []
    frame = 0
    split = [36, 37, 37, 37, 37, 37, 37]
    for i in range(7):
        cur_img = frame + split[i] // 2 + 1 - int(split[i] % 2  == 0)
        img = read_exr(''.join(path[cur_img]))
        files_utils.imshow(img)
        frame += split[i]
        files_utils.save_image(img, f'./assets/exp_mid/{i:02d}')



def main():
    root = '/mnt/r/projects/facesimile/shots/101/beigefox_front/lighting/images/lte_main_char_genman_090deg/v001/exr'
    # path = files_utils.collect(r'C:\Users\hertz\Downloads\mnt\r\projects\facesimile\assets\char\GEN_WOMAN\lighting\images\lte_main_slapcomp\v001\exr/', '.exr')
    path = files_utils.collect(root, '.exr')
    all_images = []
    for i in range(len(path)):
        # img = read_exr(''.join(path[(517 // 14) * i + 18 + i]))
        img = read_exr(''.join(path[i]))
        all_images.append(img)
        # files_utils.imshow(img)
        # if (i + 1) % 10 == 0:
        #     files_utils.save_image(all_images[-1], f'{constants.CHECKPOINTS_ROOT}pti/seq/source_{i:03d}')
        # files_utils.save_image(img, f'{constants.DATA_ROOT}/lte_main_char_genman_090deg/crop_{i:05d}')
        # all_images.append(img)
        # files_utils.save_image(img, f'./assets/exp/exp_{i:02d}')
    # all_images = pad_all_images(all_images)
    # start = 0
    # for j, sp in enumerate(split):
    #     seq = all_images[start: start+ sp]
    #     mid_seq = sp // 2 + 1
    #     seq = seq[:mid_seq]
    #     seq_rev = seq.copy()
    #     seq_rev.reverse()
    #     for i in range(len(seq_rev)):
    #         seq_rev[i] = seq_rev[i].copy()[:, ::-1]
    #     seq = seq + seq_rev
    #     seq = seq[10:-10]
    #     # else:
    #     #     seq = seq[11:-10]
    #     start += sp
    #     image_utils.gif_group(seq, f'./assets/exp/exp_{j:02d}', 20)
    # files_utils.save_np(np.stack(all_images, axis=0), f'./assets/exp/exp_all')
    # image_utils.gif_group(all_images, f'./assets/exp/lte_main_char_genman_090deg', 15)
    # files_utils.save_image(np.concatenate(all_images, axis=1), f'./assets/exp/exp_all')
    return 0


if __name__ == '__main__':
    main()