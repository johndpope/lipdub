import cv2
import torchvision
import options
from custom_types import *
import constants
from utils import files_utils, train_utils
from align_faces import FaceAlign
import imageio
from torchvision.transforms import functional as tsf
from PIL import Image
import hashlib


def get_frames_numbers(file):
    json_files = files_utils.collect(f"{file[0]}/extracted_sequences/", '.json')
    inds = []
    for json_file in json_files:
        data = files_utils.load_json(''.join(json_file))
        inds.extend(data)
    inds = torch.tensor(inds, dtype=torch.int64)
    inds = inds[torch.rand(len(inds)).argsort()]
    return inds


class ProcessFaceForensicsRoot:

    def process_frame(self, frame) -> bool:
        try:
            crop, quad, lm, lm_crop = self.aligner.crop_face(frame, 512)
            self.update_metadata(crop, quad, lm, lm_crop)
            # if DEBUG:
            #     files_utils.imshow(frame)
            #     files_utils.imshow(crop)
        except:
            return False
        return True

    @property
    def crop_path(self):
        return f"{self.out_root}{self.total_counter:07d}.png"

    def start_item(self, video_path, frame_number):
        self.cur_item = {"id": self.total_counter, "crop_path": self.crop_path,
                         "video_path": video_path, "frame_number": frame_number}

    def update_metadata(self, crop, quad, lm, lm_crop):
        files_utils.save_image(crop, self.crop_path)
        self.metadata["info"].append(self.cur_item)
        self.metadata["quads"].append(quad)
        self.metadata["landmarks"].append(lm)
        self.metadata["landmarks_crops"].append(lm_crop)
        self.total_counter += 1
        if (self.total_counter + 1) % 100 == 0:
            self.save_metadata()
        self.logger.reset_iter()

    def process_sequence(self, video_path):
        vid = imageio.get_reader("".join(video_path), 'ffmpeg')
        if self.single_mode:
            frames = torch.arange(vid.count_frames())
        else:
            frames = get_frames_numbers(video_path)
        counter = 0
        for frame_number in frames:
            frame_number = frame_number.item()
            self.start_item(video_path, frame_number)
            frame = vid.get_data(frame_number)
            counter += int(self.process_frame(frame))
            if counter == self.max_frames_per_file and not self.single_mode:
                break

    def save_metadata(self):
        metadata = {"info": self.metadata["info"],
                    "quads": np.stack(self.metadata["quads"]),
                    "landmarks": np.stack(self.metadata["landmarks"]),
                    "landmarks_crops": np.stack(self.metadata["landmarks_crops"])}
        files_utils.save_pickle(metadata, f"{self.out_root}/metadata")

    def run(self, video_root, is_single=False):
        self.single_mode = is_single
        if is_single:
            files = [files_utils.split_path(video_root)]
            self.logger.start(imageio.get_reader("".join(files[0]), 'ffmpeg').count_frames())
        else:
            files = files_utils.collect(constants.FaceForensicsRoot + 'downloaded_videos/', '.mp4')
            self.logger.start(len(files) * self.max_frames_per_file)
        for video_path in files:
            self.process_sequence(video_path)
        self.logger.stop()

    def load(self):
        self.metadata = files_utils.load_pickle(f"{self.out_root}/metadata")
        return self

    def __len__(self):
        return len(self.metadata["info"])

    def __getitem__(self, item: int):
        info, quad, lm, lm_crop = [self.metadata[name][item] for name in ("info", "quads:", "landmarks",
                                                                         "landmarks_crops")]
        crop = files_utils.load_image(f"{self.out_root}{info['id']:07d}.png")
        return crop, lm_crop
    cache = set()

    def get_landmarks(self, image):
        m = hashlib.md5()
        m.update(image)
        name = m.digest()
        path = f"{constants.MNT_ROOT}/cache/{name}.npy"
        assert name not in self.cache
        if files_utils.is_file(path):
            return files_utils.load_np(path)
        else:
            self.cache.add(name)
            result = self.aligner.get_landmark(image)
            files_utils.save_np(result, path)
            return result

    def __init__(self, out_root: str):
        self.single_mode = False
        self.max_frames_per_file = 40
        self.aligner = FaceAlign()
        self.metadata = {"info": [], "quads": [], "landmarks": [], "landmarks_crops": []}
        self.total_counter = 0
        self.cur_item = {}
        self.out_root = out_root
        # self.out_root = constants.FaceForensicsRoot + 'processed_frames/'
        self.logger = train_utils.Logger()



class LipsTransform:

    def zero_center_landmarks(self, landmarks: ARRAY):
        max_val, min_val = landmarks.max(axis=0), landmarks.min(axis=0)
        center = (max_val + min_val) / 2
        landmarks = landmarks - center[None, :]
        return landmarks


    def landmarks_only_(self, landmarks, image, coeff):
        h, w, c = image.shape
        if h != self.res:
            landmarks = landmarks * (self.res / h)
        if coeff is not None:
            x, y = landmarks[:, 0], landmarks[:, 1]
            x_new = (coeff[0] * x + coeff[1] * y + coeff[2]) / (coeff[6] * x + coeff[7] * y + 1)
            y_new = (coeff[3] * x + coeff[4] * y + coeff[5]) / (coeff[6] * x + coeff[7] * y + 1)
            landmarks = np.stack((x_new, y_new), axis=1)
        landmarks = (landmarks / self.res) * 2 - 1
        return landmarks
    
    def landmarks_only(self, landmarks, image, train=True):
        width = height = self.res
        if train:
            startpoints, endpoints = self.perspective_aug.get_params(width, height, self.perspective_aug.distortion_scale)
            coeff = tsf._get_perspective_coeffs(endpoints, startpoints)
        else:
            coeff = None
        return self.landmarks_only_(landmarks, image, coeff)

    def image_only(self, image):
        image = self.resize(image)
        image = self.to_tensor(image) * 2 - 1
        return image

    def __call__(self, images, landmarks, train=True):
        squeeze = type(images) is ARRAY
        if squeeze:
            images = [images]
        base_image = images[0]
        is_mask = [image.ndim == 2 for image in images]
        images = [Image.fromarray(image) for image in images]
        images = [self.resize(image) for image in images]
        if torch.rand((1,)).item() < .5 and train:
            h, w, c = base_image.shape
            landmarks[:, 0] = w - landmarks[:, 0]
            images = [tsf.hflip(image) for image in images]
        if self.color_jitter and torch.rand((1,)).item() < .5 and train:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.color_jitter.get_params(self.color_jitter.brightness, self.color_jitter.contrast, self.color_jitter.saturation, self.color_jitter.hue)
            images_ = []
            for i, image in enumerate(images):
                if not is_mask[i]:
                    for fn_id in fn_idx:
                        if fn_id == 0 and brightness_factor is not None:
                            image = tsf.adjust_brightness(image, brightness_factor)
                        elif fn_id == 1 and contrast_factor is not None:
                            image = tsf.adjust_contrast(image, contrast_factor)
                        elif fn_id == 2 and saturation_factor is not None:
                            image = tsf.adjust_saturation(image, saturation_factor)
                        elif fn_id == 3 and hue_factor is not None:
                            image = tsf.adjust_hue(image, hue_factor)
                images_.append(image)
            images = images_
        if train and self.blur:
            images = [image if is_mask[i] else self.blur(image) for i, image in enumerate(images)]
        if self.perspective_aug and torch.rand(1) < self.perspective_aug.p and train:
            fills = [[255] if is_mask[i] else [255, 255, 255] for i in range(len(images))]
            width, height = tsf.get_image_size(images[0])
            startpoints, endpoints = self.perspective_aug.get_params(width, height, self.perspective_aug.distortion_scale)
            coeff = tsf._get_perspective_coeffs(endpoints, startpoints)
            images = [tsf.perspective(image, startpoints, endpoints, self.perspective_aug.interpolation, fills[i]) for
                      i, image in enumerate(images)]
        else:
            coeff = None
        landmarks = self.landmarks_only_(landmarks, base_image, coeff).astype(np.float32)
        images = [self.to_tensor(image) if is_mask[i] else self.to_tensor(image) * 2 - 1 for i, image in enumerate(images)]
        if squeeze:
            return images[0], landmarks
        return images, landmarks

    def __init__(self, res, color_jitter=True, blur=True, distortion_scale=.4):
        self.res = res
        self.resize = torchvision.transforms.Resize(res)
        self.to_tensor = torchvision.transforms.ToTensor()
        if color_jitter:
            self.color_jitter = torchvision.transforms.ColorJitter(hue=0.2, brightness=.8, saturation=.8, contrast=.8)
        else:
            self.color_jitter = None
        kernel_size = int(.05 * res)
        if blur:
            if kernel_size % 2 == 0:
                kernel_size = kernel_size + 1
            blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(.1, 2.))
            self.blur = torchvision.transforms.RandomApply([blur], .3)
        else:
            self.blur = None
        if distortion_scale > 0:
            self.perspective_aug = torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=.5, fill=(255, 255, 255),
                                                                       interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            self.perspective_aug = None


class LipsConditionedDS(Dataset, ProcessFaceForensicsRoot):

    def prep_infer(self, path_base, path_driving):
        crop_base = files_utils.load_np("".join(path_base))
        lm_base = self.get_landmarks(crop_base)
        driving_base = files_utils.load_np("".join(path_driving))
        lm_driving = self.get_landmarks(driving_base)
        image_in, _, image_full, mask, _ = self.get_item_(crop_base, lm_base)
        _, lm_driving, driving_image, _, _ = self.get_item_(driving_base, lm_driving)
        lm_driving = torch.from_numpy(lm_driving)
        return image_in, lm_driving, image_full, driving_image, mask

    def get_item_(self, crop_base, lm_base):
        mask = get_mask(crop_base, lm_base)
        (crop, mask), _ = self.transform([crop_base, mask], lm_base, train=self.is_train)
        lm_lips = self.transform.landmarks_only(lm_base, crop_base, self.is_train)[48:, :]
        lm_lips = self.transform.zero_center_landmarks(lm_lips)
        masked = crop * (1 - mask) + mask * torch.ones_like(crop)
        # image_ = vis_landmark_on_img(crop.copy(), (lm_crop + 1) * 256)
        # files_utils.imshow(image_)
        return masked, lm_lips.astype(np.float32), crop, mask, mask.sum()



    def __getitem__(self, item: int):
        info,  lm_base = [self.metadata[name][item] for name in ("info", "landmarks_crops")]
        crop_base = files_utils.load_image(f"{self.out_root}{info['id']:07d}.png")
        return self.get_item_(crop_base, lm_base)

    def __init__(self, opt):
        super(LipsConditionedDS, self).__init__(opt.data_dir)
        self.load()
        self.is_train = True
        self.transform = LipsTransform(256, distortion_scale=.2, blur=False, color_jitter=False)


class LipsLandmarksDS(Dataset, ProcessFaceForensicsRoot):

    def __getitem__(self, item: int):
        info,  lm_crop = [self.metadata[name][item] for name in ("info", "landmarks_crops")]
        crop = files_utils.load_image(f"{self.out_root}{info['id']:07d}.png")
        crop, lm_crop = self.transform(crop, lm_crop)
        lm_lips = lm_crop[48:, :]


        if DEBUG:
            crop = files_utils.image_to_display(crop)
            image_ = draw_lips(crop.copy(), (lm_crop + 1) * self.transform.res / 2.)
            files_utils.imshow(image_)
        return crop, lm_lips

    def __init__(self, opt):
        super(LipsLandmarksDS, self).__init__(opt.data_dir)
        self.load()
        self.transform = LipsTransform(240)


def get_drawer(img, shape, line_width):


    def draw_curve_(idx_list, color=(0, 255, 0), loop=False):
        for i in idx_list:
            cv2.line(img, (shape[i, 0], shape[i, 1]), (shape[i + 1, 0], shape[i + 1, 1]), color, line_width)
        if (loop):
            cv2.line(img, (shape[idx_list[0], 0], shape[idx_list[0], 1]),
                     (shape[idx_list[-1] + 1, 0], shape[idx_list[-1] + 1, 1]), color, line_width)

    return draw_curve_


def draw_lips(img, lips):
    img = img.copy()
    if lips.shape[0] > 30:
        lips = lips[48:, :]
    lips = lips.astype(np.int32)
    draw_curve = get_drawer(img, lips, 1)
    draw_curve(list(range(0, 11)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(12, 19)), loop=True, color=(238, 130, 238))
    # for i in (12, 16):
    #     print(i)
    #     cv2.circle(img, lips[i], 2, (255, 255, 255), thickness=-1)
    return img


def vis_landmark_on_img(img, shape, line_width=2):
    '''
    Visualize landmark on images.
    '''


    shape = shape.astype(np.int32)
    draw_curve = get_drawer(img, shape, line_width)
    draw_curve(list(range(0, 16)), color=(255, 144, 25))  # jaw
    draw_curve(list(range(17, 21)), color=(50, 205, 50))  # eye brow
    draw_curve(list(range(22, 26)), color=(50, 205, 50))
    draw_curve(list(range(27, 35)), color=(208, 224, 63))  # nose
    draw_curve(list(range(36, 41)), loop=True, color=(71, 99, 255))  # eyes
    draw_curve(list(range(42, 47)), loop=True, color=(71, 99, 255))
    draw_curve(list(range(48, 59)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(60, 67)), loop=True, color=(238, 130, 238))

    return img


def get_mask(img, shape):
    points = shape[2:15]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def infer():
    dataset = LipsLandmarksDS(options.OptionsLipsDetection())

    for i in range(20):
        x = dataset[i * 40]
        # files_utils.imshow(image)
        # im_lm = vis_landmark_on_img(255 * np.ones_like(image), lm)
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # mask = get_mask(mask, lm).astype(np.bool)
        # im_masked = image.copy()
        # im_masked[mask] = 255
        # # files_utils.imshow(im_masked)
        #
        # files_utils.save_image(im_masked, f"{constants.CACHE_ROOT}/for_slides/ff_example/ff_{i:02d}_maksed")
        # files_utils.save_image(im_lm, f"{constants.CACHE_ROOT}/for_slides/ff_example/ff_{i:02d}_lm")



def main():
    video = "../assets/raw_videos/obama_062814.mp4"
    ProcessFaceForensicsRoot(constants.MNT_ROOT + 'video_frames/obama/').run(video, True)


if __name__ == '__main__':
    infer()
