from __future__ import annotations
import cv2
import torchvision
import options
from custom_types import *
import constants
from utils import files_utils, train_utils  # , frontalize_landmarks
from align_faces import FaceAlign
import imageio
from torchvision.transforms import functional as tsf
from PIL import Image


def get_frames_numbers(file, shuffle=True, get_single=False):
    json_files = files_utils.collect(f"{file[0]}/extracted_sequences/", '.json')
    inds = []
    for json_file in json_files:
        data = files_utils.load_json(''.join(json_file))
        inds.extend(data)
        if get_single:
            break
    inds = torch.tensor(inds, dtype=torch.int64)
    if shuffle:
        inds = inds[torch.rand(len(inds)).argsort()]
    return inds


class ProcessFaceForensicsRoot:

    def process_frame(self, frame) -> bool:
        try:
            crop, quad, lm, lm_crop = self.aligner.crop_face(frame, self.res)
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
                         "video_path": self.seq_number, "frame_number": frame_number}

    def update_metadata(self, crop, quad, lm, lm_crop):
        files_utils.save_image(crop, self.crop_path)
        self.metadata["info"].append(self.cur_item)
        self.metadata["quads"].append(quad)
        self.metadata["landmarks"].append(lm)
        self.metadata["landmarks_crops"].append(lm_crop)
        self.total_counter += 1
        if (self.total_counter + 1) % 10000 == 0:
            self.save_metadata()
        self.logger.reset_iter()

    def process_sequence(self, video_path, frames=None):
        vid = imageio.get_reader("".join(video_path), 'ffmpeg')
        if frames is None:
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
            if self.max_frames_per_file > 0 and counter == self.max_frames_per_file and not self.single_mode:
                break

    def save_metadata(self):
        metadata = {"info": self.metadata["info"],
                    "quads": np.stack(self.metadata["quads"]),
                    "landmarks": np.stack(self.metadata["landmarks"]),
                    "landmarks_crops": np.stack(self.metadata["landmarks_crops"]),
                    "sequence_length": V(self.seq_arr).astype(np.int32)}
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
        self.save_metadata()
        self.logger.stop()

    def run_all(self):
        files = files_utils.collect(constants.FaceForensicsRoot + 'downloaded_videos/', '.mp4')
        frames = [get_frames_numbers(video_path, False, True) for video_path in files]
        self.logger.start(sum([len(item) for item in frames]))
        for video_path, frames_ in zip(files, frames):
            self.process_sequence(video_path, frames_)
            self.seq_arr.append(self.total_counter)
            self.seq_number += 1
        self.logger.stop()
        self.save_metadata()

    def load(self):
        self.metadata = files_utils.load_pickle(f"{self.out_root}/metadata")
        video2id = {}
        item2id = {}
        id2items = {}
        counter = 0
        for i, item in enumerate(self.metadata["info"]):
            video_path = item["video_path"]
            if type(video_path) is not int:
                video_path = ''.join(item["video_path"])
            if video_path not in video2id:
                video2id[video_path] = counter
                id2items[counter] = []
                counter += 1
            item2id[i] = video2id[video_path]
            id2items[video2id[video_path]].append(i)
        self.item2id = item2id
        self.id2items = id2items
        return self

    def __len__(self):
        return len(self.metadata["info"])

    def __getitem__(self, item: int):
        info, quad, lm, lm_crop = [self.metadata[name][item] for name in ("info", "quads:", "landmarks",
                                                                          "landmarks_crops")]
        crop = files_utils.load_image(f"{self.out_root}{info['id']:07d}.png")
        return crop, lm_crop

    cache = set()

    def get_landmarks(self, path: List[str], image: Optional[ARRAY] = None) -> ARRAY:
        name = path[0].split('/')[-2:] + [path[1]]
        name = 'landmarks_'.join(name)
        path_cache = f"{constants.MNT_ROOT}/cache/{name}.npy"
        loaded = False
        if files_utils.is_file(path_cache):
            try:
                landmarks = files_utils.load_np(path_cache)
                loaded = True
            except:
                pass
        if not loaded:
            if image is None:
                if path[-1] == '.npy':
                    image = files_utils.load_np(''.join(path))
                else:
                    image = files_utils.load_image(''.join(path))
            landmarks = self.aligner.get_landmark(image)
            files_utils.save_np(landmarks, path_cache)
        return landmarks

    @property
    def num_ids(self):
        return len(set([item for key, item in self.item2id.items()]))

    def __init__(self, out_root: str, max_frames_per_file):
        self.res = 256
        self.single_mode = False
        self.max_frames_per_file = max_frames_per_file
        self.aligner = FaceAlign()
        self.metadata = {"info": [], "quads": [], "landmarks": [], "landmarks_crops": []}
        self.total_counter = 0
        self.seq_number = 0
        self.seq_arr = []
        self.cur_item = {}
        self.item2id = {}
        self.id2items = {}
        self.out_root = out_root
        # self.out_root = constants.FaceForensicsRoot + 'processed_frames/'
        self.logger = train_utils.Logger()


class LipsTransform:

    def zero_center_landmarks(self, landmarks: ARRAY):
        if landmarks.ndim == 2:
            max_val, min_val = landmarks.max(axis=0), landmarks.min(axis=0)
            center = (max_val + min_val) / 2
            landmarks = landmarks - center[None, :]
        else:
            max_val, min_val = landmarks.max(axis=1), landmarks.min(axis=1)
            center = (max_val + min_val) / 2
            landmarks = landmarks - center[:, None, :]
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
            startpoints, endpoints = self.perspective_aug.get_params(width, height,
                                                                     self.perspective_aug.distortion_scale)
            coeff = tsf._get_perspective_coeffs(endpoints, startpoints)
        else:
            coeff = None
        return self.landmarks_only_(landmarks, image, coeff)

    def image_only(self, image, train=True):
        image = self.resize(image)
        image = self.to_tensor(image) * 2 - 1
        return image

    def __call__(self, images, landmarks=None, train=True):
        squeeze = type(images) is ARRAY
        if squeeze:
            images = [images]
        base_image = images[0]
        is_mask = [image.ndim == 2 for image in images]
        images = [Image.fromarray(image) for image in images]
        images = [self.resize(image) for image in images]
        if torch.rand((1,)).item() < .5 and train:
            h, w, c = base_image.shape
            if landmarks is not None:
                landmarks[:, 0] = w - landmarks[:, 0]
            images = [tsf.hflip(image) for image in images]
        if self.color_jitter and torch.rand((1,)).item() < .5 and train:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.color_jitter.get_params(
                self.color_jitter.brightness, self.color_jitter.contrast, self.color_jitter.saturation,
                self.color_jitter.hue)
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
            startpoints, endpoints = self.perspective_aug.get_params(width, height,
                                                                     self.perspective_aug.distortion_scale)
            coeff = tsf._get_perspective_coeffs(endpoints, startpoints)
            images = [tsf.perspective(image, startpoints, endpoints, self.perspective_aug.interpolation, fills[i]) for
                      i, image in enumerate(images)]
        else:
            coeff = None
        if landmarks is not None:
            landmarks = self.landmarks_only_(landmarks, base_image, coeff).astype(np.float32)
        images = [self.to_tensor(image) if is_mask[i] else self.to_tensor(image) * 2 - 1 for i, image in
                  enumerate(images)]
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
            self.perspective_aug = torchvision.transforms.RandomPerspective(distortion_scale=0.4, p=.5,
                                                                            fill=(255, 255, 255),
                                                                            interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
        else:
            self.perspective_aug = None


class LipsConditionedDS(Dataset, ProcessFaceForensicsRoot):

    def prep_infer(self, paths_base, paths_driving, item):
        crop_base = files_utils.load_np("".join(paths_base[item]))
        lm_base = self.get_landmarks(paths_base[item], crop_base)
        driving_base = files_utils.load_np("".join(paths_driving[item]))
        lm_driving = self.get_landmarks(paths_driving[item], driving_base)
        image_in, image_ref, _, image_full, mask, _, _ = self.get_item_(crop_base, lm_base, crop_base)
        _, _, lm_driving, driving_image, _, _, _ = self.get_item_(driving_base, lm_driving, driving_base)
        lm_driving = torch.from_numpy(lm_driving)
        return image_in, image_ref, lm_driving, image_full, driving_image, mask

    @staticmethod
    def extract_lines(crop_base, lm_base):
        lips = lm_base[48:, :]
        key_points = [3, 9, 0, 6]  # height, width
        out = lips[key_points]
        out = (out / crop_base.shape[0]) * 2 - 1
        out = out.reshape(2, 2, 2)
        if np.equal(out[0, 0], out[0, 1]).all():
            out[0, 0, 1] += 1
        return out

    def get_item_(self, crop_base, lm_base, reference_image):
        mask = get_mask(crop_base, lm_base)
        mask_bg = np.ones_like(crop_base) * 255
        lines = self.extract_lines(crop_base, lm_base)
        if self.opt.draw_lips_lines:
            mask_bg = draw_lips_lines(mask_bg, lm_base)
        # files_utils.imshow(crop_lines)
        (crop, mask_bg, mask), _ = self.transform([crop_base, mask_bg, mask], lm_base, train=self.is_train)
        mask = mask.unsqueeze(0)
        # mask = (1 - nnf.max_pool2d(1 - mask, (5, 5), (1, 1), padding=2))[0]
        reference_image, _ = self.transform(reference_image, train=self.is_train)
        lm_lips = self.transform.landmarks_only(lm_base, crop_base, self.is_train)[48:, :]
        lm_lips = self.transform.zero_center_landmarks(lm_lips)
        masked = crop * (1 - mask) + mask * mask_bg
        # image_ = vis_landmark_on_img(crop.copy(), (lm_crop + 1) * 256)
        # files_utils.imshow(masked)
        return masked, reference_image, lm_lips.astype(np.float32), crop, mask, mask.sum(), lines

    def get_random_image_from_same_id(self, person_id):
        person_items = self.id2items[person_id]
        select = np.random.choice(len(person_items), 1)[0]
        info = self.metadata["info"][person_items[select]]
        crop_ref = files_utils.load_image(f"{self.out_root}{info['id']:07d}.png")
        return crop_ref

    def __getitem__(self, item: int):
        info, lm_base = [self.metadata[name][item] for name in ("info", "landmarks_crops")]
        crop_base = files_utils.load_image(f"{self.out_root}{info['id']:07d}.png")
        person_id = self.item2id[int(item)]
        crop_ref = self.get_random_image_from_same_id(person_id)
        data = self.get_item_(crop_base, lm_base, crop_ref)
        return person_id, *data

    def __init__(self, opt: options.OptionsLipsGenerator):
        super(LipsConditionedDS, self).__init__(opt.data_dir, -1)
        self.load()
        self.opt = opt
        opt.num_ids = self.num_ids
        self.is_train = True
        self.transform = LipsTransform(256, distortion_scale=.2, blur=False, color_jitter=False)


class LipsSeqDS(LipsConditionedDS):

    @staticmethod
    def explode_item_infer(item, history, max_len):
        items_seq = [item - history // 2 + i for i in range(history)]
        items_seq = [min(max(0, item), max_len - 1) for item in items_seq]
        return items_seq

    def prep_infer(self, paths_base, paths_driving, item):
        items_seq = self.explode_item_infer(item, self.opt.image_seq, len(paths_base))
        items_lips = self.explode_item_infer(item, self.opt.lips_seq, len(paths_driving))
        crops_base = [files_utils.load_np("".join(paths_base[item])) for item in items_seq]
        lms_base = [self.get_landmarks(paths_base[item], crops_base[i]) for i, item in enumerate(items_seq)]
        driving_base = [files_utils.load_np("".join(paths_driving[item])) for item in items_lips]
        lm_driving = [self.get_landmarks(paths_driving[item], driving_base[i]) for i, item in enumerate(items_lips)]
        image_in, image_ref, _, image_full, mask, _, _ = self.get_item_(crops_base, lms_base,
                                                                        crops_base[self.opt.image_seq // 2])
        lm_driving = self.transform_landmarks(lm_driving, driving_base[0])
        driving_image, _ = self.transform(driving_base[self.opt.lips_seq // 2], None, train=False)
        lm_driving = torch.from_numpy(lm_driving)
        return image_in, image_ref, lm_driving, image_full, driving_image, mask

    def transform_landmarks(self, landmarks, image):
        lms_base = np.concatenate(landmarks, axis=0)
        lm_lips = self.transform.landmarks_only(lms_base, image, self.is_train)
        lm_lips = lm_lips.reshape((lm_lips.shape[0] // 68, 68, 2))[:, 48:, :]
        lm_lips = self.transform.zero_center_landmarks(lm_lips)
        return lm_lips.astype(np.float32)

    def get_item_(self, crops_base, lms_base, reference_image):
        offset = (len(lms_base) - len(crops_base)) // 2
        mask = [get_mask(crop_base, lm_base) for crop_base, lm_base in zip(crops_base, lms_base)]
        mask_bgs = [np.ones_like(crops_base[0]) * 255 for _ in range(len(crops_base))]
        lines = [self.extract_lines(crop_base, lm_base) for crop_base, lm_base in zip(crops_base, lms_base[offset:])]
        if self.opt.draw_lips_lines:
            mask_bgs = [draw_lips_lines(mask_bg, lm_base) for mask_bg, lm_base in zip(mask_bgs, lms_base[offset:])]
        # files_utils.imshow(crop_lines)
        all_transform = crops_base + mask_bgs + mask
        all_transform, _ = self.transform(all_transform, None, train=self.is_train)
        (crop, mask_bg, mask) = all_transform[:self.opt.image_seq], all_transform[
                                                                    self.opt.image_seq: 2 * self.opt.image_seq], all_transform[
                                                                                                                 -self.opt.image_seq:]
        mask = torch.stack(mask)
        crop = torch.stack(crop)
        mask_bg = torch.stack(mask_bg)
        mask = 1 - nnf.max_pool2d(1 - mask, (5, 5), (1, 1), padding=2)
        reference_image, _ = self.transform(reference_image, train=self.is_train)
        lm_lips = self.transform_landmarks(lms_base, crops_base[0])
        masked = crop * (1 - mask) + mask * mask_bg
        lines = np.stack(lines, axis=0)
        masked = masked[:, :, 128:, 64: -64]
        reference_image = reference_image[:, 128:, 64: -64]
        mask = mask[:, :, 128:, 64: -64]
        return masked, reference_image, lm_lips.astype(np.float32), crop, mask, mask.sum(dim=(1, 2, 3)), lines

    def explode_items(self, item, history):
        out = [item]
        seq_number = self.metadata["info"][item]["video_path"]
        for i in range(history // 2):
            item_ = max(0, item - i - 1)
            seq_number_ = self.metadata["info"][item_]["video_path"]
            if seq_number_ != seq_number:
                item_ = out[0]
            out = [item_] + out
        for i in range(history // 2):
            item_ = min(len(self) - 1, item + i + 1)
            seq_number_ = self.metadata["info"][item_]["video_path"]
            if seq_number_ != seq_number:
                item_ = out[-1]
            out.append(item_)
        return out

    def __getitem__(self, item: int):
        items_seq = self.explode_items(item, self.opt.image_seq)
        items_lips = self.explode_items(item, self.opt.lips_seq)
        infos = [self.metadata['info'][item_] for item_ in items_seq]
        lms_base = [self.metadata['landmarks_crops'][item_] for item_ in items_lips]
        # img = np.ones((256, 256, 3)).astype(np.uint8)
        crops_base = [files_utils.load_image(f"{self.out_root}{info['id']:07d}.png") for info in infos]
        person_id = self.item2id[int(item)]
        crop_ref = self.get_random_image_from_same_id(person_id)
        try:
            data = self.get_item_(crops_base, lms_base, crop_ref)
        except BaseException as e:
            print(f"\n error: {item}:\n{e}")
            raise ValueError
        return person_id, *data

    def get_split(self, frac):
        sequence_length = self.metadata['sequence_length']
        select = np.random.choice(len(sequence_length), int(len(sequence_length) * frac), False).tolist()
        split_train = []
        split_val = []
        prev = 0
        for i in range(len(sequence_length)):
            inds = torch.arange(sequence_length[i] - prev) + prev
            if i in select:
                split_val.append(inds)
            else:
                split_train.append(inds)
            prev = sequence_length[i]
        return torch.cat(split_train), torch.cat(split_val)

    def __init__(self, opt: options.OptionsLipsGeneratorSeq):
        super(LipsSeqDS, self).__init__(opt)
        self.opt = opt
        # self.frontalize = frontalize_landmarks.FrontalizeLandmarks()


class LipsLandmarksDS(Dataset, ProcessFaceForensicsRoot):

    def __getitem__(self, item: int):
        info, lm_crop = [self.metadata[name][item] for name in ("info", "landmarks_crops")]
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


def get_lips_landmarks(img, lips):
    if type(lips) is T:
        lips: T = lips.detach().cpu()
        if lips.dim() > 2:
            lips = lips.squeeze(0)
        lips = (lips + 1) * (img.shape[0] / 2)
    if lips.shape[0] > 30:
        lips = lips[48:, :]
    return lips


class Line:

    def intersect_bounds(self, target, dim: int, bounds: Optional[Tuple[float, float]]):
        if self.direction[dim] == 0:
            return None
        t = (target - self.start[dim]) / self.direction[dim]
        intersection = self.start + t * self.direction
        if bounds[0] <= intersection[1 - dim] <= bounds[1]:
            return intersection
        else:
            return None

    def intersect_x(self, target_x, bounds_y):
        return self.intersect_bounds(target_x, 0, bounds_y)

    def intersect_y(self, target_y, bounds_x):
        return self.intersect_bounds(target_y, 1, bounds_x)

    def __init__(self, points):
        self.start = points[0]
        self.direction = points[1] - points[0]


def get_intersections(pair, image) -> ARRAY:
    line = Line(pair)
    out = []
    bound = (0, image.shape[0])
    for target, dim in zip((0, 0, image.shape[0], image.shape[0]), (0, 1, 0, 1)):
        intersection = line.intersect_bounds(target, dim, bound)
        if intersection is not None:
            out.append(intersection)
        if len(out) == 2:
            break
    out = V(out)
    return out


def draw_lips_lines(image, landmarks, in_place=True):
    lips = get_lips_landmarks(image, landmarks)
    key_points = ([3, 9], [0, 6])  # height, width
    if not in_place:
        image = image.copy()
    # lips = lips.astype(np.int32)
    for i, pair in enumerate(key_points):
        points = lips[pair]
        line = get_intersections(points, image).astype(np.int32)
        if len(line) == 0:
            if i == 0:
                line = [[points[0][0], 0], [points[0][0], 256]]
            else:
                line = [[0, points[0][1]], [256, points[0][1]]]
        cv2.line(image, line[0], line[1], (0, 0, 0), 1)
    return image


def draw_lips(img, lips, scale=1):
    if type(img) is int:
        img = np.ones((img, img, 3), dtype=np.uint8) * 255
    else:
        img = img.copy()
    lips = get_lips_landmarks(img, lips)
    if scale != 1:
        lips = scale * lips + .5 * img.shape[0] * (1 - scale)
    lips = lips.astype(np.int32)
    draw_curve = get_drawer(img, lips, 1)
    draw_curve(list(range(0, 11)), loop=True, color=(238, 130, 238))  # mouth
    draw_curve(list(range(12, 19)), loop=True, color=(238, 130, 238))
    # for i in range(20):
    #     print(i)
    #     img_ = img.copy()
    #     cv2.circle(img_, lips[i], 2, (255, 255, 255), thickness=-1)
    #     files_utils.imshow(img_)
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
    # points[0] = ((shape[1] + shape[2]) / 2).astype(shape.dtype)
    points[-1] = ((shape[15] + shape[14]) / 2).astype(shape.dtype)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    nose_y = shape[33, 1]
    mask[:nose_y] = 0
    return mask


def infer():
    dataset = LipsConditionedDS(options.OptionsLipsGenerator())
    dataset.is_train = False
    for i in range(5, len(dataset)):
        x = dataset[40 * i]
        files_utils.imshow(x[1])
        # files_utils.imshow(x[2])
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
    video = "../assets/raw_videos/office_michael.mp4"
    # video = constants.FaceForensicsRoot + 'downloaded_videos'
    # ProcessFaceForensicsRoot(constants.FaceForensicsRoot + 'processed_frames_all/', -1).run_all()
    ProcessFaceForensicsRoot(constants.MNT_ROOT + 'video_frames/office_jim/', -1).run(video, True)


if __name__ == '__main__':
    main()
