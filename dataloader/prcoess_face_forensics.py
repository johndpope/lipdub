from __future__ import annotations
from scipy.signal import savgol_filter
import torchvision
import align_faces
from utils import image_utils
import options
from custom_types import *
import constants
from utils import files_utils, train_utils, transformation_utils, landmarks_utils  #, frontalize_landmarks
from align_faces import FaceAlign
import imageio
from torchvision.transforms import functional as tsf
from PIL import Image
from models.sync_net import process_audio


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


def filter_lips(lips):
    lips = lips.reshape((-1, 204))
    lips[:, :48 * 3] = savgol_filter(lips[:, :48 * 3], 15, 3, axis=0)
    lips[:, 48 * 3:] = savgol_filter(lips[:, 48 * 3:], 5, 3, axis=0)
    lips = lips.reshape((-1, 68, 3))
    return lips


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

    def get_landmarks(self, path: List[str], image: Optional[ARRAY] = None) -> ARRAY:
        return self.aligner.get_landmarks_cached(path, image)

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

    @staticmethod
    def zero_center_landmarks(landmarks: ARRAY):
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
            startpoints, endpoints = self.perspective_aug.get_params(width, height, self.perspective_aug.distortion_scale)
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
        if landmarks is not None:
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
        mask = landmarks_utils.get_mask(crop_base, lm_base)
        mask_bg = np.ones_like(crop_base) * 255
        lines = self.extract_lines(crop_base, lm_base)
        if self.opt.draw_lips_lines:
            mask_bg = landmarks_utils.draw_lips_lines(mask_bg, lm_base)
        # files_utils.imshow(crop_lines)
        (crop, mask_bg, mask), _ = self.transform([crop_base, mask_bg, mask], lm_base, train=self.is_train)
        # mask = mask.unsqueeze(0)
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
        info,  lm_base = [self.metadata[name][item] for name in ("info", "landmarks_crops")]
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
        image_in, image_ref, _, image_full, mask, _, _ = self.get_item_(crops_base, lms_base, crops_base[self.opt.image_seq // 2])
        lm_driving = self.transform_landmarks(lm_driving, driving_base[0])
        driving_image, _ = self.transform(driving_base[self.opt.lips_seq // 2 ], None, train=False)
        lm_driving = torch.from_numpy(lm_driving)
        return image_in, image_ref, lm_driving, image_full, driving_image, mask

    def transform_landmarks(self, landmarks, image, is_train: Optional[bool]= None):
        lms_base = np.concatenate(landmarks, axis=0)
        lm_lips = self.transform.landmarks_only(lms_base, image, self.is_train if is_train is None else is_train)
        lm_lips = lm_lips.reshape((lm_lips.shape[0] // 68, 68, 2))[:, 48:, :]
        lm_lips = self.transform.zero_center_landmarks(lm_lips)
        return lm_lips.astype(np.float32)

    def get_item_(self, crops_base, lms_base, reference_image):
        offset = (len(lms_base) - len(crops_base)) // 2
        mask = [landmarks_utils.get_mask(crop_base, lm_base) for crop_base, lm_base in zip(crops_base, lms_base)]
        mask_bgs = [np.ones_like(crops_base[0]) * 255 for _ in range(len(crops_base))]
        lines = [self.extract_lines(crop_base, lm_base) for crop_base, lm_base in zip(crops_base, lms_base[offset:])]
        if self.opt.draw_lips_lines:
            mask_bgs = [landmarks_utils.draw_lips_lines(mask_bg, lm_base) for mask_bg, lm_base in zip(mask_bgs, lms_base[offset:])]
        # files_utils.imshow(crop_lines)
        all_transform = crops_base + mask_bgs + mask
        all_transform, _ = self.transform(all_transform, None, train=self.is_train)
        (crop, mask_bg, mask) = all_transform[:self.opt.image_seq], all_transform[self.opt.image_seq: 2 * self.opt.image_seq], all_transform[-self.opt.image_seq:]
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

    def get_split_single(self, frac):
        val_length = int(len(self) * frac)
        select = np.random.choice(len(self), val_length, False).tolist()
        val_inds = []
        for item in select:
            val_inds += list(range(item, min(len(self), item + 100)))
            val_inds = list(set(val_inds))
            if len(val_inds) >= val_length:
                break
        train_inds = torch.tensor([i for i in range(len(self)) if i not in val_inds], dtype=torch.int64)
        val_inds = torch.tensor([i for i in val_inds if i < len(self)], dtype=torch.int64)
        return train_inds, val_inds

    def get_split(self, frac):
        sequence_length = self.metadata['sequence_length']
        if len(sequence_length) == 0:
            return self.get_split_single(frac)
        target_val_length =  int(len(self) * frac)
        if target_val_length > 10000:
            frac = 10000. / len(self)
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


class LipsSeqDSDual(LipsSeqDS):

    def __getitem__(self, item):
        person_id, masked, reference_image, _, crop, mask, mask_sum, lines = super(LipsSeqDSDual, self).__getitem__(item)
        items_lips = self.explode_items(item, self.opt.lips_seq)
        lms_base = [self.landmarks_sec[item_] for item_ in items_lips]
        lm_lips = self.transform_landmarks(lms_base, self.crop_base, False)
        return person_id, masked, reference_image, lm_lips, crop, mask, mask_sum, lines

    def __len__(self):
        return min(super(LipsSeqDSDual, self).__len__(), len(self.landmarks_sec))

    def __init__(self, opt: options.OptionsLipsGeneratorSeq, landmarks_root: str):
        super(LipsSeqDSDual, self).__init__(opt)
        self.is_train = True
        metadata = files_utils.load_pickle(f'{constants.MNT_ROOT}video_frames/{landmarks_root}/metadata')
        self.landmarks_sec = metadata['landmarks_crops']
        self.crop_base = files_utils.load_image(f"{constants.MNT_ROOT}video_frames/{landmarks_root}/{metadata['info'][0]['id']:07d}.png")



class LipsSeqDSInfer(Dataset):

    @staticmethod
    def explode_item_infer(item, history, max_len):
        items_seq = [item - history // 2 + i for i in range(history)]
        items_seq = [min(max(0, item), max_len - 1) for item in items_seq]
        return items_seq

    def __getitem__(self, item):
        items_seq = self.explode_item_infer(item, self.opt.image_seq, len(self.paths_base))
        items_lips = self.explode_item_infer(item, self.opt.lips_seq, len(self.paths_driving))
        crops_base = [files_utils.load_np("".join(self.paths_base[item])) for item in items_seq]
        crops_base = [image_utils.resize(item, 256) for item in crops_base]
        lms_base = [self.lm_base[item] for item in items_seq]
        lm_driving = np.stack([self.lm_driving[item] for item in items_lips])
        image_in, image_ref, _, image_full, mask, _, _ = self.train_ds.get_item_(crops_base, lms_base,
                                                                        crops_base[self.opt.image_seq // 2])
        lm_driving = torch.from_numpy(lm_driving).float()
        return image_in, image_ref, lm_driving, image_full, mask, self.audio[item]

    @property
    def opt(self):
        return self.train_ds.opt

    def __len__(self):
        return min(len(self.paths_base), len(self.paths_driving), len(self.audio))

    @staticmethod
    def get_frames(folder, starting_time, ending_time):
        paths = files_utils.collect(folder, '.npy')
        paths = [path for path in paths if 'image_' in path[1] or 'crop_' in path[1]]
        metadata = files_utils.load_pickle(f"{folder}/metadata")
        landmarks_2d = metadata['landmarks_2d']
        depth = metadata['landmarks'][:, :, -1:]
        landmarks = np.concatenate((landmarks_2d, depth), axis=2)
        fps = metadata['fps']
        if starting_time > 0:
            paths = paths[int(fps * starting_time):]
            landmarks = landmarks[int(fps * starting_time):]
        else:
            starting_time = 0
        if ending_time > 0:
            num_frames = int((ending_time - starting_time) * fps)
            paths = paths[:num_frames]
            landmarks = landmarks[:num_frames]
        return paths, landmarks

    def load_makeittalk_lm(self, path):
        fl_driving = np.load(path).reshape((-1, 68, 3))
        fl_driving = torch.from_numpy(fl_driving).reshape(1, -1, 68 * 3).permute(0, 2, 1)
        fl_driving = nnf.interpolate(fl_driving, size=len(self.paths_driving), mode='linear')
        fl_driving = fl_driving.permute(0, 2, 1).squeeze().reshape(-1, 68, 3).numpy()
        fl_driving = fl_driving * 4
        return fl_driving

    @staticmethod
    def get_normalized_lips(lm, ref_image):
        lips = (lm[:, 48:] / ref_image.shape[0]) * 2 - 1
        lips = LipsTransform.zero_center_landmarks(lips)
        n, l, d = lips.shape
        std = lips.reshape((n*l, d)).std(axis=0)
        return lips, std

    def align_lips(self, lm_driving, lm_base, align_driving):
        if align_driving:
            lm_driving = adjust_lm(lm_driving, lm_base[:len(lm_driving)])
        ref_image = files_utils.load_np("".join(self.paths_base[0]))
        lips_driving, std_driving = self.get_normalized_lips(lm_driving, ref_image)
        return lips_driving[:, :, :2], lm_base[:, :, :2] / 4

    # def align_lips(self, lm, lm_base):
    #     aligned_lm = transformation_utils.align_landmarks(lm, lm_base[:len(self)])
    #     # aligned_lm = aligned_lm.reshape((-1, 204))
    #     # aligned_lm[:, :48 * 3] = savgol_filter(aligned_lm[:, :48 * 3], 15, 3, axis=0)
    #     # aligned_lm[:, 48 * 3:] = savgol_filter(aligned_lm[:, 48 * 3:], 5, 3, axis=0)
    #     # aligned_lm = aligned_lm.reshape((-1, 68, 3))
    #     ref_image = files_utils.load_np("".join(self.paths_base[0]))
    #     lips_driving, std_driving = self.get_normalized_lips(aligned_lm, ref_image)
    #     _, std_base = self.get_normalized_lips(lm_base, ref_image)
    #     scale = std_base / std_driving
    #     lips_driving = lips_driving * scale[None, None, :]
    #     return lips_driving[:, :, :2] , lm_base[:, :, :2] / 4

    def get_mel_data(self, folder):
        mel_data_path = f"{folder}/mel_data.npy"
        if files_utils.is_file(mel_data_path):
            mel_data = files_utils.load_np(mel_data_path)
        else:
            fps = files_utils.load_pickle(f"{folder}/metadata")["fps"]
            mel_data = process_audio(f"{folder}/audio.wav", fps)
            files_utils.save_np(mel_data, mel_data_path)
        return mel_data

    def __init__(self, train_ds: LipsSeqDS, base_folder: str, driving_folder: str, starting_time: int = -1,
                 ending_time: int = -1, align_driving=True):
        self.train_ds = train_ds
        self.train_ds.is_train = False
        self.paths_base, lm_base = self.get_frames(base_folder, starting_time, ending_time)
        self.paths_driving, lm_driving = self.get_frames(driving_folder, -1, -1)
        if len(self.paths_base) < len(self.paths_driving):
            self.paths_driving, lm_driving = self.paths_driving[:len(self.paths_base)], lm_driving[:len(self.paths_base)]
        # if makeittalk_lm_path is not None:
        #     lm_driving = self.load_makeittalk_lm(makeittalk_lm_path)
        self.lm_driving, self.lm_base = self.align_lips(lm_driving, lm_base, align_driving)
        self.audio = self.get_mel_data(driving_folder)
        # self.lm_base, self.lm_driving = pass


class LipsLandmarksDS(Dataset, ProcessFaceForensicsRoot):

    def __getitem__(self, item: int):
        info,  lm_crop = [self.metadata[name][item] for name in ("info", "landmarks_crops")]
        crop = files_utils.load_image(f"{self.out_root}{info['id']:07d}.png")
        crop, lm_crop = self.transform(crop, lm_crop)
        lm_lips = lm_crop[48:, :]


        if DEBUG:
            crop = files_utils.image_to_display(crop)
            image_ = landmarks_utils.draw_lips(crop.copy(), (lm_crop + 1) * self.transform.res / 2.)
            files_utils.imshow(image_)
        return crop, lm_lips

    def __init__(self, opt):
        super(LipsLandmarksDS, self).__init__(opt.data_dir)
        self.load()
        self.transform = LipsTransform(240)


def infer():
    dataset = LipsConditionedDS(options.OptionsLipsGenerator())
    dataset.is_train = False
    align = align_faces.FaceAlign()
    for i in range(5, len(dataset), 40):
        info, lm_base = [dataset.metadata[name][55] for name in ("info", "landmarks_crops")]
        crop_base = files_utils.load_image(f"{dataset.out_root}{info['id']:07d}.png")
        lm_3d = align.predictor_3d.get_landmarks(crop_base)[0]

        lm_3d[:, :2] = lm_base
        files_utils.save_np(lm_3d, f"{constants.CACHE_ROOT}/template_landmarks")
        return
        image = landmarks_utils.vis_landmark_on_img(np.ones((512, 512, 3), dtype=np.uint8) * 255, lm_3d)
        files_utils.imshow(image)
        # x = dataset[40 * i]
        # files_utils.imshow(x[1])
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
    video = f"{constants.MNT_ROOT}/marz_dub/anthony_bourdain_english_faceformer.mp4"
    # video = constants.FaceForensicsRoot + 'downloaded_videos'
    # ProcessFaceForensicsRoot(constants.FaceForensicsRoot + 'processed_frames_all/', -1).run_all()
    ProcessFaceForensicsRoot(constants.MNT_ROOT + 'video_frames/anthony_bourdain_english_faceformer/', -1).run(video, True)


def finalize_video(sequence, driven_dir, name):
    from moviepy import editor
    metadata = files_utils.load_pickle(f"{driven_dir}/metadata")
    sample_rate, audio = files_utils.load_wav(f"{driven_dir}/audio")
    audio_out_len = float(audio.shape[0] * len(sequence)) / metadata['frames_count']
    files_utils.save_wav(audio[:int(audio_out_len)], sample_rate, name + '.wav')
    image_utils.gif_group(sequence, name + 'tmp', metadata['fps'])
    video_clip = editor.VideoFileClip(name + 'tmp.mp4')
    audio_clip = editor.AudioFileClip(name + '.wav')
    audio_clip = editor.CompositeAudioClip([audio_clip])
    video_clip.audio = audio_clip
    video_clip.write_videofile(f'{name}.mp4')
    video_clip.close()
    files_utils.delete_single(name + 'tmp.mp4')
    files_utils.delete_single(name + '.wav')


def debug_landmarks(root, max_len=-1):
    key = 'landmarks_2d'
    name: str = root.split('/')[-2]
    paths = files_utils.collect(root, '.npy')
    paths = [path for path in paths if 'crop' in path[1]]
    metadata = files_utils.load_pickle(f'{root}/metadata')
    lms = metadata[key]
    # lms = lms.reshape((-1, 204))
    # lms[:, :48 * 3] = savgol_filter(lms[:, :48 * 3], 15, 3, axis=0)
    # lms[:, 48 * 3:] = savgol_filter(lms[:, 48 * 3:], 5, 3, axis=0)
    # lms = lms.reshape((-1, 68, 3))
    out = []
    if max_len > 0 and metadata['fps'] * max_len < len(paths):
        lms = lms[:int(max_len * metadata['fps'])]
        paths = paths[:int(max_len * metadata['fps'])]
    for path, lm in zip(paths, lms):
        crop = files_utils.load_np(''.join(path))
        crop = landmarks_utils.vis_landmark_on_img(crop, lm)
        out.append(crop)
    finalize_video(out, root, f'{constants.DATA_ROOT}/debug/{name}_lm2d')
    # image_utils.gif_group(out, f'{constants.DATA_ROOT}/debug/{name}_lm2d_tmp', 30)


def get_center_std(lips):
    center = (lips.max(axis=(0, 1)) + lips.min(axis=(0, 1))) / 2
    lips = lips - center[None, None, :]
    return lips, center, lips.std(axis=(0, 1))

def lips_squares(lips_target, lips_source):
    # mean_source, std_source = lips_source.mean(axis=(0, 1)), lips_source.std(axis=(0, 1))
    # mean_target, std_target = lips_target.mean(axis=(0, 1)), lips_target.std(axis=(0, 1))
    # lips_target = lips_target - mean_target[None, None, :]
    # lips_source = lips_source - mean_source[None, None, :]
    lips_out = lips_source[:, :12]
    lips_in = lips_source[:, 12:]
    # lips_out_target = lips_target[:, :12].reshape((lips_target.shape[0], 36))
    lips_in_target = lips_target[:, 12:]
    lips_in_target, center_target, std_target = get_center_std(lips_in_target)
    lips_in, center, std = get_center_std(lips_in)
    lips_in_target = lips_in_target * (std / std_target)[None, None, :]
    lips_out = lips_out - center[None, None, :]
    lips_in_target = lips_in_target.reshape((lips_in_target.shape[0], 24))
    lips_out = lips_out.reshape((lips_source.shape[0], 36))
    lips_in = lips_in.reshape((lips_source.shape[0], 24))
    in2out, residual = np.linalg.lstsq(lips_in, lips_out, rcond=None)[:2]
    out2in, residual = np.linalg.lstsq(lips_out, lips_in, rcond=None)[:2]
    lips_out_predict = np.einsum('nm,mk->nk', lips_in_target, in2out)
    lips_in_predict = lips_in_target
    # lips_in_predict = np.einsum('nm,mk->nk', lips_out_predict, out2in)
    lips_new = np.concatenate((lips_out_predict, lips_in_predict), axis=1)
    lips_new = lips_new.reshape((lips_in_target.shape[0], 20, 3)) + center_target[None, None, :]
    return lips_new


def adjust_lm(fl_driving, fl_source):
    template = files_utils.load_np(f"{constants.CACHE_ROOT}/template_landmarks") * 2
    template = template.astype(fl_driving.dtype)
    template = np.expand_dims(template, axis=0)
    template = np.repeat(template, fl_driving.shape[0], axis=0)
    aligned_all_fl_driving = transformation_utils.align_landmarks(fl_driving, template)
    aligned_all_fl_source = transformation_utils.align_landmarks(fl_source, template)
    lips_source = aligned_all_fl_source[:, 48:]
    lips_target = aligned_all_fl_driving[:, 48:]
    lips_target = lips_squares(lips_target, lips_source)
    # lips_target = lips_target * std_source[None, :, :] / std_target[None, :, :]
    aligned_all_fl_driving[:, 48:] = lips_target
    aligned_all_fl_driving = transformation_utils.align_landmarks(aligned_all_fl_driving, fl_source)
    return aligned_all_fl_driving


def makeittalk(starttime=10):
    name = "Eilish"
    out_path = f"{constants.MNT_ROOT}/processed_infer/Eilish"
    driving = f'{constants.MNT_ROOT}/processed_infer/Eilish_French052522'
    key = 'landmarks_2d'
    fl_source = files_utils.load_pickle(f'{out_path}/metadata')[key][starttime * 30:]
    fl_driving = files_utils.load_pickle(f'{driving}/metadata')[key]
    if key == 'landmarks':
        fl_source = filter_lips(fl_source)
        fl_driving = filter_lips(fl_driving)
    else:
        fl_source_depth = files_utils.load_pickle(f'{out_path}/metadata')['landmarks'][starttime * 30:, :, -1:]
        fl_driving_depth = files_utils.load_pickle(f'{driving}/metadata')['landmarks'][:, :, -1:]
        fl_source = np.concatenate((fl_source, fl_source_depth), axis=2)
        fl_driving = np.concatenate((fl_driving, fl_driving_depth), axis=2)
    # fl_source[:, :, 2] = fl_source[:, :, 2] - fl_source[:, :, 2].min()
    # fl_source[:, :, 2] = fl_source[:, :, 2] / fl_source[:, :, 2].max()
    paths = files_utils.collect(out_path, '.npy')
    paths_base = [path for path in paths if 'crop' in path[1]][starttime * 30:]

    paths_driving = files_utils.collect(driving, '.npy')
    paths_driving = [path for path in paths_driving if 'crop' in path[1]]
    fl_source = fl_source[:len(fl_driving)]
    base = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    aligned_all = adjust_lm(fl_driving, fl_source)
    out = []
    aligned_mid = transformation_utils.align_landmarks(fl_driving, fl_source)
    for i in range(0, len(fl_driving)):
        # crop = files_utils.load_np(''.join(paths[i]))
        image_a = landmarks_utils.vis_landmark_on_img(base.copy(), fl_source[i])
        image_b = landmarks_utils.vis_landmark_on_img(base.copy(), fl_driving[i])
        image_c = landmarks_utils.vis_landmark_on_img(base.copy(), aligned_mid[i])
        image_d = landmarks_utils.vis_landmark_on_img(base.copy(), aligned_all[i])

        image_all = np.concatenate((image_a, image_b, image_c, image_d), axis=1)
        image_all = Image.fromarray(image_all)
        image_all = V(image_all.resize((256*3, 256), resample=Image.BICUBIC))
        out.append(image_all)

        # files_utils.imshow(image_all)
        # files_utils.imshow(image_b)
        # files_utils.imshow(image_c)
    finalize_video(out, driving, f'{constants.DATA_ROOT}/debug/{name}_transfer_2d_scale')
    # image = np.ones((256, 256, 3), dtype=np.uint8) * 255
    # images = []
    # for i in range(len(fl)):
    #     img = vis_landmark_on_img(image.copy(), fl[i])
    #     images.append(img)
    #     # files_utils.imshow(img)
    # image_utils.gif_group(images, '/home/ahertz/projects/MakeItTalk/examples/pred_fls_BillieEilish_French_lm', 30)


if __name__ == '__main__':
    # infer()
    makeittalk()
    # main()
