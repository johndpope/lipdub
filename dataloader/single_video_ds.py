import cv2
from custom_types import *
from options import OptionsA2S
from utils import files_utils
import constants


FACEMESH_FACE_OVAL_BOTTOM = np.arange(10, 27)


class SingleVideoDS(Dataset):

    def load_image(self, item: int):
        path = f"{self.root}/image_{self.frame_numbers[item]:08d}"
        image = files_utils.load_np(path)
        image = image.astype(np.float32) / 127.5 - 1
        image = torch.from_numpy(image)
        return image

    def load_mask(self, item: int):
        if self.mask[item] is None:
            contours = self.contours[self.frame_numbers[item]]
            mask = np.zeros((256, 256), dtype=np.uint8)
            # mask = cv2.polylines(mask, [contours], True, 255, torch.randint(2, 6, (1,)).item())
            mask = cv2.polylines(mask, [contours], True, 255, 4)
            mask_pt = contours[FACEMESH_FACE_OVAL_BOTTOM][:, 1].min()
            mask[mask_pt:, :] = 255
            # path = f"{self.root}/{self.name}_mask_{self.frame_numbers[item]:08d}"
            # mask = files_utils.load_np(path)
            mask = torch.from_numpy(mask)
            self.mask[item] = mask
        return self.mask[item]

    def load_images(self, item):
        images = [self.load_image(item + i) for i in range(self.frames_per_item)]
        return images

    def load_masks(self, item):
        masks = [self.load_mask(item + i) for i in range(self.frames_per_item)]
        return masks

    def prepare_audio_input(self, item):
        item = self.sound_offset + item
        audio_2_frame = self.audio.shape[0] / self.metadata['num_frames']
        start = int(audio_2_frame * (self.frame_numbers[item] - (self.opt.audio_per_item - self.opt.frames_per_item) // 2))
        out = self.audio[start: start + self.audio_seq_size]
        out = out.view(-1, 13 * self.opt.audio_multiplier)
        return out

    def prepare_images_input(self, images, masks):
        # out = []
        # for image, mask in zip(images, masks):
        #     masked = image.clone()
        #     masked[~mask[0]] = -1
        #     masked[mask[1]] = 1
        #     out.append(masked.permute(2, 0, 1))
        images_input = torch.stack(masks, dim=0).unsqueeze(0).float() / 255.
        if self.opt.image_input_resolution != images_input.shape[-1]:
            images_input = nnf.interpolate(images_input, self.opt.image_input_resolution, mode='bicubic', align_corners=True)
        images_input = (images_input > .25).float()
        return images_input

    def get_mask_output(self, item: int) -> T:
        if self.mask_out[item] is None:
            contours = self.contours[self.frame_numbers[item]]
            # contours = contours[FACEMESH_FACE_OVAL_BOTTOM]
            mask = np.zeros((256, 256), dtype=np.uint8)
            mask = cv2.fillPoly(mask, [contours], 255)
            mask = torch.from_numpy(mask > 1).float()
            self.mask_out[item] = mask
        return self.mask_out[item]

    def prepare_image_output(self, images, item):
        if self.linear_mode:
            select = self.opt.frames_per_item // 2
        else:
            select = torch.randint(len(images), (1,)).item()
        image = images[select].permute(2, 0, 1)
        mask = self.get_mask_output(item + select)
        # contours = self.contours[self.frame_numbers[item]]
        # mask = masks[select][1].float()
        if self.opt.stylegan_size != image.shape[-1]:
            image = nnf.interpolate(image.unsqueeze(0), self.opt.stylegan_size, mode='bicubic', align_corners=True)[0]
            mask = nnf.interpolate(mask.unsqueeze(0).unsqueeze(0), self.opt.stylegan_size)[0]
        return image, mask, select

    def __getitem__(self, item: int):
        images = self.load_images(item)
        masks = self.load_masks(item)
        images_input = self.prepare_images_input(images, masks)
        image_output, mask_output, t = self.prepare_image_output(images, item)
        audio_input = self.prepare_audio_input(item)
        return audio_input, images_input * 0, image_output, mask_output, t

    def __len__(self):
        return len(self.frame_numbers) - self.frames_per_item

    @property
    def name(self):
        return self.opt.video_name

    @property
    def frames_per_item(self):
        return self.opt.frames_per_item

    @staticmethod
    def init_audio(sound_path: str):
        audio = files_utils.load_np(f"{sound_path}_mfcc")
        audio = audio - audio.mean(0)[None, :]
        audio = audio / audio.std(0)[None, :]
        audio = torch.from_numpy(audio).float()
        return audio

    def set_sound(self, sound_path: str):
        self.audio = self.init_audio(sound_path)
        self.audio_wav = sound_path

    def set_sound_offset(self, offset):
        self.sound_offset = offset

    def to_linear(self):
        self.linear_mode = True

    def to_shuffle(self):
        self.linear_mode = False

    def export_sub_sound(self, start_item, end_item):
        audio_path = f'{self.audio_wav}.wav'
        sample_rate, data = files_utils.load_wav(audio_path)
        frame2sound = float(sample_rate) / self.metadata['fps']
        # start_frame = self.frame_numbers[start_item] - (self.opt.audio_per_item - self.opt.frames_per_item) // 2
        start_frame = self.frame_numbers[start_item + self.opt.frames_per_item // 2]
        end_frame = start_frame + end_item
        audio_seq = data[int(start_frame * frame2sound): int(end_frame * frame2sound)]
        files_utils.save_wav(audio_seq, sample_rate, f'{self.opt.cp_folder}/predict/seq_obama_b.wav')
        return

    def __init__(self, opt: OptionsA2S):
        self.opt = opt
        self.root = f"{constants.CACHE_ROOT}/{self.name}/"
        self.metadata = files_utils.load_pickle(f"{self.root}/{self.name}_metadata")
        contours = files_utils.load_np(f'{self.root}/face_contours').round().astype(np.int32)
        self.contours = np.roll(contours, 1, -1)
        paths = files_utils.collect(self.root, '.npy', prefix=f'image_')
        self.frame_numbers = list(map(lambda x: int(x[1].split('_')[-1]), paths))
        self.audio_wav = f"{self.root}/{self.name}"
        self.audio = self.init_audio(self.audio_wav)
        audio_seq_size = int(self.audio.shape[0] / self.metadata['num_frames'] * self.opt.audio_per_item)
        self.audio_seq_size = audio_seq_size + audio_seq_size % self.opt.audio_multiplier
        self.linear_mode = False
        self.sound_offset = 0
        self.mask: TNS = [None] * len(self)
        self.mask_out: TNS = [None] * len(self)


if __name__ == '__main__':
    ds = SingleVideoDS(OptionsA2S())
    for i in range(len(ds)):
        mask = ds.get_mask_output(i)
        mask = mask.numpy().astype(np.bool)
        files_utils.save_np(mask, f"{constants.CACHE_ROOT}/{ds.name}_maskrcnn/mask_{ds.frame_numbers[i]:08d}")
    # for i in range(100):
    #     audio_input, images_input, image_output, mask_output, t = ds[i]
    #     files_utils.imshow(images_input[0])
    #     files_utils.imshow(image_output * mask_output[None])
    # ds.set_sound('/home/ahertz/projects/StyleFusion-main/assets/cache//obama_b/obama_b')
    # ds.export_sub_sound(1000, 600)

