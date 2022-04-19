import os.path
from utils import files_utils, train_utils, image_utils
import dlib
import tqdm
from custom_types import *
import imageio
import PIL
import scipy
from scipy.ndimage.filters import gaussian_filter1d
import constants


IMAGE_SIZE = 1024

def crop_image(filepath, quad, enable_padding=False):
    x = (quad[3] - quad[1]) / 2
    qsize = np.hypot(*x) * 2
    # read image
    if isinstance(filepath, PIL.Image.Image):
        img = filepath
    elif isinstance(filepath, ARRAY):
        img = PIL.Image.fromarray(filepath)
    else:
        img = PIL.Image.open(filepath)
    transform_size = IMAGE_SIZE
    # Shrink.
    shrink = int(np.floor(qsize / IMAGE_SIZE * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if (crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]):
        img = img.crop(crop)
        quad -= crop[0:2]
    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if IMAGE_SIZE < transform_size:
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), PIL.Image.ANTIALIAS)
    return img


class FaceAlign:

    def get_landmark(self, image):
        """get landmark with dlib
        :return: np.array shape=(68, 2)
        """
        if self.fa is not None:
            lms, _, bboxes = self.fa.get_landmarks(image, return_bboxes=True)
            if len(lms) == 0:
                return None
            return lms[0]

        if self.detector is None:
            self.detector = dlib.get_frontal_face_detector()
        dets = self.detector(image)

        for k, d in enumerate(dets):
            shape = self.predictor(image, d)
            break
        else:
            return None
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        return lm

    def compute_transform(self, image, scale=1.0):
        lm = self.get_landmark(image)
        if lm is None:
            raise Exception(f'Did not detect any faces in image')
        lm_chin = lm[0: 17]  # left-right
        lm_eyebrow_left = lm[17: 22]  # left-right
        lm_eyebrow_right = lm[22: 27]  # left-right
        lm_nose = lm[27: 31]  # top-down
        lm_nostrils = lm[31: 36]  # top-down
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        lm_mouth_inner = lm[60: 68]  # left-clockwise
        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

        x *= scale
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        return c, x, y

    def crop_faces_by_quads(self, vid, quads):
        orig_images, crops = [], []
        self.logger.start(len(quads))
        for i, quad in enumerate(quads):
            frame = vid.get_data(i)
            crop = crop_image(frame, quad.copy())
            orig_images.append(frame)
            crops.append(crop)
            self.logger.reset_iter()
        self.logger.stop()
        return crops, orig_images

    def crop_faces(self, vid, num_frames, scale=1., center_sigma=1., xy_sigma=3.0, use_fa=False):
        cs, xs, ys = [], [], []
        self.logger.start(num_frames)
        for i in range(num_frames):
            frame = vid.get_data(i)
            c, x, y = self.compute_transform(frame, scale=scale)
            cs.append(c)
            xs.append(x)
            ys.append(y)
            self.logger.reset_iter()
        self.logger.stop()
        cs = np.stack(cs)
        xs = np.stack(xs)
        ys = np.stack(ys)
        if center_sigma != 0:
            cs = gaussian_filter1d(cs, sigma=center_sigma, axis=0)

        if xy_sigma != 0:
            xs = gaussian_filter1d(xs, sigma=xy_sigma, axis=0)
            ys = gaussian_filter1d(ys, sigma=xy_sigma, axis=0)

        quads = np.stack([cs - xs - ys, cs - xs + ys, cs + xs + ys, cs + xs - ys], axis=1)
        crops, orig_images = self.crop_faces_by_quads(vid, list(quads))

        return crops, orig_images, quads

    def crop_video(self, video_path: str, out_path: str, max_len: int):
        vid = imageio.get_reader(f"{video_path}", 'ffmpeg')
        fps = vid._meta['fps']
        num_frames = min(int(fps * max_len), vid.count_frames())
        crops, orig_images, quads = self.crop_faces(vid, num_frames)
        files_utils.save_np(quads, f'{out_path}/quads')
        image_utils.gif_group(crops, f'{out_path}/orig_align', fps)
        for i, crop in enumerate(crops):
            files_utils.save_np(crop, f'{out_path}/crop_{i:04d}')

    def __init__(self):
        self.predictor = dlib.shape_predictor("./weights/ffhq/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.fa = None
        self.logger = train_utils.Logger()
        # self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)


if __name__ == '__main__':
    aligner = FaceAlign()
    aligner.crop_video(f"/mnt/r/projects/facesimile/shots/101/purpledino_genwoman/comp_wip/images/comp_main_publish_qt/101_purpledino_genwoman_comp_v010.mov",
                       f"{constants.DATA_ROOT}/101_purpledino_genwoman_comp_v010", 10000)

    # crop_video(f"{constants.DATA_ROOT}/raw_videos/obama_062814", 30, f"{constants.DATA_ROOT}/processed")
    # viseme2vec(f"{constants.DATA_ROOT}/raw_videos/obama_062814",
    #             f"{constants.DATA_ROOT}/processed/viseme_obama_062814",
    #            f"{constants.DATA_ROOT}/processed/viseme_vec_obama_062814")
    # p2fa_folder(f"{constants.DATA_ROOT}/processed/")
    # split_audio(f"{constants.DATA_ROOT}/raw_videos/obama_062814",
    #             f"{constants.DATA_ROOT}/processed/obama_062814")
    # split_text(f"{constants.DATA_ROOT}/raw_videos/obama_062814.txt",
    #            f"{constants.DATA_ROOT}/processed/obama_062814.txt")