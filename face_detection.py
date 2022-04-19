import os.path

import dlib
from encoder4editing.e4e_utils.alignment import get_landmark, align_by_lm
from custom_types import *
import cv2
import mediapipe as mp
import imageio
from utils import files_utils, image_utils, train_utils
import constants
import ffmpeg
from scipy.io import wavfile
import python_speech_features
from scipy.ndimage.filters import gaussian_filter1d



FACEMESH_FACE_OVAL = V([(10, 338), (338, 297), (297, 332), (332, 284),
                                (284, 251), (251, 389), (389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162), (162, 21), (21, 54),
                                (54, 103), (103, 67), (67, 109), (109, 10)], dtype=np.int32)

FACEMESH_FACE_OVAL_BOTTOM = V([(361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 361)], dtype=np.int32)
# FACEMESH_LIPS = V([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
#                            (17, 314), (314, 405), (405, 321), (321, 375),
#                            (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
#                            (37, 0), (0, 267),
#                            (267, 269), (269, 270), (270, 409), (409, 291), (291, 146)], dtype=np.int32)
#                            # (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
#                            # (14, 317), (317, 402), (402, 318), (318, 324),
#                            # (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
#                            # (82, 13), (13, 312), (312, 311), (311, 310),
#                            # (310, 415), (415, 308)], dtype=np.int32)
FACEMESH_LIPS = V([(291, 409), (409, 270), (270, 269), (269, 267), (267, 0), (0, 37), (37, 39), (39, 40), (40, 185), (185, 61), (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                   (17, 314), (314, 405), (405, 321), (321, 375),
                   (375, 291), ], dtype=np.int32)
                   # (61, 185), (185, 40), (40, 39), (39, 37),
                   #         (37, 0), (0, 267),
                   #         (267, 269), (269, 270), (270, 409), (409, 291), (291, 61)], dtype=np.int32)
                           # (78, 95), (95, 88), (88, 178), (178, 87), (87, 14)
                           # (14, 317), (317, 402), (402, 318), (318, 324),
                           # (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           # (82, 13), (13, 312), (312, 311), (311, 310),
                           # (310, 415), (415, 308)], dtype=np.int32)


def get_oval(image, landmarks):
    landmark_arr = landmarks_to_arr(landmarks, image)
    landmark_oval = landmark_arr[FACEMESH_FACE_OVAL][:, 0]
    return landmark_oval
    # landmark_oval = landmark_arr[FACEMESH_FACE_OVAL][:, 0]
    masks = []
    for select in (FACEMESH_FACE_OVAL, FACEMESH_FACE_OVAL_BOTTOM):
        points = landmark_arr[select][:, 0]
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [np.roll(points, 1, -1)], False, 255)
        masks.append(mask > 0)
    masks = np.stack(masks, axis=0)
    return masks


def get_lips(image, landmarks):
    # landmark_arr = landmarks_to_arr(landmarks, image)
    points = landmarks[FACEMESH_LIPS][:, 0]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = cv2.fillPoly(mask, [np.roll(points, 1, -1).astype(np.int32)], True,  255)
    files_utils.imshow(mask)
    return mask



def get_poly(image, landmarks):
    landmark_arr = landmarks_to_arr(landmarks, image).astype(np.int32)
    masks = []
    for select in (FACEMESH_FACE_OVAL, FACEMESH_FACE_OVAL_BOTTOM):
        points = landmark_arr[select][:, 0]
        mask = image.copy()
        mask = cv2.polylines(mask, [np.roll(points, 1, -1)], True, (0, 255, 0), 3)

        masks.append(mask)
    # masks = np.stack(masks, axis=0)
    files_utils.save_image(V(masks[0]), "./assets/cache/for_slides/image_full")
    return masks


class VidDs(Dataset):

    def __len__(self):
        return self.length

    def fill_cache(self, item: int) -> ARRAY:
        if self.cache[item] is None:
            self.cache[item] = self.vid.get_data(item)
            if self.to_tensor:
                self.cache[item] = image_utils.im2tensor(self.cache[item])
        return self.cache[item]

    def __getitem__(self, item):
        frame_image: ARRAY = self.fill_cache(item)
        return frame_image, self.t[:, item]

    def get_metadata(self):
        metadata = self.vid._meta
        return {'num_frames': self.length, 'fps': metadata['fps'], 'duration': metadata['duration']}

    def get_length(self):
        counter = 0
        # return self.vid.count_frames()
        for _ in self.vid.iter_data():
            counter += 1
        return counter

    def __init__(self, video_file: str, to_tensor: bool):
        self.vid = imageio.get_reader(video_file, 'ffmpeg')
        # print(self.vid.count_frames())
        self.length = self.get_length()
        self.to_tensor = to_tensor
        self.cache: List[Union[ARRAY, N]] = [None] * self.length
        # self.cache = [self.vid.get_data(item) for item in range(self.length)]
        # if self.to_tensor:
        #     self.cache = [image_utils.im2tensor(item) for item in self.cache]
        self.t = torch.linspace(0, 1, self.length).unsqueeze(0)


def landmarks_to_arr(landmark: List[Dict[str, float]], image: ARRAY) -> ARRAY:
    h_pix, w_pix, _ = image.shape
    landmark_arr = map(lambda x: [x.y, x.x], landmark)
    landmark_arr = V(list(landmark_arr))
    landmark_arr[:, 0] *= h_pix
    landmark_arr[:, 1] *= w_pix
    return landmark_arr


def crop_by_landmarks(landmarks, image, edge_size: Optional[float], last_corners):
    top_offset = .32
    bottom_offset = .1
    lanmark_arr = landmarks_to_arr(landmarks, image)
    h_pix, w_pix, _ = image.shape
    box = lanmark_arr.max(0), lanmark_arr.min(0)
    height_width = box[0] - box[1]
    if edge_size is None:
        offset_h_top = height_width[0] * top_offset
        offset_h_bottom = height_width[0] * bottom_offset
        edge_size = offset_h_top + offset_h_bottom + height_width[0]
    else:
        offset_h_top_and_offset_h_bottom = edge_size - height_width[0]
        offset_h_top = offset_h_top_and_offset_h_bottom * (top_offset / (top_offset + bottom_offset))
    offset_w = (edge_size - height_width[1]) / 2
    corners = [int(box[1][0] - offset_h_top), int(box[1][1] - offset_w)]
    if last_corners is not None:
        if last_corners[0] > corners[0]:
            corners[0] = last_corners[0] - 1
        elif last_corners[0] < corners[0]:
            corners[0] = last_corners[0] + 1
        if last_corners[1] > corners[1]:
            corners[1] = last_corners[1] - 1
        elif last_corners[1] < corners[1]:
            corners[1] = last_corners[1] + 1
    edge_size_int = int(edge_size)
    corners = [max(corners[0], 0), max(corners[1], 0)]
    if corners[0] + edge_size_int > h_pix:
        corners[0] = h_pix - edge_size_int
    if corners[1] + edge_size_int > w_pix:
        corners[1] = w_pix - edge_size_int
    return image[corners[0]: corners[0] + edge_size_int, corners[1]: corners[1] + edge_size_int], corners, edge_size


class PreprocessVideo:

    def iter(self, face_mesh, landmarks):
        image, _ = self.ds[self.counter]
        aligned_image = align_by_lm(landmarks, image, pad_type='constant')
        # if aligned_image is None:
        #     return False
        aligned_image = V(aligned_image)
        results = face_mesh.process(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return False
        files_utils.save_np(aligned_image, f'{self.root}image_{self.counter:08d}')
        landmark_oval = get_oval(aligned_image, results.multi_face_landmarks[0].landmark)
        return landmark_oval
        # mask_a = np.expand_dims(masks[0], 2).astype(np.uint8) * 255
        # mask_a = np.expand_dims(masks[0], 2).repeat(3, axis=2).astype(np.uint8) * 255
        mask_b = np.expand_dims(masks[1], 2).repeat(3, axis=2).astype(np.uint8) * 255
        # mask_a[~masks[0]] = 60
        # masked_a = aligned_image.copy()
        # masked_a[~masks[0]] = 0
        # masked_a = np.concatenate((aligned_image, mask_a), axis=2)

        # masked_b = masked_a.copy()
        # masked_b[masks[1]] = 255
        # masked_b[masks[1], -1] = 255
        # files_utils.save_image(aligned_image, f'{self.root}image_{self.counter:08d}')
        # if self.counter > 300:
        #     files_utils.imshow(masked_a)
        #     files_utils.imshow(masked_b)
        # files_utils.save_image(aligned_image, f'{self.root}image_{self.counter:08d}')
        # files_utils.save_image(mask_a, f'{self.root}maskA_{self.counter:08d}')
        # files_utils.save_image(mask_b, f'{self.root}maskB_{self.counter:08d}')
        # files_utils.save_image(masked_a, f'{self.root}maskedA_{self.counter:08d}')
        # files_utils.save_image(masked_b, f'{self.root}maskedB_{self.counter:08d}')

        # files_utils.save_np(masks, f'{self.root}_mask_{self.counter:08d}')

        # x = aligned_image.copy()
        # x[masks[0] == 0] = 0
        # files_utils.imshow(x)
        # x = aligned_image.copy()
        # x[masks[1] == 0] = 0
        # files_utils.imshow(x)
        return True

    def skip_iter(self, face_mesh):
        self.increase_logger()
        while self.counter < len(self.ds):
            image, _ = self.ds[self.counter]
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                return
            self.increase_logger()

    def increase_logger(self):
        if self.verbose:
            self.logger.reset_iter()
        self.counter += 1

    @property
    def metadata_path(self):
        return f'{self.root}_metadata'

    def get_metadata_path(self):
        metadata = files_utils.load_pickle(self.metadata_path)
        if metadata is None:
            metadata = {}
        return metadata

    def get_lm(self):
        image, _ = self.ds[self.counter]
        lm = get_landmark(image, self.predictor, self.detector)
        return lm

    def get_all_lm(self):
        lm_path = f'{self.root}landmarks'
        all_landmarks_data = files_utils.load_pickle(lm_path)
        if all_landmarks_data is None:
            self.counter = 0
            if self.verbose:
                self.logger.start(len(self.ds))
            is_face = np.zeros(len(self.ds), dtype=np.bool_)
            mp_face_mesh = mp.solutions.face_mesh
            all_landmarks = np.zeros((len(self.ds), 68, 2))
            with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5) as face_mesh:
                while self.counter < len(self.ds):
                    lm = self.get_lm()
                    is_face[self.counter] = lm is not None
                    if is_face[self.counter]:
                    #     continue
                    #     # self.skip_iter(face_mesh)
                    # else:
                        all_landmarks[self.counter] = lm
                    self.increase_logger()
            # if is_face.sum() >= (len(self.ds) * .75):
            files_utils.save_pickle({'is_face': is_face, 'landmarks': all_landmarks}, lm_path)
            if self.verbose:
                self.logger.stop()
        else:
            is_face, all_landmarks = all_landmarks_data['is_face'], all_landmarks_data['landmarks']
        return is_face, all_landmarks

    def process_audio(self, audio_path=None):
        # metadata = self.get_metadata_path()
        if audio_path is None:
            audio_path = f'{self.root}audio.wav'
            mfcc_path = f'{self.root}mfcc'
        else:
            mfcc_path = audio_path.replace('.wav', '_mfcc')
        if not files_utils.is_file(audio_path):
            files_utils.init_folders(audio_path)
            in1 = ffmpeg.input(self.video_path)
            a1 = in1.audio
            out = ffmpeg.output(a1, audio_path)
            out.run()
        sample_rate, audio = wavfile.read(audio_path)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc], axis=1)
        files_utils.save_np(mfcc, mfcc_path)
        return
        # metadata['sample_rate'] = sample_rate

    # 44100
    @staticmethod
    def blur_seq(seq, max_iters=3):
        seq = torch.from_numpy(seq)
        diffs = (seq[1:] - seq[:-1]).abs().sum(-1).sum(-1)
        diffs = diffs / (seq.shape[1] * seq.shape[2])
        th = diffs.mean() + diffs.std()
        check = 0
        while (diffs > th).sum() > 0 and check < max_iters:
            seq_new = seq.clone()
            last_valid = 0
            is_valid = True
            for i in range(1, len(diffs)):
                if diffs[i - 1] > th:
                    is_valid = False
                else:
                    if not is_valid:
                        seq_new[last_valid + 1: i] = (seq[last_valid] + seq[i]) / 2
                    is_valid = True
                    last_valid = i
            seq = seq_new
            diffs = (seq[1:] - seq[:-1]).abs().sum(-1).sum(-1)
            diffs = diffs / (seq.shape[1] * seq.shape[2])
            check += 1
        seq = seq.numpy()
        out = gaussian_filter1d(seq, 5, axis=0)
        return out

    def blur_landmarks(self, is_face, all_landmarks):
        end = start = 0
        changed = False
        all_landmarks = self.pad_landmarks(is_face, all_landmarks)
        all_landmarks = self.blur_seq(all_landmarks)
        # for i in range(is_face.shape[0]):
        #     if is_face[i]:
        #         end = i + 1
        #         changed = True
        #     else:
        #         if changed:
        #             all_landmarks[start: end] = self.blur_seq(all_landmarks[start: end])
        #             changed = False
        #         start = i + 1
        # if changed:
        #     all_landmarks[start: end] = self.blur_seq(all_landmarks[start: end])
        return all_landmarks

    @staticmethod
    def pad_landmarks(is_face, all_landmarks):
        changed = False
        last_lm = None
        stash = []
        padedd_landmarks = all_landmarks.copy()
        for i in range(is_face.shape[0]):
            if is_face[i]:
                if changed:
                    pad_lam = all_landmarks[i]
                    if last_lm is not None:
                        pad_lam = (pad_lam + all_landmarks[last_lm]) / 2
                    for j in stash:
                        padedd_landmarks[j] = pad_lam
                    stash = []
                    changed = False
                last_lm = i
            else:
                changed = True
                stash.append(i)
        if changed:
            for j in stash:
                padedd_landmarks[j] = all_landmarks[last_lm]
        return padedd_landmarks

    def process_images(self):
        is_face, all_landmarks = self.get_all_lm()
        # print(is_face.sum())
        # if is_face.sum() < (len(self.ds) * .75):
        #     return False
        # all_landmarks = self.blur_landmarks(is_face, all_landmarks)
        self.counter = 0
        if self.verbose:
            self.logger.start(len(self.ds))
        metadata = self.ds.get_metadata()
        mp_face_mesh = mp.solutions.face_mesh
        all_images = []
        all_landmark_oval = np.zeros((len(self.ds), 36, 2))
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            for i in range(is_face.shape[0]):
                if is_face[i]:
                    oval = self.iter(face_mesh, all_landmarks[i])
                    all_landmark_oval[i] = oval
                # if image is not None:
                #     all_images.append(image)
                self.increase_logger()
                # if len(all_images) >= 500:
                #     break
        files_utils.save_np(all_landmark_oval, f'{self.root}face_contours')
        # image_utils.gif_group(all_images, f'{constants.CACHE_ROOT}/tmp/seq', 29.7, True)
        files_utils.save_pickle(metadata, f'{self.root}metadata')
        if self.verbose:
            self.logger.stop()
        return True

    def __init__(self, video_file: str, name, verbose: bool = True):
        self.video_path = video_file
        self.predictor = dlib.shape_predictor("./weights/ffhq/shape_predictor_68_face_landmarks.dat")
        self.detector = dlib.get_frontal_face_detector()
        self.counter = 0
        self.is_skipped = False
        self.root = f'{constants.CACHE_ROOT}/{name}/'
        self.logger = train_utils.Logger()
        self.ds = VidDs(self.video_path, False)
        self.verbose = verbose


def from_video(video_file: str):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    ds = VidDs(video_file, False)
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    # last_corners = edge_size = None
    files_utils.init_folders('./assets/tmp/')
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for idx in range(370, 372, 1):
            image, _ = ds[idx]
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue
            # image, last_corners, edge_size = crop_by_landmarks(results.multi_face_landmarks[0].landmark, image, edge_size, last_corners)
            # files_utils.imshow(image)
            # return
            mask = get_poly(image, results.multi_face_landmarks[0].landmark)
            return
            # files_utils.imshow(mask)
            # return
            # files_utils.save_image(V(image), f'./assets/tmp/obama_crop_{idx:04d}.png')
            # for face_landmarks in results.multi_face_landmarks:
            #     print('face_landmarks:', face_landmarks)
            #     mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_tesselation_style())
            #     mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            #     mp_drawing.draw_landmarks(
            #     image=annotated_image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_iris_connections_style())


def crop_video(name_in, name_out):
    ds = VidDs(name_in, False)
    images = []
    for i in range(len(ds)):
        frame, _ = ds[i]
        images.append(frame[200: 200 + 660, 320:-100])
    image_utils.gif_group(images, name_out, 29.7, True)


def make_lots():
    num_collect = 1000
    all_mp4 = files_utils.collect('/home/ahertz/Downloads/mvlrs_v1/main/', '.mp4')
    visited = set()
    for path in all_mp4:
        vid_dir = path[0].split('/')[-2]
        vid_name = f"{vid_dir}_{path[1]}"
        if os.path.isdir(f'{constants.LRS2_PROCESS}/{vid_name}'):
            num_collect -= 1
            visited.add(vid_dir)
            continue
        if vid_dir in visited:
            continue
        else:
            vid = PreprocessVideo(''.join(path),
                                  vid_name, verbose=False)
            success = vid.process_images()
            if success:
                visited.add(vid_dir)
                # vid.process_audio()
                num_collect -= 1
                print(num_collect)
        if num_collect == 0:
            break



def main_single():
    predictor = dlib.shape_predictor("./weights/ffhq/shape_predictor_68_face_landmarks.dat")
    detector = dlib.get_frontal_face_detector()
    landmarks = None
    all_landmark = files_utils.load_np(f'./assets/exp_mid/landmarks')
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        for i in range(7):
            image = files_utils.load_image(f'./assets/exp_mid/{i:02d}')
            # if landmarks is None:
            landmarks = get_landmark(image, predictor, detector)
            aligned_image = align_by_lm(landmarks, image, pad_type='constant')
            # aligned_image = files_utils.load_np( f'./assets/exp_mid/aligned_{i:02d}')
            aligned_image = V(aligned_image)
            files_utils.imshow(aligned_image)
            results = face_mesh.process(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
            landmarks_face = results.multi_face_landmarks[0].landmark
            # all_landmark
            landmark_arr = landmarks_to_arr(landmarks_face, aligned_image)
            # landmark_arr = all_landmark[i]
            mask = get_lips(aligned_image, landmark_arr).astype(np.bool_)
            # aligned_image[~mask] = (aligned_image[~mask] + 100)
            # aligned_image = np.clip(aligned_image, 0, 255)
            # files_utils.imshow(aligned_image)
            # files_utils.save_np(aligned_image, f'./assets/exp_mid/aligned_{i:02d}')
            files_utils.save_image(aligned_image, f'./assets/exp_mid/aligned_{i:02d}')
            all_landmark[i] = landmark_arr
        # files_utils.save_np(all_landmark, f'./assets/exp_mid/landmarks')




def main():

    # crop_video('./assets/raw_videos/me_lyps.MOV', './assets/raw_videos/me_lips_crop')
    for i in range(7):
        vid = PreprocessVideo(f'{constants.DATA_ROOT}/raw_videos/exp_{i:02d}.mp4', f'exp_{i:02d}')
        vid.process_images()
    # from_video('./assets/raw_videos/obama_a.mp4')


if __name__ == '__main__':
    main_single()
