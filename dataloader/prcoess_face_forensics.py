from custom_types import *
import constants
from utils import files_utils, train_utils
from align_faces import FaceAlign
import imageio


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
        frames = get_frames_numbers(video_path)
        vid = imageio.get_reader("".join(video_path), 'ffmpeg')
        counter = 0
        for frame_number in frames:
            frame_number = frame_number.item()
            self.start_item(video_path, frame_number)
            frame = vid.get_data(frame_number)
            counter += int(self.process_frame(frame))
            if counter == self.max_frames_per_file:
                break

    def save_metadata(self):
        metadata = {"info": self.metadata["info"],
                    "quads:": np.stack(self.metadata["quads"]),
                    "landmarks": np.stack(self.metadata["landmarks"]),
                    "landmarks_crops": np.stack(self.metadata["landmarks_crops"])}
        files_utils.save_pickle(metadata, f"{self.out_root}/metadata")

    def run(self):
        files = files_utils.collect(constants.FaceForensicsRoot + 'downloaded_videos/', '.mp4')
        self.logger.start(len(files) * self.max_frames_per_file)
        for video_path in files:
            self.process_sequence(video_path)
        self.logger.stop()

    def __init__(self):
        self.max_frames_per_file = 40
        self.aligner = FaceAlign()
        self.metadata = {"info": [], "quads": [], "landmarks": [], "landmarks_crops": []}
        self.total_counter = 0
        self.cur_item = {}
        self.out_root = constants.FaceForensicsRoot + 'processed_frames/'
        self.logger = train_utils.Logger()


def main():
    ProcessFaceForensicsRoot().run()


if __name__ == '__main__':
    main()
