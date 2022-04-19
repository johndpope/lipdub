from custom_types import *
from scipy.io import wavfile
# import python_speech_features
import imageio
import ffmpeg


def audio2features(path: str):
    # sample_rate, audio =
    pass


def split_video(video_path: str):
    # vid = imageio.get_reader(video_path, 'ffmpeg')
    in1 = ffmpeg.input(video_path)
    a1 = in1.audio
    out = ffmpeg.output(a1, video_path.replace('.mp4', '.wav'))
    # v2 = in2.video.filter('reverse').filter('hue', s=0)
    # a2 = in2.audio.filter('areverse').filter('aphaser')
    # joined = ffmpeg.concat(v1, a1, v2, a2, v=1, a=1).node
    # v3 = joined[0]
    # a3 = joined[1].filter('volume', 0.8)
    # out = ffmpeg.output(v3, a3, 'out.mp4')
    out.run()
    pass


def main():
    split_video('../assets/raw_videos/obama_a.mp4')
    pass

if __name__ == '__main__':
    main()