from p2fa import align
from custom_types import *
import imageio
from utils import files_utils, image_utils, train_utils
import constants
import ffmpeg
from scipy.io import wavfile


# viseme2phonemes = {0: ('sp', ), 1: ('ay', 'ah'), 2: ('ey', 'eh', 'ae'),
#                   3: ('er', ), 4: ('ix', 'iy', 'ih', 'ax', 'axr', 'y'), 5: ('uw', 'uh', 'w'),
#                   6: ('ao', 'aa', 'oy', 'ow', 'aw'), 7: ('aw'), 8: ('g', 'hh', 'k', 'ng'),
#                   9: ('r'), 10: ('l', 'd', 'n', 'en', 'el', 't'), 11: ('s', 'z'),
#                   12: ('ch', 'sh', 'jh', 'zh'), 13: ('th', 'dh'), 14: ('f', 'v'),
#                   15: ('m', 'em', 'b', 'p')
#                   }



viseme2phonemes = {0: ('aa',), 1:('ah', 'ax'  'axr'), 2:('ao',), 3: ('ow', 'aw'),
                   4: ('uw', 'uh', 'oy'), 5: ('eh', 'ae'), 6: ('ay', 'ih'), 7: ('ey',),
                   8:('ix', 'iy', 'y'), 9:('er', 'r'), 10: ('l', 'el', 'en'), 11: ('w',),
                   12: ('m', 'em', 'b', 'p'), 13: ('g', 'hh', 'k', 'ng', 'd', 'n', 't', 's', 'z', 'th', 'dh'),
                   14:('ch', 'sh', 'jh', 'zh'), 15: ('f', 'v'),  16: ('sp',),
                  }


phoneme2viseme = {}
for viseme, phonemes in viseme2phonemes.items():
    for phoneme in phonemes:
        phoneme2viseme[phoneme] = viseme


def split_text(file, out):
    text = files_utils.load_txt(file)
    words = []
    for line in text:
        words += line.split()
    files_utils.save_txt(words, out)


def split_video(root, name):
    vid = imageio.get_reader(f'{root}/{name}.mp4', 'ffmpeg')
    fps = vid._meta['fps']
    texts = files_utils.collect(root, ".txt")
    for path in texts:
        start, end = path[1].split('_')[-2:]
        start, end = int(start), int(end)
        first_frame = int(start * fps)
        last_frame = int(end * fps)
        frames = [vid.get_data(i) for i in range(first_frame, last_frame)]
        image_utils.gif_group(frames, f"{path[0]}/{path[1]}", fps)


def split_audio(video_path, audio_path):
    # audio_path = f'{root}/{name}.wav'
    if not files_utils.is_file(audio_path + '.wav'):
        in1 = ffmpeg.input(f'{video_path}')
        a1 = in1.audio
        out = ffmpeg.output(a1, audio_path + '.wav')
        out.run()
    # vid = imageio.get_reader(f'{root}/{name}.mp4', 'ffmpeg')
    # fps = vid._meta['fps']
    # sample_rate, audio = wavfile.read(audio_path)
    # texts = files_utils.collect(root, ".txt")
    # for path in texts:
    #     start, end = path[1].split('_')[-2:]
    #     start, end = int(start), int(end)
    #     first_frame = int(start * sample_rate)
    #     last_frame = int(end * sample_rate)
    #     frames = audio[first_frame:last_frame]
    #     files_utils.save_wav(frames, sample_rate, f"{path[0]}/{path[1]}")


def translate_phonemes(phonemes_list: List):
    out = []
    for row in phonemes_list:
        phoneme, start, end = row
        phoneme = phoneme.lower()
        while (ord(phoneme[-1]) >= 48 and ord(phoneme[-1]) <= 57):
            phoneme = phoneme[:-1]
        if phoneme not in phoneme2viseme:
            print(f"error: {phoneme}")
        else:
            out.append([phoneme2viseme[phoneme], start, end ])
    return out


def viseme2vec(video_path, viseme_path, out_path):
    # path = files_utils.collect(root, ".txt")[0]
    # viseme_path = f"{path[0]}/viseme_{path[1]}"
    max_len = 30

    viseme = files_utils.load_pickle(viseme_path)['viseme']
    vid = imageio.get_reader(f"{video_path}", 'ffmpeg')
    fps = vid._meta['fps']
    num_frames = min(vid.count_frames(), int(fps * max_len))
    vis_vec = torch.zeros(num_frames, len(viseme2phonemes))
    cur_viseme_idx = 0
    cur_viseme = viseme[cur_viseme_idx]

    is_viseme = torch.zeros(num_frames).bool()
    frame_time = torch.arange(num_frames).float() / float(fps)

    for i, viseme_ in enumerate(viseme):
        timing = (viseme_[1] + viseme_[2]) / 2.
        dist = (frame_time - timing).abs()
        closest = dist.argmin()
        is_viseme[closest] = True
    for i in range(len(is_viseme)):
        if is_viseme[i]:
            break
        is_viseme[i] = True

    for i in range(len(is_viseme) - 1, -1, -1):
        if is_viseme[i]:
            break
        is_viseme[i] = True

    for i in range(num_frames):
        cur_time = i / float(fps)
        if cur_time > cur_viseme[2]:
            cur_viseme_idx += 1
            cur_viseme = viseme[cur_viseme_idx]
        cur_viseme_mid = (cur_viseme[1] + cur_viseme[2]) / 2.
        if cur_time < (cur_viseme[1] + cur_viseme[2]) / 2.:
            if cur_viseme_idx == 0:
                vis_vec[i, cur_viseme[0]] = 1.
            else:
                prev_viseme = viseme[cur_viseme_idx - 1]
                prev_viseme_mid = (prev_viseme[1] + prev_viseme[2]) / 2.
                total_time = cur_viseme_mid - prev_viseme_mid
                alpha = (cur_time - prev_viseme_mid) / total_time
                beta = (cur_viseme_mid - cur_time) / total_time
                vis_vec[i, cur_viseme[0]] += alpha
                vis_vec[i, prev_viseme[0]] += beta
        else:
            if cur_viseme_idx >= len(viseme) - 1:
                vis_vec[i, cur_viseme[0]] += 1
            else:
                next_viseme = viseme[cur_viseme_idx + 1]
                next_viseme_mid = (next_viseme[1] + next_viseme[2]) / 2.
                total_time = next_viseme_mid - cur_viseme_mid
                alpha = (next_viseme_mid - cur_time) / total_time
                beta = (cur_time - cur_viseme_mid) / total_time
                vis_vec[i, cur_viseme[0]] += alpha
                vis_vec[i, next_viseme[0]] += beta

    files_utils.save_pickle({'is_center': is_viseme, 'vis_vec': vis_vec}, out_path)
    # for viseme_ in viseme:
    #     if viseme_[0] != 0:
    #         viseme_frame = .5 * fps * (viseme_[1] + viseme_[2])
    #         frame = vid.get_data(int(viseme_frame))
    #         files_utils.imshow(frame)

    return


def align_visemes(database, target):
    source_visemes = files_utils.load_pickle(database)['vis_vec']
    target = files_utils.load_pickle(target)['vis_vec']
    diff = target[:, None] - source_visemes[None]
    diff = (diff ** 2).sum(-1)
    select = diff.argmin(1)
    return select


def sub_audio(root, vid, start_frame: int, end_frame: int, fps):
    sample_rate, audio = files_utils.load_wav(root)
    vid = imageio.get_reader(f"{vid}.mp4", 'ffmpeg')
    fps_base = vid._meta['fps']
    total_frames = vid.count_frames()
    total_audio = audio.shape[0]

    start_sec = float(start_frame) / fps_base
    end_sec = float(end_frame) / fps_base

    start = start_sec * sample_rate
    end = end_sec * sample_rate
    sub_audio = audio[int(start): int(end)]
    sample_rate = int(sample_rate * float(fps) / fps_base)
    files_utils.save_wav(sub_audio, sample_rate, f'{root}_{start_frame}_{end_frame}_{fps}')


    # vid = imageio.get_reader(f"{root}.mp4", 'ffmpeg')
    return


def p2fa_file(text_file):
    wav_path = f'{text_file}.wav'
    if files_utils.is_file(wav_path):
        phoneme_list, word_list, _ = align.align(wav_path, f'{text_file}.txt')
        viseme_list = translate_phonemes(phoneme_list)
        files_utils.save_pickle({'phoneme': phoneme_list,
                                 'viseme': viseme_list,
                                 'word': word_list}, f'{text_file}_viseme')


def p2fa_folder(root):
    texts = files_utils.collect(root, ".txt")
    phonemes_set = set()
    total = 0
    for path in texts:
        wav_path = f'{path[0]}{path[1]}.wav'
        if files_utils.is_file(wav_path):
            phoneme_list, word_list, _ = align.align(wav_path, f'{path[0]}{path[1]}.txt')
            viseme_list = translate_phonemes(phoneme_list)
            for ph in phoneme_list:
                phonemes_set.add(ph[0])
            total += len(phoneme_list[0])
            files_utils.save_pickle({'phoneme': phoneme_list,
                                     'viseme': viseme_list,
                                     'word': word_list}, f'{path[0]}viseme_{path[1]}')


def make_histogram(lst):
    histogram = {}
    for item in lst:
        if item[0] not in histogram:
            histogram[item[0]] = [0, []]
        histogram[item[0]][0] += 1
        histogram[item[0]][1].append(item)
    out = list(histogram.items())
    out = sorted(out, key=lambda x: -x[1][0])
    return out


def plot_word(word_info, frames, fps):
    frame_start = word_info[1] * fps
    frame_end = word_info[2] * fps
    if frame_end > len(frames):
        return None
    images = [files_utils.load_np(''.join(path)) for path in frames[int(frame_start): int(frame_end)]]
    image = image_utils.make_row(images, 5)
    return image


def plot_viseme(visemes_arr, frames, fps, max_viseme = 5):
    images = []
    for i in range(min(len(visemes_arr), max_viseme)):
        viseme_info = visemes_arr[i]
        frame_start = viseme_info[1] * fps
        frame_end = viseme_info[2] * fps
        if frame_end > len(frames):
            continue
        frame = frames[int(frame_start + frame_end) // 2]
        images.append(files_utils.load_np(''.join(frame)))
    if len(images) < max_viseme:
        return None
    image = image_utils.make_row(images, 5)
    return image

def np_video(viseme_root, out_folder):
    vid = imageio.get_reader(viseme_root, 'ffmpeg')
    for i in range(vid.count_frames()):
        image = vid.get_data(i)[:, (1920 - 1080) // 2: -(1920 - 1080) // 2]
        files_utils.save_np(image, f'{out_folder}/image_{i:04d}')


def look_on_viseme(viseme_root, frames_root):
    data = files_utils.load_pickle(f'{viseme_root}_viseme')
    vid = imageio.get_reader(f"{viseme_root}.mp4", 'ffmpeg')
    fps = vid._meta['fps']
    frames = files_utils.collect(frames_root, '.npy', prefix='crop')
    visemes = data['phoneme']
    visemes_histogram = make_histogram(visemes)
    for i in range(len(visemes_histogram)):
        viseme = visemes_histogram[i][0]
        visemes_arr = visemes_histogram[i][1][1]
        image = plot_viseme(visemes_arr, frames, fps, max_viseme=3)
        if image is not None:
            files_utils.save_image(image, f'{constants.DATA_ROOT}/cache/p2fa_check/phoneme_{viseme}')


def look_on_words(viseme_root, frames_root):
    data = files_utils.load_pickle(f'{viseme_root}_viseme')
    vid = imageio.get_reader(f"{viseme_root}.mp4", 'ffmpeg')
    fps = vid._meta['fps']
    frames = files_utils.collect(frames_root, '.npy', prefix='crop')
    words = data['word']
    word_histogram = make_histogram(words)
    word_index = 6
    for j in range(min(5, len(word_histogram[word_index][1][1]))):
        word = word_histogram[word_index][0]
        image = plot_word(word_histogram[word_index][1][1][j], frames, fps)
        if image is not None:
            files_utils.save_image(image, f'{constants.DATA_ROOT}/cache/p2fa_check/{word}_{j:02d}')

    return


def label_frames(root, viseme_path):
    all_frames = files_utils.collect(root, '.npy')
    all_frames = [file for file in all_frames if 'image_' in file[1]]
    viseme = files_utils.load_pickle(root + viseme_path)
    for i in range(len(all_frames)):
        cur_time = i / 24.
        for j in range(len(viseme['viseme'])):
            if viseme['viseme'][j][2] > cur_time:
                for k in range(len(viseme['word'])):
                    if viseme['word'][k][2] > cur_time:
                        selected_word = viseme['word'][k][0].lower()
                        break
                name = f'{i:03d}_{selected_word}_{viseme["phoneme"][j][0]}_{viseme["viseme"][j][0]:02d}'
                files_utils.save_image(files_utils.load_np(''.join(all_frames[i])), f'{root}/{name}')
                break


if __name__ == '__main__':
    viseme2vec(f'{constants.DATA_ROOT}/raw_videos/101_purpledino_front_comp_v019.mov',
               f'{constants.DATA_ROOT}/raw_videos/101_purpledino_front_comp_v019_viseme.pkl',
               f'{constants.DATA_ROOT}/raw_videos/101_purpledino_front_comp_v019_viseme_vec.pkl')
    # label_frames(f'{constants.DATA_ROOT}/101_beigefox_front_comp_v017/', '101_beigefox_front_comp_v017_viseme')
    # np_video(f'{constants.DATA_ROOT}/raw_videos/101_beigefox_front_comp_v017.mov',
    #          f'{constants.DATA_ROOT}/101_beigefox_front_comp_v017/')
    # p2fa_file(f'{constants.DATA_ROOT}/raw_videos/101_beigefox_front_comp_v017')
    # split_audio(f'{constants.DATA_ROOT}/raw_videos/101_beigefox_front_comp_v017.mov',
    #             f'{constants.DATA_ROOT}/raw_videos/101_beigefox_front_comp_v017')
    # look_on_viseme(f"{constants.DATA_ROOT}/raw_videos/obama_062814", f"{constants.DATA_ROOT}/processed")
