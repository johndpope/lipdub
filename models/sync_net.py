###  from https://github.com/Rudrabha/Wav2Lip
import constants
from custom_types import *
import librosa
from scipy import signal


def _amp_to_db(x):
    min_level = np.exp(-100 / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _linear_to_mel(spectogram):
    _mel_basis = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    return np.dot(_mel_basis, spectogram)


def _normalize(S):
    return np.clip((2 * 4.) * ((S + 100.) / 100.) - 4.,
                   -4., 4.)


def melspectrogram(wav):
    D = signal.lfilter([1, -0.97], [1], wav)
    D = librosa.stft(y=D, n_fft=800, hop_length=200, win_length=800)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - 20
    return _normalize(S)


def process_audio(wav_path, fps):
    mel_step_size = 16
    wav = librosa.core.load(wav_path, sr=16000)[0]
    mel = melspectrogram(wav)
    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    mel_chunks = np.stack(mel_chunks).astype(np.float32)
    mel_chunks = np.expand_dims(mel_chunks, axis=1)
    return mel_chunks


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class SyncNet(nn.Module):
    def __init__(self):
        super(SyncNet, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        state_dict = torch.load(f'{constants.PROJECT_ROOT}/weights/lipsync_expert.pth', map_location=CPU)['state_dict']
        self.load_state_dict(state_dict)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = nnf.normalize(audio_embedding, p=2, dim=1)
        face_embedding = nnf.normalize(face_embedding, p=2, dim=1)
        return audio_embedding, face_embedding


def main():
    net = SyncNet()
    image = torch.rand(5, 15, 96 // 2, 96)
    mel_chunks = process_audio("/mnt/ml/projects/dubbing/processed_infer/BourdainT/audio.wav", 30)
    # mel_input = torch.rand(5, 1, 80, 16)
    mel_input = mel_chunks[:5]
    mel_input = np.stack(mel_input, axis=0)
    mel_input = torch.from_numpy(mel_input).float().unsqueeze(-1).permute(0, 3, 1, 2)
    out = net(mel_input, image)
    print(out[0].shape)
    print(out[1].shape)
    pass



if __name__ == '__main__':
    main()