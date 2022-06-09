from models import stylegan_wrapper, sync_net
from dataloader import prcoess_face_forensics
from custom_types import *
from utils import train_utils, files_utils, image_utils
import constants
import options



class Trainer:

    def get_sync_loss(self, images, audio, key: Optional[str] = None):
        images = (images + 1) / 2
        images = nnf.interpolate(images, 96, mode='bilinear')[:, :, 96 // 2:, :]
        images = images.view(-1, 15, 96 // 2, 96)
        audio_embedding, face_embedding = self.sync_net(audio, images)
        loss = (1 - torch.einsum('bd,bd->b', audio_embedding, face_embedding)).mean()
        if key is not None:
            self.logger.stash_iter(key, loss)
        return loss


    def finalize_video(self, sequence, driven_dir, out_path):
        from moviepy import editor
        metadata = files_utils.load_pickle(f"{driven_dir}/metadata")
        sample_rate, audio = files_utils.load_wav(f"{driven_dir}/audio")
        audio_out_len = float(audio.shape[0] * len(sequence)) / metadata['frames_count']
        files_utils.save_wav(audio[:int(audio_out_len)], sample_rate, f'{out_path}.wav')
        image_utils.gif_group(sequence, f'{out_path}_tmp', metadata['fps'])
        video_clip = editor.VideoFileClip(f'{out_path}_tmp.mp4')
        audio_clip = editor.AudioFileClip(f'{out_path}.wav')
        audio_clip = editor.CompositeAudioClip([audio_clip])
        video_clip.audio = audio_clip
        video_clip.write_videofile(f'{out_path}.mp4')
        video_clip.close()
        files_utils.delete_single(f'{out_path}_tmp.mp4')
        files_utils.delete_single(f'{out_path}.wav')

    def train(self, base_folder: str, driving_folder: str, name, starting_time: int = -1, ending_time: int = -1,
                    num_epochs: int = 100):
        batch_size = 20
        infer_ds = prcoess_face_forensics.LipsSeqDSInfer(prcoess_face_forensics.LipsSeqDS(options.OptionsLipsGeneratorSeq()), base_folder, driving_folder, starting_time,
                                                         ending_time)
        dataloader = DataLoader(infer_ds, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0 if DEBUG else 8)
        w = nn.Embedding(len(infer_ds) + 4, 512).to(self.device)
        with torch.no_grad():
            latent = self.latent_avg.unsqueeze(0).repeat(len(infer_ds) + 4, 1)
            w.weight.data = latent
        optimizer = Optimizer(w.parameters(), lr=1e-4)
        for epoch in range(num_epochs):
            last_seq = 0
            self.logger.start(len(dataloader))
            for i, data in enumerate(dataloader):
                audio = data[-1].to(self.device)
                b = audio.shape[0]
                select = torch.arange(b + 4).to(self.device) + last_seq
                w_input = w(select)
                w_input = w_input.unsqueeze(1).expand(w_input.shape[0], 18, 512)
                out = self.model(w_input)
                select_seq = torch.arange(5).to(self.device)
                prefix = torch.arange(b).to(self.device)
                select_seq = select_seq[None, :] + prefix[:, None]
                out = out[select_seq]
                out = out.reshape(b * 5, 3, 1024, 1024)
                loss = self.get_sync_loss(out, audio, "sync")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.logger.reset_iter()
                last_seq += b
            self.logger.stop()
        files_utils.save_model(w, f"{constants.CHECKPOINTS_ROOT}/w_sync/{name}")


    @property
    def device(self):
        return CUDA(0)

    @property
    def sync_net(self):
        if self.sync_net_ is None:
            self.sync_net_ = sync_net.SyncNet().eval().to(self.device)
        return self.sync_net_


    @torch.no_grad()
    def export_video(self, name, driven_dir):
        stat_dict = torch.load(f"{constants.CHECKPOINTS_ROOT}/w_sync/{name}")
        w = stat_dict['weight']
        b = 20
        num_iters = w.shape[0] // b + int(w.shape[0] % b != 0)
        seq = []
        self.logger.start(num_iters)
        for i in range(num_iters):
            if i == num_iters - 1:
                w_input = w[i * b:]
            else:
                w_input = w[i * b:i * b + b]
            w_input = w_input.to(self.device)
            w_input = w_input.unsqueeze(1).expand(w_input.shape[0], 18, 512)
            out = self.model(w_input)
            out = nnf.interpolate(out, 256, mode='bicubic', align_corners=True)
            out = [files_utils.image_to_display(img) for img in out]
            seq += out
            self.logger.reset_iter()
        self.logger.stop()
        self.finalize_video(seq, driven_dir, f"{constants.CHECKPOINTS_ROOT}/w_sync/{name}")




    def __init__(self):
        self.model = stylegan_wrapper.get_model().eval()
        self.sync_net_: Optional[sync_net.SyncNet] = None
        self.logger = train_utils.Logger()
        ckpt = torch.load(f'{constants.PROJECT_ROOT}/weights/ffhq/e4e_ffhq_encode.pt')
        self.latent_avg = ckpt['latent_avg'][0].to(self.device)


def main():
    info_dict = {'Eilish': {'name_base': 'Eilish', 'name_driving': 'Eilish_French052522', 'tag': 'Eilish'},
                 'Johnson': {'name_base': 'Johnson', 'name_driving': 'Dwayne_Spanish_FaceFormer', 'tag': 'Johnson'},
                 'Bourdain': {'name_base': 'BourdainT', 'name_driving': 'Bourdain_Italian_faceformer',
                              'tag': 'Bourdain'}}

    name = "Eilish"
    info = info_dict[name]
    # Trainer().export_video(name, f'{constants.MNT_ROOT}/processed_infer/{info["name_driving"]}/')
    Trainer().train(f'{constants.MNT_ROOT}/processed_infer/{info["name_base"]}/',
                      f'{constants.MNT_ROOT}/processed_infer/{info["name_driving"]}/', name,
                      starting_time=10, ending_time=-1, num_epochs=100)




if __name__ == '__main__':
    main()
