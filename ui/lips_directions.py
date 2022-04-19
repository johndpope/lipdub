import numpy as np

from models import models_utils
from ui import ui_utils
from custom_types import *
from models import stylegan_wrapper
import constants
from utils import files_utils, image_utils
import vtkmodules.vtkInteractionStyle
import vtkmodules.vtkRenderingOpenGL2
import vtk
import vtk.util.numpy_support as numpy_support
from vtkmodules.vtkImagingCore import vtkImageCast
from vtkmodules.vtkRenderingCore import (
    vtkImageActor,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)


def show_something(idx, w_plus, dirs, stylegan):
    with torch.no_grad():
        num_between = 3
        base_vec = w_plus[-1][idx]
        alpha = torch.linspace(0, 3., num_between)
        w = base_vec[None, :, :] + dirs[None, :, :] * alpha[:, None, None]
        out = stylegan(w.cuda())
        out_low = nnf.interpolate(out, size=(128, 128))
        out_low = torch.cat([out_low[i] for i in range(num_between)], dim=2)
        files_utils.imshow(out_low)
    return

@models_utils.torch_no_grad
def main():
    from moviepy import editor
    stylegan = stylegan_wrapper.get_model()
    valid = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_valid')
    w_plus = [files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}')[valid] for i in range(17)]
    dirs = [w_plus[i] - w_plus[-1] for i in range(16)]
    dirs = [item.mean(0) for item in dirs]
    dirs = torch.stack(dirs, dim=0)
    viseme_vec = files_utils.load_pickle(f'{constants.DATA_ROOT}/raw_videos/101_purpledino_front_comp_v019_viseme_vec.pkl')['vis_vec']
    viseme_vec = viseme_vec[:, :-1]
    w_base = w_plus[-1][12]
    dirs_vec = torch.einsum('nwd,fn->fwd', dirs, viseme_vec) * 2
    w_frames = w_base.unsqueeze(0).repeat(dirs_vec.shape[0], 1, 1)
    w_frames[:, :8] += dirs_vec[:, :8]
    frames = []
    for i in range(len(w_frames)):
        out = stylegan(w_frames[i].unsqueeze(0).cuda())
        out = nnf.interpolate(out, (256, 256)).clip(-1, 1)
        out = files_utils.image_to_display(out)
        frames.append(out)
    image_utils.gif_group(frames, f'{constants.CACHE_ROOT}/syn_video/tmp', 24)
    video_clip = editor.VideoFileClip(f'{constants.CACHE_ROOT}/syn_video/tmp.mp4')
    audio_clip = editor.AudioFileClip(f'{constants.DATA_ROOT}/raw_videos/101_purpledino_front_comp_v019.wav')
    audio_clip = editor.CompositeAudioClip([audio_clip])
    video_clip.audio = audio_clip
    video_clip.write_videofile(f'{constants.CACHE_ROOT}/syn_video/{12}_101_purpledino.mp4')
    video_clip.close()
    files_utils.delete_single(f'{constants.CACHE_ROOT}/syn_video/tmp.mp4')


class InteractorStyle(vtk.vtkInteractorStyleTrackballCamera):


    def replace_image(self, w):

        with torch.no_grad():
            if w.dim() < 3:
                w = w.unsqueeze(0)
            # x = torch.randn(1, 512).cuda()
            # w = self.model.generator.style(x).unsqueeze(1).repeat(1, 18, 1)
            out = self.model(w.cuda())
            out = nnf.interpolate(out, size=(512, 512))
            out = out[0].detach().cpu().permute(1, 2, 0)
            out = ((out + 1) * 127.5).clamp(0, 255)
        out = out.numpy().astype(np.uint8)
        out = np.flipud(out)
        out = out.reshape(out.shape[0] * out.shape[1], 3)
        out = numpy_support.numpy_to_vtk(out)

        # depthImageData.SetDimensions((512, 512, 1))
        # assume 0,0 origin and 1,1 spacing.
        # depthImageData.SetSpacing(1, 1, 1)
        # depthImageData.SetOrigin(0, 0, 0)
        self.image_data.GetPointData().SetScalars(out)

    def update(self):
        delta = 3 * self.alphas[:, None, None] * self.dirs[:16]
        # w_new = self.w_cur.clone() + delta.sum(0)
        w_new = self.w_base[self.image_id].clone()
        # w_new[8:] = self.w_cur[8:]
        w_new[:10] += delta.sum(0)[:10]

        w_new = w_new + self.w_pose * self.pose_value
        self.replace_image(w_new)

    def on_key_press(self, obj, event):
        key = self.interactor.GetKeySym().lower()
        if key == 'return' or key == 'kp_enter':
            # self.w_cur = self.get_rand_w()
            self.image_id = torch.randint(199, (1,)).item()
            # self.w_cur = self.get_rand_w()
        self.update()

    def slider_event(self, slider, event):
        value = slider.GetRepresentation().GetValue()

        self.alphas[self.sliders_dict[slider.GetAddressAsString('')]] = -1 + 2 * value / 100.
        # self.update()

    def pose_event(self, slider, event):
        self.pose_value = -6  + 12 * slider.GetRepresentation().GetValue() / 100.

    @property
    def interactor(self):
        return self.GetInteractor()

    def center_w_base(self):
        for _ in range(3):
            for i in range(16):
                projection = torch.einsum('nwd,wd->n', self.w_base, self.dirs[i])
                delta = .5 * (self.base_magnitude[i] - projection) / (self.dirs[i] ** 2).sum()
                self.w_base += self.dirs[i][None, :, :] * delta[:, None, None]


    def get_rand_w(self):
        with torch.no_grad():
            x = torch.randn(1, 512).cuda()
            w = self.model.generator.style(x).unsqueeze(1).repeat(1, 18, 1).cpu()[0]
            # for _ in range(3):
            #     for i in range(16):
            #         projection = torch.einsum('wd,wd', w, self.dirs[i])
            #         delta = .5 * (self.base_magnitude[i] - projection) / (self.dirs[i] ** 2).sum()
            #         w = w + self.dirs[i] * delta
        return w

    def set_scroller(self):
        viseme = ['AA', 'AH', 'AO', 'OW', 'UH', 'EH', 'AY', 'EY', 'IY', 'R', 'L', 'W', 'M/P/B', 'NG/D/TH', 'CH/J/SH',
                  'F/V', 'X']
        for i in range(16):
            slider = ui_utils.make_slider(self.interactor, self.slider_event, self.render, .1 + .8 * i / 16.,
                                          viseme[i], 50.)
            self.sliders.append(slider)
            self.sliders_dict[slider[0].GetAddressAsString('')] = i
        self.pose_slider = ui_utils.make_slider(self.interactor, self.pose_event, self.render, .9, 'pose', 50.)

    def __init__(self, image_data, render):
        super(InteractorStyle, self).__init__()
        self.AddObserver(vtk.vtkCommand.KeyPressEvent, self.on_key_press)
        self.model = stylegan_wrapper.get_model()
        self.render = render
        self.image_data = image_data
        self.sliders = []
        self.sliders_dict = {}
        self.w_pose = torch.load("../encoder4editing/editings/interfacegan_directions/pose.pt").to(CPU)[0]
        valid = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_valid_v3')
        w_plus = []
        for i in range(17):
            try:
                w_ = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}_v3')[valid]
            except:
                w_ = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}')[valid]
            w_plus.append(w_)
        w_plus_v1 = []
        for i in range(17):
            w_ = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}')[valid]
            w_plus_v1.append(w_)
        # w_plus = [files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}_v2')[valid] for i in
        #           range(17)]
        dirs_all = [w_plus_v1[i] - w_plus_v1[-1] for i in range(17)]
        dirs = torch.stack([dirs_all[i].mean(0) for i in range(17)], dim=0)
        dirs_all = [w_plus[i] - w_plus[-1] for i in range(17)]
        self.dirs = torch.stack([dirs_all[i].mean(0) for i in range(17)], dim=0)
        self.dirs[12] = dirs[12]
        self.dirs[4] = dirs[4]
        # self.dirs[12] = self.dirs[12] * 2
        self.image_id = torch.randint(199, (1,)).item()
        self.w_base = w_plus[-1]
        # self.w_base = self.w_base[:, :1].repeat(1, 18, 1)
        # self.base_magnitude = torch.einsum('nwd,rwd->nr', self.w_base, self.dirs).mean(0)
        # self.center_w_base()
        self.image_id = torch.randint(199, (1,)).item()
        self.w_cur = self.get_rand_w()
        self.alphas = torch.zeros(16)
        self.pose_value = 0.
        self.replace_image(self.w_base[self.image_id])


def vtk_show():
    fileName = "../assets/grid/grid_0_0.png"
    # Read the image.
    readerFactory = vtk.vtkImageReader2Factory()
    reader = readerFactory.CreateImageReader2(fileName)
    reader.SetFileName(fileName)
    reader.Update()

    scalarRange = [0, 255.]
    middleSlice = (reader.GetOutput().GetExtent()[5] - reader.GetOutput().GetExtent()[4]) // 2

    # Work with double images
    cast = vtkImageCast()
    cast.SetInputConnection(reader.GetOutputPort())
    cast.SetOutputScalarTypeToDouble()
    cast.Update()

    originalData = vtk.vtkImageData()
    originalData.DeepCopy(cast.GetOutput())


    # Create actors
    source1Actor = vtkImageActor()
    source1Actor.GetMapper().SetInputData(originalData)
    # There will be one render window
    renderWindow = vtkRenderWindow()
    renderWindow.SetSize(600, 300)

    # And one interactor
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)


    interactor.Initialize()
    centerRenderer = vtkRenderer()
    style = InteractorStyle(originalData, centerRenderer)
    interactor.SetInteractorStyle(style)
    # Define viewport ranges
    # (xmin, ymin, xmax, ymax)
    leftViewport = [0.0, 0.0, 0.5, 1.0]
    centerViewport = [0.5, 0.0, 1., 1.0]

    # Setup renderers
    leftRenderer = vtkRenderer()
    renderWindow.AddRenderer(leftRenderer)
    leftRenderer.SetViewport(leftViewport)
    leftRenderer.SetBackground(1., 1., 1.)

    renderWindow.AddRenderer(centerRenderer)
    centerRenderer.SetViewport(centerViewport)
    centerRenderer.SetBackground(1., 1., 1.)
    leftRenderer.AddActor(source1Actor)
    leftRenderer.ResetCamera()
    centerRenderer.ResetCamera()
    renderWindow.SetWindowName('viseme control')
    style.SetCurrentRenderer(leftRenderer)
    style.set_scroller()
    renderWindow.Render()
    interactor.Start()


@models_utils.torch_no_grad
def create_viseme_grid():
    seed = 99
    torch.manual_seed(seed)
    np.random.seed(seed)
    def get_image(row, cols):
        if row == 0:
            return files_utils.load_image(f"{constants.CACHE_ROOT}/viseme/{dirs_offset + cols:02d}")
        return all_images[(row - 1) * num_cols + cols]
    num_rows = 8
    num_cols = 16
    stylegan = stylegan_wrapper.get_model()
    valid = files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_valid_v3')
    w_plus = [files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}_v3')[valid] for i in range(17)]
    w_plus_v1 = [files_utils.load_pickle(f'{constants.CACHE_ROOT}/viseme/viseme_w_plus_{i:02d}')[valid] for i in
                 range(17)]
    dirs = [w_plus_v1[i] - w_plus_v1[-1] for i in range(17)]
    dirs_v1 = torch.stack([dirs[i].mean(0) for i in range(17)], dim=0)
    dirs = [w_plus[i] - w_plus[-1] for i in range(16)]
    dirs = [item.mean(0) for item in dirs]
    dirs = torch.stack(dirs, dim=0)
    dirs[12] = dirs_v1[12]
    dirs[4] = dirs_v1[4]
    alphas = torch.ones(16) * 2.5
    alphas[:2] = 1
    alphas[2] = 2
    alphas[3] = 1
    alphas[5] = 1
    alphas[15] = 2
    alphas[14] = 2
    alphas[12] = 2
    alphas[11] = 1.5
    alphas[10] = 1.5
    alphas[9] = 1
    alphas[8] = 1
    alphas[7] = 1
    w_pose = torch.load("../encoder4editing/editings/interfacegan_directions/pose.pt").to(CPU)[0]
    dirs = dirs * alphas[:, None, None]
    w_base = w_plus[-1]
    # w_base = w_base[:, :1].repeat(1, 18, 1)
    all_images = []
    select = np.random.choice(len(w_base), num_rows, replace=False)
    dirs_offset = 0
    for i in range(num_rows):
        for j in range(num_cols):
            w_new = w_base[select[i]].clone()
            w_new[:8] += dirs[dirs_offset + j][:8]
            w_new -= 4 * w_pose
            out = stylegan(w_new.cuda().unsqueeze(0))
            out = nnf.interpolate(out, size=(256, 256))
            out = out[0].detach().cpu().permute(1, 2, 0)
            out = ((out + 1) * 127.5).clamp(0, 255).numpy()
            all_images.append(out)
    grid = image_utils.simple_grid(num_rows + 1, num_cols, 5, 256, get_image)
    files_utils.save_image(grid, f'./grid_{dirs_offset}_{dirs_offset + num_cols}_left.png')


if __name__ == '__main__':
    vtk_show()
