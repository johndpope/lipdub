import os
import sys


IS_WINDOWS = sys.platform == 'win32'
get_trace = getattr(sys, 'gettrace', None)
DEBUG = get_trace is not None and get_trace() is not None
EPSILON = 1e-4
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = f'{PROJECT_ROOT}/assets/'
RAW_ROOT = f'{DATA_ROOT}raw/'
PLOT_ROOT = f'{PROJECT_ROOT}/plots/'
OUT_ROOT = f'{DATA_ROOT}out/'
CHECKPOINTS_ROOT = f'{DATA_ROOT}checkpoints/'
CACHE_ROOT = f'{DATA_ROOT}cache/'



stylegan_weights = f'{PROJECT_ROOT}/weights/ffhq/stylegan2-ffhq-config-f.pt'
MNT_ROOT = f'{DATA_ROOT}dubbing/'
LRS2_PROCESS = f'{MNT_ROOT}lrs2_process/'
LRS2_ROOT = f'/home/ahertz/Downloads/mvlrs_v1/main'

COLORS = [[231, 231, 91], [103, 157, 200], [177, 116, 76], [88, 164, 149],
         [236, 150, 130], [80, 176, 70], [108, 136, 66], [78, 78, 75],
         [41, 44, 104], [217, 49, 50], [87, 40, 96], [85, 109, 115], [234, 234, 230],
          [30, 30, 30]]
