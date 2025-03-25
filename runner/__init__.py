'''
1. define some shared parameters
'''
from pathlib import Path
DATA_FOLDER = Path('/data/qh/DoTA/data')
FONT_FOLDER = "/data/qh/dependency/arial.ttf"
CHFONT_FOLDER = "/data/qh/dependency/microsoft_yahei.ttf"

# train / test folder
pretrained_weight_path = Path('/data/qh/pretrain_models/')
vst_pretrained_weight_path = pretrained_weight_path / 'swin_base_patch244_window1677_sthv2.pth'                        

DoTA_FOLDER = Path('/ssd/qh/DoTA/data') 
DADA_FOLDER = Path('/ssd/qh/DADA2000/')
META_FOLDER = DATA_FOLDER / 'metadata'
FONT_FOLDER = "/data/qh/dependency/arial.ttf"

DEBUG_FOLDER = Path('/data/qh/DoTA/Intervl-TAD/debug')

# __all__ =[

# ]