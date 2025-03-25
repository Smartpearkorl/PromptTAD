from alchemy_cat.dl_config import load_config, Config , DEP
import sys
from runner import DoTA_FOLDER
cfg = Config()

# Dataset
import torch
from torchvision import transforms
from pytorchvideo import transforms as T
from runner.src.data_transform import pad_frames
from runner.src.dota import Dota

input_shape = [480, 640]

cfg.dataset.type = Dota
cfg.dataset.name = 'dota'
cfg.dataset.root_path = str(DoTA_FOLDER)
cfg.dataset.phase = 'train'
cfg.dataset.pre_process_type = 'sam_emb' # 'rgb' or 'emb'
cfg.dataset.transforms = {
        'emb': transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x,dtype=torch.float)),
        ])}
cfg.dataset.VCL = 16
cfg.dataset.sorted_num_frames = False
cfg.dataset.data_type = ''

# prepare train dataset
# cfg.train_dataset.data = cfg.dataset.type(**cfg.dataset)
cfg.train_dataset.cfg = cfg.dataset

cfg.dataset.phase = 'val'
cfg.dataset.VCL = None
cfg.dataset.sorted_num_frames = True
cfg.dataset.data_type = ''

# prepare test dataset
# cfg.test_dataset.data = cfg.dataset.type(**cfg.dataset)
cfg.test_dataset.cfg = cfg.dataset
# delete template variable data
cfg.dataset = None
