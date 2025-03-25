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
cfg.dataset.pre_process_type = 'rgb' # rgb or sam_emb
cfg.dataset.data_mean = [0.5, 0.5, 0.5]
cfg.dataset.data_std = [0.5, 0.5, 0.5]
cfg.dataset.transforms = {
        'image': transforms.Compose([
            pad_frames(input_shape),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [T, H, W, C] -> [T, C, H, W]
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            T.AugMix(),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(cfg.dataset.data_mean, cfg.dataset.data_std),
            # [T, C, H, W]
        ])}
cfg.dataset.VCL = 8
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.5
cfg.dataset.sorted_num_frames = False
cfg.dataset.data_type = ''

# prepare train dataset
# cfg.train_dataset.data = cfg.dataset.type(**cfg.dataset)
cfg.train_dataset.cfg = cfg.dataset

cfg.dataset.phase = 'val'
cfg.dataset.data_mean = [0.5, 0.5, 0.5]
cfg.dataset.data_std = [0.5, 0.5, 0.5]
cfg.dataset.transforms = {
        'image': transforms.Compose([
            pad_frames(input_shape),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [T, H, W, C] -> [T, C, H, W]
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(cfg.dataset.data_mean, cfg.dataset.data_std),
            # [T, C, H, W]
        ]),
    }

cfg.dataset.VCL = None
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.0
cfg.dataset.sorted_num_frames = True
cfg.dataset.data_type = 'tiny_' # 'select_train_' 'sub_'

# prepare test dataset
# cfg.test_dataset.data = cfg.dataset.type(**cfg.dataset)
cfg.test_dataset.cfg = cfg.dataset

# delete template variable data
cfg.dataset = None
