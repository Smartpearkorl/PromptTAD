from alchemy_cat.dl_config import load_config, Config , DEP
import sys
from runner import DoTA_FOLDER
cfg = Config()

# Dataset
import torch
import numpy as np
from torchvision import transforms
from runner.src.dota import Dota
from models.sam_model.utils.transforms import ResizeLongestSide

processor = ResizeLongestSide(1024)


input_shape = [480, 640]

cfg.dataset.type = Dota
cfg.dataset.name = 'dota'
cfg.dataset.root_path = str(DoTA_FOLDER)
cfg.dataset.phase = 'train'
cfg.dataset.pre_process_type = 'rgb' # rgb or sam_emb
cfg.dataset.transforms = {
        'image': transforms.Compose([
            transforms.Lambda(lambda x: [processor.apply_image(i) for i in x] ),
            transforms.Lambda(lambda x: torch.tensor(np.array(x)).permute(0,3,1,2)), # list -> np.array -> tensor -> [B,H,W,3] -> [B,3,H,W]
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
cfg.dataset.data_type = 'mini_'

# prepare test dataset
# cfg.test_dataset.data = cfg.dataset.type(**cfg.dataset)
cfg.test_dataset.cfg = cfg.dataset

# delete template variable data
cfg.dataset = None
