import typing as t
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

# 定义图像转换操作
import cv2
import numpy as np

scale = 4
patch_h , patch_w = int(9*scale) , int(16*scale)
patch_size = 14
input_shape = (patch_h * patch_size, patch_w * patch_size)

cfg.dataset.type = Dota
cfg.dataset.name = 'dota'
cfg.dataset.root_path = str(DoTA_FOLDER)
cfg.dataset.phase = 'train'
cfg.dataset.pre_process_type = 'dinov2_emb' # rgb or sam_emb
cfg.dataset.transforms = {
        'emb': transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x,dtype=torch.float)),
        ])}

cfg.dataset.VCL = None
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.0
cfg.dataset.sorted_num_frames = False
cfg.dataset.data_type = ''
cfg.train_dataset.cfg = cfg.dataset

cfg.dataset.phase = 'val'
cfg.dataset.transforms = {
        'emb': transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x,dtype=torch.float)),
        ])}

cfg.dataset.VCL = None
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.0
cfg.dataset.sorted_num_frames = True
cfg.dataset.data_type = ''

# prepare test dataset
# cfg.test_dataset.data = cfg.dataset.type(**cfg.dataset)
cfg.test_dataset.cfg = cfg.dataset

# delete template variable data
cfg.dataset = None


