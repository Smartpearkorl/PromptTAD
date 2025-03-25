import typing as t
from alchemy_cat.dl_config import load_config, Config , DEP
import sys
from runner import DoTA_FOLDER
type_std_mean: t.TypeAlias = tuple[float, float, float]
OPENAI_DATASET_MEAN: t.Final[type_std_mean] = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD: t.Final[type_std_mean] = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN: t.Final[type_std_mean] = (0.485, 0.456, 0.406)
IMAGENET_STD: t.Final[type_std_mean] = (0.229, 0.224, 0.225)
INCEPTION_MEAN: t.Final[type_std_mean] = (0.5, 0.5, 0.5)
INCEPTION_STD: t.Final[type_std_mean] = (0.5, 0.5, 0.5)

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
cfg.dataset.pre_process_type = 'rgb' # rgb or sam_emb
cfg.dataset.data_mean = IMAGENET_MEAN
cfg.dataset.data_std = IMAGENET_STD

cfg.dataset.transforms = {
        'image': transforms.Compose([
            # transforms.Lambda(lambda x: np.array([cv2.GaussianBlur(img, (9, 9), sigmaX=0.1) for img in x])), 
            transforms.Lambda(lambda x: np.array([cv2.resize(img, (input_shape[1],input_shape[0]), interpolation=cv2.INTER_LINEAR) for img in x])),       
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [ H, W, C] -> [ C, H, W]
            transforms.Lambda(lambda x: x.permute(0,3,1,2)),    
            transforms.Normalize(cfg.dataset.data_mean, cfg.dataset.data_std),
        ])}

cfg.dataset.VCL = None
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.5
cfg.dataset.sorted_num_frames = False
cfg.dataset.data_type = ''
cfg.train_dataset.cfg = cfg.dataset

cfg.dataset.phase = 'val'
cfg.dataset.data_mean = IMAGENET_MEAN
cfg.dataset.data_std = IMAGENET_STD
cfg.dataset.transforms = {
        'image': transforms.Compose([
            transforms.Lambda(lambda x: np.array([cv2.resize(img, (input_shape[1],input_shape[0]), interpolation=cv2.INTER_LINEAR) for img in x])),       
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [ H, W, C] -> [ C, H, W]
            transforms.Lambda(lambda x: x.permute(0,3,1,2)),    
            transforms.Normalize(cfg.dataset.data_mean, cfg.dataset.data_std),
        ])}

cfg.dataset.VCL = None
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.0
cfg.dataset.sorted_num_frames = True
cfg.dataset.data_type = 'select_train_' # '' 'sub_' 'select_train_'

# prepare test dataset
# cfg.test_dataset.data = cfg.dataset.type(**cfg.dataset)
cfg.test_dataset.cfg = cfg.dataset

# delete template variable data
cfg.dataset = None


