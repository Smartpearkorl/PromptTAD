from alchemy_cat.dl_config import load_config, Config , DEP
import sys
from models.TTHF.datasets.transform_list import *
cfg = Config()
from runner import DoTA_FOLDER
# Dataset
import torch
from torchvision import transforms
from pytorchvideo import transforms as T
from runner.src.data_transform import pad_frames
from runner.src.dota import Dota
import cv2

input_shape = (224, 224)
rand_crop = RandomCropNumpy(input_shape)
rand_color = RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25))
cfg.dataset.type = Dota
cfg.dataset.name = 'dota'
cfg.dataset.root_path = str(DoTA_FOLDER)
cfg.dataset.phase = 'train'
cfg.dataset.pre_process_type = 'rgb' # rgb or sam_emb
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.5
cfg.dataset.data_mean = [0.485, 0.456, 0.406]
cfg.dataset.data_std = [0.229, 0.224, 0.225]
cfg.dataset.transforms = {
        'image': transforms.Compose([
            transforms.Lambda(lambda x: np.array([cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR) for img in x])),       
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Lambda(lambda x: np.array(rand_crop(x))), 
            transforms.Lambda(lambda x: np.array([rand_color(img) for img in x])),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [ H, W, C] -> [ C, H, W]
            transforms.Lambda(lambda x: x.permute(0,3,1,2)),    
            transforms.Normalize(cfg.dataset.data_mean, cfg.dataset.data_std),
        ])}
cfg.dataset.VCL = 8
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.5
cfg.dataset.sorted_num_frames = False
cfg.dataset.data_type = ''

# prepare train dataset
cfg.train_dataset.cfg = cfg.dataset

cfg.dataset.phase = 'val'
cfg.dataset.data_mean = [0.485, 0.456, 0.406]
cfg.dataset.data_std = [0.229, 0.224, 0.225]
cfg.dataset.transforms = {
        'image': transforms.Compose([           
            transforms.Lambda(lambda x: np.array([cv2.resize(img, input_shape, interpolation=cv2.INTER_LINEAR) for img in x])),       
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Lambda(lambda x: np.array(rand_crop(x))), 
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [ H, W, C] -> [ C, H, W]
            transforms.Lambda(lambda x: x.permute(0,3,1,2)),    
            transforms.Normalize(cfg.dataset.data_mean, cfg.dataset.data_std),
        ])}
cfg.dataset.VCL = None
cfg.dataset.vertical_flip_prob = 0.0
cfg.dataset.horizontal_flip_prob = 0.0
cfg.dataset.sorted_num_frames = True
cfg.dataset.data_type = ''

# prepare test dataset
cfg.test_dataset.cfg = cfg.dataset

# delete template variable data
cfg.dataset = None
