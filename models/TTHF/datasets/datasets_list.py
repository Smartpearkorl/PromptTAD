import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import glob
from collections import OrderedDict
from PIL import ImageFile
from datasets.transform_list import RandomCropNumpy,EnhancedCompose,RandomColor,RandomHorizontalFlip,ArrayToTensorNumpy,CropNumpy
from torchvision import transforms
import os
import json
import joblib
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _is_pil_image(img):
    return isinstance(img, Image.Image)


class DoTADatasetSubMPM(data.Dataset):
    def __init__(self, args, train=True, return_filename=False):
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.height = args.height
        self.width = args.width
        if train is True:
            self.datafile = args.trainfile_dota
        else:
            self.datafile = args.testfile_dota
        self.train = train
        self.transform = Transformer(args)
        self.args = args
        self.data_path = "/ssd/qh/DoTA/data/frames"  # path to the video frames of DoTA dataset
        if train is True:
            self.label_file = '/ssd/qh/DoTA/data/metadata/metadata_train.json'  # path to gt
            self.one_hot_file_M = '/ssd/qh/DoTA/data/ground_truth_demo/train_one_hot_label.json'  # path to label
            self.one_hot_file_S = '/ssd/qh/DoTA/data/ground_truth_demo/train_one_hot_label_bi.json'
            self.one_hot_label_M = joblib.load(self.one_hot_file_M)
            self.one_hot_label_S = joblib.load(self.one_hot_file_S)
            # self.one_hot_label_M = joblib.load(open(self.one_hot_file_M, 'rb'))
            # self.one_hot_label_S = joblib.load(open(self.one_hot_file_S, 'rb'))
        else:
            self.label_file = '/ssd/qh/DoTA/data/metadata/metadata_val.json'
        self.return_filename = return_filename
        self.fileset = []
        with open(self.datafile, 'r') as f:
            for row in f:
                self.fileset.append(row.strip('\n'))
        self.fileset = sorted(self.fileset)
        self.label = json.load(open(self.label_file, 'rb'))
        self.triplets = list()
        idx = 1  # video idx
        for video in sorted(self.fileset):
            video_name = video
            video_label = self.label[video_name]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(
                os.path.join(self.data_path, video, 'images', '*' + '.jpg'))
            self.videos[video_name]['frame'].sort()
            video_frame = self.videos[video_name]['frame']
            self.videos[video_name]['length'] = int(video_label['video_end']) - int(video_label['video_start']) + 1
            if len(self.videos[video_name]['frame']) == self.videos[video_name]['length']:
                pass
            else:
                self.videos[video_name]['frame'] = self.videos[video_name]['frame'][:self.videos[video_name]['length']]
            for i in range(len(self.videos[video_name]['frame']) - 1):
                triplet = (video_frame[i], video_frame[i + 1])
                self.triplets.append(triplet)

            idx += 1


        self.tot_frame_num = len(self.triplets)

    def __getitem__(self, index):
        # Opening image files.   rgb: input color image, gt: sparse depth map
        rgb_file = self.triplets[index]
        rgb_p = Image.open(rgb_file[0])
        rgb_c = Image.open(rgb_file[1])
        one_hot_label_M = False
        one_hot_label_S = False
        text = False
        divided_file_ = rgb_file[-1].split('/')
        filename = divided_file_[-3] + '_' + divided_file_[-1]
        # load gt label (one-hot label)
        if self.train is True:
            one_hot_label_text_M = self.one_hot_label_M[divided_file_[-3]][int(divided_file_[-1].split('.')[0])]
            one_hot_label_M = torch.tensor(one_hot_label_text_M[1])
            one_hot_label_text_S = self.one_hot_label_S[divided_file_[-3]][int(divided_file_[-1].split('.')[0])]
            one_hot_label_S = torch.tensor(one_hot_label_text_S[1])

        rgb_p = rgb_p.resize((self.width, self.height), 2)
        rgb_c = rgb_c.resize((self.width, self.height), 2)

        rgb_p = np.asarray(rgb_p, dtype=np.float32) / 255.0
        rgb_c = np.asarray(rgb_c, dtype=np.float32) / 255.0

        rgb_p, rgb_c = self.transform([rgb_p, rgb_c], self.train)

        if self.return_filename is True:
            return rgb_p, rgb_c, one_hot_label_M, one_hot_label_S, text, filename
        else:
            return rgb_p, rgb_c, one_hot_label_M, one_hot_label_S, text

    def __len__(self):
        return self.tot_frame_num


class DADADatasetSubMPM(data.Dataset):
    def __init__(self, args, train=True):
        self.videos = OrderedDict()
        self.all_frame_addr = list()
        self.height = args.height
        self.width = args.width
        self.data_path = "/data/lrq/DADA-2000"
        self.data_dir = os.path.join(self.data_path, 'DADA_dataset')
        if train is True:
            raise ("Training mode is currently not supported for DADA dataset")
        else:
            self.datafile = os.path.join(self.data_path, 'test_file.json')
        self.train = train
        self.transform = Transformer(args)
        self.args = args
        self.fileset = []
        with open(self.datafile, 'r') as f:
            ind = json.load(f)
            for row in ind:
                class_id = row[0][0]
                video_id = row[0][1]
                video_name = class_id + '_' + video_id
                self.fileset.append(video_name)
        self.triplets = list()
        idx = 1  # video idx
        for video in sorted(self.fileset):
            video_name = video
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            video_path = video.split('_')
            self.videos[video_name]['frame'] = glob.glob(
                os.path.join(self.data_dir, video_path[0], video_path[1], 'images',
                             '*' + '.jpg'))
            self.videos[video_name]['frame'].sort()
            video_frame = self.videos[video_name]['frame']
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

            for i in range(len(self.videos[video_name]['frame']) - 1):
                triplet = (video_frame[i], video_frame[i + 1])
                self.triplets.append(triplet)

            idx += 1

        self.tot_frame_num = len(self.triplets)


    def __getitem__(self, index):
        # Opening image files.   rgb: input color image, gt: sparse depth map
        rgb_file = self.triplets[index]
        rgb_p = Image.open(rgb_file[0])
        rgb_c = Image.open(rgb_file[1])
        one_hot_label_M = False
        one_hot_label_S = False
        text = False
        if self.train is True:
            raise NotImplementedError

        rgb_p = rgb_p.resize((self.width, self.height), 2)
        rgb_c = rgb_c.resize((self.width, self.height), 2)

        rgb_p = np.asarray(rgb_p, dtype=np.float32) / 255.0
        rgb_c = np.asarray(rgb_c, dtype=np.float32) / 255.0

        rgb_p, rgb_c = self.transform([rgb_p, rgb_c], self.train)
        return rgb_p, rgb_c, one_hot_label_M, one_hot_label_S, text

    def __len__(self):
        return self.tot_frame_num


class Transformer(object):
    def __init__(self, args):
        if args.dataset in ['DoTA', 'DADA']:
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25)),
                 RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25))],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])
            self.test_transform = EnhancedCompose([
                CropNumpy((args.height, args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            ])

    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
