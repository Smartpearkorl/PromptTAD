""" This file is modified from:
https://raw.githubusercontent.com/piergiaj/pytorch-i3d/master/videotransforms.py
"""
import numpy as np
import cv2
import torch

from torchvision import transforms as T

def pad_frames(input_shape):
    def f(imgs):
        t, h, w, c = imgs.shape
        shape_r, shape_c = input_shape

        ims = np.zeros((t, shape_r, shape_c, c), dtype=np.float32)
        for i, im in enumerate(imgs):
            padded_image = padding(im, shape_r, shape_c, c)
            if c == 1:
                padded_image = np.expand_dims(padded_image, axis=-1)
            ims[i] = padded_image.astype(np.float32)
        return ims
    return f


def padding(img, shape_r=480, shape_c=640, channels=3):
    img_padded = np.zeros((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)

    original_shape = img.shape
    rows_rate = original_shape[0]/shape_r
    cols_rate = original_shape[1]/shape_c

    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:, ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))
        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows), :] = img

    return img_padded


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, img_size , p=0.5):
        super().__init__()
        self.image_height, self.image_width = img_size
        self.p = p

    def forward(self, img, boxes):
        coins = torch.rand(img.shape[0], device=img.device)
        indexes = (coins > self.p).nonzero().squeeze(1).tolist()
        img[indexes] = T.functional.vflip(img[indexes])
        # 将框垂直翻转： (x,y) -> (x,h-y)
        for index in indexes:
           tmp_boxes = boxes[index]
           if tmp_boxes.shape[0]!=0:
                boxes[index] =  tmp_boxes[:, [0, 3, 2, 1]] * np.array([1, -1, 1, -1]) + np.array([0, self.image_height, 0, self.image_height])
        return indexes, img , boxes


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, img_size, p=0.5):
        super().__init__()
        self.image_height, self.image_width = img_size
        self.p = p

    def forward(self, img, boxes):
        coins = torch.rand(img.shape[0], device=img.device)
        indexes = (coins > self.p).nonzero().squeeze(1).tolist()
        img[indexes] = T.functional.hflip(img[indexes])
        # 将框水平翻转: （x,y）-> （w-x,y）
        for index in indexes:
           tmp_boxes = boxes[index]
           if tmp_boxes.shape[0]!=0:
                boxes[index] =  tmp_boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([self.image_width, 0, self.image_width, 0])
        return indexes, img , boxes
