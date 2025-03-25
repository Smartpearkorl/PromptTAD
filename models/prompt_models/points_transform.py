
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple,List
from torch import Tensor
from runner.src.data_transform import padding
"""
Utilities for bounding box manipulation and GIoU.
"""
from torchvision.ops.boxes import box_area

# modified from torchvision to also return the union
def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def generate_random_boxes(batch_size, num_boxes, image_size=(640, 480)):
    """生成随机的边界框，保证右下角坐标大于或等于左上角坐标。

    Args:
        num_boxes (int): 要生成的边界框数量。
        image_size (tuple): 图像的尺寸，格式为 (宽, 高)。

    Returns:
        torch.Tensor: 形状为 (num_boxes, 4) 的张量，每行是一个边界框的 (x_min, y_min, x_max, y_max)。
    """
    batch_boxes = []
    for i in range(batch_size):
        width, height = image_size
        # 生成左上角的坐标
        x_min = torch.randint(0, width, (num_boxes,))
        y_min = torch.randint(0, height, (num_boxes,))

        # 生成右下角的坐标，保证右下角坐标大于或等于左上角坐标
        x_max = x_min + torch.randint(0, width - 10, (num_boxes,))
        y_max = y_min + torch.randint(0, height - 10, (num_boxes,))

        # 合并坐标
        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

        # 检查条件
        # assert (boxes[:, 2:] >= boxes[:, :2]).all(), "生成的边界框不满足条件"
        batch_boxes.append(boxes)
    batch_boxes = torch.stack(batch_boxes,dim=0).float()
    return batch_boxes


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def build_all_layer_point_grids(
    n_per_side: int, n_layers: int, scale_per_layer: int
) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer**i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


class ResizeCoordinates:
    def __init__(self, target_size: Tuple[int, int] , original_size: Tuple[int, int]  ) -> None:
        self.target_size = target_size
        self.original_size = original_size
        
    @staticmethod
    def get_preprocess_shape2D( old_shape:Tuple[int, int], tar_shape:Tuple[int, int]) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        oldh,oldw =  old_shape
        tarh,tarw =  tar_shape
        scale = min ( tarh * 1.0 / oldh, tarw * 1.0 / oldw )
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
    
    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape2D( original_size, self.target_size )
        # way of image paddding is middle, so need to shift boxes 
        # point(x,y) -> (w,h)
        delta_h = (self.target_size[0] - new_h)//2 
        delta_w = (self.target_size[1] - new_w)//2
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w) + delta_w
        coords[..., 1] = coords[..., 1] * (new_h / old_h) + delta_h
        return coords
    
    def apply_boxes(self, boxes: np.ndarray, original_size = None) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        original_size = original_size if original_size else self.original_size
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def normalize_box_coordinate(self,coords: np.ndarray)-> np.ndarray:
        h , w = self.target_size
        coords[..., 0] = coords[..., 0] / w
        coords[..., 2] = coords[..., 2] / w
        coords[..., 1] = coords[..., 1] / h 
        coords[..., 3] = coords[..., 3] / h
        return coords

    def denormalize_box_coordinate(self,coords: np.ndarray)-> np.ndarray:
        h , w = self.target_size
        coords[..., 0] = coords[..., 0] * w
        coords[..., 2] = coords[..., 2] * w
        coords[..., 1] = coords[..., 1] * h 
        coords[..., 3] = coords[..., 3] * h
        return coords

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # target_size = self.get_preprocess_shape2D(image.shape[0], image.shape[1], self.target_size)
        pad_img = padding(image, shape_r = self.target_size[0], shape_c = self.target_size[1] )
        return pad_img


