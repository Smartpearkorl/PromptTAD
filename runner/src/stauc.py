"""
STAUCMetrics class and utility functions for the spatial-temporal area under ROC curve (STAUC).

To use stauc metrics, first initialize a STAUCMetrics object,
then update the metrics for each video in the test dataset,
finally run get_stauc to compute the STAUC value.
"""
import numpy as np
import torch
import copy
import pickle
import json
from datetime import datetime
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import column_or_1d, check_consistent_length, assert_all_finite
from sklearn.utils.multiclass import type_of_target
from sklearn import metrics
import warnings
from tqdm import tqdm
import itertools
import os
import argparse

# Custom imports
import sys
from pathlib import Path
FILE = Path(__file__).resolve() # /home/qh/TDD/pama/runner/src/stauc.py
sys.path.insert(0, str(FILE.parents[2]))
import os 
os.chdir(FILE.parents[2])
from runner.src.tools import custom_print
from runner.src.metrics import normalize_video 
from runner import DATA_FOLDER

H = 256
W = 256

# H = 1280
# W = 720

def bbox_to_score_map(bboxes, scores, image_size=(1280,720)):
    '''
    Params:
        bboxes: a BoxList object or a tensor bboxes, in x1y1x2y2 format
        scores: scores of each bbox
    Return:
        score_map: (H, W)
    '''
    bboxes = copy.deepcopy(bboxes)
    bboxes = torch.as_tensor(bboxes).type(torch.float)
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes)
    score_map = torch.zeros(H, W)
    if bboxes.max() > 1:
        # normalize then denormalize to correct size
        bboxes[:,[0,2]] /= image_size[0] 
        bboxes[:,[1,3]] /= image_size[1]
    bboxes[:,[0,2]] *= W
    bboxes[:,[1,3]] *= H
    bboxes = bboxes.type(torch.long)
    bboxes[:,[0,2]] = torch.clamp(bboxes[:,[0,2]], min=0, max=W)
    bboxes[:,[1,3]] = torch.clamp(bboxes[:,[1,3]], min=0, max=H)
    
    # Generate gaussian
    for bbox, score in zip(bboxes, scores):
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        sigma = torch.tensor([w, h])
        
        x_locs = torch.arange(0, w, 1).type(torch.float)
        y_locs = torch.arange(0, h, 1).type(torch.float)
        y_locs = y_locs[:, np.newaxis]

        x0 = w // 2
        y0 = h // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- (((x_locs - x0) ** 2)/sigma[0] + ((y_locs - y0) ** 2)/sigma[1]) / 2 )        
        score = g * score
        score_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] += score
        
    return score_map

def get_num_pixels_in_region(bboxes, H, W):
    mask = torch.zeros([H, W])
    for bbox in bboxes:
        mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    K = int(mask.sum())
    return K

def get_tarr(difference_map, 
             label, 
             bboxes, 
             image_size=(1280, 720), 
             method_type='frame', 
             obj_bboxes=None, 
             obj_scores=None,
             top_percent=-1
             ):
    '''
    Given a difference map and annotations, compute the 
    True Anomaly Region Rate
    difference_map: (H, W)
    bboxes: a list of gt anomalous box, x1y1x2y2, not normalized 
    label: 0/1, normal or abnormal
    input_type: 'object' or 'frame' 
    '''
    if label == 0:
        return 0, []
    elif len(bboxes) == 0:
        return 1, []
    else:
        if method_type == 'object':
            if len(obj_bboxes) == 0 or len(obj_scores) == 0:
                return 1, []
            else:
                difference_map = bbox_to_score_map(obj_bboxes, obj_scores, image_size=image_size)
        H, W = difference_map.shape

        if not isinstance(difference_map, torch.Tensor):
            difference_map = torch.FloatTensor(difference_map)

        bboxes = copy.deepcopy(bboxes)
        if isinstance(bboxes, (list, np.ndarray)):
            bboxes = torch.FloatTensor(bboxes)

        if bboxes.max() > 1:
            # normalize then denormalize to correct size
            bboxes[:,[0,2]] /= image_size[0] 
            bboxes[:,[1,3]] /= image_size[1]
        bboxes[:,[0,2]] *= W
        bboxes[:,[1,3]] *= H
        bboxes = bboxes.type(torch.long)
        
        if top_percent != -1:
            K = int(top_percent * difference_map.shape[0] * difference_map.shape[1])
        else:
            K = get_num_pixels_in_region(bboxes, H, W)

        values, indices = torch.topk(difference_map.view(-1), k=K)
        h_coord = (indices // W)#.type(torch.float)
        w_coord = (indices % W)#.type(torch.float)
        
        mask = torch.zeros([H, W])
        for bbox in bboxes:
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        true_anomaly_idx = mask[h_coord, w_coord] == 1
        tarr = values[true_anomaly_idx].sum() / (values.sum() + 1e-6)
        
        return tarr, mask

class STAUCMetrics():
    def __init__(self):
        self.labels = []
        self.scores = []
        self.tarrs = []

    def update(self, gt_labels, gt_bboxes, pred_scores, pred_bboxes, pred_bboxes_score):
        """
        Update the metrics given the labels and predictions of a video sample.

        gt_labels: a list of labels of each frame in a video
        gt_bboxes: a list of np arraies of annotated bounding boxes in each frame of a video.
        pred_scores: a list of np arraies of predicted anomaly scores of each predicted bbox.
        pred_bboxes: a list of np arraies of predicted bounding boxes in each frame of a video.
        """
        for frame_id in tqdm(range(len(gt_labels))):
            tarr, mask = get_tarr(difference_map=None, 
                                    label=gt_labels[frame_id], 
                                    bboxes=gt_bboxes[frame_id],
                                    method_type='object',
                                    obj_bboxes=pred_bboxes[frame_id],
                                    obj_scores=pred_bboxes_score[frame_id])
            self.tarrs.append(tarr)
            self.labels.append(gt_labels[frame_id])
            self.scores.append(pred_scores[frame_id])

    def get_stauc(self, pos_label=1):
        """Compute the Spatio-temperal STAUC, regular AUC, and the score gap."""
        fpr, tpr, sttpr, thresholds = self.stroc_curve(pos_label)
        stauc = metrics.auc(fpr, sttpr)
        auc = metrics.auc(fpr, tpr)
        self.labels = torch.tensor(self.labels)
        self.scores = torch.tensor(self.scores)
        gap = self.scores[self.labels == 1].mean() - self.scores[self.labels == 0].mean()
        gap = gap.item()
        return stauc, auc, gap

    def stroc_curve(self, pos_label=0, drop_intermediate=True):
        """Compute the ST-ROC curve."""
        fps, tps, thresholds, positives = self._binary_clf_curve(
                                            y_true=self.labels, 
                                            y_score=self.scores, 
                                            pos_label=pos_label, 
                                            sample_weight=self.tarrs)

        if drop_intermediate and len(fps) > 2:
            optimal_idxs = np.where(np.r_[True,
                                          np.logical_or(np.diff(fps, 2),
                                                        np.diff(tps, 2)),
                                          True])[0]
            fps = fps[optimal_idxs]
            tps = tps[optimal_idxs]
            positives = positives[optimal_idxs]
            thresholds = thresholds[optimal_idxs]

        # Add an extra threshold position
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        positives = np.r_[0, positives]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

        if fps[-1] <= 0:
            warnings.warn("No negative samples in y_true, "
                        "false positive value should be meaningless")
            fpr = np.repeat(np.nan, fps.shape)
        else:
            fpr = fps / fps[-1]

        if tps[-1] <= 0:
            warnings.warn("No positive samples in y_true, "
                        "true positive value should be meaningless")
            tpr = np.repeat(np.nan, tps.shape)
            sttpr =  np.repeat(np.nan, tps.shape)
        else:
            sttpr = tps / positives[-1] #tps[-1]
            tpr = positives / positives[-1]
        return fpr, tpr, sttpr, thresholds
    
    def _binary_clf_curve(self, y_true, y_score, pos_label=None, sample_weight=None):
        """Calculate true and false positives per binary classification threshold.
        Parameters
        ----------
        y_true : array, shape = [n_samples]
            True targets of binary classification
        y_score : array, shape = [n_samples]
            Estimated probabilities or decision function
        pos_label : int or str, default=None
            The label of the positive class
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        Returns
        -------
        fps : array, shape = [n_thresholds]
            A count of false positives, at index i being the number of negative
            samples assigned a score >= thresholds[i]. The total number of
            negative samples is equal to fps[-1] (thus true negatives are given by
            fps[-1] - fps).
        tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
            An increasing count of true positives, at index i being the number
            of positive samples assigned a score >= thresholds[i]. The total
            number of positive samples is equal to tps[-1] (thus false negatives
            are given by tps[-1] - tps).
        thresholds : array, shape = [n_thresholds]
            Decreasing score values.
        """
        # Check to make sure y_true is valid
        y_type = type_of_target(y_true)
        if not (y_type == "binary" or
                (y_type == "multiclass" and pos_label is not None)):
            raise ValueError("{0} format is not supported".format(y_type))

        check_consistent_length(y_true, y_score, sample_weight)
        y_true = column_or_1d(y_true)
        y_score = column_or_1d(y_score)
        assert_all_finite(y_true)
        assert_all_finite(y_score)

        if sample_weight is not None:
            sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
        # triggering a FutureWarning by calling np.array_equal(a, b)
        # when elements in the two arrays are not comparable.
        classes = np.unique(y_true)
        if (pos_label is None and (
                classes.dtype.kind in ('O', 'U', 'S') or
                not (np.array_equal(classes, [0, 1]) or
                    np.array_equal(classes, [-1, 1]) or
                    np.array_equal(classes, [0]) or
                    np.array_equal(classes, [-1]) or
                    np.array_equal(classes, [1])))):
            classes_repr = ", ".join(repr(c) for c in classes)
            raise ValueError("y_true takes value in {{{classes_repr}}} and "
                            "pos_label is not specified: either make y_true "
                            "take value in {{0, 1}} or {{-1, 1}} or "
                            "pass pos_label explicitly.".format(
                                classes_repr=classes_repr))
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        if sample_weight is not None:
            weight = sample_weight[desc_score_indices]
        else:
            weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = stable_cumsum(y_true * weight)[threshold_idxs]
        positives = stable_cumsum(y_true)[threshold_idxs] # Note that the number of positive should be computed differently
        if sample_weight is not None:
            # express fps as a cumsum to ensure fps is increasing even in
            # the presence of floating point errors
            fps = stable_cumsum((1 - y_true))[threshold_idxs]
        else:
            fps = 1 + threshold_idxs - tps
        return fps, tps, y_score[threshold_idxs], positives

'''
计算stauc指标
'''
def calculate_stauc_score(model_folder: str ,  specific_peoch : list = [] , popr = False): # scenes:List[str]=None
    
    def flat_list(list_):
        if isinstance(list_, (np.ndarray, np.generic)):
            # to be retrocompatible
            return list_
        return list(itertools.chain(*list_))
    
    def read_video_boxes(labels,begin,end):
        frames_boxes = []
        for frame_data in labels[begin:end]:
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            frames_boxes.append(np.array(boxes))
        return frames_boxes

    def post_process(outputs, kernel_size = 31):
        import scipy.signal as signal
        post_outputs = []
        for preds in outputs:
            ks = len(preds) if len(preds)%2!=0 else len(preds)-1 # make sure odd
            now_ks = min(kernel_size,ks)
            preds = signal.medfilt(preds, kernel_size = now_ks)     
            scores_each_video = normalize_video(np.array(preds))
            post_outputs.append(scores_each_video)
        return post_outputs

    file_name = os.path.join(model_folder,'evaluation',f'stauc_eval.txt')
    for epoch in specific_peoch:
        pkl_path = os.path.join(model_folder,'eval',f'results-{epoch}.pkl')
        NF = 4
        assert os.path.exists(pkl_path),f'{pkl_path} is not existed'
        with open(pkl_path, 'rb') as f:
            content = pickle.load(f)
        assert 'obj_targets' in content ,f'instance anomal info is not in {pkl_path}'
        gt_targets = flat_list(content['targets'])
        L = len(gt_targets)  # for debug
        # L = 5
        if popr:
            content['outputs'] = post_process(content['outputs'])   
        gt_targets = gt_targets[:L]
        gt_targets = flat_list(content['targets'][:L])
        pred_scores = flat_list(content['outputs'][:L])
        pred_bboxes_scores = flat_list(content['obj_outputs'][:L])
        scenes = content['video_name'][:L]
        gt_bboxes, pred_bboxes = [], []  
        for scene in tqdm(scenes,desc="Scenes : "):
            yolo_path = os.path.join(DATA_FOLDER / 'yolov9', scene+'.json')
            gt_path = os.path.join(DATA_FOLDER / 'annotations',scene+'.json')
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
                        
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            num_frames = yolo_data['num_frames']
            ori_yolo_boxes = read_video_boxes(yolo_labels,NF-1,num_frames-1)
            ori_gt_boxes = read_video_boxes(gt_labels,NF-1,num_frames-1)
            gt_bboxes.append(ori_gt_boxes)
            pred_bboxes.append(ori_yolo_boxes)

        gt_bboxes = flat_list(gt_bboxes)
        pred_bboxes = flat_list(pred_bboxes)
        assert len(gt_bboxes) == len(gt_targets),f' frame_bbox not equal to frame'

        stauc_metrics = STAUCMetrics()
        stauc_metrics.update(gt_targets,gt_bboxes,pred_scores,pred_bboxes,pred_bboxes_scores)
        stauc, auc, gap = stauc_metrics.get_stauc()

        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(file_name, 'a') as file:
            file.write("\n######################### EPOCH #########################\n")
            # 写入评估指标
            file.write(f'{formatted_time}')
            file.write(f'\npost-process is {popr}\n')
            file.write(f"\n----------------STAUC eval on epoch = {epoch}----------------\n")
            file.write("[Correctness] total frame = %d\n" % (len(gt_bboxes)))
            file.write("              ST-AUC = %.5f\n" % (stauc))
            file.write("              AUC = %.5f\n" % (auc))
            file.write("              gap = %.5f\n" % (gap))
            file.write("\n")            
        print(f'{stauc=:.2f} {auc=:.2f} {gap=:.2f}')
        custom_print(f'write to {file_name}')


def parse_args():
    parser = argparse.ArgumentParser(description="Script for STAUC evaluation.")

    parser.add_argument(
        '--model_folder',
        type=str,
        required=True,
        help='Path to the model folder.'
    )
    
    parser.add_argument(
        '--specific_epoch',
        type=str,
        required=True,
        help='List of specific epochs to process.'
    )
    
    parser.add_argument(
        '--popr',
        action='store_true',
        help='Flag to enable or disable the popr option (default: False).'
    )
    
    args = parser.parse_args()
    args.specific_epoch = [int(epoch) for epoch in args.specific_epoch.split(',')]
    return args

if __name__ == "__main__":
    args = parse_args()
    calculate_stauc_score(args.model_folder, args.specific_epoch, args.popr)

if __name__ == '__main__2':
    # 创建一个简单的STAUCMetrics对象
    stauc_metrics = STAUCMetrics()
    num_frames = 10
    gt_labels = [1 if i % 2 == 0 else 0 for i in range(num_frames)]  # 奇数帧为正常帧，偶数帧为异常帧
    pred_scores = [
        np.array([0.8]) if gt_labels[i] == 1 else np.array([0.5])
        for i in range(num_frames)
    ]

    gt_bboxes = [
        np.array([[50.0, 50, 100, 100]]) if gt_labels[i] == 1 else np.array([])
        for i in range(num_frames)
    ]

    pred_bboxes = [
        np.array([[48.0, 48, 102, 102]]) if gt_labels[i] == 1 else np.array([])
        for i in range(num_frames)
    ]

    pred_bboex_scores = [
        np.array([0.8]) if gt_labels[i] == 1 else np.array([0.5])
        for i in range(num_frames)
    ]

    stauc_metrics.update(gt_labels,gt_bboxes,pred_scores,pred_bboxes,pred_bboex_scores)

    # 计算STAUC值
    stauc, auc, gap = stauc_metrics.get_stauc()

    # 输出结果
    print(f"STAUC: {stauc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Score Gap: {gap:.4f}")
