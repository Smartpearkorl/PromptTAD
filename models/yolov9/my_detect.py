import argparse
import os
import platform
import sys
from pathlib import Path
import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory /home/qh/TDD/PromptTAD/models/yolov9/my_detect.py
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
sys.path.append(str(FILE.parents[2]))
from runner import *
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from tqdm import tqdm
import os
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

import json
class LoadSceneFrames:
    def __init__(self, scene_frames, scene_path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1,batchsize=None):
        self.scene_frames = scene_frames
        self.scene_path = scene_path
        self.img_size = img_size
        self.stride = stride
        self.len_frames = len(scene_frames)
        self.auto = auto
        self.transforms = transforms
        self.batchsize = batchsize if batchsize else self.len_frames


    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count >= self.len_frames:
            raise StopIteration
        imgs = []
        img_trans = []
        text = []
        for i in range(self.batchsize):
            #read image
            file_path = os.path.join(self.scene_path,self.scene_frames[self.count])
            img = cv2.imread(file_path)
            assert  img is not None, f'Image Not Found {self.scene_frames[self.count]}'       
            if self.transforms:
                im = self.transforms(img)
            else:
                im = letterbox(img, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
                im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                im = np.ascontiguousarray(im)  # contiguous
            imgs.append(img)
            img_trans.append(im)
            text.append(self.scene_frames[self.count])
            self.count += 1
        return imgs, img_trans, text

import yaml
def check_boxes():
    json_path = DATA_FOLDER / 'yolov9'
    image_path = DATA_FOLDER / 'frames'
    yaml_path = os.path.join(ROOT,"data/coco.yaml")
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    names = data.get('names', {})
    json_files = [f for f in os.listdir(json_path)]
    for json_file in json_files:
        with open(os.path.join(json_path,json_file), 'r') as f:
            scene_dic = json.load(f)
        
        video_name = scene_dic['video_name'] 
        num_frames = scene_dic['num_frames']
        scene_labels = scene_dic['lables']
        assert len(scene_labels)==num_frames,f'scene_labels is not equal to num_frames'
        for frame in scene_labels:
            frame_id =  frame['frame_id']
            frame_path = os.path.join(image_path,video_name,"images",frame_id)
            img = cv2.imread(frame_path)
            annotator = Annotator(img, line_width=2, example=str(names))
            # category_ID = []
            # confidence = []
            # boxes = []
            for obj in frame['objects']:
                # boxes.append(obj["bbox"])
                # confidence.append(obj["confidence"])
                # category_ID.append(obj["category ID"])
                boxes = obj["bbox"]
                confidence = obj["confidence"]
                category_ID = obj["category ID"]
                label = f'{names[category_ID]} {confidence:.2f}'
                annotator.box_label(boxes, label, color=colors(confidence, True))
            img = annotator.result()
            print()

# check_boxes()
                      
@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold 
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    dataset_type = 'DADA' # 'DoTA' 'DADA' 
    data_type = 'val' # 'train' 'val' 'test'
    source = str(source)
    vis_sample_per = 5 

    # DoTA
    if dataset_type == 'DoTA':
        scenes_split_path = META_FOLDER / "val_split.txt"
        scenes_file_path = DATA_FOLDER / 'frames' 
        scenes_json_path = DATA_FOLDER / 'yolov9'
        vis_save_path = DATA_FOLDER / 'yolo_vis'
        with open(scenes_split_path, 'r') as f:
            scenes = f.read().splitlines()

    # DADA
    elif dataset_type == 'DADA':
        scenes_split_path = DADA_FOLDER / f"metadata/metadata_{data_type}.json" 
        scenes_file_path =  DADA_FOLDER / 'frames'
        scenes_json_path = DADA_FOLDER / 'yolov9' 
        vis_save_path =  DADA_FOLDER / 'yolo_vis' 
        with open(scenes_split_path, 'r') as f:
            scenes = list(json.load(f).keys())

    os.makedirs(scenes_json_path,exist_ok=True) 
    # 加载模型
    device = select_device(device)
    weights = yolov9_c_convertd_weight_path
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    is_visualize = False
    vis_sample_per = 5 
    for scene in tqdm(scenes):
        # 对一个场景
        if dataset_type == 'DoTA':
            scene_path = os.path.join(scenes_file_path,scene,"images")
        else:
            scene_path = os.path.join(scenes_file_path,scene)
        
        save_path = os.path.join(scenes_json_path,scene+".json")

        if os.path.exists(save_path):
            continue

        scene_frames = sorted([f for f in os.listdir(scene_path)])

        # 写入json
        scene_dic = {}
        scene_dic['video_name'] = scene
        scene_dic['num_frames'] = len(scene_frames)
        scene_dic['lables'] = None 

        # length
        if len(scene_frames)>300:
            scene_frames = [scene_frames[i:i+250] for i in range(0, len(scene_frames), 250) ]

        else:
            scene_frames = [scene_frames]

        scene_labels = []
        for frames in scene_frames:        
            dataset = LoadSceneFrames(frames, scene_path = scene_path, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            if is_visualize:
                vis_scene_path = os.path.join(vis_save_path,scene)         
                os.makedirs(vis_scene_path,exist_ok=True)
            for imgs, im, text in dataset:
                
                #load 
                im =  torch.from_numpy(np.stack(im,axis=0)).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                #inference
                pred = model(im, augment=augment, visualize=visualize)
                #NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                assert len(imgs) == len(pred) == len(text), f'prediction number mismatch'
                #per_image_boxes
                for i, det in enumerate(pred):
                    #visualize
                    if is_visualize and i % vis_sample_per == 0:
                        im0 = imgs[i].copy()
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        vis_det = det.clone()
                        vis_det[:, :4] = scale_boxes(im.shape[2:], vis_det[:, :4], im0.shape)
                        for *xyxy, conf, cls in reversed(vis_det):
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        im0 = annotator.result()
                        frame_save_path = os.path.join(vis_scene_path,text[i])
                        cv2.imwrite(frame_save_path, im0)

                    frame_labels = {}
                    frame_labels["frame_id"] =  text[i]
                    frame_objects = []
                    if (len(det)):
                        # Rescale boxes from img_size to imgs size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], imgs[i].shape)
                        # filter by class
                        normal_ids = ['person','bicycle','car','motorcycle','bus','truck']
                        for i in range(det.shape[0]):
                            if names[int(det[i][5].data)] in normal_ids:
                                objetct = {
                                    "category ID":int(det[i][5].data),
                                    "category": names[int(det[i][5].data)],
                                    "bbox":det[i][0:4].tolist(),
                                    "confidence":float(det[i][4].data)
                                }
                                frame_objects.append(objetct)
                    # all objects per image 
                    frame_labels["objects"]=frame_objects
                    # all info per image
                    scene_labels.append(frame_labels)
                
            #video frames 
        scene_dic['lables']=scene_labels
       
        with open(save_path, 'w') as f:
            json.dump(scene_dic, f,indent=4)
            

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
