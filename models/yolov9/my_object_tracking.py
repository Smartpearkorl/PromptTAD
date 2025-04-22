import sys

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))  

import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import glob
from tqdm import tqdm
import json
import os
from PIL import Image, ImageDraw, ImageFont

class TAD_dataset_deepsort():
    def __init__(
        self,
    ):
        self.yolo_folder = '/data/qh/DoTA/data/yolov9/'
        self.gt_folder = "/data/qh/DoTA/data/annotations/"
        self.image_folder = "/data/qh/DoTA/data/frames/"
        self.train_txt = "/data/qh/DoTA/data/metadata/train_split.txt"
        self.val_txt = "/data/qh/DoTA/data/metadata/val_split.txt" 
        self.minibox_val_txt = "/data/qh/DoTA/data/metadata/minibox_val_split.txt"
        self.train_json = "/data/qh/DoTA/data/metadata/metadata_train.json"
        self.val_json = "/data/qh/DoTA/data/metadata/metadata_val.json"
        self.minibox_val_json = "/data/qh/DoTA/data/metadata/minibox_metadata_val.json"
        # Load the COCO class labels
        classes_path = os.path.join(ROOT,"YOLOv9_DeepSORT/configs/coco.names")
        with open(classes_path, "r") as f:
            self.class_names = f.read().strip().split("\n")
        self.conf = 0.1

    '''
    根据图片制造视频
    '''
    def create_video_from_folders(self, scenes, output_folder, name_prefix = '', fps=10, target_folder=None ,sort_keys = None):   
        if target_folder!=None:
            folder_images = [sorted(glob.glob(os.path.join(glob.escape(folder),'*.jpg')),key=sort_keys) for folder in target_folder]
        else:
            folder_images = [sorted(glob.glob(os.path.join(self.image_folder,scene,'images','*.jpg'))) for scene in scenes]
        example_image = cv2.imread(folder_images[0][0])
        height, width, channels = example_image.shape
        canvas_width = width
        canvas_height = height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
        os.makedirs(output_folder,exist_ok=True)
        for scene_idx, scene in enumerate(scenes):
            output_path =os.path.join(output_folder, name_prefix + f'{scene}.mp4')
            out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
            for image_path in tqdm(folder_images[scene_idx],desc='frames: '):  
                frame = cv2.imread(image_path)  
                if (canvas_height,canvas_width) != (frame.shape[0], frame.shape[1]):  
                    frame = cv2.resize(frame, (canvas_height,canvas_width))  
                out.write(frame)  
        # 清理资源
        out.release()
        cv2.destroyAllWindows()

    def read_video_boxes(self,labels,begin,end):
        frames_boxes = []
        for frame_data in labels[begin:end]:
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            frames_boxes.append(np.array(boxes))
        return frames_boxes

    def run(self, folder):
        # Initialize the DeepSort tracker
        tracker = DeepSort(n_init=1,max_age=1)
        with open(self.val_json) as f:
            meta_datas = json.load(f)

        scenes = list(meta_datas.keys())
        for ind , scene in enumerate(tqdm(scenes,desc="Scenes : ")):
            savefolder = os.path.join(folder,scene)
            os.makedirs(savefolder,exist_ok=True)
            metadata = meta_datas[scene]
            image_path = os.path.join(self.image_folder,scene,'images')
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            yolo_labels = yolo_data['lables']                     
            anomaly_start , anomaly_end , num_frames = metadata['anomaly_start'] , metadata['anomaly_end'] , metadata['num_frames']
            frames_path = [ os.path.join(image_path, yolo_labels[i]['frame_id']) for i in range(num_frames)]
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            # ori_yolo_boxes = self.read_video_boxes(yolo_labels,0,num_frames)
            # Create a list of random colors to represent each class
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(self.class_names), 3)) 
            for index, frame in enumerate(tqdm(frames,desc='images: ')):
                detect = []
                objectdata = yolo_labels[index]
                for det in objectdata['objects']:
                    label, confidence, bbox = det['category ID'], det['confidence'], det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    class_id = int(label)

                    # Filter out weak detections by confidence threshold and class_id

                    if confidence < self.conf:
                        continue

                    detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

                tracks = tracker.update_tracks(detect, frame=frame)

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    class_id = track.get_det_class()
                    x1, y1, x2, y2 = map(int, ltrb)
                    color = colors[class_id]
                    B, G, R = map(int, color)
                    text = f"{track_id} - {self.class_names[class_id]}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                    cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                    cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Apply Gaussian Blur
                    if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

                cv2.imshow('YOLOv9 Object tracking', frame)     
                cv2.waitKey(1)       
                cv2.imwrite(os.path.join(savefolder,f'{index:06d}.jpg'), frame)  # 保存图像文件
                
    def run_video(self,videofolder, savefolder, scene):
        videopath = os.path.join(videofolder,scene+'.mp4')
        os.makedirs(os.path.join(savefolder,'videos'), exist_ok=True)
        image_folder = os.path.join(savefolder,'images',scene)
        os.makedirs(image_folder, exist_ok=True)
        # Initialize the video capture
        cap = cv2.VideoCapture(videopath)           
        if not cap.isOpened():
            print('Error: Unable to open video source.')
            return
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # video writer objects
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        writer = cv2.VideoWriter(os.path.join(savefolder,'videos',f'{scene}.mp4'), fourcc, fps, (frame_width, frame_height))
        # Initialize the DeepSort tracker
        tracker = DeepSort(n_init=1, max_age=1) # n_init=1,max_age=1
        # select device (CPU or GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load YOLO model
        weights = os.path.join(ROOT,'yolov9-c-converted.pt')
        model = DetectMultiBackend(weights=weights,device=device, fuse=True)
        model = AutoShape(model)

        # Load the COCO class labels
        classes_path = os.path.join(ROOT,"YOLOv9_DeepSORT/configs/coco.names")
        with open(classes_path, "r") as f:
            class_names = f.read().strip().split("\n")

        # Create a list of random colors to represent each class
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(class_names), 3)) 
        index = -1
        normal_ids = ['person','bicycle','car','motorcycle','bus','truck']
        noraml_indices = [class_names.index(item) for item in normal_ids if item in class_names]  
        while True:
            index += 1 
            ret, frame = cap.read()
            if not ret:
                break
            # Run model on each frame
            results = model(frame)
            detect = []
            for det in results.pred[0]:
                label, confidence, bbox = det[5], det[4], det[:4]
                x1, y1, x2, y2 = map(int, bbox)
                class_id = int(label)

                # Filter out weak detections by confidence threshold and class_id
                if not class_id in noraml_indices or confidence < self.conf:
                    continue
                detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
                tracks = tracker.update_tracks(detect, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                color = colors[class_id]
                B, G, R = map(int, color)
                text = f"{track_id} - {class_names[class_id]}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Apply Gaussian Blur
                # if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                #     frame[y1:y2, x1:x2] = cv2.GaussianBlur(frame[y1:y2, x1:x2], (99, 99), 3)

            cv2.imshow('YOLOv9 Object tracking', frame)
            writer.write(frame)
            cv2.imwrite(os.path.join(image_folder,f'{index:06d}.jpg'), frame)  # 保存图像文件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and writer
        cap.release()
        writer.release()

if __name__ == '__main__':
    runner = TAD_dataset_deepsort()
    # runner.run("/data/qh/DoTA/data/deepsort/")

    # scenes = ['a_VxrUq9PmA_002682']
    # output_folder = "/data/qh/DoTA/output/debug/visualization/videos/original_video"
    # runner.create_video_from_folders(scenes=scenes, output_folder=output_folder, sort_keys=None)

    videopath = "/data/qh/DoTA/output/debug/visualization/videos/original_video/a_VxrUq9PmA_002682.mp4"
    outputpath = "/data/qh/DoTA/output/debug/visualization/videos/deepsort/a_VxrUq9PmA_002682.mp4"

    videofolder = "/data/qh/DoTA/output/debug/visualization/videos/original_video"
    savefolder = "/data/qh/DoTA/output/debug/visualization/videos/deepsort"
    scene = 'a_VxrUq9PmA_002682'
    runner.run_video(videofolder, savefolder, scene)
    pass
