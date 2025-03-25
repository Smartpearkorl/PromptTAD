import glob
import os
from tqdm import tqdm 
import cv2
import re
import json
import numpy as np
import copy
from PIL import Image, ImageDraw, ImageFont
from runner import DADA_FOLDER , FONT_FOLDER

'''
给DADA构造dota数据集的metadata
1. scene.json
2. etadata_train.json, metadata_test.json, metadata_val.json, 
3. 查看yolo和gt的point
4. 划分 ego-involves and non-ego-involves
'''


class DADA_Prepare():
    def __init__(self):
        self.folder = DADA_FOLDER
        self.videos_pth = DADA_FOLDER / "videos/*"
        self.image_folder = os.path.join(self.folder, 'frames')
        self.yolo_folder = os.path.join(self.folder, 'yolov9')
        self.gt_folder = os.path.join(self.folder,'annotations')
        self.meta_folder = os.path.join(self.folder,'metadata')
        pass
    
    def video2image(self,):
        def extract_numbers(file_path):
            match = re.search(r"images_(\d+)_(\d+).avi", file_path)
            if match:
                return int(match.group(1)), int(match.group(2))
            return None
        
        def single_process(video_pth, save_pth):
            vc = cv2.VideoCapture(video_pth)  #
            c = 0
            rval = vc.isOpened()
            while rval:  #
                c = c + 1
                rval, frame = vc.read()
                if rval:
                    name = os.path.join(save_pth, str(c).zfill(4) + '.jpg')
                    cv2.imwrite(name, frame)
                    cv2.waitKey(1)
                else:
                    break
            vc.release()
        
        all_videos = glob.glob(self.videos_pth)
        all_videos = sorted(all_videos,key=extract_numbers)
        for video in tqdm(all_videos):
            video_name = os.path.basename(video)
            temp = video_name.split('_')
            cc, category, folder = temp[0], temp[1], temp[2].split('.')[0]
            scene_name = category + '_' + folder
            save_images_pth = os.path.join(self.folder, 'frames', scene_name)
            os.makedirs(save_images_pth,exist_ok=True)         
            single_process(video, save_images_pth)

    '''
    生成DoTA标签格式的metadata文件:
    1. scene.json
    2.metadata_train.json, metadata_test.json, metadata_val.json, 
    '''
    def creat_metadata(self,data_type='val'): # 'train' 'val' 'test'  
        os.makedirs(os.path.join(self.folder,'annotations'),exist_ok=True)
        meta_folder = os.path.join(self.folder,'metadata')
        frame_folder = os.path.join(self.folder,'frames')
        datafile = os.path.join(meta_folder,f'{data_type}_file.json')
        ego_involved_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                            '12', '13', '14', '15', '16', '17', '18', '51', '52']
        with open(datafile, 'r') as f:
            ind = json.load(f)
        scene_index = [int(x[1]) for x in ind]
        scene_name = [(x[0][0], x[0][1], x[0][0]+'_'+x[0][1]) for x in ind]
        no_anomaly_scene = []
        all_data = {}
        for (scene_data,scene_ind) in tqdm(zip(scene_name,scene_index), total=len(scene_name)):
            class_id , video_id , scene = scene_data
            single_json_path = os.path.join(self.folder,'annotations',f'{scene}.json')
            # if os.path.exists(single_json_path):
            #     continue
            frame_list = os.listdir(os.path.join(frame_folder,scene))
            frame_list = sorted(frame_list)
            anomaly_txt_path = os.path.join(self.folder, 'DADA_dataset', class_id, video_id, 'attach', f'{video_id}_anomaly.txt'  )
            coord_txt_path = os.path.join(self.folder, 'DADA_dataset', class_id, video_id, 'attach', f'{video_id}_coordinate.txt')
           
            if not os.path.exists(anomaly_txt_path):
                anomaly_txt_path = os.path.join(self.folder, 'DADA_dataset', class_id, video_id, 'attach', f'anomaly.txt')
                if not os.path.exists(anomaly_txt_path):
                    no_anomaly_scene.append(scene)
                    continue
            
            if not os.path.exists(coord_txt_path):
                coord_txt_path = os.path.join(self.folder, 'DADA_dataset', class_id, video_id, 'attach', f'coordinate.txt')
                if not os.path.exists(anomaly_txt_path):
                    no_anomaly_scene.append(scene)
                    continue

            frame_labels = []
            # label
            with open(anomaly_txt_path, 'r') as labels_file:
                labels = [line.strip() for line in labels_file.readlines()]
            # coordinate
            with open(coord_txt_path, 'r') as coordinate_file:
                coordinates = [line.strip() for line in coordinate_file.readlines()]

            assert len(labels) == len(coordinates), f'{scene} has different number of labels and coordinates'

            label_list , anomaly , cordinate = [] , [], [] 
            for i,(label,coor) in enumerate(zip(labels,coordinates)):
                label = int(label)
                x,y = int(coor.split(',')[0]), int(coor.split(',')[1])
                label_list.append(label)
                objects = []
                # anomaly
                if label == 1:
                    anomaly.append(i)
                    if x >0 or y>0:
                        objects.append({'point':[x,y]})
                accident_name = "normal" if label == 0 else "anomaly"
                image_path = os.path.join('frames',scene,str(i+1).zfill(4)+'.jpg')
                assert image_path.split('/')[-1] == frame_list[i]
                
                tmp_labels = {"frame_id":i,
                              "image_path": image_path,
                              "accident_id": label,
                              "accident_name": accident_name,
                              "objects": objects,
                              }
                frame_labels.append(tmp_labels)

            if len(anomaly):       
                anomaly_start, anomaly_end = anomaly[0], anomaly[-1]
                num_frames = len(label_list)
            else:
                no_anomaly_scene.append(scene)
                anomaly_start, anomaly_end = -1, -1
                num_frames = len(label_list)   
            
            ego_involve = class_id in ego_involved_list
            # single_scene.json
            single_data = {
                "video_name": scene,
                "channel": "xxxx",
                "num_frames": num_frames,
                "ignore": False,
                "ego_involve": ego_involve,
                "night": False,
                "anomaly_start": anomaly_start,
                "anomaly_end": anomaly_end,
                "accident_id": int(class_id),
                "accident_name": "to check",
                "labels": frame_labels,
            }
            # single_scene.json
            with open(single_json_path,'w') as f:
                json.dump(single_data, f, indent=4)

            all_data[scene] = {"scene_index":scene_ind,
                               "anomaly_start": anomaly_start,
                               "anomaly_end": anomaly_end,
                               "num_frames":num_frames,
                               "subset":type,
                               }
        all_json_path = os.path.join(meta_folder, f'metadata_{type}.json')
        with open(all_json_path,'w') as f:
            json.dump(all_data, f, indent=4)
        print(no_anomaly_scene)
    

    def read_video_boxes_points(self,labels,begin,end,type='bbox'):
        if type == 'bbox':
            frames_boxes = []
            for frame_data in labels[begin:end]:
                boxes = [ obj['bbox'] for obj in frame_data['objects'] if 'bbox' in obj ]    
                frames_boxes.append(np.array(boxes))
            return frames_boxes
        
        elif type == 'point':
            frames_points = []
            for frame_data in labels[begin:end]:
                boxes = [ obj['point'] for obj in frame_data['objects'] ]
                frames_points.append(np.array(boxes))
            return frames_points
        
        elif type == 'distance':
            frames_distance = []
            for frame_data in labels[begin:end]:
                distances = [ obj['distance'] for obj in frame_data['objects'] if 'distance' in obj]
                frames_distance.append(np.array(distances))
            return frames_distance
        
    '''
    在图像上画框, 并可能有每个框的特有信息： 用于 instance anomal score
    '''
    def draw_boxes_with_info(self, image, boxes_list, set_color = None, info_list=None , special_index=None):
        N,h,w,c = image.shape
        box_color  = ['red', 'blue', 'black' ,'yellow' , 'green'] 
        text_color = ['red', 'green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        if set_color is not None:
            box_color = [set_color]*5
            text_color = [set_color]*5

        font = ImageFont.truetype(FONT_FOLDER,25)
        info_list = info_list if info_list else [[] for _ in range(N)]
        special_index = special_index if special_index else [[] for _ in range(N)]
        image_with_boxes = []
        for i in range(N):
            image_pil = Image.fromarray(image[i])
            draw = ImageDraw.Draw(image_pil)
            # pad empty when no info 
            if len(info_list[i])==0:
                info_list[i] = [[] for _ in range(boxes_list[i].shape[0])]
            
            for box_ind,(box,info) in enumerate(zip(boxes_list[i],info_list[i])):
                bbox_color = box_color[1] if box_ind in special_index[i] else box_color[0]
                draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=bbox_color, width=4)
                if len(info):
                    draw.text( (box[0],box[1]), f"{info}",fill=text_color[0], font=font) 
            image_with_boxes.append(np.array(image_pil))
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes
    
    def draw_point(self, image, points_list):
        N,h,w,c = image.shape  
        image_with_points = []
        for i in range(N):
            image_pil = Image.fromarray(image[i])
            draw = ImageDraw.Draw(image_pil)
            for point in points_list[i]:  
                size = 10
                left_up_point = (max(0,point[0] - size), max(0,point[1] - size))
                right_down_point = (min(w,point[0] + size), min(h,point[1] + size))
                # draw.rectangle([left_up_point, right_down_point], outline=(0, 0, 255))
                draw.ellipse([left_up_point, right_down_point], fill=(0, 0, 255))
            image_with_points.append(np.array(image_pil))
        image_with_points =  np.array(image_with_points)
        return image_with_points
    
    '''
    gt point 与 yolo 匹配的bbox 
    '''
    def find_closest_bbox_index(self, point, bboxes, max_distance=100):  
        def is_point_in_bbox(point, bbox):  
            """检查点是否在bbox内"""  
            x, y = point  
            x_min, y_min, x_max, y_max = bbox  
            return x_min <= x <= x_max and y_min <= y <= y_max  

        def calculate_center(bbox):  
            """计算bbox的中心点"""  
            x_min, y_min, x_max, y_max = bbox  
            center_x = (x_min + x_max) / 2  
            center_y = (y_min + y_max) / 2  
            return np.array([center_x, center_y])  

        def euclidean_distance(point1, point2):  
            """计算两个点间的欧氏距离"""  
            return np.linalg.norm(point1 - point2)  
        
        """找到距离point最近的bbox的索引"""  
        min_distance = float('inf')  
        closest_index = -1  

        for i, bbox in enumerate(bboxes):  
            if is_point_in_bbox(point, bbox):  
                center = calculate_center(bbox)  
                distance = euclidean_distance(point, center)
                
                # 计算 bbox 对角线长度  
                x_min, y_min, x_max, y_max = bbox  
                diagonal_distance = euclidean_distance(np.array([x_min, y_min]), np.array([x_max, y_max]))  
                cmp_distance = max(0.5*diagonal_distance,max_distance)
                if distance < min_distance and distance < cmp_distance:  
                    min_distance = distance  
                    closest_index = i  
        
        return [closest_index,min_distance] # index:distance  


    def add_boxes_to_labels(self,data_type='val'):
       
        with open(os.path.join(self.meta_folder,f'metadata_{data_type}.json'), 'r') as f:
            scenes = list(json.load(f).keys())

        for scene in tqdm(scenes,desc="Scenes : "):
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            image_path = os.path.join(self.image_folder,scene)
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"     
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            ori_gt_points = self.read_video_boxes_points(gt_labels,anomaly_start,anomaly_end,type='point')
            ori_yolo_boxes = self.read_video_boxes_points(yolo_labels,anomaly_start,anomaly_end,type='bbox')
            match_infoes = []
            for pnt,bbox in zip(ori_gt_points,ori_yolo_boxes):
                if len(pnt)>0:
                    match_infoes.append(self.find_closest_bbox_index(pnt[0],bbox))
                else:
                    match_infoes.append([-1,float('inf')])
            # update gt
            gt_labels_update = copy.deepcopy(gt_labels) 
            for i,(match_info, frame_bbox) in enumerate(zip(match_infoes,ori_yolo_boxes)):
                ind , dis = match_info
                if ind!= -1:
                   px, py = gt_labels_update[anomaly_start+i]['objects'][0]['point']
                   # ori_gt_points[i].shape == (1,2)
                   assert px == ori_gt_points[i][0][0] and py == ori_gt_points[i][0][1], f'{scene} {anomaly_start+i} mismatch point'   
                   gt_labels_update[anomaly_start+i]['objects'][0]['bbox'] = frame_bbox[ind].tolist()
                   gt_labels_update[anomaly_start+i]['objects'][0]['distance'] = dis

            gt_data['labels'] =  gt_labels_update
            with open(gt_path,'w') as f:
                json.dump(gt_data, f, indent=4)


    '''
    查看yolo和gt的point
    '''
    def check_point_box(self,data_type='test',check_type='before'):
        if check_type=='before':
            save_folder = os.path.join(self.folder,'debug','check_point_box')
        elif check_type=='after':
            save_folder = os.path.join(self.folder,'debug','check_point_box_after')
        os.makedirs(save_folder,exist_ok=True)
        with open(os.path.join(self.meta_folder,f'metadata_{data_type}.json'), 'r') as f:
            scenes = list(json.load(f).keys())

        scenes = scenes[::30]
        for scene in tqdm(scenes,desc="Scenes : "):
            save_path = os.path.join(save_folder,scene)
            # if os.path.exists(save_path):
            #     continue
            os.makedirs(save_path,exist_ok=True)
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            image_path = os.path.join(self.image_folder,scene)
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"     
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(anomaly_start,anomaly_end)]
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            ori_gt_points = self.read_video_boxes_points(gt_labels,anomaly_start,anomaly_end,type='point')
            ori_yolo_boxes = self.read_video_boxes_points(yolo_labels,anomaly_start,anomaly_end,type='bbox')
            # add box to gt 前检测
            if check_type== 'before':
                match_infoes = []
                for pnt,bbox in zip(ori_gt_points,ori_yolo_boxes):
                    if len(pnt)>0:
                        match_infoes.append(self.find_closest_bbox_index(pnt[0],bbox))
                    else:
                        match_infoes.append([-1,float('inf')])

                # 添加info list
                info_list, match_ind = [], []
                for match_info, frame_bbox in zip(match_infoes,ori_yolo_boxes):
                    ind , dis = match_info
                    N_obj = frame_bbox.shape[0]
                    info = [[] for x in range(N_obj)]
                    if ind != -1:
                        info[ind] = f'{dis:.2f}'
                        match_ind.append([ind])
                    else:
                        match_ind.append([])
                    info_list.append(info)

                frames = self.draw_point(frames,ori_gt_points)
                frames = self.draw_boxes_with_info(frames,ori_yolo_boxes, info_list = info_list , special_index = match_ind )
            # add box to gt 后检测
            elif check_type== 'after': 
                ori_gt_boxes = self.read_video_boxes_points(gt_labels,anomaly_start,anomaly_end,type='bbox')      
                ori_gt_distance = self.read_video_boxes_points(gt_labels,anomaly_start,anomaly_end,type='distance')
                info_list = []
                for dis,frame_bbox in zip(ori_gt_distance,ori_gt_boxes):
                    info = []
                    # 0个 或 1个 bbox
                    if len(dis)>0:
                        info.append([f'{float(dis):.2f}'])
                    else:
                        info.append([])
                    info_list.append(info)
                frames = self.draw_point(frames,ori_gt_points)
                frames = self.draw_boxes_with_info(frames,ori_yolo_boxes,set_color='red')
                frames = self.draw_boxes_with_info(frames,ori_gt_boxes, set_color='green',info_list = info_list)
                       
            # 储存图像
            for i, frame in enumerate(frames):
                frame_name = yolo_labels[i+anomaly_start]['frame_id']
                image_pil = Image.fromarray(frame)
                image_pil.save(os.path.join(save_path,frame_name))
    
    def multi_points(self,data_type='test'):
        save_folder = os.path.join(self.folder,'debug','check_point_box')
        os.makedirs(save_folder,exist_ok=True)
        with open(os.path.join(self.meta_folder,f'metadata_{data_type}.json'), 'r') as f:
            scenes = list(json.load(f).keys())

        scenes = scenes[::10]
        for scene in tqdm(scenes,desc="Scenes : "):
            save_path = os.path.join(save_folder,scene)
            # os.makedirs(save_path,exist_ok=True)
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            image_path = os.path.join(self.image_folder,scene)
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"     
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            ori_gt_points = self.read_video_boxes_points(gt_labels,anomaly_start,anomaly_end,type='point')
            for point in ori_gt_points:
                if len(point)>1:
                    print(f'more : {scenes}')
                elif len(point)==0:
                    print(f'zero : {scenes}')
            pass
            # frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(anomaly_start,anomaly_end)]
            # frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            
            # ori_yolo_boxes = self.read_video_boxes_points(yolo_labels,anomaly_start,anomaly_end,type='bbox')
            # frames = self.draw_point(frames,ori_gt_points)
            # frames = self.draw_boxes_with_info(frames,ori_yolo_boxes)
            # # 储存图像
            # for i, frame in enumerate(frames):
            #     frame_name = yolo_labels[i+anomaly_start]['frame_id']
            #     image_pil = Image.fromarray(frame)
            #     image_pil.save(os.path.join(save_path,frame_name))

    '''
    划分 ego-involves and non-ego-involves
    '''
    def split_ego_involves(self,data_type = 'test'):
        ego_involved_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                            '12', '13', '14', '15', '16', '17', '18', '51', '52']
        
        with open(os.path.join(self.meta_folder,f'metadata_{data_type}.json'), 'r') as f:
            scenes_data = json.load(f)
            scenes = list(scenes_data.keys())

        ego_involves , non_ego_involves = {} , {} 
        for scene in tqdm(scenes,desc="Scenes : "):
            cls = scene.split('_')[0]
            if cls in ego_involved_list:
                ego_involves[scene] = scenes_data[scene]
            else:
                non_ego_involves[scene] = scenes_data[scene]

        ego_path = os.path.join(self.meta_folder,f'ego_metadata_{data_type}.json')
        with open(ego_path,'w') as f:
            json.dump(ego_involves, f, indent=4)
        no_ego_path = os.path.join(self.meta_folder,f'noego_metadata_{data_type}.json')
        with open(no_ego_path,'w') as f:
            json.dump(non_ego_involves, f, indent=4)

    
if __name__ == '__main__':
    tool = DADA_Prepare()
    type='train'# 'train' 'val' 'test' 
    # tool.creat_metadata(type=type)
    # tool.add_boxes_to_labels(data_type=type)
    # tool.check_point_box(data_type=type ,check_type='after')
    # tool.multi_points()
    # tool.split_ego_involves()

    