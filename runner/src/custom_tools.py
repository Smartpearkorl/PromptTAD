### some custom tools ###
from typing import List, Tuple
import os 
import torch
from torch import nn
from typing import List
import json
from PIL import Image, ImageDraw, ImageFont
import matplotlib
# matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap , Normalize
import numpy as np
from tqdm import tqdm as tqdm
import pickle
import random
import re
import cv2
import yaml
import itertools
from datetime import datetime
import glob
from sklearn.metrics import roc_auc_score  

# Custom imports
import sys
from pathlib import Path
FILE = Path(__file__).resolve() # /home/qh/TDD/pama/runner/src/custom_tools.py
sys.path.insert(0, str(FILE.parents[2]))
import os 
os.chdir(FILE.parents[2])

from models.componets import HungarianMatcher
from runner.src.tools import custom_print
from runner.src.metrics import  write_results ,evaluation_on_obj, evaluation , AUC_on_scene , Accuracy_on_scene, normalize_video ,print_results
from runner.src.stauc import STAUCMetrics
from runner import DATA_FOLDER, META_FOLDER, FONT_FOLDER, DEBUG_FOLDER,CHFONT_FOLDER, DADA_FOLDER

'''
对比模型的state: parameters or gradients
'''
class State_Comparison():
    def __init__(
            self,
            folder: str,
            path_1: str,
            path_2: str,
            para_com_path:str = 'para_compare',
            grad_com_path:str = 'grad_compare',
    ):
        super().__init__()
        self.folder = folder
        self.path_1 = path_1
        self.path_2 = path_2
        self.para_com_path = para_com_path
        self.grad_com_path = grad_com_path
        self.para_com_str = '----- Parameter Compare ------'
        self.grad_com_str = '----- Gradient Compare ------'
        custom_print(f'save folder :{self.folder}')
        os.makedirs(os.path.join(folder,para_com_path),exist_ok=True)
        os.makedirs(os.path.join(folder,grad_com_path),exist_ok=True)
    
    '''
    如果有prompt model,需要将其放在state_dict1  
    '''
    def compare_state(self, state_dict1, state_dict2, output_file):
        # 先判断对比的类型: 0 ： 同一个模型。1： state_dict1 -> prompt state_dict2-> baseline or fpn
        type_1 , type_2 = 0 , 0
        for key in state_dict1.keys():
            if 'prompt_model' in key:
                type_1 = 1
                break

        for key in state_dict2.keys():
            if 'prompt_model' in key:
                type_2 = 1
                break
        
        cmp_type = (type_1==1 and type_2==0 )

        with open(output_file, 'w') as f:
            no_diff = []
            no_module = []
            mean_diff = []
            big_error = []
            for key in state_dict1.keys():
                ke_replace = key.replace('prompt_model.backbone', 'model') if cmp_type and 'prompt_model.backbone' in key else key
                if ke_replace in state_dict2:
                    if not torch.equal(state_dict1[key], state_dict2[ke_replace]):
                        p_mean = torch.norm(state_dict1[key])
                        m_mean = torch.norm(state_dict2[ke_replace])
                        percent_1, percent_2 = (p_mean - m_mean)/p_mean*100, (p_mean - m_mean)/m_mean*100
                        diff_str = (f"{key}\n state_1 mean : {p_mean:.5e}  state_2  mean : {m_mean:.5e}  mean_diff: {(p_mean - m_mean):.5e} "
                                    f"Percent_1: {percent_1:.2f} Percent_2: {percent_2:.2f}" )                             
                        if abs(percent_1) > 30 or abs(percent_2) > 30 :
                            big_error.append(diff_str)
                        mean_diff.append(diff_str)                                
                        if state_dict1[key].shape == state_dict2[ke_replace].shape:
                            diff = state_dict1[key] - state_dict2[ke_replace]
                            f.write(f"{key}")
                            f.write(f"state_1 mean : {p_mean:.5e} ; state_2 mean : {m_mean:.5e}; mean_diff: {(p_mean - m_mean):.5e} "
                                    f"Percent_1: {percent_1:.2f} Percent_2: {percent_2:.2f}\nDifference: \n{diff}\n\n") 
                                    
                        else:
                            f.write(f"Difference:\n {p_mean - m_mean}\n\n")
                    else:
                        no_diff.append(f'{key}')
                else:
                    no_module.append(f'{key}')
                    f.write(f"Parameter {key} is not found in the second state_dict\n") 

            f.write(f"\n\n--- no diff module ---\n\n")
            for key in no_diff:
                f.write(f"{key}\n\n")
            
            f.write(f"\n\n--- no module ---\n\n")
            for key in no_module:
                f.write(f"{key}\n\n")
            
            f.write(f"\n\n--- mean diff ---\n\n")
            for key in mean_diff:
                f.write(f"{key}\n\n")
            
            f.write(f"\n\n--- big mean diff ---\n\n")
            for key in big_error:
                f.write(f"{key}\n\n")
    
    def compare_para(self,epoch_1,epoch_2,save_name):
        ex_path = os.path.join(self.path_1,"checkpoints",f'model-{epoch_1}.pt')
        ey_path = os.path.join(self.path_2,"checkpoints",f'model-{epoch_2}.pt')
        custom_print(f'{self.para_com_str}')
        custom_print(f'begin')
        c_ex = torch.load(ex_path)
        c_ey = torch.load(ey_path)
        model_state_dict_ex = c_ex['model_state_dict']
        model_state_dict_ey = c_ey['model_state_dict']
        save_path = os.path.join(self.folder,self.para_com_path,save_name)
        self.compare_state(model_state_dict_ex, model_state_dict_ey, save_path)
        custom_print(f'save to {save_path}')
        custom_print(f'end')
        custom_print(f'{self.para_com_str}')
    
    def compare_grad(self,epoch_1,epoch_2,save_name):
        ex_path = os.path.join(self.path_1,"checkpoints",f'model-{epoch_1}.pt')
        ey_path = os.path.join(self.path_2,"checkpoints",f'model-{epoch_2}.pt')
        custom_print(f'{self.grad_com_str}')
        custom_print(f'begin')
        c_ex = torch.load(ex_path)
        c_ey = torch.load(ey_path)
        model_state_dict_ex = c_ex['grad']
        model_state_dict_ey = c_ey['grad']
        save_path = os.path.join(self.folder,self.para_com_path,save_name)
        self.compare_state(model_state_dict_ex, model_state_dict_ey,save_path)
        custom_print(f'save to {save_path}')
        custom_print(f'end')
        custom_print(f'{self.grad_com_str}')

'''
对比模型参数或者梯度
'''
if __name__ == "__main__":
    pass
    # '''
    # "/data/qh/DoTA/output/Prompt/standart_fpn_256/"
    # "/data/qh/DoTA/output/Prompt/box_loss_weight=0.0002_flip/"
    # "/data/qh/DoTA/output/Prompt/box_loss_weight=0.0002/"
    # '''
    # folder = '/data/qh/DoTA/output/debug/' 
    # path_1 = "/data/qh/DoTA/output/Prompt/box_loss_weight=0.0002_flip/"
    # path_2 = "/data/qh/DoTA/output/Prompt/box_loss_weight=0.0002/"
    # ### compare parameters ###  
    # model_1 = "box_loss_flip"
    # model_2 = "box_loss"
    # epoch_1 = 200
    # epoch_2 = 200
    # cmp_name = f'{model_1}_epoch_{epoch_1}_vs_{model_2}_epoch_{epoch_2}.txt'
    # cmp = State_Comparison(folder,path_1,path_2)
    # cmp.compare_para(epoch_1,epoch_2,cmp_name)
    # custom_print()

'''
定义一些文件路径
'''
class PathConfig:      
    def __init__(self):  
        self.yolo_folder = DATA_FOLDER / 'yolov9'  
        self.gt_folder = DATA_FOLDER / 'annotations'  
        self.image_folder = DATA_FOLDER / 'frames'  
        self.train_txt = META_FOLDER / "train_split.txt"  
        self.val_txt = META_FOLDER / "val_split.txt"  
        self.minibox_val_txt = META_FOLDER / "minibox_val_split.txt"  
        self.train_json = META_FOLDER / "metadata_train.json"  
        self.val_json = META_FOLDER / "metadata_val.json"  
        self.minibox_val_json = META_FOLDER / "minibox_metadata_val.json"  

'''
一些工具类
1. 根据图片制造视频
2. 将测试集中场景分类(ego/non-ego/per-class)存到json文件中
3. 统计所有训练集下coco标签的数量,统计每个类别的比例
4. 记录异常coco类所在图片的位置
5. 画出异常coco的照片
6. 更新yolov9的框
7. kill所有用到nvidia的进程
8. 删掉'results-200_rank1.pkl' 等中间pkl

'''
class ProcessorUtils(PathConfig):

    def __init__(self) -> None:
        super().__init__()
    
    '''
    根据图片制造视频:
    args : scenes : List[str] : 场景名字
           output_folder : str : 输出文件夹 , 储存路径  output_folder/scene.mp4
           target_folder: 读取图像的文件夹, 默认为None, 此时读取 self.image_folder/scene/images下的图片
           sort_keys: 读取文件夹的排序方式
    '''
    def create_video_from_folders(self, scenes, output_folder, fps=10, target_folder=None ,sort_keys = None):   
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
            output_path =os.path.join(output_folder,f'{scene}.mp4')
            out = cv2.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
            for image_path in tqdm(folder_images[scene_idx],desc='frames: '):  
                frame = cv2.imread(image_path)  
                if (canvas_height,canvas_width) != (frame.shape[0], frame.shape[1]):  
                    frame = cv2.resize(frame, (canvas_height,canvas_width))  
                out.write(frame)
                print(f'video saved at {output_folder}')
        out.release()
        cv2.destroyAllWindows()

    '''
    将测试集中场景分类: involve/non-involve and  ego+class / non-ego+class 
    储存的路径： META_FOLDER /"class_aware/class_aware.json" 
    '''
    def split_class_set(self,):
        anomalies = [
            'start_stop_or_stationary',  # 1 ST
            'moving_ahead_or_waiting',  # 2 AH
            'lateral',  # 3 LA
            'oncoming',  # 4 OC
            'turning',  # 5 TC
            'pedestrian',  # 6 VP
            'obstacle',  # 7 VO
            'leave_to_right',  # 8 OO-r
            'leave_to_left',  # 9 OO-l
            'unknown',  # 10 UK
            'out_of_control', # 11 OO Combine from OO-r and OO-l
            ]
        with open(self.val_txt,'r') as f:
            scenes =  f.read().split('\n')
        save_path = str(META_FOLDER /"class_aware/class_aware.json")
        save_data = {}
        save_data['involve ego'] = []
        save_data['non-involve ego'] = []
        save_data['per-class']  = {**{f'ego_{x}':[] for x in anomalies} , **{f'no-ego_{x}':[] for x in anomalies}}

        for scene in tqdm(scenes,desc='scenes: '):
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            ego_involve, accident_id, accident_name  = gt_data['ego_involve'] ,gt_data['accident_id'],gt_data['accident_name']
            assert anomalies[int(accident_id)-1] == accident_name, f'{scene=} accident is not matched {anomalies[int(accident_id)-1]} {accident_name}'
            # ego involved or not |  per class
            if ego_involve:
                save_data['involve ego'].append(scene)
                save_data['per-class'][f'ego_{accident_name}'].append(scene)
            else:
                save_data['non-involve ego'].append(scene)
                save_data['per-class'][f'no-ego_{accident_name}'].append(scene)
        # combine oo
        save_data['per-class']['ego_out_of_control'] = save_data['per-class']['ego_leave_to_right']+save_data['per-class']['ego_leave_to_left']
        save_data['per-class']['no-ego_out_of_control'] = save_data['per-class']['no-ego_leave_to_right']+save_data['per-class']['no-ego_leave_to_left']    
        with open(save_path,'w') as f:
            json.dump(save_data, f, indent=4)


    '''
    统计所有训练集下coco标签的数量,统计每个类别的比例
    储存位置: META_FOLDER / 'yolov9_files' / 'coco_class_count_update.txt'
    '''
    def yolo_label_count(self,):
        coco_path = './runner/src/coco.yaml'
        with open(coco_path,'r') as f:
            coco = yaml.safe_load(f)
        classes = coco['names']
        count = [0]*len(classes)
        json_path = os.listdir(self.yolo_folder)
        for path in tqdm(json_path):
            path_ = os.path.join(self.yolo_folder,path)
            with open(path_, 'r') as f:
                yolo_data = json.load(f)
            labels = yolo_data['lables']
            for frame in labels:
                for obj in frame['objects']:
                    count[obj["category ID"]] = count[obj["category ID"]]+1

        save_path =  META_FOLDER / 'yolov9_files' / 'coco_class_count_update.txt'
        all_num = sum(count)
        with open(save_path,'w') as f: 
            for k,v in classes.items():
                f.write(f'{k:>2}: {v:>20} {count[k]:>6} {(count[k]/all_num):.2f}\n')
        print(f'write to {save_path}')
    
    """
    记录异常coco类所在图片的位置
    储存位置: META_FOLDER / 'yolov9_files' / 'unexpected_object.json'
    """
    def record_unexpected_object(self,):      
        coco_path = './runner/src/coco.yaml'
        with open(coco_path,'r') as f:
            coco = yaml.safe_load(f)
        classes = coco['names']
        json_path = os.listdir(self.yolo_folder)
        normal_ids = ['person','bicycle','car','motorcycle','bus','truck']
        count = [0]*len(classes)
        max_record = 20
        record_data = {x:[] for x in classes.values()}
        for path in tqdm(json_path):
            path_ = os.path.join(self.yolo_folder,path)
            with open(path_, 'r') as f:
                yolo_data = json.load(f)
            scene , labels  = yolo_data['video_name'] ,yolo_data['lables']  
            for frame_index, frame in enumerate(labels):
                for obj_index, obj in enumerate(frame['objects']):         
                    now_category , now_ID = obj['category'] , obj["category ID"]
                    if now_category not in normal_ids and count[now_ID] < max_record:
                        count[now_ID] = count[now_ID] + 1
                        record_data[now_category].append([scene,frame_index,obj_index])

        save_path =  META_FOLDER / 'yolov9_files' / 'unexpected_object.json'
        with open(save_path,'w') as f: 
            json.dump(record_data, f, indent=4)
        print(f'write to {save_path}')

    '''
    画出异常coco的照片: 
    args: json_path 中记录了异常coco类所在图片的位置, 来自 record_unexpected_object
          max_frame: 某个类别存储的最大帧数
          save_folder: 储存的目录
    
    '''
    def plot_unexpect_class(self, json_path , save_folder, max_frame = 5):

        image_folder = self.image_folder 
        yolo_folder = self.yolo_folder
        font = FONT_FOLDER
        
        def plot_image(scene,frame_index,obj_index,save_path):
            yolo_path = os.path.join(yolo_folder,scene+'.json')
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            labels = yolo_data['lables']  
            frame_path = os.path.join(image_folder,scene,'images',labels[frame_index]['frame_id'])
            frame = np.asarray(Image.open(frame_path))
            box = labels[frame_index]['objects'][obj_index]['bbox']
            category = labels[frame_index]['objects'][obj_index]['category']
             
            image_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(image_pil)
            draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline='red', width=4)
            draw.text( (box[0],box[1]), f"{category}", fill='red', font=font) 
            image_pil.save(save_path)

        with open(json_path,'r') as f:
            json_data = json.load(f)

        for key,value in tqdm(json_data.items()):
            if len(value)>0:
                sub_path = os.path.join(save_folder,key)
                os.makedirs(sub_path,exist_ok=True)
                if max_frame >= len(value):
                    index_list = [i for i in range(len(value))]
                else:
                    index_list = [int(len(value)*i/max_frame) for i in range(0,max_frame)]
                for i in index_list:
                    scene,frame_index,obj_index = value[i]
                    save_path = os.path.join(sub_path,f'{scene}_frmame_{frame_index}_obj_{obj_index}.png')
                    plot_image(scene,frame_index,obj_index,save_path)


    '''
    更新yolov9的框: 选出特定的类别
    '''
    def update_yolov9_object(self, update_yolo_folder):      
        json_path = os.listdir(self.yolo_folder)
        normal_ids = ['person','bicycle','car','motorcycle','bus','truck']

        for json_name in tqdm(json_path):
            path_ = os.path.join(self.yolo_folder, json_name)
            with open(path_, 'r') as f:
                yolo_data = json.load(f)
            labels = yolo_data['lables']  
            for frame in labels:
                frame['objects'] = [obj for obj in frame['objects'] if obj['category'] in normal_ids]
            
            save_path = os.path.join(update_yolo_folder,json_name)
            with open(save_path,'w') as f: 
                json.dump(yolo_data, f, indent=4)

    '''
    该脚本会清除 当前登陆用户导致的GPU显存不释放问题
    '''
    @staticmethod
    def release_memory():
        result = os.popen("fuser -v /dev/nvidia*").read()
        results = result.split()
        for pid in results:
            os.system(f"kill -9 {int(pid)}")

    '''
    删掉： 'results-200_rank1.pkl' 等中间pkl
    '''
    @staticmethod
    def remove_prefix(main_folder):
        pathes = os.listdir(main_folder)
        for path in pathes:
            model_path = os.path.join(main_folder,path,'eval')
            eva_pathes = os.listdir(model_path)
            for eva_path in eva_pathes:
                if 'rank' in eva_path:
                    os.remove(os.path.join(model_path,eva_path))
                
'''
 关于一些模型后处理的工具
 1. 计算 stauc
 2. 重新生成eval.txt (包括: dota/dada mini ego / non-ego ):
 3. 对比两个模型在每个场景下的AUC
 4. 对比同一个pkl在 正常处理 和 后处理 下每个场景的auc
'''
class Post_Process_Tools(PathConfig):
    def __init__(self) -> None:
        super().__init__()

    '''
    计算stauc指标
    '''
    def calculate_stauc_score(self, model_folder: str ,  specific_peoch : list = [] , popr = False): # scenes:List[str]=None
        
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
                yolo_path = os.path.join(self.yolo_folder,scene+'.json')
                gt_path = os.path.join(self.gt_folder,scene+'.json')
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

    '''
    给定某个模型的文件夹,自动根据eval下pkl,得到eval结果。
    如果给了specific_peoch, 则只eval 指定的epoch。
    '''
    @staticmethod
    def eval_on_epoches(model_folder: str ,  specific_peoch : list = [], post_process = False, val_type = 'all', specific_info = None ,test_kernelsize = False):
        def filter_eval_pkl(files: List[str]) -> Tuple[List[str], List[str]]:
            dic = {}
            pattern = re.compile(r'results-(\d+).pkl')
            for file in files:
                match = pattern.match(file)
                if match:
                    epoch = int(match.group(1))
                    pkl = match.group(0)
                    dic[epoch] = pkl
            dic = dict(sorted(dic.items(), key = lambda x: x[0]))
            epoch = list(dic.keys())
            pkls = list(dic.values())
            return epoch , pkls

        def extract_elements(data_dict, index_list):
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, list):
                    # 如果是由list组成的list
                    result[key] = [value[i] for i in index_list if i < len(value)]
                elif isinstance(value, np.ndarray):
                    # 如果是numpy数组，检查其维度
                    if value.ndim == 1:
                        result[key] = value[index_list] if max(index_list) < value.shape[0] else None
                    else:
                        result[key] = value[index_list] if max(index_list) < value.shape[0] else None
            return result

        eval_folder = os.path.join(model_folder,'eval')
        assert os.path.exists(eval_folder),f'there is no eval folder in {model_folder}'
        eval_epoch_names = os.listdir(eval_folder) 
        # 只要eval 'results-140.pkl' 这个格式的
        if not specific_peoch:
            epoches , pkl_names = filter_eval_pkl(eval_epoch_names)
        else:
            epoches = specific_peoch
            pkl_names = [f'results-{e}.pkl'for e in specific_peoch  ] 

        assert epoches,f'there is no file like results-xxx.pkl in {eval_folder}'
        model_name = model_folder.split('/')[-1] if model_folder.split('/')[-1] else model_folder.split('/')[-2]
        
        # txt 名字： remake_eval_[Template]__[model_name]
        names_template = '' if not post_process else 'popr_' # popr_midfilter popr_meanfilter auc>50
        names_template = names_template if not test_kernelsize else 'ks_' + names_template # popr_midfilter popr_meanfilter auc>50
        names_template = names_template if not specific_info else names_template + specific_info['name'] + '_'
        txt_name = 'remake_eval_' + names_template + model_name +'.txt'

        per_class =True # DADA no perclass
        # val_type
        if val_type == 'mini_box':
            txt_name = f'{val_type}_{txt_name}'
        elif val_type == 'dada_ego':
            txt_name = f'ego_{txt_name}'
            per_class =False
        elif val_type == 'dada_noego':
            txt_name = f'noego_{txt_name}'
            per_class =False
        elif val_type == 'dada_all':
            per_class =False

        os.makedirs(os.path.join(model_folder,'evaluation'),exist_ok=True)
        txt_path = os.path.join(model_folder,'evaluation',txt_name)
        for (epoch, pkl_name) in tqdm(zip(epoches,pkl_names),total=len(epoches),desc="eval process : "):
            path = os.path.join(eval_folder,pkl_name)
            with open(path, 'rb') as f:
                content = pickle.load(f)
      
            # 指定在某些场景下
            if specific_info != None:
                specific_scenes = specific_info['scenes']
                indexs = [idx[0] for x in specific_scenes for idx in [np.where(content['video_name'] == x)[0]] if idx.size > 0]
                # indexs = [ np.where(content['video_name'] == x)[0][0]  for x in specific_scenes]
                for key, value in content.items():
                    if isinstance(value, list):
                        content[key] = [value[i] for i in indexs]
                    elif isinstance(value, np.ndarray):
                        content[key] = value[indexs]

            if not 'all' in val_type :
                # eval 在 mini_box 场景下
                if val_type == 'mini_box':
                    minibox_path = META_FOLDER / "minibox_val_split.txt" 
                    with open(minibox_path,'r') as f:
                        mini_scenes = f.read().split()

                elif val_type == 'dota_ego':
                    with open(META_FOLDER / "metadata_val.json" , 'r') as f:
                        meta = json.load(f)
                    mini_scenes = [k for k, v in meta.items() if 'ego' in v["anomaly_class"]]  

                elif val_type == 'dota_noego':  
                    with open( META_FOLDER / "metadata_val.json" , 'r') as f:
                        meta = json.load(f)
                    mini_scenes = [k for k, v in meta.items() if not ('ego' in v["anomaly_class"]) ]   

                # dada eval 在 ego-involves 场景下
                elif val_type == 'dada_ego':
                    mini_path =  DADA_FOLDER / "metadata/ego_metadata_test.json"
                    with open(mini_path, 'r') as f:
                        mini_scenes = list(json.load(f).keys())
                
                # dada eval 在 ego-involves 场景下
                elif val_type == 'dada_noego':
                    mini_path = DADA_FOLDER / "metadata/noego_metadata_test.json" 
                    with open(mini_path, 'r') as f:
                        mini_scenes = list(json.load(f).keys())
                
                mini_index , miss_scenes = [],[]
                for sce in mini_scenes:
                    index = np.where(sce==content['video_name'])[0]
                    if index.size > 0:
                        mini_index.append(index[0])
                    else:
                        miss_scenes.append(sce)
                content = extract_elements(content,mini_index)

            if test_kernelsize:
                kernel_size = [ i for i in range(1,42,2)]  
                for ks in tqdm(kernel_size):       
                    # write whole model frame anomal eval
                    write_results(txt_path, epoch , *evaluation(FPS=10, **content, post_process = post_process, kernel_size = ks , per_class=per_class ), prefix = f'kernel_size={ks}')
            else:
                write_results(txt_path, epoch , *evaluation(FPS=10, **content, post_process = post_process, per_class=per_class ))
                # instance level eval
                if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
                    write_results(txt_path,epoch,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instacne')
                # frame level eval
                if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
                    write_results(txt_path, epoch , *evaluation(FPS=10, outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')

        custom_print(f'save path to {txt_path}')
         
    '''
    记录所有模型的frame anoaml AUC
    通过模型文件下 remake_eval_xxx.txt统计
    统计格式：
        {
            'model_name': 
                    'epoch' : [20,40,60,...]
                    'f-AUC' : [xx, xx, xx] 
                    ...
        }           
    '''
    @staticmethod
    def Collect_all_AUC(model_list: List[str], save_path: str, post_process = False, val_type = 'all'):
        AUC_data = {}
        if os.path.exists(save_path): 
            with open(save_path,'r') as f:
                AUC_data = json.load(f)
        for model_folder in tqdm(model_list,desc="models : "):
            # 检查remake_eval_xxx.txt是否存在
            model_name = model_folder.split('/')[-1] if model_folder.split('/')[-1] else model_folder.split('/')[-2]
            if model_name in AUC_data:
                continue
            
            # eval_txt path
            if post_process:
                eval_txt_path = os.path.join(model_folder,'evaluation',f'remake_eval_popr_{model_name}.txt')
                if val_type == 'mini_box':
                    eval_txt_path = os.path.join(model_folder,'evaluation',f'{val_type}remake_eval_popr_{model_name}.txt')
          
            else:
                eval_txt_path = os.path.join(model_folder,'evaluation',f'remake_eval_{model_name}.txt')
                if val_type == 'mini_box':
                    eval_txt_path = os.path.join(model_folder,'evaluation',f'{val_type}_remake_eval_{model_name}.txt')
            
            if not os.path.exists(eval_txt_path): 
               Post_Process_Tools.eval_on_epoches( model_folder , post_process = post_process , val_type = val_type)

            with open(eval_txt_path,'r') as f:
                file_data = f.read().split('\n')
            # 得到epoch下AUC结果: 区分三种，'frame eval on epoch' 'prompt frame eval on epoch' 'instacne eval on epoch'
            lables = ['-frame' ,  'prompt frame', 'instacne']
            AUC_patterns = [f'{x} eval on epoch' for x in lables]
            model_level_data = {}
            for lable ,AUC_pattern in zip(lables , AUC_patterns):
                ind = [ i for i,s in enumerate(file_data) if re.findall(AUC_pattern, s)]
                # 准备参数的pattern
                cri_labels = {1:'f-AUC', 2:'PR-AUC', 3:'F1-Score', 4:'F1-Mean', 5:'Accuracy'}
                cri_patterns = [ re.compile(f'{label}\s+= (\d*\.\d+)') for label in cri_labels.values()]
                # class_label = { 9:'normal', 10:'anomaly'}
                cri_patterns.append(re.compile(r'(\d*\.\d+)\s+(\d*\.\d+)\s+(\d*\.\d+)\s+(\d+)'))
                data = {label:[] for label in cri_labels.values()}
                data['epoches'] = []
                sub_labels = ['precision','recall','f1-score','support']
                data['normal'] = {x:[] for x in sub_labels}
                data['anomaly'] = {x:[] for x in sub_labels}
                for i in ind:
                    # 先检查是否是整个测试集,即比较support的数量
                    sp_pattern = re.compile(r'(\d*\.\d+)\s+(\d+)')
                    supports_num = int(re.search(sp_pattern, file_data[i+12]).group(2))
                    if val_type == 'all' and supports_num < 130000:
                            continue                 
                    epo_pattern = re.compile(r'eval on epoch = (\d+)')
                    epoch = int(re.search(epo_pattern, file_data[i]).group(1))
                    data['epoches'].append(epoch)
                    for j,(k,v) in enumerate(cri_labels.items()):
                        data[v].append(float(re.search(cri_patterns[j],file_data[i+k]).group(1)))
                    normal_groups = re.search(cri_patterns[-1],file_data[i+9]).groups()
                    anomal_groups = re.search(cri_patterns[-1],file_data[i+10]).groups()
                    for l,n_d,a_d in zip(sub_labels,normal_groups,anomal_groups):
                        data['normal'][l].append(float(n_d))
                        data['anomaly'][l].append(float(a_d))
                model_level_data[lable] = data
            AUC_data[model_name] = model_level_data 
        with open(save_path,'w') as f:
            json.dump(AUC_data, f, indent=4)
        custom_print(f'write to {save_path}')
        return AUC_data
           
    '''
    对比两个模型在每个场景下的auc,
    返回差别大的、AUC都低的场景
    '''
    @staticmethod
    def compare_auc_on_per_scene(folder_1, epoch_1, folder_2 , epoch_2, save_path, metric_type = 'AUC', post_process=False):
        def format_dict(d):
            if isinstance(list(d.values())[0],float):
                return {k: round(v, 3) for k, v in d.items()} 
            elif isinstance(list(d.values())[0],list):
                return {k: [round(v, 3) for v in lst] for k, lst in d.items()}
        cfg_name_1 = folder_1.split('/')[-1] if folder_1.split('/')[-1] else folder_1.split('/')[-2]
        cfg_name_2 = folder_2.split('/')[-1] if folder_2.split('/')[-1] else folder_2.split('/')[-2]
        pkl_1 = os.path.join(folder_1,"eval",f'results-{epoch_1:02d}.pkl')
        pkl_2 = os.path.join(folder_2,"eval",f'results-{epoch_2:02d}.pkl')
        assert os.path.exists(pkl_1), f'file {pkl_1} is not existed'
        assert os.path.exists(pkl_2), f'file {pkl_2} is not existed'

        if metric_type == 'AUC':
            all_scenes_1 = AUC_on_scene(pkl_1,post_process=post_process)
            all_scenes_2 = AUC_on_scene(pkl_2,post_process=post_process)
        elif metric_type == 'Accuracy':
            all_scenes_1 = Accuracy_on_scene(pkl_1,post_process=post_process)
            all_scenes_2 = Accuracy_on_scene(pkl_2,post_process=post_process)

        scene_keys = list(set(all_scenes_1.keys()) & set(all_scenes_2.keys()))
        combine = {key:[all_scenes_1[key],all_scenes_2[key]] for key in scene_keys}
        big_diff = dict(sorted(combine.items(),key=lambda item: abs(item[1][0] - item[1][1]), reverse=True))
        both_bad = dict(sorted(combine.items(),key=lambda item: (item[1][0] + item[1][1])/2))
        # 增加 差值和平均数
        big_diff ={k: v + [v[0] - v[1]] for k, v in big_diff.items()}
        both_bad ={k: v + [(v[0] + v[1])/2] for k, v in both_bad.items()}
        # 保留3位小数
        big_diff = format_dict(big_diff)
        both_bad = format_dict(both_bad)
        assert len(big_diff) == len(both_bad),"length of big_diff-dict and both_bad-dict is not equal"
        if save_path:
            # 计算键的最大长度
            max_key_length1 = max(len(key) for key in big_diff.keys())
            max_key_length2 = max(len(key) for key in both_bad.keys())
            txt_path = DEBUG_FOLDER / 'per_scenes_compare' / metric_type / save_path
            with open(txt_path, 'w') as file:
                file.write(f'------ {metric_type} on per scene between {cfg_name_1} and {cfg_name_2} ------ \n')
                file.write(f'big difference                                                    both low {metric_type}\n')
                for (key1, value1), (key2, value2) in zip(big_diff.items(), both_bad.items()):
                    # 将键和值写入文件，每一行包含两个键和各自的值列表
                    file.write(f"{key1.ljust(max_key_length1)}: {str(value1).ljust(30)} {key2.ljust(max_key_length2)}: {value2}\n")
            custom_print(f'{metric_type} on per scene between {cfg_name_1} and {cfg_name_2} save to {txt_path}')
        return big_diff, both_bad

    '''
    对比同一个pkl在 正常处理 和 后处理 下每个场景的auc
    '''
    @staticmethod
    def compare_popr_auc_on_per_scene(folder, epochs , save_path):
        def format_dict(d):
            if isinstance(list(d.values())[0],float):
                return {k: round(v, 3) for k, v in d.items()} 
            elif isinstance(list(d.values())[0],list):
                return {k: [round(v, 3) for v in lst] for k, lst in d.items()}
        
        for epoch in tqdm(epochs,desc='epochs: '):
            pkl = os.path.join(folder,"eval",f'results-{epoch:02d}.pkl')
            cfg_name = folder.split('/')[-1] if folder.split('/')[-1] else folder.split('/')[-2]
            assert os.path.exists(pkl), f'file {pkl} is not existed'
            all_scenes_1 = AUC_on_scene(pkl,post_process=False)
            all_scenes_2 = AUC_on_scene(pkl,post_process=True)
            scene_keys = list(set(all_scenes_1.keys()) & set(all_scenes_2.keys()))
            combine = {key:[all_scenes_1[key],all_scenes_2[key]] for key in scene_keys}
            big_diff = dict(sorted(combine.items(),key=lambda item: abs(item[1][0] - item[1][1]), reverse=True))
            both_bad = dict(sorted(combine.items(),key=lambda item: (item[1][0] + item[1][1])/2))
            # 增加 差值和平均数
            big_diff ={k: v + [v[0] - v[1]] for k, v in big_diff.items()}
            both_bad ={k: v + [(v[0] + v[1])/2] for k, v in both_bad.items()}
            # 保留3位小数
            big_diff = format_dict(big_diff)
            both_bad = format_dict(both_bad)
            assert len(big_diff) == len(both_bad),"length of big_diff-dict and both_bad-dict is not equal"
            if save_path:
                # 计算键的最大长度
                max_key_length1 = max(len(key) for key in big_diff.keys())
                max_key_length2 = max(len(key) for key in both_bad.keys())
                txt_path = DEBUG_FOLDER / 'per_scenes_compare' / 'popr' / save_path
                with open(txt_path, 'w') as file:
                    file.write(f'------ auc on per scene {cfg_name} between [normal] and [post process] ------ \n')
                    file.write(f'big difference                                                    both low auc\n')
                    for (key1, value1), (key2, value2) in zip(big_diff.items(), both_bad.items()):
                        # 将键和值写入文件，每一行包含两个键和各自的值列表
                        file.write(f"{key1.ljust(max_key_length1)}: {str(value1).ljust(30)} {key2.ljust(max_key_length2)}: {value2}\n")
                custom_print(f'auc on per scene between {cfg_name} between [normal] and [post process] save to {txt_path}')
        return big_diff, both_bad

    '''
    生成dinov2 emb 搭配main.py
    '''
    @staticmethod
    def save_dinov2_emb(cfg, model, train_sampler, traindata_loader, scaler, optimizer,lr_scheduler, 
                test_sampler=None, testdata_loader=None , index_video = 0 , index_frame= 0 ):
            # NF for vst
        fb = cfg.get('NF',0)
        # clip model use two image in every iteration 
        clip_model =  'clip' in cfg.get('model_type')
        save_path = '/ssd/qh/DoTA/data/dinov2_emb/vitl_scale=4'
        # run in single video
        from tqdm import tqdm as tqdm
        for j, (video_data, data_info, yolo_boxes, frames_boxes, video_name) in enumerate(tqdm(testdata_loader)):
            # prepare data for model and loss func
            video_data = video_data.to(cfg.device, non_blocking=True) # [B,T,C,H,W]
            data_info = data_info.to(cfg.device, non_blocking=True)
            # yolo_boxes : list B x list T x nparray(N_obj, 4)
            yolo_boxes = np.array(yolo_boxes,dtype=object)

            # record whole video data
            B,T = video_data.shape[:2]
            assert B==1,f'Batch_size must be 1'
            emb_path = os.path.join(save_path,video_name[0])
            if os.path.exists(emb_path):
                print(f'{emb_path} is existed')
                continue  
            os.makedirs(emb_path,exist_ok=True)
            # loop in video frames
            batch_size = 8
            split_frames = [video_data[0,i:i + batch_size] for i in range(0, T, batch_size)]
            
            for batch_count,batch_frame in enumerate(split_frames):       
                ret = model(batch_frame, boxes=None, rnn_state=None, frame_state=None)              
                cls_token, patch_token = ret['x_norm_clstoken'], ret['x_norm_patchtokens']
                # loop in batch
                Now_Batch = cls_token.shape[0]
                for index in range(Now_Batch):
                    frame_index = batch_count*batch_size+index
                    emb_path = os.path.join(save_path,video_name[0])
                    np.save(f'{emb_path}/cls_token_{frame_index}.npy',cls_token[index].cpu().numpy())
                    np.save(f'{emb_path}/patch_token_{frame_index}.npy',patch_token[index].cpu().numpy().astype(np.float16))

'''
关于scene的工具:
1. 挑选测试集中异常目标是远景（小框）的场景
2. 图像分为哦10x5的网格,根据bbox网格可视化
2. 将 val 拆分到10x5的grid中, 并写入json
3. 任意集合场景下,bbox面积的分布
4. 任意集合场景下,pkl在指定grid上的AUC(包含对比不同pkl)
5. 挑选测试集中异常目标是远景（小框）的场景
6. 挑选训练集10%的场景(查看过拟合)
'''
class Scene_Aware_Tools(PathConfig):
    def __init__(self, folder : str) -> None:
        super().__init__()
        self.folder = folder
        self.box_threshold = [2667,5115,8507,13792,21904,36676,65353,111277,189312,886728]
        os.makedirs(self.folder,exist_ok=True)

    def read_video_boxes(self,labels,begin,end):
        frames_boxes = []
        for frame_data in labels[begin:end]:
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            frames_boxes.append(np.array(boxes))
        return frames_boxes

    def box_area(self,boxes):
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    def box_center(self,boxes):
        x = (boxes[:, 2] + boxes[:, 0])/2
        y = (boxes[:, 3] + boxes[:, 1])/2
        return np.stack( (x, y) , axis=1 )
    
    '''
    计算box的像素面积: box (x1, y1, x2, y2),小于阈值为1
    '''
    def is_mini_boxes(self, boxes, area_threshold = 2667):  
        # area_threshold = self.box_threshold[0]
        return self.box_area(boxes)<area_threshold

    def is_big_boxes(self,boxes, area_threshold = 189312):
        # area_threshold = self.box_threshold[-2]
        return self.box_area(boxes)>area_threshold
    
    '''
    根据bbox的中心位置所在网格将场景分组
    '''
    def split_scenebox_in_grid(self, data_type = 'val' ,image_size = (1280,720) , bins = (10,5) ,thresh = 0.2):
        if isinstance(data_type,str):
            if data_type == 'val':
                file_text = self.val_txt
            elif data_type == 'minibox_val':
                file_text = self.minibox_val_txt   
            with open(file_text) as f:       
                all_scenes = f.read().split()
        elif isinstance(data_type,list):
            all_scenes = data_type
            data_type = 'no_ego'   

        secen_in_grid = [[] for i in range(bins[0]*bins[1])]
        no_box_scene = []
        for i , scene in enumerate(tqdm(all_scenes,desc='scenes: ')):
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            gt_labels = gt_data['labels']
            ori_gt_boxes = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            filter_box = [ self.box_center(x) for x in ori_gt_boxes if x.shape[0]>0]
            if not filter_box:
                no_box_scene.append(scene)
                continue
            result = np.concatenate(filter_box, axis=0)
            center_x, center_y = result[:,0], result[:,1]
            grid_size_x = image_size[0]//bins[0]
            grid_size_y = image_size[1]//bins[1]
            x_edges = np.arange(0, image_size[0]+1 , grid_size_x)
            y_edges = np.arange(0, image_size[1]+1, grid_size_y)
            heatmap, _, _ = np.histogram2d(center_x, center_y, bins=[x_edges, y_edges])
            heatmap = heatmap.T
            total_points = np.sum(heatmap)
            percentages = (heatmap / total_points) 
            sorted_indices = np.argsort(-percentages, axis=None)
            flatten_per = percentages.flatten()
            for i, index in enumerate(sorted_indices):
                if i == 0 or flatten_per[index] > thresh:
                    secen_in_grid[index].append(scene)
                else:
                    break
        save_path = os.path.join(self.folder , f'{data_type}_split_in_grid_{bins[0]}x{bins[1]}.json')
        json_data = {i:x for i,x in enumerate(secen_in_grid)} 
        json_data['No_box'] = no_box_scene
        with open(save_path,'w') as f:
            json.dump(json_data, f, indent=4)
    
    '''
    不同位置的bbox场景的AUC统计,绘制热力图
    '''
    def eval_on_grids(self, model_name ,grid_sceene_path:str, pkl_path:str , image_size = (1280,720) , bins = (10,5) , post_process =False , no_bar = False):
        def load_heatmap_data(scene_data, pkl_path):
            auc_grid, num_grid = np.zeros(bins[0]*bins[1]+1), np.zeros(bins[0]*bins[1]+1)
            with open(pkl_path, 'rb') as f:
                results = pickle.load(f)
            for i, value in enumerate(scene_data.values()):
                content = {}
                specific_scenes = value
                if len(specific_scenes) == 0:
                    continue
                indexs = [idx[0] for x in specific_scenes for idx in [np.where(results['video_name'] == x)[0]] if idx.size > 0]
                for key, value in results.items():
                    if isinstance(value, list):
                        content[key] = [value[i] for i in indexs]
                    elif isinstance(value, np.ndarray):
                        content[key] = value[indexs]
                auc_grid[i] = evaluation(FPS=10, **content, post_process = post_process )[0]
                num_grid[i] = len(specific_scenes)
            return auc_grid , num_grid

        def plot_single_heatmap(model_name, auc_grid, num_grid, is_cmp = False):
            # 这里的heatmap已经转置然后flatten,所以reshape为(y,x)
            heatmap , nummap = auc_grid[:-1].reshape((bins[1],bins[0])) , num_grid[:-1].reshape((bins[1],bins[0]))
            masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
            grid_size_x = image_size[0]//bins[0]
            grid_size_y = image_size[1]//bins[1]
            x_edges = np.arange(0, image_size[0]+1 , grid_size_x)
            y_edges = np.arange(0, image_size[1]+1, grid_size_y)
            xedges, yedges =  x_edges, y_edges
            plt.figure(figsize=(15, 9))
            # plt.gcf().subplots_adjust(top=0.85, bottom=0.15)  # top控制标题与图形之间的间距，bottom控制x轴与图形的间距
            if is_cmp:
                data_positive = np.ma.masked_where(heatmap < 0, heatmap)
                data_negative = np.ma.masked_where(heatmap > 0, heatmap)
                cmap_positive = plt.cm.YlOrRd  # 例如使用黄色到红色渐变表示正值
                cmap_negative = plt.cm.Blues_r  # 例如使用蓝色表示负值（反转的）          
                norm_positive = Normalize(vmin=0, vmax=np.max(data_positive))
                norm_negative = Normalize(vmin=np.min(data_negative), vmax=0)
                pos_img = plt.imshow(np.flip(data_positive, axis=0), origin='upper', cmap=cmap_positive, norm=norm_positive, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                neg_img = plt.imshow(np.flip(data_negative, axis=0), origin='upper', cmap=cmap_negative, norm=norm_negative, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
                
                if not no_bar:
                    ax = plt.gca()
                    cbar_pos = plt.colorbar(pos_img, ax=ax, fraction=0.046, pad=0.04, shrink=0.8, location='right')
                    cbar_pos.set_label('Positive Values')
                    cbar_neg = plt.colorbar(neg_img, ax=ax, fraction=0.046, pad=0.09, shrink=0.8, location='left')
                    cbar_neg.set_label('Negative Values')

                heatmap_zeros = np.ma.masked_where(heatmap != 0, heatmap)
                plt.imshow(np.flip(heatmap_zeros, axis=0), origin='upper', cmap=ListedColormap(['white']), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])   
            else:
                vmin = np.min(heatmap[heatmap>0]) 
                vmax = np.max(heatmap) 
                im = plt.imshow(np.flip(masked_heatmap, axis=0), origin='upper', cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],vmin=vmin, vmax=vmax)
                heatmap_zeros = np.ma.masked_where(heatmap != 0, heatmap)
                plt.imshow(np.flip(heatmap_zeros, axis=0), origin='upper', cmap=ListedColormap(['white']), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],vmin=vmin, vmax=vmax)
                plt.colorbar(im,label='grid AUC')  
            for t_y in range(len(yedges)-1):
                for t_x in range(len(xedges)-1):
                    value ,num = heatmap[t_y, t_x] , nummap[t_y, t_x]
                    pos_x , pos_y = xedges[t_x] + grid_size_x / 2 , yedges[t_y] + grid_size_y / 2
                    plt.text(pos_x, pos_y-grid_size_y / 8,f'{value*100:.2f}%', color='green', ha='center', va='center', fontsize=12)           
                    plt.text(pos_x, pos_y+grid_size_y / 8,f'{int(num):d}', color='blue', ha='center', va='center', fontsize=12)
            ax = plt.gca()
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')

            if not no_bar:
                plt.xlabel(f'no box auc {auc_grid[-1]*100:.2f}% | {num_grid[-1]}',fontsize=14)

            plt.xticks(x_edges)
            plt.yticks(y_edges)
            plt.gca().invert_yaxis()
            plt.title(model_name +f'_grid_{bins[0]}x{bins[1]}_AUC',fontsize=16,pad=30)
            fig_name = model_name + f'_grid_{bins[0]}x{bins[1]}_AUC_hotmap.png'
            plt.savefig(os.path.join(self.folder,'visualization','AUC_on_grid','v2',fig_name))

        scene_data = json.load(open(grid_sceene_path,'r'))
        if isinstance(pkl_path,str):
             auc_grid, num_grid = load_heatmap_data(scene_data, pkl_path)
             plot_single_heatmap(model_name, auc_grid, num_grid)
        elif isinstance(pkl_path,list) and len(pkl_path)==2: 
             auc_grid0, num_grid = load_heatmap_data(scene_data , pkl_path[0])
             plot_single_heatmap(model_name[0], auc_grid0, num_grid)
             auc_grid1, _ = load_heatmap_data(scene_data , pkl_path[1])
             plot_single_heatmap(model_name[1], auc_grid1, num_grid)
             auc_grid = auc_grid1 - auc_grid0 
             cmp_name = f'{model_name[0]} VS {model_name[1]}'
             plot_single_heatmap(cmp_name, auc_grid, num_grid, is_cmp=True)

    '''
    热力图：单独/对比  ego/non-ego 在grid上数量的分布 
    '''
    def class_on_grids(self, scene_type = 'all', class_type = 'compare', image_size = (1280,720) , bins = (10,5) ):
        anomalies = [
            'start_stop_or_stationary',  # 1 ST
            'moving_ahead_or_waiting',  # 2 AH
            'lateral',  # 3 LA
            'oncoming',  # 4 OC
            'turning',  # 5 TC
            'pedestrian',  # 6 VP
            'obstacle',  # 7 VO
            'leave_to_right',  # 8 OO-r
            'leave_to_left',  # 9 OO-l
            'unknown',  # 10 UK
            'out_of_control', # 11 OO Combine from OO-r and OO-l
            ]
        ego_json_path = META_FOLDER/ "class_aware/ego_split_in_grid_10x5.json" # no_ego_split_in_grid_10x5        
        non_ego_json_path = META_FOLDER/ "class_aware/no_ego_split_in_grid_10x5.json" # no_ego_split_in_grid_10x5        
        ego_data = json.load(open(ego_json_path,'r'))
        non_ego_data = json.load(open(non_ego_json_path,'r'))
        prefix = ''
        if scene_type == 'mini_box':
            prefix = 'mini_'
            # 保留mini_box中scenes
            file_text = self.minibox_val_txt   
            with open(file_text) as f:       
                all_scenes = f.read().split()
            update_ego_data , update_non_ego_data = {},{}
            for (key,value),(key2,value2) in zip(ego_data.items(),non_ego_data.items()):
                update_ego_data[key] = [x for x in value if x in all_scenes]
                update_non_ego_data[key2] = [x for x in value2 if x in all_scenes]
            ego_data,non_ego_data = update_ego_data , update_non_ego_data

        # class_type: 'Compare':  non_ego/all * 100%  |  'ego': ego scenes on grid | 'non-ego': non-ego scenes on grid
        if class_type == 'compare':
            class_name = 'Compare'
            ego_grid, non_ego_grid, num_grid = np.zeros(bins[0]*bins[1]+1) , np.zeros(bins[0]*bins[1]+1) , np.zeros(bins[0]*bins[1]+1)
            for i, (d1, d2) in enumerate(zip(ego_data.values(),non_ego_data.values())):
                ego_grid[i] = len(d1)
                non_ego_grid[i] = len(d2)
                if ego_grid[i]>0 or non_ego_grid[i]>0:
                    num_grid[i] = non_ego_grid[i]/(non_ego_grid[i]+ego_grid[i])

            ego_grid = ego_grid[:-1].reshape((bins[1],bins[0]))
            non_ego_grid = non_ego_grid[:-1].reshape((bins[1],bins[0]))

        elif class_type == 'ego':
            class_name = 'Ego'
            num_grid = np.zeros(bins[0]*bins[1]+1) 
            for i, d1 in enumerate(ego_data.values()):
                num_grid[i] = len(d1)

        elif class_type == 'non-ego':
            class_name = 'Non-Ego'
            num_grid = np.zeros(bins[0]*bins[1]+1) 
            for i, d1 in enumerate(non_ego_data.values()):
                num_grid[i] = len(d1)
   
        # 这里的heatmap已经转置然后flatten,所以reshape为(y,x)
        heatmap  = num_grid[:-1].reshape((bins[1],bins[0]))
        masked_heatmap = np.ma.masked_where(heatmap == 0, heatmap)
        grid_size_x = image_size[0]//bins[0]
        grid_size_y = image_size[1]//bins[1]
        x_edges = np.arange(0, image_size[0]+1 , grid_size_x)
        y_edges = np.arange(0, image_size[1]+1, grid_size_y)
        xedges, yedges =  x_edges, y_edges
        plt.figure(figsize=(15, 9))
        vmin = np.min(heatmap[heatmap>0]) 
        vmax = np.max(heatmap) 
        im = plt.imshow(np.flip(masked_heatmap, axis=0), origin='upper', cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],vmin=vmin, vmax=vmax)
        heatmap_zeros = np.ma.masked_where(heatmap != 0, heatmap)
        plt.imshow(np.flip(heatmap_zeros, axis=0), origin='upper', cmap=ListedColormap(['green']), extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],vmin=vmin, vmax=vmax)
        plt.colorbar(im,label='grid Numbers')  
        for t_y in range(len(yedges)-1):
            for t_x in range(len(xedges)-1):
                pos_x , pos_y = xedges[t_x] + grid_size_x / 2 , yedges[t_y] + grid_size_y / 2
                if class_type == 'compare':
                    ego_num , non_num , value = ego_grid[t_y, t_x]  , non_ego_grid[t_y, t_x]  , heatmap[t_y, t_x]        
                    plt.text(pos_x, pos_y,f'{value*100:.2f}%', color='green', ha='center', va='center', fontsize=12)
                    plt.text(pos_x, pos_y-grid_size_y / 4,f'{int(ego_num):d}', color='blue', ha='center', va='center', fontsize=12)           
                    plt.text(pos_x, pos_y+grid_size_y / 4,f'{int(non_num):d}', color='blue', ha='center', va='center', fontsize=12)
                else:
                    num = heatmap[t_y, t_x]
                    plt.text(pos_x, pos_y,f'{int(num):d}', color='blue', ha='center', va='center', fontsize=12)
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        if class_type != 'compare':
            plt.xlabel(f'no box numbers : {int(num_grid[-1])}',fontsize=14)
        else:
            plt.xlabel(f'upper: ego-involved | down: non-ego',fontsize=14)
        plt.xticks(x_edges)
        plt.yticks(y_edges)
        plt.gca().invert_yaxis()
        plt.title(prefix+class_name +f' grid {bins[0]}x{bins[1]} Numerbers of scenes',fontsize=16,pad=30)
        fig_name = prefix+class_name + f' grid {bins[0]}x{bins[1]} Numerbers of scenes.png'
        plt.savefig(os.path.join(self.folder,fig_name))

    '''
    统计框面积分布
    '''
    def plot_box_area_distribution(self,specific_info = None):
        if specific_info:
           figName = specific_info['name'] + '_'
           all_scenes = specific_info['scenes']
        else:
            figName = ''
            with open(self.val_txt) as f:       
                all_scenes = f.read().split()
        cal_all_boxes = []
        for scene in tqdm(all_scenes,desc='scenes: '):
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            gt_labels = gt_data['labels']
            ori_gt_boxes = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            filter_box = [ self.box_area(x) for x in ori_gt_boxes if x.shape[0]>0]
            cal_all_boxes.extend(filter_box)
        result = np.concatenate(cal_all_boxes, axis=0)
        data = sorted(result)
        frequency = [1] * len(data)
        frequency = np.cumsum(frequency)
        frequency = frequency/frequency[-1]
        plt.figure(figsize=(12, 8))
        plt.plot(data, frequency, marker='o', linestyle='-', color='b')

        # 标注分区间的比例
        intervals = [0.10] * 10
        interval_labels = np.cumsum(intervals)
        data_ind = [int(x*len(data)) for x in interval_labels]
        for i,label in enumerate(interval_labels):
            plt.scatter(data[data_ind[i]], label)
            plt.text(data[data_ind[i]], label, f'{int(label * 100)}% -> {data[data_ind[i]]:.1f}', color='r', va='bottom')
        
        specified_ticks = [data[x] for x in data_ind]
        plt.xticks(specified_ticks, labels=[f'{x:.2f}' for x in specified_ticks],rotation=90)

        # 设置标题和标签
        plt.title("Cumulative Distribution of Data")
        plt.xlabel("Value")
        plt.ylabel("Cumulative Probability")
        fig_name = figName +'val_box_area_distri.png'
        plt.savefig(os.path.join(self.folder,fig_name))
            
    '''
    统计框位置分布
    '''
    def plot_box_pos_distribution(self, specific_info = None , data_type = 'val' , bins = (5,5)):
        if specific_info:
           figName = specific_info['name'] + '_'
           all_scenes = specific_info['scenes']
        else:
            figName = ''
            file_text = self.val_txt if data_type == 'val'else self.train_txt
            with open(file_text) as f:       
                all_scenes = f.read().split()
        cal_all_boxes = []
        for i , scene in enumerate(tqdm(all_scenes,desc='scenes: ')):
            # if i>10:
            #     break
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            gt_labels = gt_data['labels']
            ori_gt_boxes = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            filter_box = [ self.box_center(x) for x in ori_gt_boxes if x.shape[0]>0]
            cal_all_boxes.extend(filter_box)
        if not cal_all_boxes:
            print(f'No box found in {figName}')
            return
        result = np.concatenate(cal_all_boxes, axis=0)
        center_x, center_y = result[:,0], result[:,1]
        image_size = (1280,720)

        '''
        热力图
        '''
        # 设置网格的大小
        grid_size_x = image_size[0]//bins[0]
        grid_size_y = image_size[1]//bins[1]

        # 定义网格的边界
        x_edges = np.arange(0, image_size[0]+1 , grid_size_x)
        y_edges = np.arange(0, image_size[1]+1, grid_size_y)

        # 使用 histogram2d 统计每个网格内的点数
        heatmap, xedges, yedges = np.histogram2d(center_x, center_y, bins=[x_edges, y_edges])
        heatmap = heatmap.T

        # 计算总点数和百分比
        total_points = np.sum(heatmap)
        percentages = (heatmap / total_points) * 100

        # 绘制热力图
        plt.figure(figsize=(10, 6))
        plt.imshow(np.flip(heatmap, axis=0), origin='upper', cmap='hot', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

        # 在每个网格内显示百分比，根据背景颜色调整字体颜色
        for t_y in range(len(yedges)-1):
            for t_x in range(len(xedges)-1):
                value = percentages[t_y, t_x]
                color = "green" if value < np.max(heatmap) * 0.1 else "blue"
                pos_x , pos_y = xedges[t_x] + grid_size_x / 2 , yedges[t_y] + grid_size_y / 2
                plt.text(pos_x, pos_y,f'{value:.2f}%', color=color, ha='center', va='center', fontsize=8)
                   
        # 设置 x 轴在上方
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')

        plt.xticks(x_edges)
        plt.yticks(y_edges)
        plt.gca().invert_yaxis()
        plt.colorbar(label='Number of Points')
        plt.title(figName +'{data_type}_{bins[0]}x{bins[1]}_BBox Position Distribution')
        fig_name = figName + f'{data_type}_{bins[0]}x{bins[1]}_box_posi_distri_hotmap.png'
        plt.savefig(os.path.join(self.folder,'box_posi_heatmap',fig_name))

        '''
        分布图
        '''
        # plt.figure(figsize=(8, 8))
        # ax = plt.gca()
        # plt.xlim(0, 1280)  
        # plt.ylim(0, 720) 

        # plt.xticks(x_edges, [f'{center}' for center in x_edges])
        # plt.yticks(y_edges, [f'{center}' for center in y_edges])

        # ax.spines['top'].set_position(('outward', 0))
        # ax.spines['bottom'].set_position(('outward', -9999))  # 隐藏底部 x 轴
        # ax.xaxis.set_label_position('top')
        # ax.xaxis.tick_top()

        # plt.scatter(center_x, center_y, c='red', marker='o', alpha=0.5)
        # plt.title(figName +'{data_type}_{bins[0]}x{bins[1]}_Box Position Distribution')
        # plt.grid(True)
        # plt.gca().invert_yaxis() 
        # fig_name = figName +f'{data_type}_{bins[0]}x{bins[1]}_box_posi_distri.png'
        # plt.savefig(os.path.join(self.folder,'box_posi_plot',fig_name))
    
    '''
    挑选测试集中异常目标是远景（小框）的场景：
    '''
    def selet_mini_box(self, area_threshold = 2667):
        with open(self.val_txt) as f:       
            all_scenes = f.read().split()
        
        with open(self.val_json) as f:
            val_metadata = json.load(f)

        mini_scene = []
        json_data = {}
        for scene in tqdm(all_scenes):
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            gt_labels = gt_data['labels']
            ori_gt_boxes = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            box_area_cmp = [ self.is_mini_boxes(x,area_threshold) for x in ori_gt_boxes if x.shape[0]>0 ]
            
            if len(box_area_cmp)<2:
                continue
            reduce_ = np.concatenate(box_area_cmp,axis=0)
            if np.any(reduce_):
                mini_scene.append(scene)
                json_data[scene] = val_metadata[scene]
        txt_save_path =  os.path.join(self.folder,'Less_Threshold=5K_val_split.txt')
        with open(txt_save_path,'w') as f:
            for x in mini_scene:
                f.write(f'{x}\n')
        custom_print(f'save txt to {txt_save_path}')
        json_save_path = os.path.join(self.folder,'Less_Threshold=5K_metadata_val.json')
        with open(json_save_path,'w') as f:
            json.dump(json_data, f, indent=4)
        custom_print(f'save json to {txt_save_path}')

        '''
    挑选测试集中异常目标是远景（小框）的场景：
    '''
    
    '''
    挑选训练集10%的场景
    '''
    def selet_train_sample(self,):
        with open(self.train_txt) as f:       
            all_scenes = f.read().split()[::10]
        
        with open(self.train_json) as f:
            val_metadata = json.load(f)

        select_scene = []
        json_data = {}
        for scene in tqdm(all_scenes):
            select_scene.append(scene)
            json_data[scene] = val_metadata[scene]
        txt_save_path =  os.path.join(self.folder,'select_train_split.txt')
        with open(txt_save_path,'w') as f:
            for x in select_scene:
                f.write(f'{x}\n')
        custom_print(f'save txt to {txt_save_path}')
        json_save_path = os.path.join(self.folder,'select_train_metadata_val.json')
        with open(json_save_path,'w') as f:
            json.dump(json_data, f, indent=4)
        custom_print(f'save json to {txt_save_path}')

'''
关于框的一些工具
1. 对比 yolo 的框 和  GT 的框 
2. 对比 yolo 的框 和  output 的框
3. intance anomaly 标注在对应的框上
4. 可视化某个场景下异常得分情况
5. 可视化某个场景下gt

1. 为测试集的yolo标注instance的异常标签: yolo_instance_match.json
2. 查看每帧中框(yolo,gt)的数量
3. 画框：
   * 在图像上画两组框,并且匹配的框在左上角用同一个index标注: 用于验证匈牙利匹配的正确性,即对比 yolo 和 gt
   * 在图像上画两组框, 对比 <某个配置下模型预测的框(注意: box_loss,即用了一个回归头用于得到bbox坐标没有用到)> 和yolo框, 同一个位置的框用同一个索引
   * 在图像上画框, 并可能有每个框的特有信息： 用于 instance anomal score
   * 单独画gt的框
   * 单独画yolo的框
4. 画得分+网格
   * 在图像上画网格
   * 只画gt score即标准的得分(只有得分)
5. 可视化：框+得分
   * 可视化 instance anomaly: lable from yolo , prediction from pkl
   * 可视化某个场景下异常得分情况
   * 可视化某个场景下gt
'''
class Boxes_Comparison(PathConfig):
    def __init__(self, folder : str) -> None:
        super().__init__()
        self.folder = folder
        self.matcher = HungarianMatcher()

    '''
    为测试集的yolo标注instance的异常标签
    即生成 match between yolo and gt : yolo_instance_match.json
    '''
    def get_instance_label_on_yolo(self,save_path):
        with open(self.val_txt,'r') as f:
            scenes =  f.read().split('\n')
        instance_labels = {}
        for scene in tqdm(scenes,desc='scenes: '):
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            num_frames , anomaly_start , anomaly_end =  gt_data['num_frames'], gt_data['anomaly_start'] , gt_data['anomaly_end'] 
            yolo_boxes , gt_boxes = [np.empty((0,4))]*num_frames ,  [np.empty((0,4))]*num_frames
            yolo_boxes[anomaly_start:anomaly_end] = self.read_video_boxes(yolo_labels,anomaly_start,anomaly_end)
            gt_boxes[anomaly_start:anomaly_end] = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            match_index = self.matcher(yolo_boxes,gt_boxes)
            matched_data = {}
            for ind , (yolo_match,gt_match) in enumerate(match_index):
                matched_data[ind] = [yolo_match.tolist(),gt_match.tolist()]
            instance_labels[scene] = matched_data
        with open(save_path,'w') as f:
            json.dump(instance_labels, f, indent=4)
    
    '''
    查看每帧中框(yolo,gt)的数量
    return train_info, val_info
    train_info, val_info: dict{scene_name: [ str, str , str, [int, int] ]}: 根据 (int+int) 排序
            例如：
            jFFhYwgepmY_003332  : 
                      anomaly_start:  33    anomaly_end:  54  length:  21
                      no box count : yolo   0   per frame box :   5   5   4   5   4   5   5   5   8   9   7   9   9   9   9   9   9   9   9   9   8
                      no box count : gt     0   per frame box :   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   2   1
                      [0,0]
    '''
    def calculate_box_in_scenes(self,):
        def calculate_box(yolo_folder: str,  gt_folder:str , scenes:List[str]):
            res = {}
            for scene in scenes:
                yolo_path = os.path.join(yolo_folder,scene+'.json')
                gt_path = os.path.join(gt_folder,scene+'.json')
                assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
                assert os.path.exists(gt_path),f'{gt_path} is not existed'
                with open(yolo_path, 'r') as f:
                    yolo_data = json.load(f)
                with open(gt_path, 'r') as f:
                    gt_data = json.load(f)
                yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
                anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
                yolo_box_count = [len(yolo_labels[i]['objects']) for i in range(anomaly_start,anomaly_end)]
                gt_box_count = [len(gt_labels[i]['objects']) for i in range(anomaly_start,anomaly_end)]
                yolo_zero_box = yolo_box_count.count(0)
                gt_zero_box = gt_box_count.count(0)
                info = []
                info.append(f'anomaly_start: {anomaly_start:>3}    anomaly_end: {anomaly_end:>3}  length: {(anomaly_end-anomaly_start):>3}')
                info.append(f'no box count : yolo  {yolo_zero_box:>2}' + '   per frame box : '+ ' '.join(f'{num:>3}' for num in yolo_box_count))
                info.append(f'no box count : gt    {gt_zero_box:>2}' + '   per frame box : '+ ' '.join(f'{num:>3}' for num in gt_box_count))
                info.append([yolo_zero_box,gt_zero_box])
                res[scene] = info
            return res

        with open(self.train_txt,'r') as f:
            train_scenes =  f.read().split()[:500]
        with open(self.val_txt,'r') as f:
            val_scenes =  f.read().split()[:500]
        
        # 对比yolo 和 gt 在异常部分框数量
        train_info = calculate_box(self.yolo_folder,self.gt_folder,train_scenes)
        val_info = calculate_box(self.yolo_folder,self.gt_folder,val_scenes)

        # 按照框数量排序
        train_info = dict(sorted(train_info.items(), key=lambda x: x[1][-1][0]+x[1][-1][1], reverse=True))
        val_info = dict(sorted(val_info.items(), key=lambda x: x[1][-1][0]+x[1][-1][1],reverse=True))
        
        # 写入txt
        txt_path = os.path.join(self.folder,'box_num_compare.txt')
        with open(txt_path,'w') as file:
            file.write(f'------ TRAIN SET : compare the number of boxes in anormal snippets between yolo and ground truth ------ \n')
            # train 
            for key , value in train_info.items():
                file.write(f'{key}  : ' + f'{value[0]}\n')
                file.write(f"{' ' * len(f'{key}  : ')}" + f"{value[1]}\n")
                file.write(f"{' ' * len(f'{key}  : ')}" + f"{value[2]}\n")
            file.write(f'\n\n\n')
            file.write(f'------ VAL SET : compare the number of boxes in anormal snippets between yolo and ground truth ------ \n')
            # train 
            for key , value in val_info.items():
                file.write(f'{key}  : ' + f'{value[0]}\n')
                file.write(f"{' ' * len(f'{key}  : ')}" + f"{value[1]}\n")
                file.write(f"{' ' * len(f'{key}  : ')}" + f"{value[2]}\n")
        
        return train_info, val_info
    
    '''
    在图像上画两组框,并且匹配的框在左上角用同一个index标注: 用于验证匈牙利匹配的正确性,即对比 yolo 和 gt
    文件位置："/data/qh/DoTA/output/debug/box_compare/yolo_gt/" + [scene name]
    '''     
    def draw_boxes_with_match(self, image, boxes_1, boxes_2, match_index):
        N,h,w,c = image.shape
        box_color  = ['red', 'blue', 'black' ,'yellow' , 'green'] 
        text_color = ['green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        font = ImageFont.truetype(FONT_FOLDER,25)
        image_with_boxes = []
        for i in range(N):
            image_pil = Image.fromarray(image[i])
            draw = ImageDraw.Draw(image_pil) 
            predict_box , gt_box, now_macth = boxes_1[i], boxes_2[i], match_index[i]
            box_num1 , box_num2 = predict_box.shape[0], gt_box.shape[0]
            # 标注 macthed boxes 
            match_len = len(now_macth[0])
            for ind in range(match_len):
                pre_b = predict_box[now_macth[0][ind]]
                gt_b = gt_box[now_macth[1][ind]]
                draw.rectangle(((pre_b[0],pre_b[1]),(pre_b[2],pre_b[3])), outline=box_color[0], width=2) 
                draw.text( (pre_b[0],pre_b[1]), f"{ind}",fill=text_color[ind%len(text_color)], font=font)
                draw.rectangle(((gt_b[0],gt_b[1]),(gt_b[2],gt_b[3])), outline=box_color[1], width=2)
                draw.text( (gt_b[0],gt_b[1]), f"{ind}",fill=text_color[ind%len(text_color)], font=font)

            # 标注 predict boxes 
            for ind, box in enumerate(predict_box):
                if ind not in now_macth[0]:
                    draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=box_color[0], width=2) 
            
            # 标注 gt boxes 
            for ind, box in enumerate(gt_box):
                if ind not in now_macth[1]:
                    draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=box_color[1], width=2) 
                    
            image_with_boxes.append(np.array(image_pil))
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes

    '''
    在图像上画两组框, 对比某个配置下模型预测的框和yolo框, 同一个位置的框用同一个索引 
    文件位置："/data/qh/DoTA/output/debug/box_compare/output_yolo/" + 模型config +  [scene name]
    '''
    def draw_boxes_output_yolo(self, scene, image, boxes_1, boxes_2):
        N,h,w,c = image.shape
        box_color  = ['red', 'blue', 'black' ,'yellow' , 'green'] 
        text_color = ['green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        font = ImageFont.truetype(FONT_FOLDER,25)
        image_with_boxes = []
        for i in range(N):
            image_pil = Image.fromarray(image[i])
            draw = ImageDraw.Draw(image_pil) 
            output_box , yolo_box = boxes_1[i], boxes_2[i]
            ### prompt test 的 bug 导致背景的points 没有去掉
            output_box = output_box[:yolo_box.shape[0]]
            box_num1 , box_num2 =  output_box.shape[0], yolo_box.shape[0]
            assert box_num1==box_num2,f'scene {scene} frame {i+3} box between output({box_num1}) and yolo({box_num2}) is not equal'

            for ind,(out_box,yolo_box) in enumerate(zip(output_box,yolo_box)):
                draw.rectangle(((out_box[0],out_box[1]),(out_box[2],out_box[3])), outline=box_color[0], width=2)
                draw.text( (out_box[0],out_box[1]), f"{ind}",fill=text_color[ind%len(text_color)], font=font)
                draw.rectangle(((yolo_box[0],yolo_box[1]),(yolo_box[2],yolo_box[3])), outline=box_color[1], width=2)
                draw.text( (yolo_box[0],yolo_box[1]), f"{ind}",fill=text_color[ind%len(text_color)], font=font) 

            image_with_boxes.append(np.array(image_pil))
                           
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes
      
    '''
    在图像上画框, 并可能有每个框的特有信息： 用于 instance anomal score
    '''
    def draw_boxes_with_info(self, image, boxes_list, info_list=None):
        N,h,w,c = image.shape
        box_color  = ['red', 'blue', 'black' ,'yellow' , 'green'] 
        text_color = ['red', 'green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        font = ImageFont.truetype(FONT_FOLDER,25)
        info_list = info_list if info_list else [[] for _ in range(N)]
        image_with_boxes = []
        for i in range(N):
            image_pil = Image.fromarray(image[i])
            draw = ImageDraw.Draw(image_pil)
            # pad empty when no info 
            if len(info_list[i])==0:
                info_list[i] = [[] for _ in range(boxes_list[i].shape[0])]
            for box,info in zip(boxes_list[i],info_list[i]):
                draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=box_color[0], width=4)
                if not isinstance(info,list) or len(info):
                    text = f"{info:.2f}"
                    left, top, right, bottom = font.getbbox(text)
                    w = right - left
                    h = bottom - top
                    outside = box[1] - h >= 0  # label fits outside box
                    draw.rectangle(
                        (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                        box[1] + 1 if outside else box[1] + h + 1),
                        fill=text_color[0],
                    )
                    draw.text((box[0], box[1] - (h+2) if outside else box[1]), text, fill='white', font=font)
            image_with_boxes.append(np.array(image_pil))
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes

    '''
    为instance-anomaly画不同颜色的框 
    '''
    def instance_anonaly_visualize(self, image, boxes_list, info):
        N,h,w,c = image.shape
        box_color  = ['red', 'blue', 'black' ,'yellow' , 'green'] 
        text_color = ['red', 'green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        font = ImageFont.truetype(FONT_FOLDER,25)
        image_with_boxes = []
        for i in range(N):
            image_pil = Image.fromarray(image[i])
            draw = ImageDraw.Draw(image_pil)    
            for box,data in zip(boxes_list[i],info[i]):
                lable,score = data
                b_color = 'green' if lable==0 else 'blue'
                t_color = 'green' if lable==0 else 'blue'
                draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=b_color, width=4)
                text = f"{score:.2f}"
                # draw.text( (box[0],box[1]), text,fill=t_color, font=font)    
                left, top, right, bottom = font.getbbox(text)
                # 计算文本的宽度和高度
                w = right - left
                h = bottom - top

                outside = box[1] - h >= 0  # label fits outside box
                draw.rectangle(
                    (box[0], box[1] - h if outside else box[1], box[0] + w + 1,
                     box[1] + 1 if outside else box[1] + h + 1),
                    fill=t_color,
                )
                draw.text((box[0], box[1] - (h+2) if outside else box[1]), text, fill='white', font=font)
                                            
            image_with_boxes.append(np.array(image_pil))
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes

    def padding_boxes(self,ori_boxes:np.array,target_size=20):
        # 将 box  padding 到 20 个 
        pad_rows = target_size-ori_boxes.shape[0]
        padding = np.zeros((pad_rows, 4), dtype=ori_boxes.dtype)
        padded_boxes = np.vstack((ori_boxes, padding))
        return padded_boxes
    
    def read_video_boxes(self,labels,begin,end):
        frames_boxes = []
        for frame_data in labels[begin:end]:
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            frames_boxes.append(np.array(boxes))
        return frames_boxes
    
    '''
    只画gt score 即标准的得分
    '''
    def plot_gt_score(self, scenes):
        with open(self.val_json, 'r') as f:
            val_gt_data = json.load(f)
        for scene in tqdm(scenes,desc="Scenes : "):
            save_path = os.path.join(self.folder,'gt','gt_score',scene)
            gt_data = val_gt_data[scene]
            n_frames , toa, tea = gt_data['num_frames'], gt_data['anomaly_start'] , gt_data['anomaly_end']
            pred_scores = np.zeros(n_frames, dtype=float)  
            pred_scores[toa:tea+1] = 1 
            # background
            fig, ax = plt.subplots(1, figsize=(30, 5))
            fontsize = 25
            plt.ylim(0, 1.0)
            plt.xlim(0, n_frames+1)

            xvals = np.arange(n_frames)
            plt.plot(xvals, pred_scores, linewidth=5.0, color='r')
            plt.axhline(
                y=0.5, xmin=0, xmax=n_frames + 1,
                linewidth=3.0, color='g', linestyle='--')
            if toa >= 0 and tea >= 0:
                plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                plt.axvline(x=tea, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                x = [toa, tea]
                y1 = [0, 0]
                y2 = [1, 1]
                ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)

            plt.xticks(range(0, n_frames + 1, 10), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

    '''
    在图像上画网格
    '''
    def plot_grid_on_image(self,frames, bins = (10,5) ):
        image_with_grid = []
        for frame in frames:
            image_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(image_pil)
            grid_cols, grid_rows = bins
            image_width, image_height = image_pil.size
            row_height = image_height // grid_rows
            col_width = image_width // grid_cols
            for row in range(1, grid_rows):
                y = row * row_height
                draw.line([(0, y), (image_width, y)], fill="white", width=2)
            for col in range(1, grid_cols):
                x = col * col_width
                draw.line([(x, 0), (x, image_height)], fill="white", width=2)
            
            image_with_grid.append(image_pil)
        image_with_grid =  np.array(image_with_grid)
        return image_with_grid

    '''
    单独画gt
    '''
    def draw_gt_box(self,scenes):
        for scene in tqdm(scenes,desc="Scenes : "):
            save_path = os.path.join(self.folder,'gt','gt_no_grid',scene)
            # if os.path.exists(save_path):
            #     continue
            os.makedirs(save_path,exist_ok=True)
            image_path = os.path.join(self.image_folder,scene,'images')
            #检测yolo和gt对应的文件时候存在
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            #检测视频帧数是否相同
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"                             
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(anomaly_start,anomaly_end)]
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))     
            ori_gt_boxes = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            frames = self.draw_boxes_with_info(frames,ori_gt_boxes)
            # frames = self.plot_grid_on_image(frames)
            # 储存图像
            for i, frame in enumerate(frames):
                frame_name = yolo_labels[i+anomaly_start]['frame_id']
                image_pil = Image.fromarray(frame)
                image_pil.save(os.path.join(save_path,frame_name))

    '''
    单独画yolo
    '''
    def draw_yolo_box(self,scenes):
        for scene in tqdm(scenes,desc="Scenes : "):
            save_path = os.path.join(self.folder,'yolo_vis',scene)
            # if os.path.exists(save_path):
            #     continue
            os.makedirs(save_path,exist_ok=True)
            image_path = os.path.join(self.image_folder,scene,'images')
            #检测yolo和gt对应的文件时候存在
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            #检测视频帧数是否相同
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"                             
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            # anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            start, end = 0, gt_data['num_frames']
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(start,end)]
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))     
            ori_yolo_boxes = self.read_video_boxes(yolo_labels,start,end)
            info_list = [list(range(boxes.shape[0])) for boxes in ori_yolo_boxes]
            frames = self.draw_boxes_with_info(frames,ori_yolo_boxes,info_list)
            # frames = self.plot_grid_on_image(frames)
            # 储存图像
            for i, frame in enumerate(frames):
                frame_name = yolo_labels[i]['frame_id']
                image_pil = Image.fromarray(frame)
                image_pil.save(os.path.join(save_path,frame_name))

    '''
    对比 yolo 和 gt 的框
    '''
    def compare_boxes_yolo_and_gt(self,scenes:List[str]):
        for scene in tqdm(scenes,desc="Scenes : "):
            save_path = os.path.join(self.folder,'yolo_gt',scene)
            if os.path.exists(save_path):
                continue
            os.makedirs(save_path,exist_ok=True)
            image_path = os.path.join(self.image_folder,scene,'images')
            #检测yolo和gt对应的文件时候存在
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            #检测视频帧数是否相同
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"                             
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(anomaly_start,anomaly_end)]
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            ori_yolo_boxes = self.read_video_boxes(yolo_labels,anomaly_start,anomaly_end)
            ori_gt_boxes = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            match_index = self.matcher(ori_yolo_boxes,ori_gt_boxes)
            frames = self.draw_boxes_with_match(frames,ori_yolo_boxes,ori_gt_boxes,match_index)

            # pad_yolo_boxes = np.array(list(map(self.padding_boxes,ori_yolo_boxes)))
            # pad_gt_boxes = np.array(list(map(self.padding_boxes,ori_gt_boxes)))
            # frames = self.draw_boxes(frames,pad_yolo_boxes,pad_gt_boxes,match_index)
            
            # 储存图像
            for i, frame in enumerate(frames):
                frame_name = yolo_labels[i+anomaly_start]['frame_id']
                image_pil = Image.fromarray(frame)
                image_pil.save(os.path.join(save_path,frame_name))

    '''
    对比 outputs 和 yolo 的框
    '''
    def compare_boxes_outputs_and_yolo(self,pkl_path:str, model_config:str, scenes:List[str]):
        assert os.path.exists(pkl_path),f'{pkl_path} is not existed'
        with open(pkl_path, 'rb') as f:
            content = pickle.load(f)
        assert 'bbox_all' in content ,f'output boxes is not in {pkl_path}'       
        outputs ={k:v for k,v in zip(content['video_name'],content['bbox_all'])}
        for scene in tqdm(scenes,desc="Scenes : ",):
            image_path = os.path.join(self.image_folder,scene,'images')
            save_path = os.path.join(self.folder,'output_yolo',model_config, scene)
            if os.path.exists(save_path) and len(os.listdir(save_path))!=0:
                continue
            os.makedirs(save_path,exist_ok=True)
            # outputs, yolo 对应的文件时候存在    
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            yolo_labels = yolo_data['lables']
            ouputs_boxes = outputs[scene]
            frame_begin, frame_end = 3, 3 + len(ouputs_boxes)
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(frame_begin,frame_end)] 
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            ori_yolo_boxes = self.read_video_boxes(yolo_labels,frame_begin,frame_end)
            assert len(frames_path) == len(ouputs_boxes) == len(ori_yolo_boxes),\
                f'image frames : {len(frames_path)}  output frames : {len(ouputs_boxes) }  yolo frames : {len(ori_yolo_boxes)} , not equal'
            frames = self.draw_boxes_output_yolo(scene, frames,ouputs_boxes,ori_yolo_boxes)

            # 储存图像
            for i, frame in enumerate(frames):
                frame_name = yolo_labels[i+frame_begin]['frame_id']
                image_pil = Image.fromarray(frame)
                image_pil.save(os.path.join(save_path,frame_name))
    
    '''
    可视化 instance anomaly: lable from yolo , prediction from pkl 
    '''
    def plot_instance_score(self, pkl_path:str, model_config:str, scenes:List[str], NF=4):
        shift = NF - 1
        assert os.path.exists(pkl_path),f'{pkl_path} is not existed'
        with open(pkl_path, 'rb') as f:
            content = pickle.load(f)
        assert 'obj_targets' in content ,f'instance anomal info is not in {pkl_path}'
        outputs ={k:[v,y] for k,v,y in zip(content['video_name'],content['obj_targets'],content['obj_outputs'])}
        for scene in tqdm(scenes,desc="Scenes : "):
            image_path = os.path.join(self.image_folder,scene,'images')
            save_path = os.path.join(self.folder,'instance_anomal',model_config, scene)
            if os.path.exists(save_path) and len(os.listdir(save_path))!=0:
                continue
            os.makedirs(save_path,exist_ok=True)
                        
            # yolo boxes
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            yolo_labels = yolo_data['lables']
            # instance anomal score
            obj_tar , obj_out =  outputs[scene][0] , outputs[scene][1]
            frame_begin, frame_end = shift, shift + len(obj_tar)
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(frame_begin,frame_end)] 
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            ori_yolo_boxes = self.read_video_boxes(yolo_labels,frame_begin,frame_end)
            assert len(frames_path) == len(obj_tar) == len(obj_out) == len(ori_yolo_boxes),\
                f'image frames : {len(frames_path)}  instance anomal score : outputs  {len(obj_out)} targets {len(obj_tar)}  yolo frames : {len(ori_yolo_boxes)} , not equal'
            
            # prepare info: formate : 'label (0,1) -> score (:.2f)'
            # info = []
            # for label,score in zip(obj_tar,obj_out):
            #     info.append([f'{l} -> {s:.1f}' for l,s in  zip(list(label),list(score))])
            # frames = self.draw_boxes_with_info(frames,ori_yolo_boxes,info)

            # add gt boxes in anomaly frames
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)       
            gt_labels = gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            ori_gt_boxes = self.read_video_boxes(gt_labels,anomaly_start,anomaly_end)
            frames[anomaly_start-shift:anomaly_end-shift] = self.draw_boxes_with_info(frames[anomaly_start-shift:anomaly_end-shift],ori_gt_boxes)
            
            info = []
            for label,score in zip(obj_tar,obj_out):
                info.append([(l,s) for l,s in  zip(list(label),list(score))])

            frames = self.instance_anonaly_visualize(frames, ori_yolo_boxes, info)

            # 储存图像
            for i, frame in enumerate(frames):
                frame_name = yolo_labels[i+frame_begin]['frame_id']
                image_pil = Image.fromarray(frame)
                image_pil.save(os.path.join(save_path,frame_name))

    @staticmethod
    def load_pickle(filename):
        with open(filename, 'rb') as f:
            content = pickle.load(f)
        return content

    @staticmethod
    def post_process_data(outputs , kernel_size = 31, popr_type='mid'):
        import scipy.signal as signal
        from scipy.ndimage import uniform_filter1d
        preds = np.array(outputs)
        if len(preds)>kernel_size:
            if popr_type == 'mid':
                preds = signal.medfilt(preds, kernel_size=kernel_size)
            elif popr_type == 'mean':
                preds = uniform_filter1d(preds, size=kernel_size)
        else:
            ks = len(preds) if len(preds)%2!=0 else len(preds)-1
            if popr_type == 'mid':
                preds = signal.medfilt(preds, kernel_size=ks)
            elif popr_type == 'mean':
                preds = uniform_filter1d(preds, size=kernel_size)
        preds = normalize_video(np.array(preds))
        return preds.tolist()

    '''
    可视化某个场景下异常得分情况
    '''
    def plot_anomaly_score_on_scenes(self, save_name, scenes, pkl_paths, names, NFs , highlight_x_axis = None,
                                    left_ha = [],  right_ha = [] ,scores_info = None, sub_folder_name = None, 
                                    add_popr = False, only_popr = False, last_frame = False,
                                    high_quality = False, instance_cfg = None):
        '''
        save_path = os.path.join(self.folder,'anomaly_score',sub_folder_name[optional], save_name, scene_score[optional]) 
        highlight_x_axis: List 额外标注的x轴 
        left_ha: List 在x_axis 上的额外标注的x轴坐标 往左移（避免于默认的标注重叠）
        right_ha: List 在x_axis 上的额外标注的x轴坐标 往右移（避免于默认的标注重叠）  
        scores_info: List:生成文件夹scene前缀[score]scene_name
        last_frame: bool 只生成最后一帧
        instance_cfg: 将每个实例得分添加到原图
        '''
        pkl_datas = [self.load_pickle(f) for f in pkl_paths]
        if add_popr:
            colores = ['red', 'blue','brown', 'orange', 'purple', 'pink', 'gray', 'cyan']
            linestyle = ['-','--','-','--','-','--','-','--',]
        else:
            colores = ['red', 'blue','brown', 'orange', 'purple', 'pink', 'gray', 'cyan']
            linestyle = ['-','-','-','-','-','-','-','-',]
        for ind , scene in enumerate(tqdm(scenes,desc="Scenes : ")):
            scene_score = scene if not scores_info else f'[{scores_info[ind]*100:.1f}]'+scene
            if sub_folder_name:
                save_path = os.path.join(self.folder,'anomaly_score',sub_folder_name,save_name, scene_score)
            else:
                save_path = os.path.join(self.folder,'anomaly_score',save_name, scene_score)

            # if os.path.exists(save_path):
            #     continue
            os.makedirs(save_path,exist_ok=True)
            image_path = os.path.join(self.image_folder,scene,'images')
            #检测yolo和gt对应的文件时候存在
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            #检测视频帧数是否相同
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"                             
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(0,gt_data['num_frames'])]
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            # 根据 NFs ，选出截取的帧数
            max_NF =  max(NFs)
            NF = max_NF
            diff_NF = [max_NF - x for x in NFs]
            if max_NF:
                frames_slice = frames[max_NF-1:-1] # 根据不同的模型而定
            else:
                frames_slice = frames
            
            # 画出 instance_anomaly score : 
            if  instance_cfg:
                pkl_path = instance_cfg['pkl_path']
                with open(pkl_path, 'rb') as f:
                    content = pickle.load(f)
                assert 'obj_targets' in content ,f'instance anomal info is not in {pkl_path}'
                outputs ={k:[v,y] for k,v,y in zip(content['video_name'],content['obj_targets'],content['obj_outputs'])}
                # yolo boxes
                yolo_path = os.path.join(self.yolo_folder,scene+'.json')
                assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
                with open(yolo_path, 'r') as f:
                    yolo_data = json.load(f)
                yolo_labels = yolo_data['lables']
                # instance anomal score
                obj_tar , obj_out =  outputs[scene][0] , outputs[scene][1]
                frame_begin, frame_end = max_NF-1, max_NF-1 + len(obj_tar)
                ori_yolo_boxes = self.read_video_boxes(yolo_labels,frame_begin,frame_end)
                assert len(frames_slice) == len(obj_tar) == len(obj_out) == len(ori_yolo_boxes),\
                    f'image frames : {len(frames_slice)}  instance anomal score : outputs  {len(obj_out)} targets {len(obj_tar)}  yolo frames : {len(ori_yolo_boxes)} , not equal'
                info = []
                for label,score in zip(obj_tar,obj_out):
                    info.append([(l,s) for l,s in  zip(list(label),list(score))])

                frames_slice = self.instance_anonaly_visualize(frames_slice,ori_yolo_boxes,info)

            # 加载 pkl 数据
            named_scores , named_AUC , score_lenth = {} , {} , []
            for pkl_data,name,d_NF in zip(pkl_datas,names,diff_NF):
                # np.where : (array([0]),...): [0][0] prevent warning : Ensure you extract a single element from your array before performing this operation.
                indices = np.where(pkl_data['video_name'] == scene)[0][0]
                targets = pkl_data['targets'][int(indices)][d_NF-1:-1] if d_NF else pkl_data['targets'][int(indices)]
                outputs = pkl_data['outputs'][int(indices)][d_NF-1:-1] if d_NF else pkl_data['outputs'][int(indices)]            
                # 只要后处理的结果
                if only_popr:             
                    outputs = self.post_process_data(outputs)
                    score_lenth.append(len(outputs))
                    named_scores[name] = outputs            
                    # 计算当前场景下的AUC
                    named_AUC[name] = roc_auc_score(targets,named_scores[name])
                else:    
                    score_lenth.append(len(outputs))
                    named_scores[name] = outputs            
                    named_AUC[name] = roc_auc_score(targets,named_scores[name])
                    if add_popr:
                        named_scores[f'popr_{name}'] = self.post_process_data(outputs)
                        named_AUC[f'popr_{name}'] = roc_auc_score(targets,named_scores[f'popr_{name}'])
                
            assert all(x == len(frames_slice) for x in score_lenth)      
            
            # 储存图像
            for index, frame in enumerate(tqdm(frames_slice,desc='draw images: ')):
                # only plot the last frame
                if last_frame and index != len(frames_slice)-1 :
                    continue 
                fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12, 6))
                # fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12, 12))
                # ax1 : 原图和标题为各个AUC
                ax1.imshow(frame)
                title_auc = []
                for name,score in named_AUC.items():
                    title_auc.append('{} {:.1f}'.format(name,score*100))
                title1 = 'AUC : '+' | '.join(title_auc)
                ax1.set_title(title1)
                ax1.axis('off')
                # ax2 : 异常得分可视化和标题为各个异常得分
                title_score = []
                for name,score in named_scores.items():
                    title_score.append('{} {:.1f}'.format(name,score[index]*100))
                scores = ' | '.join(title_score)
                title2 = 'frame {:3d} score : '.format(index+NF-1) + scores
                ax2.set_title(title2,pad=10)
                ax2.set_ylabel('Frame Anomaly Score', labelpad=5, fontsize=10)
                # 修改纵轴刻度字体大小 
                for label in ax2.get_yticklabels():  
                    label.set_fontsize(10)  
                # curve plot
                toa , tea = anomaly_start-NF+1 , anomaly_end-NF+1
                toa = max(toa,0)
                n_frames = score_lenth[0]
                xvals = np.arange(n_frames)
                ax2.set_aspect(n_frames*0.135)
                ax2.set_ylim(0, 1.0)
                ax2.set_xlim(0, n_frames)
                
                # Adjust x-axis ticks to be offset by 3  
                offset = NF-1   
                current_ticks  = ax2.get_xticks()
                shift_ticks = [tick - offset for tick in current_ticks  if tick>=offset ]
                # add 0 position or not 
                # shift_ticks.insert(0,0) 
                ax2.set_xticks(shift_ticks)
                offset_labels = [int(x + offset) for x in shift_ticks]
                ax2.set_xticklabels(offset_labels)  
                # 修改横轴刻度字体大小 
                for label in ax2.get_xticklabels():  
                    label.set_fontsize(12) 

                # plot highlight line and x_ticks
                if highlight_x_axis != None:
                    for h_x in highlight_x_axis[ind]:
                        ax2.axvline(x=h_x-NF+1, ymax=1.0, linewidth=2, color='g', linestyle=':')
                        if h_x in left_ha:
                            ax2.text(h_x-NF+1+0.5, 0, str(h_x), color='g', fontsize=12, ha='left', va='bottom')  
                        elif h_x in right_ha:
                            ax2.text(h_x-NF+1-0.2, -0.05, str(h_x), color='g', fontsize=12, ha='center', va='top') 
                        else:
                            ax2.text(h_x-NF+1, -0.05, str(h_x), color='g', fontsize=12, ha='center', va='top')  
                                    
                # plot gt line
                # ax2.axhline(y=0.5, xmin=0, xmax=n_frames, linewidth=3.0, color='g', linestyle='--') # threshold line = 0.5
                if toa >= 0 and tea >= 0:
                    ax2.axvline(x=toa, ymax=1.0, linewidth=1.5, color='r', linestyle='--')
                    ax2.axvline(x=tea, ymax=1.0, linewidth=1.5, color='r', linestyle='--')
                    x = [toa, tea]
                    y1 = [0, 0]
                    y2 = [1, 1]
                    ax2.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
                # plot scores
                for i,(name,score) in enumerate(named_scores.items()):
                    ax2.plot(xvals[:index+1],score[:index+1], label=name, color=colores[i], linestyle=linestyle[i],linewidth=2)
                
                image_savepath = os.path.join(save_path,yolo_labels[index+NF-1]['frame_id'])
                # label fontsize
                plt.legend(loc='best', fontsize=12, frameon=True)
                plt.subplots_adjust(hspace=0)  
                plt.tight_layout()
                dpi = 300 if high_quality else 150    
                plt.savefig(image_savepath,dpi=dpi,bbox_inches='tight', pad_inches=0.1)  
                plt.close('all')

    '''
    可视化某个场景下gt
    '''
    def plot_gt_score_on_scenes(self,scenes):
        
        for ind , scene in enumerate(tqdm(scenes,desc="Scenes : ")):
            save_path = os.path.join(self.folder,'gt','gt_score',scene)
            # if os.path.exists(save_path):
            #     continue
            os.makedirs(save_path,exist_ok=True)
            image_path = os.path.join(self.image_folder,scene,'images')
            #检测yolo和gt对应的文件时候存在
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            gt_path = os.path.join(self.gt_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            assert os.path.exists(gt_path),f'{gt_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            #检测视频帧数是否相同
            assert yolo_data['num_frames'] == gt_data['num_frames'] , \
                f"yolo frames: {yolo_data['num_frames']} is not equal to gt: {gt_data['num_frames']}"                             
            yolo_labels , gt_labels  =  yolo_data['lables'] , gt_data['labels']
            anomaly_start , anomaly_end = gt_data['anomaly_start'] , gt_data['anomaly_end']
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(0,gt_data['num_frames'])]
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            score = [1 if anomaly_start<=x< anomaly_end else 0 for x in range(gt_data['num_frames'])]
            NF = 1
            # 储存图像
            for index, frame in enumerate(tqdm(frames,desc='draw images: ')):
                fig,(ax1,ax2) = plt.subplots(2,1,figsize=(12, 12))
                # ax1 : 原图和标题为各个AUC
                ax1.imshow(frame)
                ax1.axis('off')
                # curve plot
                toa , tea = anomaly_start-NF+1 , anomaly_end-NF+1
                toa = max(toa,0)
                n_frames = gt_data['num_frames']
                xvals = np.arange(n_frames)

                # 准备用于绘制曲线的 ax2  
                toa, tea = anomaly_start - NF + 1, anomaly_end - NF + 1  
                toa = max(toa, 0)  
                n_frames = gt_data['num_frames']  
                ax2.set_aspect(n_frames*0.18)

                ax2.set_ylim(0, 1.0)
                ax2.set_xlim(0, n_frames)
        
                # plot gt line
                # ax2.axhline(y=0.5, xmin=0, xmax=n_frames, linewidth=3.0, color='g', linestyle='--') # threshold line = 0.5
                if toa >= 0 and tea >= 0:
                    ax2.axvline(x=toa, ymax=1.0, linewidth=1.5, color='r', linestyle='--')
                    ax2.axvline(x=tea, ymax=1.0, linewidth=1.5, color='r', linestyle='--')
                    x = [toa, tea]
                    y1 = [0, 0]
                    y2 = [1, 1]
                    ax2.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
                # plot scores
                ax2.plot(xvals[:index+1],score[:index+1], color='red',linewidth=6)             
                image_savepath = os.path.join(save_path,yolo_labels[index+NF-1]['frame_id'])
                plt.subplots_adjust(hspace=0)  
                plt.tight_layout()
                dpi = 300     
                plt.savefig(image_savepath,dpi=dpi,bbox_inches='tight', pad_inches=0.1)  
                plt.close('all')


'''
一些用到sam的工具
1. 测试 multi-object 功能  
'''
class sam_tools(PathConfig):
    def __init__(self, folder : str) -> None:
        super().__init__()
        self.folder = folder
        self.predictor = self.load_sam_model()

    def load_sam_model(self):
        sys.path.insert(0,"/home/qh/TDD/sam")
        from segment_anything import SamPredictor, sam_model_registry
        custom_print('Load SAM model....')
        sam = sam_model_registry["default"](checkpoint="/home/qh/TDD/sam/pretrained/sam_vit_h_4b8939.pth")
        sam.to(device='cuda')
        predictor = SamPredictor(sam)
        return predictor

    def read_video_boxes(self,labels,begin,end):
        frames_boxes = []
        for frame_data in labels[begin:end]:
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            frames_boxes.append(np.array(boxes))
        return frames_boxes

    @staticmethod
    def show_mask(mask, ax, color = None):
        color = np.array([30/255, 144/255, 255/255, 0.6]) if color is None else color
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    
    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

    @staticmethod        
    def show_box(box, ax, color = None):
        color = 'green' if color is None else color
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor = color, facecolor=(0,0,0,0), lw=2))   

    def draw_with_mask_and_box(self, image, boxes, masks, info_list=None):
        assert boxes.shape[0] == masks.shape[0], f'length of boxes is {boxes.shape[0]} while masks is {masks.shape[0]}'
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        colors = [np.concatenate([np.random.random(3), np.array([0.6])], axis=0) for _ in range(boxes.shape[0])]
        for color,mask,box in zip(colors,masks,boxes):
            sam_tools.show_mask(mask.cpu().numpy(), plt.gca(), color)
            sam_tools.show_box(box.cpu().numpy(), plt.gca(),color)
        plt.axis('off')
        return plt.gcf()

    def multi_object_sam(self,scenes):
        for scene in scenes:
            # prepare save folder
            scene_path = os.path.join(self.folder,scene)
            # if os.path.exists(scene_path) and len(os.listdir(scene_path))!=0:
            #     continue
            os.makedirs(scene_path,exist_ok=True)
            # get yolo boxes
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            yolo_labels = yolo_data['lables']
            frame_begin, frame_end = 0, yolo_data['num_frames']
            yolo_boxes = self.read_video_boxes(yolo_labels,frame_begin,frame_end)
            # get images
            image_path = os.path.join(self.image_folder,scene,'images')
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(frame_begin,frame_end)] 
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            for i in tqdm(range(frame_begin,frame_end),desc='images: '):
                image = frames[i]
                frame_box = torch.tensor(yolo_boxes[i], device=self.predictor.device)
                self.predictor.set_image(image)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(frame_box, image.shape[:2])
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                fig = self.draw_with_mask_and_box(image,frame_box,masks)
                save_name = os.path.join(scene_path,f"{yolo_labels[i]['frame_id']}")
                fig.savefig(save_name,dpi=300)

    '''
    记录： image frames 有 4990个scenes , 而 train_list(3275) + val_list(1402) = 4677 和 metadata 相等
    '''
    @staticmethod
    def compress_npdata():
        # 创建示例数据
        data = np.load("/data/qh/DoTA/data/sam_emb/a_VxrUq9PmA_001718/vit_h_emb.npy")
        # 压缩数据
        
        com_path= "/data/qh/DoTA/data/sam_emb/a_VxrUq9PmA_001718/com_vit_h_emb.npz"
        np.savez_compressed(com_path,data=data) 
        com_data=np.load(com_path)['data'] 
        # 验证数据是否一致
        assert np.array_equal(data, com_data)

    '''
    生成image_encoder image embs
    '''
    def sam_vis_encoder_embs(self,savefoler):
        scenes = []
        with open(self.train_txt,'r') as f:
            scenes.extend(f.read().split())
        with open(self.val_txt,'r') as f:
            scenes.extend(f.read().split())
        half_len = int(len(scenes)/2)
        # scenes = scenes[:half_len]
        scenes = scenes[half_len:]
        for scene in tqdm(scenes,desc='Scenes: '):
            # prepare save folder
            scene_path = os.path.join(savefoler,scene)
            if os.path.exists(scene_path) and len(os.listdir(scene_path))!=0:
                continue
            os.makedirs(scene_path,exist_ok=True)
            # get yolo data
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            with open(yolo_path, 'r') as f:
                yolo_data = json.load(f)
            yolo_labels = yolo_data['lables']
            frame_begin, frame_end = 0, yolo_data['num_frames']
            # get images
            image_path = os.path.join(self.image_folder,scene,'images')
            frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(frame_begin,frame_end)] 
            frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
            # 定义 batch_size
            batch_size = 8
            split_frames = [frames[i:i + batch_size] for i in range(0, frames.shape[0], batch_size)]
            embs_list = []
            for batch_frame in split_frames:       
                self.predictor.set_image(batch_frame)
                embs_list.append(self.predictor.features.cpu().numpy().astype(np.float16))
                self.predictor.reset_image()
            np.save(f'{scene_path}/vit_h_emb2.npy', np.concatenate(embs_list, axis=0))

    '''
    将video level的emb 拆分为frame level
    '''
    def split_embs_to_frame(self,savefoler):
        scenes = []
        with open(self.train_txt,'r') as f:
            scenes.extend(f.read().split())
        with open(self.val_txt,'r') as f:
            scenes.extend(f.read().split())
        scenes = sorted(scenes)
        half_len = int(len(scenes)/2)
        scenes = scenes[:half_len]
        # scenes = scenes[half_len:]
        for scene in tqdm(scenes,desc='Scenes: '):
            # prepare save folder
            scene_path = os.path.join(savefoler,scene)
            whole_emb_path = os.path.join(scene_path,'vit_h_emb2.npy')
            if not os.path.exists(whole_emb_path):
                continue

            # get yolo data
            yolo_path = os.path.join(self.yolo_folder,scene+'.json')
            assert os.path.exists(yolo_path),f'{yolo_path} is not existed'
            with open(yolo_path, 'r') as f:
                num_frames = json.load(f)['num_frames']

            vit_emb = np.array(np.load(whole_emb_path))
            assert num_frames == vit_emb.shape[0], f'{scenes} error'
            for i,frame_emb in enumerate(vit_emb):
                np.save(f'{scene_path}/{i}.npy',frame_emb)
            os.remove(whole_emb_path)



'''
生成视频
'''
if __name__ == '__main__':
    pass
    # pp_tools = ProcessorUtils()
    # scenes = ['[score]a_VxrUq9PmA_002682_AUC']
    # output_folder = "/data/qh/DoTA/output/debug/visualization/videos/others"
    # # target_folder = ["/data/qh/DoTA/poma_v2/rnn/base,no_fpn,rnn,vcl=8,lr=0.002/Gradcam_Frame/a_VxrUq9PmA_002682_AUC-58.90"]
    # target_folder = ['/data/qh/DoTA/output/debug/box_compare/anomaly_score/baseline VS Fullmodel[v2]/[demoL-paper]/a_VxrUq9PmA_002682']
    # # def get_sort_key(path):  
    # #     match = re.search(r'gradcam_(\d+)', path)    
    # #     return int(match.group(1)) if match else 0  
    # pp_tools.create_video_from_folders(scenes=scenes, output_folder=output_folder, target_folder=target_folder, sort_keys=None)
    
    # # ego-involved / non-ego
    # val_json = "/data/qh/DoTA/data/metadata/metadata_val.json"
    # with open(val_json, 'r') as f:
    #     val_gt_data = json.load(f)
    # ego_involved_scenes = [x for x in val_gt_data.keys() if 'ego' in  val_gt_data[x]['anomaly_class']]
    # non_ego_scenes = [x for x in val_gt_data.keys() if 'other' in  val_gt_data[x]['anomaly_class']]
    # ego_involved_scenes = {x:all_scenes[x] for x in ego_involved_scenes if x in keys}
    # non_ego_scenes = {x:all_scenes[x] for x in non_ego_scenes if x in keys}

    # # sorted
    # ego_involved_scenes = dict(sorted(ego_involved_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    # non_ego_scenes = dict(sorted(non_ego_scenes.items(), key=lambda x:x[1][-1], reverse=True))

    # scenes = list(all_scenes.keys())[-200:-10:20]
    # scenes = ['0qfbmt4G8Rw_004485','0qfbmt4G8Rw_004658','0RJPQ_97dcs_002834','0RJPQ_97dcs_000387']
    
    # model_config = '[ego]poma,vst,ins,prompt,rnn,depth=4'
    # scenes = list(ego_involved_scenes.keys())[:8]
    # Tools.plot_instance_score(pkl_path,model_config,scenes)
    # model_config = '[non-ego]poma,vst,ins,prompt,rnn,depth=4'
    # scenes = list(non_ego_scenes.keys())[8:16]
    # Tools.plot_instance_score(pkl_path,model_config,scenes)

    # folder = '/data/qh/DoTA/output/debug/box_compare' 
    # Tools = Boxes_Comparison(folder)
    # # scenes = ['D_pyFV4nKd4_001836','a_VxrUq9PmA_002682','Pbw0A-RCjcw_002938','pQdl0apLT70_004976']  

    # # pkl_path = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt,rnn,depth=4/eval/results-160.pkl"
    # pkl_path = "/data/qh/DoTA/poma_v2/instance/[v2]vst,ins,prompt,rnn,depth=4/eval/results-120.pkl"
    # # metrics : AUC+ACC/2  
    # auc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'frame', post_process=False)
    # acc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'instance')
    # keys = list(acc_scenes.keys())
    # all_scenes = {x:[auc_scenes[x],acc_scenes[x],(auc_scenes[x]+acc_scenes[x])/2] for x in keys}
    # all_scenes = dict(sorted(all_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    
    # output_folder = "/data/qh/DoTA/output/debug/visualization/videos/High_performence"  
    # scenes = list(all_scenes.keys())[:8]
    # pp_tools.create_video_from_folders(scenes=scenes, output_folder=output_folder,)

    # output_folder = "/data/qh/DoTA/output/debug/visualization/videos/Poor_performence"  
    # scenes = list(all_scenes.keys())[-8:]
    # pp_tools.create_video_from_folders(scenes=scenes, output_folder=output_folder,)

'''
计算stauc
'''
if __name__ == '__main__':
    pass
    pp_tools = Post_Process_Tools()
    # pkl_path = "/data/qh/DoTA/poma_v2/instance/vst,ins,bottle,prompt,rnn,depth=4/"
    # pkl_path = "/data/qh/DoTA/poma_v2/instance/vst,ins,bottle,prompt,rnn,depth=4/eval/results-180.pkl"
    # model_folder =  "/data/qh/DoTA/poma_v2/instance/vst,ins,bottle,prompt,rnn,depth=4" 
    # model_folder = "/data/qh/DoTA/poma_v2/instance/[v2]vst,ins,prompt,rnn,depth=4/"
    # specific_peoch  = [120]
    # model_folder = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt,rnn,depth=4/"
    # specific_peoch  = [160]
    # pp_tools.calculate_stauc_score(model_folder,specific_peoch,popr=True)


'''
单独生成eval, 调post process 
'''
if __name__ == '__main__':
    pass
    # model_folder = "/data/qh/DoTA/data/metadata/"
    # Tools = Scene_Aware_Tools(model_folder)
    # Tools.selet_mini_box(area_threshold=5000)

    # model_folder = "/data/qh/DoTA/output/Prompt/cross_attn_trans_depth_4_load_from_prompt/"
    # epochs = [80]

    # model_folder = "/data/qh/DoTA/output/Baseline/v4_2/"
    # epochs = [690]

    # model_folder = "/data/qh/DoTA/output/Prompt/standart_fpn_lr=0.002/"
    # epochs = [200]

    # model_folder = "/data/qh/DoTA/poma_v2/instance/[v2]vst,ins,prompt,rnn,depth=4/"
    # epochs = [120]
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = None , post_process = True , specific_info = None)

    # model_folder = "/data/qh/DoTA/output/Baseline/v4_2/"
    # epochs = [690]
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , post_process = True , specific_info = None ,test_kernelsize = False)

    # specific_info = {}
    # specific_info['name'] = 'minibox_10K'
    # with open("/data/qh/DoTA/data/metadata/minibox_val_split.txt",'r') as f :
    #     specific_info['scenes'] = f.read().split()
    #     print(f'LENGTH = {len(specific_info["scenes"])}')

    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , post_process = False , specific_info = specific_info)

    '''
    DADA
    '''
    # model_folder = "/data/qh/DADA/poma/[no_train]vst,ins,decoderV2,prompt,rnn/"
    # epochs = [120]
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , val_type='dada_all', post_process = True , specific_info = None)
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , val_type='dada_ego', post_process = False , specific_info = None)
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , val_type='dada_noego', post_process = False , specific_info = None)
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , val_type='dada_ego', post_process = True , specific_info = None)
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , val_type='dada_noego', post_process = True , specific_info = None)

'''
sam 给box 查看mask
'''
if __name__ == '__main__':
    pass
    # folder = '/data/qh/DoTA/data/sam'
    # Tools = sam_tools(folder)
    # # Tools.split_embs_to_frame("/data/qh/DoTA/data/sam_emb/")
    # # Tools.sam_vis_encoder_embs("/data/qh/DoTA/data/sam_emb/")
    # scenes = ['eWWgJznGg6U_006506',
    #           '3Sqeb-l1RPA_000983',
    #           'yhtzAKqRyXw_004811',
    #           'lmykriTbncM_003990'
    #           ]
    # Tools.multi_object_sam(scenes)


'''
plot frame-level anomaly score
'''
if __name__ == "__main__": 
    pass
    '''
    with / without popr in single model
    '''
    # folder = "/data/qh/DoTA/output/Prompt/cross_attn_trans_depth_4_load_from_prompt/"
    # epochs = [80]
    # big_diff, both_bad = Post_Process_Tools().compare_popr_auc_on_per_scene(folder, epochs , save_path = None )

    # # 区分：positive 和 negtive big_diff
    # # neg_big_diff = {item[0]:item[1] for item in big_diff.items() if item[1][-1] > 0}
    # # pos_big_diff = {item[0]:item[1] for item in big_diff.items() if item[1][-1] < 0}
    # # scenes = list(neg_big_diff.keys())[:3]+list(pos_big_diff.keys())[:3]+list(big_diff.keys())[-3:]+list(both_bad.keys())[:3]+list(both_bad.keys())[-3:]

    # scenes = [item[0] for item in both_bad.items() if item[1][-1] < 0.5]
    # scores = [item[1][-1] for item in both_bad.items() if item[1][-1] < 0.5]
    # scenes = scenes[::3]
    # scores = scores[::3]

    '''
    sigle model : high ACC for visualization
    '''
    # folder = '/data/qh/DoTA/output/debug/box_compare' 
    # cmp = Boxes_Comparison(folder)
    # pkl_paths = ["/data/qh/DoTA/poma_v2/instance/vst,ins,prompt,rnn,depth=4/eval/results-160.pkl"] 
    # # average ACC on instance and frame
    # pkl_path = pkl_paths[0]
    # auc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'frame', post_process=False)
    # acc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'instance')
    # keys = list(acc_scenes.keys())
    # all_scenes = {x:[auc_scenes[x],acc_scenes[x],(auc_scenes[x]+acc_scenes[x])/2] for x in keys}
    # # all_scenes = dict(sorted(all_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    # # scenes = list(all_scenes.keys())[:8]

    # # ego-involved / non-ego
    # val_json = "/data/qh/DoTA/data/metadata/metadata_val.json"
    # with open(val_json, 'r') as f:
    #     val_gt_data = json.load(f)
    # non_ego_scenes = [x for x in val_gt_data.keys() if 'other' in  val_gt_data[x]['anomaly_class']]
    # non_ego_scenes = {x:all_scenes[x] for x in non_ego_scenes if x in keys}
    # non_ego_scenes = dict(sorted(non_ego_scenes.items(), key=lambda x:x[1][-1], reverse=True))

    # scenes = list(non_ego_scenes.keys())[:3]
    # print(scenes)
    # scores = [x[-1] for x in list(all_scenes.values())[:8]]
    # save_name = '[only][average_ACC]full_model_v1' # folder name
    # names = [''] # compare model name
    # NFs = [4]
    # cmp.plot_anomaly_score_on_scenes(scenes = scenes, scores_info = scores, sub_folder_name ='[high_ACC]', save_name = save_name ,pkl_paths = pkl_paths , NFs = NFs, names = names)

    '''
    numerous model 
    '''
    # folder_1 = "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8"
    # epoch_1 = 160
    # folder_2 = "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8"
    # epoch_2 =  120
    
    # # 对比两个模型在每个scene的AUC / Accuracy
    # model_1 = f'[poma]baseline'
    # model_2 = '[v2]full model'
    # save_path = F"{model_1} VS {model_2}.txt"
    # # big_diff, both_bad = Post_Process_Tools().compare_auc_on_per_scene(folder_1, epoch_1, folder_2 , epoch_2, metric_type = 'Accuracy', save_path = save_path , post_process=False)
    # big_diff, both_bad = Post_Process_Tools().compare_auc_on_per_scene(folder_1, epoch_1, folder_2 , epoch_2, metric_type = 'AUC', save_path = None , post_process=False)
    # # scenes = list(big_diff.keys())[:100]
    # scenes =  list(both_bad.keys())[:30]
    
    # pkl_path = "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8/eval/results-120.pkl"
    # auc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'frame', post_process=False)
    # acc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'instance')
    # keys = list(acc_scenes.keys())
    # all_scenes = {x:[auc_scenes[x],acc_scenes[x],(auc_scenes[x]+acc_scenes[x])/2] for x in keys}
    # all_scenes = dict(sorted(all_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    # all scenes
    # scenes = list(all_scenes.keys())[-20:]

    # select night
    # low_auc_scenes = list(all_scenes.keys())[-50:]
    # night_low_auc = []
    # for x in low_auc_scenes:
    #     with open(DATA_FOLDER / 'annotations' /f"{x}.json" , 'r') as f:
    #         meta = json.load(f)
    #     if meta['night']:
    #       night_low_auc.append(str(x))
    # scenes = night_low_auc

    # scenes = ['JHfjuFrnpjA_005930','JV0D-YkWHD8_002421','UjkeNSr2dXQ_000394']
    # highlight_x_axis = None
    # left_ha = []
    # right_ha = []

    # # ego-involved / non-ego
    # val_json = "/data/qh/DoTA/data/metadata/metadata_val.json"
    # with open(val_json, 'r') as f:
    #     val_gt_data = json.load(f)
    # ego_involved_scenes = [x for x in val_gt_data.keys() if 'ego' in  val_gt_data[x]['anomaly_class']]
    # non_ego_scenes = [x for x in val_gt_data.keys() if 'other' in  val_gt_data[x]['anomaly_class']]
    # ego_involved_scenes = {x:all_scenes[x] for x in ego_involved_scenes if x in keys}
    # non_ego_scenes = {x:all_scenes[x] for x in non_ego_scenes if x in keys}
    
    # # sorted
    # ego_involved_scenes = dict(sorted(ego_involved_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    # non_ego_scenes = dict(sorted(non_ego_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    # scenes = list(ego_involved_scenes.keys())[100:]
    # # scenes = list(non_ego_scenes.keys())[100:]

    folder = '/data/qh/DoTA/output/debug/box_compare' 
    cmp = Boxes_Comparison(folder)
    pkl_paths = [
                 "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8/eval/results-120.pkl",
                 "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8/eval/results-160.pkl",
                #  "/ssd/qh/DoTA/data/TTHF/TDAFF_BASE_RN50/popr_eval.pkl",
                 "/ssd/qh/DoTA/data/TTHF/TDAFF_BASE_RN50/eval.pkl",
                ]
    sub_folder_name = 'Promppt VS Others'
    # save_name = '[video][online,dark,lowAUC]FullModel_Baseline_TTHF'
    save_name = '[selected]FullModel_Baseline_TTHF'
    names = ['FullModel','Baseline','TTHF'] # compare model name
    only_popr = False
    # save_name = '[popr]FullModel_Baseline_TTHF'
    # names = ['FullModel†','Baseline†','TTHF†'] # compare model name
    # only_popr = True
    NFs = [4,4,0]
    last_frame = True # True  False
    high_quality = True # True  False
    instance_cfg = {}
    instance_cfg['pkl_path'] = "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8/eval/results-120.pkl"
    # instance_cfg = None

    # select to discuss
    # scenes = ['eWWgJznGg6U_006506','yhtzAKqRyXw_004811','pQdl0apLT70_004976','hy433gakFeo_001156','4wKjxDXnmYs_001051']
    # scenes = ['DyzL2sahobA_002611']

    # positive-selected fixed x_ticks
    # scenes = ['a_VxrUq9PmA_002682','Pbw0A-RCjcw_002938','PqbpIHZvjMA_002926','Hd2IzHAfkCI_001091','HzVbo46kkBA_004124',
    #           'Q7VBPeGwJWw_001631','D_pyFV4nKd4_001836',]
    
    # scenes = ['D_pyFV4nKd4_001836','a_VxrUq9PmA_002682','Pbw0A-RCjcw_002938','pQdl0apLT70_004976']
    # highlight_x_axis = [[34,48,58],[27,50,61],[15,53,113],[21,56,75]]
    # left_ha = [21,61]
    # right_ha = [58] 

    scenes = ['JHfjuFrnpjA_005930']
    highlight_x_axis = [[39,43,48,52,57,66]]
    left_ha = [39,48]
    right_ha = []

    cmp.plot_anomaly_score_on_scenes(scenes = scenes, sub_folder_name = sub_folder_name, 
                                     save_name = save_name , pkl_paths = pkl_paths , only_popr = only_popr,
                                     NFs = NFs, names = names , highlight_x_axis = highlight_x_axis , 
                                     left_ha=left_ha , right_ha = right_ha , last_frame=last_frame,
                                     high_quality=high_quality, instance_cfg=instance_cfg)


'''
可视化instance anomaly
1. 为测试集的yolo标注instance的异常标签
2. 对比两个test result,在逐场景下测量AUC
'''
if __name__ == "__main__":
    '''
    为测试集的yolo标注instance的异常标签: 
    match between yolo and gt : yolo_instance_match.json
    '''
    # folder = '/data/qh/DoTA/output/debug/box_compare' 
    # Tools = Boxes_Comparison(folder)
    # Tools.get_instance_label_on_yolo('/data/qh/DoTA/data/metadata/yolo_instance_match.json')

    '''
    "/data/qh/DoTA/output/Prompt/box_obj_loss_weight=0.0002_0.02/eval/results-200.pkl"
    "/data/qh/DoTA/output/Prompt/only_obj_loss_load_from_prompt/eval/results-200.pkl"
    
    scenes 选择方式：
    scenes = ['0RJPQ_97dcs_000387']

    mini_val_path = "/data/qh/DoTA/data/metadata/mini_metadata_val.json"
    with open(mini_val_path,'r') as f:
        scene_data = json.load(f)
    scenes = list(scene_data.keys())

    scenes = os.listdir("/data/qh/DoTA/output/debug/box_compare/instance_anomal/basic_obj_loss/")

    # '''
    folder = '/data/qh/DoTA/output/debug/box_compare' 
    Tools = Boxes_Comparison(folder)
    # pkl_path = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt,rnn,depth=4/eval/results-160.pkl"
    pkl_path = "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8/eval/results-120.pkl"
    
    # auc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'frame', post_process=False)
    # acc_scenes = Accuracy_on_scene(pkl_path, acc_type = 'instance')
    # keys = list(acc_scenes.keys())
    # all_scenes = {x:[auc_scenes[x],acc_scenes[x],(auc_scenes[x]+acc_scenes[x])/2] for x in keys}
    # all_scenes = dict(sorted(all_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    # scenes = list(all_scenes.keys())[:8]

    # scenes = ['JkYzYrJpSoQ_005468', 'ZIts2XH28SA_001782', 'hdOtS-BpUPY_003911']
    # model_config = '[high_avg_acc]poma,vst,ins,prompt,rnn,depth=4'
    # Tools.plot_instance_score(pkl_path,model_config,scenes)

    # # ego-involved / non-ego
    # val_json = "/data/qh/DoTA/data/metadata/metadata_val.json"
    # with open(val_json, 'r') as f:
    #     val_gt_data = json.load(f)
    # ego_involved_scenes = [x for x in val_gt_data.keys() if 'ego' in  val_gt_data[x]['anomaly_class']]
    # non_ego_scenes = [x for x in val_gt_data.keys() if 'other' in  val_gt_data[x]['anomaly_class']]
    # ego_involved_scenes = {x:all_scenes[x] for x in ego_involved_scenes if x in keys}
    # non_ego_scenes = {x:all_scenes[x] for x in non_ego_scenes if x in keys}

    # # sorted
    # ego_involved_scenes = dict(sorted(ego_involved_scenes.items(), key=lambda x:x[1][-1], reverse=True))
    # non_ego_scenes = dict(sorted(non_ego_scenes.items(), key=lambda x:x[1][-1], reverse=True))

    # scenes = list(all_scenes.keys())[-200:-10:20]
    # scenes = ['0qfbmt4G8Rw_004485','0qfbmt4G8Rw_004658','0RJPQ_97dcs_002834','0RJPQ_97dcs_000387']
    
    # model_config = '[ego]poma,vst,ins,prompt,rnn,depth=4'
    # scenes = list(ego_involved_scenes.keys())[:8]
    # Tools.plot_instance_score(pkl_path,model_config,scenes)
    # model_config = '[non-ego]poma,vst,ins,prompt,rnn,depth=4'
    # scenes = list(non_ego_scenes.keys())[8:16]
    # Tools.plot_instance_score(pkl_path,model_config,scenes)

    # select scenes 
    # scenes = ['O9uvBFovKj8_005244', 'Pbw0A-RCjcw_002938', 'PqbpIHZvjMA_002926','a_VxrUq9PmA_002682'] #non-ego
    # scenes = ['4wKjxDXnmYs_000311','cfrLchAShxQ_001515','cfrLchAShxQ_002469','Hd2IzHAfkCI_001091','HzVbo46kkBA_004124','JV0D-YkWHD8_001349'] #ego
    # scenes = ['3Sqeb-l1RPA_000983','6E2N9ld0eHg_003418','EY8x-fyQkbk_003863','gXezhrOijmQ_003917','K1r3m5OrmB4_001155',
    #           'zRZ9PJguIfE_003237','4wKjxDXnmYs_005595','bhA2ckvE-TQ_005336','DyzL2sahobA_002611','eWWgJznGg6U_006506',
    #           '4wKjxDXnmYs_001051','bhA2ckvE-TQ_004401','HaC3LrJiTmQ_005454','Hkui6PJboFg_003202','hy433gakFeo_001156',
    #           'IQwS7BEjUxM_001783','kjVywUi1WK4_003685','pQdl0apLT70_004976','vdLn-qswnRo_001707','yhtzAKqRyXw_004811',] # bad
    # model_config = '[poor]poma,vst,ins,prompt,rnn,depth=4'
    # scenes = ['D_pyFV4nKd4_001836','Pbw0A-RCjcw_002938','a_VxrUq9PmA_002682','pQdl0apLT70_004976']
    # scenes = ['JHfjuFrnpjA_005930',]
    # model_config = '[final select]'
    
    # Tools.plot_instance_score(pkl_path,model_config,scenes)    

'''
重新生成eval.txt
'''
if __name__ == "__main__":
    pass
    '''
    生成单个模型的eval结果
    '''
    # model_folder = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt,rnn,depth=4/"
    # specific_peoch = [160]
    # Post_Process_Tools.eval_on_epoches(model_folder,specific_peoch=None , val_type='mini_box',  post_process=True) # val_type='mini_box'
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = None , post_process=True)
    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = specific_peoch , post_process=True)

    '''
    sampled_scenes: 对特定的scenes集合(如auc>50部分的scene)求AUC 
    '''
    # # model_folder = "/data/qh/DoTA/output/Prompt/cross_attn_trans_depth_4_load_from_prompt/"
    # # epochs = [80]
    # model_folder = "/data/qh/DoTA/output/Baseline/v4_2/"
    # epochs = [690]
    # big_diff, both_bad = Post_Process_Tools().compare_popr_auc_on_per_scene(model_folder, epochs , save_path='author_baseline.json' )
    # # # 区分：positive 和 negtive big_diff
    # # neg_big_diff = {item[0]:item[1] for item in big_diff.items() if item[1][-1] > 0}
    # # pos_big_diff = {item[0]:item[1] for item in big_diff.items() if item[1][-1] < 0}
    # # scenes = list(neg_big_diff.keys())[:3]+list(pos_big_diff.keys())[:3]+list(big_diff.keys())[-3:]+list(both_bad.keys())[:3]+list(both_bad.keys())[-3:]
    # specific_info = {}
    # specific_info['name'] = 'auc_G50'
    # specific_info['scenes'] = [item[0] for item in both_bad.items() if item[1][-1] > 0.5]

    # Post_Process_Tools.eval_on_epoches(model_folder, specific_peoch = epochs , post_process = False , specific_info = specific_info)

    # Tools = Scene_Aware_Tools(model_folder)
    # Tools.plot_box_area_distribution(specific_info)

    '''
    汇总所有模型的eval结果
    '''
    # path = [          
    #         # baseline
    #         "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8/",
    #         "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=256,fpn(0),prompt(0),rnn,vcl=8/",
    #         "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(0),prompt(1),rnn,vcl=8/",
    #         "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(1),prompt(0),rnn,vcl=8/",
    #         "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(1),prompt(1),rnn,vcl=8/",

    #         # ablation
    #         "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(1),prompt(no_RA),rnn,vcl=8/",
    #         "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(no_RA),rnn,vcl=8/",

    #         # instanc,
    #         "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(0),prompt(1),rnn,vcl=8/",
    #         "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8/",

    #        ]
    
    # save_path = "/data/qh/DoTA/output/debug/eval_resulst_compare/[v3]PromptTAD_eval_compare.json"
    # Post_Process_Tools().Collect_all_AUC(path,save_path, post_process=False) # val_type='mini_box'
    
    # save_path = "/data/qh/DoTA/output/debug/eval_resulst_compare/[v3]PromptTAD_mini_box_eval_compare.json"
    # Post_Process_Tools().Collect_all_AUC(path,save_path, post_process=False, val_type='mini_box') # val_type='mini_box'      

    # save_path = "/data/qh/DoTA/output/debug/eval_resulst_compare/[v3]PromptTAD_popr_eval_compare.json"
    # Post_Process_Tools().Collect_all_AUC(path,save_path,post_process=True) # val_type='mini_box'

'''
逐个场景下AUC计算: 不同pkl文件, 同一个pkl的加不加后处理
'''
if __name__ == '__main__':  
    pass
    '''
    对比两个模型在每个场景上AUC
    # baseline
    "/data/qh/DoTA/output/Baseline/L1_train_v4_2/",
    "/data/qh/DoTA/output/Baseline/Train_lr_Linear_cosineAnn/",
    "/data/qh/DoTA/output/Baseline/v4_2/",
    # fpn and prompt
    "/data/qh/DoTA/output/Prompt/standart_fpn_lr=0.002/",
    "/data/qh/DoTA/output/Prompt/prompt_fpn_lr=0.002/",

    # loss
    "/data/qh/DoTA/output/Prompt/box_w=0.0002/",
    "/data/qh/DoTA/output/Prompt/box_w=0.0002_flip/",
    "/data/qh/DoTA/output/Prompt/box_w=0.0002_obj_w=0.02/",
    "/data/qh/DoTA/output/Prompt/box_w=0.008_frame_w=1/",
    "/data/qh/DoTA/output/Prompt/box_w=0.008_rnn_1024/",
    "/data/qh/DoTA/output/Prompt/box_w=0.008_weighted_res_load_from_flip/",
    "/data/qh/DoTA/output/Prompt/frame_w=1_load_from_prompt/",
    "/data/qh/DoTA/output/Prompt/obj_w=0.8_load_from_prompt/",

    # cross attn
    "/data/qh/DoTA/output/Prompt/cross_attn_load_from_prompt/",
    "/data/qh/DoTA/output/Prompt/cross_attn_frame_loss_load_from_prompt/",
    "/data/qh/DoTA/output/Prompt/cross_attn_aug_frame_load_from_prompt/",
    "/data/qh/DoTA/output/Prompt/cross_attn_trans_depth_4_load_from_prompt/",

    # poma
    "/data/qh/DoTA/poma/base,fpn,rnn,vcl=8,lr=0.002/"

    "baseline_vs_standard_fpn.txt"
    "prompt_vs_standard_fpn.txt"
    "boxloss_vs_prompt.txt"
    '''
    # folder_1 = "/data/qh/DoTA/poma/base,fpn,rnn,vcl=8,lr=0.002/"
    # epoch_1 = 200
    # folder_2 = "/data/qh/DoTA/output/Prompt/cross_attn_trans_depth_4_load_from_prompt/"
    # epoch_2 =  80

    # folder_1 = "/data/qh/DoTA/dinov2/trainer_v2/vit_l,base,rnn,vcl=8/"
    # epoch_1 = 200
    # folder_2 = "/data/qh/DoTA/output/Prompt/cross_attn_trans_depth_4_load_from_prompt/"
    # epoch_2 =  80
    
    # 对比两个模型在每个scene的AUC
    # model_1 = '[normal] dinov2-vit_l'
    # model_2 = 'ori_depth4'
    # save_path = F"{model_1} VS {model_2}.txt"
    # Post_Process_Tools().compare_auc_on_per_scene(folder_1, epoch_1, folder_2 , epoch_2, save_path , post_process=False)

    '''
    对比一个pkl在 正常处理 和 后处理 下每个场景的auc
    '''
    # folder = "/data/qh/DoTA/output/Prompt/cross_attn_trans_depth_4_load_from_prompt/"
    # epochs = [80]
    # save_path = '[self]'+'cross_attn_trans_depth_4_load_from_prompt' + '.txt'
    # Post_Process_Tools().compare_popr_auc_on_per_scene(folder, epochs , save_path )

'''
1. 将 val 拆分到10x5的grid中, 并写入json
2. 任意集合场景下,bbox面积的分布
3. 任意集合场景下,pkl在指定grid上的AUC(包含对比不同pkl)
'''
if __name__ == "__main__":
    pass

    '''
    将 val 拆分到10x5的grid中, 并写入json
    '''
    # folder = "/data/qh/DoTA/data/metadata/"
    # Tools = Scene_Aware_Tools(folder)
    # Tools.split_scenebox_in_grid(data_type='val')
    # Tools.split_scenebox_in_grid(data_type='minibox_val')

    '''
    将 class-aware 的scene 拆分到10x5的grid中, 并写入json / 再根据数量得到heatmap
    '''
    # folder = "/data/qh/DoTA/data/metadata/class_aware"
    # Tools = Scene_Aware_Tools(folder)
    # json_path = "/data/qh/DoTA/data/metadata/class_aware/class_aware.json"
    # class_data = json.load(open(json_path,'r'))
    # Tools.split_scenebox_in_grid(data_type=class_data["non-involve ego"])

    '''
    per-class anomoly scenes 在grid 上的分布
    '''
    # folder = "/data/qh/DoTA/data/metadata/class_aware"
    # Tools = Scene_Aware_Tools(folder)
    # scene_type = ''
    # Tools.class_on_grids(scene_type = scene_type,class_type='compare')
    # Tools.class_on_grids(scene_type = scene_type,class_type='ego')
    # Tools.class_on_grids(scene_type = scene_type,class_type='ego')


    '''
    任意集合场景下,bbox面积的分布
    '''
    # folder = '/data/qh/DoTA/output/debug/'
    # Tools = Scene_Aware_Tools(folder)
    # specific_info = {}
    # specific_info['name'] = 'auc_G50'
    # specific_info['scenes'] = [item[0] for item in both_bad.items() if item[1][-1] > 0.5]
    # Tools.plot_box_pos_distribution(specific_info = specific_info , bins = (10,5))

    '''
    对比不同pkl在指定grid上的AUC
    '''
    # folder = '/data/qh/DoTA/output/debug/'
    # Tools = Scene_Aware_Tools(folder)

    # model_name = ['[fig][val]baseline','[fig][val]full_model_v2']
    # grid_sceene_path = "/data/qh/DoTA/data/metadata/val_split_in_grid_10x5.json"
    
    # # model_name = ['[mini]baseline','[mini]full_model_v2']
    # # grid_sceene_path = "/data/qh/DoTA/data/metadata/minibox_val_split_in_grid_10x5.json"

    # pkl_path = ["/data/qh/DoTA/output/Baseline/v4_2/eval/results-690.pkl", 
    #            "/data/qh/DoTA/poma_v2/instance/[v2]vst,ins,prompt,rnn,depth=4/eval/results-120.pkl"]
    # Tools.eval_on_grids(model_name=model_name,grid_sceene_path=grid_sceene_path,pkl_path=pkl_path,no_bar=True)


    '''
    单个场景下在grid上的AUC
    '''
    # folder = '/data/qh/DoTA/output/debug/'
    # Tools = Scene_Aware_Tools(folder)
    # epochs = [80]
    # big_diff, both_bad = Post_Process_Tools().compare_popr_auc_on_per_scene(model_folder, epochs , save_path=None )
    # both_bad_scenes = list(both_bad.keys())[::-1]
    # for i in range(10):
    #     specific_info = {}
    #     specific_info['name'] = both_bad_scenes[i]
    #     specific_info['scenes'] = [both_bad_scenes[i]]
    #     Tools.plot_box_pos_distribution(specific_info = specific_info , bins = (10,5))

'''
对比框
1. gt score在每帧/单独gt框/单独yolo框
2. 对比 yolo 和 gt 
3. 对比 yolo 和 outputs
'''
if __name__ == "__main__":
    pass
    #  gt score在每帧
    # folder = '/data/qh/DoTA/output/debug/box_compare' 
    # cmp = Boxes_Comparison(folder)
    # scenes =['PqbpIHZvjMA_002926']
    # # scenes = ['D_pyFV4nKd4_001836','a_VxrUq9PmA_002682','Pbw0A-RCjcw_002938','pQdl0apLT70_004976']
    # cmp.plot_gt_score_on_scenes(scenes)

    # 单独gt框
    # folder = '/data/qh/DoTA/output/debug/box_compare' 
    # cmp = Boxes_Comparison(folder)
    # scenes = ['Eq7_uD2yN5Y_004339']
    # cmp.draw_gt_box(scenes)
    # cmp.plot_gt_score(scenes)


    # 单独yolo框
    # folder = '/data/qh/DoTA/data' 
    # cmp = Boxes_Comparison(folder)
    # scenes = ['a_VxrUq9PmA_002682','D_pyFV4nKd4_001836','Pbw0A-RCjcw_002938','a_VxrUq9PmA_002682','pQdl0apLT70_004976']
    # scenes = ['a_VxrUq9PmA_002682']
    # cmp.draw_yolo_box(scenes)


    # # 对比 yolo 和 gt     
    # folder = '/data/qh/DoTA/output/debug/box_compare' 
    # cmp = Boxes_Comparison(folder)
    # train_info, val_info = cmp.calculate_box_in_scenes()
    # scenes = list(val_info.keys())[-10:] + list(val_info.keys())[0:5]
    # cmp.compare_boxes_yolo_and_gt(scenes)

    # 对比 yolo 和 outputs
    '''
    "/data/qh/DoTA/output/Prompt/box_loss_weight=0.0002/eval/results-200.pkl"
    "/data/qh/DoTA/output/Prompt/box_loss_weightedresnet/eval/results-200.pkl"
    '''
    # folder = '/data/qh/DoTA/output/debug/box_compare' 
    # cmp = Boxes_Comparison(folder)
    # # with open("/data/qh/DoTA/data/metadata/minibox_val_split.txt",'r') as f:
    # #     scenes = f.read().split()[:10]
    # scenes = os.listdir("/data/qh/DoTA/output/debug/box_compare/output_yolo/basic_box_loss/")
    # pkl_path = "/data/qh/DoTA/output/Prompt/box_loss_weightedresnet/eval/results-200.pkl"
    # model_config = 'weighted_resnet_epoch_200'
    # cmp.compare_boxes_outputs_and_yolo(pkl_path, model_config ,scenes)



