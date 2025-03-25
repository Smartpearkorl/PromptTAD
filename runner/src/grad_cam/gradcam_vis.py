import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict
import torch
import os
from collections import OrderedDict
from tqdm import tqdm as tqdm
from torchvision import transforms
import json

# Custom imports
import sys
from pathlib import Path
FILE = Path(__file__).resolve() # /home/qh/TDD/PromptTAD/runner/src/grad_cam/gradcam_vis.py
sys.path.insert(0, str(FILE.parents[3]))
import os 
os.chdir(FILE.parents[3])

from runner.src.metrics import  AUC_on_scene
from runner.src.tools import custom_print 
from runner.src.data_transform import pad_frames
from runner.src.grad_cam.gradcam import GradCAM, show_cam_on_image, center_crop_img, gradcam_vis,ResizeTransform
from runner.src.custom_tools import Post_Process_Tools as PP_Tools
from models.prompt_models.points_transform import ResizeCoordinates
from runner import DATA_FOLDER, META_FOLDER, FONT_FOLDER, DEBUG_FOLDER,CHFONT_FOLDER, DADA_FOLDER

torch.backends.cudnn.enabled = False # when RNN is in model, cudnn need train mode
def read_file(path):
    return np.asarray(Image.open(path))

'''
同一个scene下不同模型grad cam可视化放在一起
'''
def plt_grad_cam(folder_1, folder_2, scenes, model_1 = None, model_2 = None):
    
    def find_and_split_subfolder(parent_dir, prefix):
        for subdir in os.listdir(parent_dir):
            if subdir.startswith(prefix):
                part2 = subdir[len(prefix)+1:]
                return subdir, part2
        return None, None
   
    cfg_name_1 = folder_1.split('/')[-1] if folder_1.split('/')[-1] else folder_1.split('/')[-2]
    cfg_name_2 = folder_2.split('/')[-1] if folder_2.split('/')[-1] else folder_2.split('/')[-2]
    vis_path_1 = os.path.join(folder_1,f'Gradcam_Frame')
    vis_path_2 = os.path.join(folder_2,f'Gradcam_Frame')
    save_name = f'{model_1}_vs_{model_2}' if model_1 is not None else f'{cfg_name_1} vs {cfg_name_2}'
    save_folder = os.path.join('/data/qh/DoTA/output/debug/gradcam_compare',save_name)
    os.makedirs(save_folder,exist_ok=True)
    scenes = scenes if isinstance(scenes,list) else [scenes]
    
    for scene in scenes:
        # 如果当前文件夹已经生成了grad_cam则跳过
        if os.path.exists(os.path.join(save_folder,scene)):
                continue
        # 找到对应的文件夹，文件夹格式为：3sGShQb_HwU_001607_AUC-100.00
        image_folder_1, auc_1  = find_and_split_subfolder(vis_path_1,scene)
        image_folder_2, auc_2  = find_and_split_subfolder(vis_path_2,scene)
        if not image_folder_1:
            print(f'no grad cam visulization of {scene} in {vis_path_1}')
            continue   
        if not image_folder_2:
            print(f'no grad cam visulization of {scene} in {vis_path_2}')
            continue
        image_folder_1 = os.path.join(vis_path_1,image_folder_1)
        image_folder_2 = os.path.join(vis_path_2,image_folder_2)
        # 准备text
        text_1 = f'{model_1} {auc_1}' if  model_1  else f'{cfg_name_1} {auc_1}'
        text_2 = f'{model_2} {auc_2}' if  model_2  else f'{cfg_name_2} {auc_2}'
        assert len(os.listdir(image_folder_1)) == len(os.listdir(image_folder_2)), \
                f'{image_folder_1} has {len(os.listdir(image_folder_1))} images but {image_folder_1} has {len(os.listdir(image_folder_2))}'
        save_folder_scene = os.path.join(save_folder,scene)
        os.makedirs(save_folder_scene,exist_ok=True)
        for grad_img in tqdm(os.listdir(image_folder_1),desc="Combine img: "):
            grad_img_path_1 = os.path.join(image_folder_1,grad_img)
            grad_img_path_2 = os.path.join(image_folder_2,grad_img)
            assert os.path.exists(grad_img_path_2),f'{grad_img_path_2} is not existed' 
            image1 = Image.open(grad_img_path_1)
            image2 = Image.open(grad_img_path_2)
            width1, height1 = image1.size
            width2, height2 = image2.size
            new_width = width1 + width2
            new_height = max(height1, height2) + 50  # 额外的高度用于添加文本
            new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
            new_image.paste(image1, (0, 0))
            new_image.paste(image2, (width1, 0))
            draw = ImageDraw.Draw(new_image)      
            # 在图像1下方添加文本
            # font = ImageFont.load_default()
            font = ImageFont.truetype(FONT_FOLDER,40)
            text1_bbox = draw.textbbox((0, 0), text_1, font=font)
            text1_width = text1_bbox[2] - text1_bbox[0]
            text1_x = (width1 - text1_width) // 2
            draw.text((text1_x, height1), text_1, fill=(0, 0, 0), font=font)

            # 在图像2下方添加文本
            text2_bbox = draw.textbbox((0, 0), text_2, font=font)
            text2_width = text2_bbox[2] - text2_bbox[0]
            text2_x = width1 + (width2 - text2_width) // 2
            draw.text((text2_x, height2), text_2, fill=(0, 0, 0), font=font)
            new_image.save(os.path.join(save_folder_scene,grad_img))

'''
grad_cam 可视化：
1. 传入modelcfg.pkl , 加载模型
2. 传入result.pkl   
'''
class vis_grad_cam():
    def __init__(
            self,
            folder: str,
            epoch: int,
            loss_type: str='Frame_loss', # grad loss 的类型： Frame_loss or Instance_loss
    ):
        self.root_path = DATA_FOLDER
        self.folder = folder
        self.epoch = epoch
        self.loss_type = loss_type
        self.cfg_name = folder.split('/')[-1] if folder.split('/')[-1] else folder.split('/')[-2]
        custom_print(f'model folder name: {self.cfg_name}')
        # prepare dir
        self.cfg_path = os.path.join(self.folder,"modelcfg.pkl")
        self.ckp_path = os.path.join(self.folder,'checkpoints',f'model-{self.epoch:02d}.pt')
        self.pkl_path = os.path.join(self.folder,"eval",f'results-{self.epoch:02d}.pkl')
        assert os.path.exists(self.cfg_path), f'file {self.cfg_path} is not existed'    
        assert os.path.exists(self.ckp_path), f'file {self.ckp_path} is not existed'    
        assert os.path.exists(self.pkl_path), f'file {self.pkl_path} is not existed'
        # read config
        self.cfg = pickle.load(open(self.cfg_path,'rb'))
        # set device
        self.cfg.device , self.cfg.input_shape = 'cpu' ,  [480, 640]     
        if loss_type == 'Frame_loss':                 
            self.save_path = os.path.join(self.folder,f"Gradcam_Frame")
            self.target_size = (480 , 640) 
            self.original_size = (720, 1280) # default conifg for draw box  
        elif loss_type == 'Instance_loss':
            self.save_path = os.path.join(self.folder,f"Gradcam_Instance")
            self.target_size = self.cfg.ins_encoder.resize.target_size
            self.original_size = self.cfg.ins_encoder.resize.original_size       
        else:
            raise Exception(f'loss_type : {loss_type} is not supported' )   
        self.box_transform = ResizeCoordinates(self.target_size, self.original_size ) # box processing   
        os.makedirs(self.save_path,exist_ok=True)

        #prepare image transform
        self.data_transform = transforms.Compose([
                                                pad_frames(self.cfg.input_shape),
                                                transforms.Lambda(lambda x: torch.tensor(x)),
                                                # [T, H, W, C] -> [T, C, H, W]
                                                transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
                                                transforms.Lambda(lambda x: x / 255.0),
                                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])   
        #prepare model
        self.model , self.target_layers = self.load_model(self.cfg, self.ckp_path)


    def load_model(self, modelcfg, ckp_path):   
        assert modelcfg.model_type == 'poma', f'unsupported model type : {modelcfg.model_type}'
        model =  modelcfg.model( dinov2_cfg = modelcfg.dinov2 ,
                                    clip_cfg = modelcfg.clip,
                                    vst_cfg = modelcfg.vst,
                                    fpn_cfg = modelcfg.fpn,
                                    ins_encoder_cfg = modelcfg.ins_encoder , 
                                    bottle_aug_cfg = modelcfg.bottle_aug, 
                                    ins_decoder_cfg = modelcfg.ins_decoder, 
                                    ano_decoder_cfg = modelcfg.ano_decoder,
                                    proxy_task_cfg = modelcfg.proxy_task)
        assert  modelcfg.vst, f'only support vst as backbone for gradcam'
        if self.loss_type == 'Frame_loss':
            self.vst_norm = True
            target_layers = [model.vst_model.norm]
            # target_layers = [model.vst_model.layers[3].blocks[1].norm1]
        elif self.loss_type == 'Instance_loss':
            target_layers = [model.vst_model.norm]
            self.vst_norm = True
            # target_layers = [model.instance_encoder.prompt_decoder.transformer.layers[3].norm4]
            # self.vst_norm = False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        # load checkpoints
        custom_print(f'load file {ckp_path}')
        checkpoint = torch.load(ckp_path, map_location = modelcfg.device)
        # 判断是不是nn.DataParallel包装的模型
        if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
            weights = checkpoint['model_state_dict']
            checkpoint['model_state_dict'] = OrderedDict([(k[7:], v) for k, v in weights.items()])
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        assert not unexpected_keys and not missing_keys , f'{unexpected_keys=}\n{missing_keys=}'
        return model, target_layers

    
    # get yolov9 boxes
    def get_yolo_boxes(self, video_name, sub_batch):
        boxes_dir = os.path.join(self.root_path, "yolov9",video_name+'.json')
        with open(boxes_dir, 'r', encoding='utf-8') as file:
            yolov9 = json.load(file)
        yolo_boxes = []
        for frame_data in yolov9['lables'][sub_batch['begin']:sub_batch['end']]:
            # 如果当前帧没有物体返回的是空array([],dtype=float64)
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            yolo_boxes.append(np.array(boxes))        
        return yolo_boxes

    def draw_boxes(self, image, boxes, obj_dict={}):
        N,h,w,c = image.shape
        image_with_boxes = []
        box_color  = {1:'red',2:'blue',3:'black',4:'yellow',5:'green' }
        text_color = ['red','green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        font = ImageFont.truetype(FONT_FOLDER,20)
        for i in range(N):
            img_pillow = Image.fromarray(image[i].astype(np.uint8))
            draw = ImageDraw.Draw(img_pillow)
            obj_score = obj_dict[i] if i in obj_dict else []
            if len(obj_score):
                assert  len(obj_score) == boxes[i].shape[0] , f' No.{i} frame has {boxes[i].shape[0]} yolo boxes , {len(obj_score)} obj scores'
            # Draw boxes on the image
            if boxes[i].shape[0] != 0:
                for ind, box in enumerate(boxes[i]):
                    draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=box_color[1], width=2)
                    if len(obj_score): 
                        draw.text( (box[0],box[1]), f"{ind} : {obj_score[ind]*100:.2f}",fill=text_color[0], font=font)
                    else:
                        draw.text( (box[0],box[1]), f"{ind}",fill=text_color[0], font=font)
                
            image_array = np.array(img_pillow).astype('float32')
            image_with_boxes.append(image_array)
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes
    
    '''
    在图像上画框, 并可能有每个框的特有信息： 用于 instance anomal score
    '''
    def draw_boxes_with_info(self, image, boxes_list, info_list=None, write_index = False):
        N,h,w,c = image.shape
        box_color  = ['red', 'blue', 'black' ,'yellow' , 'green'] 
        text_color = ['red', 'green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        font = ImageFont.truetype(FONT_FOLDER,25)
        info_list = info_list if info_list else [[] for _ in range(N)]
        image_with_boxes = []
        for i in range(N):
            image_pil = Image.fromarray(image[i].astype(np.uint8))
            draw = ImageDraw.Draw(image_pil)
            # pad empty when no info 
            if len(info_list[i])==0:
                info_list[i] = [[] for _ in range(boxes_list[i].shape[0])]
            for ind,(box,info) in enumerate(zip(boxes_list[i],info_list[i])):
                if write_index:
                    info = f'{ind} {info}'
                draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=box_color[0], width=2)
                if len(info):
                    draw.text( (box[0],box[1]), f"{info}",fill=text_color[0], font=font) 
            image_with_boxes.append(np.array(image_pil))
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes

    def save_auc_on_scene(self,vis_scenes,all_scenes):
        txt_path = os.path.join(self.save_path,f'epoch-{self.epoch}_auc_on_scenes.txt')
        vis_scenes = {scene:all_scenes[scene] for scene in vis_scenes}
        vis_scenes_data = json.dumps(vis_scenes, indent=4)
        all_scenes_data = json.dumps(all_scenes, indent=4)
        with open(txt_path, 'w') as file:
            file.write(f'visualization on scenes as follows\n')
            file.write(vis_scenes_data)
            file.write(f'\nvisualization on scenes as follows\n')
            file.write(all_scenes_data)
        custom_print(f'epoch {self.epoch} AUC score on per scene save to {txt_path}')
    
    def check_duplicated_scene(self,scene_name):
        for sub_scene in os.listdir(self.save_path):
            if sub_scene.startswith(scene_name):
                print(f'{scene_name} has been in {self.save_path}')
                return True
        return False

    def forward(self,
                good_num:int=5, # AUC 最好的前几个scenes 
                bad_num:int=5, # AUC 最差的前几个scenes
                specific_scenes: list=None, # 指定scenes 
                post_process = False,
                draw_box = True,
                ):
        
        all_scenes = AUC_on_scene(self.pkl_path,post_process=post_process)
        # 指定 scenes or auc最小场景all_scenes:  Dict{'scene_name' : auc_score,}
        scenes = list(all_scenes.keys())[-good_num:]+list(all_scenes.keys())[:bad_num] if not specific_scenes else specific_scenes
        auc_scores =[all_scenes[x] for x in scenes]
        scenes  = scenes if isinstance(scenes, list) else [scenes]
        # 保存每个scene的AUC得分    
        self.save_auc_on_scene(scenes,all_scenes)

        img_size_h, img_size_w = self.cfg.input_shape
        # ground truth of scenes
        with open( META_FOLDER / 'metadata_val.json', 'r') as f:
            scenes_gt = json.load(f)
        
        scenes_gt = {key:scenes_gt[key] for key in scenes}

        # yolo instanced matched index with gt
        if self.loss_type == 'Instance_loss':
            with open( META_FOLDER / 'yolo_instance_match.json', 'r') as f:
                instance_matches = json.load(f)
        
        # cfg.NF 
        NF_sub_1 = self.cfg.NF - 1
        for i,scene in enumerate(scenes):
            # 如果当前文件夹已经生成了grad_cam则跳过
            # if self.check_duplicated_scene(scene):
            #     continue
            scean_path = DATA_FOLDER / 'frames' / f'{scene}' / 'images'
            video_frames = sorted(os.listdir(scean_path))
            # snippets 在scene 起始帧和结束帧 
            sub_batch={"begin":0,"end":len(video_frames)}
            # sub_batch={"begin":0,"end":5}
            names = [os.path.join(scean_path,name) for name in video_frames[sub_batch["begin"]:sub_batch["end"]]]
            frames = np.array(list(map(read_file, names))).astype('float32')
            yolo_boxes = self.get_yolo_boxes(scene,sub_batch)
            fixed_yolo_boxes = [self.box_transform.apply_boxes(x) for x in yolo_boxes ]
            # [T, C, H, W]
            img_tensor = self.data_transform(frames)
            frames = pad_frames(self.cfg.input_shape)(frames) # to plot in image

            # expand batch dimension
            # [T, C, H, W] -> [B, T, C, H, W] -> [B, C, T , H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            t_type = 'Frame_loss' if self.vst_norm else self.loss_type # 如果是 vst_norm ，强制ResizeTransform为 Frame_loss 格式
            cam = GradCAM(cfg=self.cfg,model=self.model, target_layers=self.target_layers, use_cuda=True,
                         reshape_transform=ResizeTransform(im_h=img_size_h, im_w=img_size_w, t_type = t_type))
            target_category = 1 
            snippets_gradcam, outputs = cam(video_data = input_tensor , yolo_boxes = yolo_boxes, \
                                            target_category=target_category, loss_type = self.loss_type ,vst_norm = self.vst_norm)
            
            # continue

            # add boxes , 如果要加入object level 的anomal score ，需要先拿到 score ,故放在GradCAM后。   
            if self.loss_type == 'Frame_loss' and draw_box:                
                frames = self.draw_boxes_with_info(frames, fixed_yolo_boxes, info_list=None, write_index = False)
            
            # prepare instance box info: Frame_loss 只标出框的索引； Instance_loss 标出框 标签和异常得分               
            if self.loss_type == 'Instance_loss':
                instance_scores = {}
                for index , frame_data in zip(range(sub_batch["begin"],sub_batch["end"]),snippets_gradcam):
                    instance_scores[index+NF_sub_1] = frame_data['obj_score']
            
                # matched_ind 格式： dict {frame_index:[[yolo_index],[gt_index]],}
                matched_ind = instance_matches[scene] 
                
                # 得到 instance level label :
                # instance_labels： 格式： dict {frame_index:[labels,],}
                instance_labels = {}
                for keys in instance_scores.keys():
                    gt_index = matched_ind[f'{keys}'][0]
                    gt_labes = np.zeros(len(instance_scores[keys]), dtype=int)
                    gt_labes[gt_index] = 1
                    instance_labels[keys] = gt_labes 
                
                # 得到 每个框准备标注信息
                # info： 格式： dict {frame_index:['0 -> 1.0', '0 -> 0.0',...],}
                instance_info = {}
                for keys in instance_scores.keys():
                    obj_tar , obj_out = instance_labels[keys] , instance_scores[keys]
                    instance_info[keys] = [[f'{l} -> {s:.1f}'] for l,s in  zip(list(obj_tar),list(obj_out))]
            
            # frame loss plot
            if self.loss_type == 'Frame_loss':
                save_path = os.path.join(self.save_path,f'{scene}_AUC-{(auc_scores[i]*100):.2f}')
                os.makedirs(save_path,exist_ok=True)
                for index,grayscale_cam in enumerate(tqdm(snippets_gradcam, desc="Draw Picture ")):
                    # 修正index , 因为开始不一定是第0帧
                    grayscale_cam = grayscale_cam[0, :]
                    fixed_index = sub_batch["begin"] + index
                    #gradcam vis: ndarray(h,w)  0~1 
                    visualization = show_cam_on_image(frames[fixed_index+NF_sub_1] / 255., grayscale_cam, use_rgb=True ,crop_size=(self.target_size,self.original_size))
                    #curve_plot
                    gradcam_vis(visualization,outputs,save_path,fixed_index,scenes_gt[scene])
                    # 额外存gradcam可视化(不带异常得分折线图)： 论文中用于展示
                    plt.imshow(visualization)
                    plt.axis('off')
                    folder = os.path.join("/data/qh/DoTA/output/debug/gradcam_compare/[final-select]baseline/",scene)
                    # folder = os.path.join("/data/qh/DoTA/output/debug/gradcam_compare/[final-select]fullmodel/",scene)
                    # folder = os.path.join("/data/qh/DoTA/output/debug/gradcam_compare/[layer3 norm1][final-select]baseline/",scene)    
                    # folder = os.path.join("/data/qh/DoTA/output/debug/gradcam_compare/[layer3 norm1][final-select]fullmodel/",scene)  
                    os.makedirs(folder,exist_ok=True)
                    savename = os.path.join(folder,f'gradcam_{index+3}.jpg')          
                    plt.savefig(savename,dpi=300)
                    plt.clf() 
            
            #  在 Instance_loss 下会查看 instance 的 grad cam ，为了便于观察，只画出单个instance的框
            elif self.loss_type == 'Instance_loss':
                scene_save_path = os.path.join(self.save_path,f'{scene}_AUC-{(auc_scores[i]*100):.2f}')
                os.makedirs(scene_save_path,exist_ok=True)
                for index, frame_data in enumerate(tqdm(snippets_gradcam, desc="Draw Picture ")):
                    # 修正index , 因为开始不一定是第0帧
                    fixed_index = sub_batch["begin"] + index
                    frame_gradcams , obj_gradcams, obj_scores = frame_data['frame_gradcam'], frame_data['obj_gradcam'], frame_data['obj_score']
                    if self.vst_norm:
                        # instance loss to every box GradCAM
                        frame_save_path = os.path.join(scene_save_path,'[vst_emb]instance_loss')
                        os.makedirs(frame_save_path,exist_ok=True) 
                        for sub_ind , obj_cam in enumerate(obj_gradcams):
                            obj_dic ={'obj_ind': sub_ind, 'obj_score': obj_scores[sub_ind]}
                            obj_cam = obj_cam[0,:]
                            # 给当前图像画对应的instance box 并标注 score,
                            '''
                            # 注意: draw_boxes_with_info ( now_frame: np.shape(N,h,w,3) , boxes_list : [N: np.shape(N_points,4)]  info_list: [N:[N_points]]
                            故： frame, box 为np,取一的时候要增加一维,且boxes_list,info_list送的是list,故送入函数还要外接一个[]
                            '''
                            now_frame, now_box = np.expand_dims(frames[fixed_index+NF_sub_1], axis=0), np.expand_dims(fixed_yolo_boxes[fixed_index+NF_sub_1][sub_ind], axis=0)
                            now_frame = self.draw_boxes_with_info(now_frame, [now_box],[instance_info[fixed_index+NF_sub_1][sub_ind]])
                            visualization = show_cam_on_image(now_frame[0]/ 255., obj_cam, use_rgb=True)
                            gradcam_vis(visualization,outputs,frame_save_path,fixed_index,scenes_gt[scene], level='instance',obj_dic = obj_dic )                  
                    else:
                        # frame loss to every box GradCAM
                        frame_save_path = os.path.join(scene_save_path,'[ins_emb]frame_loss')
                        os.makedirs(frame_save_path,exist_ok=True) 
                        for sub_ind , obj_cam in enumerate(frame_gradcams):
                            obj_dic ={'obj_ind': sub_ind, 'obj_score': obj_scores[sub_ind]}
                            # 给当前图像画对应的instance box 并标注 score,
                            '''
                            # 注意: draw_boxes_with_info ( now_frame: np.shape(N,h,w,3) , boxes_list : [N: np.shape(N_points,4)]  info_list: [N:[N_points]]
                            故： frame, box 为np,取一的时候要增加一维,且boxes_list,info_list送的是list,故送入函数还要外接一个[]
                            '''
                            now_frame, now_box = np.expand_dims(frames[fixed_index+NF_sub_1], axis=0), np.expand_dims(fixed_yolo_boxes[fixed_index+NF_sub_1][sub_ind], axis=0)
                            now_frame = self.draw_boxes_with_info(now_frame, [now_box],[instance_info[fixed_index+NF_sub_1][sub_ind]])
                            visualization = show_cam_on_image(now_frame[0]/ 255., obj_cam, use_rgb=True)
                            gradcam_vis(visualization,outputs,frame_save_path,fixed_index,scenes_gt[scene], level='instance',obj_dic = obj_dic )

                        # instance loss to every box GradCAM
                        instance_save_path = os.path.join(scene_save_path,'[ins_emb]instance_loss')
                        os.makedirs(instance_save_path,exist_ok=True) 
                        for sub_ind , obj_cam in enumerate(obj_gradcams):
                            obj_dic ={'obj_ind': sub_ind, 'obj_score': obj_scores[sub_ind]}
                            obj_cam = obj_cam[0,:]
                            # 给当前图像画对应的instance box 并标注 score,
                            '''
                            # 注意: draw_boxes_with_info ( now_frame: np.shape(N,h,w,3) , boxes_list : [N: np.shape(N_points,4)]  info_list: [N:[N_points]]
                            故： frame, box 为np,取一的时候要增加一维,且boxes_list,info_list送的是list,故送入函数还要外接一个[]
                            '''
                            now_frame, now_box = np.expand_dims(frames[fixed_index+NF_sub_1], axis=0), np.expand_dims(fixed_yolo_boxes[fixed_index+NF_sub_1][sub_ind], axis=0)
                            now_frame = self.draw_boxes_with_info(now_frame, [now_box],[instance_info[fixed_index+NF_sub_1][sub_ind]])
                            visualization = show_cam_on_image(now_frame[0]/ 255., obj_cam, use_rgb=True)
                            gradcam_vis(visualization,outputs,instance_save_path,fixed_index,scenes_gt[scene], level='instance', obj_dic = obj_dic )

'''
对比两个模型在每个场景上Grad Cam
'''
if __name__ == '__main__':
    pass
    '''
    对比两个模型在每个场景上AUC
    '''
    folder_1 =  "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8"
    epoch_1 = 160
    folder_2 = "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8"
    epoch_2 =  120
    # # 对比两个模型在每个scene的AUC
    model_1 = '[no norm][poma v2 selected]baseline'
    model_2 = 'fullmodel'
    # save_path = F"{model_1}_vs_{model_2}.txt"
    # big_diff, both_bad = PP_Tools.compare_auc_on_per_scene(folder_1, epoch_1, folder_2 , epoch_2, save_path=None)
    # scenes = list(big_diff.keys())[:5] + list(both_bad.keys())[:3] + list(both_bad.keys())[-3:]
    # scenes = ['Pbw0A-RCjcw_002938','ha-IeID24As_001147','Q7VBPeGwJWw_000261','ha-IeID24As_001147','fWJbp43k644_001672'] 
    # scenes = ['fWJbp43k644_001672','p-fBcE77G4c_005054','r6_ZhT7rmhM_000644','YBEYOS3A3Ic_001426','4K_6s1n6BpU_001222','DpQBo4gb6Lw_001365',
    #           'Eq7_uD2yN5Y_004339','gZbGLa253Ak_000932',]

    # scenes = ['a_VxrUq9PmA_002682','Pbw0A-RCjcw_002938','PqbpIHZvjMA_002926','Hd2IzHAfkCI_001091','HzVbo46kkBA_004124',
    #         'Q7VBPeGwJWw_001631','D_pyFV4nKd4_001836',]

    # scenes = ['D_pyFV4nKd4_001836','Pbw0A-RCjcw_002938','a_VxrUq9PmA_002682','pQdl0apLT70_004976']
    scenes = ['JHfjuFrnpjA_005930']
    # 单个模型得到grad_cam
    # vis = vis_grad_cam(folder_1,epoch_1)
    # vis.forward(good_num=1,bad_num=1,specific_scenes = scenes, draw_box=False)
    # 单个模型得到grad_cam
    # vis = vis_grad_cam(folder_2,epoch_2)
    # vis.forward(good_num=1,bad_num=1,specific_scenes = scenes, draw_box=False)
    # 将grad_cam 画在一起
    plt_grad_cam(folder_1,folder_2,scenes,model_1,model_2)


'''
instance gradcam
'''
if __name__ == '__main__':
    pass
    '''
    "/data/qh/DoTA/output/Prompt/box_w=0.0002_obj_w=0.02/"
    /data/qh/DoTA/output/Prompt/obj_w=0.8_load_from_prompt

    specific_scenes=['Nc9SDBksPPA_000075']
    scenes=['lmykriTbncM_003990']
    scenes = ['RPNiqsdt6-w_004024']
    '''
    # folder_1 = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt,rnn,depth=4/"
    # epoch_1 = 160
    # pkl_path = os.path.join(folder_1,'eval',f'results-{epoch_1}.pkl')
    # # all_scenes = AUC_on_scene(pkl_path,auc_type = 'instance', post_process=False)
    # # scenes = list(all_scenes.keys())[-5:]
    # scenes = ['pOJpwWmEcFg_005342','OPj7lUQOre8_004526','8dI7OolIEXY_000708']
    # # scenes = ['eWWgJznGg6U_006506']
    # vis = vis_grad_cam(folder_1,epoch_1,loss_type = 'Instance_loss')
    # vis.forward(good_num=5,bad_num=5,specific_scenes = scenes)
    # # vis.forward(good_num=5,bad_num=5)
    # # vis = vis_grad_cam(folder_1,epoch_1)
    # # vis.forward(good_num=5,bad_num=5)


