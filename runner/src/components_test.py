from alchemy_cat.dl_config import load_config, Config ,Param2Tune,IL
from alchemy_cat.py_tools import Logger,get_local_time_str
import torch
import argparse
import os
import sys

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import json

from runner.src.dota import prepare_dataset
from runner.src.tools import *
from runner.src.utils import resume_from_checkpoint , prepare_optim_sched


'''
Damn it !!!!!!!! 
对比模型参数
'''
def compare_para():
    from collections import OrderedDict
    path = '/data/qh/DoTA/pretrained/swin_base_patch244_window1677_sthv2.pth'
    path_1 = "/data/qh/DoTA/output/Baseline/L2_train_v4_2/checkpoints/model-180.pt"
    path_2 = "/data/qh/DoTA/output/Baseline/L2_train_v4_2/checkpoints/model-20.pt"

    with open(path, "rb") as f:
        checkpoint = torch.load(f)

    m0 = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        if 'backbone' in k:
            name = 'module.model.' + k[9:]
            m0[name] = v

    with open(path_1, "rb") as f:
        data_1 = torch.load(f)
    with open(path_2, "rb") as f:
        data_2 = torch.load(f)
    
    print()
    m1,m2 = data_1['model_state_dict'], data_2['model_state_dict']
    com01 , com02 , com12= {} , {} , {}
    for k,v in m0.items():
        com01[k] = torch.allclose(v.cuda(),m1[k])
        com02[k] = torch.allclose(v.cuda(),m2[k].cuda())
    
    for k in m1.keys():
        com12[k] = torch.allclose(m1[k],m2[k])
    
    print()
        
def compare_ckp():
    p1 = "/data/qh/DoTA/poma/prompt,rnn,vcl=8,lr=0.002/checkpoints/model-99.pt"
    p2 = "/data/qh/DoTA/poma/prompt,rnn,vcl=8,lr=0.002/checkpoints/model-120.pt"
    ckp1 , ckp2 = torch.load(p1) , torch.load(p2)
    d1 , d2 = ckp1['optimizer_state_dict'] , ckp2['optimizer_state_dict']
    pl = []
    for (k1,v1),(k2,v2) in zip(d1['state'].items(),d2['state'].items()):
        pl.append(v1['momentum_buffer'].shape==v2['momentum_buffer'].shape)
        print(f"{v1['momentum_buffer'].shape} {v2['momentum_buffer'].shape}")

    print()

    
def parse_config():
    parser = argparse.ArgumentParser(description='PAMA implementation')

    parser.add_argument('--local_rank','--local-rank',
                        type=int,
                        default=0,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--distributed',
                        action='store_true',
                        help='if DDP is applied.')
    
    parser.add_argument('--fp16',
                        action='store_true',
                        help='if fp16 is applied.')
    
    parser.add_argument('--phase',
                    default='train',
                    choices=['train', 'test', 'play'],
                    help='Training or testing or play phase.')
    
    parser.add_argument('--num_workers',
                    type=int,
                    default=1,
                    metavar='N',)
    
    help_epoch = 'The epoch to restart from (training) or to eval (testing).'
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help=help_epoch)

    parser.add_argument('--output',
                        default='/home/qh/TDD/pama/tmp_output',
                        help='Directory where save the output.')
    
    args = parser.parse_args()
    cfg = vars(args)

    device = torch.device(f'cuda:{cfg["local_rank"]}') if torch.cuda.is_available() else torch.device('cpu')
    n_nodes = torch.cuda.device_count()
    cfg.update(device=device)
    cfg.update(n_nodes=n_nodes)
    return cfg

class plt_tools():
    def __init__(self) -> None:
        pass

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

    @staticmethod
    def draw_with_mask_and_box(image, boxes, masks, info_list=None):
        assert boxes.shape[0] == masks.shape[0], f'length of boxes is {boxes.shape[0]} while masks is {masks.shape[0]}'
        plt.figure(figsize=(15, 15))
        plt.imshow(image)
        colors = [np.concatenate([np.random.random(3), np.array([0.6])], axis=0) for _ in range(boxes.shape[0])]
        for color,mask,box in zip(colors,masks,boxes):
            plt_tools.show_mask(mask.cpu().numpy(), plt.gca(), color)
            plt_tools.show_box(box.cpu().numpy(), plt.gca(),color)
        plt.axis('off')
        return plt.gcf()

   
    '''
    在图像上画框, 并可能有每个框的特有信息： 用于 instance anomal score
    '''
    @staticmethod
    def draw_boxes_with_info(image, boxes_list, info_list=None):
        N,h,w,c = image.shape
        box_color  = ['red', 'blue', 'black' ,'yellow' , 'green'] 
        text_color = ['red', 'green','yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
        font = ImageFont.truetype("/home/qh/TDD/movad/grad_cam/arial.ttf",25)
        info_list = info_list if info_list else [[]]*N
        image_with_boxes = []
        for i in range(N):
            image_pil = Image.fromarray(image[i])
            draw = ImageDraw.Draw(image_pil)
            info_list[i] = info_list[i] if info_list[i] else [[]]*boxes_list[i].shape[0]
            for box,info in zip(boxes_list[i],info_list[i]):
                draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline=box_color[0], width=2)
                if len(info):
                    draw.text( (box[0],box[1]), f"{info}",fill=text_color[0], font=font) 
            image_with_boxes.append(np.array(image_pil))
        image_with_boxes =  np.array(image_with_boxes)
        return image_with_boxes
    

'''
test model compomnents
'''
class Component_Tools():
    def __init__(self, is_init_trainer) -> None:

        self.yolo_folder = '/data/qh/DoTA/data/yolov9/'
        self.gt_folder = "/data/qh/DoTA/data/annotations/"
        self.image_folder = "/data/qh/DoTA/data/frames/"
        self.train_scenes_folder = "/data/qh/DoTA/data/metadata/train_split.txt"
        self.val_scenes_folder = "/data/qh/DoTA/data/metadata/val_split.txt"
        self.plt_tool = plt_tools()
        if is_init_trainer:
            # prepare base config
            parse_cfg = parse_config()
            trainer_path = '/home/qh/TDD/pama/configs/train/multitrainer_basecfg.py'
            SoC = load_config(trainer_path)
            self.basecfg , self.datacfg  = SoC.basecfg , SoC.datacfg 
            self.basecfg.basic.unfreeze()
            self.basecfg.basic.batch_size = 2
            self.basecfg.basic.update(parse_cfg)
            self.basecfg.basic.freeze() 
            setup_seed(self.basecfg.basic.seed)
            self.train_sampler, self.test_sampler, self.traindata_loader, self.testdata_loader = prepare_dataset(self.basecfg.basic, self.datacfg.train_dataset.data, self.datacfg.test_dataset.data)
        
    def test_sam(self,folder):
        from models.sam_model.TAD_SAM import SamTAD
        sam_type = 'vit_b' # 'vit_h' 'vit_l' 'vit_b'
        # checkpoint = "/home/qh/TDD/sam/pretrained/sam_vit_h_4b8939.pth"
        # checkpoint = "/home/qh/TDD/sam/pretrained/sam_vit_l_0b3195.pth"
        checkpoint = "/home/qh/TDD/sam/pretrained/sam_vit_b_01ec64.pth"    
        input_type = 'rgb' # 'rgb' 'emb'
        decoder_type = 'full'

        print('loading sam model...')
        sam_model = SamTAD(input_type,decoder_type,sam_type,checkpoint).cuda()
        for (video_data, data_info, yolo_boxes, frames_boxes, video_name) in tqdm(self.traindata_loader,desc='dataloader:'):
            video_data = video_data.to(sam_model.device, non_blocking=True)
            begin_batch , end_batch = data_info[:,5].type(torch.int).numpy() , data_info[:,6].type(torch.int).numpy()
            for images_data, frame_begin , frame_end , boxes , scene in zip(video_data,begin_batch,end_batch,yolo_boxes,video_name):

                scene_path = os.path.join(folder,scene)
                os.makedirs(scene_path,exist_ok=True)
                yolo_path = os.path.join(self.yolo_folder,scene+'.json')
                with open(yolo_path, 'r') as f:
                    yolo_data = json.load(f)
                yolo_labels = yolo_data['lables']

                # prepare origin video frames
                image_path = os.path.join(self.image_folder,scene,'images')
                frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(frame_begin,frame_end)] 
                frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))

                for i in tqdm(range(0,frame_end-frame_begin),desc='images: '):
                    image_data , frame = images_data[i].unsqueeze(dim=0), frames[i]
                    frame_box = torch.tensor(boxes[i], device=sam_model.device)
                    transformed_boxes = sam_model.transform.apply_boxes_torch(frame_box, sam_model.original_size)
                    masks,_,_ = sam_model.predict_torch( images = image_data, point_coords=None, boxes = transformed_boxes, point_labels=None)                                     
                    fig = self.plt_tool.draw_with_mask_and_box(frame,frame_box,masks)
                    save_name = os.path.join(scene_path,f"{yolo_labels[i+frame_begin]['frame_id']}")
                    fig.savefig(save_name,dpi=300)
                    print()
    
    def read_video_boxes(self,labels,begin,end):
        frames_boxes = []
        for frame_data in labels[begin:end]:
            boxes = [ obj['bbox'] for obj in frame_data['objects'] ]
            frames_boxes.append(np.array(boxes))
        return frames_boxes

    def test_ResizeCoordinates(self,):
        from models.prompt_models.points_transform import ResizeCoordinates
        trans = ResizeCoordinates(target_size=(480, 640))
        scene = 'lmykriTbncM_003990'
        frame_begin , frame_end = 0 , 10
        yolo_path = os.path.join(self.yolo_folder,scene+'.json')
        with open(yolo_path, 'r') as f:
            yolo_data = json.load(f)
        yolo_labels = yolo_data['lables']
        # prepare origin video frames
        image_path = os.path.join(self.image_folder,scene,'images')
        frames_path = [ os.path.join(image_path,yolo_labels[i]['frame_id']) for i in range(frame_begin,frame_end)] 
        frames = np.array(list(map(lambda x:np.asarray(Image.open(x)),frames_path)))
        yolo_boxes = self.read_video_boxes(yolo_labels,frame_begin,frame_end)
        for frame,boxes in zip(frames,yolo_boxes):
            pad_imag = trans.apply_image(frame)
            pad_boxes = trans.apply_boxes(boxes,original_size=(720, 1280))
            image_with_boxes = self.plt_tool.draw_boxes_with_info(np.expand_dims(pad_imag, axis=0) , [pad_boxes])
            print()



if __name__ == "__main__":
    To_test = Component_Tools(is_init_trainer=False)
    To_test.test_ResizeCoordinates()
    '''
    sam_test_folder = ''
    '''
    # sam_test_folder = "/data/qh/DoTA/pama/test/sam_vit_b_rgb_to_mask"
    # To_test.test_sam(sam_test_folder)

    


