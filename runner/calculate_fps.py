from torch.utils.tensorboard import SummaryWriter
from alchemy_cat.dl_config import load_config, Config ,Param2Tune,IL
from alchemy_cat.py_tools import Logger,get_local_time_str
import torch
import argparse
import os
import sys
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler as GradScaler
import numpy as np
from tqdm import tqdm
from calflops import calculate_flops 
# Custom imports
import sys
from pathlib import Path
FILE = Path(__file__).resolve() # /home/qh/TDD/pama/runner/main.py
sys.path.insert(0, str(FILE.parents[1]))
import os 
os.chdir(FILE.parents[1])

from runner.src.tools import CEloss
from runner.src.dota import gt_cls_target
from models.componets import HungarianMatcher

# Custom imports
import sys
from pathlib import Path
FILE = Path(__file__).resolve() # /home/qh/TDD/pama/runner/main.py
sys.path.insert(0, str(FILE.parents[1]))
import os 
os.chdir(FILE.parents[1])

from runner.train import pama_train 
from runner.test import pama_test 
from runner.src.dota import prepare_dataset
from runner.src.tools import *
from runner.src.utils import resume_from_checkpoint , prepare_optim_sched , get_result_filename , load_results , freeze_vit_backbone
from runner.src.metrics import evaluation, print_results , write_results , evaluation_on_obj

class FrameState():
    def __init__(self) -> None:
        self.t = 0
        self.begin_t = 0
        self.T = 0
        pass

# 查看每部分时间
from time import perf_counter
flag_debug_t = False
debug_t = 0
def updata_debug_t():
        global debug_t
        debug_t = perf_counter()
        
def print_t(process='unknown process'):
    global flag_debug_t
    if flag_debug_t:
        print(f"{process} takes {(perf_counter() - debug_t):.4f}",force=True)
        updata_debug_t()      

def Calculator(STUDY_TYPE, cfg, model, test_sampler, testdata_loader, epoch, filename):
    '''
    when DDP test, every node makes a single .pkl file, then rank_0 node load and combine all .pkl data  
    '''
    # NF for vst
    fb = cfg.get('NF',0)
    # clip model use two image in every iteration 
    clip_model =  'clip' in cfg.get('model_type')
    
    index_frame = 0
    celoss = CEloss(cfg)
    # tqdm
    if cfg.local_rank == 0:
        pbar = tqdm(total=len(testdata_loader),desc='Test Epoch: %d / %d' % (epoch, cfg.total_epoch))

    model.eval()
    for j, (video_data, data_info, yolo_boxes, frames_boxes, video_name) in enumerate(testdata_loader):
        print_t(process=f"video {j}: {video_name[0]}:Enter iterator")
        # prepare data for model and loss func
        video_data = video_data.to(cfg.device, non_blocking=True) # [B,T,C,H,W]  emb is float16->float32
        data_info = data_info.to(cfg.device, non_blocking=True)
        # yolo_boxes : list B x list T x nparray(N_obj, 4)
        yolo_boxes = np.array(yolo_boxes,dtype=object)

        # record whole video data
        B,T = video_data.shape[:2]
        t_shape = (B,T-fb)

        # video data
        idx_batch,toa_batch,tea_batch = data_info[:, 1] , data_info[:, 2] , data_info[:, 3]
        info_batch = data_info[:, 7:11] # accident info: accident_id ego_involve
        # loop in video frames     
        rnn_state , frame_state = None , FrameState()
        print_t(process=f"video {j}: {video_name[0]}: Loop begin....")
        start_time = time.time()  # 开始计时
        for i in range(fb + clip_model , T):
            # preparation
            frame_state.t = i-fb
            frame_state.begin_t = 1 if clip_model else 0
            frame_state.T = T-1-fb
            #clip 
            if clip_model:
                batch_image_data , batch_boxes = video_data[:,i-1:i+1] , yolo_boxes[:,i]
            # poma
            elif fb:         
                batch_image_data , batch_boxes = video_data[:,i-fb:i] , yolo_boxes[:,i-1]
            # pama  
            else:
                batch_image_data , batch_boxes = video_data[:,i] , yolo_boxes[:,i]

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                if clip_model:
                    logits_per_image, logits_per_text = model(batch_image_data, batch_boxes, frame_state)
                    output = logits_per_image
                else:

                    if STUDY_TYPE=='para':
                        flops, macs, params = calculate_flops(model = model, args=[batch_image_data, batch_boxes, rnn_state, frame_state])
                        print(f"\n**************************\n")
                        with open(filename, "a") as f:  # 使用 "a" 模式追加内容  
                            f.write(f"\n\nmodel: {cfg.model_type} flops: {flops}   macs: {macs}   params: {params}")  
                        print(f"\n\nmodel: {cfg.model_type} flops: {flops}   macs: {macs}   params: {params}")
                        print(f"\n**************************\n")
                        return

                    ret = model(batch_image_data, batch_boxes, rnn_state, frame_state)
                    output , rnn_state, outputs_ins_anormal= ret['output'] , ret['rnn_state'] , ret['ins_anomaly']
                    output = output.softmax(dim=-1)

            print_t(process=f"video {j}: {video_name[0]}: Loop {i} end....")
        
        if STUDY_TYPE=='fps':
            end = time.time()  # 开始计时
            print(f"\n**************************\n")
            with open(filename, "a") as f:  # 使用 "a" 模式追加内容  
                f.write(f"\n\nmodel: {cfg.model_type}\nvideo {j}: {video_name[0]}: FPS: {(T-fb)/(end-start_time)}")  
            print(f"model: {cfg.model_type} video {j}: {video_name[0]}: FPS: {(T-fb)/(end-start_time)}")
            print(f"\n**************************\n")
     
# torch.autograd.set_detect_anomaly(True)
def parse_config():
    parser = argparse.ArgumentParser(description='PromptTAD implementation')

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
                    default='test',
                    choices=['test', 'train', 'play'],
                    help='Training or testing or play phase.')
    
    parser.add_argument('--num_workers',
                    type = int,
                    default = 4,
                    metavar='N',)
    
    help_epoch = 'The epoch to restart from (training) or to eval (testing).'
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help=help_epoch)

    parser.add_argument('--config',
                        default='no_config')
                        
    parser.add_argument('--output',
                        # default = "/data/qh/DoTA/poma_v2/Benchmark/FPS/debug",
                        default = "/data/qh/DoTA/poma_v2/Benchmark/debug",
                        # default = "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8/",
                        help='Directory where save the output.')
    
    args = parser.parse_args()
    cfg = vars(args)

    device = torch.device(f'cuda:{cfg["local_rank"]}') if torch.cuda.is_available() else torch.device('cpu')
    n_nodes = torch.cuda.device_count()
    cfg.update(device=device)
    cfg.update(n_nodes=n_nodes)
    return cfg

def Single_calculate(cfg_path, weight_path):
    parse_cfg = parse_config()
    if parse_cfg['config'] == 'no_config':
        '''
                                           base_detection
        cfg_path = './configs/train/poma/vst/base_detection/vst,base,dim=256,fpn(0),prompt(0),rnn,vcl=8.py'
        cfg_path = './configs/train/poma/vst/base_detection/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8.py'
        cfg_path = './configs/train/poma/vst/base_detection/vst,base,fpn(0),prompt(1),rnn,vcl=8.py'
        cfg_path = './configs/train/poma/vst/base_detection/vst,base,fpn(1),prompt(0),rnn,vcl=8.py'
        cfg_path = './configs/train/poma/vst/base_detection/vst,base,fpn(1),prompt(1),rnn,vcl=8.py'
                                           instance_detection
        cfg_path =  './configs/train/poma/vst/instance_detection/vst,ins,fpn(0),prompt(1),rnn,vcl=8.py'
        cfg_path =  './configs/train/poma/vst/instance_detection/vst,ins,fpn(1),prompt(1),rnn,vcl=8.py'
                                           ablation_study
        cfg_path = './configs/train/poma/vst/ablation/vst,base,fpn(1),prompt(no_RA),rnn,vcl=8.py'
        cfg_path = './configs/train/poma/vst/ablation/vst,ins,fpn(1),prompt(no_RA),rnn,vcl=8.py'

        '''        
        cfg_path = cfg_path
    else:
        cfg_path = parse_cfg['config']

    SoC = load_config(cfg_path)
    basecfg , datacfg , modelcfg =SoC.basecfg , SoC.datacfg , SoC.modelcfg
    
    basecfg.basic.unfreeze()    
    basecfg.basic.update(parse_cfg)

    basecfg.basic.directly_load = weight_path
    basecfg.basic.whole_load = True
    
    basecfg.basic.test_batch_size = 1
    # init_distributed(basecfg.basic) no ddp and reload print
    setup_seed(basecfg.basic.seed)
    print(basecfg) 
    
    # study type : fps / para
    STUDY_TYPE = 'para' # fps / para
    rank = basecfg.basic.local_rank
    name = f'{basecfg.basic.output}/{STUDY_TYPE}_{basecfg.basic.model_type}.log'
    Logger(out_file = name, real_time = True)

    print('prepare dataset...')
    train_sampler, test_sampler, traindata_loader, testdata_loader = prepare_dataset(basecfg.basic, datacfg.train_dataset.data,datacfg.test_dataset.data)
    print('loading model...')
    if modelcfg.model_type == 'pama':
        model =  modelcfg.model( sam_cfg = modelcfg.sam, 
                                 bottle_aug_cfg = modelcfg.bottle_aug, 
                                 ins_decoder_cfg = modelcfg.ins_decoder, 
                                 ano_decoder_cfg = modelcfg.ano_decoder,
                                 proxy_task_cfg = modelcfg.proxy_task )
    
    elif modelcfg.model_type == 'poma':
        model =  modelcfg.model( dinov2_cfg = modelcfg.dinov2 ,
                                 clip_cfg = modelcfg.clip,
                                 vst_cfg = modelcfg.vst,
                                 fpn_cfg = modelcfg.fpn,
                                 ins_encoder_cfg = modelcfg.ins_encoder , 
                                 bottle_aug_cfg = modelcfg.bottle_aug, 
                                 ins_decoder_cfg = modelcfg.ins_decoder, 
                                 ano_decoder_cfg = modelcfg.ano_decoder,
                                 proxy_task_cfg = modelcfg.proxy_task)
    
    elif  modelcfg.model_type == 'clip_poma':
        model =  modelcfg.model( clip_cfg = modelcfg.clip ,
                                 ins_encoder_cfg = modelcfg.ins_encoder ,
                                 bottle_aug_cfg = modelcfg.bottle_aug , 
                                 ins_decoder_cfg = modelcfg.ins_decoder, 
                                 memorybank_cfg = modelcfg.memorybank )
        
    if basecfg.basic.distributed:
        model.cuda(basecfg.basic.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[basecfg.basic.local_rank], find_unused_parameters=True)
    else:
        model.to(basecfg.basic.device)
        model = nn.DataParallel(model) # 需要指定CUDA_VISIBLE_DEVICES 否则rnn会错误
    
    if basecfg.basic.phase == 'train':
        print('prepare optimizer...')
        optimizer, lr_scheduler = prepare_optim_sched(model, basecfg.optimizer, basecfg.sched)
        ckp = resume_from_checkpoint(basecfg.basic,  model.module , optimizer , lr_scheduler)
        # resume summarywriter index for tensorboard
        index_video = ckp.get('index_video', 0)
        index_frame = ckp.get('index_frame', 0)
    else:
        resume_from_checkpoint(basecfg.basic,  model.module , None , None)
    
    # basecfg.basic.fp16 = True
    scaler = GradScaler(enabled=basecfg.basic.fp16)
    print(f'apply FP16 {basecfg.basic.fp16}')
        
    if basecfg.basic.phase=='test':
        cfg = basecfg.basic
        epoch = cfg.epoch 
        filename = os.path.join(basecfg.basic.output,f'{STUDY_TYPE}.txt')
        with torch.no_grad():
            Calculator(STUDY_TYPE,basecfg.basic, model, test_sampler, testdata_loader, basecfg.basic.epoch, filename)

if __name__ == "__main__":
    cfg_paths = [  
        # './configs/train/poma/vst/base_detection/vst,base,dim=256,fpn(0),prompt(0),rnn,vcl=8.py',  
        # './configs/train/poma/vst/base_detection/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8.py',  
        # './configs/train/poma/vst/base_detection/vst,base,fpn(0),prompt(1),rnn,vcl=8.py',  
        # './configs/train/poma/vst/base_detection/vst,base,fpn(1),prompt(0),rnn,vcl=8.py',  
        # './configs/train/poma/vst/base_detection/vst,base,fpn(1),prompt(1),rnn,vcl=8.py',  
        # './configs/train/poma/vst/instance_detection/vst,ins,fpn(0),prompt(1),rnn,vcl=8.py',  
        './configs/train/poma/vst/instance_detection/vst,ins,fpn(1),prompt(1),rnn,vcl=8.py',  
        # './configs/train/poma/vst/ablation/vst,base,fpn(1),prompt(no_RA),rnn,vcl=8.py',
        # './configs/train/poma/vst/ablation/vst,ins,fpn(1),prompt(no_RA),rnn,vcl=8.py',
    ]  

    weight_path = [
        # "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=256,fpn(0),prompt(0),rnn,vcl=8/checkpoints/model-200.pt",
        # "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8/checkpoints/model-200.pt",
        # "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(0),prompt(1),rnn,vcl=8/checkpoints/model-200.pt",
        # "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(1),prompt(0),rnn,vcl=8/checkpoints/model-200.pt",
        # "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(1),prompt(1),rnn,vcl=8/checkpoints/model-200.pt",
        # "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(0),prompt(1),rnn,vcl=8/checkpoints/model-200.pt",
        "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(1),rnn,vcl=8/checkpoints/model-200.pt",
        # "/data/qh/DoTA/poma_v2/rnn/vst,base,fpn(1),prompt(no_RA),rnn,vcl=8/checkpoints/model-200.pt",
        # "/data/qh/DoTA/poma_v2/instance/vst,ins,fpn(1),prompt(no_RA),rnn,vcl=8/checkpoints/model-200.pt",
    ]
    for p,w in zip(cfg_paths,weight_path):
        Single_calculate(p,w)


            


        

    


    
