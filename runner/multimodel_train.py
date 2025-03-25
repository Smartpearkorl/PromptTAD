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
from torch.cuda.amp import GradScaler as GradScaler

import sys
from pathlib import Path
FILE = Path(__file__).resolve() # /home/qh/TDD/pama/runner/main.py
sys.path.insert(0, str(FILE.parents[2]))
import os 
os.chdir(FILE.parents[2])

from runner.train_plus import pama_train_plus 
from runner.src.dota import prepare_dataset
from runner.src.tools import *
from runner.src.utils import resume_from_checkpoint , prepare_optim_sched , get_result_filename , load_results ,freeze_vit_backbone
from runner.src.metrics import evaluation, print_results , write_results , evaluation_on_obj

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
                    type = int,
                    default = 8,
                    metavar='N',)
    
    help_epoch = 'The epoch to restart from (training) or to eval (testing).'
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help=help_epoch)

    parser.add_argument('--output',
                    default="/data/qh/DoTA/dinov2/vscode_debug/",
                    # default='/data/qh/DoTA/dinov2/trainer/',
                    help='Directory where save the output.')
    
    args = parser.parse_args()
    cfg = vars(args)

    device = torch.device(f'cuda:{cfg["local_rank"]}') if torch.cuda.is_available() else torch.device('cpu')
    n_nodes = torch.cuda.device_count()
    cfg.update(device=device)
    cfg.update(n_nodes=n_nodes)
    return cfg

'''
load single trainer
'''
def load_single_trainer(cfg_path , output_path , parse_cfg):
    SoC = load_config(cfg_path)
    basecfg , datacfg , modelcfg =SoC.basecfg , SoC.datacfg , SoC.modelcfg

    basecfg.basic.unfreeze()    
    basecfg.basic.update(parse_cfg)
    basecfg.basic.output = output_path

    name = basecfg.basic.model_type
    print(f'trainer {name=} is running')
    print(basecfg) 

    print('loading model...')
    if modelcfg.model_type == 'pama':
        model =  modelcfg.model( sam_cfg = modelcfg.sam, 
                                 bottle_aug_cfg = modelcfg.bottle_aug, 
                                 ins_decoder_cfg = modelcfg.ins_decoder, 
                                 ano_decoder_cfg = modelcfg.ano_decoder )
    
    elif modelcfg.model_type == 'poma':
        model =  modelcfg.model( dinov2_cfg = modelcfg.dinov2 ,
                                 clip_cfg = modelcfg.clip,
                                 vst_cfg = modelcfg.vst,
                                 fpn_cfg = modelcfg.fpn,
                                 ins_encoder_cfg = modelcfg.ins_encoder , 
                                 bottle_aug_cfg = modelcfg.bottle_aug, 
                                 ins_decoder_cfg = modelcfg.ins_decoder, 
                                 ano_decoder_cfg = modelcfg.ano_decoder )
    
    elif  modelcfg.model_type == 'clip_poma':
        model =  modelcfg.model( clip_cfg = modelcfg.clip ,
                                 ins_encoder_cfg = modelcfg.ins_encoder ,
                                 bottle_aug_cfg = modelcfg.bottle_aug , 
                                 ins_decoder_cfg = modelcfg.ins_decoder, 
                                 memorybank_cfg = modelcfg.memorybank )
  
    # freeze vit: misconvergence for sam
    if datacfg.train_dataset.cfg.pre_process_type == 'rgb':
        freeze_vit_backbone(basecfg.basic.model_type,model)

    if basecfg.basic.distributed:
        model.cuda(basecfg.basic.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[basecfg.basic.local_rank], output_device=basecfg.basic.local_rank,find_unused_parameters=True)
    else:
        model.to(basecfg.basic.device)
        model = nn.DataParallel(model)
    
    if basecfg.basic.phase == 'train':
        print('prepare optimizer...')
        optimizer, lr_scheduler = prepare_optim_sched(model, basecfg.optimizer, basecfg.sched)
        ckp = resume_from_checkpoint(basecfg.basic,  model.module , optimizer , lr_scheduler)
        # resume summarywriter index for tensorboard
        index_video = ckp.get('index_video', 0)
        index_frame = ckp.get('index_frame', 0)
    else:
        resume_from_checkpoint(basecfg.basic,  model.module , None , None)
            
    scaler = GradScaler(enabled=basecfg.basic.fp16)
    print(f'apply FP16 {basecfg.basic.fp16}')

    backup_file(SoC, modelcfg, model)
   
    data = {'cfg':basecfg.basic , 'model':model , 'scaler':scaler, 'optimizer':optimizer , 'lr_scheduler':lr_scheduler ,
            'index_video':index_video , 'index_frame':index_frame}
    
    return name, data

    

if __name__ == "__main__":
    parse_cfg = parse_config()

    trainer_path = '/home/qh/TDD/pama/configs/base/multitrainer_basecfg.py'
    SoC = load_config(trainer_path)
    basecfg , datacfg  = SoC.basecfg , SoC.datacfg 
    basecfg.basic.unfreeze()    
    basecfg.basic.update(parse_cfg)
    basecfg.basic.output = parse_cfg['output']
    init_distributed(basecfg.basic)
    setup_seed(basecfg.basic.seed)
    
    rank = basecfg.basic.local_rank
    name = f'{basecfg.basic.output}/{get_local_time_str(for_file_name=True)}-{rank=}.log'
    Logger(out_file = name, real_time = True)

    print('prepare dataset...')
    train_sampler, test_sampler, traindata_loader, testdata_loader = prepare_dataset(basecfg.basic, datacfg.train_dataset.data,datacfg.test_dataset.data)

    print('prepare trainer...')
    cfg_path = [
                '/home/qh/TDD/pama/configs/train/poma/dinov2/vit_l,base,plain,vcl=8.py',
                '/home/qh/TDD/pama/configs/train/poma/dinov2/vit_l,base,rnn,vcl=8.py',        
                '/home/qh/TDD/pama/configs/train/poma/dinov2/vit_l,prompt,rnn,vcl=8.py',
                # '/home/qh/TDD/pama/configs/train/poma/dinov2/vit_l,base,mb=5,vcl=8.py',
                ]
    
    output_path = [
                   f'{basecfg.basic.output}/vit_l,base,plain,vcl=8',
                   f'{basecfg.basic.output}/vit_l,base,rnn,vcl=8',
                   f'{basecfg.basic.output}/vit_l,prompt,rnn,vcl=8',
                #    f'{basecfg.basic.output}/vit_l,base,mb=5,vcl=8',
                  ]
    trainers = {}
    for C,P in zip(cfg_path,output_path):
        name, data = load_single_trainer(C,P,parse_cfg)
        trainers[name] = data
        # update epoch
        if data['cfg'].epoch != basecfg.basic.epoch:
            basecfg.basic.epoch = data['cfg'].epoch
            print(f'Set epoch={data["cfg"].epoch} from trainer :{name}')

    if basecfg.basic.phase=='train':   
        pama_train_plus(basecfg.basic, trainers, train_sampler, traindata_loader, test_sampler ,testdata_loader )
        
        


