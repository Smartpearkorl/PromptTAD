from alchemy_cat.dl_config import load_config, Config ,Param2Tune,IL
from alchemy_cat.py_tools import Logger,get_local_time_str
import torch
import argparse
import os
import sys

import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler as GradScaler

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
                    default='train',
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
                        # default = "/data/qh/DoTA/pama/test/vscode_debug/",
                        # default = "/data/qh/DoTA/VS/",
                        default = "/data/qh/DoTA/poma_v2/rnn/vst,base,dim=1024,fpn(0),prompt(0),rnn,vcl=8/",
                        help='Directory where save the output.')
    
    args = parser.parse_args()
    cfg = vars(args)

    device = torch.device(f'cuda:{cfg["local_rank"]}') if torch.cuda.is_available() else torch.device('cpu')
    n_nodes = torch.cuda.device_count()
    cfg.update(device=device)
    cfg.update(n_nodes=n_nodes)
    return cfg

if __name__ == "__main__":
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
        cfg_path = './configs/train/poma/vst/ablation/vst,ins,fpn(1),prompt(no_RA),rnn,vcl=8.py'
    else:
        cfg_path = parse_cfg['config']

    SoC = load_config(cfg_path)
    basecfg , datacfg , modelcfg =SoC.basecfg , SoC.datacfg , SoC.modelcfg
    
    basecfg.basic.unfreeze()    
    basecfg.basic.update(parse_cfg)

    # basecfg.basic.directly_load = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt_mean,rnn,depth=4/checkpoints/model-200.pt"
    # basecfg.basic.whole_load = True
  
    init_distributed(basecfg.basic)
    setup_seed(basecfg.basic.seed)
    print(basecfg) 
    
    rank = basecfg.basic.local_rank
    name = f'{basecfg.basic.output}/{get_local_time_str(for_file_name=True)}-{rank=}.log'
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
        
    # freeze vit: misconvergence for sam
    # if datacfg.train_dataset.cfg.pre_process_type == 'rgb':
    #     freeze_vit_backbone(basecfg.basic.model_type,model)

    if basecfg.basic.distributed:
        model.cuda(basecfg.basic.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[basecfg.basic.local_rank], find_unused_parameters=True)
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
    
    # basecfg.basic.fp16 = True
    scaler = GradScaler(enabled=basecfg.basic.fp16)
    print(f'apply FP16 {basecfg.basic.fp16}')
        
    if basecfg.basic.phase=='train':
        # backup config
        backup_file(SoC, modelcfg, model)
        pama_train(basecfg.basic, model, train_sampler, traindata_loader, scaler,
                optimizer, lr_scheduler, test_sampler, testdata_loader, index_video, index_frame )
        
    if basecfg.basic.phase=='test':
        cfg = basecfg.basic
        epoch = cfg.epoch 
        filename = get_result_filename(basecfg.basic, basecfg.basic.epoch)
        if not os.path.exists(filename):
            with torch.no_grad():
                pama_test(basecfg.basic, model, test_sampler, testdata_loader, basecfg.basic.epoch, filename)
                   
        if dist.is_initialized() and basecfg.basic.local_rank != 0:
            dist.destroy_process_group()
        
        # write test results
        if cfg.local_rank == 0: 
            content = load_results(filename)
            # apply per-class
            per_class = datacfg.train_dataset.cfg.name =='dota'
            # frame level eval
            print(f'post_process: True')
            print_results(cfg, *evaluation(**content,post_process=True,per_class = per_class))
            print(f'post_process: False')
            print_results(cfg, *evaluation(**content,post_process=False,per_class = per_class))
            txt_folder = os.path.join(cfg.output, 'evaluation')
            os.makedirs(txt_folder,exist_ok=True)
            txt_path = os.path.join(txt_folder, 'eval.txt')
            write_results(txt_path, epoch , *evaluation( **content,per_class = per_class))
            # instance level eval
            if 'obj_targets' in content and sum([len(x) for x in content['obj_targets']]):
                write_results(txt_path,epoch,*evaluation_on_obj(content['obj_outputs'],content['obj_targets'],content['video_name']),eval_type='instacne')
            # frame level eval
            if 'fra_outputs' in content and content['fra_outputs'][0][0] != -100:
                write_results(txt_path, epoch , *evaluation(outputs = content['fra_outputs'], targets = content['targets']) , eval_type ='prompt frame')
     


