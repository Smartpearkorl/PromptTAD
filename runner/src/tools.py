import torch
import numpy as np
import random
import logging
import time
import os
import torch.distributed as dist

'''
prepare for training 
'''
def CEloss(cfg):
    return torch.nn.CrossEntropyLoss( weight=torch.tensor(cfg.class_weights).to(cfg.device),)
       

def NLLloss(cfg):
    return torch.nn.NLLLoss( weight=torch.tensor(cfg.class_weights).to(cfg.device))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

'''
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
'''
def Prepare_logger(logpath):
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)
    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logfile = os.path.join(logpath,f'{date}.log')
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

'''
custom_tools.py applies 
'''
def custom_print(*args, level='INFO', local_rank=0, only_rank0=True, **kwargs):
    
    if only_rank0 and local_rank != 0:
        return
    
    levels = {
        'DEBUG': '\033[94m[DEBUG]\033[0m',
        'INFO': '\033[92m[INFO]\033[0m',
        'WARNING': '\033[93m[WARNING]\033[0m',
        'ERROR': '\033[91m[ERROR]\033[0m'
    }
    level_str = levels.get(level, '[INFO]')

    rank_colors = ['\033[96m', '\033[95m', '\033[94m', '\033[93m', '\033[92m']
    rank_color = rank_colors[local_rank % len(rank_colors)]

    rank_prefix = f'{rank_color}[RANK {local_rank}]\033[0m'
    print(f'{rank_prefix} {level_str}', *args, **kwargs)

def reload_print(local_rank):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print
    is_master = local_rank== 0
    def print( *args, level='INFO', **kwargs):
        levels = {
            'DEBUG': '\033[94m[DEBUG]\033[0m',
            'INFO': '\033[92m[INFO]\033[0m',
            'WARNING': '\033[93m[WARNING]\033[0m',
            'ERROR': '\033[91m[ERROR]\033[0m'
        }
        level_str = levels.get(level, '[INFO]')
        rank_colors = ['\033[96m', '\033[95m', '\033[94m', '\033[93m', '\033[92m']
        rank_color = rank_colors[local_rank % len(rank_colors)]
        rank_prefix = f'{rank_color}[RANK {local_rank}]\033[0m'
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(f'{rank_prefix} {level_str}', *args, **kwargs)
    __builtin__.print = print

def init_distributed(cfg):
    if cfg.distributed:
        dist.init_process_group(backend="nccl")
        # dist.barrier()
    reload_print(cfg.local_rank)   
    print(f'distributed: {cfg.distributed}  n_nodes={cfg.n_nodes}',force=True)
       

def backup_file(SoC, modelcfg, model):
    if SoC.basecfg.basic.local_rank ==0:
        SoC.save_py(os.path.join(SoC.basecfg.basic.output,'basecfg.py'))
        modelcfg.save_py(os.path.join(SoC.basecfg.basic.output,'modelcfg.py'))
        modelcfg.save_pkl(os.path.join(SoC.basecfg.basic.output,'modelcfg.pkl'))
        with open(os.path.join(SoC.basecfg.basic.output,'model.log'),'w') as f:
            f.write(str(model))

