from alchemy_cat.dl_config import Config ,IL
from torch.optim.lr_scheduler  import CosineAnnealingLR, CosineAnnealingWarmRestarts,StepLR, OneCycleLR,SequentialLR,LinearLR
import torch
cfg = Config()

# basic
cfg.basic.seed = 123
cfg.basic.total_epoch = 100
cfg.basic.snapshot_interval = 10

cfg.basic.test_inteval = 10
cfg.basic.batch_size = 128
cfg.basic.test_batch_size = 8
cfg.basic.num_workers = 0

cfg.basic.n_nodes = 1
cfg.basic.lr = IL(lambda c: 5e-6 * c.basic.batch_size / 128 , priority=0)
cfg.basic.max_iter = IL(lambda c:  int(c.basic.total_epoch * 3196 // (c.basic.n_nodes*c.basic.batch_size)) , priority=0) 
cfg.basic.class_weights = (1.0 , 1.0)

cfg.basic.train_debug.debug_train_weight = False # False
cfg.basic.train_debug.debug_train_weight_level = 2
cfg.basic.train_debug.debug_train_grad = False
cfg.basic.train_debug.debug_train_grad_level = 2
cfg.basic.train_debug.debug_loss = True

# optimizer and lr_scheduler strategy
cfg.optimizer.type = 'adam'
cfg.optimizer.cls = torch.optim.Adam
cfg.optimizer.lr = IL(lambda c: c.basic.lr , priority=1)
cfg.optimizer.weight_decay = 1e-4

cfg.sched.cls = SequentialLR
cfg.sched.warm.warm_iters = IL(lambda c: int(3196 // (c.basic.n_nodes * c.basic.batch_size )), priority=1)  # ~ 1 epochs
cfg.sched.warm.ini.start_factor = 0.25
cfg.sched.warm.ini.end_factor = 1.0
cfg.sched.warm.ini.total_iters = IL(lambda c: c.sched.warm.warm_iters)
cfg.sched.warm.cls = LinearLR

cfg.sched.main.ini.T_max = IL(lambda c: c.basic.max_iter - c.sched.warm.warm_iters)
cfg.sched.main.ini.eta_min = IL(lambda c: 0.25*c.optimizer.lr )
cfg.sched.main.cls = CosineAnnealingLR






