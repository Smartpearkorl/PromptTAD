from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL
import sys
sys.path.insert(0,'/home/qh/TDD/pama')
sys.path.insert(0,'/home/qh/TDD/pama/runner')

basecfg_path = '/home/qh/TDD/pama/configs/base/basecfg.py'
datacfg_path = '/home/qh/TDD/pama/configs/base/dada_movad_rgb_cfg.py'
modelcfg_path = '/home/qh/TDD/pama/configs/model/poma/vst/vst_base_rnn.py'

cfg = Config()
cfg.basecfg = load_config(basecfg_path)
cfg.datacfg = load_config(datacfg_path)
cfg.modelcfg = load_config(modelcfg_path)

cfg.unfreeze()
# update VCL
cfg.datacfg.train_dataset.cfg.VCL = 8
# prepare train dataset
cfg.datacfg.train_dataset.data = cfg.datacfg.train_dataset.cfg.type(**cfg.datacfg.train_dataset.cfg)
# prepare test dataset
cfg.datacfg.test_dataset.data = cfg.datacfg.test_dataset.cfg.type(**cfg.datacfg.test_dataset.cfg)

dataset_type = 'dota' if  'DoTA' in cfg.train_dataset.cfg.root_path else 'dada' 
cfg.basecfg.basic.dataset_type = dataset_type

# update some data
cfg.basecfg.basic.model_type = 'poma_base_no_fpn_rnn'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF
