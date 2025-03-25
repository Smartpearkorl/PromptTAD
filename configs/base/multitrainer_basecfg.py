from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL
import sys
sys.path.insert(0,'/home/qh/TDD/pama')
sys.path.insert(0,'/home/qh/TDD/pama/runner')

basecfg_path = '/home/qh/TDD/pama/configs/base/basecfg.py'
# datacfg_path = '/home/qh/TDD/pama/configs/base/dota_rgb_cfg.py'
datacfg_path = '/home/qh/TDD/pama/configs/base/dota_dinov2_emb.py'

cfg = Config()
cfg.basecfg = load_config(basecfg_path)
cfg.datacfg = load_config(datacfg_path)

cfg.unfreeze()
# update VCL
cfg.datacfg.train_dataset.cfg.VCL = 8
# prepare train dataset
cfg.datacfg.train_dataset.data = cfg.datacfg.train_dataset.cfg.type(**cfg.datacfg.train_dataset.cfg)
# prepare test dataset
cfg.datacfg.test_dataset.data = cfg.datacfg.test_dataset.cfg.type(**cfg.datacfg.test_dataset.cfg)
