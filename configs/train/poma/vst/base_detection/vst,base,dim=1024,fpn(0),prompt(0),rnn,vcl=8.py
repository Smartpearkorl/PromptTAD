from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL

basecfg_path = '/configs/base/basecfg.py'
datacfg_path = '/configs/base/dota_movad_rgb_cfg.py'
modelcfg_path = './configs/model/poma/vst/base_detection/vst(dim=1024)_base_fpn(0)_prompt(0)_rnn.py'

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

cfg.unfreeze()
# update some data
# cfg.basecfg.basic.model_type = 'poma_base_no_fpn_rnn_1024'
cfg.basecfg.basic.model_type = 'poma_base,dim=1024,fpn(0),prompt(0)'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF
