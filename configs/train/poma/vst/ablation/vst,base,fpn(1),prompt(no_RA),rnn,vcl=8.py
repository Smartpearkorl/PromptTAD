from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL

basecfg_path = './configs/base/basecfg.py'
datacfg_path = './configs/base/dota_movad_rgb_cfg.py'
modelcfg_path = './configs/model/poma/vst/ablation/vst_base_fpn(1)_prompt(no_RA)_rnn.py'


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

# update some data
cfg.basecfg.basic.model_type = 'poma_base,fpn(1),prompt(no_RA)'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

# update model-aware cfg
cfg.modelcfg.ins_encoder.pmt_decoder.twoway.depth = 4
# cfg.basecfg.basic.directly_load = "/data/qh/DoTA/poma/base,fpn,rnn,vcl=8,lr=0.002/checkpoints/model-200.pt"
