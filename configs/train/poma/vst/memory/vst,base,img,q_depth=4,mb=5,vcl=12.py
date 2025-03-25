from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL
import sys
sys.path.insert(0,'/home/qh/TDD/pama')
sys.path.insert(0,'/home/qh/TDD/pama/runner')

basecfg_path = '/home/qh/TDD/pama/configs/base/basecfg.py'
datacfg_path = '/home/qh/TDD/pama/configs/base/dota_movad_rgb_cfg.py'
modelcfg_path = '/home/qh/TDD/pama/configs/model/poma/vst/memory/vst_base_memory.py'


cfg = Config()
cfg.basecfg = load_config(basecfg_path)
cfg.datacfg = load_config(datacfg_path)
cfg.modelcfg = load_config(modelcfg_path)

cfg.unfreeze()

# update VCL
cfg.datacfg.train_dataset.cfg.VCL = 12
# prepare train dataset
cfg.datacfg.train_dataset.data = cfg.datacfg.train_dataset.cfg.type(**cfg.datacfg.train_dataset.cfg)
# prepare test dataset
cfg.datacfg.test_dataset.data = cfg.datacfg.test_dataset.cfg.type(**cfg.datacfg.test_dataset.cfg)

# update some data
cfg.basecfg.basic.model_type = 'poma_base_img_memory'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

# update model-aware cfg
cfg.modelcfg.ano_decoder.apply_vis_mb = True
cfg.modelcfg.ano_decoder.apply_obj_mb = False
cfg.modelcfg.ano_decoder.regressor.Qformer_depth = 4
cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_vis_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_vis_mb)
cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_obj_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_obj_mb)
cfg.modelcfg.ano_decoder.memory_bank_length = 5
cfg.modelcfg.ano_decoder.visual_pe.peroid = IL(lambda c: c.modelcfg.ano_decoder.memory_bank_length)

# update model-aware cfg
# cfg.basecfg.basic.directly_load = "/data/qh/DoTA/poma/base,fpn,rnn,vcl=8,lr=0.002/checkpoints/model-200.pt"
# cfg.modelcfg 