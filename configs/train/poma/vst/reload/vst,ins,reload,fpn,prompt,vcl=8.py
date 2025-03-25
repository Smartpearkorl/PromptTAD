from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL
import sys
sys.path.insert(0,'/home/qh/TDD/pama')
sys.path.insert(0,'/home/qh/TDD/pama/runner')

basecfg_path = '/home/qh/TDD/pama/configs/base/basecfg.py'
datacfg_path = '/home/qh/TDD/pama/configs/base/dota_movad_rgb_cfg.py'
modelcfg_path = '/home/qh/TDD/pama/configs/model/poma/vst/instance_detection/vst_ins_prompt_rnn.py'


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
cfg.basecfg.basic.model_type = 'poma_prompt_fpn_rnn'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

cfg.basecfg.basic.proxy_tasks = cfg.modelcfg.proxy_task.task_names
if cfg.modelcfg.proxy_task.use_ins_anomaly:
    cfg.basecfg.basic.apply_ins_loss = True
    cfg.basecfg.basic.ins_loss_weight = 1

# update model-aware cfg
cfg.modelcfg.ins_encoder.pmt_decoder.twoway.depth = 4
cfg.modelcfg.ins_decoder.block_depth = 4

cfg.basecfg.basic.directly_load = "/data/qh/DoTA/poma_v2/instance/vst,ins,prompt,rnn,depth=4/checkpoints/model-160.pt"
cfg.basecfg.basic.whole_load = True