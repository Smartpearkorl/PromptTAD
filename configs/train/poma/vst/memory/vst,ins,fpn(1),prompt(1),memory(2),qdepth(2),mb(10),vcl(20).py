from alchemy_cat.dl_config import Config, load_config, Config ,Param2Tune,IL
import sys
basecfg_path = './configs/base/basecfg.py'
datacfg_path = './configs/base/dota_movad_rgb_cfg.py'
modelcfg_path = './configs/model/poma/vst/memory/vst_ins_fpn(1)_prompt(1)_memory(1).py'

cfg = Config()
cfg.basecfg = load_config(basecfg_path)
cfg.datacfg = load_config(datacfg_path)
cfg.modelcfg = load_config(modelcfg_path)

cfg.unfreeze()

# update VCL
cfg.datacfg.train_dataset.cfg.VCL = 20
# prepare train dataset
cfg.datacfg.train_dataset.data = cfg.datacfg.train_dataset.cfg.type(**cfg.datacfg.train_dataset.cfg)
# prepare test dataset
cfg.datacfg.test_dataset.data = cfg.datacfg.test_dataset.cfg.type(**cfg.datacfg.test_dataset.cfg)

# update some data
cfg.basecfg.basic.model_type = 'poma_prompt_memory'
cfg.basecfg.basic.VCL = cfg.datacfg.train_dataset.cfg.VCL
cfg.basecfg.basic.NF = cfg.modelcfg.NF

cfg.basecfg.basic.proxy_tasks = cfg.modelcfg.proxy_task.task_names
if cfg.modelcfg.proxy_task.use_ins_anomaly:
    cfg.basecfg.basic.apply_ins_loss = True
    cfg.basecfg.basic.ins_loss_weight = 1

# update model-aware cfg
cfg.modelcfg.ins_encoder.pmt_decoder.twoway.depth = 4
cfg.modelcfg.ins_decoder.block_depth = 4
cfg.modelcfg.ano_decoder.regressor.Qformer_depth = 2

cfg.modelcfg.ano_decoder.apply_vis_mb = True
cfg.modelcfg.ano_decoder.apply_obj_mb = True
cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_vis_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_vis_mb)
cfg.modelcfg.ano_decoder.regressor.Qformer_cfg.apply_obj_mb = IL(lambda c: c.modelcfg.ano_decoder.apply_obj_mb)
cfg.modelcfg.ano_decoder.memory_bank_length = 10
cfg.modelcfg.ano_decoder.visual_pe.peroid = IL(lambda c: c.modelcfg.ano_decoder.memory_bank_length)

# cfg.basecfg.basic.directly_load = "/data/qh/DoTA/poma/base,fpn,rnn,vcl=8,lr=0.002/checkpoints/model-200.pt"
# cfg.basecfg.basic.whole_load = False
