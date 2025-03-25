from alchemy_cat.dl_config import Config,IL
import sys

from models.prompt_models.poma import poma
cfg = Config()
from runner import vst_pretrained_weight_path
cfg.model = poma
cfg.model_type = 'poma'
cfg.NF = 4
cfg.basic.hid_dim = 256
cfg.basic.dropout = 0.3

# transformer block
cfg.basic.trans.hid_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)
cfg.basic.trans.nhead = 8
cfg.basic.trans.ffn_dim = 256
cfg.basic.trans.dropout = 0.1
cfg.basic.selected_ano_decoder = 'rnn' # 'rnn' or 'memory' or 'plain'

# video swin transformer
cfg.vst.type = 'swin_base_patch244_window1677_sthv2'
cfg.vst.checkpoint = str(vst_pretrained_weight_path)

# fpn 
cfg.fpn.dimen_reduce_type = 'mlp'

cfg.basic.use_ins = False
cfg.basic.selected_ano_decoder = 'rnn' # 'rnn' or 'memory' or 'plain'

if cfg.basic.use_ins:
    # instace encoder : prompt encoder
    cfg.ins_encoder.pmt_encoder.prompt_type = 'boxes'
    cfg.ins_encoder.pmt_encoder.embed_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)
    # update from Dota
    cfg.ins_encoder.pmt_encoder.image_embedding_size = (60 , 80)
    cfg.ins_encoder.pmt_encoder.input_image_size = (480 , 640)
    cfg.ins_encoder.resize.target_size = (480 , 640)
    cfg.ins_encoder.resize.original_size = (720, 1280)
    # instace encoder : prompt decoder
    cfg.ins_encoder.pmt_decoder.transformer_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)
    cfg.ins_encoder.pmt_decoder.twoway.depth = 2
    cfg.ins_encoder.pmt_decoder.twoway.embedding_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)
    cfg.ins_encoder.pmt_decoder.twoway.num_heads = 8
    cfg.ins_encoder.pmt_decoder.twoway.mlp_dim = 2048
    # instance decoder
    cfg.ins_decoder.num_query_token = 12
    cfg.ins_decoder.hid_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ins_decoder.initializer_range = 0.02
    cfg.ins_decoder.block_depth = 2
    cfg.ins_decoder.block = cfg.basic.trans

if cfg.basic.selected_ano_decoder == 'plain':
    # Anomaly Decoder : plain regressor
    cfg.ano_decoder.type = 'plain'
    cfg.ano_decoder.regressor.pool_shape = (6,6) 
    cfg.ano_decoder.regressor.pool_dim = IL(lambda c: c.basic.hid_dim*
                                                c.ano_decoder.regressor.pool_shape[-2]*
                                                c.ano_decoder.regressor.pool_shape[-1])
    cfg.ano_decoder.regressor.dim_latent = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)
    cfg.ano_decoder.regressor.dropout = IL(lambda c: c.basic.dropout)


elif cfg.basic.selected_ano_decoder == 'rnn':
    # Anomaly Decoder : rnn regressor
    cfg.ano_decoder.type = 'rnn'
    cfg.ano_decoder.reducer.has_obj_embs = IL(lambda c: c.basic.use_ins)
    # MLP for object embedding to latent
    if cfg.basic.use_ins:
        cfg.ano_decoder.reducer.mlp_dim = IL(lambda c: c.ins_decoder.num_query_token*c.basic.hid_dim)
        cfg.ano_decoder.reducer.mlp_depth = 2

    cfg.ano_decoder.reducer.pool_shape = (6,6) 
    cfg.ano_decoder.reducer.pool_dim = IL(lambda c: c.basic.hid_dim*
                                                c.ano_decoder.reducer.pool_shape[0]*
                                                c.ano_decoder.reducer.pool_shape[1])
    cfg.ano_decoder.reducer.dim_latent = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)
    cfg.ano_decoder.reducer.dropout = IL(lambda c: c.basic.dropout)
    
    cfg.ano_decoder.regressor.dim_latent = IL(lambda c: c.ano_decoder.reducer.dim_latent)
    cfg.ano_decoder.regressor.rnn_state_size = 256
    cfg.ano_decoder.regressor.rnn_cell_num = 3
    cfg.ano_decoder.regressor.dropout = IL(lambda c: c.basic.dropout)

elif cfg.basic.selected_ano_decoder == 'memory':
    # Anomaly Decoder : Memorybank regressor
    cfg.ano_decoder.type = 'memory'
    cfg.ano_decoder.apply_vis_mb = True
    cfg.ano_decoder.apply_obj_mb = True
    cfg.ano_decoder.memory_bank_length = 5 
    cfg.ano_decoder.num_query_token = 1
    cfg.ano_decoder.initializer_range =0.02
    cfg.ano_decoder.hid_dim = IL(lambda c: c.basic.hid_dim, priority=0)
    cfg.ano_decoder.visual_pe.peroid = IL(lambda c: c.ano_decoder.memory_bank_length)
    cfg.ano_decoder.visual_pe.emb_dim = IL(lambda c: c.basic.hid_dim)
    cfg.ano_decoder.regressor.Qformer_depth = 2
    cfg.ano_decoder.regressor.Qformer_cfg = cfg.basic.trans
    cfg.ano_decoder.regressor.Qformer_cfg.apply_vis_mb = IL(lambda c: c.ano_decoder.apply_vis_mb)
    cfg.ano_decoder.regressor.Qformer_cfg.apply_obj_mb = IL(lambda c: c.ano_decoder.apply_obj_mb)
    cfg.ano_decoder.regressor.Qformer_cfg.memory_bank_length = IL(lambda c: c.ano_decoder.memory_bank_length)
    cfg.ano_decoder.regressor.hidden_size = IL(lambda c: c.ano_decoder.num_query_token*c.ano_decoder.hid_dim )                                                      
    cfg.ano_decoder.regressor.dim_latent = IL(lambda c: c.basic.hid_dim)
    cfg.ano_decoder.regressor.dropout = IL(lambda c: c.basic.dropout)



