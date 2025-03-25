from alchemy_cat.dl_config import Config,IL
import sys
from runner import vst_pretrained_weight_path
from models.prompt_models.poma import poma
cfg = Config()

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
# cfg.fpn.dimen_reduce_type = 'mlp'
cfg.fpn.apply_linar = True
cfg.fpn.linear.vst_dim = 1024
cfg.fpn.linear.target_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)

cfg.basic.use_ins = True
cfg.basic.selected_ano_decoder = 'rnn' # 'rnn' or 'memory' or 'plain'

cfg.proxy_task.task_names = []
cfg.proxy_task.use_ins_anomaly = True
if cfg.proxy_task.use_ins_anomaly:
    cfg.proxy_task.task_names.append('instance')
    cfg.proxy_task.ins_anomaly_type = 'ffn' # 'ffn' or 'memoery'
    if cfg.proxy_task.ins_anomaly_type == 'ffn':
        cfg.proxy_task.ins_anomaly_ffn.input_dim = IL(lambda c: c.basic.hid_dim)
        cfg.proxy_task.ins_anomaly_ffn.hidden_dim = IL(lambda c: c.basic.hid_dim)
        cfg.proxy_task.ins_anomaly_ffn.output_dim = 2
        cfg.proxy_task.ins_anomaly_ffn.num_layers = 2

    elif cfg.proxy_task.ins_anomaly_type == 'memory': 
        cfg.proxy_task.regressor.N_for_cls = True  # instance anomaly detection : N dimension for classification
        cfg.proxy_task.regressor.Qformer_depth = 2  
        cfg.proxy_task.regressor.Qformer_cfg = cfg.basic.trans
        cfg.proxy_task.regressor.Qformer_cfg.Skip_selfattn = IL(lambda c: c.proxy_task.regressor.N_for_cls) 
        cfg.proxy_task.regressor.Qformer_cfg.apply_vis_mb = True
        cfg.proxy_task.regressor.Qformer_cfg.apply_obj_mb = True
        cfg.proxy_task.regressor.Qformer_cfg.memory_bank_length = IL(lambda c: c.ano_decoder.memory_bank_length)
        cfg.proxy_task.regressor.hidden_size = IL(lambda c: c.basic.hid_dim )                                                      
        cfg.proxy_task.regressor.dim_latent = IL(lambda c: c.basic.hid_dim)
        cfg.proxy_task.regressor.dropout = IL(lambda c: c.basic.dropout)
     
if cfg.basic.use_ins:
    # instace encoder : prompt encoder
    cfg.ins_encoder.pmt_encoder.prompt_type = 'boxes'
    cfg.ins_encoder.pmt_encoder.embed_dim = IL(lambda c: c.basic.hid_dim, priority=0, rel=False)
    # update from Dota
    cfg.ins_encoder.pmt_encoder.image_embedding_size = (15 , 20)
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
                                                c.ano_decoder.regressor.pool_shape[0]*
                                                c.ano_decoder.regressor.pool_shape[1])
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
                                                c.ano_decoder.reducer.pool_shape[-2]*
                                                c.ano_decoder.reducer.pool_shape[-1])
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


