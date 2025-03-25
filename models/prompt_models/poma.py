import torch
import math

from torch import nn
from torch.nn import functional as F
import functools
import operator
from models.TTHF import open_clip_local as clip
from models.prompt_models.multiscale_vst import register_vst_model
from models.prompt_models.vst_fpn import Standard_FPN
from models.prompt_models.aggregation import Instance_Encoder, Instance_Decoder , Instance_Decoder_V2
from models.Transformer import *
from models.componets import *
from models.dinov2.dinov2.models import register_vit_model


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class poma(nn.Module):
    def __init__(
        self,
        # dinov2
        dinov2_cfg:dict = {},
        # clip
        clip_cfg: dict = {},
        # vst
        vst_cfg: dict = {},
        fpn_cfg: dict = {},

        ins_encoder_cfg: dict ={},

        bottle_aug_cfg: dict = {},    
        ins_decoder_cfg = {}, 
        ano_decoder_cfg: dict = {},

        reduce_cfg: dict = {},
        return_cfg: dict = {},
        proxy_task_cfg: dict = {},
        ):
        super().__init__()
        empy_dict = {}
        # backbone type
        configs = {'clip': clip_cfg, 'vst': vst_cfg, 'dinov2': dinov2_cfg}
        self.vit_type = next((key for key, cfg in configs.items() if cfg != empy_dict), None)

        if self.vit_type == 'vst':
            self.vst_model = register_vst_model(**vst_cfg)
            self.use_fpn = fpn_cfg.dimen_reduce_type != empy_dict
            if self.use_fpn:
                self.fpn = Standard_FPN(**fpn_cfg)     
            else:
                self.use_linear = fpn_cfg.get('apply_linar',False) # 兼容之前的cfg
                if self.use_linear:
                    self.linear = nn.Sequential(nn.Linear(fpn_cfg.linear.vst_dim , fpn_cfg.linear.target_dim),nn.ReLU())

        if self.vit_type == 'clip':
            self.clip_vit = clip.create_model(**clip_cfg.model)
            self.clip_vit = self.clip_vit.visual
            self.clip_patch_head = nn.Sequential( nn.Linear(clip_cfg.patch_head.hid_dim * 2, clip_cfg.patch_head.hid_dim),nn.Dropout(0.2))
        
        if self.vit_type == 'dinov2':
            self.vit_model = register_vit_model(**dinov2_cfg)

        # build Instance Encoder-Decoder
        self.use_ins_decoder = ins_decoder_cfg != empy_dict
        if self.use_ins_decoder :
            self.instance_encoder = Instance_Encoder(ins_encoder_cfg.pmt_encoder, ins_encoder_cfg.resize,ins_encoder_cfg.pmt_decoder )
            # instance_decoder ablation experiment 
            if ins_decoder_cfg == 'mean':
                self.instance_decoder = 'mean' 
                self.ins_reducer = Reducer(**ano_decoder_cfg.reducer) # for object-aug image embs              
            else:
                self.instance_query = nn.Parameter(torch.zeros(1, ins_decoder_cfg.num_query_token, ins_decoder_cfg.hid_dim))
                self.instance_query.data.normal_(mean=0.0, std=ins_decoder_cfg.initializer_range)
                self.instance_decoder = Instance_Decoder_V2(ins_decoder_cfg.block_depth, ins_decoder_cfg.block)

        # way of augmenting instance-aware embedding: bottleneck or average
        self.use_bottle_aug =  bottle_aug_cfg != empy_dict
        if self.use_bottle_aug:
            self.bottle_aug = SelfAttentionBlock(TransformerLayer(bottle_aug_cfg.hid_dim, MultiHeadAttention(bottle_aug_cfg.nhead, bottle_aug_cfg.hid_dim),
                                                                 PositionwiseFeedForward(bottle_aug_cfg.hid_dim, bottle_aug_cfg.ffn_dim), bottle_aug_cfg.dropout))
            self.bottle_regressor = nn.Sequential(nn.Linear(bottle_aug_cfg.hid_dim, 64), nn.ReLU(),
                                                 nn.Dropout(0.3),nn.Linear(64, 32), nn.Dropout(0.3),
                                                 nn.Linear(32, 1), nn.Sigmoid())                                     
            
        # build Anomaly Decoder:  'plain' , 'rnn' or 'memory bank'
        self.anomaly_decoder_type =  ano_decoder_cfg.type
        if ano_decoder_cfg.type == 'plain':
            self.anomaly_regressor = Plain_regressor(**ano_decoder_cfg.regressor)

        elif ano_decoder_cfg.type == 'rnn':
            self.reducer = Reducer(**ano_decoder_cfg.reducer)
            self.anomaly_regressor = Rnn_regressor(**ano_decoder_cfg.regressor)
            
        elif ano_decoder_cfg.type == 'memory':
            self.query_tokens = nn.Parameter(torch.zeros(1, ano_decoder_cfg.num_query_token, ano_decoder_cfg.hid_dim))   
            self.query_tokens.data.normal_(mean=0.0, std=ano_decoder_cfg.initializer_range)
            self.memory_bank_length = ano_decoder_cfg.memory_bank_length
            self.apply_vis_mb = ano_decoder_cfg.apply_vis_mb
            self.apply_obj_mb = ano_decoder_cfg.apply_obj_mb
            if self.apply_vis_mb:
                # self.add_image_pe = CircularPositionalEmbedding(ano_decoder_cfg.visual_pe.peroid,ano_decoder_cfg.visual_pe.emb_dim)
                self.add_image_pe = SinusoidalPositionEmbeddings(ano_decoder_cfg.visual_pe.peroid,ano_decoder_cfg.visual_pe.emb_dim)
                self.ave_pool =  nn.AdaptiveAvgPool2d(ano_decoder_cfg.pool_shape)
            if self.apply_obj_mb:
                # self.add_object_pe = CircularPositionalEmbedding(ano_decoder_cfg.visual_pe.peroid,ano_decoder_cfg.visual_pe.emb_dim)   
                self.add_object_pe = SinusoidalPositionEmbeddings(ano_decoder_cfg.visual_pe.peroid,ano_decoder_cfg.visual_pe.emb_dim)   
            self.anomaly_regressor = Memoery_regressor(**ano_decoder_cfg.regressor)

        # proxy task
        self.use_proxy_task = proxy_task_cfg != empy_dict
        if self.use_proxy_task:
            self.proxy_tasks = proxy_task_cfg.task_names
            # instance anomaly detection
            if proxy_task_cfg.use_ins_anomaly:
                assert self.use_ins_decoder, 'instance anomaly detection requires instance decoder'
                if proxy_task_cfg.ins_anomaly_type == 'ffn':
                    self.ins_anomaly_type = 'ffn'
                    self.ins_anomaly_regressor = MLP(**proxy_task_cfg.ins_anomaly_ffn)
                elif proxy_task_cfg.ins_anomaly_type == 'memory':
                    self.ins_anomaly_type = 'memory'
                    # self.ins_anomaly_regressor = self.anomaly_regressor
                    self.ins_anomaly_regressor = Memoery_regressor(**proxy_task_cfg.regressor)

            if proxy_task_cfg.use_box_anormal:
                self.box_detect_regressor = MLP(**proxy_task_cfg.box_detect_ffn)
                         
        # self.apply(weights_init_)
        # init weights before attach the vision transformer 
        # self.vst_model = register_vst_model(**vst_cfg)
            
    def forward(self, imgs , boxes , rnn_state = None, frame_state = None):
        if self.vit_type == 'vst':
            # imgs: B, T, C, W, H -> B, C, T, W, H
            vst_img_emb = self.vst_model(imgs.permute(0,2,1,3,4)) # list[]
            if self.use_fpn:
                img_embs = self.fpn(vst_img_emb) # [B, C, H, W]
            else:
                img_embs = vst_img_emb[-1] # [B, C, T, H, W]
                if self.use_linear:
                    B, C, T, H, W = img_embs.shape
                    img_embs = img_embs.flatten(2).transpose(1, 2).contiguous()
                    img_embs = self.linear(img_embs)
                    img_embs = img_embs.permute(0,2,1).view(B,-1,T,H,W).contiguous()
                    img_embs = torch.mean(img_embs,dim=2)

        if self.vit_type == 'clip':
            _,img_embs = self.clip_vit(imgs) # [B, C, H*W]
            img_embs = self.clip_patch_head(img_embs.permute(0, 2, 1))
            B, HW, C = img_embs.shape
            img_embs = img_embs.view(B,C,int(HW**0.5),int(HW**0.5))
        
        img_cls_token = None # prompt decoder is mean
        if self.vit_type == 'dinov2':
            ret = self.vit_model(imgs)     
            img_cls_token , img_embs  = ret['x_norm_clstoken'] , ret['x_norm_patchtokens']
            B, HW, C = img_embs.shape
            # imput image size satisfy : h:w = 9:16, so scale = (HW/(9*16))**0.5
            scale = (HW/(9*16))**0.5
            H, W = int(9*scale) , int(16*scale)
            img_embs = img_embs.permute(0, 2, 1).view(B,C,H,W) # [B,H*W,C] ->[B,C,H,W]

        object_embs = None    
        if self.use_ins_decoder :
            # image to instance : similar to SAM-Mask Encoder 
            instance_tokens , instacne_embs = self.instance_encoder( img_embs , boxes )
            B,C,H,W = img_embs.shape
            if self.instance_decoder == 'mean':
                instacne_embs_mean = torch.stack([ torch.mean(x,dim=0) for x in instacne_embs],dim=0)
                instance_tokens_mean = torch.concat([torch.mean(x,dim=0,keepdim=True) for x in instance_tokens],dim=0)
                img_cls_token = self.ins_reducer(instacne_embs_mean,object_embs=None,cls_tokens=instance_tokens_mean)
                batch_object_tokens = instance_tokens 
                pass
                
            else:
                instance_query = self.instance_query.expand(B,-1,-1)
                # instance to ins_query: loop in batch  
                batch_object_embs = []
                batch_object_tokens = []
                for ins_tokens, ins_embs, ins_query in zip(instance_tokens,instacne_embs,instance_query):
                    # ins_tokens tensor[1,N_obj,C]   instance_query tensor[1,N_query,C]
                    ins_tokens , ins_query = ins_tokens.unsqueeze(dim=0) , ins_query.unsqueeze(dim=0)
                    # augment intance-aware embedding
                    if self.use_bottle_aug:
                        bottle_weight = self.bottle_aug(ins_tokens)
                        bottle_weight = self.bottle_regressor(bottle_weight)
                        ins_embs = bottle_weight.view(-1,1,1,1)*ins_embs
                        ins_embs = torch.sum(ins_embs, dim=0, keepdim=True)
                    else:
                        ins_embs = torch.mean(ins_embs,dim=0,keepdim=True)

                    # BxCxHxW -> BxHWxC == B x N_image_tokens x C
                    ins_embs = ins_embs.flatten(2).permute(0, 2, 1)
                    # add image position encoding: [1, C, H, W] -> [1, C, HW] -> [1,HW,C]
                    embs_pe = self.instance_encoder.prompt_encoder.get_dense_pe().flatten(2).permute(0,2,1)
                    object_embs, object_tokens =  self.instance_decoder(ins_query, ins_embs, ins_tokens, img_pe = embs_pe)
                    batch_object_tokens.append(object_tokens.squeeze(dim=0))
                    # object_embs =  self.instance_decoder(ins_query, ins_embs, ins_tokens)
                    batch_object_embs.append(object_embs)
                # object_embs : [B, 1, num_query, C] -> [B, num_query, C]
                object_embs = torch.stack(batch_object_embs,dim=0).squeeze(dim=1)
        
        '''
        return :

        '''
        ret = {'output':None, 'rnn_state':None , 'ins_anomaly':None}


        '''
        ffn for instance anomaly detection 
        '''
        if self.use_proxy_task and 'instance' in self.proxy_tasks and self.ins_anomaly_type == 'ffn': 
            ret['ins_anomaly'] = []
            for frame_instance_tokens in batch_object_tokens: # instance_tokens | batch_object_tokens
                ins_anomaly_output = self.ins_anomaly_regressor(frame_instance_tokens)
                ret['ins_anomaly'].append(ins_anomaly_output)
      
        if self.anomaly_decoder_type == 'plain':
            if self.vit_type == 'dinov2':
                output = self.anomaly_regressor(img_embs, cls_tokens = img_cls_token)
            else:
                output = self.anomaly_regressor(img_embs)      
            ret['output'] = output 
            
        elif self.anomaly_decoder_type == 'rnn':
            if self.vit_type == 'dinov2':
                x = self.reducer(img_embs, object_embs, cls_tokens = img_cls_token)
            else:
                x = self.reducer(img_embs, object_embs, cls_tokens = img_cls_token)
                output, rnn_state = self.anomaly_regressor(x,rnn_state)
            ret['output'], ret['rnn_state'] = output , rnn_state
 
        elif self.anomaly_decoder_type == 'memory':
            if self.vit_type == 'vst':
                # only use the last vst emb : [B , C , 15 , 20]       
                # img_embs = vst_img_emb[-1].flatten(2).permute(0, 2, 1) # [B, N=HxW, C]
                # ave-pooling
                img_embs = self.ave_pool(img_embs).flatten(2).permute(0, 2, 1) # [B , C , 60 , 80] -> [B , C , 6 , 8] -> [B , 36, C]    

            elif self.vit_type == 'dinov2':
                # only use the last vst emb : [B , C , 15 , 20]       
                img_embs = img_embs.flatten(2).permute(0, 2, 1) # [B, N=HxW, C]

            B, N_img, C = img_embs.shape
            
            query_tokens = self.query_tokens.expand(B, -1, -1) # [B, N_query, C]
            if self.apply_vis_mb:
                img_embs = self.add_image_pe(img_embs, frame_state.t)
                img_embs = img_embs.unsqueeze(1) # [B, 1, N, C]

            if self.apply_obj_mb:
                N_obj = object_embs.shape[1]
                object_embs = self.add_object_pe(object_embs, frame_state.t) # [B, obj_query, C]
                object_embs = object_embs.unsqueeze(1) # [B, 1, obj_query, C]
            
            img_hidden_states , obj_hidden_states = None , None
            if frame_state.t == frame_state.begin_t:
                if self.apply_vis_mb:
                    img_hidden_states  = img_embs # [B, 1, N, C]
                    self.img_size_constant = torch.ones(B, 1, N_img).to(img_embs.device) # [B, 1, N]
                if self.apply_obj_mb:
                    obj_hidden_states  = object_embs # [B, 1, obj_query, C]
                    self.obj_size_constant = torch.ones(B, 1, N_obj).to(object_embs.device) # [B, 1, obj_query]
            else:
                if self.apply_vis_mb:
                    img_hidden_states = torch.cat([self.visual_memory_bank, img_embs], dim=1) # [B, (t+1), N, C]
                if self.apply_obj_mb:
                    obj_hidden_states = torch.cat([self.object_memory_bank, object_embs], dim=1) # [B, (t+1), obj_query, C]

            img_hidden_states = img_hidden_states.view(B, -1, C) if img_hidden_states is not None else img_hidden_states
            obj_hidden_states = obj_hidden_states.view(B, -1, C) if obj_hidden_states is not None else obj_hidden_states
            output = self.anomaly_regressor( query =  query_tokens ,img_emb = img_hidden_states , obj_emb = obj_hidden_states , frame_state = frame_state)
            
            '''
            mememory_regressor for instance anomaly detection 
            '''
            if self.use_proxy_task and 'instance' in self.proxy_tasks and self.ins_anomaly_type == 'memory':
                ret['ins_anomaly'] = []
                for frame_ins_tokens, frame_img_states, frame_obj_states in zip(instance_tokens,img_hidden_states,obj_hidden_states):
                    frame_ins_tokens = frame_ins_tokens.unsqueeze(dim=0)
                    frame_img_states = frame_img_states.unsqueeze(dim=0)
                    frame_obj_states = frame_obj_states.unsqueeze(dim=0)
                    ins_anomaly_output = self.ins_anomaly_regressor( frame_ins_tokens, frame_img_states, frame_obj_states , frame_state )
                    ret['ins_anomaly'].append(ins_anomaly_output)

            # If it is the first frame, initialize the visual_memory_bank as the embedding of the first frame
            # If not, concatenate the visual_memory_bank with the current frame embedding and update the compression_size
            if frame_state.t == frame_state.begin_t:
                if self.apply_vis_mb: 
                    self.visual_memory_bank = img_embs.detach()  # [B, 1, N, C]
                    self.visual_compression_size = self.img_size_constant  # [B, 1, N]
                if self.apply_obj_mb:
                    self.object_memory_bank = object_embs.detach()  # [B, 1, obj_query, C]
                    self.object_compression_size = self.obj_size_constant  # [B, 1, obj_query]
            else:
                if self.apply_vis_mb: 
                    self.visual_memory_bank = torch.cat([self.visual_memory_bank, img_embs.detach()], dim=1)  # [B, t+1, N, C]
                    self.visual_compression_size = torch.cat([self.visual_compression_size, self.img_size_constant], dim=1)  # [B, t+1, N]
                if self.apply_obj_mb:
                    self.object_memory_bank = torch.cat([self.object_memory_bank, object_embs.detach()], dim=1)  # [B, t+1, obj_query, C]
                    self.object_compression_size = torch.cat([self.object_compression_size, self.obj_size_constant], dim=1)  # [B, t+1, obj_query]

            # If it is the last frame, delete the visual_memory_bank and compression_size
            # Else, if the current length of the visual_memory_bank exceeds the threshold, compress the visual_memory_bank
            if frame_state.t == frame_state.T :
                if self.apply_vis_mb: 
                    del self.visual_memory_bank
                    del self.visual_compression_size
                if self.apply_obj_mb:
                    del self.object_memory_bank
                    del self.object_compression_size
            else:
                if self.apply_vis_mb and self.visual_memory_bank.size(1) > self.memory_bank_length: 
                    self.visual_memory_bank, self.visual_compression_size = memory_bank_compress(self.visual_memory_bank, self.visual_compression_size)
                if self.apply_obj_mb and self.object_memory_bank.size(1) > self.memory_bank_length:
                    self.object_memory_bank, self.object_compression_size = memory_bank_compress(self.object_memory_bank, self.object_compression_size)

            ret['output'] = output 
        
        # proxy task : instance anormal detection in one frame
        
        return ret
