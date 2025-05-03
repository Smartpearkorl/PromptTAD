import torch
import math

from torch import nn
from torch.nn import functional as F
import functools
import operator
from models.TTHF import open_clip_local as clip
from models.prompt_models.multiscale_vst import register_vst_model
from models.prompt_models.vst_fpn import Standard_FPN
from models.prompt_models.aggregation import Instance_Encoder, Instance_Decoder_V2
from models.Transformer import *
from models.componets import *


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

        img_cls_token = None # prompt decoder is mean
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
            for frame_instance_tokens in batch_object_tokens: # batch_object_tokens = instance_tokens when self.instance_decoder == 'mean' 
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
       
        return ret
