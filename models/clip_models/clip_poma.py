import torch
import math

from torch import nn
from torch.nn import functional as F
import functools
import operator


from models.TTHF import open_clip_local as clip
from models.prompt_models.multiscale_vst import register_vst_model
from models.prompt_models.vst_fpn import Standard_FPN
from models.prompt_models.aggregation import Instance_Encoder
from models.Transformer import *
from models.componets import *

class clip_poma(nn.Module):
    def __init__(
        self,
        clip_cfg: dict = None,
        ins_encoder_cfg: dict =None,
        bottle_aug_cfg: dict = None,    

        ins_decoder_cfg = None, 
        memorybank_cfg = None,

        loss_cfg: dict = None,
        ):
        super().__init__()
        empy_dict = {}

        # prepare clip model
        self.clip = clip.create_model(**clip_cfg.model)
        tokenizer = clip.get_tokenizer(clip_cfg.model.model_name)
        self.accident_prompt = ['The traffic in this scenario is normal', 'A traffic anomaly occurred in the scene']
        self.tokenized_text = tokenizer(self.accident_prompt)
        # hf_vit_encoder
        self.diff_encoder = copy.deepcopy(self.clip.visual)
        # frezee clip vit
        for _, params in self.clip.visual.named_parameters():
            params.requires_grad = False
        self.clip.positional_embedding.requires_grad = False
        self.clip.text_projection.requires_grad = False
        
        self.clip_patch_head = nn.Sequential( nn.Linear(clip_cfg.patch_head.hid_dim * 2, clip_cfg.patch_head.hid_dim),nn.Dropout(0.2))
        self.vis_dim = self.clip.visual.output_dim

        # build Instance Encoder
        self.use_ins_encoder = ins_decoder_cfg != empy_dict
        if self.use_ins_encoder:
            self.instance_encoder = Instance_Encoder(ins_encoder_cfg.pmt_encoder, ins_encoder_cfg.resize,ins_encoder_cfg.pmt_decoder )

        # build Instance Decoder
        self.use_ins_decoder = ins_decoder_cfg != empy_dict
        if self.use_ins_decoder :
            self.ins_query_is_img_vec = ins_decoder_cfg.num_query_token == 0 # use clip vectors 
            if not self.ins_query_is_img_vec:
                self.instance_query = nn.Parameter(torch.zeros(1, ins_decoder_cfg.num_query_token, ins_decoder_cfg.hid_dim))
                self.instance_query.data.normal_(mean=0.0, std=ins_decoder_cfg.initializer_range)
            self.instance_decoder = Instance_Decoder(ins_decoder_cfg.block_depth, ins_decoder_cfg.block)

        # way of augmenting instance-aware embedding: bottleneck or average
        self.use_bottle_aug =  bottle_aug_cfg != empy_dict
        if self.use_bottle_aug:
            self.bottle_aug = SelfAttentionBlock(TransformerLayer(bottle_aug_cfg.hid_dim, MultiHeadAttention(bottle_aug_cfg.nhead, bottle_aug_cfg.hid_dim),
                                                                 PositionwiseFeedForward(bottle_aug_cfg.hid_dim, bottle_aug_cfg.ffn_dim), bottle_aug_cfg.dropout))
            self.bottle_regressor = nn.Sequential(nn.Linear(bottle_aug_cfg.hid_dim, 64), nn.ReLU(),
                                                 nn.Dropout(0.3),nn.Linear(64, 32), nn.Dropout(0.3),
                                                 nn.Linear(32, 1), nn.Sigmoid())                                     

        self.use_memory_bank =  memorybank_cfg != empy_dict                   
        if self.use_memory_bank:
            self.query_is_img_vec = memorybank_cfg.num_query_token == 0 # use clip vectors  
            if not self.query_is_img_vec:
                self.query_tokens = nn.Parameter(torch.zeros(1, memorybank_cfg.num_query_token, memorybank_cfg.hid_dim))   
                self.query_tokens.data.normal_(mean=0.0, std=memorybank_cfg.initializer_range)
            self.memory_bank_length = memorybank_cfg.memory_bank_length
            self.apply_vis_mb = memorybank_cfg.apply_vis_mb
            self.apply_obj_mb = memorybank_cfg.apply_obj_mb
            if self.apply_vis_mb:
                self.add_image_pe = CircularPositionalEmbedding(memorybank_cfg.visual_pe.peroid,memorybank_cfg.visual_pe.emb_dim)
            if self.apply_obj_mb:
                self.add_object_pe = CircularPositionalEmbedding(memorybank_cfg.visual_pe.peroid,memorybank_cfg.visual_pe.emb_dim)   
            self.memoery = Memoery_Block(**memorybank_cfg.regressor)

        N_times = self.use_ins_decoder + self.use_memory_bank
        if N_times:
            self.fusion_layer = nn.Sequential(nn.Linear((N_times+1) * self.vis_dim, self.vis_dim), nn.Dropout(0.2))
           
    def forward(self, imgs , boxes , frame_state = None):
        # vecs: [B,1024]  embs:  [B,2048，HW=49]
        img_vecs, img_embs = self.clip.visual(imgs[:,1]) 
        
        diff_img = imgs[:,1] - imgs[:,0]
        diff_vecs, diff_embs = self.diff_encoder(diff_img)
        
        img_diff_aug_vecs = img_vecs + diff_vecs
        img_diff_aug_vecs = img_diff_aug_vecs/img_diff_aug_vecs.norm(dim=-1,keepdim=True)

        img_embs = self.clip_patch_head(img_embs.permute(0, 2, 1))
        B, HW, C = img_embs.shape
        img_embs = img_embs.view(B,C,int(HW**0.5),int(HW**0.5))

        object_embs = None        
        if self.use_ins_decoder :
            instance_tokens , instacne_embs = self.instance_encoder( img_embs , boxes )
            B,C,H,W = img_embs.shape

            if self.ins_query_is_img_vec:
                instance_query = img_vecs.unsqueeze(dim=1) # [B,1,C] 
            else:
                instance_query = self.instance_query.expand(B,-1,-1)

            # loop in batch 
            batch_object_embs = []
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
                object_embs =  self.instance_decoder(ins_query, ins_embs, ins_tokens)
                batch_object_embs.append(object_embs)
            # object_embs : [B, 1, num_query, C] -> [B, num_query, C]
            object_embs = torch.stack(batch_object_embs,dim=0).squeeze(dim=1)
            img_ins_aug_vecs = object_embs[:,0,:]

        if  self.use_memory_bank:
            # only use the last vst emb : [B , C , 15 , 20]       
            img_embs = img_embs.flatten(2).permute(0, 2, 1) # [B, N=HxW, C]
            # img_embs = self.add_image_pe(img_embs ,frame_state.t)
            B, N_img, C = img_embs.shape
            
            if self.query_is_img_vec: 
                query_tokens = img_vecs.unsqueeze(dim=1) # [B,1,C]
            else:
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
            mb_query = self.memoery( query =  query_tokens ,img_emb = img_hidden_states , obj_emb = obj_hidden_states , frame_state = frame_state)
         
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

            img_mb_aug_vecs = mb_query[:,0,:]

        if self.use_ins_decoder and self.use_memory_bank:
            img_tokens = self.fusion_layer(torch.cat([img_diff_aug_vecs, img_ins_aug_vecs, img_mb_aug_vecs], -1))
        elif self.use_ins_decoder: # only use_ins_decoder
            img_tokens = self.fusion_layer(torch.cat([img_diff_aug_vecs, img_ins_aug_vecs], -1))
        elif self.use_memory_bank: # only use_memory_bank
            img_tokens = self.fusion_layer(torch.cat([img_diff_aug_vecs, img_mb_aug_vecs], -1))
        else: # plain
            img_tokens = img_diff_aug_vecs
        
        if self.use_ins_decoder or self.use_memory_bank:
            img_tokens = img_tokens / img_tokens.norm(dim=-1,keepdim=True)

        text_vecs = self.clip.encode_text(self.tokenized_text.to(img_embs.device))
        text_tokens = text_vecs / text_vecs.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = logit_scale * img_tokens @ text_tokens.t() # [B,C] @ [2,1024]
        logits_per_text = logit_scale * text_tokens @ img_tokens.t()

        return logits_per_image, logits_per_text
      

