from torch import nn ,Tensor
import torch
import numpy as np

from models.prompt_models.points_transform import ResizeCoordinates
from models.prompt_models.prompt_modules import *
from models.Transformer import *

'''
                 Instance-wise Aggregation
target:
    transform object bounding box to instance-aware feature embedding with image feature embedding
                 
module included:
       1. Object-promopt Encoder: encode bounding box coordinate into feature embedding
       2. Instance-wise Aggregation: aggregate instance-aware feature embedding 
'''
class Instance_Encoder(nn.Module):
    def __init__(self,    
        prompt_encoder_cfg:dict,
        resize_cfg:dict,
        prompt_decoder_cfg:dict,
        ):
        super().__init__()
        self.prompt_encoder = PromptEncoder(**prompt_encoder_cfg)
        self.transform = ResizeCoordinates(**resize_cfg)
        self.prompt_decoder = PromptDecoder(**prompt_decoder_cfg)

    def forward(self, img_emb: Tensor , boxes_batch: np.array ): # points: np.array
        # need to loop in batch for building instnce-aware embedding 
        batch_ins_tokens , batch_ins_embs = [], []
        for frame_emb , frame_boxes  in zip(img_emb,boxes_batch):  
            frame_boxes = self.transform.apply_boxes(frame_boxes)
            box_torch = torch.as_tensor(frame_boxes, dtype=torch.float, device=img_emb.device)
           
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.prompt_encoder(points = None, boxes = box_torch, masks = None)
            instance_tokens, instacne_embs = self.prompt_decoder(
                image_embeddings=frame_emb.unsqueeze(dim=0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,)

            batch_ins_tokens.append(instance_tokens)
            batch_ins_embs.append(instacne_embs)
        return  batch_ins_tokens , batch_ins_embs
    

'''
                Relation-wise Aggregation: [add positional encoding]
target:
    aggregate all instance-aware feature embedding into inhanced frame token
'''
class Instance_block_V2(nn.Module):
    def __init__(self,hid_dim,nhead,ffn_dim,dropout):
        super(Instance_block_V2, self).__init__()
        self.query2object = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
        self.query2frame = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
        self.object2qurey = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))   

    def forward(self, query: Tensor, img_emb: Tensor , obj_emb: Tensor , query_pe:Tensor , img_pe : Tensor, obj_pe: Tensor):         
        attn_out = self.query2object(query + query_pe, obj_emb + obj_pe, obj_emb )
        query = query + attn_out
        attn_out = self.query2frame(query + query_pe, img_emb + img_pe, img_emb )
        query = query + attn_out
        attn_out = self.object2qurey(obj_emb + obj_pe, query + query_pe, query )
        obj_emb = obj_emb + attn_out

        return query , img_emb , obj_emb 

class Instance_Decoder_V2(nn.Module):
    def __init__(self,block_depth,block_cfg):
        super(Instance_Decoder_V2, self).__init__()
        self.layer = clones(Instance_block_V2(**block_cfg),block_depth)
        self.n_layers = block_depth
    def forward(self, query: Tensor ,img_emb: Tensor , obj_emb: Tensor , img_pe : Tensor):
        for layer_i in range(self.n_layers):
            query, img_emb, obj_emb = self.layer[layer_i](query, img_emb, obj_emb , query_pe = query, img_pe = img_pe , obj_pe = obj_emb )
        return query , obj_emb


'''
Relation-wise Aggregation(Deprecated)
'''
class Instance_block(nn.Module):
    def __init__(self,hid_dim,nhead,ffn_dim,dropout,apply_obj2query=True):
        super(Instance_block, self).__init__()
        self.apply_obj2query = apply_obj2query
        self.query2object = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
        self.query2frame = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
        # last block no object2qurey for no grad in it        
        if apply_obj2query:
            self.object2qurey = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))   
            # self.frame2query = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
            
    def forward(self, query: Tensor, img_emb: Tensor , obj_emb: Tensor ,img_pe : Tensor): 
        if img_pe!= None:
            img_emb = img_pe + img_emb
        query = self.query2object(query, obj_emb, obj_emb )
        query = self.query2frame(query, img_emb, img_emb )
        if self.apply_obj2query:
            obj_emb = self.object2qurey(obj_emb, query, query )
            # img_emb = self.frame2query(img_emb, query, query )
        return query , img_emb , obj_emb

class Instance_Decoder(nn.Module):
    def __init__(self,block_depth,block_cfg):
        super(Instance_Decoder, self).__init__()
        self.layer = clones(Instance_block(**block_cfg),block_depth-1)
        self.layer.append(Instance_block(**block_cfg, apply_obj2query=False))
        self.n_layers = block_depth
    def forward(self, query: Tensor ,img_emb: Tensor , obj_emb: Tensor , img_pe = None):
        for layer_i in range(self.n_layers):
            query, img_emb, obj_emb = self.layer[layer_i](query, img_emb, obj_emb , img_pe = img_pe)
        return query

