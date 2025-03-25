import torch
from torch import nn ,Tensor
from .Transformer import *
from models.prompt_models.points_transform import generalized_box_iou
from scipy.optimize import linear_sum_assignment
import numpy as np
'''
匈牙利匹配: 
'''
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: type is list and len is N_frames , consist of tensor of dim ([N_object,4] ) with the predicted box coordinates(xyxy)
            targets: type is list and len is N_frames , consist of tensor of dim ([N_object,4] ) with the target box coordinates(xyxy)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_predict_object, num_target_boxes)
        """
        bs , bs_check = len(outputs) , len(targets)
        assert bs == bs_check, f'output frames : {bs} is not equal to target {bs_check}'
        output_size = [ x.shape[0] for x in outputs]
        target_size = [ x.shape[0] for x in targets]

        out_bbox = torch.cat([torch.as_tensor(x) for x in outputs])
        tgt_bbox = torch.cat([torch.as_tensor(x) for x in targets])

        # 如果所有 frame 都没有框 直接返回 空 list
        if out_bbox.shape[0] ==0 or tgt_bbox.shape[0]==0:
            return [(np.array([], dtype=np.int64), np.array([], dtype=np.int64)) for _ in range(bs)]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
        C = C.split(output_size, dim = 0)
        C2 = [x.split(target_size, dim = -1) for x in C] 
        indices = [linear_sum_assignment(C2[i][i]) for i in range(bs)]
        return indices

'''
Instancce Decoder: tokenize object-aware feature embedding
'''
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(RotaryPositionalEmbedding, self).__init__()
    
        self.rotation_matrix = torch.zeros(d_model, d_model )
        self.positional_embedding = torch.zeros(max_seq_len, d_model)
        
        for i in range(d_model):
            for j in range(d_model):
                self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)

        for i in range(max_seq_len):
            for j in range(d_model):
                self.positional_embedding[i, j] = torch.cos(i * j * 0.01)

    def forward(self, x ):
        """
        Args:
            x: 一个形状为 (batch_size, seq_len, d_model) 的张量。

        Returns:
            返回一个形状为 (batch_size, seq_len, d_model) 的张量。
        """
        # 添加位置嵌入到输入张量
        x = x + self.positional_embedding[::x.size(1), :]

        # 应用旋转矩阵到输入张量
        x = torch.matmul(x, self.rotation_matrix)

        return x

class CircularPositionalEmbedding(nn.Module):
    def __init__(self, period, embedding_dim):
        super(CircularPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.period = period
        self.image_pe = nn.Embedding(period, embedding_dim)
        nn.init.constant_(self.image_pe.weight, 0.0)

    def forward(self, image_embeds, t):
        B, N, C = image_embeds.shape

        # 计算位置编码索引
        if isinstance(t, int):
            position_ids = torch.tensor([t % self.period]).long().to(image_embeds.device)  # [1]
        elif isinstance(t, torch.Tensor):
            position_ids = (t % self.period).long().to(image_embeds.device)

        position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, 1]
        image_embeds = image_embeds + self.image_pe(position_ids)  # [B, N, C]
        return image_embeds

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, period,  embedding_dim):
        super().__init__()
        self.period = period
        total_time_steps = period
        time_emb_dims = 128
        half_dim = time_emb_dims // 2   
        time_emb_dims_exp = embedding_dim
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        ts = torch.arange(total_time_steps, dtype=torch.float32)
        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, image_embeds, t):
        B, N, C = image_embeds.shape

        # 计算位置编码索引
        if isinstance(t, int):
            position_ids = torch.tensor([t % self.period]).long().to(image_embeds.device)  # [1]
        elif isinstance(t, torch.Tensor):
            position_ids = (t % self.period).long().to(image_embeds.device)

        return image_embeds + self.time_blocks(position_ids)

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.contiguous().view(self.shape)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Reducer(nn.Module):
    def __init__(self, has_obj_embs, pool_shape, pool_dim, dim_latent, dropout, mlp_dim = None, mlp_depth = None,  dim_scale = 1 ):
        super(Reducer, self).__init__()
         # scale for dim_latent : dinov2 scale = 2 (cls_token + reduced image embbedins)
        cls_dim_latent = int(dim_latent*dim_scale)
        pooling_func = nn.AdaptiveAvgPool3d if len(pool_shape) == 3 else nn.AdaptiveAvgPool2d
        self.reducer = nn.Sequential(pooling_func(pool_shape),
                                     View((-1,pool_dim)),
                                     nn.LayerNorm(pool_dim),
                                     nn.Linear(pool_dim,dim_latent),             
                                     nn.ReLU(),nn.Dropout(dropout))
        self.has_obj_embs = has_obj_embs # 'True' ->mlp for obj_embs
        if has_obj_embs:
            self.mlp = MLP(mlp_dim,dim_latent,dim_latent,mlp_depth)
            self.cls = nn.Sequential(nn.Linear(dim_latent+cls_dim_latent,dim_latent),
                                        nn.ReLU(),nn.Dropout(dropout))
        else:
            self.cls = nn.Sequential(nn.Linear(cls_dim_latent,dim_latent),
                                        nn.ReLU(),nn.Dropout(dropout))

    def forward(self,img_embs, object_embs, cls_tokens=None):
      
        x = self.reducer(img_embs)
        if cls_tokens is not None:
            x = torch.cat((x,cls_tokens),dim=1)

        if self.has_obj_embs:
            B = img_embs.shape[0]
            obj_x = self.mlp(object_embs.view(B,-1))
            x = torch.cat((x,obj_x),dim=1)
               
        x = self.cls(x)
        return x  

class Plain_regressor(nn.Module):
    def __init__(self, pool_shape, pool_dim, dim_latent, dropout, dim_scale =1):
        super(Plain_regressor, self).__init__()
        # scale for dim_latent
        cls_dim_latent = int(dim_latent*dim_scale)
        self.reducer = nn.Sequential(nn.AdaptiveAvgPool2d(pool_shape),
                                     View((-1,pool_dim)),
                                     nn.LayerNorm(pool_dim),
                                     nn.Linear(pool_dim,dim_latent),             
                                     nn.ReLU(),nn.Dropout(dropout))
        
        self.cls = nn.Sequential( nn.Linear(cls_dim_latent, dim_latent),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(dim_latent, 2))
        
    def forward(self,img_embs,cls_tokens=None):
        x = self.reducer(img_embs)
        # dinov2 :concate
        if cls_tokens!=None: 
            x = self.cls(torch.concat((x,cls_tokens),dim=1))
        else:
            x = self.cls(x)
        return x

class Rnn_regressor(nn.Module):
    def __init__(self, dim_latent , rnn_state_size , rnn_cell_num, dropout ):
        super(Rnn_regressor, self).__init__()
        self.ln = nn.LayerNorm(dim_latent)
        self.rnn = nn.LSTM(dim_latent,rnn_state_size,rnn_cell_num)
        self.cls = nn.Sequential( nn.Linear(rnn_state_size, rnn_state_size),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(rnn_state_size, 2))

    def forward(self,x,rnn_state):
        x = self.ln(x)
        # [B,C] -> [1,B,C] for LSTM input [T,B,C]
        x = x.unsqueeze(0)
        x, rnn_state = self.rnn(x, rnn_state)
        x = x.squeeze(0)
        hx, cx = rnn_state
        rnn_state = (hx.detach(), cx.detach())
        x = self.cls(x)
        return x , rnn_state
    
def memory_bank_compress(memory_bank: torch.Tensor, compression_size: torch.Tensor) -> tuple:
    """
    Compresses the memory bank if the current memory bank length is greater than the threshold.
    Compression_size is the number of frames that are compressed into each position.
    
    Args:
        memory_bank (torch.Tensor): The input memory bank to compress. Shape: (B, T, N, C)
        compression_size (torch.Tensor): The number of frames to compress into each position. Shape: (B, T, N)
    
    Returns:
        compressed_memory_bank (torch.Tensor): The compressed memory bank. Shape: (B, T-1, N, C)
        compressed_size (torch.Tensor): The number of frames compressed into each position. Shape: (B, T-1, N)
    """
    B, T, N, C = memory_bank.shape
    # Calculate the cosine similarity between adjacent frames
    similarity_matrix = F.cosine_similarity(memory_bank[:, :-1, :], memory_bank[:, 1:, :], dim=-1)
    # Select the frame indices with the top-1 similarity 
    _, max_similarity_indices = torch.max(similarity_matrix, dim=1, keepdim=True)

    # Calculate source and dst indices for compression
    src_indices = max_similarity_indices + 1
    dst_indices = torch.arange(T - 1).to(memory_bank.device)[None, :, None].repeat(B, 1, N)
    dst_indices[dst_indices > max_similarity_indices] += 1

    # Gather source and dst memory banks and sizes
    src_memory_bank = memory_bank.gather(dim=1, index=src_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    dst_memory_bank = memory_bank.gather(dim=1, index=dst_indices.unsqueeze(-1).expand(-1, -1, -1, C))
    src_size = compression_size.gather(dim=1, index=src_indices)
    dst_size = compression_size.gather(dim=1, index=dst_indices)

    # Multiply the memory banks by their corresponding sizes
    src_memory_bank *= src_size.unsqueeze(-1)
    dst_memory_bank *= dst_size.unsqueeze(-1)

    # Compress the memory bank by adding the source memory bank to the dst memory bank
    dst_memory_bank.scatter_add_(dim=1, index=max_similarity_indices.unsqueeze(-1).expand(-1, -1, -1, C), src=src_memory_bank)
    dst_size.scatter_add_(dim=1, index=max_similarity_indices, src=src_size)

    # Normalize the dst memory bank by its size
    compressed_memory_bank = dst_memory_bank / dst_size.unsqueeze(-1)
    return compressed_memory_bank, dst_size

class Query_transformer(nn.Module):
    def __init__(self, apply_vis_mb, apply_obj_mb, hid_dim , nhead , ffn_dim , dropout , memory_bank_length, Skip_selfattn = False):
        super(Query_transformer, self).__init__()
        self.apply_vis_mb = apply_vis_mb
        self.apply_obj_mb = apply_obj_mb
        self.memory_bank_length = memory_bank_length
        self.Skip_selfattn = Skip_selfattn
        # instance anomaly detection : skip query2query block for no time connection among querys
        if not self.Skip_selfattn:
            self.query2query =  CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
         
        if self.apply_vis_mb:
            self.query2frame = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
        if self.apply_obj_mb:
            self.query2object = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))     
                    
    def forward(self,hidden_states,frame_states,obj_states,frame_state):
        if  self.Skip_selfattn:
            if self.apply_obj_mb:
                x = self.query2object(hidden_states,obj_states,obj_states)
            if self.apply_vis_mb:
                x = self.query2frame(x,frame_states,frame_states)
            return x , frame_states , obj_states
        
        else:
            # prepare memmory bank
            if hasattr(self, 'query_memory_bank'):
                B, T, N, C = self.query_memory_bank.shape
                query_memory_bank = self.query_memory_bank.view(B, -1, C) # [B, T*N_query, C]
                q_with_bank = torch.cat([query_memory_bank,hidden_states], dim=1) #[B, (T+1)*N_query, C]
            else:
                q_with_bank = hidden_states
            
            x = self.query2query(hidden_states,q_with_bank,q_with_bank)

            if self.apply_obj_mb:
                x = self.query2object(x,obj_states,obj_states)
            if self.apply_vis_mb:
                x = self.query2frame(x,frame_states,frame_states)

            B, N, C = hidden_states.shape
            # if it is the first frame, initialize the query_memory_bank as the first frame's query embedding
            # if not, concatenate the query_memory_bank with the current frame embedding and update the compression_size
            if not hasattr(self, 'query_memory_bank'):
                self.query_memory_bank = hidden_states[:, None, :, :].detach() # [B, 1, N_query, C]
                self.size_constant = torch.ones(B, 1, N).to(hidden_states.device) # [B, 1, N_query]
                self.compression_size = self.size_constant
            else:
                self.query_memory_bank = torch.cat([self.query_memory_bank, hidden_states[:, None, :, :].detach()], dim=1) # [B, t+1, N_query, C]
                self.compression_size = torch.cat([self.compression_size, self.size_constant], dim=1) # [B, t+1, N_query]

            # if it is the last frame, delete the query_memory_bank and compression_size
            # else if the current length of the query_memory_bank exceeds the threshold, compress the query_memory_bank
            if frame_state.t == frame_state.T:
                del self.query_memory_bank
                del self.compression_size
            elif self.query_memory_bank.size(1) > self.memory_bank_length:
                self.query_memory_bank, self.compression_size = memory_bank_compress(self.query_memory_bank, self.compression_size)

            return x , frame_states , obj_states 

class Memoery_regressor(nn.Module):
    def __init__(self,  Qformer_cfg , Qformer_depth, hidden_size , dim_latent , dropout , N_for_cls = False):
        super(Memoery_regressor, self).__init__()
        self.query_layer = clones(Query_transformer(**Qformer_cfg), Qformer_depth)
        self.n_layers = Qformer_depth
        self.cls = nn.Sequential( nn.Linear(hidden_size, dim_latent),
                                  nn.ReLU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(dim_latent, 2))
        self.N_for_cls = N_for_cls

    def forward(self, query: Tensor ,img_emb: Tensor , obj_emb: Tensor , frame_state ):   
        for layer_i in range(self.n_layers):
            query, img_emb, obj_emb = self.query_layer[layer_i](query, img_emb, obj_emb, frame_state)
        
        B,N,C = query.shape
        if self.N_for_cls:
            assert B==1, f'when N dimension for clssification ,Batchsize should be 1, but got {B}'
            query = query.view(N,C)
            output = self.cls(query)
            return output

        else:
            query = query.view(B,-1)
            output = self.cls(query)
            return output

class Memoery_Block(nn.Module):
    def __init__(self,  Qformer_cfg , Qformer_depth):
        super(Memoery_Block, self).__init__()
        self.query_layer = clones(Query_transformer(**Qformer_cfg), Qformer_depth)
        self.n_layers = Qformer_depth

    def forward(self, query: Tensor ,img_emb: Tensor , obj_emb: Tensor , frame_state ):   
        for layer_i in range(self.n_layers):
            query, img_emb, obj_emb = self.query_layer[layer_i](query, img_emb, obj_emb, frame_state)

        return query