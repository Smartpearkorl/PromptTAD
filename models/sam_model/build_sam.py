# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None , no_load_vit=False, decoder_type='full'):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        no_load_vit=no_load_vit,
        decoder_type=decoder_type,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None, no_load_vit=False, decoder_type='full'):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        no_load_vit=no_load_vit,
        decoder_type=decoder_type,
    )


def build_sam_vit_b(checkpoint=None,no_load_vit=False, decoder_type='full'):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        no_load_vit=no_load_vit,
        decoder_type=decoder_type,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    for_anomaly_detection = False, 
    no_load_vit = False,
    decoder_type = 'full', 
):
    '''
    decoder_type:
    'full': original sam for segmentation
    'TAD_base': only load sam image-encoder VIT (case no_load_vit is True)
    'TAD_prompt': sam for TAD  
    '''
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # whether load vit
    if no_load_vit:
        image_encoder = None
    else:
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
    # which Encoder Decoder type
    if decoder_type == 'TAD_base':
        prompt_encoder , mask_decoder = None , None 
    else:
        for_anomaly_detection = decoder_type == 'TAD_prompt'
        prompt_encoder=PromptEncoder(
            for_anomaly_detection = for_anomaly_detection,
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16)
        
        mask_decoder=MaskDecoder(
            for_anomaly_detection = for_anomaly_detection,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256)
    
    sam = Sam(
        image_encoder = image_encoder,
        prompt_encoder = prompt_encoder,
        mask_decoder = mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    # sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
            # DDP need to find_unused_parameters = False to accelerate 
            if for_anomaly_detection:
                state_dict['prompt_encoder.point_embeddings.0.weight'] = state_dict['prompt_encoder.point_embeddings.2.weight']  
                state_dict['prompt_encoder.point_embeddings.1.weight'] = state_dict['prompt_encoder.point_embeddings.3.weight']  
                
        sam.load_state_dict(state_dict,strict=False)
    return sam
