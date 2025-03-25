# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import vision_transformer as vits
from models.dinov2.dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14,dinov2_vitg14
from models.dinov2.dinov2.models.dinov2_lora import DINOV2EncoderLoRA
import torch

logger = logging.getLogger("dinov2")


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)

vit_dim_map = {'vits14':384, 'vitb14':768, 'vitl14':1024, 'vitg14':1536}

dinov2_model_mapping = {
    'vits14': dinov2_vits14,  
    'vitb14': dinov2_vitb14,
    'vitl14': dinov2_vitl14,
    'vitg14': dinov2_vitg14
}

class Fake_vit():
    def __init__(self) -> None:
        pass

    def __call__(self, emb):
        emb = emb.squeeze(dim=1)
        return {"x_norm_clstoken":emb[:,0],
                "x_norm_patchtokens":emb[:, 1:]}
         
def register_vit_model(type,checkpoint,load_emb=False,use_lora=False):
    assert not (load_emb and use_lora) , f'load_emb and use_lora cannot be True at the same time' 
    # 正常load vit
    if not load_emb:
        encoder = dinov2_model_mapping[type](pretrained=False)
        model_dict = torch.load(checkpoint, map_location="cpu")
        encoder.load_state_dict(model_dict, strict=True)
        model = DINOV2EncoderLoRA(encoder, emb_dim = vit_dim_map[type] , use_lora=use_lora )
    else:
        model = Fake_vit()
    return model
