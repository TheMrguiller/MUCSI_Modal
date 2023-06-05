# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np

import utils
from modeling_utils import BEiT3Wrapper, _get_base_config, _get_large_config, _get_mini_config, _get_tiny_config, _get_pipi_config


class TwoLayerMLP(nn.Module):
    def __init__(
            self, 
            in_features, 
            hidden_features, 
            out_features, 
            norm_layer, 
            norm_input=True, 
    ):
        super().__init__()
        self.norm1 = norm_layer(in_features) if norm_input else nn.Identity()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.norm2 = norm_layer(hidden_features)
        self.act = nn.GELU()
        self.dense2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.norm1(x)
        x = self.dense1(x)
        x = self.norm2(x)
        x = self.act(x)
        return self.dense2(x)


class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
            
        cls_rep = self.norm(cls_rep)
            
        pooled_output = self.dense(cls_rep)
            
        pooled_output = self.activation(pooled_output)
            
        return pooled_output



class BEiT3ForVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer, 
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), 
            norm_layer(embed_dim * 2), 
            nn.GELU(), 
            nn.Linear(embed_dim * 2, num_classes), 
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=question, 
            visual_tokens=image, 
            text_padding_position=padding_mask, 
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)

        return self.head(cls_rep)

@register_model
def beit3_base_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=224, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_base_patch16_224_pipi_vqav2(pretrained=False, **kwargs):
    args = _get_pipi_config(img_size=224, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model

@register_model
def beit3_base_patch16_224_tiny_vqav2(pretrained=False, **kwargs):
    args = _get_tiny_config(img_size=224, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model

@register_model
def beit3_base_patch16_224_mini_vqav2(pretrained=False, **kwargs):
    args = _get_mini_config(img_size=224, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model

@register_model
def beit3_base_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_base_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_384_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_480_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=480, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model


@register_model
def beit3_large_patch16_768_vqav2(pretrained=False, **kwargs):
    args = _get_large_config(img_size=768, **kwargs)
    args.normalize_output = False
    model = BEiT3ForVisualQuestionAnswering(args, num_classes=3129, **kwargs)
    return model