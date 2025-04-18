from typing import Dict, List, Optional, Tuple, Union, Sequence

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from ..modeling.tuna_injector import TunaInjector
from ..utils.model_utils import c2_xavier_fill
from ..modeling.vit import ImageEncoderViT

def load_checkpoint():
    state_dict = torch.load('pretrain/sam_vit_b_01ec64.pth')
    re = {}
    for k, v in state_dict.items():
        if k.startswith("image_encoder"):
            re[k.replace("image_encoder.", "")] = v

    return re

@MODELS.register_module()
class ViTSAM(BaseModule):
    def __init__(self, **kwargs):
        super().__init__()
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        self.model = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=prompt_embed_dim,
            use_abs_pos=False,
        ).cuda()
        TunaInjector.inject_vit_sam_mona(self.model)
        self.model.load_state_dict(load_checkpoint(), strict=False)
        # Freeze parameters
        self.freeze_parameters()

    def freeze_parameters(self):
        self.model.eval()
        for name, param in self.model.named_parameters():
            if 'tuna_' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode=mode)
        
        self.freeze_parameters()
            
        
    def forward(self, x):
        x = self.model(x)
        # for i in x:
        #     print(i.shape)
        # exit()
        return x