from typing import Dict, List, Optional, Tuple, Union, Sequence

import addict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from ..modeling.tuna_injector import TunaInjector
from ..modeling.swin_sam import SwinTransformerSemanticSAM
from ..utils.model_utils import c2_xavier_fill

def load_checkpoint():
    state_dict = torch.load("pretrain/swinl_only_sam_many2many.pth")
    modified = {}
    for k, v in state_dict['model'].items():
        if not "model.backbone" in k: continue
        new_key = k.replace("model.backbone.", "")
            
        modified[new_key] = v
    return modified

@MODELS.register_module()
class SwinTransformerTunaSAM(BaseModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SwinTransformerSemanticSAM(**kwargs)
        # Inject mona modules
        TunaInjector.inject_sam_swin(self.model)
        self.model.load_state_dict(load_checkpoint(), strict=False)
        # Freeze parameters
        self.freeze_parameters()

    def freeze_parameters(self):
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
        return x