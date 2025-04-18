from typing import Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from ..modeling.tuna_injector import TunaInjector

@MODELS.register_module()
class ViTTuna(BaseModule):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs['type'] = 'VisionTransformer'
        self.model: nn.Module = MODELS.build(kwargs)
        # Inject mona modules
        TunaInjector.inject_vit(self.model)
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
        return x