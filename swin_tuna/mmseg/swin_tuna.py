from typing import Dict, List, Optional, Tuple, Union, Sequence

import addict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from ..modeling.tuna_injector import TunaInjector
from ..utils.model_utils import c2_xavier_fill

@MODELS.register_module()
class SwinTransformerTuna(BaseModule):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs['type'] = 'SwinTransformer'
        self.model: nn.Module = MODELS.build(kwargs)
        # Inject mona modules
        TunaInjector.inject_tuna(self.model)
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