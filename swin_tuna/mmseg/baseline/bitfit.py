import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.backbones.swin import SwinTransformer, SwinBlock, SwinBlockSequence

from ..PEFT import PEFT


@MODELS.register_module() 
class SwinTransformerBitFit(PEFT):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = self.make_model(kwargs)
        # Freeze parameters
        self.freeze_parameters()
        
    def make_model(self, kwargs)-> SwinTransformer:
        kwargs['type'] = 'SwinTransformer'
        model = MODELS.build(kwargs)
        return model
        
    def freeze_parameters(self, mode=True):
        for name, param in self.named_parameters():
            if 'bias' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            
        
    def forward(self, x):
        x = self.model(x)
        return x