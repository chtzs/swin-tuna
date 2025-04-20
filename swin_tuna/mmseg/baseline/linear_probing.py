import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.backbones.swin import SwinTransformer

from ..PEFT import PEFT

@MODELS.register_module() 
class SwinTransformerFixed(PEFT):
    def __init__(self, **kwargs):
        super().__init__()
        # Inject module into SwinTransformer
        self.model = self.make_model(kwargs)
        # Freeze parameters
        self.freeze_parameters()
        
    def make_model(self, kwargs)-> SwinTransformer:
        kwargs['type'] = 'SwinTransformer'
        model = MODELS.build(kwargs)
        return model
        
    def freeze_parameters(self, mode=True):
        self.model.eval()
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.model(x)
        return x