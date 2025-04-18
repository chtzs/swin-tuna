import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.backbones.swin import SwinTransformer, SwinBlockSequence, SwinBlock
from .tuna import Tuna
        

def inject_attributes(target: nn.Module, conv_size, dim, hidden_dim, device=None, clazz=Tuna):
    print("Injected")
    setattr(target, 'tuna_1', clazz(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
    setattr(target, 'tuna_2', clazz(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
    setattr(target, 'tuna_scale1', nn.Parameter(torch.ones(dim) * 1e-6, requires_grad=True).to(device=device))
    setattr(target, 'tuna_scale2', nn.Parameter(torch.ones(dim) * 1e-6, requires_grad=True).to(device=device))
    setattr(target, 'tuna_x_scale1', nn.Parameter(torch.ones(dim), requires_grad=True).to(device=device))
    setattr(target, 'tuna_x_scale2', nn.Parameter(torch.ones(dim), requires_grad=True).to(device=device))

class TunaInjector:
    @staticmethod
    def inject_tuna(target_model: nn.Module):
        assert isinstance(target_model, SwinTransformer), "Wrong target model"
        device = next(target_model.parameters()).device
        conv_size_list = [5, 5, 5, 3]# [3, 5, 7, 9]
        hidden_dim_list = [64, 64, 96, 192]
        def forward(self, x, hw_shape):
            def _inner_forward(x):
                identity = x
                # tuna module
                x_tuna = self.tuna_1(identity, hw_shape)
                # original modules
                x = self.norm1(x)
                x = self.attn(x, hw_shape)
                # inject back
                raw = x + identity
                x = self.tuna_x_scale1 * raw + self.tuna_scale1 * x_tuna
                
                # original modules
                identity = x
                # tuna module
                x_tuna = self.tuna_2(identity, hw_shape)
                x = self.norm2(x)
                x = self.ffn(x, identity=identity)
                # inject back
                x = self.tuna_x_scale2 * x + self.tuna_scale2 * x_tuna
                
                return x

            x = _inner_forward(x)
            return x
        
        for i, sequence_block in enumerate(target_model.stages):
            sequence_block: SwinBlockSequence
            for target in sequence_block.blocks:
                target: SwinBlock
                dim = target.ffn.embed_dims
                inject_attributes(target, conv_size=conv_size_list[i], dim=dim, hidden_dim=hidden_dim_list[i], device=device)
    
        SwinBlock.forward = forward