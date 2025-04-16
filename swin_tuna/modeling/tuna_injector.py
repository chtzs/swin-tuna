import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple

from .tuna import Tuna
from .swin_sam import SwinTransformerSemanticSAM, SwinTransformerBlock
from mmseg.models.backbones.swin import SwinTransformer, SwinBlockSequence, SwinBlock
        

def inject_attributes(target: nn.Module, conv_size, dim, hidden_dim, device=None):
    print("Injected SAM")
    setattr(target, 'tuna_1', Tuna(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
    setattr(target, 'tuna_2', Tuna(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
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
                x_tuna = self.tuna_1(x, hw_shape)
                # original modules
                x = self.norm1(x)
                x = self.attn(x, hw_shape)
                # inject back
                raw = x + identity
                x = self.tuna_x_scale1 * raw + self.tuna_scale1 * x_tuna
                
                # original modules
                identity = x
                x = self.norm2(x)
                x = self.ffn(x, identity=identity)
                # tuna module
                x_tuna = self.tuna_2(x, hw_shape)
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

    @staticmethod
    def inject_sam_swin(target_model: nn.Module):
        assert isinstance(target_model, SwinTransformerSemanticSAM), "Wrong target model"
        device = next(target_model.parameters()).device
        conv_size_list = [5, 5, 5, 3]# [3, 5, 7, 9]
        hidden_dim_list = [64, 64, 96, 192]
        def forward(self, x, mask_matrix):
            """Forward function.
            Args:
                x: Input feature, tensor size (B, H*W, C).
                H, W: Spatial resolution of the input feature.
                mask_matrix: Attention mask for cyclic shift.
            """
            B, L, C = x.shape
            H, W = self.H, self.W
            hw_shape = (H, W)
            assert L == H * W, "input feature has wrong size"

            # HACK model will not upsampling
            # if min([H, W]) <= self.window_size:
                # if window size is larger than input resolution, we don't partition windows
                # self.shift_size = 0
                # self.window_size = min([H,W])

            shortcut = x
            # tuna module
            x_tuna = self.tuna_1(x, hw_shape)
            x = self.norm1(x)
            x = self.w_msa(x, mask_matrix)
            # inject back
            raw =  self.drop_path(x) + shortcut
            x = self.tuna_x_scale1 * raw + self.tuna_scale1 * x_tuna
            
            # FFN
            x = x + self.drop_path(self.mlp(self.norm2(x)))
             # tuna module
            x_tuna = self.tuna_2(x, hw_shape)
            # inject back
            x = self.tuna_x_scale2 * x + self.tuna_scale2 * x_tuna
            return x
        
        for i, layer in enumerate(target_model.layers):
            for target in layer.blocks:
                dim = target.dim
                inject_attributes(target, conv_size=conv_size_list[i], dim=dim, hidden_dim=hidden_dim_list[i], device=device)
    
        SwinTransformerBlock.forward = forward
    

    @staticmethod
    def inject_mona(target_model: nn.Module):
        from mmseg.models.backbones.swin import SwinBlock
        from .mona import Mona
        device = next(target_model.parameters()).device
        
        def forward(self, x, hw_shape):
            def _inner_forward(x):
                identity = x
                x = self.norm1(x)
                x = self.attn(x, hw_shape)

                x = x + identity
                x = self.my_module_1(x, hw_shape)
                
                identity = x
                x = self.norm2(x)
                x = self.ffn(x, identity=identity)

                x = self.my_module_2(x, hw_shape)
                return x

            x = _inner_forward(x)
            return x
        
        def search_and_modify(model):
            for name, module in model.named_children():
                if isinstance(module, SwinBlock):
                    print("Injected mona")
                    target = getattr(model, name)
                    dim = target.ffn.embed_dims
                    setattr(target, 'my_module_1', Mona(dim, 8).to(device=device))
                    setattr(target, 'my_module_2', Mona(dim, 8).to(device=device))
                else:
                    search_and_modify(module)
        
        # Search for SwinBlock
        SwinBlock.forward = forward
        search_and_modify(target_model)

