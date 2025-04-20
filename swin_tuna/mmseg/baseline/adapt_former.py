import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from mmseg.models.backbones.swin import SwinTransformer, SwinBlock, SwinBlockSequence
from ..PEFT import PEFT

class Adapter(nn.Module):
    def __init__(self,
                 d_model,
                 bottleneck=64,
                 dropout=0.0,
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output
   
@MODELS.register_module() 
class SwinTransformerAdapter(PEFT):
    def __init__(self, **kwargs):
        super().__init__()
        # Inject module into SwinTransformer
        self.model = self.make_model(kwargs)
        self.adapt_model()
        # Freeze parameters
        self.freeze_parameters()
        
    def make_model(self, kwargs)-> SwinTransformer:
        kwargs['type'] = 'SwinTransformer'
        model = MODELS.build(kwargs)
        return model
        
    def adapt_model(self):
        def forward(self, x, hw_shape):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            # Parellel
            adapt_x = self.adaptmlp(x, add_residual=False)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            x = x + adapt_x
            
            return x
        
        for sequence_block in self.model.stages:
            sequence_block: SwinBlockSequence
            for target in sequence_block.blocks:
                target: SwinBlock
                dim = target.ffn.embed_dims
                setattr(target, 'adaptmlp', Adapter(d_model=dim).cuda())
    
        SwinBlock.forward = forward
        
    def freeze_parameters(self, mode=True):
        # Original dropout helps model
        # self.model.eval()
        for name, param in self.model.named_parameters():
            if 'adaptmlp' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            
    def forward(self, x):
        x = self.model(x)
        return x