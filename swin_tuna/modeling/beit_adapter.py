import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple

from .tuna import Tuna
from .beit import BEiT, Block

def inject_attributes(target: nn.Module, conv_size, dim, hidden_dim, device=None):
    print("Injected")
    setattr(target, 'tuna_1', Tuna(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
    setattr(target, 'tuna_2', Tuna(dim, hidden_dim=hidden_dim, conv_size=conv_size).to(device=device))
    setattr(target, 'tuna_scale1', nn.Parameter(torch.ones(dim) * 1e-6, requires_grad=True).to(device=device))
    setattr(target, 'tuna_scale2', nn.Parameter(torch.ones(dim) * 1e-6, requires_grad=True).to(device=device))
    setattr(target, 'tuna_x_scale1', nn.Parameter(torch.ones(dim), requires_grad=True).to(device=device))
    setattr(target, 'tuna_x_scale2', nn.Parameter(torch.ones(dim), requires_grad=True).to(device=device))


class BEiTAdapter:
    @staticmethod
    def make_beit(**kwargs) -> BEiT:
        beit = BEiT(kwargs)
        beit.init_weights("/data/diskb/quyang/haotian/beit_large_patch16_224_pt22k_ft22k.pth")

        def forward(self, x, rel_pos_bias=None):
            # x: B x (HW + 1) x C
            # 第一个是cls
            _, size, _ = x.shape
            hw = int((size - 1) ** 0.5)
            hw_shape = (hw, hw)
            if self.gamma_1 is None:
                raise Exception()
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                identity = x
                # tuna module
                x_tuna = self.tuna_1(x, hw_shape)
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x
        
        Block.forward = forward
        return beit