import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, Tuple

from ..utils.model_utils import c2_xavier_fill

_size_2_t = Union[int, Tuple[int, int]]

# 49.63
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        c2_xavier_fill(self.conv2d)
        
    def forward(self, x: Tensor, hw_shapes: tuple) -> Tensor:
        # B x HW x C
        b, hw, c = x.shape
        # B x HW x C => B x C x H x W
        x = x.permute(0, 2, 1).view(b, c, *hw_shapes)
        x = self.conv2d(x)
        # B x C x H x W => B x HW x C
        x = x.view(b, c, hw).permute(0, 2, 1)
        return x

class Tuna(nn.Module):
    def __init__(self, in_features, hidden_dim=64, conv_size=3):
        super().__init__()
        # Follow ConvNeXt
        self.hidden_dim = hidden_dim
        # depth-wise convolution
        self.conv2d = Conv2d(hidden_dim, hidden_dim, kernel_size=conv_size, padding=conv_size // 2, groups=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        
        # point-wise convolution
        self.pwconv1 = nn.Linear(in_features, hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, in_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        
        # init
        c2_xavier_fill(self.pwconv1)
        c2_xavier_fill(self.pwconv2)
        
    def conv(self, x: Tensor, hw_shapes: tuple) -> Tensor:
        identity = x
        
        x = self.norm(x)
        x = self.conv2d(x, hw_shapes)
        
        x = x + identity
        return x
        
    def forward(self, x: Tensor, hw_shapes: tuple):
        assert len(x.shape) == 3, 'Wrong shape!'
        
        identity = x
        # Downsample
        x = self.pwconv1(x)
        # Conv
        x = self.conv(x, hw_shapes)
        # Upsample
        x = self.pwconv2(x)
        
        x = self.act(x)
        x = self.dropout(x)
        x = x + identity
        
        return x
