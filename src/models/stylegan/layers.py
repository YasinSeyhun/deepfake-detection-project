import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class EqualLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        bias_init: float = 0,
        lr_mul: float = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None
        
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
    
    def forward(self, x):
        if self.activation == "fused_lrelu":
            out = F.linear(x, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2 * self.lr_mul)
        else:
            out = F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul)
        
        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        blur_kernel: List[int] = [1, 3, 3, 1],
    ):
        super().__init__()
        
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )
        
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)
    
    def forward(self, x, style, noise=None):
        out = self.conv(x, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        
        return out

class ToRGB(nn.Module):
    def __init__(
        self,
        in_channel: int,
        style_dim: int,
        upsample: bool = True,
        blur_kernel: List[int] = [1, 3, 3, 1],
    ):
        super().__init__()
        
        if upsample:
            self.upsample = Upsample(blur_kernel)
        
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
    
    def forward(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias
        
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        
        return out

class ConstantInput(nn.Module):
    def __init__(self, channel: int, size: int = 4):
        super().__init__()
        
        self.input = nn.Parameter(torch.randn(1, channel, size, size))
    
    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        
        return out 