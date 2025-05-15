import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = latent_dim if i == 0 else hidden_dim
            out_dim = latent_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                self.layers.append(nn.LeakyReLU(0.2))

    def forward(self, z):
        x = z
        for layer in self.layers:
            x = layer(x)
        return x

class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.style1 = nn.Linear(style_dim, out_channels)
        self.style2 = nn.Linear(style_dim, out_channels)
        self.noise1 = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.noise2 = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        x = self.conv1(x)
        x = x + self.noise1 * torch.randn_like(x)
        x = self.activation(x)
        x = x * self.style1(style).view(-1, -1, 1, 1)
        
        x = self.conv2(x)
        x = x + self.noise2 * torch.randn_like(x)
        x = self.activation(x)
        x = x * self.style2(style).view(-1, -1, 1, 1)
        
        return x

class StyleGAN(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=512, num_layers=8, img_size=256):
        super().__init__()
        self.mapping = MappingNetwork(latent_dim, hidden_dim, num_layers)
        
        # Initial constant
        self.initial = nn.Parameter(torch.randn(1, hidden_dim, 4, 4))
        
        # Synthesis blocks
        self.blocks = nn.ModuleList()
        current_size = 4
        while current_size < img_size:
            in_channels = hidden_dim if current_size == 4 else hidden_dim * 2
            out_channels = hidden_dim * 2 if current_size * 2 < img_size else 3
            self.blocks.append(SynthesisBlock(in_channels, out_channels, latent_dim))
            current_size *= 2
        
        # To RGB layers
        self.to_rgb = nn.ModuleList()
        for _ in range(len(self.blocks)):
            self.to_rgb.append(nn.Conv2d(hidden_dim * 2, 3, 1))

    def forward(self, z):
        # Map latent to style
        style = self.mapping(z)
        
        # Initial synthesis
        x = self.initial.repeat(z.shape[0], 1, 1, 1)
        
        # Progressive synthesis
        for i, block in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x, style)
            if i == len(self.blocks) - 1:
                x = self.to_rgb[i](x)
        
        return torch.tanh(x) 