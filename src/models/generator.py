import torch
import torch.nn as nn
from typing import List

class AdaptiveInstanceNorm1d(nn.Module):
    """Adaptive Instance Normalization for style injection"""
    
    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
            style: (batch, style_dim)
        """
        # Normalize
        x = self.norm(x)
        
        # Compute style parameters
        style_params = self.fc(style).unsqueeze(2)
        gamma, beta = style_params.chunk(2, dim=1)
        
        # Apply affine transformation
        return gamma * x + beta

class ResidualBlock(nn.Module):
    """Residual block with AdaIN"""
    
    def __init__(self, channels: int, style_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.adain1 = AdaptiveInstanceNorm1d(channels, style_dim)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm1d(channels, style_dim)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.adain2(out, style)
        
        return out + residual

class Generator(nn.Module):
    """
    Generator: Fuses content and style to generate target mel-spectrogram
    """
    
    def __init__(
        self,
        content_dim: int = 512,
        style_dim: int = 256,
        channels: List[int] = [512, 256, 128, 64],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        upsample_rates: List[int] = [2, 2, 2, 1],
        output_channels: int = 80,
        num_residual_blocks: int = 4
    ):
        super().__init__()
        
        input_dim = content_dim + style_dim
        
        # Initial projection
        self.input_conv = nn.Conv1d(content_dim, channels[0], 1)
        
        # Residual blocks with style injection
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels[0], style_dim)
            for _ in range(num_residual_blocks)
        ])
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        
        for i, (out_ch, k, rate) in enumerate(zip(channels, kernel_sizes, upsample_rates)):
            in_ch = channels[i-1] if i > 0 else channels[0]
            
            if rate > 1:
                # Upsample
                layer = nn.Sequential(
                    nn.Upsample(scale_factor=rate, mode='nearest'),
                    nn.Conv1d(in_ch, out_ch, k, padding=(k-1)//2),
                    nn.InstanceNorm1d(out_ch),
                    nn.ReLU()
                )
            else:
                # No upsampling
                layer = nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, k, padding=(k-1)//2),
                    nn.InstanceNorm1d(out_ch),
                    nn.ReLU()
                )
            
            self.upsample_layers.append(layer)
        
        # Output layer
        self.output_conv = nn.Conv1d(channels[-1], output_channels, 1)
    
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content: (batch, content_dim, time)
            style: (batch, style_dim)
        Returns:
            mel: (batch, output_channels, time')
        """
        # Initial projection
        x = self.input_conv(content)
        
        # Apply residual blocks with style
        for res_block in self.residual_blocks:
            x = res_block(x, style)
        
        # Upsample
        for upsample in self.upsample_layers:
            x = upsample(x)
        
        # Output projection
        mel = self.output_conv(x)
        
        return mel