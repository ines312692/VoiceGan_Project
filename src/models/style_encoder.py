import torch
import torch.nn as nn
from typing import List

class StyleEncoder(nn.Module):
    """
    Style Encoder: CNN-based
    Extracts speaker identity and vocal characteristics from reference audio
    """
    
    def __init__(
        self,
        input_channels: int = 80,
        channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3],
        strides: List[int] = [2, 2, 2, 2],
        style_dim: int = 256
    ):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        in_ch = input_channels
        
        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            padding = (k - 1) // 2
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, s, padding),
                nn.InstanceNorm1d(out_ch),
                nn.LeakyReLU(0.2)
            ))
            in_ch = out_ch
        
        # Global pooling and projection to style vector
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        self.style_dim = style_dim
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time)
        Returns:
            style: (batch, style_dim)
        """
        x = mel
        
        # CNN encoding
        for conv in self.conv_layers:
            x = conv(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Project to style vector
        style = self.fc(x)
        
        return style