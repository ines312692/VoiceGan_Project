import torch
import torch.nn as nn
from typing import List, Tuple

class Discriminator(nn.Module):
    """
    Multi-scale Discriminator
    Distinguishes between real and generated mel-spectrograms
    """
    
    def __init__(
        self,
        input_channels: int = 80,
        channels: List[int] = [64, 128, 256, 512, 1024],
        kernel_sizes: List[int] = [4, 4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2, 1],
        num_scales: int = 3
    ):
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            SingleDiscriminator(input_channels, channels, kernel_sizes, strides)
            for _ in range(num_scales)
        ])
        
        # Downsample layers for multi-scale
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            outputs: List of discriminator outputs
            features: List of intermediate features for feature matching loss
        """
        outputs = []
        features_list = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            
            out, feats = disc(x)
            outputs.append(out)
            features_list.append(feats)
        
        return outputs, features_list

class SingleDiscriminator(nn.Module):
    """Single scale discriminator"""
    
    def __init__(
        self,
        input_channels: int,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int]
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_ch = input_channels
        
        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            padding = (k - 1) // 2
            self.layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, s, padding),
                nn.LeakyReLU(0.2)
            ))
            in_ch = out_ch
        
        # Final output layer
        self.output_layer = nn.Conv1d(channels[-1], 1, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            output: Discriminator output
            features: Intermediate features
        """
        features = []
        
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        output = self.output_layer(x)
        
        return output, features