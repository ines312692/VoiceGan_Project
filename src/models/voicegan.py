import torch
import torch.nn as nn
from typing import Dict, Tuple

from .content_encoder import ContentEncoder
from .style_encoder import StyleEncoder
from .generator import Generator
from .discriminator import Discriminator

class VoiceGAN(nn.Module):
    """
    Complete VoiceGAN model for voice conversion A→B
    """

    def __init__(
        self,
        # Audio config
        n_mels: int = 80,

        # Content encoder config
        content_channels: list = [64, 128, 256, 512],
        content_kernel_sizes: list = [3, 3, 3, 3],
        content_strides: list = [1, 2, 2, 1],
        transformer_dim: int = 512,
        num_heads: int = 8,
        num_transformer_layers: int = 4,

        # Style encoder config
        style_channels: list = [64, 128, 256, 512],
        style_kernel_sizes: list = [3, 3, 3, 3],
        style_strides: list = [2, 2, 2, 2],
        style_dim: int = 256,

        # Generator config
        generator_channels: list = [512, 256, 128, 64],
        generator_kernel_sizes: list = [3, 3, 3, 3],
        upsample_rates: list = [2, 2, 2, 1],

        # Discriminator config
        discriminator_channels: list = [64, 128, 256, 512, 1024],
        discriminator_kernel_sizes: list = [4, 4, 4, 4, 4],
        discriminator_strides: list = [2, 2, 2, 2, 1],
        num_discriminator_scales: int = 3
    ):
        super().__init__()

        # Encoders
        self.content_encoder = ContentEncoder(
            input_channels=n_mels,
            channels=content_channels,
            kernel_sizes=content_kernel_sizes,
            strides=content_strides,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers
        )

        self.style_encoder = StyleEncoder(
            input_channels=n_mels,
            channels=style_channels,
            kernel_sizes=style_kernel_sizes,
            strides=style_strides,
            style_dim=style_dim
        )

        # Generator
        self.generator = Generator(
            content_dim=transformer_dim,
            style_dim=style_dim,
            channels=generator_channels,
            kernel_sizes=generator_kernel_sizes,
            upsample_rates=upsample_rates,
            output_channels=n_mels
        )

        # Discriminator
        self.discriminator = Discriminator(
            input_channels=n_mels,
            channels=discriminator_channels,
            kernel_sizes=discriminator_kernel_sizes,
            strides=discriminator_strides,
            num_scales=num_discriminator_scales
        )

    def forward(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training

        Args:
            source_mel: Mel-spectrogram of source speaker A (batch, n_mels, time)
            target_mel: Mel-spectrogram of target speaker B (batch, n_mels, time)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary containing:
                - generated_mel: Generated mel-spectrogram
                - content_features: Content encoding
                - style_features: Style encoding
        """
        # Encode content from source
        content = self.content_encoder(source_mel)

        # Encode style from target
        style = self.style_encoder(target_mel)

        # Generate mel-spectrogram with source content and target style
        generated_mel = self.generator(content, style)

        results = {
            'generated_mel': generated_mel,
            'content': content,
            'style': style
        }

        return results

    def convert(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert voice from source to target

        Args:
            source_mel: Source mel-spectrogram (batch, n_mels, time)
            target_mel: Target reference mel-spectrogram (batch, n_mels, time)

        Returns:
            Converted mel-spectrogram (batch, n_mels, time)
        """
        with torch.no_grad():
            content = self.content_encoder(source_mel)
            style = self.style_encoder(target_mel)
            converted_mel = self.generator(content, style)

        return converted_mel

    def encode_content(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract content features"""
        with torch.no_grad():
            return self.content_encoder(mel)

    def encode_style(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract style features"""
        with torch.no_grad():
            return self.style_encoder(mel)

    def discriminate(
        self,
        mel: torch.Tensor
    ) -> Tuple[list, list]:
        """
        Run discriminator

        Returns:
            outputs: List of discriminator outputs
            features: List of intermediate features
        """
        return self.discriminator(mel)