"""
Vocoder implementation
For production, consider using pretrained HiFi-GAN or MelGAN
This is a simplified implementation for educational purposes
"""

import torch
import torch.nn as nn
from typing import List


class ResBlock(nn.Module):
    """Residual block for vocoder"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: tuple = (1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=d, padding=self._get_padding(kernel_size, d))
            for d in dilation
        ])

        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                      dilation=1, padding=self._get_padding(kernel_size, 1))
            for _ in dilation
        ])

        self.activations = nn.ModuleList([nn.LeakyReLU(0.1) for _ in range(len(dilation) * 2)])

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2,
                                  self.activations[::2], self.activations[1::2]):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    @staticmethod
    def _get_padding(kernel_size: int, dilation: int = 1):
        return int((kernel_size * dilation - dilation) / 2)


class HiFiGANVocoder(nn.Module):
    """
    Simplified HiFi-GAN vocoder
    Converts mel-spectrogram to waveform

    Note: For best results, use a pretrained vocoder from:
    - https://github.com/jik876/hifi-gan
    - https://github.com/descriptinc/melgan-neurips
    """

    def __init__(
            self,
            n_mels: int = 80,
            upsample_rates: List[int] = [8, 8, 2, 2],
            upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
            upsample_initial_channel: int = 512,
            resblock_kernel_sizes: List[int] = [3, 7, 11],
            resblock_dilation_sizes: List[tuple] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # Initial convolution
        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3)

        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                )
            )

        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        # Post convolution
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3)

        # Activation
        self.activation = nn.LeakyReLU(0.1)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel-spectrogram (batch, n_mels, time)

        Returns:
            audio: Waveform (batch, 1, samples)
        """
        x = self.conv_pre(mel)

        for i, upsample in enumerate(self.ups):
            x = self.activation(x)
            x = upsample(x)

            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs += self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = self.activation(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def _init_weights(self, m):
        """Initialize weights"""
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class MelGANVocoder(nn.Module):
    """
    Simplified MelGAN vocoder (alternative to HiFi-GAN)
    Generally faster but slightly lower quality
    """

    def __init__(
            self,
            n_mels: int = 80,
            ngf: int = 32,
            n_residual_layers: int = 3
    ):
        super().__init__()

        ratios = [8, 8, 2, 2]  # Upsampling ratios

        # Initial layer
        mult = int(2 ** len(ratios))
        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(n_mels, mult * ngf, kernel_size=7, padding=0)
        ]

        # Upsampling layers
        for i, r in enumerate(ratios):
            mult = int(2 ** (len(ratios) - i))

            model += [
                nn.LeakyReLU(0.2),
                nn.ConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2
                )
            ]

            # Residual blocks
            for j in range(n_residual_layers):
                model += [ResBlock(mult * ngf // 2)]

        # Output layer
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.Conv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: Mel-spectrogram (batch, n_mels, time)

        Returns:
            audio: Waveform (batch, 1, samples)
        """
        return self.model(mel)


def load_pretrained_vocoder(
        vocoder_type: str = 'hifigan',
        checkpoint_path: str = None,
        device: str = 'cuda'
) -> nn.Module:
    """
    Load a pretrained vocoder

    Args:
        vocoder_type: 'hifigan' or 'melgan'
        checkpoint_path: Path to vocoder checkpoint
        device: Device to load model on

    Returns:
        Loaded vocoder model
    """
    if vocoder_type == 'hifigan':
        vocoder = HiFiGANVocoder()
    elif vocoder_type == 'melgan':
        vocoder = MelGANVocoder()
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vocoder.load_state_dict(checkpoint['model_state_dict'])

    vocoder.to(device)
    vocoder.eval()

    return vocoder