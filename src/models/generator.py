"""
Generator: Fusionne contenu + style pour générer le melspectrogram A→B
"""
import torch
import torch.nn as nn
from .style_encoder import AdaptiveInstanceNorm


class ResidualBlock(nn.Module):
    """Bloc résiduel avec AdaIN pour injection du style"""

    def __init__(self, channels, style_dim, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.adain1 = AdaptiveInstanceNorm(style_dim, channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm(style_dim, channels)

    def forward(self, x, style):
        residual = x

        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.adain2(out, style)

        out = out + residual
        out = self.relu(out)

        return out


class UpsampleBlock(nn.Module):
    """Bloc d'upsampling avec AdaIN"""

    def __init__(self, in_channels, out_channels, style_dim, upsample_rate=2, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2

        self.upsample = nn.Upsample(scale_factor=upsample_rate, mode='nearest')
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.adain = AdaptiveInstanceNorm(style_dim, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.adain(x, style)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    """
    Generator: Fusionne contenu (de A) + style (de B)

    Architecture:
        1. Projection du contenu encodé
        2. Blocs résiduels avec injection de style via AdaIN
        3. Upsampling pour reconstruire la résolution temporelle
        4. Convolution finale pour générer le melspectrogram

    Input:
        - content: (batch, time, content_dim) depuis Content Encoder
        - style: (batch, style_dim) depuis Style Encoder

    Output:
        - mel_spec: (batch, n_mels, time) - Melspectrogram transformé
    """

    def __init__(self,
                 content_dim=256,
                 style_dim=128,
                 n_mels=80,
                 hidden_channels=[512, 256, 128],
                 n_residual_blocks=6,
                 upsample_rates=[2, 2, 2],
                 kernel_size=5):
        super().__init__()

        self.content_dim = content_dim
        self.style_dim = style_dim
        self.n_mels = n_mels

        # Input projection: content → channels
        self.input_proj = nn.Sequential(
            nn.Linear(content_dim, hidden_channels[0]),
            nn.ReLU()
        )

        # Residual blocks avec style injection
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels[0], style_dim, kernel_size)
            for _ in range(n_residual_blocks)
        ])

        # Upsampling blocks
        self.upsample_blocks = nn.ModuleList()
        in_ch = hidden_channels[0]
        for i, (out_ch, rate) in enumerate(zip(hidden_channels[1:], upsample_rates)):
            self.upsample_blocks.append(
                UpsampleBlock(in_ch, out_ch, style_dim, rate, kernel_size)
            )
            in_ch = out_ch

        # Output convolution pour générer le melspectrogram
        self.output_conv = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], hidden_channels[-1], kernel_size,
                      padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels[-1], n_mels, kernel_size,
                      padding=kernel_size // 2),
            nn.Tanh()  # Normaliser entre -1 et 1
        )

    def forward(self, content, style):
        """
        Args:
            content: (batch, time, content_dim) - Encodage du contenu de A
            style: (batch, style_dim) - Encodage du style de B

        Returns:
            mel_spec: (batch, n_mels, time) - Melspectrogram généré (A→B)
        """
        batch_size, time, _ = content.shape

        # Project content to hidden dimension
        x = self.input_proj(content)  # (batch, time, channels)

        # Transpose pour CNN: (batch, channels, time)
        x = x.transpose(1, 2)

        # Apply residual blocks with style injection
        for res_block in self.residual_blocks:
            x = res_block(x, style)

        # Upsample to reconstruct temporal resolution
        for upsample_block in self.upsample_blocks:
            x = upsample_block(x, style)

        # Generate final melspectrogram
        mel_spec = self.output_conv(x)  # (batch, n_mels, time)

        return mel_spec

    def inference(self, content, style):
        """Mode inférence sans gradient"""
        with torch.no_grad():
            return self.forward(content, style)


class MultiScaleGenerator(Generator):
    """
    Générateur multi-échelle pour améliorer la qualité
    Génère des melspectrograms à différentes résolutions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Outputs intermédiaires
        self.intermediate_outputs = nn.ModuleList([
            nn.Conv1d(ch, self.n_mels, 1)
            for ch in [512, 256, 128]
        ])

    def forward(self, content, style, return_intermediates=False):
        batch_size, time, _ = content.shape

        x = self.input_proj(content)
        x = x.transpose(1, 2)

        # Residual blocks
        for res_block in self.residual_blocks:
            x = res_block(x, style)

        intermediates = []

        # Upsample with intermediate outputs
        for i, upsample_block in enumerate(self.upsample_blocks):
            x = upsample_block(x, style)
            if return_intermediates and i < len(self.intermediate_outputs):
                inter = self.intermediate_outputs[i](x)
                intermediates.append(inter)

        # Final output
        mel_spec = self.output_conv(x)

        if return_intermediates:
            return mel_spec, intermediates
        return mel_spec


if __name__ == "__main__":
    # Test du Generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paramètres
    batch_size = 4
    time_steps = 200
    content_dim = 256
    style_dim = 128
    n_mels = 80

    # Créer le modèle
    generator = Generator(
        content_dim=content_dim,
        style_dim=style_dim,
        n_mels=n_mels,
        hidden_channels=[512, 256, 128],
        n_residual_blocks=6,
        upsample_rates=[2, 2, 2]
    ).to(device)

    # Créer des inputs de test
    content = torch.randn(batch_size, time_steps, content_dim).to(device)
    style = torch.randn(batch_size, style_dim).to(device)

    print(f"Content shape: {content.shape}")
    print(f"Style shape: {style.shape}")

    # Forward pass
    mel_spec = generator(content, style)
    print(f"Generated melspectrogram shape: {mel_spec.shape}")
    print(f"Expected shape: (batch={batch_size}, n_mels={n_mels}, time={time_steps * 8})")

    # Test multi-scale generator
    ms_generator = MultiScaleGenerator(
        content_dim=content_dim,
        style_dim=style_dim,
        n_mels=n_mels
    ).to(device)

    mel_spec, intermediates = ms_generator(content, style, return_intermediates=True)
    print(f"\nMulti-scale output: {mel_spec.shape}")
    print(f"Number of intermediate outputs: {len(intermediates)}")

    # Nombre de paramètres
    n_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"\nNumber of parameters: {n_params:,}")

    print("\n✓ Generator test passed!")