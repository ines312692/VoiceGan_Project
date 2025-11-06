"""
Discriminator: Distingue les vrais melspectrograms de B des spectrograms générés
"""
import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """Bloc de base du discriminateur"""

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminateur CNN pour évaluer le réalisme des melspectrograms

    Architecture PatchGAN-style:
        - Plusieurs couches CNN avec downsampling
        - Classifie des "patches" du spectrogram
        - Output: probabilité que le spectrogram soit réel

    Input: mel_spec (batch, n_mels, time)
    Output: validity (batch, 1, patches)
    """

    def __init__(self,
                 n_mels=80,
                 channels=[80, 128, 256, 512],
                 kernel_size=4,
                 stride=2):
        super().__init__()

        self.n_mels = n_mels

        # Build discriminator blocks
        layers = []

        # First layer (no normalization)
        layers.append(nn.Conv1d(n_mels, channels[0], kernel_size, stride, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Middle layers
        in_ch = channels[0]
        for out_ch in channels[1:]:
            layers.append(DiscriminatorBlock(in_ch, out_ch, kernel_size, stride))
            in_ch = out_ch

        # Output layer
        layers.append(nn.Conv1d(channels[-1], 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (batch, n_mels, time)

        Returns:
            validity: (batch, 1, patches) - Scores de validité pour chaque patch
        """
        validity = self.model(mel_spec)
        return validity


class MultiScaleDiscriminator(nn.Module):
    """
    Discriminateur multi-échelle
    Évalue le réalisme à différentes échelles temporelles
    """

    def __init__(self,
                 n_mels=80,
                 num_scales=3,
                 channels=[80, 128, 256, 512]):
        super().__init__()

        self.discriminators = nn.ModuleList([
            Discriminator(n_mels, channels) for _ in range(num_scales)
        ])

        # Downsampling pour les différentes échelles
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (batch, n_mels, time)

        Returns:
            outputs: Liste de scores de validité à différentes échelles
        """
        outputs = []
        x = mel_spec

        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)

        return outputs


class ProjectionDiscriminator(nn.Module):
    """
    Discriminateur avec projection conditionnelle
    Peut conditionner sur l'identité du locuteur cible
    """

    def __init__(self,
                 n_mels=80,
                 style_dim=128,
                 channels=[80, 128, 256, 512]):
        super().__init__()

        # Feature extraction
        layers = []
        layers.append(nn.Conv1d(n_mels, channels[0], 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2))

        in_ch = channels[0]
        for out_ch in channels[1:]:
            layers.append(DiscriminatorBlock(in_ch, out_ch, 4, 2))
            in_ch = out_ch

        self.features = nn.Sequential(*layers)

        # Projection du style
        self.style_proj = nn.Linear(style_dim, channels[-1])

        # Output layer
        self.output = nn.Conv1d(channels[-1], 1, 3, 1, 1)

    def forward(self, mel_spec, style=None):
        """
        Args:
            mel_spec: (batch, n_mels, time)
            style: (batch, style_dim) - Optionnel, pour conditioning

        Returns:
            validity: (batch, 1, patches)
        """
        # Extract features
        features = self.features(mel_spec)  # (batch, channels, time)

        # Conditional projection (si style fourni)
        if style is not None:
            style_emb = self.style_proj(style)  # (batch, channels)
            style_emb = style_emb.unsqueeze(2)  # (batch, channels, 1)

            # Inner product pour conditioning
            features = features + style_emb

        # Output
        validity = self.output(features)

        return validity


class SpectralNormDiscriminator(Discriminator):
    """
    Discriminateur avec Spectral Normalization
    Améliore la stabilité de l'entraînement GAN
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Appliquer spectral norm à toutes les convolutions
        self._apply_spectral_norm()

    def _apply_spectral_norm(self):
        """Applique spectral normalization aux couches conv"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.utils.spectral_norm(module)


if __name__ == "__main__":
    # Test des discriminateurs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4
    n_mels = 80
    time_steps = 200
    style_dim = 128

    mel_spec = torch.randn(batch_size, n_mels, time_steps).to(device)
    style = torch.randn(batch_size, style_dim).to(device)

    print("=" * 60)
    print("Testing Discriminators")
    print("=" * 60)

    # 1. Discriminateur simple
    print("\n1. Simple Discriminator")
    disc = Discriminator(n_mels=n_mels).to(device)
    validity = disc(mel_spec)
    print(f"Input shape: {mel_spec.shape}")
    print(f"Output shape: {validity.shape}")
    n_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # 2. Discriminateur multi-échelle
    print("\n2. Multi-Scale Discriminator")
    ms_disc = MultiScaleDiscriminator(n_mels=n_mels, num_scales=3).to(device)
    outputs = ms_disc(mel_spec)
    print(f"Number of scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Scale {i} output shape: {out.shape}")
    n_params = sum(p.numel() for p in ms_disc.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # 3. Discriminateur avec projection
    print("\n3. Projection Discriminator")
    proj_disc = ProjectionDiscriminator(n_mels=n_mels, style_dim=style_dim).to(device)
    validity = proj_disc(mel_spec, style)
    print(f"Output shape (with style): {validity.shape}")
    validity_uncond = proj_disc(mel_spec)
    print(f"Output shape (without style): {validity_uncond.shape}")
    n_params = sum(p.numel() for p in proj_disc.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # 4. Discriminateur avec Spectral Norm
    print("\n4. Spectral Norm Discriminator")
    sn_disc = SpectralNormDiscriminator(n_mels=n_mels).to(device)
    validity = sn_disc(mel_spec)
    print(f"Output shape: {validity.shape}")
    n_params = sum(p.numel() for p in sn_disc.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    print("\n" + "=" * 60)
    print("✓ All discriminator tests passed!")
    print("=" * 60)