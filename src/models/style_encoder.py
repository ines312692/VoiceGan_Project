"""
Style Encoder: CNN
Capture les caractéristiques vocales (timbre, intonation) de la voix cible B
"""
import torch
import torch.nn as nn


class StyleCNNBlock(nn.Module):
    """Bloc CNN pour l'extraction de features de style"""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.InstanceNorm1d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class GlobalStylePooling(nn.Module):
    """Pooling global pour agréger les features de style"""

    def __init__(self, pooling_type='adaptive'):
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
        Returns:
            pooled: (batch, channels)
        """
        if self.pooling_type == 'adaptive':
            # Adaptive average pooling
            pooled = torch.mean(x, dim=2)
        elif self.pooling_type == 'max':
            pooled = torch.max(x, dim=2)[0]
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attention = torch.softmax(x, dim=2)
            pooled = torch.sum(x * attention, dim=2)
        else:
            pooled = torch.mean(x, dim=2)

        return pooled


class StyleEncoder(nn.Module):
    """
    Style Encoder avec CNN

    Architecture:
        1. Plusieurs couches CNN pour extraire les caractéristiques vocales
        2. Downsampling progressif avec stride=2
        3. Global pooling pour obtenir un vecteur de style fixe
        4. FC layers pour projection finale

    Le style encode: timbre, pitch moyen, vitesse de parole, qualité vocale
    """

    def __init__(self,
                 n_mels=80,
                 style_dim=128,
                 cnn_channels=[80, 128, 256, 512],
                 kernel_size=5,
                 pooling_type='adaptive'):
        super().__init__()

        self.n_mels = n_mels
        self.style_dim = style_dim

        # CNN Layers avec downsampling
        cnn_layers = []
        in_ch = n_mels
        for out_ch in cnn_channels:
            cnn_layers.append(
                StyleCNNBlock(in_ch, out_ch, kernel_size, stride=2)
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # Global pooling
        self.global_pool = GlobalStylePooling(pooling_type)

        # FC layers pour obtenir le vecteur de style final
        self.fc = nn.Sequential(
            nn.Linear(cnn_channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, style_dim)
        )

        # Normalisation du vecteur de style
        self.style_norm = nn.LayerNorm(style_dim)

    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (batch, n_mels, time) - Melspectrogram de référence (speaker B)

        Returns:
            style: (batch, style_dim) - Vecteur de style global
        """
        # Extract style features through CNN
        x = self.cnn(mel_spec)  # (batch, channels, time_downsampled)

        # Global pooling to get fixed-size representation
        x = self.global_pool(x)  # (batch, channels)

        # Project to style dimension
        style = self.fc(x)  # (batch, style_dim)

        # Normalize
        style = self.style_norm(style)

        return style

    def get_multi_scale_style(self, mel_spec):
        """
        Extraction de style multi-échelle (pour des applications avancées)
        Retourne les features à différentes échelles
        """
        styles = []
        x = mel_spec

        for layer in self.cnn:
            x = layer(x)
            # Pooling à chaque échelle
            pooled = torch.mean(x, dim=2)
            styles.append(pooled)

        return styles


class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN)
    Utilisé pour injecter le style dans le générateur
    """

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)

        # Apprendre les paramètres affines depuis le style
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, style):
        """
        Args:
            x: (batch, channels, time) - Features à normaliser
            style: (batch, style_dim) - Vecteur de style

        Returns:
            normalized: Features avec le style appliqué
        """
        # Instance normalization
        x = self.norm(x)

        # Compute affine parameters from style
        style_params = self.fc(style)  # (batch, channels * 2)
        gamma, beta = style_params.chunk(2, dim=1)  # Each (batch, channels)

        # Apply affine transformation
        gamma = gamma.unsqueeze(2)  # (batch, channels, 1)
        beta = beta.unsqueeze(2)  # (batch, channels, 1)

        out = gamma * x + beta

        return out


if __name__ == "__main__":
    # Test du Style Encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Créer le modèle
    encoder = StyleEncoder(
        n_mels=80,
        style_dim=128,
        cnn_channels=[80, 128, 256, 512],
        kernel_size=5
    ).to(device)

    # Test avec un batch
    batch_size = 4
    n_mels = 80
    time_steps = 200

    mel_spec = torch.randn(batch_size, n_mels, time_steps).to(device)

    print(f"Input shape: {mel_spec.shape}")
    style = encoder(mel_spec)
    print(f"Style vector shape: {style.shape}")

    # Test AdaIN
    adain = AdaptiveInstanceNorm(style_dim=128, num_features=256).to(device)
    features = torch.randn(batch_size, 256, 100).to(device)
    normalized = adain(features, style)
    print(f"AdaIN output shape: {normalized.shape}")

    # Afficher le nombre de paramètres
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params:,}")

    print("\n✓ Style Encoder test passed!")