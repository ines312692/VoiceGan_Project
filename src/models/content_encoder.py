"""
Content Encoder: CNN + Transformer
Capture le contenu linguistique (ce qui est dit) de la voix source A
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Encodage positionnel pour le Transformer"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNBlock(nn.Module):
    """Bloc CNN pour l'extraction de features locales"""

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ContentEncoder(nn.Module):
    """
    Content Encoder avec CNN (local features) + Transformer (global context)

    Architecture:
        1. CNN Layers: Extraction de features locales depuis le melspectrogram
        2. Transformer Layers: Capture des dépendances temporelles longues
        3. Output: Représentation du contenu linguistique
    """

    def __init__(self,
                 n_mels=80,
                 content_dim=256,
                 cnn_channels=[80, 128, 256],
                 transformer_layers=4,
                 transformer_heads=8,
                 dropout=0.1):
        super().__init__()

        self.n_mels = n_mels
        self.content_dim = content_dim

        # CNN Layers pour features locales
        cnn_layers = []
        in_ch = n_mels
        for out_ch in cnn_channels:
            cnn_layers.append(CNNBlock(in_ch, out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Projection vers la dimension du transformer
        self.proj = nn.Linear(cnn_channels[-1], content_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(content_dim, dropout=dropout)

        # Transformer Encoder pour contexte global
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=content_dim,
            nhead=transformer_heads,
            dim_feedforward=content_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        # Layer finale
        self.output_layer = nn.Sequential(
            nn.Linear(content_dim, content_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(content_dim, content_dim)
        )

    def forward(self, mel_spec):
        """
        Args:
            mel_spec: (batch, n_mels, time) - Melspectrogram d'entrée

        Returns:
            content: (batch, time, content_dim) - Encodage du contenu
        """
        # CNN: Extract local features
        # Input: (batch, n_mels, time)
        x = self.cnn(mel_spec)  # (batch, channels, time)

        # Transpose pour le transformer
        x = x.transpose(1, 2)  # (batch, time, channels)

        # Projection vers content_dim
        x = self.proj(x)  # (batch, time, content_dim)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer: Capture global context
        content = self.transformer(x)  # (batch, time, content_dim)

        # Output layer
        content = self.output_layer(content)

        return content

    def get_content_features(self, mel_spec):
        """Extraction des features de contenu pour l'analyse"""
        with torch.no_grad():
            return self.forward(mel_spec)


if __name__ == "__main__":
    # Test du Content Encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Créer le modèle
    encoder = ContentEncoder(
        n_mels=80,
        content_dim=256,
        cnn_channels=[80, 128, 256],
        transformer_layers=4,
        transformer_heads=8
    ).to(device)

    # Test avec un batch
    batch_size = 4
    n_mels = 80
    time_steps = 200

    mel_spec = torch.randn(batch_size, n_mels, time_steps).to(device)

    print(f"Input shape: {mel_spec.shape}")
    content = encoder(mel_spec)
    print(f"Output shape: {content.shape}")

    # Afficher le nombre de paramètres
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params:,}")

    print("\n✓ Content Encoder test passed!")