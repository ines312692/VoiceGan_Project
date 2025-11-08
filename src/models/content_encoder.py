import torch
import torch.nn as nn
import math
from typing import List, Optional


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            activation: str = "relu"
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_linear(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int = 2048,
            dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class ContentEncoder(nn.Module):
    """
    Content Encoder: CNN + Transformer
    Extracts linguistic content from source audio
    """

    def __init__(
            self,
            input_channels: int = 80,
            channels: List[int] = [64, 128, 256, 512],
            kernel_sizes: List[int] = [3, 3, 3, 3],
            strides: List[int] = [1, 2, 2, 1],
            transformer_dim: int = 512,
            num_heads: int = 8,
            num_layers: int = 4,
            dropout: float = 0.1
    ):
        super().__init__()

        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_ch = input_channels

        for out_ch, k, s in zip(channels, kernel_sizes, strides):
            padding = (k - 1) // 2
            self.conv_layers.append(
                ConvBlock(in_ch, out_ch, k, s, padding)
            )
            in_ch = out_ch

        # Ensure final channel matches transformer dimension
        if channels[-1] != transformer_dim:
            self.projection = nn.Conv1d(channels[-1], transformer_dim, 1)
        else:
            self.projection = nn.Identity()

        # Positional encoding
        self.pos_encoding = PositionalEncoding(transformer_dim, dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(transformer_dim, num_heads, transformer_dim * 4, dropout)
            for _ in range(num_layers)
        ])

        self.output_dim = transformer_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, time)
        Returns:
            content: (batch, transformer_dim, time')
        """
        x = mel

        # CNN encoding
        for conv in self.conv_layers:
            x = conv(x)

        # Project to transformer dimension
        x = self.projection(x)

        # Transpose for transformer (batch, time, channels)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        for layer in self.transformer_layers:
            x = layer(x)

        # Transpose back to (batch, channels, time)
        x = x.transpose(1, 2)

        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)