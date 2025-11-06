"""
Modèle VoiceGAN complet - Conversion de voix A→B
"""
import torch
import torch.nn as nn
from .content_encoder import ContentEncoder
from .style_encoder import StyleEncoder
from .generator import Generator
from .discriminator import MultiScaleDiscriminator


class VoiceGAN(nn.Module):
    """
    Modèle VoiceGAN complet pour la conversion de voix A→B

    Architecture:
        1. Content Encoder (CNN + Transformer) - Encode le contenu de A
        2. Style Encoder (CNN) - Encode le style de B
        3. Generator - Fusionne contenu + style pour générer A→B
        4. Discriminator - Évalue le réalisme du résultat
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Content Encoder: CNN + Transformer
        self.content_encoder = ContentEncoder(
            n_mels=config.N_MELS,
            content_dim=config.CONTENT_DIM,
            cnn_channels=config.CONTENT_CNN_CHANNELS,
            transformer_layers=config.CONTENT_TRANSFORMER_LAYERS,
            transformer_heads=config.CONTENT_TRANSFORMER_HEADS,
            dropout=config.CONTENT_DROPOUT
        )

        # Style Encoder: CNN
        self.style_encoder = StyleEncoder(
            n_mels=config.N_MELS,
            style_dim=config.STYLE_DIM,
            cnn_channels=config.STYLE_CNN_CHANNELS,
            kernel_size=config.STYLE_KERNEL_SIZE,
            pooling_type=config.STYLE_POOLING
        )

        # Generator
        self.generator = Generator(
            content_dim=config.CONTENT_DIM,
            style_dim=config.STYLE_DIM,
            n_mels=config.N_MELS,
            hidden_channels=config.GENERATOR_CHANNELS,
            n_residual_blocks=6,
            upsample_rates=config.GENERATOR_UPSAMPLE_RATES,
            kernel_size=config.GENERATOR_KERNEL_SIZE
        )

        # Discriminator (Multi-Scale)
        self.discriminator = MultiScaleDiscriminator(
            n_mels=config.N_MELS,
            num_scales=3,
            channels=config.DISC_CHANNELS[:-1]  # Enlever le dernier (1)
        )

    def forward(self, source_mel, target_mel, mode='train'):
        """
        Forward pass

        Args:
            source_mel: (batch, n_mels, time) - Melspectrogram de A (source)
            target_mel: (batch, n_mels, time) - Melspectrogram de B (target/référence)
            mode: 'train' ou 'inference'

        Returns:
            Si mode='train': dict avec tous les outputs nécessaires
            Si mode='inference': generated_mel seulement
        """
        # 1. Encoder le contenu de A
        content = self.content_encoder(source_mel)  # (batch, time, content_dim)

        # 2. Encoder le style de B
        style = self.style_encoder(target_mel)  # (batch, style_dim)

        # 3. Générer le melspectrogram A→B
        generated_mel = self.generator(content, style)  # (batch, n_mels, time)

        if mode == 'inference':
            return generated_mel

        # Pour l'entraînement, retourner tout
        outputs = {
            'generated_mel': generated_mel,
            'content': content,
            'style': style
        }

        return outputs

    def convert_voice(self, source_mel, target_mel):
        """
        Convertit la voix A→B (mode inférence)

        Args:
            source_mel: (batch, n_mels, time) - Voix source A
            target_mel: (batch, n_mels, time) - Voix de référence B

        Returns:
            generated_mel: (batch, n_mels, time) - Voix convertie
        """
        self.eval()
        with torch.no_grad():
            return self.forward(source_mel, target_mel, mode='inference')

    def discriminate(self, mel_spec):
        """
        Évalue le réalisme d'un melspectrogram

        Args:
            mel_spec: (batch, n_mels, time)

        Returns:
            validity_scores: Liste de scores à différentes échelles
        """
        return self.discriminator(mel_spec)

    def get_content_representation(self, mel_spec):
        """Extrait uniquement la représentation du contenu"""
        with torch.no_grad():
            return self.content_encoder(mel_spec)

    def get_style_representation(self, mel_spec):
        """Extrait uniquement la représentation du style"""
        with torch.no_grad():
            return self.style_encoder(mel_spec)

    def save_checkpoint(self, path, optimizer_g=None, optimizer_d=None,
                        epoch=0, step=0, losses=None):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'content_encoder': self.content_encoder.state_dict(),
            'style_encoder': self.style_encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
        }

        if optimizer_g is not None:
            checkpoint['optimizer_g'] = optimizer_g.state_dict()
        if optimizer_d is not None:
            checkpoint['optimizer_d'] = optimizer_d.state_dict()
        if losses is not None:
            checkpoint['losses'] = losses

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path, load_optimizers=False,
                        optimizer_g=None, optimizer_d=None):
        """Charge un checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')

        self.content_encoder.load_state_dict(checkpoint['content_encoder'])
        self.style_encoder.load_state_dict(checkpoint['style_encoder'])
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

        if load_optimizers and optimizer_g is not None and 'optimizer_g' in checkpoint:
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        if load_optimizers and optimizer_d is not None and 'optimizer_d' in checkpoint:
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])

        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)

        print(f"Checkpoint loaded from {path} (epoch {epoch}, step {step})")

        return epoch, step

    def count_parameters(self):
        """Compte le nombre de paramètres de chaque composant"""
        counts = {
            'content_encoder': sum(p.numel() for p in self.content_encoder.parameters()),
            'style_encoder': sum(p.numel() for p in self.style_encoder.parameters()),
            'generator': sum(p.numel() for p in self.generator.parameters()),
            'discriminator': sum(p.numel() for p in self.discriminator.parameters()),
        }
        counts['total'] = sum(counts.values())

        return counts


def create_voicegan_model(config):
    """Factory function pour créer le modèle"""
    model = VoiceGAN(config)

    # Afficher les informations du modèle
    param_counts = model.count_parameters()
    print("\n" + "=" * 60)
    print("VoiceGAN Model Created")
    print("=" * 60)
    print(f"Content Encoder parameters: {param_counts['content_encoder']:,}")
    print(f"Style Encoder parameters: {param_counts['style_encoder']:,}")
    print(f"Generator parameters: {param_counts['generator']:,}")
    print(f"Discriminator parameters: {param_counts['discriminator']:,}")
    print(f"Total parameters: {param_counts['total']:,}")
    print("=" * 60 + "\n")

    return model


if __name__ == "__main__":
    # Test du modèle complet
    from config.model_config import ModelConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Créer le modèle
    model = create_voicegan_model(ModelConfig).to(device)

    # Test avec des données synthétiques
    batch_size = 2
    n_mels = 80
    time_steps = 200

    source_mel = torch.randn(batch_size, n_mels, time_steps).to(device)
    target_mel = torch.randn(batch_size, n_mels, time_steps).to(device)

    print("Testing forward pass...")
    outputs = model(source_mel, target_mel, mode='train')

    print(f"Source mel shape: {source_mel.shape}")
    print(f"Target mel shape: {target_mel.shape}")
    print(f"Generated mel shape: {outputs['generated_mel'].shape}")
    print(f"Content shape: {outputs['content'].shape}")
    print(f"Style shape: {outputs['style'].shape}")

    # Test discrimination
    print("\nTesting discriminator...")
    disc_outputs = model.discriminate(outputs['generated_mel'])
    print(f"Number of discriminator scales: {len(disc_outputs)}")
    for i, out in enumerate(disc_outputs):
        print(f"  Scale {i} output shape: {out.shape}")

    # Test conversion
    print("\nTesting voice conversion...")
    converted = model.convert_voice(source_mel, target_mel)
    print(f"Converted voice shape: {converted.shape}")

    # Test checkpoint
    print("\nTesting checkpoint save/load...")
    model.save_checkpoint('test_checkpoint.pth', epoch=1, step=100)
    epoch, step = model.load_checkpoint('test_checkpoint.pth')
    print(f"Loaded checkpoint: epoch={epoch}, step={step}")

    print("\n✓ VoiceGAN model test passed!")