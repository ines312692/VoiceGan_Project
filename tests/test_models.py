import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.content_encoder import ContentEncoder
from src.models.style_encoder import StyleEncoder
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.models.voicegan import VoiceGAN


class TestContentEncoder:
    def test_forward_shape(self):
        """Test that content encoder produces correct output shape"""
        encoder = ContentEncoder(
            input_channels=80,
            channels=[64, 128, 256, 512],
            transformer_dim=512
        )

        # Input: (batch, n_mels, time)
        x = torch.randn(2, 80, 100)
        output = encoder(x)

        # Output should be (batch, transformer_dim, time')
        assert output.dim() == 3
        assert output.size(0) == 2
        assert output.size(1) == 512


class TestStyleEncoder:
    def test_forward_shape(self):
        """Test that style encoder produces correct output shape"""
        encoder = StyleEncoder(
            input_channels=80,
            style_dim=256
        )

        # Input: (batch, n_mels, time)
        x = torch.randn(2, 80, 100)
        output = encoder(x)

        # Output should be (batch, style_dim)
        assert output.dim() == 2
        assert output.size(0) == 2
        assert output.size(1) == 256


class TestGenerator:
    def test_forward_shape(self):
        """Test that generator produces correct output shape"""
        generator = Generator(
            content_dim=512,
            style_dim=256,
            output_channels=80
        )

        # Inputs
        content = torch.randn(2, 512, 25)
        style = torch.randn(2, 256)

        output = generator(content, style)

        # Output should be (batch, output_channels, time')
        assert output.dim() == 3
        assert output.size(0) == 2
        assert output.size(1) == 80


class TestDiscriminator:
    def test_forward_shape(self):
        """Test that discriminator produces correct outputs"""
        discriminator = Discriminator(
            input_channels=80,
            num_scales=3
        )

        # Input: (batch, n_mels, time)
        x = torch.randn(2, 80, 100)
        outputs, features = discriminator(x)

        # Should return list of outputs and features for each scale
        assert len(outputs) == 3
        assert len(features) == 3

        for output in outputs:
            assert output.size(0) == 2  # batch size


class TestVoiceGAN:
    def test_forward(self):
        """Test VoiceGAN forward pass"""
        model = VoiceGAN(
            n_mels=80,
            transformer_dim=512,
            style_dim=256
        )

        source_mel = torch.randn(2, 80, 100)
        target_mel = torch.randn(2, 80, 100)

        results = model(source_mel, target_mel)

        assert 'generated_mel' in results
        assert 'content' in results
        assert 'style' in results

        assert results['generated_mel'].size(0) == 2
        assert results['generated_mel'].size(1) == 80

    def test_convert(self):
        """Test voice conversion"""
        model = VoiceGAN(n_mels=80)
        model.eval()

        source_mel = torch.randn(1, 80, 100)
        target_mel = torch.randn(1, 80, 100)

        converted = model.convert(source_mel, target_mel)

        assert converted.size(0) == 1
        assert converted.size(1) == 80

    def test_trainable_parameters(self):
        """Test that model has trainable parameters"""
        model = VoiceGAN(n_mels=80)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert params > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])