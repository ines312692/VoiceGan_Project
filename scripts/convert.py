#!/usr/bin/env python
"""Voice conversion script"""

import torch
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import Config
from src.models.voicegan import VoiceGAN
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.mel_spectrogram import MelSpectrogramProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Convert voice A to B')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to source audio file')
    parser.add_argument('--target', type=str, required=True,
                        help='Path to target reference audio file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save converted audio')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    return parser.parse_args()

def load_model(checkpoint_path: str, config: Config, device: str):
    """Load trained model from checkpoint"""
    model = VoiceGAN(
        n_mels=config.audio.n_mels,
        content_channels=config.content_encoder.channels,
        content_kernel_sizes=config.content_encoder.kernel_sizes,
        content_strides=config.content_encoder.strides,
        transformer_dim=config.content_encoder.transformer_dim,
        num_heads=config.content_encoder.num_heads,
        num_transformer_layers=config.content_encoder.num_layers,
        style_channels=config.style_encoder.channels,
        style_kernel_sizes=config.style_encoder.kernel_sizes,
        style_strides=config.style_encoder.strides,
        style_dim=config.style_encoder.style_dim,
        generator_channels=config.generator.channels,
        generator_kernel_sizes=config.generator.kernel_sizes,
        upsample_rates=config.generator.upsample_rates,
        discriminator_channels=config.discriminator.channels,
        discriminator_kernel_sizes=config.discriminator.kernel_sizes,
        discriminator_strides=config.discriminator.strides
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

def main():
    args = parse_args()

    # Load config
    config = Config(args.config)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)

    # Initialize processors
    audio_processor = AudioProcessor(
        sample_rate=config.audio.sample_rate,
        segment_length=config.audio.segment_length
    )

    mel_processor = MelSpectrogramProcessor(
        sample_rate=config.audio.sample_rate,
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        win_length=config.audio.win_length,
        n_mels=config.audio.n_mels,
        fmin=config.audio.fmin,
        fmax=config.audio.fmax
    )

    # Load source audio
    print(f"Loading source audio: {args.source}")
    source_audio = audio_processor.load_audio(args.source)
    source_mel = mel_processor.wav_to_mel(source_audio).unsqueeze(0).to(device)

    # Load target reference audio
    print(f"Loading target reference: {args.target}")
    target_audio = audio_processor.load_audio(args.target)
    target_mel = mel_processor.wav_to_mel(target_audio).unsqueeze(0).to(device)

    # Convert
    print("Converting voice...")
    with torch.no_grad():
        converted_mel = model.convert(source_mel, target_mel)

    # Convert mel back to audio
    print("Converting to audio...")
    converted_audio = mel_processor.mel_to_wav(converted_mel.squeeze(0).cpu())

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_processor.save_audio(converted_audio, output_path)

    print(f"Converted audio saved to: {output_path}")

if __name__ == '__main__':
    main()