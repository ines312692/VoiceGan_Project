#!/usr/bin/env python
"""Evaluation script for VoiceGAN"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys
import json
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import Config
from src.models.voicegan import VoiceGAN
from src.training.dataset import VoiceDataset, VoiceCollator
from src.evaluation.metrics import VoiceConversionMetrics, evaluate_batch
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.mel_spectrogram import MelSpectrogramProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate VoiceGAN')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None = all)')
    parser.add_argument('--save_audio', action='store_true',
                        help='Save converted audio samples')
    return parser.parse_args()

def load_model(checkpoint_path: str, config: Config, device: str):
    """Load trained model"""
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

    # Setup
    device = args.device if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")

    # Load config
    config = Config(args.config)

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, config, device)

    # Create test dataset
    print("Loading test dataset...")
    test_dataset = VoiceDataset(
        data_dir=args.test_dir,
        audio_config=config.audio.__dict__,
        segment_length=config.audio.segment_length,
        split='test'
    )

    # Limit number of samples if specified
    if args.num_samples:
        test_dataset.pairs = test_dataset.pairs[:args.num_samples]

    print(f"Evaluating on {len(test_dataset)} samples")

    # Create data loader
    collator = VoiceCollator()
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator
    )

    # Initialize metrics
    metrics_calculator = VoiceConversionMetrics()

    # Evaluation
    all_metrics = {
        'mcd': [],
        'cosine_similarity': [],
        'spectral_convergence': [],
        'log_spectral_distance': []
    }

    print("Evaluating...")

    # Audio processors for saving
    if args.save_audio:
        audio_dir = output_dir / 'audio_samples'
        audio_dir.mkdir(exist_ok=True)

        audio_processor = AudioProcessor(
            sample_rate=config.audio.sample_rate
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

    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (source_mels, target_mels) in enumerate(tqdm(test_loader)):
            # Evaluate batch
            batch_metrics = evaluate_batch(model, source_mels, target_mels, device)

            # Accumulate metrics
            for k, v in batch_metrics.items():
                all_metrics[k].append(v)

            # Save audio samples
            if args.save_audio and batch_idx < 5:  # Save first 5 batches
                source_mels = source_mels.to(device)
                target_mels = target_mels.to(device)
                converted_mels = model.convert(source_mels, target_mels)

                for i in range(min(2, len(source_mels))):  # Save 2 per batch
                    # Convert to audio
                    source_audio = mel_processor.mel_to_wav(source_mels[i].cpu())
                    target_audio = mel_processor.mel_to_wav(target_mels[i].cpu())
                    converted_audio = mel_processor.mel_to_wav(converted_mels[i].cpu())

                    # Save
                    audio_processor.save_audio(
                        source_audio,
                        audio_dir / f'sample_{sample_idx}_source.wav'
                    )
                    audio_processor.save_audio(
                        target_audio,
                        audio_dir / f'sample_{sample_idx}_target.wav'
                    )
                    audio_processor.save_audio(
                        converted_audio,
                        audio_dir / f'sample_{sample_idx}_converted.wav'
                    )

                    sample_idx += 1

    # Compute statistics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    results = {}
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        results[metric_name] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'min': float(min_val),
            'max': float(max_val)
        }

        print(f"\n{metric_name.upper()}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Min:  {min_val:.4f}")
        print(f"  Max:  {max_val:.4f}")

    # Save results
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    if args.save_audio:
        print(f"Audio samples saved to: {audio_dir}")

    print("\nEvaluation completed!")

if __name__ == '__main__':
    main()