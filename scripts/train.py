#!/usr/bin/env python
"""Training script for VoiceGAN"""

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import Config
from src.models.voicegan import VoiceGAN
from src.training.dataset import VoiceDataset, VoiceCollator
from src.training.trainer import VoiceGANTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train VoiceGAN')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    config = Config(args.config)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = VoiceDataset(
        data_dir=args.data_dir + '/train',
        audio_config=config.audio.__dict__,
        segment_length=config.audio.segment_length,
        split='train'
    )

    val_dataset = VoiceDataset(
        data_dir=args.data_dir + '/val',
        audio_config=config.audio.__dict__,
        segment_length=config.audio.segment_length,
        split='val'
    )

    # Create data loaders
    collator = VoiceCollator()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create model
    print("Creating model...")
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

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer_config = {
        'learning_rate_g': config.training.learning_rate_g,
        'learning_rate_d': config.training.learning_rate_d,
        'beta1': config.training.beta1,
        'beta2': config.training.beta2,
        'lambda_reconstruction': config.training.lambda_reconstruction,
        'lambda_adversarial': config.training.lambda_adversarial,
        'lambda_identity': config.training.lambda_identity,
        'lambda_content': config.training.lambda_content,
        'lambda_feature_matching': config.training.lambda_feature_matching,
        'discriminator_start_epoch': config.training.discriminator_start_epoch,
        'grad_clip': config.training.grad_clip,
        'save_every': config.training.save_every,
        'log_every': config.training.log_every,
        'checkpoint_dir': config.logging['checkpoint_dir'],
        'log_dir': config.logging['log_dir']
    }

    trainer = VoiceGANTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=config.training.num_epochs)

    print("Training completed!")

if __name__ == '__main__':
    main()