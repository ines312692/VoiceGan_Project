import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import numpy as np

from src.models.voicegan import VoiceGAN
from src.losses.losses import VoiceGANLoss


class VoiceGANTrainer:
    """Trainer for VoiceGAN model"""

    def __init__(
            self,
            model: VoiceGAN,
            train_loader: DataLoader,
            val_loader: DataLoader,
            config: dict,
            device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            list(model.content_encoder.parameters()) +
            list(model.style_encoder.parameters()) +
            list(model.generator.parameters()),
            lr=config['learning_rate_g'],
            betas=(config['beta1'], config['beta2'])
        )

        self.optimizer_d = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=config['learning_rate_d'],
            betas=(config['beta1'], config['beta2'])
        )

        # Loss
        self.criterion = VoiceGANLoss(
            lambda_reconstruction=config['lambda_reconstruction'],
            lambda_adversarial=config['lambda_adversarial'],
            lambda_identity=config['lambda_identity'],
            lambda_content=config['lambda_content'],
            lambda_feature_matching=config['lambda_feature_matching']
        )

        # Directories
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.writer = SummaryWriter(self.log_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Config
        self.discriminator_start_epoch = config.get('discriminator_start_epoch', 5)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.save_every = config.get('save_every', 10)
        self.log_every = config.get('log_every', 100)

    def train(self, num_epochs: int):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Log
            self._log_epoch(epoch, train_losses, val_losses)

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        print("Training completed!")
        self.writer.close()

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        epoch_losses = {
            'g_total': [], 'g_recon': [], 'g_adv': [],
            'g_identity': [], 'g_content': [], 'g_fm': [],
            'd_total': [], 'd_real': [], 'd_fake': []
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')

        for batch_idx, (source_mel, target_mel) in enumerate(pbar):
            source_mel = source_mel.to(self.device)
            target_mel = target_mel.to(self.device)

            # Forward pass
            results = self.model(source_mel, target_mel)
            fake_mel = results['generated_mel']

            # === Train Generator ===
            self.optimizer_g.zero_grad()

            # Get discriminator outputs for fake
            disc_fake_out, disc_fake_feats = self.model.discriminate(fake_mel)
            disc_real_out, disc_real_feats = self.model.discriminate(target_mel)

            # Re-encode fake mel for losses
            content_fake = self.model.encode_content(fake_mel)
            style_fake = self.model.encode_style(fake_mel)

            # Compute generator loss
            g_loss, g_loss_dict = self.criterion.generator_loss(
                real_mel=target_mel,
                fake_mel=fake_mel,
                disc_fake_outputs=disc_fake_out,
                disc_fake_features=disc_fake_feats,
                disc_real_features=disc_real_feats,
                content_source=results['content'],
                content_fake=content_fake,
                style_target=results['style'],
                style_fake=style_fake
            )

            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.content_encoder.parameters()) +
                list(self.model.style_encoder.parameters()) +
                list(self.model.generator.parameters()),
                self.grad_clip
            )
            self.optimizer_g.step()

            # === Train Discriminator ===
            if self.current_epoch >= self.discriminator_start_epoch:
                self.optimizer_d.zero_grad()

                # Get discriminator outputs
                disc_real_out, _ = self.model.discriminate(target_mel)
                disc_fake_out, _ = self.model.discriminate(fake_mel.detach())

                # Compute discriminator loss
                d_loss, d_loss_dict = self.criterion.discriminator_loss(
                    disc_real_out, disc_fake_out
                )

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(),
                    self.grad_clip
                )
                self.optimizer_d.step()
            else:
                d_loss_dict = {'d_total': 0, 'd_real': 0, 'd_fake': 0}

            # Update losses
            for k, v in {**g_loss_dict, **d_loss_dict}.items():
                epoch_losses[k].append(v)

            # Log
            if self.global_step % self.log_every == 0:
                for k, v in {**g_loss_dict, **d_loss_dict}.items():
                    self.writer.add_scalar(f'train/{k}', v, self.global_step)

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'g_loss': f"{g_loss_dict['g_total']:.4f}",
                'd_loss': f"{d_loss_dict['d_total']:.4f}"
            })

        # Average losses
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def validate(self):
        """Validation loop"""
        self.model.eval()

        val_losses = {
            'g_total': [], 'g_recon': []
        }

        with torch.no_grad():
            for source_mel, target_mel in self.val_loader:
                source_mel = source_mel.to(self.device)
                target_mel = target_mel.to(self.device)

                # Forward
                results = self.model(source_mel, target_mel)
                fake_mel = results['generated_mel']

                # Reconstruction loss only for validation
                recon_loss = torch.nn.functional.l1_loss(fake_mel, target_mel)

                val_losses['g_recon'].append(recon_loss.item())

        return {k: np.mean(v) for k, v in val_losses.items()}

    def _log_epoch(self, epoch, train_losses, val_losses):
        """Log epoch results"""
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - G: {train_losses['g_total']:.4f}, D: {train_losses['d_total']:.4f}")
        print(f"  Val   - Recon: {val_losses['g_recon']:.4f}")

        # TensorBoard
        for k, v in train_losses.items():
            self.writer.add_scalar(f'epoch/train_{k}', v, epoch)
        for k, v in val_losses.items():
            self.writer.add_scalar(f'epoch/val_{k}', v, epoch)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': self.config
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Checkpoint loaded: {path}")
        print(f"Resuming from epoch {self.current_epoch}")