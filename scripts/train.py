"""
Script d'entraînement principal pour VoiceGAN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from config.model_config import ModelConfig, TrainingConfig, DataConfig
from src.models.voicegan import create_voicegan_model
from src.losses.losses import TotalLoss
from src.training.dataset import create_dataloaders
from src.preprocessing.audio_processor import AudioProcessor


class Trainer:
    """Classe pour gérer l'entraînement de VoiceGAN"""

    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizers
        self.optimizer_g = optim.Adam(
            list(model.content_encoder.parameters()) +
            list(model.style_encoder.parameters()) +
            list(model.generator.parameters()),
            lr=config.LEARNING_RATE_G,
            betas=(config.BETA1, config.BETA2)
        )

        self.optimizer_d = optim.Adam(
            model.discriminator.parameters(),
            lr=config.LEARNING_RATE_D,
            betas=(config.BETA1, config.BETA2)
        )

        # Learning rate schedulers
        if TrainingConfig.USE_SCHEDULER:
            self.scheduler_g = self._create_scheduler(self.optimizer_g)
            self.scheduler_d = self._create_scheduler(self.optimizer_d)
        else:
            self.scheduler_g = None
            self.scheduler_d = None

        # Loss function
        self.criterion = TotalLoss(
            lambda_recon=config.LAMBDA_RECON,
            lambda_identity=config.LAMBDA_IDENTITY,
            lambda_content=config.LAMBDA_CONTENT,
            lambda_adv=config.LAMBDA_ADV
        )

        # TensorBoard
        self.writer = SummaryWriter(TrainingConfig.LOG_DIR)

        # Checkpoints
        self.checkpoint_dir = Path(TrainingConfig.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Mixed precision training
        self.scaler_g = torch.cuda.amp.GradScaler() if TrainingConfig.USE_AMP else None
        self.scaler_d = torch.cuda.amp.GradScaler() if TrainingConfig.USE_AMP else None

    def _create_scheduler(self, optimizer):
        """Crée un learning rate scheduler"""
        if TrainingConfig.SCHEDULER_TYPE == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=TrainingConfig.SCHEDULER_STEP_SIZE,
                gamma=TrainingConfig.SCHEDULER_GAMMA
            )
        elif TrainingConfig.SCHEDULER_TYPE == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.NUM_EPOCHS
            )
        elif TrainingConfig.SCHEDULER_TYPE == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        return None

    def train_epoch(self):
        """Entraîne le modèle pour une epoch"""
        self.model.train()

        epoch_losses_g = {'total': 0, 'recon': 0, 'adv': 0, 'identity': 0, 'content': 0}
        epoch_losses_d = {'total': 0, 'real': 0, 'fake': 0}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            source_mel = batch['source_mel'].to(self.device)
            target_mel = batch['target_mel'].to(self.device)

            # ========== Train Discriminator ==========
            self.optimizer_d.zero_grad()

            with torch.cuda.amp.autocast(enabled=TrainingConfig.USE_AMP):
                # Forward pass
                outputs = self.model(source_mel, target_mel, mode='train')
                generated_mel = outputs['generated_mel']

                # Discriminator outputs
                disc_real = self.model.discriminate(target_mel)
                disc_fake = self.model.discriminate(generated_mel.detach())

                # Discriminator loss
                d_loss = 0
                for real, fake in zip(disc_real, disc_fake):
                    loss_real = self.criterion.adv_loss(real, target_is_real=True)
                    loss_fake = self.criterion.adv_loss(fake, target_is_real=False)
                    d_loss += (loss_real + loss_fake) * 0.5
                d_loss /= len(disc_real)

            # Backward pass
            if self.scaler_d is not None:
                self.scaler_d.scale(d_loss).backward()
                self.scaler_d.unscale_(self.optimizer_d)
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(),
                    TrainingConfig.GRAD_CLIP_VALUE
                )
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.discriminator.parameters(),
                    TrainingConfig.GRAD_CLIP_VALUE
                )
                self.optimizer_d.step()

            # ========== Train Generator ==========
            self.optimizer_g.zero_grad()

            with torch.cuda.amp.autocast(enabled=TrainingConfig.USE_AMP):
                # Forward pass
                outputs = self.model(source_mel, target_mel, mode='train')
                generated_mel = outputs['generated_mel']
                content = outputs['content']

                # Re-encode content from generated
                generated_content = self.model.content_encoder(generated_mel)

                # Discriminator output for generator
                disc_fake_for_g = self.model.discriminate(generated_mel)

                # Generator loss
                g_loss, g_losses = self.criterion.generator_loss(
                    generated_mel=generated_mel,
                    target_mel=target_mel,
                    source_content=content,
                    generated_content=generated_content,
                    disc_fake_output=disc_fake_for_g[0],  # Use first scale
                    style_encoder=self.model.style_encoder
                )

            # Backward pass
            if self.scaler_g is not None:
                self.scaler_g.scale(g_loss).backward()
                self.scaler_g.unscale_(self.optimizer_g)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.content_encoder.parameters()) +
                    list(self.model.style_encoder.parameters()) +
                    list(self.model.generator.parameters()),
                    TrainingConfig.GRAD_CLIP_VALUE
                )
                self.scaler_g.step(self.optimizer_g)
                self.scaler_g.update()
            else:
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.content_encoder.parameters()) +
                    list(self.model.style_encoder.parameters()) +
                    list(self.model.generator.parameters()),
                    TrainingConfig.GRAD_CLIP_VALUE
                )
                self.optimizer_g.step()

            # Accumuler les pertes
            epoch_losses_d['total'] += d_loss.item()
            for key in g_losses:
                if key in epoch_losses_g:
                    epoch_losses_g[key] += g_losses[key].item()

            # Logging
            if self.global_step % self.config.LOG_INTERVAL == 0:
                self.writer.add_scalar('train/d_loss', d_loss.item(), self.global_step)
                self.writer.add_scalar('train/g_loss', g_loss.item(), self.global_step)
                for key, value in g_losses.items():
                    self.writer.add_scalar(f'train/g_{key}', value.item(), self.global_step)

            # Update progress bar
            pbar.set_postfix({
                'g_loss': f"{g_loss.item():.4f}",
                'd_loss': f"{d_loss.item():.4f}"
            })

            self.global_step += 1

        # Moyennes sur l'epoch
        for key in epoch_losses_g:
            epoch_losses_g[key] /= len(self.train_loader)
        for key in epoch_losses_d:
            epoch_losses_d[key] /= len(self.train_loader)

        return epoch_losses_g, epoch_losses_d

    @torch.no_grad()
    def validate(self):
        """Validation"""
        self.model.eval()

        val_losses_g = {'total': 0, 'recon': 0, 'adv': 0, 'identity': 0, 'content': 0}
        val_losses_d = {'total': 0}

        for batch in tqdm(self.val_loader, desc="Validation"):
            source_mel = batch['source_mel'].to(self.device)
            target_mel = batch['target_mel'].to(self.device)

            # Forward pass
            outputs = self.model(source_mel, target_mel, mode='train')
            generated_mel = outputs['generated_mel']
            content = outputs['content']

            # Re-encode content
            generated_content = self.model.content_encoder(generated_mel)

            # Discriminator
            disc_fake = self.model.discriminate(generated_mel)

            # Losses
            g_loss, g_losses = self.criterion.generator_loss(
                generated_mel, target_mel, content, generated_content,
                disc_fake[0], self.model.style_encoder
            )

            # Accumuler
            for key in g_losses:
                if key in val_losses_g:
                    val_losses_g[key] += g_losses[key].item()

        # Moyennes
        for key in val_losses_g:
            val_losses_g[key] /= len(self.val_loader)

        return val_losses_g

    def train(self):
        """Boucle d'entraînement principale"""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch

            # Train
            train_losses_g, train_losses_d = self.train_epoch()

            print(f"\nEpoch {epoch} - Train:")
            print(f"  Generator Loss: {train_losses_g['total']:.4f}")
            print(f"  Discriminator Loss: {train_losses_d['total']:.4f}")

            # Validate
            val_losses_g = self.validate()

            print(f"Epoch {epoch} - Validation:")
            print(f"  Generator Loss: {val_losses_g['total']:.4f}")

            # TensorBoard
            self.writer.add_scalar('epoch/train_g_loss', train_losses_g['total'], epoch)
            self.writer.add_scalar('epoch/train_d_loss', train_losses_d['total'], epoch)
            self.writer.add_scalar('epoch/val_g_loss', val_losses_g['total'], epoch)

            # Learning rate scheduling
            if self.scheduler_g is not None:
                if TrainingConfig.SCHEDULER_TYPE == 'plateau':
                    self.scheduler_g.step(val_losses_g['total'])
                    self.scheduler_d.step(val_losses_g['total'])
                else:
                    self.scheduler_g.step()
                    self.scheduler_d.step()

            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                self.model.save_checkpoint(
                    checkpoint_path,
                    self.optimizer_g,
                    self.optimizer_d,
                    epoch,
                    self.global_step,
                    {'train_g': train_losses_g, 'val_g': val_losses_g}
                )

            # Save best model
            if val_losses_g['total'] < self.best_val_loss:
                self.best_val_loss = val_losses_g['total']
                best_path = self.checkpoint_dir / "best_model.pth"
                self.model.save_checkpoint(best_path, self.optimizer_g, self.optimizer_d,
                                           epoch, self.global_step)
                print(f"✓ New best model saved! Val loss: {self.best_val_loss:.4f}")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train VoiceGAN')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    # Configuration
    config = ModelConfig()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size

    # Device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Create model
    model = create_voicegan_model(config).to(device)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config, DataConfig)

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Resume from checkpoint
    if args.resume:
        epoch, step = model.load_checkpoint(args.resume, True,
                                            trainer.optimizer_g, trainer.optimizer_d)
        trainer.current_epoch = epoch + 1
        trainer.global_step = step

    # Train
    trainer.train()


if __name__ == "__main__":
    main()