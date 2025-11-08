import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class VoiceGANLoss:
    """Combined loss for VoiceGAN training"""
    
    def __init__(
        self,
        lambda_reconstruction: float = 10.0,
        lambda_adversarial: float = 1.0,
        lambda_identity: float = 5.0,
        lambda_content: float = 1.0,
        lambda_feature_matching: float = 10.0
    ):
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_adversarial = lambda_adversarial
        self.lambda_identity = lambda_identity
        self.lambda_content = lambda_content
        self.lambda_feature_matching = lambda_feature_matching
        
        self.reconstruction_loss = ReconstructionLoss()
        self.adversarial_loss = AdversarialLoss()
        self.identity_loss = IdentityLoss()
        self.content_loss = ContentLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
    
    def generator_loss(
        self,
        real_mel: torch.Tensor,
        fake_mel: torch.Tensor,
        disc_fake_outputs: List[torch.Tensor],
        disc_fake_features: List[List[torch.Tensor]],
        disc_real_features: List[List[torch.Tensor]],
        content_source: torch.Tensor,
        content_fake: torch.Tensor,
        style_target: torch.Tensor,
        style_fake: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute generator losses
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss (L1)
        loss_recon = self.reconstruction_loss(fake_mel, real_mel)
        
        # Adversarial loss
        loss_adv = self.adversarial_loss.generator_loss(disc_fake_outputs)
        
        # Identity loss (style should match target)
        loss_identity = self.identity_loss(style_fake, style_target)
        
        # Content preservation loss
        loss_content = self.content_loss(content_fake, content_source)
        
        # Feature matching loss
        loss_fm = self.feature_matching_loss(disc_fake_features, disc_real_features)
        
        # Total loss
        total_loss = (
            self.lambda_reconstruction * loss_recon +
            self.lambda_adversarial * loss_adv +
            self.lambda_identity * loss_identity +
            self.lambda_content * loss_content +
            self.lambda_feature_matching * loss_fm
        )
        
        loss_dict = {
            'g_total': total_loss.item(),
            'g_recon': loss_recon.item(),
            'g_adv': loss_adv.item(),
            'g_identity': loss_identity.item(),
            'g_content': loss_content.item(),
            'g_fm': loss_fm.item()
        }
        
        return total_loss, loss_dict
    
    def discriminator_loss(
        self,
        disc_real_outputs: List[torch.Tensor],
        disc_fake_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute discriminator loss
        
        Returns:
            total_loss: Discriminator loss
            loss_dict: Dictionary of losses
        """
        loss_real = self.adversarial_loss.discriminator_loss_real(disc_real_outputs)
        loss_fake = self.adversarial_loss.discriminator_loss_fake(disc_fake_outputs)
        
        total_loss = loss_real + loss_fake
        
        loss_dict = {
            'd_total': total_loss.item(),
            'd_real': loss_real.item(),
            'd_fake': loss_fake.item()
        }
        
        return total_loss, loss_dict

class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss between generated and real mel-spectrograms"""
    
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(fake, real)

class AdversarialLoss(nn.Module):
    """GAN adversarial loss (LSGAN)"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def generator_loss(self, disc_fake_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Generator tries to fool discriminator"""
        loss = 0
        for fake_out in disc_fake_outputs:
            loss += self.mse_loss(fake_out, torch.ones_like(fake_out))
        return loss / len(disc_fake_outputs)
    
    def discriminator_loss_real(self, disc_real_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Discriminator should classify real as 1"""
        loss = 0
        for real_out in disc_real_outputs:
            loss += self.mse_loss(real_out, torch.ones_like(real_out))
        return loss / len(disc_real_outputs)
    
    def discriminator_loss_fake(self, disc_fake_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Discriminator should classify fake as 0"""
        loss = 0
        for fake_out in disc_fake_outputs:
            loss += self.mse_loss(fake_out, torch.zeros_like(fake_out))
        return loss / len(disc_fake_outputs)

class IdentityLoss(nn.Module):
    """Ensures generated style matches target style"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, style_fake: torch.Tensor, style_target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(style_fake, style_target)

class ContentLoss(nn.Module):
    """Ensures content is preserved from source"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, content_fake: torch.Tensor, content_source: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(content_fake, content_source)

class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for more stable training"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        fake_features: List[List[torch.Tensor]],
        real_features: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        loss = 0
        for fake_feats, real_feats in zip(fake_features, real_features):
            for fake_feat, real_feat in zip(fake_feats, real_feats):
                loss += F.l1_loss(fake_feat, real_feat.detach())
        
        num_features = sum(len(feats) for feats in fake_features)
        return loss / num_features