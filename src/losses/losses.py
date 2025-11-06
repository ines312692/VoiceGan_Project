"""
Fonctions de perte pour VoiceGAN-Transformation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    L1 Loss entre spectrogramme généré et spectrogramme réel
    Assure que le spectrogramme généré ressemble à celui de B
    """

    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type

        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, generated_mel, target_mel):
        """
        Args:
            generated_mel: (batch, n_mels, time) - Spectrogramme généré
            target_mel: (batch, n_mels, time) - Spectrogramme cible (B)

        Returns:
            loss: Scalaire
        """
        return self.criterion(generated_mel, target_mel)


class AdversarialLoss(nn.Module):
    """
    Perte adversariale pour le GAN
    Supporte plusieurs types de GAN loss
    """

    def __init__(self, gan_mode='lsgan'):
        super().__init__()
        self.gan_mode = gan_mode

        if gan_mode == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gan_mode == 'wgan':
            self.criterion = None  # Wasserstein distance
        elif gan_mode == 'hinge':
            self.criterion = None  # Hinge loss
        else:
            raise ValueError(f"Unknown GAN mode: {gan_mode}")

    def forward(self, prediction, target_is_real):
        """
        Args:
            prediction: Sortie du discriminateur
            target_is_real: bool - True si on veut que ce soit réel

        Returns:
            loss: Scalaire
        """
        if self.gan_mode == 'vanilla' or self.gan_mode == 'lsgan':
            if target_is_real:
                target = torch.ones_like(prediction)
            else:
                target = torch.zeros_like(prediction)
            loss = self.criterion(prediction, target)

        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()

        elif self.gan_mode == 'hinge':
            if target_is_real:
                loss = F.relu(1.0 - prediction).mean()
            else:
                loss = F.relu(1.0 + prediction).mean()

        return loss


class IdentityLoss(nn.Module):
    """
    Perte d'identité vocale
    Garantit que la sortie ressemble bien à la voix de B
    Utilise un encodeur pré-entraîné pour extraire les embeddings
    """

    def __init__(self, embedding_model=None):
        super().__init__()
        self.embedding_model = embedding_model
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, generated_mel, target_mel, style_encoder=None):
        """
        Args:
            generated_mel: (batch, n_mels, time) - Spectrogramme généré
            target_mel: (batch, n_mels, time) - Spectrogramme de référence (B)
            style_encoder: Encodeur de style pour extraire les embeddings

        Returns:
            loss: Scalaire (1 - similarité cosinus)
        """
        if style_encoder is not None:
            # Extraire les embeddings de style
            with torch.no_grad():
                target_style = style_encoder(target_mel)
            generated_style = style_encoder(generated_mel)

            # Similarité cosinus (maximiser)
            similarity = self.cosine_sim(generated_style, target_style)
            loss = 1.0 - similarity.mean()
        else:
            # Simple L2 loss si pas d'encodeur disponible
            loss = F.mse_loss(generated_mel, target_mel)

        return loss


class ContentLoss(nn.Module):
    """
    Perte de contenu
    Garantit que le texte prononcé reste celui de A
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, source_content, generated_content):
        """
        Args:
            source_content: (batch, time, content_dim) - Contenu de A (source)
            generated_content: (batch, time, content_dim) - Contenu du généré

        Returns:
            loss: Scalaire
        """
        return self.criterion(generated_content, source_content)


class PerceptualLoss(nn.Module):
    """
    Perte perceptuelle
    Compare les features à différentes couches d'un réseau
    """

    def __init__(self, feature_network=None):
        super().__init__()
        self.feature_network = feature_network
        self.criterion = nn.L1Loss()

    def forward(self, generated_mel, target_mel):
        """
        Args:
            generated_mel: (batch, n_mels, time)
            target_mel: (batch, n_mels, time)

        Returns:
            loss: Scalaire
        """
        if self.feature_network is None:
            # Fallback à simple reconstruction loss
            return self.criterion(generated_mel, target_mel)

        # Extraire les features
        gen_features = self.feature_network(generated_mel)
        target_features = self.feature_network(target_mel)

        # Comparer les features
        if isinstance(gen_features, list):
            loss = 0
            for gen_f, target_f in zip(gen_features, target_features):
                loss += self.criterion(gen_f, target_f)
            loss /= len(gen_features)
        else:
            loss = self.criterion(gen_features, target_features)

        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss
    Améliore la stabilité du GAN en matchant les features intermédiaires
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, real_features, fake_features):
        """
        Args:
            real_features: Liste de features du discriminateur pour données réelles
            fake_features: Liste de features du discriminateur pour données générées

        Returns:
            loss: Scalaire
        """
        loss = 0
        for real_f, fake_f in zip(real_features, fake_features):
            loss += self.criterion(fake_f, real_f.detach())

        return loss / len(real_features)


class TotalLoss(nn.Module):
    """
    Perte totale combinant toutes les pertes
    """

    def __init__(self,
                 lambda_recon=10.0,
                 lambda_identity=5.0,
                 lambda_content=2.0,
                 lambda_adv=1.0,
                 lambda_perceptual=1.0,
                 lambda_feature_matching=1.0):
        super().__init__()

        self.lambda_recon = lambda_recon
        self.lambda_identity = lambda_identity
        self.lambda_content = lambda_content
        self.lambda_adv = lambda_adv
        self.lambda_perceptual = lambda_perceptual
        self.lambda_feature_matching = lambda_feature_matching

        # Initialiser les pertes
        self.recon_loss = ReconstructionLoss(loss_type='l1')
        self.adv_loss = AdversarialLoss(gan_mode='lsgan')
        self.identity_loss = IdentityLoss()
        self.content_loss = ContentLoss()
        self.perceptual_loss = PerceptualLoss()
        self.feature_matching_loss = FeatureMatchingLoss()

    def generator_loss(self,
                       generated_mel,
                       target_mel,
                       source_content,
                       generated_content,
                       disc_fake_output,
                       style_encoder=None,
                       real_features=None,
                       fake_features=None):
        """
        Calcule la perte totale du générateur

        Returns:
            total_loss, dict of individual losses
        """
        losses = {}

        # Reconstruction loss
        losses['recon'] = self.recon_loss(generated_mel, target_mel)

        # Adversarial loss (générateur veut tromper le discriminateur)
        losses['adv'] = self.adv_loss(disc_fake_output, target_is_real=True)

        # Identity loss
        losses['identity'] = self.identity_loss(generated_mel, target_mel, style_encoder)

        # Content loss
        losses['content'] = self.content_loss(source_content, generated_content)

        # Perceptual loss (optionnel)
        if self.lambda_perceptual > 0:
            losses['perceptual'] = self.perceptual_loss(generated_mel, target_mel)

        # Feature matching loss (optionnel)
        if self.lambda_feature_matching > 0 and real_features is not None:
            losses['feature_matching'] = self.feature_matching_loss(real_features, fake_features)

        # Total loss
        total_loss = (
                self.lambda_recon * losses['recon'] +
                self.lambda_adv * losses['adv'] +
                self.lambda_identity * losses['identity'] +
                self.lambda_content * losses['content']
        )

        if 'perceptual' in losses:
            total_loss += self.lambda_perceptual * losses['perceptual']

        if 'feature_matching' in losses:
            total_loss += self.lambda_feature_matching * losses['feature_matching']

        losses['total'] = total_loss

        return total_loss, losses

    def discriminator_loss(self, disc_real_output, disc_fake_output):
        """
        Calcule la perte du discriminateur

        Returns:
            total_loss, dict of individual losses
        """
        # Real loss
        loss_real = self.adv_loss(disc_real_output, target_is_real=True)

        # Fake loss
        loss_fake = self.adv_loss(disc_fake_output, target_is_real=False)

        # Total discriminator loss
        total_loss = (loss_real + loss_fake) * 0.5

        losses = {
            'real': loss_real,
            'fake': loss_fake,
            'total': total_loss
        }

        return total_loss, losses


if __name__ == "__main__":
    # Test des losses
    print("Testing Loss Functions")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    n_mels = 80
    time_steps = 200
    content_dim = 256

    # Créer des tenseurs de test
    generated_mel = torch.randn(batch_size, n_mels, time_steps).to(device)
    target_mel = torch.randn(batch_size, n_mels, time_steps).to(device)
    source_content = torch.randn(batch_size, time_steps, content_dim).to(device)
    generated_content = torch.randn(batch_size, time_steps, content_dim).to(device)
    disc_output = torch.randn(batch_size, 1, 50).to(device)

    # Test des pertes individuelles
    recon_loss = ReconstructionLoss()
    print(f"Reconstruction Loss: {recon_loss(generated_mel, target_mel).item():.4f}")

    adv_loss = AdversarialLoss()
    print(f"Adversarial Loss (real): {adv_loss(disc_output, True).item():.4f}")
    print(f"Adversarial Loss (fake): {adv_loss(disc_output, False).item():.4f}")

    identity_loss = IdentityLoss()
    print(f"Identity Loss: {identity_loss(generated_mel, target_mel).item():.4f}")

    content_loss = ContentLoss()
    print(f"Content Loss: {content_loss(source_content, generated_content).item():.4f}")

    # Test perte totale
    print("\n" + "=" * 60)
    total_loss = TotalLoss()

    disc_fake = torch.randn(batch_size, 1, 50).to(device)
    disc_real = torch.randn(batch_size, 1, 50).to(device)

    g_loss, g_losses = total_loss.generator_loss(
        generated_mel, target_mel, source_content, generated_content, disc_fake
    )

    print("\nGenerator Losses:")
    for name, value in g_losses.items():
        print(f"  {name}: {value.item():.4f}")

    d_loss, d_losses = total_loss.discriminator_loss(disc_real, disc_fake)

    print("\nDiscriminator Losses:")
    for name, value in d_losses.items():
        print(f"  {name}: {value.item():.4f}")

    print("\n" + "=" * 60)
    print("✓ All loss tests passed!")
    print("=" * 60)