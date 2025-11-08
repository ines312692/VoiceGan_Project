import torch
import numpy as np
from scipy import spatial
import librosa
from typing import Tuple


class VoiceConversionMetrics:
    """Metrics for evaluating voice conversion quality"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def mel_cepstral_distortion(
            self,
            generated: torch.Tensor,
            target: torch.Tensor,
            n_mfcc: int = 13
    ) -> float:
        """
        Compute Mel-Cepstral Distortion (MCD)
        Lower is better (typically < 6 dB is good)

        Args:
            generated: Generated mel-spectrogram (n_mels, time)
            target: Target mel-spectrogram (n_mels, time)
            n_mfcc: Number of MFCC coefficients

        Returns:
            MCD in dB
        """
        # Convert to numpy
        if isinstance(generated, torch.Tensor):
            generated = generated.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # Compute MFCCs
        mfcc_gen = librosa.feature.mfcc(S=generated, n_mfcc=n_mfcc)
        mfcc_tgt = librosa.feature.mfcc(S=target, n_mfcc=n_mfcc)

        # Align lengths
        min_len = min(mfcc_gen.shape[1], mfcc_tgt.shape[1])
        mfcc_gen = mfcc_gen[:, :min_len]
        mfcc_tgt = mfcc_tgt[:, :min_len]

        # Compute MCD
        diff = mfcc_gen - mfcc_tgt
        mcd = np.sqrt(np.sum(diff ** 2, axis=0)).mean()
        mcd = (10.0 / np.log(10.0)) * mcd  # Convert to dB

        return float(mcd)

    def cosine_similarity(
            self,
            embedding1: torch.Tensor,
            embedding2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between style embeddings
        Higher is better (1.0 = identical)

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity [-1, 1]
        """
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.cpu().numpy()

        # Flatten if needed
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()

        # Compute cosine similarity
        similarity = 1 - spatial.distance.cosine(embedding1, embedding2)

        return float(similarity)

    def spectral_convergence(
            self,
            generated: torch.Tensor,
            target: torch.Tensor
    ) -> float:
        """
        Compute spectral convergence
        Lower is better

        Args:
            generated: Generated spectrogram
            target: Target spectrogram

        Returns:
            Spectral convergence
        """
        if isinstance(generated, torch.Tensor):
            generated = generated.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # Align lengths
        min_len = min(generated.shape[-1], target.shape[-1])
        generated = generated[..., :min_len]
        target = target[..., :min_len]

        # Compute norm
        norm_diff = np.linalg.norm(generated - target)
        norm_target = np.linalg.norm(target)

        if norm_target == 0:
            return 0.0

        return float(norm_diff / norm_target)

    def log_spectral_distance(
            self,
            generated: torch.Tensor,
            target: torch.Tensor
    ) -> float:
        """
        Compute log spectral distance
        Lower is better

        Args:
            generated: Generated spectrogram
            target: Target spectrogram

        Returns:
            Log spectral distance
        """
        if isinstance(generated, torch.Tensor):
            generated = generated.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # Add small constant to avoid log(0)
        eps = 1e-8
        generated = np.log(generated + eps)
        target = np.log(target + eps)

        # Align lengths
        min_len = min(generated.shape[-1], target.shape[-1])
        generated = generated[..., :min_len]
        target = target[..., :min_len]

        # Compute squared difference
        lsd = np.sqrt(np.mean((generated - target) ** 2))

        return float(lsd)

    def compute_all_metrics(
            self,
            generated_mel: torch.Tensor,
            target_mel: torch.Tensor,
            generated_style: torch.Tensor,
            target_style: torch.Tensor
    ) -> dict:
        """
        Compute all metrics

        Returns:
            Dictionary with all metric values
        """
        metrics = {
            'mcd': self.mel_cepstral_distortion(generated_mel, target_mel),
            'cosine_similarity': self.cosine_similarity(generated_style, target_style),
            'spectral_convergence': self.spectral_convergence(generated_mel, target_mel),
            'log_spectral_distance': self.log_spectral_distance(generated_mel, target_mel)
        }

        return metrics


def evaluate_batch(
        model,
        source_mels: torch.Tensor,
        target_mels: torch.Tensor,
        device: str = 'cuda'
) -> dict:
    """
    Evaluate a batch of conversions

    Args:
        model: VoiceGAN model
        source_mels: Source mel-spectrograms (batch, n_mels, time)
        target_mels: Target mel-spectrograms (batch, n_mels, time)
        device: Device to use

    Returns:
        Dictionary of averaged metrics
    """
    model.eval()
    metrics_calculator = VoiceConversionMetrics()

    batch_metrics = {
        'mcd': [],
        'cosine_similarity': [],
        'spectral_convergence': [],
        'log_spectral_distance': []
    }

    with torch.no_grad():
        source_mels = source_mels.to(device)
        target_mels = target_mels.to(device)

        # Convert
        converted_mels = model.convert(source_mels, target_mels)

        # Extract styles
        converted_styles = model.encode_style(converted_mels)
        target_styles = model.encode_style(target_mels)

        # Compute metrics for each sample
        for i in range(len(source_mels)):
            metrics = metrics_calculator.compute_all_metrics(
                converted_mels[i],
                target_mels[i],
                converted_styles[i],
                target_styles[i]
            )

            for k, v in metrics.items():
                batch_metrics[k].append(v)

    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in batch_metrics.items()}

    return avg_metrics