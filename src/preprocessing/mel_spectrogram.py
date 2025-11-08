import torch
import torchaudio
import librosa
import numpy as np
from typing import Optional, Tuple


class MelSpectrogramProcessor:
    """Convert audio to mel-spectrogram and vice versa"""

    def __init__(
            self,
            sample_rate: int = 22050,
            n_fft: int = 1024,
            hop_length: int = 256,
            win_length: int = 1024,
            n_mels: int = 80,
            fmin: float = 0.0,
            fmax: Optional[float] = 8000.0,
            center: bool = True,
            norm: str = "slaney",
            mel_scale: str = "slaney"
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax else sample_rate // 2
        self.center = center

        # Create mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax,
            norm=norm,
            htk=False
        )

        # PyTorch transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
            center=center,
            norm=norm,
            mel_scale=mel_scale
        )

    def wav_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel-spectrogram
        Args:
            audio: (batch, samples) or (samples,)
        Returns:
            mel: (batch, n_mels, time) or (n_mels, time)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Compute mel spectrogram
        mel = self.mel_transform(audio)

        # Convert to log scale
        mel = torch.clamp(mel, min=1e-5)
        mel = torch.log(mel)

        return mel.squeeze(0) if audio.size(0) == 1 else mel

    def mel_to_wav(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Convert mel-spectrogram back to waveform using Griffin-Lim
        Args:
            mel: (n_mels, time) or (batch, n_mels, time)
        Returns:
            audio: (samples,) or (batch, samples)
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Convert from log scale
        mel = torch.exp(mel)

        # Convert mel to linear spectrogram
        mel_basis_torch = torch.from_numpy(self.mel_basis).float().to(mel.device)
        spec = torch.matmul(mel_basis_torch.T, mel)

        # Apply Griffin-Lim
        audio = self._griffin_lim(spec)

        return audio.squeeze(0) if squeeze else audio

    def _griffin_lim(self, spec: torch.Tensor, n_iter: int = 32) -> torch.Tensor:
        """Griffin-Lim algorithm for phase reconstruction"""
        angles = torch.randn_like(spec)
        spec_complex = spec * torch.exp(1j * angles)

        for _ in range(n_iter):
            audio = torch.istft(
                spec_complex,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=self.center
            )
            spec_complex = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                center=self.center,
                return_complex=True
            )
            angles = torch.angle(spec_complex)
            spec_complex = spec * torch.exp(1j * angles)

        audio = torch.istft(
            spec_complex,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center
        )

        return audio

    def normalize(self, mel: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Normalize mel-spectrogram"""
        return (mel - mean) / (std + 1e-8)

    def denormalize(self, mel: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Denormalize mel-spectrogram"""
        return mel * std + mean

    def compute_stats(self, mel_list: list) -> Tuple[float, float]:
        """Compute mean and std from list of mel spectrograms"""
        all_mels = torch.cat([m.flatten() for m in mel_list])
        return all_mels.mean().item(), all_mels.std().item()