import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


class AudioProcessor:
    """Audio loading, processing and saving utilities"""

    def __init__(
            self,
            sample_rate: int = 22050,
            segment_length: int = 16384,
            normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.normalize = normalize

    def load_audio(
            self,
            path: Union[str, Path],
            offset: float = 0.0,
            duration: Optional[float] = None
    ) -> torch.Tensor:
        """
        Load audio file
        Args:
            path: Path to audio file
            offset: Start reading after this time (in seconds)
            duration: Only load up to this much audio (in seconds)
        Returns:
            audio: (samples,) torch tensor
        """
        audio, sr = torchaudio.load(str(path))

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.squeeze(0)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Apply offset and duration
        if offset > 0 or duration is not None:
            start_sample = int(offset * self.sample_rate)
            end_sample = int((offset + duration) * self.sample_rate) if duration else len(audio)
            audio = audio[start_sample:end_sample]

        # Normalize
        if self.normalize:
            audio = self._normalize_audio(audio)

        return audio

    def save_audio(
            self,
            audio: torch.Tensor,
            path: Union[str, Path],
            sample_rate: Optional[int] = None
    ):
        """Save audio to file"""
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Ensure audio is in correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Denormalize if needed
        audio = torch.clamp(audio, -1.0, 1.0)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(path), audio, sample_rate)

    def segment_audio(
            self,
            audio: torch.Tensor,
            segment_length: Optional[int] = None,
            random: bool = True
    ) -> torch.Tensor:
        """
        Extract segment from audio
        Args:
            audio: (samples,)
            segment_length: Length of segment to extract
            random: If True, extract random segment; otherwise from start
        Returns:
            segment: (segment_length,)
        """
        if segment_length is None:
            segment_length = self.segment_length

        if len(audio) < segment_length:
            # Pad if audio is too short
            padding = segment_length - len(audio)
            audio = torch.nn.functional.pad(audio, (0, padding))
        elif len(audio) > segment_length:
            # Extract segment
            if random:
                max_start = len(audio) - segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = 0
            audio = audio[start:start + segment_length]

        return audio

    def trim_silence(
            self,
            audio: torch.Tensor,
            top_db: float = 30.0,
            frame_length: int = 2048,
            hop_length: int = 512
    ) -> torch.Tensor:
        """Remove leading and trailing silence"""
        audio_np = audio.numpy()
        non_silent_intervals = librosa.effects.split(
            audio_np,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )

        if len(non_silent_intervals) == 0:
            return audio

        start = non_silent_intervals[0][0]
        end = non_silent_intervals[-1][1]

        return audio[start:end]

    def apply_preemphasis(
            self,
            audio: torch.Tensor,
            coef: float = 0.97
    ) -> torch.Tensor:
        """Apply pre-emphasis filter"""
        return torch.cat([
            audio[:1],
            audio[1:] - coef * audio[:-1]
        ])

    def remove_preemphasis(
            self,
            audio: torch.Tensor,
            coef: float = 0.97
    ) -> torch.Tensor:
        """Remove pre-emphasis filter"""
        result = torch.zeros_like(audio)
        result[0] = audio[0]
        for i in range(1, len(audio)):
            result[i] = audio[i] + coef * result[i - 1]
        return result

    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1]"""
        max_val = audio.abs().max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    def compute_rms(self, audio: torch.Tensor) -> float:
        """Compute RMS energy"""
        return torch.sqrt(torch.mean(audio ** 2)).item()

    def adjust_volume(
            self,
            audio: torch.Tensor,
            target_rms: float
    ) -> torch.Tensor:
        """Adjust audio volume to target RMS"""
        current_rms = self.compute_rms(audio)
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        return audio