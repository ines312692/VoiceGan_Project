import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import random

from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.mel_spectrogram import MelSpectrogramProcessor

class VoiceDataset(Dataset):
    """Dataset for voice conversion training"""
    
    def __init__(
        self,
        data_dir: str,
        audio_config: dict,
        segment_length: int = 16384,
        split: str = 'train',
        cache_mels: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.segment_length = segment_length
        self.cache_mels = cache_mels
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=audio_config['sample_rate'],
            segment_length=segment_length
        )
        
        self.mel_processor = MelSpectrogramProcessor(
            sample_rate=audio_config['sample_rate'],
            n_fft=audio_config['n_fft'],
            hop_length=audio_config['hop_length'],
            win_length=audio_config['win_length'],
            n_mels=audio_config['n_mels'],
            fmin=audio_config['fmin'],
            fmax=audio_config['fmax']
        )
        
        # Load file paths organized by speaker
        self.speakers = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.speaker_to_files = {}
        
        for speaker_dir in self.speakers:
            audio_files = list(speaker_dir.glob('*.wav')) + list(speaker_dir.glob('*.flac'))
            if audio_files:
                self.speaker_to_files[speaker_dir.name] = audio_files
        
        # Create pairs of (source_file, target_speaker)
        self.pairs = []
        for speaker, files in self.speaker_to_files.items():
            for file in files:
                # For each file, can convert to any other speaker
                target_speakers = [s for s in self.speaker_to_files.keys() if s != speaker]
                for target_speaker in target_speakers:
                    self.pairs.append((file, speaker, target_speaker))
        
        # Cache for mel-spectrograms
        self.mel_cache = {} if cache_mels else None
        
        print(f"Loaded {len(self.pairs)} pairs from {len(self.speaker_to_files)} speakers")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            source_mel: Mel-spectrogram of source audio
            target_mel: Mel-spectrogram of target speaker reference
        """
        source_file, source_speaker, target_speaker = self.pairs[idx]
        
        # Get source mel
        source_mel = self._load_mel(source_file)
        
        # Get random target file from target speaker
        target_files = self.speaker_to_files[target_speaker]
        target_file = random.choice(target_files)
        target_mel = self._load_mel(target_file)
        
        # Ensure both have same time dimension (for simplicity, use minimum)
        min_time = min(source_mel.size(1), target_mel.size(1))
        source_mel = source_mel[:, :min_time]
        target_mel = target_mel[:, :min_time]
        
        return source_mel, target_mel
    
    def _load_mel(self, audio_path: Path) -> torch.Tensor:
        """Load audio and convert to mel-spectrogram"""
        # Check cache
        if self.mel_cache is not None and str(audio_path) in self.mel_cache:
            return self.mel_cache[str(audio_path)].clone()
        
        # Load audio
        audio = self.audio_processor.load_audio(audio_path)
        
        # Segment audio
        audio = self.audio_processor.segment_audio(audio, random=True)
        
        # Convert to mel
        mel = self.mel_processor.wav_to_mel(audio)
        
        # Cache if enabled
        if self.mel_cache is not None:
            self.mel_cache[str(audio_path)] = mel.clone()
        
        return mel
    
    def get_speaker_list(self) -> List[str]:
        """Get list of all speakers"""
        return list(self.speaker_to_files.keys())

class VoiceCollator:
    """Collate function for batching"""
    
    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate batch of (source_mel, target_mel) pairs
        """
        source_mels, target_mels = zip(*batch)
        
        # Pad to same length
        source_mels = self._pad_sequence(source_mels)
        target_mels = self._pad_sequence(target_mels)
        
        return source_mels, target_mels
    
    def _pad_sequence(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to same length"""
        max_len = max(s.size(1) for s in sequences)
        n_mels = sequences[0].size(0)
        
        padded = torch.full(
            (len(sequences), n_mels, max_len),
            self.pad_value,
            dtype=sequences[0].dtype
        )
        
        for i, seq in enumerate(sequences):
            padded[i, :, :seq.size(1)] = seq
        
        return padded