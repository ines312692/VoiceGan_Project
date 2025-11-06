"""
Dataset PyTorch pour VoiceGAN
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random
import json


class VoiceConversionDataset(Dataset):
    """
    Dataset pour la conversion de voix A→B

    Structure attendue:
        data/processed/
            speaker_A/
                file1.npy
                file2.npy
                ...
            speaker_B/
                file1.npy
                file2.npy
                ...
    """

    def __init__(self, data_dir, speaker_pairs, split='train',
                 min_length=None, max_length=None, transform=None):
        """
        Args:
            data_dir: Répertoire racine des données
            speaker_pairs: Liste de tuples (speaker_A, speaker_B)
            split: 'train', 'val', ou 'test'
            min_length: Longueur minimale des spectrogrammes
            max_length: Longueur maximale des spectrogrammes
            transform: Transformations à appliquer
        """
        self.data_dir = Path(data_dir)
        self.speaker_pairs = speaker_pairs
        self.split = split
        self.min_length = min_length
        self.max_length = max_length
        self.transform = transform

        # Charger les fichiers
        self.samples = self._load_samples()

        print(f"Loaded {len(self.samples)} samples for {split} set")

    def _load_samples(self):
        """Charge la liste des paires de fichiers"""
        samples = []

        for speaker_a, speaker_b in self.speaker_pairs:
            # Chemins des répertoires
            dir_a = self.data_dir / speaker_a
            dir_b = self.data_dir / speaker_b

            if not dir_a.exists() or not dir_b.exists():
                print(f"Warning: Directory not found for pair ({speaker_a}, {speaker_b})")
                continue

            # Lister les fichiers
            files_a = sorted(list(dir_a.glob('*.npy')))
            files_b = sorted(list(dir_b.glob('*.npy')))

            # Créer les paires
            for file_a in files_a:
                # Choisir un fichier aléatoire de B pour chaque fichier de A
                for file_b in files_b:
                    samples.append({
                        'source': str(file_a),
                        'target': str(file_b),
                        'speaker_a': speaker_a,
                        'speaker_b': speaker_b
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retourne un échantillon

        Returns:
            dict avec 'source_mel', 'target_mel', et métadonnées
        """
        sample = self.samples[idx]

        # Charger les melspectrograms
        source_mel = np.load(sample['source'])  # (n_mels, time)
        target_mel = np.load(sample['target'])  # (n_mels, time)

        # Filtrer par longueur si nécessaire
        if self.min_length is not None:
            if source_mel.shape[1] < self.min_length:
                # Pad si trop court
                pad_width = self.min_length - source_mel.shape[1]
                source_mel = np.pad(source_mel, ((0, 0), (0, pad_width)), mode='constant')

            if target_mel.shape[1] < self.min_length:
                pad_width = self.min_length - target_mel.shape[1]
                target_mel = np.pad(target_mel, ((0, 0), (0, pad_width)), mode='constant')

        if self.max_length is not None:
            # Tronquer si trop long
            if source_mel.shape[1] > self.max_length:
                start = random.randint(0, source_mel.shape[1] - self.max_length)
                source_mel = source_mel[:, start:start + self.max_length]

            if target_mel.shape[1] > self.max_length:
                start = random.randint(0, target_mel.shape[1] - self.max_length)
                target_mel = target_mel[:, start:start + self.max_length]

        # Aligner les longueurs (prendre le minimum)
        min_len = min(source_mel.shape[1], target_mel.shape[1])
        source_mel = source_mel[:, :min_len]
        target_mel = target_mel[:, :min_len]

        # Appliquer les transformations
        if self.transform is not None:
            source_mel = self.transform(source_mel)
            target_mel = self.transform(target_mel)

        # Convertir en tenseurs
        source_mel = torch.from_numpy(source_mel).float()
        target_mel = torch.from_numpy(target_mel).float()

        return {
            'source_mel': source_mel,
            'target_mel': target_mel,
            'speaker_a': sample['speaker_a'],
            'speaker_b': sample['speaker_b'],
            'source_path': sample['source'],
            'target_path': sample['target']
        }


class RandomPairDataset(Dataset):
    """
    Dataset avec paires aléatoires de locuteurs
    Plus flexible pour l'entraînement
    """

    def __init__(self, data_dir, speakers, split='train',
                 fixed_length=200, samples_per_epoch=10000):
        """
        Args:
            data_dir: Répertoire racine des données
            speakers: Liste de locuteurs disponibles
            split: 'train', 'val', ou 'test'
            fixed_length: Longueur fixe des spectrogrammes
            samples_per_epoch: Nombre d'échantillons par epoch
        """
        self.data_dir = Path(data_dir)
        self.speakers = speakers
        self.split = split
        self.fixed_length = fixed_length
        self.samples_per_epoch = samples_per_epoch

        # Charger tous les fichiers par locuteur
        self.speaker_files = self._load_speaker_files()

        print(f"Loaded dataset with {len(self.speakers)} speakers")
        for speaker, files in self.speaker_files.items():
            print(f"  {speaker}: {len(files)} files")

    def _load_speaker_files(self):
        """Charge les fichiers pour chaque locuteur"""
        speaker_files = {}

        for speaker in self.speakers:
            speaker_dir = self.data_dir / speaker
            if speaker_dir.exists():
                files = sorted(list(speaker_dir.glob('*.npy')))
                speaker_files[speaker] = files

        return speaker_files

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        """
        Génère une paire aléatoire

        Returns:
            dict avec 'source_mel' et 'target_mel'
        """
        # Choisir aléatoirement deux locuteurs différents
        speaker_a, speaker_b = random.sample(self.speakers, 2)

        # Choisir aléatoirement un fichier pour chaque locuteur
        file_a = random.choice(self.speaker_files[speaker_a])
        file_b = random.choice(self.speaker_files[speaker_b])

        # Charger les melspectrograms
        mel_a = np.load(file_a)
        mel_b = np.load(file_b)

        # Extraire des segments de longueur fixe
        mel_a = self._extract_segment(mel_a, self.fixed_length)
        mel_b = self._extract_segment(mel_b, self.fixed_length)

        # Convertir en tenseurs
        source_mel = torch.from_numpy(mel_a).float()
        target_mel = torch.from_numpy(mel_b).float()

        return {
            'source_mel': source_mel,
            'target_mel': target_mel,
            'speaker_a': speaker_a,
            'speaker_b': speaker_b
        }

    def _extract_segment(self, mel, length):
        """Extrait un segment de longueur fixe"""
        if mel.shape[1] >= length:
            # Choisir aléatoirement un segment
            start = random.randint(0, mel.shape[1] - length)
            segment = mel[:, start:start + length]
        else:
            # Pad si trop court
            pad_width = length - mel.shape[1]
            segment = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')

        return segment


def create_dataloaders(config, data_config):
    """
    Crée les dataloaders pour train/val/test

    Args:
        config: ModelConfig
        data_config: DataConfig

    Returns:
        train_loader, val_loader, test_loader
    """
    # Obtenir les paires de locuteurs
    speaker_pairs = data_config.get_speaker_pairs()

    # Séparer en train/val/test
    n_pairs = len(speaker_pairs)
    n_train = int(n_pairs * data_config.TRAIN_SPLIT)
    n_val = int(n_pairs * data_config.VAL_SPLIT)

    train_pairs = speaker_pairs[:n_train]
    val_pairs = speaker_pairs[n_train:n_train + n_val]
    test_pairs = speaker_pairs[n_train + n_val:]

    # Créer les datasets
    train_dataset = VoiceConversionDataset(
        data_config.PROCESSED_DIR,
        train_pairs,
        split='train',
        max_length=int(data_config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE / config.HOP_LENGTH)
    )

    val_dataset = VoiceConversionDataset(
        data_config.PROCESSED_DIR,
        val_pairs,
        split='val',
        max_length=int(data_config.MAX_AUDIO_LENGTH * config.SAMPLE_RATE / config.HOP_LENGTH)
    )

    test_dataset = VoiceConversionDataset(
        data_config.PROCESSED_DIR,
        test_pairs,
        split='test'
    )

    # Créer les dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    from config.model_config import ModelConfig, DataConfig

    # Test du dataset
    print("Testing VoiceConversionDataset...")

    # Créer un dataset de test
    speaker_pairs = [("p225", "p226"), ("p227", "p228")]

    dataset = VoiceConversionDataset(
        data_dir="data/processed",
        speaker_pairs=speaker_pairs,
        split='train',
        max_length=200
    )

    print(f"Dataset size: {len(dataset)}")

    # Test d'un échantillon
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Source mel shape: {sample['source_mel'].shape}")
        print(f"Target mel shape: {sample['target_mel'].shape}")
        print(f"Speaker A: {sample['speaker_a']}")
        print(f"Speaker B: {sample['speaker_b']}")

    print("\n✓ Dataset test passed!")