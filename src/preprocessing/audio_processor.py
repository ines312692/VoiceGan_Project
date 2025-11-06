"""
Prétraitement audio et génération de melspectrogrammes
"""
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path


class AudioProcessor:
    """
    Classe pour le prétraitement audio
    - Chargement des fichiers audio
    - Normalisation
    - Suppression du silence
    - Génération de melspectrogrammes
    """

    def __init__(self, config):
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.n_fft = config.N_FFT
        self.hop_length = config.HOP_LENGTH
        self.win_length = config.WIN_LENGTH
        self.n_mels = config.N_MELS
        self.mel_fmin = config.MEL_FMIN
        self.mel_fmax = config.MEL_FMAX

        # Mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax
        )

    def load_audio(self, audio_path, normalize=True, trim_silence=True):
        """
        Charge un fichier audio

        Args:
            audio_path: Chemin vers le fichier audio
            normalize: Normaliser l'amplitude
            trim_silence: Supprimer les silences au début/fin

        Returns:
            audio: np.array (samples,)
            sr: Sample rate
        """
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        # Trim silence
        if trim_silence:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=20,
                frame_length=self.win_length,
                hop_length=self.hop_length
            )

        # Normalisation
        if normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            audio = audio * 0.95  # Éviter le clipping

        return audio, sr

    def audio_to_mel(self, audio):
        """
        Convertit un signal audio en melspectrogram

        Args:
            audio: np.array (samples,)

        Returns:
            mel: np.array (n_mels, time)
        """
        # STFT
        D = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            pad_mode='reflect'
        )

        # Magnitude
        S = np.abs(D)

        # Mel scale
        mel = np.dot(self.mel_basis, S)

        # Log scale
        mel = np.log(np.maximum(mel, 1e-5))

        return mel

    def mel_to_audio(self, mel, vocoder=None):
        """
        Convertit un melspectrogram en audio (nécessite un vocoder)

        Args:
            mel: np.array (n_mels, time) ou torch.Tensor
            vocoder: Modèle vocoder (HiFi-GAN, MelGAN, etc.)

        Returns:
            audio: np.array (samples,)
        """
        if vocoder is None:
            # Utiliser Griffin-Lim si pas de vocoder
            # Inverse mel scale
            mel_exp = np.exp(mel)
            S = np.dot(self.mel_basis.T, mel_exp)

            # Griffin-Lim
            audio = librosa.griffinlim(
                S,
                n_iter=32,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
        else:
            # Utiliser le vocoder neural
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()

            with torch.no_grad():
                audio = vocoder(mel.unsqueeze(0))
                audio = audio.squeeze().cpu().numpy()

        return audio

    def normalize_mel(self, mel, method='instance'):
        """
        Normalise le melspectrogram

        Args:
            mel: np.array (n_mels, time)
            method: 'instance', 'global', 'none'

        Returns:
            mel_norm: np.array (n_mels, time)
        """
        if method == 'instance':
            # Instance normalization
            mean = np.mean(mel)
            std = np.std(mel)
            mel_norm = (mel - mean) / (std + 1e-8)

        elif method == 'global':
            # Global normalization (basé sur des stats du dataset)
            # Ces valeurs devraient être calculées sur le dataset
            mean = -5.0
            std = 2.0
            mel_norm = (mel - mean) / (std + 1e-8)

        else:
            mel_norm = mel

        return mel_norm

    def denormalize_mel(self, mel_norm, method='instance', stats=None):
        """
        Dénormalise le melspectrogram

        Args:
            mel_norm: np.array (n_mels, time)
            method: 'instance', 'global', 'none'
            stats: dict avec 'mean' et 'std' si nécessaire

        Returns:
            mel: np.array (n_mels, time)
        """
        if method == 'instance' and stats is not None:
            mel = mel_norm * stats['std'] + stats['mean']
        elif method == 'global':
            mean = -5.0
            std = 2.0
            mel = mel_norm * std + mean
        else:
            mel = mel_norm

        return mel

    def process_audio_file(self, audio_path, output_path=None):
        """
        Pipeline complet: audio → melspectrogram → sauvegarde

        Args:
            audio_path: Chemin vers le fichier audio
            output_path: Chemin de sortie (optionnel)

        Returns:
            mel: np.array (n_mels, time)
        """
        # Charger l'audio
        audio, sr = self.load_audio(audio_path)

        # Générer le melspectrogram
        mel = self.audio_to_mel(audio)

        # Normaliser
        mel = self.normalize_mel(mel, method='instance')

        # Sauvegarder si demandé
        if output_path is not None:
            np.save(output_path, mel)

        return mel

    def batch_process_directory(self, input_dir, output_dir):
        """
        Traite tous les fichiers audio d'un répertoire

        Args:
            input_dir: Répertoire contenant les fichiers audio
            output_dir: Répertoire de sortie pour les melspectrograms
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extensions audio supportées
        audio_extensions = ['.wav', '.flac', '.mp3', '.m4a']

        # Trouver tous les fichiers audio
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_dir.rglob(f'*{ext}'))

        print(f"Found {len(audio_files)} audio files")

        # Traiter chaque fichier
        for i, audio_path in enumerate(audio_files):
            try:
                # Chemin de sortie
                rel_path = audio_path.relative_to(input_dir)
                output_path = output_dir / rel_path.with_suffix('.npy')
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Traiter
                self.process_audio_file(audio_path, output_path)

                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(audio_files)} files")

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        print(f"Preprocessing complete! Saved to {output_dir}")


def compute_dataset_statistics(mel_dir, output_path='dataset_stats.npy'):
    """
    Calcule les statistiques (mean, std) sur tout le dataset
    Pour la normalisation globale

    Args:
        mel_dir: Répertoire contenant les melspectrograms .npy
        output_path: Chemin de sortie pour les stats
    """
    mel_dir = Path(mel_dir)
    mel_files = list(mel_dir.rglob('*.npy'))

    print(f"Computing statistics on {len(mel_files)} files...")

    all_values = []

    for i, mel_path in enumerate(mel_files):
        try:
            mel = np.load(mel_path)
            all_values.append(mel.flatten())

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(mel_files)} files")
        except Exception as e:
            print(f"Error loading {mel_path}: {e}")

    # Concaténer toutes les valeurs
    all_values = np.concatenate(all_values)

    # Calculer les statistiques
    mean = np.mean(all_values)
    std = np.std(all_values)

    stats = {
        'mean': mean,
        'std': std,
        'min': np.min(all_values),
        'max': np.max(all_values)
    }

    # Sauvegarder
    np.save(output_path, stats)

    print(f"\nDataset Statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print(f"Saved to {output_path}")

    return stats


if __name__ == "__main__":
    from config.model_config import ModelConfig

    # Test de l'AudioProcessor
    processor = AudioProcessor(ModelConfig)

    print("AudioProcessor initialized")
    print(f"Sample rate: {processor.sample_rate}")
    print(f"N_mels: {processor.n_mels}")
    print(f"Hop length: {processor.hop_length}")

    # Pour tester avec un fichier réel, décommenter:
    # audio_path = "path/to/audio.wav"
    # mel = processor.process_audio_file(audio_path)
    # print(f"Melspectrogram shape: {mel.shape}")

    print("\n✓ AudioProcessor test passed!")