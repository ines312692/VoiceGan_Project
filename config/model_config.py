"""
Configuration du modèle VoiceGAN-Transformation
"""
import torch


class ModelConfig:
    """Configuration pour le modèle VoiceGAN"""

    # Audio parameters
    SAMPLE_RATE = 22050
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 80
    MEL_FMIN = 0
    MEL_FMAX = 8000

    # Model dimensions
    CONTENT_DIM = 256
    STYLE_DIM = 128
    HIDDEN_DIM = 512

    # Content Encoder (CNN + Transformer)
    CONTENT_CNN_CHANNELS = [80, 128, 256]
    CONTENT_TRANSFORMER_LAYERS = 4
    CONTENT_TRANSFORMER_HEADS = 8
    CONTENT_DROPOUT = 0.1

    # Style Encoder (CNN)
    STYLE_CNN_CHANNELS = [80, 128, 256, 512]
    STYLE_KERNEL_SIZE = 5
    STYLE_POOLING = "adaptive"

    # Generator
    GENERATOR_CHANNELS = [512, 256, 128, 80]
    GENERATOR_KERNEL_SIZE = 5
    GENERATOR_UPSAMPLE_RATES = [2, 2, 2, 2]

    # Discriminator
    DISC_CHANNELS = [80, 128, 256, 512, 1]
    DISC_KERNEL_SIZE = 4
    DISC_STRIDE = 2

    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE_G = 0.0002
    LEARNING_RATE_D = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999

    # Loss weights
    LAMBDA_RECON = 10.0
    LAMBDA_IDENTITY = 5.0
    LAMBDA_CONTENT = 2.0
    LAMBDA_ADV = 1.0

    # Training settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SAVE_INTERVAL = 5
    LOG_INTERVAL = 10

    # Data augmentation
    USE_AUGMENTATION = True
    PITCH_SHIFT_RANGE = (-2, 2)
    TIME_STRETCH_RANGE = (0.9, 1.1)

    # Vocoder
    VOCODER_TYPE = "hifigan"  # or "melgan"
    VOCODER_CHECKPOINT = "checkpoints/vocoder/hifigan.pth"

    @classmethod
    def to_dict(cls):
        """Convertit la config en dictionnaire"""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('_') and not callable(v)}

    @classmethod
    def update(cls, config_dict):
        """Met à jour la configuration depuis un dictionnaire"""
        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


class TrainingConfig:
    """Configuration spécifique à l'entraînement"""

    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    OUTPUT_DIR = "outputs"

    # Early stopping
    PATIENCE = 10
    MIN_DELTA = 0.001

    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"  # "step", "cosine", "plateau"
    SCHEDULER_STEP_SIZE = 30
    SCHEDULER_GAMMA = 0.5

    # Gradient clipping
    GRAD_CLIP_VALUE = 1.0

    # Mixed precision training
    USE_AMP = True

    # Distributed training
    DISTRIBUTED = False
    WORLD_SIZE = 1


class DataConfig:
    """Configuration des données"""

    # Dataset paths
    DATASET_NAME = "VCTK"  # or "LibriTTS"
    DATA_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"

    # Train/Val/Test split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1

    # Audio processing
    MIN_AUDIO_LENGTH = 1.0  # seconds
    MAX_AUDIO_LENGTH = 10.0  # seconds
    TRIM_SILENCE = True
    NORMALIZE = True

    # Spectrogram processing
    SPEC_NORMALIZE = "instance"  # "instance", "batch", "none"

    @classmethod
    def get_speaker_pairs(cls):
        """Retourne les paires de locuteurs pour l'entraînement"""
        return [
            ("p225", "p226"),  # Female to Female
            ("p226", "p225"),  # Female to Female
            ("p227", "p228"),  # Male to Male
            ("p228", "p227"),  # Male to Male
        ]