import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: int = 0
    fmax: int = 8000
    segment_length: int = 16384

@dataclass
class ContentEncoderConfig:
    channels: List[int] = None
    kernel_sizes: List[int] = None
    strides: List[int] = None
    transformer_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [64, 128, 256, 512]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3, 3]
        if self.strides is None:
            self.strides = [1, 2, 2, 1]

@dataclass
class StyleEncoderConfig:
    channels: List[int] = None
    kernel_sizes: List[int] = None
    strides: List[int] = None
    style_dim: int = 256
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [64, 128, 256, 512]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3, 3]
        if self.strides is None:
            self.strides = [2, 2, 2, 2]

@dataclass
class GeneratorConfig:
    input_dim: int = 768
    channels: List[int] = None
    kernel_sizes: List[int] = None
    upsample_rates: List[int] = None
    output_channels: int = 80
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [512, 256, 128, 64]
        if self.kernel_sizes is None:
            self.kernel_sizes = [3, 3, 3, 3]
        if self.upsample_rates is None:
            self.upsample_rates = [2, 2, 2, 1]

@dataclass
class DiscriminatorConfig:
    channels: List[int] = None
    kernel_sizes: List[int] = None
    strides: List[int] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [64, 128, 256, 512, 1024]
        if self.kernel_sizes is None:
            self.kernel_sizes = [4, 4, 4, 4, 4]
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 1]

@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 200
    learning_rate_g: float = 0.0002
    learning_rate_d: float = 0.0001
    beta1: float = 0.5
    beta2: float = 0.999
    grad_clip: float = 1.0
    lambda_reconstruction: float = 10.0
    lambda_adversarial: float = 1.0
    lambda_identity: float = 5.0
    lambda_content: float = 1.0
    lambda_feature_matching: float = 10.0
    discriminator_start_epoch: int = 5
    save_every: int = 10
    log_every: int = 100
    eval_every: int = 1000

class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.audio = AudioConfig(**config_dict['audio'])
        self.content_encoder = ContentEncoderConfig(**config_dict['model']['content_encoder'])
        self.style_encoder = StyleEncoderConfig(**config_dict['model']['style_encoder'])
        self.generator = GeneratorConfig(**config_dict['model']['generator'])
        self.discriminator = DiscriminatorConfig(**config_dict['model']['discriminator'])
        self.training = TrainingConfig(**config_dict['training'])
        
        self.data = config_dict['data']
        self.logging = config_dict['logging']
        self.evaluation = config_dict['evaluation']
    
    def __repr__(self):
        return f"Config(audio={self.audio}, training={self.training})"