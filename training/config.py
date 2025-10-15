"""
Training Configuration for BrainWave Analyzer
Centralized configuration for all training parameters
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import os


@dataclass
class ModelConfig:
    """Configuration for model architectures"""
    
    # EEG to Image Model
    eeg_to_image: Dict = None
    
    # Image to EEG Model  
    image_to_eeg: Dict = None
    
    def __post_init__(self):
        if self.eeg_to_image is None:
            self.eeg_to_image = {
                'time_steps': 100,
                'n_features': 1,
                'image_size': (64, 64),
                'latent_dim': 256,
                'hidden_units': 128,
                'use_attention': True,
                'use_vae': False
            }
        
        if self.image_to_eeg is None:
            self.image_to_eeg = {
                'image_size': (64, 64),
                'time_steps': 100,
                'n_features': 1,
                'latent_dim': 256,
                'hidden_units': 128,
                'use_attention': True,
                'use_teacher_forcing': False
            }


@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    
    # Dataset parameters
    n_samples: int = 1000
    batch_size: int = 32
    train_val_split: float = 0.8
    
    # Training parameters
    epochs: int = 20
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    
    # Loss weights
    eeg_to_image_loss_weights: Dict = None
    image_to_eeg_loss_weights: Dict = None
    
    # Regularization
    dropout_rate: float = 0.2
    weight_decay: float = 1e-5
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_decay_factor: float = 0.5
    lr_patience: int = 5
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = 'val_loss'
    early_stopping_min_delta: float = 1e-4
    
    # Model checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    monitor_checkpoint: str = 'val_loss'
    mode_checkpoint: str = 'min'
    
    # Logging
    log_dir: str = 'logs'
    log_frequency: int = 10
    
    def __post_init__(self):
        if self.eeg_to_image_loss_weights is None:
            self.eeg_to_image_loss_weights = {
                'reconstruction': 1.0,
                'perceptual': 0.1,
                'kl_divergence': 0.01
            }
        
        if self.image_to_eeg_loss_weights is None:
            self.image_to_eeg_loss_weights = {
                'reconstruction': 1.0,
                'dtw': 0.1,
                'frequency': 0.05
            }


@dataclass
class DataConfig:
    """Configuration for data processing"""
    
    # Dataset parameters
    data_dir: str = 'data'
    dataset_filename: str = 'brainwave_dataset.pkl'
    
    # EEG parameters
    sample_rate: int = 250
    duration: float = 4.0
    n_channels: int = 1
    
    # Image parameters
    image_size: Tuple[int, int] = (64, 64)
    n_colors: int = 3
    
    # Data augmentation
    use_augmentation: bool = True
    eeg_noise_factor: float = 0.1
    image_variation: float = 0.3
    
    # Preprocessing
    normalize_eeg: bool = True
    normalize_images: bool = True
    eeg_normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    image_normalization_method: str = 'minmax'  # 'standard', 'minmax'


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # Metrics to compute
    compute_image_metrics: bool = True
    compute_eeg_metrics: bool = True
    
    # Image quality metrics
    image_metrics: List[str] = None
    
    # EEG quality metrics
    eeg_metrics: List[str] = None
    
    # Visualization
    save_predictions: bool = True
    n_visualization_samples: int = 10
    visualization_dir: str = 'visualizations'
    
    def __post_init__(self):
        if self.image_metrics is None:
            self.image_metrics = ['mse', 'mae', 'psnr', 'ssim']
        
        if self.eeg_metrics is None:
            self.eeg_metrics = ['mse', 'mae', 'dtw', 'frequency_correlation']


class Config:
    """Main configuration class combining all configs"""
    
    def __init__(self, config_dict: Dict = None):
        """
        Initialize configuration
        
        Args:
            config_dict: Optional dictionary to override default configs
        """
        # Initialize default configs
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        
        # Override with provided config
        if config_dict:
            self.update_from_dict(config_dict)
    
    def update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary"""
        return {
            'model': {
                'eeg_to_image': self.model.eeg_to_image,
                'image_to_eeg': self.model.image_to_eeg
            },
            'training': {
                'n_samples': self.training.n_samples,
                'batch_size': self.training.batch_size,
                'epochs': self.training.epochs,
                'learning_rate': self.training.learning_rate,
                'optimizer': self.training.optimizer,
                'eeg_to_image_loss_weights': self.training.eeg_to_image_loss_weights,
                'image_to_eeg_loss_weights': self.training.image_to_eeg_loss_weights
            },
            'data': {
                'sample_rate': self.data.sample_rate,
                'duration': self.data.duration,
                'image_size': self.data.image_size,
                'use_augmentation': self.data.use_augmentation
            },
            'evaluation': {
                'image_metrics': self.evaluation.image_metrics,
                'eeg_metrics': self.evaluation.eeg_metrics
            }
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        import json
        
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str):
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.training.checkpoint_dir,
            self.training.log_dir,
            self.data.data_dir,
            self.evaluation.visualization_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Predefined configurations for different scenarios
def get_demo_config() -> Config:
    """Get configuration for demo training (fast, small dataset)"""
    config = Config()
    
    # Small dataset for quick demo
    config.training.n_samples = 200
    config.training.epochs = 10
    config.training.batch_size = 16
    
    # Simplified models for speed
    config.model.eeg_to_image['latent_dim'] = 128
    config.model.image_to_eeg['latent_dim'] = 128
    config.model.eeg_to_image['hidden_units'] = 64
    config.model.image_to_eeg['hidden_units'] = 64
    
    # Disable some features for speed
    config.model.eeg_to_image['use_vae'] = False
    config.training.use_lr_scheduler = False
    config.training.use_early_stopping = False
    
    return config


def get_full_config() -> Config:
    """Get configuration for full training (comprehensive, large dataset)"""
    config = Config()
    
    # Large dataset for full training
    config.training.n_samples = 5000
    config.training.epochs = 50
    config.training.batch_size = 32
    
    # Full model capabilities
    config.model.eeg_to_image['use_vae'] = True
    config.training.use_lr_scheduler = True
    config.training.use_early_stopping = True
    
    # Enhanced loss weights
    config.training.eeg_to_image_loss_weights.update({
        'perceptual': 0.2,
        'kl_divergence': 0.05
    })
    
    config.training.image_to_eeg_loss_weights.update({
        'dtw': 0.2,
        'frequency': 0.1
    })
    
    return config


def get_attention_config() -> Config:
    """Get configuration focusing on attention mechanisms"""
    config = Config()
    
    # Medium dataset
    config.training.n_samples = 1000
    config.training.epochs = 30
    
    # Enhanced attention models
    config.model.eeg_to_image['use_attention'] = True
    config.model.image_to_eeg['use_attention'] = True
    
    # Larger latent space for attention
    config.model.eeg_to_image['latent_dim'] = 512
    config.model.image_to_eeg['latent_dim'] = 512
    
    return config


if __name__ == "__main__":
    # Demo usage
    print("Testing configuration system...")
    
    # Test default config
    config = Config()
    print("Default config created successfully")
    print(f"Training epochs: {config.training.epochs}")
    print(f"Model latent dim: {config.model.eeg_to_image['latent_dim']}")
    
    # Test demo config
    demo_config = get_demo_config()
    print(f"Demo config epochs: {demo_config.training.epochs}")
    print(f"Demo config samples: {demo_config.training.n_samples}")
    
    # Test config saving/loading
    demo_config.save_config('demo_config.json')
    loaded_config = Config.load_config('demo_config.json')
    print(f"Loaded config epochs: {loaded_config.training.epochs}")
    
    # Create directories
    demo_config.create_directories()
    print("Directories created successfully")
    
    print("Configuration system test completed!")
