"""
Dataset Creation and Management for BrainWave Analyzer
Creates paired EEG-Image datasets for training
"""

import numpy as np
import pickle
import os
from typing import Tuple, Dict, Optional
from .synthetic_generator import create_synthetic_eeg_dataset, BrainState
from .image_generator import create_synthetic_image_dataset


class BrainWaveDataset:
    """Complete dataset for EEG-Image mapping"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 sample_rate: int = 250,
                 duration: float = 4.0,
                 image_size: Tuple[int, int] = (64, 64),
                 n_channels: int = 1):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory to save/load data
            sample_rate: EEG sampling rate
            duration: EEG duration in seconds
            image_size: Image dimensions (height, width)
            n_channels: Number of EEG channels
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.image_size = image_size
        self.n_channels = n_channels
        self.time_points = int(sample_rate * duration)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset storage
        self.train_eeg = None
        self.train_images = None
        self.train_labels = None
        self.val_eeg = None
        self.val_images = None
        self.val_labels = None
        self.metadata = None
    
    def generate_dataset(self, 
                        n_samples: int = 1000,
                        save: bool = True) -> Dict:
        """
        Generate complete paired EEG-Image dataset
        
        Args:
            n_samples: Total number of samples
            save: Whether to save dataset to disk
            
        Returns:
            Dataset dictionary
        """
        print(f"Generating {n_samples} paired EEG-Image samples...")
        
        # Generate EEG data
        print("Generating EEG data...")
        eeg_dataset = create_synthetic_eeg_dataset(
            n_samples=n_samples,
            sample_rate=self.sample_rate,
            duration=self.duration
        )
        
        # Extract labels for image generation
        train_labels = eeg_dataset['train_labels']
        val_labels = eeg_dataset['val_labels']
        
        # Generate corresponding images
        print("Generating corresponding images...")
        train_images = create_synthetic_image_dataset(
            len(train_labels), 
            train_labels, 
            self.image_size
        )
        
        val_images = create_synthetic_image_dataset(
            len(val_labels), 
            val_labels, 
            self.image_size
        )
        
        # Store dataset
        self.train_eeg = eeg_dataset['train_eeg']
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_eeg = eeg_dataset['val_eeg']
        self.val_images = val_images
        self.val_labels = val_labels
        
        # Create metadata
        self.metadata = {
            'n_train': len(train_labels),
            'n_val': len(val_labels),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'time_points': self.time_points,
            'image_size': self.image_size,
            'n_channels': self.n_channels,
            'brain_states': [state.value for state in BrainState],
            'dataset_version': '1.0'
        }
        
        if save:
            self.save_dataset()
        
        return self.get_dataset_dict()
    
    def save_dataset(self, filename: str = "brainwave_dataset.pkl"):
        """Save dataset to disk"""
        if self.train_eeg is None:
            raise ValueError("No dataset to save. Generate dataset first.")
        
        dataset_dict = self.get_dataset_dict()
        filepath = os.path.join(self.data_dir, filename)
        
        print(f"Saving dataset to {filepath}...")
        with open(filepath, 'wb') as f:
            pickle.dump(dataset_dict, f)
        
        print(f"Dataset saved successfully!")
        print(f"  Train samples: {len(self.train_labels)}")
        print(f"  Val samples: {len(self.val_labels)}")
        print(f"  EEG shape: {self.train_eeg.shape}")
        print(f"  Image shape: {self.train_images.shape}")
    
    def load_dataset(self, filename: str = "brainwave_dataset.pkl") -> Dict:
        """Load dataset from disk"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        print(f"Loading dataset from {filepath}...")
        with open(filepath, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        # Restore dataset attributes
        self.train_eeg = dataset_dict['train_eeg']
        self.train_images = dataset_dict['train_images']
        self.train_labels = dataset_dict['train_labels']
        self.val_eeg = dataset_dict['val_eeg']
        self.val_images = dataset_dict['val_images']
        self.val_labels = dataset_dict['val_labels']
        self.metadata = dataset_dict['metadata']
        
        print(f"Dataset loaded successfully!")
        print(f"  Train samples: {len(self.train_labels)}")
        print(f"  Val samples: {len(self.val_labels)}")
        
        return dataset_dict
    
    def get_dataset_dict(self) -> Dict:
        """Get complete dataset as dictionary"""
        if self.train_eeg is None:
            raise ValueError("No dataset available. Generate or load dataset first.")
        
        return {
            'train_eeg': self.train_eeg,
            'train_images': self.train_images,
            'train_labels': self.train_labels,
            'val_eeg': self.val_eeg,
            'val_images': self.val_images,
            'val_labels': self.val_labels,
            'metadata': self.metadata
        }
    
    def get_data_stats(self) -> Dict:
        """Get dataset statistics"""
        if self.train_eeg is None:
            return {"error": "No dataset loaded"}
        
        stats = {
            'n_train': len(self.train_labels),
            'n_val': len(self.val_labels),
            'eeg_stats': {
                'train_mean': float(np.mean(self.train_eeg)),
                'train_std': float(np.std(self.train_eeg)),
                'train_min': float(np.min(self.train_eeg)),
                'train_max': float(np.max(self.train_eeg)),
                'val_mean': float(np.mean(self.val_eeg)),
                'val_std': float(np.std(self.val_eeg)),
            },
            'image_stats': {
                'train_mean': float(np.mean(self.train_images)),
                'train_std': float(np.std(self.train_images)),
                'train_min': float(np.min(self.train_images)),
                'train_max': float(np.max(self.train_images)),
            },
            'label_distribution': {
                'train': np.bincount(self.train_labels).tolist(),
                'val': np.bincount(self.val_labels).tolist()
            }
        }
        
        return stats
    
    def augment_dataset(self, 
                       eeg_noise_factor: float = 0.1,
                       image_variation: float = 0.3) -> None:
        """
        Augment existing dataset with noise and variations
        
        Args:
            eeg_noise_factor: Amount of noise to add to EEG
            image_variation: Amount of variation to add to images
        """
        if self.train_eeg is None:
            raise ValueError("No dataset to augment. Generate or load dataset first.")
        
        print("Augmenting dataset...")
        
        # Augment EEG with noise
        eeg_noise = eeg_noise_factor * np.random.randn(*self.train_eeg.shape)
        self.train_eeg = np.vstack([self.train_eeg, self.train_eeg + eeg_noise])
        
        # Augment images with variations
        n_train = len(self.train_images)
        augmented_images = []
        
        for i in range(n_train):
            img = self.train_images[i].copy()
            
            # Add small random variations
            variation = image_variation * np.random.randn(*img.shape)
            augmented_img = np.clip(img + variation, 0, 1)
            augmented_images.append(augmented_img)
        
        augmented_images = np.array(augmented_images)
        self.train_images = np.vstack([self.train_images, augmented_images])
        
        # Duplicate labels
        self.train_labels = np.concatenate([self.train_labels, self.train_labels])
        
        print(f"Dataset augmented! New train size: {len(self.train_labels)}")
    
    def create_balanced_dataset(self, 
                               samples_per_class: int = 200) -> None:
        """
        Create balanced dataset with equal samples per class
        
        Args:
            samples_per_class: Number of samples per brain state
        """
        print(f"Creating balanced dataset with {samples_per_class} samples per class...")
        
        n_classes = len(BrainState)
        total_samples = n_classes * samples_per_class
        
        # Generate balanced EEG data
        balanced_eeg = []
        balanced_labels = []
        
        for class_idx in range(n_classes):
            # Generate samples for this class
            class_eeg, class_labels = self._generate_class_samples(
                samples_per_class, 
                class_idx
            )
            balanced_eeg.append(class_eeg)
            balanced_labels.extend([class_idx] * samples_per_class)
        
        balanced_eeg = np.vstack(balanced_eeg)
        balanced_labels = np.array(balanced_labels)
        
        # Generate corresponding images
        balanced_images = create_synthetic_image_dataset(
            total_samples, 
            balanced_labels, 
            self.image_size
        )
        
        # Create train/val split
        n_train = int(0.8 * total_samples)
        train_indices = np.random.choice(total_samples, n_train, replace=False)
        val_indices = np.setdiff1d(np.arange(total_samples), train_indices)
        
        # Store balanced dataset
        self.train_eeg = balanced_eeg[train_indices]
        self.train_images = balanced_images[train_indices]
        self.train_labels = balanced_labels[train_indices]
        self.val_eeg = balanced_eeg[val_indices]
        self.val_images = balanced_images[val_indices]
        self.val_labels = balanced_labels[val_indices]
        
        print(f"Balanced dataset created!")
        print(f"  Train samples: {len(self.train_labels)}")
        print(f"  Val samples: {len(self.val_labels)}")
    
    def _generate_class_samples(self, 
                               n_samples: int, 
                               class_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples for specific class"""
        from .synthetic_generator import EEGGenerator
        
        generator = EEGGenerator(
            sample_rate=self.sample_rate,
            duration=self.duration,
            n_channels=self.n_channels
        )
        
        brain_state = list(BrainState)[class_idx]
        eeg_samples = []
        
        for _ in range(n_samples):
            eeg = generator.generate_brain_state_eeg(brain_state)
            eeg_samples.append(eeg)
        
        return np.array(eeg_samples), np.full(n_samples, class_idx)


def create_demo_dataset(n_samples: int = 500,
                       data_dir: str = "data") -> BrainWaveDataset:
    """
    Create a demo dataset for BrainWave Analyzer
    
    Args:
        n_samples: Number of samples to generate
        data_dir: Directory to save data
        
    Returns:
        BrainWaveDataset instance
    """
    dataset = BrainWaveDataset(data_dir=data_dir)
    dataset.generate_dataset(n_samples=n_samples, save=True)
    return dataset


def load_demo_dataset(data_dir: str = "data") -> BrainWaveDataset:
    """
    Load existing demo dataset
    
    Args:
        data_dir: Directory containing data
        
    Returns:
        BrainWaveDataset instance
    """
    dataset = BrainWaveDataset(data_dir=data_dir)
    dataset.load_dataset()
    return dataset


if __name__ == "__main__":
    # Demo usage
    print("Creating BrainWave dataset...")
    
    # Create dataset
    dataset = create_demo_dataset(n_samples=100)
    
    # Print statistics
    stats = dataset.get_data_stats()
    print("\nDataset Statistics:")
    print(f"Train samples: {stats['n_train']}")
    print(f"Val samples: {stats['n_val']}")
    print(f"EEG range: [{stats['eeg_stats']['train_min']:.3f}, {stats['eeg_stats']['train_max']:.3f}]")
    print(f"Image range: [{stats['image_stats']['train_min']:.3f}, {stats['image_stats']['train_max']:.3f}]")
    print(f"Label distribution: {stats['label_distribution']['train']}")
    
    # Test augmentation
    print("\nTesting augmentation...")
    dataset.augment_dataset()
    new_stats = dataset.get_data_stats()
    print(f"After augmentation - Train samples: {new_stats['n_train']}")
    
    print("\nDemo completed successfully!")
