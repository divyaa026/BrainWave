"""
Training Pipeline for BrainWave Analyzer
Handles training of both EEG-to-Image and Image-to-EEG models
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    LearningRateScheduler, CSVLogger
)
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import BrainWaveDataset
from models.eeg_to_image import create_eeg_to_image_model
from models.image_to_eeg import create_image_to_eeg_model
from evaluation.metrics import ModelEvaluator
from .config import Config, get_demo_config, get_full_config


class BrainWaveTrainer:
    """Main trainer class for BrainWave models"""
    
    def __init__(self, config: Config):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.dataset = None
        self.eeg_to_image_model = None
        self.image_to_eeg_model = None
        self.training_history = {}
        
        # Create necessary directories
        self.config.create_directories()
        
        # Set up TensorFlow
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Configure GPU if available
        self._setup_gpu()
    
    def _setup_gpu(self):
        """Configure GPU settings"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU configured: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU available, using CPU")
    
    def load_dataset(self, force_regenerate: bool = False) -> BrainWaveDataset:
        """
        Load or generate dataset
        
        Args:
            force_regenerate: Whether to force dataset regeneration
            
        Returns:
            BrainWaveDataset instance
        """
        print("Loading dataset...")
        
        self.dataset = BrainWaveDataset(
            data_dir=self.config.data.data_dir,
            sample_rate=self.config.data.sample_rate,
            duration=self.config.data.duration,
            image_size=self.config.data.image_size,
            n_channels=self.config.data.n_channels
        )
        
        dataset_path = os.path.join(
            self.config.data.data_dir, 
            self.config.data.dataset_filename
        )
        
        # Try to load existing dataset
        if not force_regenerate and os.path.exists(dataset_path):
            try:
                self.dataset.load_dataset()
                print(f"Dataset loaded from {dataset_path}")
            except Exception as e:
                print(f"Failed to load dataset: {e}")
                print("Generating new dataset...")
                self.dataset.generate_dataset(
                    n_samples=self.config.training.n_samples,
                    save=True
                )
        else:
            print("Generating new dataset...")
            self.dataset.generate_dataset(
                n_samples=self.config.training.n_samples,
                save=True
            )
        
        # Apply augmentation if enabled
        if self.config.data.use_augmentation:
            print("Applying data augmentation...")
            self.dataset.augment_dataset(
                eeg_noise_factor=self.config.data.eeg_noise_factor,
                image_variation=self.config.data.image_variation
            )
        
        # Print dataset statistics
        stats = self.dataset.get_data_stats()
        print(f"Dataset loaded: {stats['n_train']} train, {stats['n_val']} val samples")
        
        return self.dataset
    
    def create_models(self):
        """Create model instances"""
        print("Creating models...")
        
        # Create EEG to Image model
        self.eeg_to_image_model = create_eeg_to_image_model(
            **self.config.model.eeg_to_image
        )
        
        # Create Image to EEG model
        self.image_to_eeg_model = create_image_to_eeg_model(
            **self.config.model.image_to_eeg
        )
        
        # Compile models with custom loss weights
        self.eeg_to_image_model.compile(
            optimizer=self.config.training.optimizer,
            learning_rate=self.config.training.learning_rate,
            loss_weights=self.config.training.eeg_to_image_loss_weights
        )
        
        self.image_to_eeg_model.compile(
            optimizer=self.config.training.optimizer,
            learning_rate=self.config.training.learning_rate,
            loss_weights=self.config.training.image_to_eeg_loss_weights
        )
        
        print("Models created and compiled successfully")
        
        # Print model info
        print(f"EEG-to-Image model params: {self.eeg_to_image_model.count_params():,}")
        print(f"Image-to-EEG model params: {self.image_to_eeg_model.count_params():,}")
    
    def create_callbacks(self, model_name: str) -> List[tf.keras.callbacks.Callback]:
        """
        Create training callbacks
        
        Args:
            model_name: Name of the model for checkpoint naming
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpointing
        if self.config.training.save_checkpoints:
            checkpoint_path = os.path.join(
                self.config.training.checkpoint_dir,
                f"{model_name}_best.h5"
            )
            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor=self.config.training.monitor_checkpoint,
                save_best_only=self.config.training.save_best_only,
                mode=self.config.training.mode_checkpoint,
                verbose=1
            )
            callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config.training.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor=self.config.training.early_stopping_monitor,
                patience=self.config.training.early_stopping_patience,
                min_delta=self.config.training.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Learning rate scheduling
        if self.config.training.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.config.training.lr_decay_factor,
                patience=self.config.training.lr_patience,
                min_lr=1e-7,
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        # CSV logging
        log_path = os.path.join(
            self.config.training.log_dir,
            f"{model_name}_training.csv"
        )
        csv_logger = CSVLogger(log_path, append=False)
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train_eeg_to_image_model(self) -> Dict:
        """
        Train EEG to Image model
        
        Returns:
            Training history
        """
        print("Training EEG to Image model...")
        
        # Prepare data
        train_eeg = self.dataset.train_eeg
        train_images = self.dataset.train_images
        val_eeg = self.dataset.val_eeg
        val_images = self.dataset.val_images
        
        # Create callbacks
        callbacks = self.create_callbacks("eeg_to_image")
        
        # Train model
        start_time = time.time()
        
        history = self.eeg_to_image_model.fit(
            train_eeg, train_images,
            validation_data=(val_eeg, val_images),
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"EEG to Image training completed in {training_time:.2f} seconds")
        
        # Store history
        self.training_history['eeg_to_image'] = history.history
        
        return history.history
    
    def train_image_to_eeg_model(self) -> Dict:
        """
        Train Image to EEG model
        
        Returns:
            Training history
        """
        print("Training Image to EEG model...")
        
        # Prepare data
        train_images = self.dataset.train_images
        train_eeg = self.dataset.train_eeg
        val_images = self.dataset.val_images
        val_eeg = self.dataset.val_eeg
        
        # Create callbacks
        callbacks = self.create_callbacks("image_to_eeg")
        
        # Train model
        start_time = time.time()
        
        history = self.image_to_eeg_model.fit(
            train_images, train_eeg,
            validation_data=(val_images, val_eeg),
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Image to EEG training completed in {training_time:.2f} seconds")
        
        # Store history
        self.training_history['image_to_eeg'] = history.history
        
        return history.history
    
    def train_both_models(self, train_order: str = 'sequential') -> Dict:
        """
        Train both models
        
        Args:
            train_order: 'sequential' or 'alternating'
            
        Returns:
            Combined training history
        """
        print(f"Training both models in {train_order} order...")
        
        if train_order == 'sequential':
            # Train one after the other
            eeg_to_image_history = self.train_eeg_to_image_model()
            image_to_eeg_history = self.train_image_to_eeg_model()
        else:
            # Alternating training (simplified version)
            print("Alternating training not implemented yet, using sequential")
            eeg_to_image_history = self.train_eeg_to_image_model()
            image_to_eeg_history = self.train_image_to_eeg_model()
        
        combined_history = {
            'eeg_to_image': eeg_to_image_history,
            'image_to_eeg': image_to_eeg_history
        }
        
        self.training_history = combined_history
        return combined_history
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.training_history:
            print("No training history available")
            return
        
        # Create subplots
        n_models = len(self.training_history)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, history) in enumerate(self.training_history.items()):
            # Plot loss
            axes[0, i].plot(history['total_loss'], label='Training Loss')
            if 'val_total_loss' in history:
                axes[0, i].plot(history['val_total_loss'], label='Validation Loss')
            axes[0, i].set_title(f'{model_name.replace("_", " ").title()} - Loss')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Loss')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # Plot reconstruction loss
            axes[1, i].plot(history['reconstruction_loss'], label='Training Reconstruction')
            if 'val_reconstruction_loss' in history:
                axes[1, i].plot(history['val_reconstruction_loss'], label='Validation Reconstruction')
            axes[1, i].set_title(f'{model_name.replace("_", " ").title()} - Reconstruction Loss')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Reconstruction Loss')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()
    
    def evaluate_models(self) -> Dict:
        """
        Evaluate trained models
        
        Returns:
            Evaluation results
        """
        print("Evaluating models...")
        
        if self.eeg_to_image_model is None or self.image_to_eeg_model is None:
            raise ValueError("Models not trained yet")
        
        evaluator = ModelEvaluator()
        
        # Prepare test data
        val_eeg = self.dataset.val_eeg
        val_images = self.dataset.val_images
        
        # Evaluate EEG to Image model
        print("Evaluating EEG to Image model...")
        eeg_to_image_results = evaluator.evaluate_eeg_to_image(
            self.eeg_to_image_model, val_eeg, val_images
        )
        
        # Evaluate Image to EEG model
        print("Evaluating Image to EEG model...")
        image_to_eeg_results = evaluator.evaluate_image_to_eeg(
            self.image_to_eeg_model, val_images, val_eeg
        )
        
        results = {
            'eeg_to_image': eeg_to_image_results,
            'image_to_eeg': image_to_eeg_results
        }
        
        # Save results
        results_path = os.path.join(
            self.config.evaluation.visualization_dir,
            'evaluation_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Evaluation results saved to {results_path}")
        
        return results
    
    def save_trained_models(self, save_dir: str = None):
        """
        Save trained models
        
        Args:
            save_dir: Directory to save models
        """
        if save_dir is None:
            save_dir = self.config.training.checkpoint_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        if self.eeg_to_image_model is not None:
            eeg_to_image_path = os.path.join(save_dir, 'eeg_to_image_model.h5')
            self.eeg_to_image_model.save_weights(eeg_to_image_path)
            print(f"EEG to Image model saved to {eeg_to_image_path}")
        
        if self.image_to_eeg_model is not None:
            image_to_eeg_path = os.path.join(save_dir, 'image_to_eeg_model.h5')
            self.image_to_eeg_model.save_weights(image_to_eeg_path)
            print(f"Image to EEG model saved to {image_to_eeg_path}")
        
        # Save configuration
        config_path = os.path.join(save_dir, 'training_config.json')
        self.config.save_config(config_path)
        print(f"Configuration saved to {config_path}")
    
    def load_trained_models(self, model_dir: str):
        """
        Load trained models
        
        Args:
            model_dir: Directory containing trained models
        """
        if self.eeg_to_image_model is None or self.image_to_eeg_model is None:
            self.create_models()
        
        eeg_to_image_path = os.path.join(model_dir, 'eeg_to_image_model.h5')
        image_to_eeg_path = os.path.join(model_dir, 'image_to_eeg_model.h5')
        
        if os.path.exists(eeg_to_image_path):
            self.eeg_to_image_model.load_weights(eeg_to_image_path)
            print(f"EEG to Image model loaded from {eeg_to_image_path}")
        
        if os.path.exists(image_to_eeg_path):
            self.image_to_eeg_model.load_weights(image_to_eeg_path)
            print(f"Image to EEG model loaded from {image_to_eeg_path}")
    
    def run_full_training_pipeline(self) -> Dict:
        """
        Run complete training pipeline
        
        Returns:
            Training and evaluation results
        """
        print("Starting full training pipeline...")
        
        # Load dataset
        self.load_dataset()
        
        # Create models
        self.create_models()
        
        # Train models
        training_history = self.train_both_models()
        
        # Evaluate models
        evaluation_results = self.evaluate_models()
        
        # Save models
        self.save_trained_models()
        
        # Plot results
        plot_path = os.path.join(
            self.config.evaluation.visualization_dir,
            'training_history.png'
        )
        self.plot_training_history(plot_path)
        
        results = {
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'config': self.config.to_dict()
        }
        
        print("Full training pipeline completed!")
        
        return results


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BrainWave models')
    parser.add_argument('--config', type=str, default='demo',
                       help='Configuration type: demo, full, attention')
    parser.add_argument('--force-regenerate', action='store_true',
                       help='Force dataset regeneration')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Get configuration
    if args.config == 'demo':
        config = get_demo_config()
    elif args.config == 'full':
        config = get_full_config()
    elif args.config == 'attention':
        config = get_attention_config()
    else:
        raise ValueError(f"Unknown config type: {args.config}")
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    
    # Create trainer and run pipeline
    trainer = BrainWaveTrainer(config)
    results = trainer.run_full_training_pipeline()
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
