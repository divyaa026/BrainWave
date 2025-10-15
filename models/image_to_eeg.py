"""
Image to EEG Model Architecture
Advanced CNN-RNN hybrid with attention mechanisms for converting images to EEG
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .attention import SpatialAttention, ChannelAttention
from .losses import DTWLoss, FrequencyDomainLoss


class ImageEncoder(layers.Layer):
    """
    Advanced image encoder with spatial attention
    """
    
    def __init__(self, 
                 latent_dim: int = 256,
                 image_size: Tuple[int, int] = (64, 64),
                 base_channels: int = 32,
                 use_attention: bool = True,
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(ImageEncoder, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.base_channels = base_channels
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # Progressive downsampling blocks
        self.downsample_blocks = []
        
        # Block 1: 64x64 -> 32x32
        self.downsample_blocks.append(self._create_downsample_block(
            base_channels, 64, name='block1'
        ))
        
        # Block 2: 32x32 -> 16x16
        self.downsample_blocks.append(self._create_downsample_block(
            base_channels * 2, base_channels, name='block2'
        ))
        
        # Block 3: 16x16 -> 8x8
        self.downsample_blocks.append(self._create_downsample_block(
            base_channels * 4, base_channels * 2, name='block3'
        ))
        
        # Attention mechanisms
        if use_attention:
            self.spatial_attention = SpatialAttention()
            self.channel_attention = ChannelAttention(reduction_ratio=8)
        
        # Global pooling and dense layers
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(base_channels * 8, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(latent_dim, activation='linear')
        
        # Batch normalization
        self.batch_norm = layers.BatchNormalization()
    
    def _create_downsample_block(self, out_channels: int, in_channels: int, name: str):
        """Create a downsampling block"""
        return tf.keras.Sequential([
            layers.Conv2D(
                out_channels, 3, strides=2, padding='same',
                activation='relu', name=f'{name}_conv'
            ),
            layers.BatchNormalization(name=f'{name}_batch_norm'),
            layers.Conv2D(
                out_channels, 3, padding='same',
                activation='relu', name=f'{name}_conv2'
            ),
            layers.BatchNormalization(name=f'{name}_batch_norm2'),
            layers.Dropout(self.dropout_rate, name=f'{name}_dropout')
        ], name=name)
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, height, width, channels)
        
        x = inputs
        
        # Progressive downsampling
        for block in self.downsample_blocks:
            x = block(x, training=training)
        
        # Apply attention mechanisms
        if self.use_attention:
            x = self.spatial_attention(x, training=training)
            x = self.channel_attention(x, training=training)
        
        # Global pooling and dense layers
        x = self.global_pool(x)
        x = self.dense1(x, training=training)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x, training=training)
        
        return x


class EEGDecoder(layers.Layer):
    """
    Advanced EEG decoder with temporal modeling
    """
    
    def __init__(self, 
                 latent_dim: int = 256,
                 time_steps: int = 100,
                 n_features: int = 1,
                 hidden_units: int = 128,
                 use_teacher_forcing: bool = False,
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(EEGDecoder, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.use_teacher_forcing = use_teacher_forcing
        self.dropout_rate = dropout_rate
        
        # Project latent vector to initial state
        self.latent_projection = layers.Dense(
            hidden_units * 2, activation='tanh'
        )
        
        # LSTM layers for temporal generation
        self.lstm1 = layers.LSTM(
            hidden_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='lstm1'
        )
        self.lstm2 = layers.LSTM(
            hidden_units // 2,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            name='lstm2'
        )
        
        # Output projection
        self.output_projection = layers.Dense(
            n_features, activation='tanh'
        )
        
        # Batch normalization
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
    
    def call(self, inputs, training=None, teacher_targets=None):
        # inputs shape: (batch_size, latent_dim)
        batch_size = tf.shape(inputs)[0]
        
        # Project latent vector
        initial_state = self.latent_projection(inputs, training=training)
        
        # Split into initial hidden and cell states
        initial_h = initial_state[:, :self.hidden_units]
        initial_c = initial_state[:, self.hidden_units:]
        
        # Create initial states for LSTM
        initial_states = [initial_h, initial_c]
        
        # Generate sequence
        if self.use_teacher_forcing and training and teacher_targets is not None:
            # Teacher forcing during training
            x = teacher_targets
        else:
            # Autoregressive generation
            x = tf.zeros((batch_size, self.time_steps, self.n_features))
        
        # LSTM processing
        x = self.lstm1(x, initial_state=initial_states, training=training)
        x = self.batch_norm1(x, training=training)
        
        x = self.lstm2(x, training=training)
        x = self.batch_norm2(x, training=training)
        
        # Output projection
        output = self.output_projection(x, training=training)
        
        return output


class ImageToEEGModel(Model):
    """
    Complete Image to EEG model with attention mechanisms
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (64, 64),
                 time_steps: int = 100,
                 n_features: int = 1,
                 latent_dim: int = 256,
                 hidden_units: int = 128,
                 use_attention: bool = True,
                 use_teacher_forcing: bool = False,
                 **kwargs):
        super(ImageToEEGModel, self).__init__(**kwargs)
        
        self.image_size = image_size
        self.time_steps = time_steps
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.use_attention = use_attention
        self.use_teacher_forcing = use_teacher_forcing
        
        # Build model components
        self.image_encoder = ImageEncoder(
            latent_dim=latent_dim,
            image_size=image_size,
            use_attention=use_attention
        )
        
        self.eeg_decoder = EEGDecoder(
            latent_dim=latent_dim,
            time_steps=time_steps,
            n_features=n_features,
            hidden_units=hidden_units,
            use_teacher_forcing=use_teacher_forcing
        )
        
        # Loss tracking
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.dtw_loss_tracker = tf.keras.metrics.Mean(name="dtw_loss")
        self.frequency_loss_tracker = tf.keras.metrics.Mean(name="frequency_loss")
    
    def call(self, inputs, training=None, teacher_targets=None):
        # inputs shape: (batch_size, height, width, channels)
        
        # Encode image
        encoded = self.image_encoder(inputs, training=training)
        
        # Decode to EEG
        generated_eeg = self.eeg_decoder(
            encoded, 
            training=training, 
            teacher_targets=teacher_targets
        )
        
        return generated_eeg
    
    def compile(self, 
                optimizer: str = 'adam',
                learning_rate: float = 0.001,
                loss_weights: Dict[str, float] = None,
                **kwargs):
        """Compile model with custom loss functions"""
        
        if loss_weights is None:
            loss_weights = {
                'reconstruction': 1.0,
                'dtw': 0.1,
                'frequency': 0.05
            }
        
        # Set up optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif optimizer.lower() == 'rmsprop':
                opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                opt = tf.keras.optimizers.get(optimizer)
        else:
            opt = optimizer
        
        super(ImageToEEGModel, self).compile(optimizer=opt, **kwargs)
        
        self.loss_weights = loss_weights
        
        # Initialize loss functions
        self.dtw_loss_fn = DTWLoss(gamma=1.0)
        self.frequency_loss_fn = FrequencyDomainLoss(sample_rate=250.0)
    
    def train_step(self, data):
        """Custom training step"""
        if self.use_teacher_forcing:
            image_input, eeg_target = data
        else:
            image_input, eeg_target = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            generated_eeg = self(
                image_input, 
                training=True,
                teacher_targets=eeg_target if self.use_teacher_forcing else None
            )
            
            # Compute losses
            reconstruction_loss = tf.reduce_mean(
                tf.square(eeg_target - generated_eeg)
            )
            
            dtw_loss = self.dtw_loss_fn(eeg_target, generated_eeg)
            frequency_loss = self.frequency_loss_fn(eeg_target, generated_eeg)
            
            # Total loss
            total_loss = (self.loss_weights['reconstruction'] * reconstruction_loss +
                         self.loss_weights['dtw'] * dtw_loss +
                         self.loss_weights['frequency'] * frequency_loss)
        
        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.dtw_loss_tracker.update_state(dtw_loss)
        self.frequency_loss_tracker.update_state(frequency_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "dtw_loss": self.dtw_loss_tracker.result(),
            "frequency_loss": self.frequency_loss_tracker.result()
        }
    
    def test_step(self, data):
        """Custom test step"""
        image_input, eeg_target = data
        
        # Forward pass (no teacher forcing during testing)
        generated_eeg = self(image_input, training=False)
        
        # Compute losses
        reconstruction_loss = tf.reduce_mean(
            tf.square(eeg_target - generated_eeg)
        )
        
        dtw_loss = self.dtw_loss_fn(eeg_target, generated_eeg)
        frequency_loss = self.frequency_loss_fn(eeg_target, generated_eeg)
        
        # Total loss
        total_loss = (self.loss_weights['reconstruction'] * reconstruction_loss +
                     self.loss_weights['dtw'] * dtw_loss +
                     self.loss_weights['frequency'] * frequency_loss)
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.dtw_loss_tracker.update_state(dtw_loss)
        self.frequency_loss_tracker.update_state(frequency_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "dtw_loss": self.dtw_loss_tracker.result(),
            "frequency_loss": self.frequency_loss_tracker.result()
        }
    
    def predict_eeg_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Predict EEG from image
        
        Args:
            image: Image data of shape (height, width, 3) in range [0, 1]
            
        Returns:
            Generated EEG of shape (time_steps, n_features)
        """
        # Preprocess input
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # Ensure correct shape
        if image.shape[1:3] != self.image_size:
            image = tf.image.resize(
                image, 
                self.image_size
            ).numpy()
        
        # Predict
        generated_eeg = self(image, training=False)
        
        # Post-process
        eeg = generated_eeg.numpy()[0]
        
        return eeg.astype(np.float32)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'ImageToEEGModel',
            'image_size': self.image_size,
            'time_steps': self.time_steps,
            'n_features': self.n_features,
            'latent_dim': self.latent_dim,
            'hidden_units': self.hidden_units,
            'use_attention': self.use_attention,
            'use_teacher_forcing': self.use_teacher_forcing,
            'total_params': self.count_params()
        }


def create_image_to_eeg_model(image_size: Tuple[int, int] = (64, 64),
                             time_steps: int = 100,
                             n_features: int = 1,
                             latent_dim: int = 256,
                             use_attention: bool = True,
                             use_teacher_forcing: bool = False) -> ImageToEEGModel:
    """
    Factory function to create Image to EEG model
    
    Args:
        image_size: Input image size (height, width)
        time_steps: Target EEG time steps
        n_features: Number of EEG features/channels
        latent_dim: Latent space dimension
        use_attention: Whether to use attention mechanisms
        use_teacher_forcing: Whether to use teacher forcing during training
        
    Returns:
        Compiled ImageToEEGModel instance
    """
    model = ImageToEEGModel(
        image_size=image_size,
        time_steps=time_steps,
        n_features=n_features,
        latent_dim=latent_dim,
        use_attention=use_attention,
        use_teacher_forcing=use_teacher_forcing
    )
    
    # Build model by calling it with dummy data
    dummy_input = tf.random.normal((1, *image_size, 3))
    _ = model(dummy_input)
    
    # Compile model
    model.compile(
        optimizer='adam',
        learning_rate=0.001
    )
    
    return model


if __name__ == "__main__":
    # Demo usage
    print("Testing Image to EEG Model...")
    
    # Create model
    model = create_image_to_eeg_model(
        image_size=(64, 64),
        time_steps=100,
        n_features=1,
        latent_dim=256,
        use_attention=True,
        use_teacher_forcing=False
    )
    
    print(f"Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # Test prediction
    dummy_image = np.random.uniform(0, 1, (64, 64, 3))
    generated_eeg = model.predict_eeg_from_image(dummy_image)
    print(f"Generated EEG shape: {generated_eeg.shape}")
    print(f"Generated EEG range: [{generated_eeg.min():.3f}, {generated_eeg.max():.3f}]")
    
    print("Image to EEG model test completed!")
