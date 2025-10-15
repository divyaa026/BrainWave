"""
EEG to Image Model Architecture
Advanced CNN-RNN hybrid with attention mechanisms for converting EEG to images
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict, Any
import numpy as np

from .attention import TemporalAttention, MultiHeadTemporalAttention
from .losses import CombinedLoss, PerceptualLoss


class EEGEncoder(layers.Layer):
    """
    Advanced EEG encoder with temporal attention
    """
    
    def __init__(self, 
                 hidden_units: int = 128,
                 latent_dim: int = 256,
                 use_attention: bool = True,
                 dropout_rate: float = 0.2,
                 **kwargs):
        super(EEGEncoder, self).__init__(**kwargs)
        
        self.hidden_units = hidden_units
        self.latent_dim = latent_dim
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # LSTM layers for temporal modeling
        self.lstm1 = layers.LSTM(
            hidden_units, 
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        )
        self.lstm2 = layers.LSTM(
            hidden_units // 2,
            return_sequences=use_attention,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate
        )
        
        # Attention mechanism
        if use_attention:
            self.temporal_attention = TemporalAttention(units=64)
        
        # Dense layers for feature extraction
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(latent_dim, activation='linear')
        
        # Batch normalization
        self.batch_norm1 = layers.BatchNormalization()
        self.batch_norm2 = layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, time_steps, features)
        
        # LSTM processing
        x = self.lstm1(inputs, training=training)
        x = self.batch_norm1(x, training=training)
        
        x = self.lstm2(x, training=training)
        x = self.batch_norm2(x, training=training)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.temporal_attention(x, training=training)
        
        # Dense feature extraction
        x = self.dense1(x, training=training)
        x = self.dropout(x, training=training)
        x = self.dense2(x, training=training)
        
        return x


class ImageDecoder(layers.Layer):
    """
    Advanced image decoder with progressive upsampling
    """
    
    def __init__(self, 
                 latent_dim: int = 256,
                 image_size: Tuple[int, int] = (64, 64),
                 base_channels: int = 512,
                 use_skip_connections: bool = True,
                 **kwargs):
        super(ImageDecoder, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.base_channels = base_channels
        self.use_skip_connections = use_skip_connections
        
        # Initial dense layer to reshape latent vector
        self.initial_size = (image_size[0] // 8, image_size[1] // 8)
        self.initial_dense = layers.Dense(
            self.initial_size[0] * self.initial_size[1] * base_channels,
            activation='relu'
        )
        self.initial_reshape = layers.Reshape(
            (self.initial_size[0], self.initial_size[1], base_channels)
        )
        
        # Progressive upsampling blocks
        self.upsample_blocks = []
        
        # Block 1: 8x8 -> 16x16
        self.upsample_blocks.append(self._create_upsample_block(
            base_channels // 2, base_channels, name='block1'
        ))
        
        # Block 2: 16x16 -> 32x32
        self.upsample_blocks.append(self._create_upsample_block(
            base_channels // 4, base_channels // 2, name='block2'
        ))
        
        # Block 3: 32x32 -> 64x64
        self.upsample_blocks.append(self._create_upsample_block(
            base_channels // 8, base_channels // 4, name='block3'
        ))
        
        # Final output layer
        self.final_conv = layers.Conv2DTranspose(
            3, 3, strides=1, padding='same', activation='sigmoid'
        )
    
    def _create_upsample_block(self, out_channels: int, in_channels: int, name: str):
        """Create an upsampling block"""
        return tf.keras.Sequential([
            layers.Conv2DTranspose(
                out_channels, 3, strides=2, padding='same',
                activation='relu', name=f'{name}_conv_transpose'
            ),
            layers.BatchNormalization(name=f'{name}_batch_norm'),
            layers.Conv2D(
                out_channels, 3, padding='same',
                activation='relu', name=f'{name}_conv'
            ),
            layers.BatchNormalization(name=f'{name}_batch_norm2')
        ], name=name)
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, latent_dim)
        
        # Initial processing
        x = self.initial_dense(inputs, training=training)
        x = self.initial_reshape(x)
        
        # Progressive upsampling
        for block in self.upsample_blocks:
            x = block(x, training=training)
        
        # Final output
        x = self.final_conv(x, training=training)
        
        return x


class EEGToImageModel(Model):
    """
    Complete EEG to Image model with attention mechanisms
    """
    
    def __init__(self, 
                 time_steps: int = 100,
                 n_features: int = 1,
                 image_size: Tuple[int, int] = (64, 64),
                 latent_dim: int = 256,
                 hidden_units: int = 128,
                 use_attention: bool = True,
                 use_vae: bool = False,
                 **kwargs):
        super(EEGToImageModel, self).__init__(**kwargs)
        
        self.time_steps = time_steps
        self.n_features = n_features
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.use_attention = use_attention
        self.use_vae = use_vae
        
        # Build model components
        self.eeg_encoder = EEGEncoder(
            hidden_units=hidden_units,
            latent_dim=latent_dim,
            use_attention=use_attention
        )
        
        if use_vae:
            # VAE components
            self.z_mean = layers.Dense(latent_dim, name='z_mean')
            self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
            self.sampling = self._sampling_layer()
        
        self.image_decoder = ImageDecoder(
            latent_dim=latent_dim,
            image_size=image_size
        )
        
        # Loss tracking
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        if use_vae:
            self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    def _sampling_layer(self):
        """Create sampling layer for VAE"""
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return sampling
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, time_steps, n_features)
        
        # Encode EEG
        encoded = self.eeg_encoder(inputs, training=training)
        
        if self.use_vae:
            # VAE sampling
            z_mean = self.z_mean(encoded)
            z_log_var = self.z_log_var(encoded)
            z = self.sampling([z_mean, z_log_var])
        else:
            z = encoded
        
        # Decode to image
        generated_image = self.image_decoder(z, training=training)
        
        if self.use_vae:
            return generated_image, z_mean, z_log_var
        else:
            return generated_image
    
    def compile(self, 
                optimizer: str = 'adam',
                learning_rate: float = 0.001,
                loss_weights: Dict[str, float] = None,
                **kwargs):
        """Compile model with custom loss functions"""
        
        if loss_weights is None:
            loss_weights = {
                'reconstruction': 1.0,
                'perceptual': 0.1,
                'kl_divergence': 0.01
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
        
        super(EEGToImageModel, self).compile(optimizer=opt, **kwargs)
        
        self.loss_weights = loss_weights
    
    def train_step(self, data):
        """Custom training step"""
        eeg_input, image_target = data
        
        with tf.GradientTape() as tape:
            if self.use_vae:
                generated_image, z_mean, z_log_var = self(eeg_input, training=True)
                
                # Reconstruction loss
                reconstruction_loss = tf.reduce_mean(
                    tf.square(image_target - generated_image)
                )
                
                # KL divergence loss
                kl_loss = -0.5 * tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
                kl_loss = tf.reduce_mean(kl_loss)
                
                # Total loss
                total_loss = (self.loss_weights['reconstruction'] * reconstruction_loss +
                             self.loss_weights['kl_divergence'] * kl_loss)
            else:
                generated_image = self(eeg_input, training=True)
                reconstruction_loss = tf.reduce_mean(
                    tf.square(image_target - generated_image)
                )
                total_loss = reconstruction_loss
                kl_loss = 0.0
        
        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        if self.use_vae:
            self.kl_loss_tracker.update_state(kl_loss)
        
        # Return metrics
        metrics = {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
        }
        if self.use_vae:
            metrics["kl_loss"] = self.kl_loss_tracker.result()
        
        return metrics
    
    def test_step(self, data):
        """Custom test step"""
        eeg_input, image_target = data
        
        if self.use_vae:
            generated_image, z_mean, z_log_var = self(eeg_input, training=False)
            
            reconstruction_loss = tf.reduce_mean(
                tf.square(image_target - generated_image)
            )
            
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
            kl_loss = tf.reduce_mean(kl_loss)
            
            total_loss = (self.loss_weights['reconstruction'] * reconstruction_loss +
                         self.loss_weights['kl_divergence'] * kl_loss)
        else:
            generated_image = self(eeg_input, training=False)
            reconstruction_loss = tf.reduce_mean(
                tf.square(image_target - generated_image)
            )
            total_loss = reconstruction_loss
            kl_loss = 0.0
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        if self.use_vae:
            self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result() if self.use_vae else 0.0
        }
    
    def predict_image_from_eeg(self, eeg_sequence: np.ndarray) -> np.ndarray:
        """
        Predict image from EEG sequence
        
        Args:
            eeg_sequence: EEG data of shape (time_steps,) or (1, time_steps, n_features)
            
        Returns:
            Generated image of shape (height, width, 3) in range [0, 1]
        """
        # Preprocess input
        if eeg_sequence.ndim == 1:
            eeg_sequence = eeg_sequence.reshape(1, self.time_steps, self.n_features)
        elif eeg_sequence.ndim == 2:
            eeg_sequence = eeg_sequence.reshape(1, self.time_steps, self.n_features)
        
        # Ensure correct shape
        if eeg_sequence.shape[1] != self.time_steps:
            # Resize if necessary
            eeg_sequence = tf.image.resize(
                tf.expand_dims(eeg_sequence, axis=-1),
                [self.time_steps, 1]
            ).numpy()
            eeg_sequence = tf.squeeze(eeg_sequence, axis=-1).numpy()
            eeg_sequence = np.expand_dims(eeg_sequence, axis=-1)
        
        # Predict
        if self.use_vae:
            generated_image, _, _ = self(eeg_sequence, training=False)
        else:
            generated_image = self(eeg_sequence, training=False)
        
        # Post-process
        image = generated_image.numpy()[0]
        image = np.clip(image, 0.0, 1.0)
        
        return image.astype(np.float32)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'EEGToImageModel',
            'time_steps': self.time_steps,
            'n_features': self.n_features,
            'image_size': self.image_size,
            'latent_dim': self.latent_dim,
            'hidden_units': self.hidden_units,
            'use_attention': self.use_attention,
            'use_vae': self.use_vae,
            'total_params': self.count_params()
        }


def create_eeg_to_image_model(time_steps: int = 100,
                             n_features: int = 1,
                             image_size: Tuple[int, int] = (64, 64),
                             latent_dim: int = 256,
                             use_attention: bool = True,
                             use_vae: bool = False) -> EEGToImageModel:
    """
    Factory function to create EEG to Image model
    
    Args:
        time_steps: Number of EEG time steps
        n_features: Number of EEG features/channels
        image_size: Target image size (height, width)
        latent_dim: Latent space dimension
        use_attention: Whether to use attention mechanisms
        use_vae: Whether to use VAE architecture
        
    Returns:
        Compiled EEGToImageModel instance
    """
    model = EEGToImageModel(
        time_steps=time_steps,
        n_features=n_features,
        image_size=image_size,
        latent_dim=latent_dim,
        use_attention=use_attention,
        use_vae=use_vae
    )
    
    # Build model by calling it with dummy data
    dummy_input = tf.random.normal((1, time_steps, n_features))
    _ = model(dummy_input)
    
    # Compile model
    model.compile(
        optimizer='adam',
        learning_rate=0.001
    )
    
    return model


if __name__ == "__main__":
    # Demo usage
    print("Testing EEG to Image Model...")
    
    # Create model
    model = create_eeg_to_image_model(
        time_steps=100,
        n_features=1,
        image_size=(64, 64),
        latent_dim=256,
        use_attention=True,
        use_vae=False
    )
    
    print(f"Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # Test prediction
    dummy_eeg = np.random.randn(100)
    generated_image = model.predict_image_from_eeg(dummy_eeg)
    print(f"Generated image shape: {generated_image.shape}")
    print(f"Generated image range: [{generated_image.min():.3f}, {generated_image.max():.3f}]")
    
    print("EEG to Image model test completed!")
