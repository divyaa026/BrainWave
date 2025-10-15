"""
Attention Mechanisms for BrainWave Analyzer
Implements temporal and spatial attention for EEG and image processing
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional


class TemporalAttention(layers.Layer):
    """
    Temporal attention mechanism for EEG sequences
    Attends to important time points in the sequence
    """
    
    def __init__(self, units: int = 64, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        self.attention_weights = None
        
    def build(self, input_shape):
        # Input shape: (batch_size, time_steps, features)
        self.W1 = self.add_weight(
            name='W1',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W2 = self.add_weight(
            name='W2', 
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b1 = self.add_weight(
            name='b1',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.b2 = self.add_weight(
            name='b2',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )
        super(TemporalAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, time_steps, features)
        
        # Compute attention scores
        # h = tanh(W1 * x + b1)
        h = tf.nn.tanh(tf.matmul(inputs, self.W1) + self.b1)
        
        # scores = W2 * h + b2
        scores = tf.matmul(h, self.W2) + self.b2
        scores = tf.squeeze(scores, axis=-1)  # Remove last dimension
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        # Store attention weights for visualization
        self.attention_weights = attention_weights
        
        # Apply attention weights
        # output = sum(attention_weights * inputs)
        attended_output = tf.reduce_sum(
            tf.expand_dims(attention_weights, axis=-1) * inputs,
            axis=1
        )
        
        return attended_output
    
    def get_attention_weights(self):
        """Get attention weights for visualization"""
        return self.attention_weights


class SpatialAttention(layers.Layer):
    """
    Spatial attention mechanism for images
    Attends to important spatial regions in the image
    """
    
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.attention_weights = None
    
    def build(self, input_shape):
        # Input shape: (batch_size, height, width, channels)
        self.conv1 = layers.Conv2D(
            filters=input_shape[-1] // 8,
            kernel_size=1,
            activation='relu',
            padding='same'
        )
        self.conv2 = layers.Conv2D(
            filters=1,
            kernel_size=1,
            activation='sigmoid',
            padding='same'
        )
        super(SpatialAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # inputs shape: (batch_size, height, width, channels)
        
        # Compute spatial attention map
        attention_map = self.conv1(inputs)
        attention_map = self.conv2(attention_map)
        
        # Store attention weights for visualization
        self.attention_weights = attention_map
        
        # Apply attention
        attended_output = inputs * attention_map
        
        return attended_output
    
    def get_attention_weights(self):
        """Get attention weights for visualization"""
        return self.attention_weights


class MultiHeadTemporalAttention(layers.Layer):
    """
    Multi-head temporal attention for better EEG sequence modeling
    """
    
    def __init__(self, num_heads: int = 4, key_dim: int = 64, **kwargs):
        super(MultiHeadTemporalAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention_weights = None
    
    def build(self, input_shape):
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.key_dim * 4, activation='relu'),
            layers.Dense(input_shape[-1])
        ])
        super(MultiHeadTemporalAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Self-attention
        attn_output = self.multi_head_attention(
            inputs, inputs, return_attention_scores=True
        )
        
        # Extract attention weights
        if isinstance(attn_output, tuple):
            attn_output, attention_weights = attn_output
            self.attention_weights = attention_weights
        
        # Residual connection + layer norm
        out1 = self.layer_norm1(inputs + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        
        # Residual connection + layer norm
        output = self.layer_norm2(out1 + ffn_output)
        
        return output


class ChannelAttention(layers.Layer):
    """
    Channel attention mechanism for feature maps
    """
    
    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.attention_weights = None
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
        self.fc = tf.keras.Sequential([
            layers.Dense(channels // self.reduction_ratio, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])
        super(ChannelAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Global average pooling
        avg_out = self.avg_pool(inputs)
        avg_out = tf.expand_dims(tf.expand_dims(avg_out, axis=1), axis=2)
        avg_out = self.fc(avg_out)
        
        # Global max pooling
        max_out = self.max_pool(inputs)
        max_out = tf.expand_dims(tf.expand_dims(max_out, axis=1), axis=2)
        max_out = self.fc(max_out)
        
        # Combine and apply attention
        attention_weights = avg_out + max_out
        self.attention_weights = attention_weights
        
        attended_output = inputs * attention_weights
        
        return attended_output


class CrossModalAttention(layers.Layer):
    """
    Cross-modal attention between EEG and image features
    """
    
    def __init__(self, units: int = 128, **kwargs):
        super(CrossModalAttention, self).__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # input_shape should be a list of two shapes: [eeg_shape, image_shape]
        eeg_shape, image_shape = input_shape
        
        self.eeg_proj = layers.Dense(self.units)
        self.image_proj = layers.Dense(self.units)
        self.attention = layers.Dot(axes=2)
        
        super(CrossModalAttention, self).build(input_shape)
    
    def call(self, inputs, training=None):
        eeg_features, image_features = inputs
        
        # Project features to common space
        eeg_proj = self.eeg_proj(eeg_features)
        image_proj = self.image_proj(image_features)
        
        # Compute attention scores
        attention_scores = self.attention([eeg_proj, image_proj])
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # Apply attention
        attended_eeg = tf.matmul(attention_weights, eeg_proj)
        attended_image = tf.matmul(tf.transpose(attention_weights, [0, 2, 1]), image_proj)
        
        return attended_eeg, attended_image


class AttentionVisualization:
    """
    Utility class for visualizing attention weights
    """
    
    @staticmethod
    def visualize_temporal_attention(attention_weights: np.ndarray,
                                   eeg_signal: np.ndarray,
                                   save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize temporal attention weights over EEG signal
        
        Args:
            attention_weights: Attention weights of shape (time_steps,)
            eeg_signal: EEG signal of shape (time_steps,)
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        # Plot EEG signal
        time_axis = np.arange(len(eeg_signal))
        ax1.plot(time_axis, eeg_signal, 'b-', linewidth=1, label='EEG Signal')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('EEG Signal with Temporal Attention')
        ax1.grid(True, alpha=0.3)
        
        # Plot attention weights
        ax2.plot(time_axis, attention_weights, 'r-', linewidth=2, label='Attention Weights')
        ax2.fill_between(time_axis, 0, attention_weights, alpha=0.3, color='red')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Temporal Attention Weights')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Convert to numpy array for return
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis_array
    
    @staticmethod
    def visualize_spatial_attention(attention_map: np.ndarray,
                                  original_image: np.ndarray,
                                  save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize spatial attention map over image
        
        Args:
            attention_map: Attention map of shape (height, width, 1)
            original_image: Original image of shape (height, width, channels)
            save_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Attention map
        attention_vis = attention_map.squeeze()
        im1 = axes[1].imshow(attention_vis, cmap='hot', interpolation='nearest')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(attention_vis, cmap='hot', alpha=0.5, interpolation='nearest')
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Convert to numpy array for return
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return vis_array


def create_attention_layer(layer_type: str, **kwargs) -> layers.Layer:
    """
    Factory function to create attention layers
    
    Args:
        layer_type: Type of attention layer
        **kwargs: Layer-specific arguments
        
    Returns:
        Attention layer instance
    """
    if layer_type == 'temporal':
        return TemporalAttention(**kwargs)
    elif layer_type == 'spatial':
        return SpatialAttention(**kwargs)
    elif layer_type == 'multi_head_temporal':
        return MultiHeadTemporalAttention(**kwargs)
    elif layer_type == 'channel':
        return ChannelAttention(**kwargs)
    else:
        raise ValueError(f"Unknown attention layer type: {layer_type}")


if __name__ == "__main__":
    # Demo usage
    print("Testing attention mechanisms...")
    
    # Test temporal attention
    print("Testing Temporal Attention...")
    batch_size, time_steps, features = 2, 100, 64
    eeg_input = tf.random.normal((batch_size, time_steps, features))
    
    temporal_attn = TemporalAttention(units=32)
    output = temporal_attn(eeg_input)
    print(f"Temporal attention output shape: {output.shape}")
    
    # Test spatial attention
    print("Testing Spatial Attention...")
    height, width, channels = 64, 64, 3
    image_input = tf.random.normal((batch_size, height, width, channels))
    
    spatial_attn = SpatialAttention()
    output = spatial_attn(image_input)
    print(f"Spatial attention output shape: {output.shape}")
    
    # Test multi-head temporal attention
    print("Testing Multi-Head Temporal Attention...")
    multi_head_attn = MultiHeadTemporalAttention(num_heads=4, key_dim=16)
    output = multi_head_attn(eeg_input)
    print(f"Multi-head temporal attention output shape: {output.shape}")
    
    print("All attention mechanisms tested successfully!")
