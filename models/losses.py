"""
Custom Loss Functions for BrainWave Analyzer
Implements specialized losses for EEG-Image mapping
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple


class PerceptualLoss(tf.keras.losses.Loss):
    """
    Perceptual loss using pre-trained VGG features
    Measures high-level feature similarity between images
    """
    
    def __init__(self, 
                 vgg_layers: list = None,
                 weights: list = None,
                 **kwargs):
        """
        Initialize perceptual loss
        
        Args:
            vgg_layers: VGG layer indices to use for feature extraction
            weights: Weights for each layer
        """
        super(PerceptualLoss, self).__init__(**kwargs)
        
        if vgg_layers is None:
            self.vgg_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2']
        else:
            self.vgg_layers = vgg_layers
            
        if weights is None:
            self.weights = [1.0, 1.0, 1.0]
        else:
            self.weights = weights
            
        # Load pre-trained VGG model
        self.vgg = self._build_vgg_model()
    
    def _build_vgg_model(self):
        """Build VGG model for feature extraction"""
        vgg = tf.keras.applications.VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(64, 64, 3)
        )
        
        # Extract specific layers
        outputs = [vgg.get_layer(name).output for name in self.vgg_layers]
        model = tf.keras.Model(vgg.input, outputs)
        model.trainable = False
        
        return model
    
    def call(self, y_true, y_pred):
        """
        Compute perceptual loss
        
        Args:
            y_true: Ground truth images
            y_pred: Predicted images
            
        Returns:
            Perceptual loss value
        """
        # Ensure images are in correct range [0, 1]
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        
        # Scale to ImageNet preprocessing
        y_true_scaled = tf.keras.applications.vgg16.preprocess_input(y_true * 255.0)
        y_pred_scaled = tf.keras.applications.vgg16.preprocess_input(y_pred * 255.0)
        
        # Extract features
        true_features = self.vgg(y_true_scaled)
        pred_features = self.vgg(y_pred_scaled)
        
        # Compute MSE loss for each layer
        loss = 0.0
        for i, (true_feat, pred_feat, weight) in enumerate(
            zip(true_features, pred_features, self.weights)
        ):
            layer_loss = tf.reduce_mean(tf.square(true_feat - pred_feat))
            loss += weight * layer_loss
        
        return loss / len(self.vgg_layers)


class DTWLoss(tf.keras.losses.Loss):
    """
    Dynamic Time Warping inspired loss for EEG sequences
    Measures temporal alignment between EEG signals
    """
    
    def __init__(self, 
                 gamma: float = 1.0,
                 **kwargs):
        """
        Initialize DTW loss
        
        Args:
            gamma: Smoothing parameter for soft DTW
        """
        super(DTWLoss, self).__init__(**kwargs)
        self.gamma = gamma
    
    def _soft_dtw_step(self, cost_matrix, gamma):
        """Single step of soft DTW computation"""
        n, m = cost_matrix.shape
        R = tf.zeros((n + 2, m + 2))
        
        # Initialize boundaries
        R = tf.tensor_scatter_nd_update(R, [[0, 0]], [0.0])
        R = tf.tensor_scatter_nd_update(R, [[0, 1]], [float('inf')])
        R = tf.tensor_scatter_nd_update(R, [[1, 0]], [float('inf')])
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Soft minimum
                values = tf.stack([
                    R[i-1, j-1],
                    R[i-1, j],
                    R[i, j-1]
                ])
                soft_min = -gamma * tf.reduce_logsumexp(-values / gamma)
                R = tf.tensor_scatter_nd_update(
                    R, [[i, j]], [cost_matrix[i-1, j-1] + soft_min]
                )
        
        return R[n, m]
    
    def call(self, y_true, y_pred):
        """
        Compute DTW loss
        
        Args:
            y_true: Ground truth EEG sequences
            y_pred: Predicted EEG sequences
            
        Returns:
            DTW loss value
        """
        batch_size = tf.shape(y_true)[0]
        losses = []
        
        for i in range(batch_size):
            true_seq = y_true[i]
            pred_seq = y_pred[i]
            
            # Compute pairwise distance matrix
            true_expanded = tf.expand_dims(true_seq, axis=1)
            pred_expanded = tf.expand_dims(pred_seq, axis=0)
            
            cost_matrix = tf.reduce_sum(tf.square(true_expanded - pred_expanded), axis=2)
            
            # Compute soft DTW
            dtw_loss = self._soft_dtw_step(cost_matrix, self.gamma)
            losses.append(dtw_loss)
        
        return tf.reduce_mean(tf.stack(losses))


class FrequencyDomainLoss(tf.keras.losses.Loss):
    """
    Loss in frequency domain for EEG signals
    Compares power spectral densities
    """
    
    def __init__(self, 
                 sample_rate: float = 250.0,
                 freq_bands: list = None,
                 **kwargs):
        """
        Initialize frequency domain loss
        
        Args:
            sample_rate: Sampling rate of EEG signals
            freq_bands: Frequency bands to compare
        """
        super(FrequencyDomainLoss, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        
        if freq_bands is None:
            self.freq_bands = [
                (0.5, 4),    # Delta
                (4, 8),      # Theta
                (8, 13),     # Alpha
                (13, 30),    # Beta
                (30, 50)     # Gamma
            ]
        else:
            self.freq_bands = freq_bands
    
    def _compute_psd(self, signal):
        """Compute power spectral density"""
        # Apply window and compute FFT
        windowed = signal * tf.signal.hann_window(tf.shape(signal)[-1])
        fft = tf.signal.fft(tf.cast(windowed, tf.complex64))
        psd = tf.abs(fft) ** 2
        
        return psd
    
    def _compute_band_power(self, psd, freq_band):
        """Compute power in specific frequency band"""
        low_freq, high_freq = freq_band
        n_samples = tf.shape(psd)[-1]
        
        low_bin = int(low_freq * n_samples / self.sample_rate)
        high_bin = int(high_freq * n_samples / self.sample_rate)
        
        band_power = tf.reduce_mean(psd[..., low_bin:high_bin], axis=-1)
        return band_power
    
    def call(self, y_true, y_pred):
        """
        Compute frequency domain loss
        
        Args:
            y_true: Ground truth EEG sequences
            y_pred: Predicted EEG sequences
            
        Returns:
            Frequency domain loss value
        """
        # Compute PSD for both signals
        true_psd = self._compute_psd(y_true)
        pred_psd = self._compute_psd(y_pred)
        
        # Compute loss for each frequency band
        total_loss = 0.0
        
        for freq_band in self.freq_bands:
            true_power = self._compute_band_power(true_psd, freq_band)
            pred_power = self._compute_band_power(pred_psd, freq_band)
            
            band_loss = tf.reduce_mean(tf.square(true_power - pred_power))
            total_loss += band_loss
        
        return total_loss / len(self.freq_bands)


class CycleConsistencyLoss(tf.keras.losses.Loss):
    """
    Cycle consistency loss for bidirectional mapping
    Ensures EEG -> Image -> EEG and Image -> EEG -> Image consistency
    """
    
    def __init__(self, 
                 eeg_loss_weight: float = 1.0,
                 image_loss_weight: float = 1.0,
                 **kwargs):
        """
        Initialize cycle consistency loss
        
        Args:
            eeg_loss_weight: Weight for EEG cycle loss
            image_loss_weight: Weight for image cycle loss
        """
        super(CycleConsistencyLoss, self).__init__(**kwargs)
        self.eeg_loss_weight = eeg_loss_weight
        self.image_loss_weight = image_loss_weight
    
    def call(self, y_true, y_pred):
        """
        Compute cycle consistency loss
        
        Args:
            y_true: Tuple of (original_eeg, original_image)
            y_pred: Tuple of (reconstructed_eeg, reconstructed_image)
            
        Returns:
            Cycle consistency loss value
        """
        original_eeg, original_image = y_true
        reconstructed_eeg, reconstructed_image = y_pred
        
        # EEG cycle loss
        eeg_loss = tf.reduce_mean(tf.square(original_eeg - reconstructed_eeg))
        
        # Image cycle loss
        image_loss = tf.reduce_mean(tf.square(original_image - reconstructed_image))
        
        # Combined loss
        total_loss = (self.eeg_loss_weight * eeg_loss + 
                     self.image_loss_weight * image_loss)
        
        return total_loss


class CombinedLoss(tf.keras.losses.Loss):
    """
    Combined loss function for training
    Combines multiple loss components
    """
    
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 dtw_weight: float = 0.1,
                 frequency_weight: float = 0.05,
                 cycle_weight: float = 0.1,
                 **kwargs):
        """
        Initialize combined loss
        
        Args:
            reconstruction_weight: Weight for reconstruction loss
            perceptual_weight: Weight for perceptual loss
            dtw_weight: Weight for DTW loss
            frequency_weight: Weight for frequency domain loss
            cycle_weight: Weight for cycle consistency loss
        """
        super(CombinedLoss, self).__init__(**kwargs)
        
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        self.dtw_weight = dtw_weight
        self.frequency_weight = frequency_weight
        self.cycle_weight = cycle_weight
        
        # Initialize individual loss functions
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.perceptual_loss = PerceptualLoss()
        self.dtw_loss = DTWLoss()
        self.frequency_loss = FrequencyDomainLoss()
        self.cycle_loss = CycleConsistencyLoss()
    
    def call(self, y_true, y_pred):
        """
        Compute combined loss
        
        Args:
            y_true: Ground truth data
            y_pred: Predicted data
            
        Returns:
            Combined loss value
        """
        total_loss = 0.0
        
        # Reconstruction loss (MSE)
        if self.reconstruction_weight > 0:
            recon_loss = self.mse_loss(y_true, y_pred)
            total_loss += self.reconstruction_weight * recon_loss
        
        # Add other losses based on input format
        # This is a simplified version - in practice, you'd need to
        # handle different input formats for different loss components
        
        return total_loss


class KLDivergenceLoss(tf.keras.losses.Loss):
    """
    KL Divergence loss for VAE-style models
    """
    
    def __init__(self, beta: float = 1.0, **kwargs):
        """
        Initialize KL divergence loss
        
        Args:
            beta: Beta parameter for beta-VAE
        """
        super(KLDivergenceLoss, self).__init__(**kwargs)
        self.beta = beta
    
    def call(self, y_true, y_pred):
        """
        Compute KL divergence loss
        
        Args:
            y_true: Should contain (z_mean, z_log_var) for VAE
            y_pred: Model predictions (not used for KL loss)
            
        Returns:
            KL divergence loss value
        """
        z_mean, z_log_var = y_true
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1
        )
        
        return self.beta * tf.reduce_mean(kl_loss)


def create_loss_function(loss_type: str, **kwargs) -> tf.keras.losses.Loss:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss function
        **kwargs: Loss-specific arguments
        
    Returns:
        Loss function instance
    """
    loss_functions = {
        'mse': tf.keras.losses.MeanSquaredError,
        'mae': tf.keras.losses.MeanAbsoluteError,
        'perceptual': PerceptualLoss,
        'dtw': DTWLoss,
        'frequency': FrequencyDomainLoss,
        'cycle': CycleConsistencyLoss,
        'combined': CombinedLoss,
        'kl_divergence': KLDivergenceLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_functions[loss_type](**kwargs)


if __name__ == "__main__":
    # Demo usage
    print("Testing custom loss functions...")
    
    # Test data
    batch_size, time_steps, features = 2, 100, 1
    height, width, channels = 64, 64, 3
    
    # Test MSE loss
    print("Testing MSE Loss...")
    y_true = tf.random.normal((batch_size, time_steps, features))
    y_pred = tf.random.normal((batch_size, time_steps, features))
    mse_loss = tf.keras.losses.MeanSquaredError()
    loss_value = mse_loss(y_true, y_pred)
    print(f"MSE Loss: {loss_value.numpy():.4f}")
    
    # Test DTW loss
    print("Testing DTW Loss...")
    dtw_loss = DTWLoss(gamma=1.0)
    loss_value = dtw_loss(y_true, y_pred)
    print(f"DTW Loss: {loss_value.numpy():.4f}")
    
    # Test frequency domain loss
    print("Testing Frequency Domain Loss...")
    freq_loss = FrequencyDomainLoss(sample_rate=250.0)
    loss_value = freq_loss(y_true, y_pred)
    print(f"Frequency Loss: {loss_value.numpy():.4f}")
    
    # Test image losses
    print("Testing Image Losses...")
    true_images = tf.random.uniform((batch_size, height, width, channels), 0, 1)
    pred_images = tf.random.uniform((batch_size, height, width, channels), 0, 1)
    
    # Test perceptual loss (simplified version)
    print("Testing Perceptual Loss...")
    try:
        perceptual_loss = PerceptualLoss()
        loss_value = perceptual_loss(true_images, pred_images)
        print(f"Perceptual Loss: {loss_value.numpy():.4f}")
    except Exception as e:
        print(f"Perceptual Loss test skipped: {e}")
    
    print("Loss function tests completed!")
