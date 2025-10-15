"""
Signal Processing Utilities for BrainWave Analyzer
EEG preprocessing, filtering, and analysis functions
"""

import numpy as np
import scipy.signal
from scipy.fft import fft, fftfreq
from scipy.stats import pearsonr
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt


class EEGProcessor:
    """EEG signal processing utilities"""
    
    def __init__(self, 
                 sample_rate: int = 250,
                 duration: float = 4.0):
        """
        Initialize EEG processor
        
        Args:
            sample_rate: Sampling rate in Hz
            duration: Signal duration in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.time_points = int(sample_rate * duration)
        
        # Define frequency bands
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def filter_signal(self, 
                     signal: np.ndarray,
                     filter_type: str = 'bandpass',
                     low_freq: float = 0.5,
                     high_freq: float = 50,
                     order: int = 4) -> np.ndarray:
        """
        Filter EEG signal
        
        Args:
            signal: Input EEG signal
            filter_type: 'bandpass', 'highpass', 'lowpass', 'notch'
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyquist = self.sample_rate / 2
        
        if filter_type == 'bandpass':
            # Bandpass filter
            low = low_freq / nyquist
            high = high_freq / nyquist
            b, a = scipy.signal.butter(order, [low, high], btype='band')
            
        elif filter_type == 'highpass':
            # High-pass filter
            cutoff = low_freq / nyquist
            b, a = scipy.signal.butter(order, cutoff, btype='high')
            
        elif filter_type == 'lowpass':
            # Low-pass filter
            cutoff = high_freq / nyquist
            b, a = scipy.signal.butter(order, cutoff, btype='low')
            
        elif filter_type == 'notch':
            # Notch filter (e.g., for 50/60 Hz noise)
            notch_freq = low_freq
            q = 30  # Quality factor
            b, a = scipy.signal.iirnotch(notch_freq / nyquist, q)
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Apply filter
        if signal.ndim == 1:
            filtered = scipy.signal.filtfilt(b, a, signal)
        else:
            filtered = np.zeros_like(signal)
            for i in range(signal.shape[1]):
                filtered[:, i] = scipy.signal.filtfilt(b, a, signal[:, i])
        
        return filtered
    
    def remove_artifacts(self, 
                        signal: np.ndarray,
                        artifact_threshold: float = 3.0) -> np.ndarray:
        """
        Remove artifacts from EEG signal
        
        Args:
            signal: Input EEG signal
            artifact_threshold: Standard deviation threshold for artifact detection
            
        Returns:
            Cleaned signal
        """
        cleaned_signal = signal.copy()
        
        # Detect and remove outliers
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        
        # Mark outliers
        outliers = np.abs(signal - mean_val) > artifact_threshold * std_val
        
        if np.any(outliers):
            # Interpolate outliers
            valid_indices = ~outliers
            if np.sum(valid_indices) > 2:
                cleaned_signal[outliers] = np.interp(
                    np.where(outliers)[0],
                    np.where(valid_indices)[0],
                    cleaned_signal[valid_indices]
                )
        
        return cleaned_signal
    
    def normalize_signal(self, 
                        signal: np.ndarray,
                        method: str = 'standard') -> np.ndarray:
        """
        Normalize EEG signal
        
        Args:
            signal: Input EEG signal
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Normalized signal
        """
        if method == 'standard':
            # Z-score normalization
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            if std_val > 0:
                normalized = (signal - mean_val) / std_val
            else:
                normalized = signal - mean_val
                
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(signal)
            max_val = np.max(signal)
            if max_val > min_val:
                normalized = (signal - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(signal)
                
        elif method == 'robust':
            # Robust normalization using median and IQR
            median_val = np.median(signal)
            q75 = np.percentile(signal, 75)
            q25 = np.percentile(signal, 25)
            iqr = q75 - q25
            if iqr > 0:
                normalized = (signal - median_val) / iqr
            else:
                normalized = signal - median_val
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def compute_power_spectral_density(self, 
                                      signal: np.ndarray,
                                      window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density
        
        Args:
            signal: Input EEG signal
            window: Window function for PSD computation
            
        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        frequencies, psd = scipy.signal.welch(
            signal,
            fs=self.sample_rate,
            window=window,
            nperseg=min(256, len(signal) // 4),
            noverlap=None
        )
        
        return frequencies, psd
    
    def extract_frequency_bands(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract power in different frequency bands
        
        Args:
            signal: Input EEG signal
            
        Returns:
            Dictionary with power in each frequency band
        """
        frequencies, psd = self.compute_power_spectral_density(signal)
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            # Find frequency indices
            band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            
            # Compute average power in band
            band_power = np.mean(psd[band_mask])
            band_powers[band_name] = float(band_power)
        
        return band_powers
    
    def compute_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute spectral features from EEG signal
        
        Args:
            signal: Input EEG signal
            
        Returns:
            Dictionary of spectral features
        """
        frequencies, psd = self.compute_power_spectral_density(signal)
        
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = float(np.sum(frequencies * psd) / np.sum(psd))
        
        # Spectral bandwidth
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = float(
            np.sqrt(np.sum(((frequencies - centroid) ** 2) * psd) / np.sum(psd))
        )
        
        # Spectral rolloff (95% of power)
        cumsum_psd = np.cumsum(psd)
        total_power = cumsum_psd[-1]
        rolloff_threshold = 0.95 * total_power
        rolloff_idx = np.where(cumsum_psd >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = float(frequencies[rolloff_idx[0]])
        else:
            features['spectral_rolloff'] = float(frequencies[-1])
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
        features['zero_crossing_rate'] = float(zero_crossings / len(signal))
        
        # RMS energy
        features['rms_energy'] = float(np.sqrt(np.mean(signal ** 2)))
        
        return features
    
    def detect_events(self, 
                     signal: np.ndarray,
                     threshold: float = 2.0,
                     min_duration: int = 10) -> List[Tuple[int, int]]:
        """
        Detect events in EEG signal
        
        Args:
            signal: Input EEG signal
            threshold: Threshold for event detection (in standard deviations)
            min_duration: Minimum event duration in samples
            
        Returns:
            List of (start, end) indices for detected events
        """
        # Compute signal envelope
        envelope = np.abs(scipy.signal.hilbert(signal))
        
        # Normalize envelope
        envelope_norm = (envelope - np.mean(envelope)) / np.std(envelope)
        
        # Detect events above threshold
        events = envelope_norm > threshold
        
        # Find event boundaries
        event_changes = np.diff(events.astype(int))
        event_starts = np.where(event_changes == 1)[0] + 1
        event_ends = np.where(event_changes == -1)[0] + 1
        
        # Handle edge cases
        if events[0]:
            event_starts = np.concatenate([[0], event_starts])
        if events[-1]:
            event_ends = np.concatenate([event_ends, [len(signal)]])
        
        # Filter events by minimum duration
        valid_events = []
        for start, end in zip(event_starts, event_ends):
            if end - start >= min_duration:
                valid_events.append((start, end))
        
        return valid_events
    
    def preprocess_eeg(self, 
                      signal: np.ndarray,
                      filter_type: str = 'bandpass',
                      remove_artifacts: bool = True,
                      normalize: bool = True,
                      normalization_method: str = 'standard') -> np.ndarray:
        """
        Complete EEG preprocessing pipeline
        
        Args:
            signal: Input EEG signal
            filter_type: Filter type to apply
            remove_artifacts: Whether to remove artifacts
            normalize: Whether to normalize signal
            normalization_method: Normalization method
            
        Returns:
            Preprocessed signal
        """
        processed_signal = signal.copy()
        
        # Apply filtering
        processed_signal = self.filter_signal(
            processed_signal, 
            filter_type=filter_type
        )
        
        # Remove artifacts
        if remove_artifacts:
            processed_signal = self.remove_artifacts(processed_signal)
        
        # Normalize signal
        if normalize:
            processed_signal = self.normalize_signal(
                processed_signal, 
                method=normalization_method
            )
        
        return processed_signal
    
    def visualize_signal(self, 
                        signal: np.ndarray,
                        title: str = "EEG Signal",
                        save_path: Optional[str] = None) -> None:
        """
        Visualize EEG signal with spectral analysis
        
        Args:
            signal: Input EEG signal
            title: Plot title
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Time domain
        time_axis = np.linspace(0, self.duration, len(signal))
        axes[0, 0].plot(time_axis, signal, 'b-', linewidth=0.8)
        axes[0, 0].set_title('Time Domain')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power spectral density
        frequencies, psd = self.compute_power_spectral_density(signal)
        axes[0, 1].semilogy(frequencies, psd, 'r-', linewidth=1)
        axes[0, 1].set_title('Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Frequency bands
        band_powers = self.extract_frequency_bands(signal)
        band_names = list(band_powers.keys())
        band_values = list(band_powers.values())
        
        axes[1, 0].bar(band_names, band_values, color=['blue', 'green', 'red', 'orange', 'purple'])
        axes[1, 0].set_title('Frequency Band Powers')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Spectral features
        spectral_features = self.compute_spectral_features(signal)
        feature_names = list(spectral_features.keys())
        feature_values = list(spectral_features.values())
        
        axes[1, 1].bar(range(len(feature_names)), feature_values, color='skyblue')
        axes[1, 1].set_title('Spectral Features')
        axes[1, 1].set_ylabel('Feature Value')
        axes[1, 1].set_xticks(range(len(feature_names)))
        axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


class ImageProcessor:
    """Image processing utilities"""
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        """
        Initialize image processor
        
        Args:
            target_size: Target image size (height, width)
        """
        self.target_size = target_size
    
    def preprocess_image(self, 
                        image: np.ndarray,
                        normalize: bool = True,
                        normalize_method: str = 'minmax') -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: Input image
            normalize: Whether to normalize image
            normalize_method: Normalization method
            
        Returns:
            Preprocessed image
        """
        # Resize image if necessary
        if image.shape[:2] != self.target_size:
            import cv2
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # Normalize if requested
        if normalize:
            if normalize_method == 'minmax':
                # Normalize to [0, 1]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            elif normalize_method == 'standard':
                # Z-score normalization
                mean = np.mean(image)
                std = np.std(image)
                if std > 0:
                    image = (image - mean) / std
        
        return image.astype(np.float32)
    
    def extract_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract features from image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of image features
        """
        features = {}
        
        # Basic statistics
        features['mean_intensity'] = float(np.mean(image))
        features['std_intensity'] = float(np.std(image))
        features['min_intensity'] = float(np.min(image))
        features['max_intensity'] = float(np.max(image))
        
        # Color features (if RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            for i, color in enumerate(['red', 'green', 'blue']):
                channel = image[:, :, i]
                features[f'{color}_mean'] = float(np.mean(channel))
                features[f'{color}_std'] = float(np.std(channel))
        
        # Texture features (simplified)
        # Gradient magnitude
        if len(image.shape) == 3:
            gray_image = np.mean(image, axis=2)
        else:
            gray_image = image
        
        grad_x = np.gradient(gray_image, axis=1)
        grad_y = np.gradient(gray_image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = float(np.mean(gradient_magnitude))
        features['gradient_std'] = float(np.std(gradient_magnitude))
        
        return features


if __name__ == "__main__":
    # Demo usage
    print("Testing signal processing utilities...")
    
    # Create processor
    processor = EEGProcessor(sample_rate=250, duration=4.0)
    
    # Generate test signal
    t = np.linspace(0, 4.0, 1000)
    test_signal = (
        np.sin(2 * np.pi * 10 * t) +  # Alpha wave
        0.5 * np.sin(2 * np.pi * 20 * t) +  # Beta wave
        0.1 * np.random.randn(len(t))  # Noise
    )
    
    print(f"Test signal shape: {test_signal.shape}")
    
    # Test filtering
    filtered = processor.filter_signal(test_signal, filter_type='bandpass')
    print(f"Filtered signal shape: {filtered.shape}")
    
    # Test preprocessing
    preprocessed = processor.preprocess_eeg(test_signal)
    print(f"Preprocessed signal shape: {preprocessed.shape}")
    
    # Test feature extraction
    band_powers = processor.extract_frequency_bands(test_signal)
    print(f"Frequency band powers: {band_powers}")
    
    spectral_features = processor.compute_spectral_features(test_signal)
    print(f"Spectral features: {spectral_features}")
    
    # Test image processing
    image_processor = ImageProcessor(target_size=(64, 64))
    test_image = np.random.rand(100, 100, 3)
    
    processed_image = image_processor.preprocess_image(test_image)
    print(f"Processed image shape: {processed_image.shape}")
    
    image_features = image_processor.extract_image_features(test_image)
    print(f"Image features: {image_features}")
    
    print("Signal processing utilities test completed!")
