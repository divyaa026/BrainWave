"""
Synthetic EEG Data Generator for BrainWave Analyzer
Generates realistic EEG patterns with different brain states
"""

import numpy as np
import scipy.signal
from typing import Tuple, Dict, List
from enum import Enum


class BrainState(Enum):
    """Different brain states for EEG generation"""
    RELAXED = "relaxed"      # Alpha waves (8-13 Hz)
    FOCUSED = "focused"      # Beta waves (13-30 Hz)
    ACTIVE = "active"        # Gamma waves (30-50 Hz)
    MOTOR_IMAGERY = "motor"  # Motor imagery patterns
    SLEEP = "sleep"          # Theta/Delta waves (0.5-8 Hz)


class EEGGenerator:
    """Generates synthetic EEG data with realistic patterns"""
    
    def __init__(self, 
                 sample_rate: int = 250,
                 duration: float = 4.0,
                 n_channels: int = 1):
        """
        Initialize EEG generator
        
        Args:
            sample_rate: Sampling rate in Hz
            duration: Duration in seconds
            n_channels: Number of EEG channels
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_channels = n_channels
        self.time_points = int(sample_rate * duration)
        
        # Define frequency bands (Hz)
        self.freq_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def generate_brain_state_eeg(self, 
                                brain_state: BrainState,
                                noise_level: float = 0.1) -> np.ndarray:
        """
        Generate EEG signal for specific brain state
        
        Args:
            brain_state: Brain state to generate
            noise_level: Level of noise to add (0-1)
            
        Returns:
            EEG signal of shape (time_points, n_channels)
        """
        t = np.linspace(0, self.duration, self.time_points)
        eeg_signal = np.zeros((self.time_points, self.n_channels))
        
        if brain_state == BrainState.RELAXED:
            # Alpha waves - relaxed state
            alpha_freq = np.random.uniform(9, 12)  # 9-12 Hz alpha
            eeg_signal[:, 0] = (
                np.sin(2 * np.pi * alpha_freq * t) +
                0.3 * np.sin(2 * np.pi * alpha_freq * 2 * t) +  # harmonics
                0.1 * np.sin(2 * np.pi * 2 * t)  # slow wave
            )
            
        elif brain_state == BrainState.FOCUSED:
            # Beta waves - focused/alert state
            beta_freq = np.random.uniform(15, 25)  # 15-25 Hz beta
            eeg_signal[:, 0] = (
                np.sin(2 * np.pi * beta_freq * t) +
                0.5 * np.sin(2 * np.pi * beta_freq * 1.5 * t) +
                0.2 * np.sin(2 * np.pi * 40 * t)  # gamma component
            )
            
        elif brain_state == BrainState.ACTIVE:
            # Gamma waves - high cognitive activity
            gamma_freq = np.random.uniform(35, 45)  # 35-45 Hz gamma
            eeg_signal[:, 0] = (
                np.sin(2 * np.pi * gamma_freq * t) +
                0.4 * np.sin(2 * np.pi * gamma_freq * 0.7 * t) +
                0.3 * np.sin(2 * np.pi * 20 * t)  # beta component
            )
            
        elif brain_state == BrainState.MOTOR_IMAGERY:
            # Motor imagery - mu rhythm suppression and beta rebound
            mu_freq = np.random.uniform(8, 12)  # mu rhythm
            beta_freq = np.random.uniform(18, 25)  # beta rebound
            
            # Create event-related pattern
            event_time = self.duration * 0.3
            event_idx = int(event_time * self.sample_rate)
            
            # Pre-event (mu rhythm)
            eeg_signal[:event_idx, 0] = np.sin(2 * np.pi * mu_freq * t[:event_idx])
            
            # During event (suppression + beta)
            duration_idx = int(0.5 * self.sample_rate)
            event_signal = (
                -0.3 * np.sin(2 * np.pi * mu_freq * t[event_idx:event_idx+duration_idx]) +
                0.8 * np.sin(2 * np.pi * beta_freq * t[event_idx:event_idx+duration_idx])
            )
            eeg_signal[event_idx:event_idx+duration_idx, 0] = event_signal
            
            # Post-event (return to baseline)
            eeg_signal[event_idx+duration_idx:, 0] = (
                0.7 * np.sin(2 * np.pi * mu_freq * t[event_idx+duration_idx:])
            )
            
        elif brain_state == BrainState.SLEEP:
            # Theta and delta waves - sleep state
            theta_freq = np.random.uniform(4, 7)  # theta
            delta_freq = np.random.uniform(1, 3)  # delta
            
            eeg_signal[:, 0] = (
                0.6 * np.sin(2 * np.pi * theta_freq * t) +
                0.4 * np.sin(2 * np.pi * delta_freq * t)
            )
        
        # Add realistic artifacts and noise
        eeg_signal = self._add_artifacts(eeg_signal, brain_state)
        eeg_signal += noise_level * np.random.randn(*eeg_signal.shape)
        
        # Apply bandpass filter to remove unrealistic frequencies
        eeg_signal = self._apply_filter(eeg_signal)
        
        return eeg_signal
    
    def generate_mixed_state_eeg(self, 
                                primary_state: BrainState,
                                secondary_state: BrainState,
                                transition_time: float = 0.5) -> np.ndarray:
        """
        Generate EEG with state transition
        
        Args:
            primary_state: Initial brain state
            secondary_state: Final brain state
            transition_time: Time to transition (seconds)
            
        Returns:
            EEG signal with state transition
        """
        # Generate both states
        primary_eeg = self.generate_brain_state_eeg(primary_state)
        secondary_eeg = self.generate_brain_state_eeg(secondary_state)
        
        # Create smooth transition
        transition_samples = int(transition_time * self.sample_rate)
        transition_idx = self.time_points // 2
        
        eeg_mixed = primary_eeg.copy()
        
        # Blend signals in transition region
        for i in range(transition_samples):
            if transition_idx + i < self.time_points:
                alpha = i / transition_samples
                eeg_mixed[transition_idx + i, 0] = (
                    (1 - alpha) * primary_eeg[transition_idx + i, 0] +
                    alpha * secondary_eeg[transition_idx + i, 0]
                )
        
        # Use secondary state for remaining time
        remaining_start = transition_idx + transition_samples
        if remaining_start < self.time_points:
            eeg_mixed[remaining_start:, 0] = secondary_eeg[remaining_start:, 0]
        
        return eeg_mixed
    
    def _add_artifacts(self, eeg_signal: np.ndarray, brain_state: BrainState) -> np.ndarray:
        """Add realistic EEG artifacts"""
        signal = eeg_signal.copy()
        
        # Eye blink artifact (occasional large amplitude)
        if np.random.random() < 0.3:  # 30% chance
            blink_time = np.random.uniform(0.2, self.duration - 0.2)
            blink_idx = int(blink_time * self.sample_rate)
            blink_duration = int(0.1 * self.sample_rate)  # 100ms
            
            start_idx = max(0, blink_idx - blink_duration // 2)
            end_idx = min(self.time_points, blink_idx + blink_duration // 2)
            
            # Add eye blink (exponential decay)
            t_blink = np.linspace(0, 0.1, end_idx - start_idx)
            blink_artifact = 50 * np.exp(-t_blink * 30) * np.random.choice([-1, 1])
            signal[start_idx:end_idx, 0] += blink_artifact
        
        # Muscle artifact (high frequency noise)
        if np.random.random() < 0.2:  # 20% chance
            muscle_start = np.random.randint(0, self.time_points // 2)
            muscle_duration = int(0.3 * self.sample_rate)
            muscle_end = min(self.time_points, muscle_start + muscle_duration)
            
            # High frequency noise
            muscle_noise = 5 * np.random.randn(muscle_end - muscle_start)
            signal[muscle_start:muscle_end, 0] += muscle_noise
        
        # Baseline drift
        drift = 0.5 * np.sin(2 * np.pi * 0.1 * np.linspace(0, self.duration, self.time_points))
        signal[:, 0] += drift
        
        return signal
    
    def _apply_filter(self, eeg_signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove unrealistic frequencies"""
        # Design bandpass filter (0.5-50 Hz)
        nyquist = self.sample_rate / 2
        low = 0.5 / nyquist
        high = 50 / nyquist
        
        # Use Butterworth filter
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        
        filtered_signal = np.zeros_like(eeg_signal)
        for ch in range(self.n_channels):
            filtered_signal[:, ch] = scipy.signal.filtfilt(b, a, eeg_signal[:, ch])
        
        return filtered_signal
    
    def generate_dataset(self, 
                        n_samples: int = 1000,
                        states: List[BrainState] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate dataset of EEG samples with labels
        
        Args:
            n_samples: Number of samples to generate
            states: List of brain states to include
            
        Returns:
            Tuple of (eeg_data, labels)
            eeg_data: Shape (n_samples, time_points, n_channels)
            labels: Shape (n_samples,) - brain state indices
        """
        if states is None:
            states = list(BrainState)
        
        eeg_data = np.zeros((n_samples, self.time_points, self.n_channels))
        labels = np.zeros(n_samples, dtype=int)
        
        samples_per_state = n_samples // len(states)
        
        for i, state in enumerate(states):
            start_idx = i * samples_per_state
            end_idx = start_idx + samples_per_state
            
            if i == len(states) - 1:  # Last state gets remaining samples
                end_idx = n_samples
            
            for j in range(start_idx, end_idx):
                eeg_data[j] = self.generate_brain_state_eeg(state)
                labels[j] = i
        
        # Shuffle the dataset
        indices = np.random.permutation(n_samples)
        eeg_data = eeg_data[indices]
        labels = labels[indices]
        
        return eeg_data, labels


def create_synthetic_eeg_dataset(n_samples: int = 1000,
                                sample_rate: int = 250,
                                duration: float = 4.0) -> Dict:
    """
    Create a complete synthetic EEG dataset for training
    
    Args:
        n_samples: Total number of samples
        sample_rate: Sampling rate in Hz
        duration: Duration in seconds
        
    Returns:
        Dictionary containing dataset and metadata
    """
    generator = EEGGenerator(sample_rate=sample_rate, duration=duration)
    
    # Generate dataset
    eeg_data, labels = generator.generate_dataset(n_samples)
    
    # Create train/validation split
    n_train = int(0.8 * n_samples)
    train_indices = np.random.choice(n_samples, n_train, replace=False)
    val_indices = np.setdiff1d(np.arange(n_samples), train_indices)
    
    dataset = {
        'train_eeg': eeg_data[train_indices],
        'train_labels': labels[train_indices],
        'val_eeg': eeg_data[val_indices],
        'val_labels': labels[val_indices],
        'states': list(BrainState),
        'sample_rate': sample_rate,
        'duration': duration,
        'time_points': generator.time_points,
        'n_channels': generator.n_channels
    }
    
    return dataset


if __name__ == "__main__":
    # Demo usage
    print("Generating synthetic EEG dataset...")
    
    # Create generator
    generator = EEGGenerator(sample_rate=250, duration=4.0)
    
    # Generate sample for each brain state
    for state in BrainState:
        print(f"Generating {state.value} state EEG...")
        eeg = generator.generate_brain_state_eeg(state)
        print(f"  Shape: {eeg.shape}, Range: [{eeg.min():.2f}, {eeg.max():.2f}]")
    
    # Generate mixed state
    print("Generating mixed state EEG (relaxed -> focused)...")
    mixed_eeg = generator.generate_mixed_state_eeg(
        BrainState.RELAXED, 
        BrainState.FOCUSED,
        transition_time=1.0
    )
    print(f"  Shape: {mixed_eeg.shape}")
    
    # Generate full dataset
    print("Creating full dataset...")
    dataset = create_synthetic_eeg_dataset(n_samples=100)
    print(f"Dataset created with {len(dataset['train_eeg'])} train and {len(dataset['val_eeg'])} val samples")
