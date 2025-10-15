"""
Evaluation Metrics for BrainWave Analyzer
Comprehensive metrics for EEG and image quality assessment
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
warnings.filterwarnings('ignore')


class EEGMetrics:
    """EEG-specific evaluation metrics"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Pearson correlation coefficient"""
        if len(y_true) != len(y_pred):
            raise ValueError("Input arrays must have the same length")
        
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0
        
        corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
        return float(corr)
    
    @staticmethod
    def frequency_correlation(y_true: np.ndarray, y_pred: np.ndarray, 
                            sample_rate: int = 250) -> float:
        """Correlation in frequency domain"""
        from scipy.fft import fft
        
        # Compute FFT
        fft_true = np.abs(fft(y_true.flatten()))
        fft_pred = np.abs(fft(y_pred.flatten()))
        
        # Ensure same length
        min_len = min(len(fft_true), len(fft_pred))
        fft_true = fft_true[:min_len]
        fft_pred = fft_pred[:min_len]
        
        if np.std(fft_true) == 0 or np.std(fft_pred) == 0:
            return 0.0
        
        corr, _ = pearsonr(fft_true, fft_pred)
        return float(corr)
    
    @staticmethod
    def spectral_distance(y_true: np.ndarray, y_pred: np.ndarray, 
                         sample_rate: int = 250) -> float:
        """Spectral distance between signals"""
        from scipy.signal import welch
        
        # Compute power spectral density
        freqs_true, psd_true = welch(y_true.flatten(), fs=sample_rate, nperseg=min(256, len(y_true)//4))
        freqs_pred, psd_pred = welch(y_pred.flatten(), fs=sample_rate, nperseg=min(256, len(y_pred)//4))
        
        # Interpolate to same frequency grid
        freq_min = max(freqs_true[0], freqs_pred[0])
        freq_max = min(freqs_true[-1], freqs_pred[-1])
        freq_grid = np.linspace(freq_min, freq_max, 100)
        
        psd_true_interp = np.interp(freq_grid, freqs_true, psd_true)
        psd_pred_interp = np.interp(freq_grid, freqs_pred, psd_pred)
        
        # Compute normalized spectral distance
        distance = euclidean(psd_true_interp, psd_pred_interp)
        max_distance = euclidean(psd_true_interp, np.zeros_like(psd_true_interp))
        
        return float(distance / (max_distance + 1e-8))
    
    @staticmethod
    def dynamic_time_warping_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simplified DTW distance"""
        try:
            from dtaidistance import dtw
            distance = dtw.distance(y_true.flatten(), y_pred.flatten())
            return float(distance)
        except ImportError:
            # Fallback to simple Euclidean distance
            return float(euclidean(y_true.flatten(), y_pred.flatten()))
    
    @staticmethod
    def signal_to_noise_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Signal to noise ratio"""
        signal_power = np.mean(y_true ** 2)
        noise_power = np.mean((y_true - y_pred) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)


class ImageMetrics:
    """Image-specific evaluation metrics"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Peak Signal to Noise Ratio"""
        try:
            # Ensure images are in [0, 1] range
            y_true = np.clip(y_true, 0, 1)
            y_pred = np.clip(y_pred, 0, 1)
            
            # Convert to [0, 255] for PSNR calculation
            y_true_255 = (y_true * 255).astype(np.uint8)
            y_pred_255 = (y_pred * 255).astype(np.uint8)
            
            if len(y_true.shape) == 3:
                # Multi-channel image
                psnr_values = []
                for i in range(y_true.shape[2]):
                    psnr_val = psnr(y_true_255[:, :, i], y_pred_255[:, :, i])
                    psnr_values.append(psnr_val)
                return float(np.mean(psnr_values))
            else:
                # Single channel image
                return float(psnr(y_true_255, y_pred_255))
        except Exception:
            return 0.0
    
    @staticmethod
    def ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Structural Similarity Index"""
        try:
            # Ensure images are in [0, 1] range
            y_true = np.clip(y_true, 0, 1)
            y_pred = np.clip(y_pred, 0, 1)
            
            if len(y_true.shape) == 3:
                # Multi-channel image
                ssim_values = []
                for i in range(y_true.shape[2]):
                    ssim_val = ssim(y_true[:, :, i], y_pred[:, :, i], 
                                  data_range=1.0, win_size=min(7, min(y_true.shape[:2])))
                    ssim_values.append(ssim_val)
                return float(np.mean(ssim_values))
            else:
                # Single channel image
                return float(ssim(y_true, y_pred, data_range=1.0, 
                                win_size=min(7, min(y_true.shape))))
        except Exception:
            return 0.0
    
    @staticmethod
    def lpips(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Learned Perceptual Image Patch Similarity (simplified version)"""
        try:
            # Simplified LPIPS using VGG features
            import tensorflow as tf
            from tensorflow.keras.applications import VGG16
            
            # Load pre-trained VGG
            vgg = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
            
            # Preprocess images
            y_true_processed = tf.keras.applications.vgg16.preprocess_input(y_true * 255.0)
            y_pred_processed = tf.keras.applications.vgg16.preprocess_input(y_pred * 255.0)
            
            # Extract features
            true_features = vgg(y_true_processed)
            pred_features = vgg(y_pred_processed)
            
            # Compute perceptual distance
            distance = tf.reduce_mean(tf.square(true_features - pred_features))
            return float(distance.numpy())
        except Exception:
            return 0.0
    
    @staticmethod
    def color_histogram_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Color histogram similarity"""
        try:
            if len(y_true.shape) != 3 or y_true.shape[2] != 3:
                return 0.0
            
            # Compute histograms for each color channel
            similarities = []
            for i in range(3):
                hist_true, _ = np.histogram(y_true[:, :, i], bins=32, range=(0, 1))
                hist_pred, _ = np.histogram(y_pred[:, :, i], bins=32, range=(0, 1))
                
                # Normalize histograms
                hist_true = hist_true / (np.sum(hist_true) + 1e-8)
                hist_pred = hist_pred / (np.sum(hist_pred) + 1e-8)
                
                # Compute histogram similarity (Bhattacharyya coefficient)
                similarity = np.sum(np.sqrt(hist_true * hist_pred))
                similarities.append(similarity)
            
            return float(np.mean(similarities))
        except Exception:
            return 0.0


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.eeg_metrics = EEGMetrics()
        self.image_metrics = ImageMetrics()
    
    def evaluate_eeg_to_image(self, 
                             model: Any,
                             test_eeg: np.ndarray,
                             test_images: np.ndarray,
                             metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate EEG to Image model
        
        Args:
            model: Trained EEG to Image model
            test_eeg: Test EEG data
            test_images: Ground truth test images
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = ['mse', 'mae', 'psnr', 'ssim', 'color_histogram_similarity']
        
        print(f"Evaluating EEG to Image model on {len(test_eeg)} samples...")
        
        # Generate predictions
        predictions = []
        for i in range(len(test_eeg)):
            pred = model.predict_image_from_eeg(test_eeg[i])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute metrics
        results = {}
        for metric in metrics:
            if metric in ['mse', 'mae']:
                # Basic metrics
                if metric == 'mse':
                    results[metric] = self.image_metrics.mse(test_images, predictions)
                elif metric == 'mae':
                    results[metric] = self.image_metrics.mae(test_images, predictions)
            
            elif metric == 'psnr':
                results[metric] = self.image_metrics.psnr(test_images, predictions)
            
            elif metric == 'ssim':
                results[metric] = self.image_metrics.ssim(test_images, predictions)
            
            elif metric == 'color_histogram_similarity':
                results[metric] = self.image_metrics.color_histogram_similarity(
                    test_images, predictions
                )
            
            elif metric == 'lpips':
                results[metric] = self.image_metrics.lpips(test_images, predictions)
            
            else:
                print(f"Unknown metric: {metric}")
        
        return results
    
    def evaluate_image_to_eeg(self, 
                             model: Any,
                             test_images: np.ndarray,
                             test_eeg: np.ndarray,
                             metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate Image to EEG model
        
        Args:
            model: Trained Image to EEG model
            test_images: Test images
            test_eeg: Ground truth test EEG data
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        if metrics is None:
            metrics = ['mse', 'mae', 'correlation_coefficient', 'frequency_correlation', 'spectral_distance']
        
        print(f"Evaluating Image to EEG model on {len(test_images)} samples...")
        
        # Generate predictions
        predictions = []
        for i in range(len(test_images)):
            pred = model.predict_eeg_from_image(test_images[i])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute metrics
        results = {}
        for metric in metrics:
            if metric in ['mse', 'mae']:
                # Basic metrics
                if metric == 'mse':
                    results[metric] = self.eeg_metrics.mse(test_eeg, predictions)
                elif metric == 'mae':
                    results[metric] = self.eeg_metrics.mae(test_eeg, predictions)
            
            elif metric == 'correlation_coefficient':
                # Compute average correlation across all samples
                correlations = []
                for i in range(len(test_eeg)):
                    corr = self.eeg_metrics.correlation_coefficient(test_eeg[i], predictions[i])
                    correlations.append(corr)
                results[metric] = float(np.mean(correlations))
            
            elif metric == 'frequency_correlation':
                # Compute average frequency domain correlation
                freq_correlations = []
                for i in range(len(test_eeg)):
                    freq_corr = self.eeg_metrics.frequency_correlation(test_eeg[i], predictions[i])
                    freq_correlations.append(freq_corr)
                results[metric] = float(np.mean(freq_correlations))
            
            elif metric == 'spectral_distance':
                # Compute average spectral distance
                spectral_distances = []
                for i in range(len(test_eeg)):
                    spec_dist = self.eeg_metrics.spectral_distance(test_eeg[i], predictions[i])
                    spectral_distances.append(spec_dist)
                results[metric] = float(np.mean(spectral_distances))
            
            elif metric == 'signal_to_noise_ratio':
                # Compute average SNR
                snrs = []
                for i in range(len(test_eeg)):
                    snr = self.eeg_metrics.signal_to_noise_ratio(test_eeg[i], predictions[i])
                    snrs.append(snr)
                results[metric] = float(np.mean(snrs))
            
            else:
                print(f"Unknown metric: {metric}")
        
        return results
    
    def evaluate_bidirectional_mapping(self, 
                                      eeg_to_image_model: Any,
                                      image_to_eeg_model: Any,
                                      test_eeg: np.ndarray,
                                      test_images: np.ndarray) -> Dict[str, float]:
        """
        Evaluate bidirectional mapping consistency
        
        Args:
            eeg_to_image_model: EEG to Image model
            image_to_eeg_model: Image to EEG model
            test_eeg: Test EEG data
            test_images: Test images
            
        Returns:
            Dictionary of bidirectional consistency metrics
        """
        print("Evaluating bidirectional mapping consistency...")
        
        # EEG -> Image -> EEG cycle
        eeg_to_image_predictions = []
        eeg_cycle_predictions = []
        
        for i in range(len(test_eeg)):
            # EEG -> Image
            image_pred = eeg_to_image_model.predict_image_from_eeg(test_eeg[i])
            eeg_to_image_predictions.append(image_pred)
            
            # Image -> EEG
            eeg_pred = image_to_eeg_model.predict_eeg_from_image(image_pred)
            eeg_cycle_predictions.append(eeg_pred)
        
        eeg_to_image_predictions = np.array(eeg_to_image_predictions)
        eeg_cycle_predictions = np.array(eeg_cycle_predictions)
        
        # Image -> EEG -> Image cycle
        image_to_eeg_predictions = []
        image_cycle_predictions = []
        
        for i in range(len(test_images)):
            # Image -> EEG
            eeg_pred = image_to_eeg_model.predict_eeg_from_image(test_images[i])
            image_to_eeg_predictions.append(eeg_pred)
            
            # EEG -> Image
            image_pred = eeg_to_image_model.predict_image_from_eeg(eeg_pred)
            image_cycle_predictions.append(image_pred)
        
        image_to_eeg_predictions = np.array(image_to_eeg_predictions)
        image_cycle_predictions = np.array(image_cycle_predictions)
        
        # Compute cycle consistency metrics
        results = {}
        
        # EEG cycle consistency
        results['eeg_cycle_mse'] = self.eeg_metrics.mse(test_eeg, eeg_cycle_predictions)
        results['eeg_cycle_mae'] = self.eeg_metrics.mae(test_eeg, eeg_cycle_predictions)
        
        eeg_correlations = []
        for i in range(len(test_eeg)):
            corr = self.eeg_metrics.correlation_coefficient(test_eeg[i], eeg_cycle_predictions[i])
            eeg_correlations.append(corr)
        results['eeg_cycle_correlation'] = float(np.mean(eeg_correlations))
        
        # Image cycle consistency
        results['image_cycle_mse'] = self.image_metrics.mse(test_images, image_cycle_predictions)
        results['image_cycle_mae'] = self.image_metrics.mae(test_images, image_cycle_predictions)
        results['image_cycle_psnr'] = self.image_metrics.psnr(test_images, image_cycle_predictions)
        results['image_cycle_ssim'] = self.image_metrics.ssim(test_images, image_cycle_predictions)
        
        return results
    
    def generate_evaluation_report(self, 
                                  eeg_to_image_results: Dict[str, float],
                                  image_to_eeg_results: Dict[str, float],
                                  bidirectional_results: Dict[str, float] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            eeg_to_image_results: EEG to Image evaluation results
            image_to_eeg_results: Image to EEG evaluation results
            bidirectional_results: Bidirectional evaluation results
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("=" * 60)
        report.append("BRAINWAVE ANALYZER - MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # EEG to Image Results
        report.append("EEG TO IMAGE MODEL EVALUATION:")
        report.append("-" * 40)
        for metric, value in eeg_to_image_results.items():
            report.append(f"  {metric:25}: {value:8.4f}")
        report.append("")
        
        # Image to EEG Results
        report.append("IMAGE TO EEG MODEL EVALUATION:")
        report.append("-" * 40)
        for metric, value in image_to_eeg_results.items():
            report.append(f"  {metric:25}: {value:8.4f}")
        report.append("")
        
        # Bidirectional Results
        if bidirectional_results:
            report.append("BIDIRECTIONAL MAPPING CONSISTENCY:")
            report.append("-" * 40)
            for metric, value in bidirectional_results.items():
                report.append(f"  {metric:25}: {value:8.4f}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        
        # Best performing metrics
        best_eeg_to_image = max(eeg_to_image_results.items(), key=lambda x: x[1] if 'mse' not in x[0] and 'mae' not in x[0] else -x[1])
        best_image_to_eeg = max(image_to_eeg_results.items(), key=lambda x: x[1] if 'mse' not in x[0] and 'mae' not in x[0] else -x[1])
        
        report.append(f"  Best EEG->Image metric: {best_eeg_to_image[0]} = {best_eeg_to_image[1]:.4f}")
        report.append(f"  Best Image->EEG metric: {best_image_to_eeg[0]} = {best_image_to_eeg[1]:.4f}")
        
        if bidirectional_results:
            best_bidirectional = max(bidirectional_results.items(), key=lambda x: x[1] if 'mse' not in x[0] and 'mae' not in x[0] else -x[1])
            report.append(f"  Best bidirectional metric: {best_bidirectional[0]} = {best_bidirectional[1]:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    print("Testing evaluation metrics...")
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Test EEG metrics
    print("Testing EEG metrics...")
    eeg_true = np.random.randn(100)
    eeg_pred = eeg_true + 0.1 * np.random.randn(100)
    
    eeg_mse = evaluator.eeg_metrics.mse(eeg_true, eeg_pred)
    eeg_mae = evaluator.eeg_metrics.mae(eeg_true, eeg_pred)
    eeg_corr = evaluator.eeg_metrics.correlation_coefficient(eeg_true, eeg_pred)
    
    print(f"EEG MSE: {eeg_mse:.4f}")
    print(f"EEG MAE: {eeg_mae:.4f}")
    print(f"EEG Correlation: {eeg_corr:.4f}")
    
    # Test image metrics
    print("Testing image metrics...")
    img_true = np.random.uniform(0, 1, (64, 64, 3))
    img_pred = img_true + 0.1 * np.random.randn(64, 64, 3)
    img_pred = np.clip(img_pred, 0, 1)
    
    img_mse = evaluator.image_metrics.mse(img_true, img_pred)
    img_mae = evaluator.image_metrics.mae(img_true, img_pred)
    img_psnr = evaluator.image_metrics.psnr(img_true, img_pred)
    img_ssim = evaluator.image_metrics.ssim(img_true, img_pred)
    
    print(f"Image MSE: {img_mse:.4f}")
    print(f"Image MAE: {img_mae:.4f}")
    print(f"Image PSNR: {img_psnr:.4f}")
    print(f"Image SSIM: {img_ssim:.4f}")
    
    print("Evaluation metrics test completed!")
