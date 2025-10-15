"""
Visualization Utilities for BrainWave Analyzer
Advanced plotting and visualization functions for EEG and images
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import cv2


class EEGVisualizer:
    """Advanced EEG signal visualization"""
    
    def __init__(self, sample_rate: int = 250):
        """
        Initialize EEG visualizer
        
        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
        
        # Define frequency bands
        self.freq_bands = {
            'Delta': (0.5, 4, '#FF6B6B'),
            'Theta': (4, 8, '#4ECDC4'),
            'Alpha': (8, 13, '#45B7D1'),
            'Beta': (13, 30, '#96CEB4'),
            'Gamma': (30, 50, '#FFEAA7')
        }
    
    def create_time_series_plot(self, 
                               eeg_data: np.ndarray,
                               title: str = "EEG Signal",
                               show_bands: bool = True,
                               interactive: bool = True) -> go.Figure:
        """
        Create interactive time series plot
        
        Args:
            eeg_data: EEG signal data
            title: Plot title
            show_bands: Whether to show frequency band annotations
            interactive: Whether to create interactive plot
            
        Returns:
            Plotly figure
        """
        # Create time axis
        time_axis = np.linspace(0, len(eeg_data) / self.sample_rate, len(eeg_data))
        
        if interactive:
            fig = go.Figure()
            
            # Main signal
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=eeg_data,
                mode='lines',
                name='EEG Signal',
                line=dict(color='#667eea', width=2),
                hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>'
            ))
            
            # Add frequency band annotations
            if show_bands:
                for band_name, (low_freq, high_freq, color) in self.freq_bands.items():
                    fig.add_annotation(
                        text=f"{band_name}<br>({low_freq}-{high_freq} Hz)",
                        xref="paper", yref="paper",
                        x=0.02 + (list(self.freq_bands.keys()).index(band_name) * 0.15),
                        y=0.95,
                        showarrow=False,
                        bgcolor=color,
                        bordercolor="white",
                        borderwidth=1,
                        font=dict(color="white", size=10)
                    )
            
            fig.update_layout(
                title=title,
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude (μV)",
                template="plotly_white",
                height=400,
                showlegend=False,
                hovermode='x unified'
            )
            
            return fig
        else:
            # Static matplotlib plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_axis, eeg_data, 'b-', linewidth=1.5)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Amplitude (μV)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            if show_bands:
                # Add frequency band legend
                legend_elements = []
                for band_name, (low_freq, high_freq, color) in self.freq_bands.items():
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=3, 
                                                     label=f'{band_name} ({low_freq}-{high_freq} Hz)'))
                ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            return fig
    
    def create_spectral_plot(self, 
                           eeg_data: np.ndarray,
                           title: str = "Power Spectral Density",
                           interactive: bool = True) -> go.Figure:
        """
        Create power spectral density plot
        
        Args:
            eeg_data: EEG signal data
            title: Plot title
            interactive: Whether to create interactive plot
            
        Returns:
            Plotly figure
        """
        from scipy.fft import fft, fftfreq
        
        # Compute FFT
        fft_data = np.abs(fft(eeg_data))
        freqs = fftfreq(len(eeg_data), 1/self.sample_rate)
        
        # Only show positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_data[:len(fft_data)//2]
        
        if interactive:
            fig = go.Figure()
            
            # Power spectrum
            fig.add_trace(go.Scatter(
                x=positive_freqs,
                y=positive_fft,
                mode='lines',
                name='Power Spectrum',
                line=dict(color='#764ba2', width=2),
                fill='tonexty'
            ))
            
            # Add frequency band regions
            for band_name, (low_freq, high_freq, color) in self.freq_bands.items():
                fig.add_vrect(
                    x0=low_freq, x1=high_freq,
                    fillcolor=color,
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                    annotation_text=band_name,
                    annotation_position="top"
                )
            
            fig.update_layout(
                title=title,
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power",
                template="plotly_white",
                height=350,
                showlegend=False,
                xaxis=dict(range=[0, 50])
            )
            
            return fig
        else:
            # Static matplotlib plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.semilogy(positive_freqs, positive_fft, 'r-', linewidth=2)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)', fontsize=12)
            ax.set_ylabel('Power', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 50)
            
            # Add frequency band regions
            for band_name, (low_freq, high_freq, color) in self.freq_bands.items():
                ax.axvspan(low_freq, high_freq, alpha=0.3, color=color, label=band_name)
            
            ax.legend()
            plt.tight_layout()
            return fig
    
    def create_spectrogram(self, 
                          eeg_data: np.ndarray,
                          title: str = "EEG Spectrogram") -> go.Figure:
        """
        Create EEG spectrogram
        
        Args:
            eeg_data: EEG signal data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        from scipy.signal import spectrogram
        
        # Compute spectrogram
        freqs, times, Sxx = spectrogram(eeg_data, fs=self.sample_rate, nperseg=64)
        
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=10 * np.log10(Sxx),
            x=times,
            y=freqs,
            colorscale='Viridis',
            name='Power (dB)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_comparison_plot(self, 
                              original: np.ndarray,
                              predicted: np.ndarray,
                              title: str = "EEG Comparison") -> go.Figure:
        """
        Create comparison plot between original and predicted EEG
        
        Args:
            original: Original EEG signal
            predicted: Predicted EEG signal
            title: Plot title
            
        Returns:
            Plotly figure
        """
        time_axis = np.linspace(0, len(original) / self.sample_rate, len(original))
        
        fig = go.Figure()
        
        # Original signal
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=original,
            mode='lines',
            name='Original',
            line=dict(color='#667eea', width=2)
        ))
        
        # Predicted signal
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=predicted,
            mode='lines',
            name='Predicted',
            line=dict(color='#ff6b6b', width=2)
        ))
        
        # Difference
        diff = original - predicted
        fig.add_trace(go.Scatter(
            x=time_axis,
            y=diff,
            mode='lines',
            name='Difference',
            line=dict(color='#95a5a6', width=1, dash='dot')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (μV)",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        return fig


class ImageVisualizer:
    """Advanced image visualization utilities"""
    
    def __init__(self):
        """Initialize image visualizer"""
        pass
    
    def create_image_grid(self, 
                         images: List[np.ndarray],
                         titles: List[str] = None,
                         n_cols: int = 4,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create grid of images
        
        Args:
            images: List of image arrays
            titles: List of titles for each image
            n_cols: Number of columns in grid
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_images = len(images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i in range(n_images):
            img = images[i]
            if img.ndim == 3:
                axes[i].imshow(img)
            else:
                axes[i].imshow(img, cmap='gray')
            
            axes[i].axis('off')
            
            if titles and i < len(titles):
                axes[i].set_title(titles[i], fontsize=10, fontweight='bold')
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, 
                              original: np.ndarray,
                              generated: np.ndarray,
                              title: str = "Image Comparison") -> go.Figure:
        """
        Create side-by-side comparison of original and generated images
        
        Args:
            original: Original image
            generated: Generated image
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Original', 'Generated'],
            horizontal_spacing=0.1
        )
        
        # Original image
        fig.add_trace(
            go.Image(z=original, name='Original'),
            row=1, col=1
        )
        
        # Generated image
        fig.add_trace(
            go.Image(z=generated, name='Generated'),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_attention_heatmap(self, 
                                image: np.ndarray,
                                attention_map: np.ndarray,
                                title: str = "Attention Visualization") -> go.Figure:
        """
        Create attention heatmap overlay
        
        Args:
            image: Original image
            attention_map: Attention weights
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Original', 'Attention Map', 'Overlay'],
            horizontal_spacing=0.1
        )
        
        # Original image
        fig.add_trace(
            go.Image(z=image, name='Original'),
            row=1, col=1
        )
        
        # Attention map
        fig.add_trace(
            go.Heatmap(z=attention_map, colorscale='Hot', name='Attention'),
            row=1, col=2
        )
        
        # Overlay
        overlay = image * attention_map[..., np.newaxis]
        fig.add_trace(
            go.Image(z=overlay, name='Overlay'),
            row=1, col=3
        )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_histogram_comparison(self, 
                                   original: np.ndarray,
                                   generated: np.ndarray,
                                   title: str = "Histogram Comparison") -> go.Figure:
        """
        Create histogram comparison of original and generated images
        
        Args:
            original: Original image
            generated: Generated image
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Original histogram
        fig.add_trace(go.Histogram(
            x=original.flatten(),
            name='Original',
            opacity=0.7,
            nbinsx=50
        ))
        
        # Generated histogram
        fig.add_trace(go.Histogram(
            x=generated.flatten(),
            name='Generated',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Pixel Value",
            yaxis_title="Count",
            template="plotly_white",
            height=400,
            barmode='overlay'
        )
        
        return fig


class ModelVisualizer:
    """Model architecture and performance visualization"""
    
    def __init__(self):
        """Initialize model visualizer"""
        pass
    
    def create_architecture_diagram(self) -> str:
        """
        Create text-based architecture diagram
        
        Returns:
            ASCII architecture diagram
        """
        diagram = """
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   EEG Input     │    │   Latent Space   │    │ Generated Image │
│  (100 samples)  │───▶│   (256D) +       │───▶│   (64×64×3)     │
│                 │    │  Attention       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
    ┌─────────┐             ┌──────────┐           ┌─────────────┐
    │ LSTM    │             │ Shared   │           │ CNN Decoder │
    │ Encoder │             │ Embedding│           │ (Progressive│
    │ + Attn  │             │ Space    │           │  Upsampling)│
    └─────────┘             └──────────┘           └─────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Generated EEG   │    │   Latent Space   │    │   Image Input   │
│ (100 samples)   │◀───│   (256D) +       │◀───│   (64×64×3)     │
│                 │    │  Attention       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
    ┌─────────────┐         ┌──────────┐           ┌─────────┐
    │ LSTM        │         │ Shared   │           │ CNN     │
    │ Decoder     │         │ Embedding│           │ Encoder │
    │ + Attn      │         │ Space    │           │ + Attn  │
    └─────────────┘         └──────────┘           └─────────┘
        """
        return diagram
    
    def create_training_curves(self, 
                              history: Dict[str, List[float]],
                              title: str = "Training Progress") -> go.Figure:
        """
        Create training progress visualization
        
        Args:
            history: Training history dictionary
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for metric_name, values in history.items():
            fig.add_trace(go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_performance_metrics(self, 
                                  metrics: Dict[str, float],
                                  title: str = "Model Performance") -> go.Figure:
        """
        Create performance metrics visualization
        
        Args:
            metrics: Dictionary of metric names and values
            title: Plot title
            
        Returns:
            Plotly figure
        """
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=metric_names,
                y=metric_values,
                marker_color=['#667eea' if v > 0.5 else '#ff6b6b' for v in metric_values],
                text=[f'{v:.3f}' for v in metric_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Score",
            template="plotly_white",
            height=400
        )
        
        return fig


def create_combined_dashboard(eeg_data: np.ndarray,
                             generated_image: np.ndarray,
                             metrics: Dict[str, float] = None) -> go.Figure:
    """
    Create comprehensive dashboard combining multiple visualizations
    
    Args:
        eeg_data: EEG signal data
        generated_image: Generated image
        metrics: Performance metrics
        
    Returns:
        Combined Plotly figure
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['EEG Signal', 'Generated Image', 'Power Spectrum', 'Performance Metrics'],
        specs=[[{"secondary_y": False}, {"type": "image"}],
               [{"secondary_y": False}, {"type": "bar"}]]
    )
    
    # EEG signal
    time_axis = np.linspace(0, len(eeg_data) / 250, len(eeg_data))
    fig.add_trace(
        go.Scatter(x=time_axis, y=eeg_data, mode='lines', name='EEG'),
        row=1, col=1
    )
    
    # Generated image
    fig.add_trace(
        go.Image(z=generated_image, name='Image'),
        row=1, col=2
    )
    
    # Power spectrum
    from scipy.fft import fft, fftfreq
    fft_data = np.abs(fft(eeg_data))
    freqs = fftfreq(len(eeg_data), 1/250)
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_data[:len(fft_data)//2]
    
    fig.add_trace(
        go.Scatter(x=positive_freqs, y=positive_fft, mode='lines', name='Spectrum'),
        row=2, col=1
    )
    
    # Performance metrics
    if metrics:
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Metrics'),
            row=2, col=2
        )
    
    fig.update_layout(
        title="BrainWave Analyzer Dashboard",
        height=800,
        showlegend=False
    )
    
    return fig


if __name__ == "__main__":
    # Demo usage
    print("Testing visualization utilities...")
    
    # Create sample data
    t = np.linspace(0, 4, 1000)
    eeg_sample = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    image_sample = np.random.rand(64, 64, 3)
    
    # Test visualizers
    eeg_viz = EEGVisualizer()
    img_viz = ImageVisualizer()
    model_viz = ModelVisualizer()
    
    print("✅ EEG Visualizer created")
    print("✅ Image Visualizer created")
    print("✅ Model Visualizer created")
    
    # Test plotting functions
    try:
        fig1 = eeg_viz.create_time_series_plot(eeg_sample)
        print("✅ Time series plot created")
        
        fig2 = eeg_viz.create_spectral_plot(eeg_sample)
        print("✅ Spectral plot created")
        
        fig3 = img_viz.create_comparison_plot(image_sample, image_sample)
        print("✅ Image comparison plot created")
        
        print("✅ All visualization utilities tested successfully!")
        
    except Exception as e:
        print(f"❌ Error in visualization testing: {e}")
