# 🧠 BrainWave Analyzer v2.0

**Advanced Multi-Modal Deep Learning Platform for EEG-Image Synthesis**

A cutting-edge demonstration of multi-modal AI that creates bidirectional mappings between EEG brain signals and visual imagery using state-of-the-art CNN-RNN hybrid architectures with attention mechanisms.

## 🎯 Project Overview

BrainWave Analyzer showcases the fascinating intersection of neuroscience and computer vision through advanced deep learning. The system demonstrates:

- **EEG → Image Synthesis**: Convert brainwave signals into generated images
- **Image → EEG Prediction**: Convert images into synthetic brainwave patterns
- **Real-time Processing**: Fast inference with professional APIs
- **Interactive Visualization**: Rich web interface with advanced analytics

## 🚀 Quick Start

### Option 1: One-Command Demo (Recommended)
```bash
python run_demo.py
```

### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server (Terminal 1)
python api.py

# 3. Start the Streamlit interface (Terminal 2)
streamlit run app.py
```

### Option 3: Training Models (Optional)
```bash
# Train models with demo configuration
python -m training.train --config demo --epochs 10

# Train models with full configuration
python -m training.train --config full --epochs 50
```

## 🏗️ Architecture

### Model Architecture

```
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
```

### Key Features

- **Hybrid CNN-RNN Architecture**: Optimal combination of spatial and temporal processing
- **Attention Mechanisms**: Temporal attention for EEG sequences, spatial attention for images
- **Multi-Modal Latent Space**: Shared 256-dimensional embedding space
- **Progressive Generation**: Multi-scale image generation with skip connections
- **Advanced Loss Functions**: Perceptual loss, DTW loss, and frequency domain losses

## 📊 Technical Specifications

### Model Parameters
- **EEG to Image Model**: ~500K parameters
- **Image to EEG Model**: ~400K parameters
- **Latent Space Dimension**: 256
- **EEG Sequence Length**: 100 samples (4 seconds at 250 Hz)
- **Image Resolution**: 64×64×3 RGB
- **Attention Heads**: 4 (temporal), spatial attention maps

### Performance Metrics
- **Inference Time**: <100ms per prediction
- **Memory Usage**: ~200MB for models
- **Supported Formats**: PNG, JPG, JPEG (images); CSV (EEG)
- **API Response Time**: <500ms including processing

## 🛠️ Technical Stack

### Backend
- **Python 3.8+**: Core language
- **TensorFlow 2.x**: Deep learning framework
- **FastAPI**: High-performance API framework
- **NumPy, SciPy**: Numerical computing
- **OpenCV**: Image processing
- **PIL/Pillow**: Image manipulation

### Frontend
- **Streamlit**: Interactive web interface
- **Plotly**: Advanced visualizations
- **Custom CSS**: Professional styling

### Machine Learning
- **Custom Architectures**: CNN-RNN hybrids
- **Attention Mechanisms**: Temporal and spatial
- **Advanced Losses**: Perceptual, DTW, frequency domain
- **Data Generation**: Synthetic EEG-image pairs

## 📁 Project Structure

```
BrainWave/backend/brainwave_app/
├── data/                          # Data generation modules
│   ├── __init__.py
│   ├── synthetic_generator.py     # EEG pattern generation
│   ├── image_generator.py         # Image generation
│   └── dataset.py                 # Dataset management
├── models/                        # Model architectures
│   ├── __init__.py
│   ├── eeg_to_image.py           # EEG to Image model
│   ├── image_to_eeg.py           # Image to EEG model
│   ├── attention.py              # Attention mechanisms
│   └── losses.py                 # Custom loss functions
├── training/                      # Training pipeline
│   ├── __init__.py
│   ├── train.py                  # Training script
│   └── config.py                 # Configuration management
├── evaluation/                    # Evaluation metrics
│   ├── __init__.py
│   └── metrics.py                # Performance metrics
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── signal_processing.py      # EEG processing
│   └── visualization.py          # Visualization helpers
├── checkpoints/                   # Model weights (generated)
├── api.py                        # FastAPI server
├── app.py                        # Streamlit interface
├── models.py                     # Model wrapper
├── run_demo.py                   # Demo launcher
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔧 API Documentation

### Endpoints

#### Health Check
```http
GET /health
```
Returns API status and model information.

#### EEG to Image
```http
POST /api/eeg-to-image
Content-Type: application/json

{
  "eeg": [0.1, 0.2, -0.1, ...],
  "brain_state": "relaxed",
  "metadata": {}
}
```

**Response:**
```json
{
  "success": true,
  "image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "confidence_score": 0.85,
  "processing_time_ms": 45.2,
  "model_info": {...},
  "metadata": {...}
}
```

#### Image to EEG
```http
POST /api/image-to-eeg
Content-Type: multipart/form-data

file: [image file]
```

**Response:**
```json
{
  "success": true,
  "eeg": [0.1, 0.2, -0.1, ...],
  "confidence_score": 0.78,
  "processing_time_ms": 67.3,
  "model_info": {...},
  "metadata": {...}
}
```

#### Generate Sample EEG
```http
POST /api/generate-sample-eeg?brain_state=relaxed
```

**Response:**
```json
{
  "success": true,
  "eeg": [0.1, 0.2, -0.1, ...],
  "brain_state": "relaxed",
  "length": 100,
  "metadata": {...}
}
```

## 🎮 Usage Examples

### Python API Client
```python
import requests
import numpy as np

# Initialize client
api_url = "http://localhost:8000"

# Generate sample EEG
response = requests.post(f"{api_url}/api/generate-sample-eeg?brain_state=focused")
eeg_data = response.json()['eeg']

# Convert EEG to image
response = requests.post(f"{api_url}/api/eeg-to-image", 
                        json={"eeg": eeg_data, "brain_state": "focused"})
image_data_url = response.json()['image_data_url']

print(f"Generated image with confidence: {response.json()['confidence_score']:.2%}")
```

### Command Line Usage
```bash
# Test API health
curl http://localhost:8000/health

# Generate sample EEG
curl -X POST "http://localhost:8000/api/generate-sample-eeg?brain_state=active"

# Convert image to EEG
curl -X POST -F "file=@image.jpg" http://localhost:8000/api/image-to-eeg
```

## 🔬 Advanced Features

### Brain States
The system recognizes and generates patterns for different brain states:

- **Relaxed (Alpha)**: 8-13 Hz, calm, meditative states
- **Focused (Beta)**: 13-30 Hz, alert, concentrated thinking
- **Active (Gamma)**: 30-50 Hz, high cognitive processing
- **Motor Imagery**: Event-related patterns, movement imagination
- **Sleep (Theta/Delta)**: 0.5-8 Hz, drowsy, sleep states

### Attention Visualization
- **Temporal Attention**: Shows which time points in EEG are most important
- **Spatial Attention**: Highlights important regions in images
- **Cross-Modal Attention**: Reveals EEG-image feature correspondences

### Evaluation Metrics
- **Image Quality**: PSNR, SSIM, perceptual similarity
- **EEG Quality**: Correlation, spectral distance, DTW alignment
- **Bidirectional Consistency**: Cycle reconstruction accuracy

## 🎓 Educational Value

This project demonstrates:

1. **Multi-Modal Deep Learning**: How to combine different data modalities
2. **Attention Mechanisms**: State-of-the-art attention architectures
3. **CNN-RNN Hybrids**: Optimal combination of spatial and temporal processing
4. **Real-World AI Applications**: Practical implementation of research concepts
5. **Professional Development**: API design, testing, and deployment practices

## 🚀 Performance Optimization

### Model Optimizations
- **Model Caching**: Models loaded once and cached for fast inference
- **Batch Processing**: Efficient tensor operations
- **Memory Management**: Optimized memory usage for large models
- **Preprocessing Pipeline**: Streamlined data preparation

### API Optimizations
- **Async Processing**: Non-blocking request handling
- **Response Caching**: Cached responses for identical inputs
- **Error Handling**: Robust error recovery and fallback mechanisms
- **Request Validation**: Input validation and sanitization

## 🔧 Configuration

### Training Configuration
```python
# Demo configuration (fast, small dataset)
config = get_demo_config()
config.training.n_samples = 200
config.training.epochs = 10

# Full configuration (comprehensive, large dataset)
config = get_full_config()
config.training.n_samples = 5000
config.training.epochs = 50
```

### Model Configuration
```python
# EEG to Image model
model_config = {
    'time_steps': 100,
    'n_features': 1,
    'image_size': (64, 64),
    'latent_dim': 256,
    'use_attention': True,
    'use_vae': False
}
```

## 🐛 Troubleshooting

### Common Issues

**Models not loading:**
```bash
# Check dependencies
pip install -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

**API connection failed:**
```bash
# Check if API is running
curl http://localhost:8000/health

# Check port availability
netstat -an | grep 8000
```

**Streamlit interface issues:**
```bash
# Clear Streamlit cache
streamlit cache clear

# Check for port conflicts
streamlit run app.py --server.port 8502
```

### Performance Issues

**Slow inference:**
- Reduce model complexity in config
- Use CPU-optimized TensorFlow build
- Enable GPU acceleration if available

**Memory issues:**
- Reduce batch size
- Use model quantization
- Clear model cache periodically

## 📈 Future Enhancements

### Planned Features
- **Real-time EEG Streaming**: Live EEG data processing
- **Advanced Visualization**: 3D brain visualization, attention heatmaps
- **Model Interpretability**: SHAP values, gradient-based attribution
- **Cloud Deployment**: Docker containers, Kubernetes orchestration
- **Mobile Support**: React Native mobile application

### Research Directions
- **Real EEG Data**: Integration with actual EEG datasets
- **Multi-Subject Models**: Person-specific and general models
- **Temporal Dynamics**: Long-term EEG pattern analysis
- **Clinical Applications**: Medical diagnosis assistance

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd BrainWave/backend/brainwave_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 .
black .
```

## 📄 License

This project is developed for educational and research purposes. Please cite appropriately if used in academic work.

## 🙏 Acknowledgments

- **TensorFlow Team**: For the excellent deep learning framework
- **Streamlit Team**: For the intuitive web interface framework
- **FastAPI Team**: For the high-performance API framework
- **Research Community**: For the foundational work in multi-modal learning

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: This README and inline code comments
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for general questions

---

**BrainWave Analyzer v2.0** - *Advanced Multi-Modal Deep Learning Platform*

Built with ❤️ using TensorFlow, Streamlit, and modern AI techniques