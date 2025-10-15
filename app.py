"""
Enhanced Streamlit Interface for BrainWave Analyzer
Professional demo interface with advanced visualizations and interactions
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import cv2
import time
import base64
from io import StringIO, BytesIO
from PIL import Image
import requests
import json
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced models
try:
    from models import DemoModels
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    st.error("Models not available. Please ensure all dependencies are installed.")

# Page configuration
st.set_page_config(
    page_title="🧠 BrainWave Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = None
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

##############################
# Helper Functions
##############################

@st.cache_data
def load_models():
    """Load models with caching"""
    if MODELS_AVAILABLE:
        try:
            models = DemoModels()
            return models
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None
    return None

def generate_sample_eeg(brain_state: str = 'relaxed') -> np.ndarray:
    """Generate sample EEG for demonstration"""
    if st.session_state.models:
        return st.session_state.models.generate_sample_eeg(brain_state)
    else:
        # Fallback synthetic generation
        t = np.linspace(0, 4.0, 100)
        if brain_state == 'relaxed':
            signal = np.sin(2 * np.pi * 10 * t)  # Alpha waves
        elif brain_state == 'focused':
            signal = np.sin(2 * np.pi * 20 * t)  # Beta waves
        elif brain_state == 'active':
            signal = np.sin(2 * np.pi * 40 * t)  # Gamma waves
        else:
            signal = np.sin(2 * np.pi * 10 * t)  # Default alpha
        
        signal += 0.1 * np.random.randn(len(signal))
        return signal.astype(np.float32)

def create_eeg_plot(eeg_data: np.ndarray, title: str = "EEG Signal") -> go.Figure:
    """Create interactive EEG plot"""
    fig = go.Figure()
    
    # Create time axis
    time_axis = np.linspace(0, 4.0, len(eeg_data))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=eeg_data,
        mode='lines',
        name='EEG Signal',
        line=dict(color='#667eea', width=2),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Amplitude:</b> %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=400,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Add frequency bands annotation
    fig.add_annotation(
        text="<b>Frequency Bands:</b><br>Delta (0.5-4 Hz)<br>Theta (4-8 Hz)<br>Alpha (8-13 Hz)<br>Beta (13-30 Hz)<br>Gamma (30-50 Hz)",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1
    )
    
    return fig

def create_spectral_analysis(eeg_data: np.ndarray) -> go.Figure:
    """Create spectral analysis plot"""
    from scipy.fft import fft, fftfreq
    
    # Compute FFT
    fft_data = np.abs(fft(eeg_data))
    freqs = fftfreq(len(eeg_data), 1/250)  # 250 Hz sampling rate
    
    # Only show positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    positive_fft = fft_data[:len(fft_data)//2]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=positive_freqs,
        y=positive_fft,
        mode='lines',
        name='Power Spectrum',
        line=dict(color='#764ba2', width=2),
        fill='tonexty'
    ))
    
    # Add frequency band regions
    bands = {
        'Delta': (0.5, 4, 'rgba(255, 99, 132, 0.2)'),
        'Theta': (4, 8, 'rgba(54, 162, 235, 0.2)'),
        'Alpha': (8, 13, 'rgba(255, 206, 86, 0.2)'),
        'Beta': (13, 30, 'rgba(75, 192, 192, 0.2)'),
        'Gamma': (30, 50, 'rgba(153, 102, 255, 0.2)')
    }
    
    for band_name, (low, high, color) in bands.items():
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=color,
            layer="below",
            line_width=0,
            annotation_text=band_name,
            annotation_position="top"
        )
    
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        template="plotly_white",
        height=300,
        showlegend=False
    )
    
    return fig

def download_eeg_csv(eeg_data: np.ndarray) -> str:
    """Create downloadable CSV for EEG data"""
    df = pd.DataFrame({
        'Time': np.linspace(0, 4.0, len(eeg_data)),
        'Amplitude': eeg_data
    })
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="eeg_signal.csv">Download EEG as CSV</a>'
    return href

def download_image_png(image_data: np.ndarray, filename: str = "generated_image") -> str:
    """Create downloadable PNG for image"""
    # Ensure image is in correct range
    img_uint8 = (np.clip(image_data, 0.0, 1.0) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download Image as PNG</a>'
    return href

##############################
# Main Application
##############################

# Header
st.markdown('<h1 class="main-header">🧠 BrainWave Analyzer</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
        Advanced Multi-Modal Deep Learning Platform for EEG-Image Synthesis
    </p>
    <p style='font-size: 1rem; color: #888;'>
        Featuring CNN-RNN Hybrid Architectures with Attention Mechanisms
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Model status
    st.markdown("### Model Status")
    if MODELS_AVAILABLE:
        if st.session_state.models is None:
            st.session_state.models = load_models()
        
        if st.session_state.models:
            st.success("✅ Models Loaded")
            model_info = st.session_state.models.get_model_info()
            st.info(f"**Architecture:** {model_info.get('architecture', 'Unknown')}")
            st.info(f"**Parameters:** {model_info.get('total_params', 'Unknown'):,}")
        else:
            st.error("❌ Models Failed to Load")
    else:
        st.error("❌ Models Not Available")
    
    # Settings
    st.markdown("### Settings")
    brain_state = st.selectbox(
        "Default Brain State",
        ["relaxed", "focused", "active", "motor", "sleep"],
        index=0,
        help="Select the default brain state for EEG generation"
    )
    
    show_spectral = st.checkbox("Show Spectral Analysis", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    
    # API Settings
    st.markdown("### API Settings")
    use_api = st.checkbox("Use API Backend", value=False, help="Use FastAPI backend instead of direct model calls")
    
    if use_api:
        api_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
        if st.button("Test API Connection"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API Connected")
                else:
                    st.error(f"❌ API Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection Failed: {e}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧠 EEG → Image", "🖼️ Image → EEG", "📊 Analysis", "ℹ️ About"])

with tab1:
    st.markdown("## 🧠 EEG to Image Synthesis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input EEG Signal")
        
        # EEG input method
        eeg_input_method = st.radio(
            "Choose EEG Input Method:",
            ["Generate Sample", "Upload CSV", "Manual Input"],
            horizontal=True
        )
        
        eeg_data = None
        
        if eeg_input_method == "Generate Sample":
            if st.button(f"🎲 Generate {brain_state.title()} EEG", type="primary"):
                eeg_data = generate_sample_eeg(brain_state)
                st.session_state.current_eeg = eeg_data
                st.success(f"Generated {brain_state} EEG signal!")
        
        elif eeg_input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload EEG CSV", type=['csv'], help="CSV file with EEG data (one column)")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    eeg_data = df.iloc[:, 0].values.astype(np.float32)
                    st.session_state.current_eeg = eeg_data
                    st.success("EEG data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
        
        else:  # Manual input
            st.markdown("**Manual EEG Input** (comma-separated values)")
            manual_input = st.text_area("Enter EEG values:", placeholder="0.1, 0.2, -0.1, 0.3, ...")
            if st.button("Load Manual EEG"):
                try:
                    eeg_data = np.array([float(x.strip()) for x in manual_input.split(',')], dtype=np.float32)
                    st.session_state.current_eeg = eeg_data
                    st.success("Manual EEG data loaded!")
                except Exception as e:
                    st.error(f"Error parsing manual input: {e}")
        
        # Display current EEG
        if 'current_eeg' in st.session_state and st.session_state.current_eeg is not None:
            eeg_data = st.session_state.current_eeg
            
            # EEG visualization
            fig_eeg = create_eeg_plot(eeg_data, f"EEG Signal ({brain_state.title()} State)")
            st.plotly_chart(fig_eeg, use_container_width=True)
            
            # Spectral analysis
            if show_spectral:
                fig_spectral = create_spectral_analysis(eeg_data)
                st.plotly_chart(fig_spectral, use_container_width=True)
            
            # EEG stats
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Length", f"{len(eeg_data)} samples")
            with col_stats2:
                st.metric("Mean", f"{np.mean(eeg_data):.3f}")
            with col_stats3:
                st.metric("Std Dev", f"{np.std(eeg_data):.3f}")
    
    with col2:
        st.markdown("### Generated Image")
        
        if eeg_data is not None:
            if st.button("🎨 Generate Image", type="primary", disabled=not MODELS_AVAILABLE):
                with st.spinner('Generating image from EEG...'):
                    start_time = time.time()
                    
                    try:
                        if use_api and 'api_url' in locals():
                            # Use API
                            response = requests.post(
                                f"{api_url}/api/eeg-to-image",
                                json={"eeg": eeg_data.tolist(), "brain_state": brain_state},
                                timeout=30
                            )
                            if response.status_code == 200:
                                result = response.json()
                                # Decode image from data URL
                                data_url = result['image_data_url']
                                header, encoded = data_url.split(',', 1)
                                image_bytes = base64.b64decode(encoded)
                                image = Image.open(BytesIO(image_bytes))
                                generated_image = np.array(image) / 255.0
                                confidence = result.get('confidence_score', 0.0)
                                processing_time = result.get('processing_time_ms', 0.0)
                            else:
                                st.error(f"API Error: {response.status_code}")
                                return
                        else:
                            # Use direct model
                            generated_image = st.session_state.models.predict_image_from_eeg(eeg_data)
                            confidence = st.session_state.models.get_confidence_score('eeg_to_image')
                            processing_time = (time.time() - start_time) * 1000
                        
                        st.session_state.generated_image = generated_image
                        st.success("Image generated successfully!")
                        
                        # Display metrics
                        if show_confidence:
                            col_metric1, col_metric2 = st.columns(2)
                            with col_metric1:
                                st.metric("Confidence", f"{confidence:.2%}")
                            with col_metric2:
                                st.metric("Processing Time", f"{processing_time:.1f} ms")
                        
                    except Exception as e:
                        st.error(f"Error generating image: {e}")
            
            # Display generated image
            if 'generated_image' in st.session_state:
                generated_image = st.session_state.generated_image
                
                st.image(generated_image, caption="Generated Image from EEG", use_column_width=True)
                
                # Image stats
                col_img1, col_img2, col_img3 = st.columns(3)
                with col_img1:
                    st.metric("Resolution", f"{generated_image.shape[0]}×{generated_image.shape[1]}")
                with col_img2:
                    st.metric("Mean Intensity", f"{np.mean(generated_image):.3f}")
                with col_img3:
                    st.metric("Color Channels", f"{generated_image.shape[2]}")
                
                # Download button
                st.markdown(download_image_png(generated_image, f"eeg_generated_{brain_state}"), unsafe_allow_html=True)
        
        else:
            st.info("👆 Load or generate an EEG signal to create an image")

with tab2:
    st.markdown("## 🖼️ Image to EEG Synthesis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Input Image")
        
        uploaded_image = st.file_uploader(
            "Upload an Image", 
            type=['png', 'jpg', 'jpeg'], 
            help="Upload an image to generate corresponding EEG signal"
        )
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            image_array = np.array(image) / 255.0
            
            st.image(image_array, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Resolution", f"{image_array.shape[0]}×{image_array.shape[1]}")
            with col_info2:
                st.metric("Mean Intensity", f"{np.mean(image_array):.3f}")
            with col_info3:
                st.metric("Color Channels", f"{image_array.shape[2]}")
            
            st.session_state.uploaded_image = image_array
    
    with col2:
        st.markdown("### Generated EEG Signal")
        
        if 'uploaded_image' in st.session_state:
            image_data = st.session_state.uploaded_image
            
            if st.button("🧠 Generate EEG", type="primary", disabled=not MODELS_AVAILABLE):
                with st.spinner('Generating EEG from image...'):
                    start_time = time.time()
                    
                    try:
                        if use_api and 'api_url' in locals():
                            # Use API
                            # Convert image to bytes
                            img_bytes = BytesIO()
                            Image.fromarray((image_data * 255).astype(np.uint8)).save(img_bytes, format='PNG')
                            img_bytes.seek(0)
                            
                            files = {'file': ('image.png', img_bytes, 'image/png')}
                            response = requests.post(f"{api_url}/api/image-to-eeg", files=files, timeout=30)
                            
                            if response.status_code == 200:
                                result = response.json()
                                generated_eeg = np.array(result['eeg'])
                                confidence = result.get('confidence_score', 0.0)
                                processing_time = result.get('processing_time_ms', 0.0)
                            else:
                                st.error(f"API Error: {response.status_code}")
                                return
                        else:
                            # Use direct model
                            generated_eeg = st.session_state.models.predict_eeg_from_image(image_data)
                            confidence = st.session_state.models.get_confidence_score('image_to_eeg')
                            processing_time = (time.time() - start_time) * 1000
                        
                        st.session_state.generated_eeg = generated_eeg
                        st.success("EEG signal generated successfully!")
                        
                        # Display metrics
                        if show_confidence:
                            col_metric1, col_metric2 = st.columns(2)
                            with col_metric1:
                                st.metric("Confidence", f"{confidence:.2%}")
                            with col_metric2:
                                st.metric("Processing Time", f"{processing_time:.1f} ms")
                        
                    except Exception as e:
                        st.error(f"Error generating EEG: {e}")
            
            # Display generated EEG
            if 'generated_eeg' in st.session_state:
                generated_eeg = st.session_state.generated_eeg
                
                # EEG visualization
                fig_eeg = create_eeg_plot(generated_eeg, "Generated EEG Signal")
                st.plotly_chart(fig_eeg, use_container_width=True)
                
                # Spectral analysis
                if show_spectral:
                    fig_spectral = create_spectral_analysis(generated_eeg)
                    st.plotly_chart(fig_spectral, use_container_width=True)
                
                # EEG stats
                col_eeg1, col_eeg2, col_eeg3 = st.columns(3)
                with col_eeg1:
                    st.metric("Length", f"{len(generated_eeg)} samples")
                with col_eeg2:
                    st.metric("Mean", f"{np.mean(generated_eeg):.3f}")
                with col_eeg3:
                    st.metric("Std Dev", f"{np.std(generated_eeg):.3f}")
                
                # Download button
                st.markdown(download_eeg_csv(generated_eeg), unsafe_allow_html=True)
        
        else:
            st.info("👆 Upload an image to generate corresponding EEG signal")

with tab3:
    st.markdown("## 📊 Advanced Analysis")
    
    # Model architecture visualization
    st.markdown("### Model Architecture")
    
    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.markdown("""
        **EEG → Image Pipeline:**
        ```
        EEG Input (Sequence)
           ↓
        LSTM Encoder + Attention
           ↓
        Latent Space (256D)
           ↓
        CNN Decoder (Progressive)
           ↓
        Generated Image (64×64×3)
        ```
        """)
    
    with col_arch2:
        st.markdown("""
        **Image → EEG Pipeline:**
        ```
        Image Input (64×64×3)
           ↓
        CNN Encoder + Attention
           ↓
        Latent Space (256D)
           ↓
        LSTM Decoder
           ↓
        Generated EEG (100 samples)
        ```
        """)
    
    # Performance metrics
    if st.session_state.models:
        st.markdown("### Performance Metrics")
        model_info = st.session_state.models.get_model_info()
        
        col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
        
        with col_perf1:
            st.metric("Model Parameters", f"{model_info.get('total_params', 0):,}")
        
        with col_perf2:
            st.metric("Latent Dimension", model_info.get('latent_dim', 'Unknown'))
        
        with col_perf3:
            st.metric("Time Steps", model_info.get('time_steps', 'Unknown'))
        
        with col_perf4:
            st.metric("Attention", "✅" if model_info.get('use_attention', False) else "❌")
        
        # Prediction counts
        if 'prediction_breakdown' in model_info:
            st.markdown("### Usage Statistics")
            pred_counts = model_info['prediction_breakdown']
            
            col_usage1, col_usage2 = st.columns(2)
            
            with col_usage1:
                st.metric("EEG → Image Predictions", pred_counts.get('eeg_to_image', 0))
            
            with col_usage2:
                st.metric("Image → EEG Predictions", pred_counts.get('image_to_eeg', 0))
    
    # Technical details
    st.markdown("### Technical Details")
    
    with st.expander("🧠 Deep Learning Architecture", expanded=False):
        st.markdown("""
        **Key Features:**
        - **Hybrid CNN-RNN Architecture**: Combines convolutional networks for spatial processing with recurrent networks for temporal modeling
        - **Attention Mechanisms**: Temporal attention for EEG sequences and spatial attention for images
        - **Multi-Modal Latent Space**: Shared 256-dimensional latent representation
        - **Progressive Generation**: Multi-scale image generation with skip connections
        - **Advanced Loss Functions**: Perceptual loss, DTW loss, and frequency domain losses
        
        **Training Data:**
        - Synthetic EEG signals with realistic frequency bands
        - Corresponding synthetic images based on brain state patterns
        - 1000+ paired samples for training
        - Data augmentation for robustness
        """)
    
    with st.expander("⚡ Performance Optimizations", expanded=False):
        st.markdown("""
        **Optimization Techniques:**
        - **Model Caching**: Models loaded once and cached for fast inference
        - **Batch Processing**: Efficient tensor operations
        - **Memory Management**: Optimized memory usage for large models
        - **Preprocessing Pipeline**: Streamlined data preparation
        - **Error Handling**: Robust error recovery and fallback mechanisms
        """)

with tab4:
    st.markdown("## ℹ️ About BrainWave Analyzer")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **BrainWave Analyzer** is an advanced multi-modal deep learning platform that demonstrates the fascinating intersection of neuroscience and computer vision. The system creates bidirectional mappings between EEG brain signals and visual imagery using state-of-the-art neural architectures.
    
    ### 🔬 Technical Innovation
    
    **Core Technologies:**
    - **TensorFlow/Keras**: Advanced deep learning framework
    - **Attention Mechanisms**: Temporal and spatial attention for improved accuracy
    - **CNN-RNN Hybrids**: Optimal combination of spatial and temporal processing
    - **Multi-Modal Learning**: Joint embedding space for cross-modal understanding
    
    **Key Achievements:**
    - Real-time EEG-to-Image synthesis
    - High-quality Image-to-EEG prediction
    - Attention-based feature extraction
    - Robust preprocessing pipelines
    - Professional API and UI interfaces
    
    ### 🚀 Use Cases
    
    **Research Applications:**
    - Neuroscience research and education
    - Brain-computer interface development
    - Cognitive state visualization
    - Medical imaging analysis
    
    **Educational Value:**
    - Deep learning architecture demonstration
    - Multi-modal AI concepts
    - Real-world AI applications
    - Interactive learning platform
    
    ### 🛠️ Technical Stack
    
    **Backend:**
    - Python 3.8+
    - TensorFlow 2.x
    - FastAPI
    - NumPy, SciPy, OpenCV
    
    **Frontend:**
    - Streamlit
    - Plotly
    - Custom CSS styling
    
    **Machine Learning:**
    - Custom CNN-RNN architectures
    - Attention mechanisms
    - Advanced loss functions
    - Synthetic data generation
    
    ### 📈 Future Enhancements
    
    **Planned Features:**
    - Real-time EEG data streaming
    - Advanced visualization tools
    - Model interpretability features
    - Cloud deployment options
    - Mobile application support
    
    ### 👥 Development Team
    
    This project demonstrates advanced deep learning concepts and serves as a comprehensive example of multi-modal AI system development.
    
    ### 📄 License
    
    This project is developed for educational and research purposes.
    """)
    
    # Contact information
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; margin-top: 2rem;'>
        <p><strong>BrainWave Analyzer v2.0</strong></p>
        <p>Advanced Multi-Modal Deep Learning Platform</p>
        <p style='color: #666; font-size: 0.9rem;'>
            Built with ❤️ using TensorFlow, Streamlit, and modern AI techniques
        </p>
    </div>
    """, unsafe_allow_html=True)

