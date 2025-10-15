import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from io import StringIO
import cv2

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

##############################
# 1. Synthetic Data Generation
##############################

def generate_synthetic_eeg(batch_size, time_steps, n_features):
    """Generates synthetic EEG data. Different patterns simulate different states."""
    # Simple patterns: sine waves with different frequencies and noises
    t = np.linspace(0, 4 * np.pi, time_steps)
    # Pattern 1: "Calm" - low frequency sine wave
    calm = np.sin(1 * t) + 0.1 * np.random.randn(time_steps)
    # Pattern 2: "Focused" - medium frequency with small bursts
    focused = np.sin(5 * t) * np.exp(-0.1 * t) + 0.1 * np.random.randn(time_steps)
    # Pattern 3: "Agitated" - high frequency noise
    agitated = 0.5 * np.sin(10 * t) + 0.5 * np.random.randn(time_steps)

    patterns = [calm, focused, agitated]
    X = []

    for _ in range(batch_size):
        chosen_pattern = np.random.choice(patterns)
        # Add some batch-specific variation
        pattern_varied = chosen_pattern + 0.05 * np.random.randn(time_steps)
        X.append(pattern_varied)

    X = np.array(X).reshape(batch_size, time_steps, n_features)
    return X


def generate_synthetic_image_from_label(pattern_id, img_height=64, img_width=64):
    """Generates a simple synthetic image based on the EEG pattern ID."""
    # Create a blank image
    img = np.zeros((img_height, img_width, 3), dtype=np.float32)
    center_x, center_y = img_width // 2, img_height // 2

    if pattern_id == 0:  # Calm - Blue circle
        cv2.circle(img, (center_x, center_y), 20, (0.3, 0.3, 1.0), -1)  # Blue
    elif pattern_id == 1:  # Focused - Green square
        cv2.rectangle(img, (center_x - 15, center_y - 15), (center_x + 15, center_y + 15), (0.3, 1.0, 0.3), -1)
    elif pattern_id == 2:  # Agitated - Red jagged lines
        points = np.array([[10, 10], [30, 50], [50, 30], [70, 70]], np.int32)
        points = points + np.random.randint(-5, 5, points.shape)
        cv2.polylines(img, [points], isClosed=True, color=(1.0, 0.3, 0.3), thickness=3)

    return img

##########################################
# 2. Model Definitions
##########################################

def build_eeg_encoder(time_steps, n_features, latent_dim):
    """LSTM-based encoder for EEG signals."""
    inputs = layers.Input(shape=(time_steps, n_features))
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
    x = layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    # For simplicity, we'll use a deterministic encoder, but this is VAE-inspired.
    encoder = Model(inputs, z_mean, name='eeg_encoder')
    return encoder


def build_image_decoder(latent_dim):
    """CNN-based decoder (generator) for images from latent vector."""
    inputs = layers.Input(shape=(latent_dim,))
    # Project and reshape
    x = layers.Dense(8 * 8 * 128, activation='relu')(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    # Upsample to 64x64
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)  # 16x16
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)  # 32x32
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(3, (3, 3), strides=2, padding='same', activation='sigmoid')(x)  # 64x64
    decoder = Model(inputs, x, name='image_decoder')
    return decoder


def build_image_encoder(latent_dim, img_height=64, img_width=64):
    """CNN-based encoder for images."""
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    z_vector = layers.Dense(latent_dim, activation='relu')(x)
    encoder = Model(inputs, z_vector, name='image_encoder')
    return encoder


def build_eeg_decoder(latent_dim, time_steps, n_features):
    """LSTM-based decoder for EEG signals from latent vector."""
    inputs = layers.Input(shape=(latent_dim,))
    # Repeat the latent vector across time steps
    x = layers.RepeatVector(time_steps)(inputs)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    outputs = layers.Dense(n_features, activation='tanh')(x)  # tanh to keep output bounded
    decoder = Model(inputs, outputs, name='eeg_decoder')
    return decoder

# Build the two combined models
LATENT_DIM = 32
TIME_STEPS = 100
N_FEATURES = 1
IMG_HEIGHT, IMG_WIDTH = 64, 64

# Model 1: EEG -> Image
eeg_encoder = build_eeg_encoder(TIME_STEPS, N_FEATURES, LATENT_DIM)
image_decoder = build_image_decoder(LATENT_DIM)

eeg_input = layers.Input(shape=(TIME_STEPS, N_FEATURES))
encoded_eeg = eeg_encoder(eeg_input)
generated_image = image_decoder(encoded_eeg)
eeg_to_image_model = Model(eeg_input, generated_image, name='eeg_to_image')
eeg_to_image_model.compile(optimizer='adam', loss='mse')  # Using MSE for simplicity

# Model 2: Image -> EEG
image_encoder = build_image_encoder(LATENT_DIM, IMG_HEIGHT, IMG_WIDTH)
eeg_decoder = build_eeg_decoder(LATENT_DIM, TIME_STEPS, N_FEATURES)

image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
encoded_image = image_encoder(image_input)
generated_eeg = eeg_decoder(encoded_image)
image_to_eeg_model = Model(image_input, generated_eeg, name='image_to_eeg')
image_to_eeg_model.compile(optimizer='adam', loss='mse')

##########################################
# 3. Training Simulation (Pseudo-Training)
##########################################
# For a demo, we'll use pre-initialized random weights.
# In a real project, you would train these models on real data.

##########################################
# 4. Streamlit UI and Application Logic
##########################################

st.set_page_config(page_title="BrainWave Analyzer", layout="wide")
st.title("🧠 BrainWave Analyzer")
st.markdown("A Multi-Modal CNN & RNN Platform for EEG and Image Synthesis")

col1, col2 = st.columns(2)

with col1:
    st.header("🧠 Brainwave to Visualization")
    eeg_input_option = st.selectbox("EEG Input Method", ["Generate Synthetic Signal", "Upload CSV"], key="eeg_input")

    if eeg_input_option == "Generate Synthetic Signal":
        # Let's generate one sample for display
        sample_eeg = generate_synthetic_eeg(1, TIME_STEPS, N_FEATURES)[0]
        st.plotly_chart(go.Figure(data=go.Scatter(y=sample_eeg.flatten(), mode='lines', line=dict(color='#00FFFF'))).update_layout(title="Synthetic EEG Signal", xaxis_title="Time", yaxis_title="Amplitude", template="plotly_dark"))
        eeg_data = sample_eeg
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], key="eeg_upload")
        if uploaded_file is not None:
            # Read the CSV, assume it's a single column with TIME_STEPS rows
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe.head())
            # Simple processing: take the first column, truncate/pad to TIME_STEPS
            eeg_signal = dataframe.iloc[:TIME_STEPS, 0].values
            if len(eeg_signal) < TIME_STEPS:
                eeg_signal = np.pad(eeg_signal, (0, TIME_STEPS - len(eeg_signal)), 'constant')
            else:
                eeg_signal = eeg_signal[:TIME_STEPS]
            eeg_data = eeg_signal.reshape(1, TIME_STEPS, 1)
            st.plotly_chart(go.Figure(data=go.Scatter(y=eeg_signal, mode='lines', line=dict(color='#00FFFF'))).update_layout(title="Uploaded EEG Signal", template="plotly_dark"))
        else:
            eeg_data = None

    if st.button("Generate Image", key="gen_img") and eeg_data is not None:
        # Run the model prediction
        with st.spinner('Generating image from EEG...'):
            # Add batch dimension and predict
            generated_img = eeg_to_image_model.predict(eeg_data, verbose=0)[0]
            # Ensure the image is in [0,1]
            img_disp = np.clip(generated_img, 0.0, 1.0)
            st.image(img_disp, caption="Generated Image from EEG", use_column_width=True)

with col2:
    st.header("🖼 Image to Brainwave")
    uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'], key="img_upload")
    if uploaded_image is not None:
        # Read and preprocess the image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            st.error("Could not decode image. Please upload a valid image file.")
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            image_display = image.astype(np.float32) / 255.0
            st.image(image_display, caption="Uploaded Image", use_column_width=True)
            # Preprocess for model: normalize and add batch dim
            image_input = np.expand_dims(image_display, axis=0)

    if st.button("Predict EEG", key="pred_eeg") and 'image_input' in locals():
        with st.spinner('Predicting EEG from image...'):
            predicted_eeg = image_to_eeg_model.predict(image_input, verbose=0)[0]
            fig = go.Figure(data=go.Scatter(y=predicted_eeg.flatten(), mode='lines', line=dict(color='#FF00FF')))
            fig.update_layout(title="Predicted EEG Signal", xaxis_title="Time", yaxis_title="Amplitude", template="plotly_dark")
            st.plotly_chart(fig)

            # Create a download link for the EEG signal
            eeg_df = pd.DataFrame(predicted_eeg.flatten(), columns=['EEG_Amplitude'])
            csv = eeg_df.to_csv(index=False)
            st.download_button(label="Download Signal as CSV", data=csv, file_name='predicted_eeg.csv', mime='text/csv')

# Model Architecture Section
with st.expander("🧬 Under the Hood: Model Architecture"):
    st.markdown("""
    *EEG -> Image Pipeline:*
    Input EEG (Sequence) -> LSTM Encoder -> Dense Layer -> CNN Decoder (Transposed Convs) -> Generated Image

    *Image -> EEG Pipeline:*
    Input Image -> CNN Encoder -> Dense Layer -> LSTM Decoder -> Predicted EEG (Sequence)
    """)

st.markdown("---")
st.markdown("Built with TensorFlow/Keras & Streamlit | Demonstrating CNNs, RNNs, and Deep Optimization")
