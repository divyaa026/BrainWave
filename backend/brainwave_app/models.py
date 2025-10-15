import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import cv2
from typing import Tuple

# Keep seeds for reproducibility in demo
tf.random.set_seed(42)
np.random.seed(42)


def build_eeg_encoder(time_steps, n_features, latent_dim):
    inputs = layers.Input(shape=(time_steps, n_features))
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
    x = layers.LSTM(64, return_sequences=False, dropout=0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    # deterministic for demo
    encoder = Model(inputs, z_mean, name='eeg_encoder')
    return encoder


def build_image_decoder(latent_dim):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(8 * 8 * 128, activation='relu')(inputs)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(3, (3, 3), strides=2, padding='same', activation='sigmoid')(x)
    decoder = Model(inputs, x, name='image_decoder')
    return decoder


def build_image_encoder(latent_dim, img_height=64, img_width=64):
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
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.RepeatVector(time_steps)(inputs)
    x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
    x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    outputs = layers.Dense(n_features, activation='tanh')(x)
    decoder = Model(inputs, outputs, name='eeg_decoder')
    return decoder


class DemoModels:
    """Container that builds demo models and exposes simple predict helpers."""

    def __init__(self, latent_dim=32, time_steps=100, n_features=1, img_h=64, img_w=64):
        self.LATENT_DIM = latent_dim
        self.TIME_STEPS = time_steps
        self.N_FEATURES = n_features
        self.IMG_H = img_h
        self.IMG_W = img_w

        # Build models
        self.eeg_encoder = build_eeg_encoder(self.TIME_STEPS, self.N_FEATURES, self.LATENT_DIM)
        self.image_decoder = build_image_decoder(self.LATENT_DIM)

        self.image_encoder = build_image_encoder(self.LATENT_DIM, self.IMG_H, self.IMG_W)
        self.eeg_decoder = build_eeg_decoder(self.LATENT_DIM, self.TIME_STEPS, self.N_FEATURES)

    def predict_image_from_eeg(self, eeg_sequence: np.ndarray) -> np.ndarray:
        """Accepts shape (time_steps,) or (1,time_steps,1) and returns image array float32 [0,1]"""
        arr = np.array(eeg_sequence)
        if arr.ndim == 1:
            arr = arr.reshape(1, self.TIME_STEPS, self.N_FEATURES)
        elif arr.ndim == 2:
            arr = arr.reshape(1, self.TIME_STEPS, self.N_FEATURES)

        pred = self.image_decoder.predict(self.eeg_encoder.predict(arr), verbose=0)
        img = np.clip(pred[0], 0.0, 1.0).astype(np.float32)
        return img

    def predict_eeg_from_image(self, image: np.ndarray) -> np.ndarray:
        """Accepts image HxWx3 float in [0,1] and returns EEG sequence shape (time_steps,)"""
        img = image.astype(np.float32)
        if img.shape != (self.IMG_H, self.IMG_W, 3):
            img = cv2.resize(img, (self.IMG_W, self.IMG_H))
        img = np.expand_dims(img, axis=0)
        latent = self.image_encoder.predict(img, verbose=0)
        eeg = self.eeg_decoder.predict(latent, verbose=0)[0]
        return eeg.flatten()
