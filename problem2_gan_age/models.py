import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_generator(latent_dim=64):
    z = keras.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(z)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation=None)(x)  # raw age value (normalized space)
    return keras.Model(z, out, name='generator')

def build_discriminator():
    x_in = keras.Input(shape=(1,))
    x = layers.Dense(128, activation='relu')(x_in)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation=None)(x)  # logit
    return keras.Model(x_in, out, name='discriminator')
