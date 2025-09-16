# problem2_gan_age/train_gan.py
# Simple/stable GAN training that ALWAYS writes generator + norm params.

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_loader import maybe_get_data

# ---- Paths anchored to this file ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(RES_DIR, exist_ok=True)

def build_generator(latent_dim: int) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="linear"),  # output is normalized age
        ],
        name="generator",
    )

def build_discriminator() -> keras.Model:
    return keras.Sequential(
        [
            layers.Input(shape=(1,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--lr-g", type=float, default=2e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Repro
    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load + normalize data ----
    ages = maybe_get_data(BASE_DIR).astype("float32")
    mean, std = float(ages.mean()), float(ages.std() + 1e-8)
    ages_norm = ((ages - mean) / std).reshape(-1, 1)

    # ---- Models & optimizers ----
    latent_dim = args.latent_dim
    G = build_generator(latent_dim)
    D = build_discriminator()
    opt_g = keras.optimizers.Adam(args.lr_g, beta_1=0.5, beta_2=0.999)
    opt_d = keras.optimizers.Adam(args.lr_d, beta_1=0.5, beta_2=0.999)
    bce = keras.losses.BinaryCrossentropy(from_logits=False)

    # ---- Dataset ----
    ds = tf.data.Dataset.from_tensor_slices(ages_norm).shuffle(10000).batch(args.batch_size).prefetch(2)

    @tf.function
    def train_step(real_batch):
        bs = tf.shape(real_batch)[0]

        # --- Train D ---
        z = tf.random.normal((bs, latent_dim))
        with tf.GradientTape() as tape_d:
            fake = G(z, training=True)
            pred_real = D(real_batch, training=True)
            pred_fake = D(fake, training=True)
            loss_d_real = bce(tf.ones_like(pred_real), pred_real)
            loss_d_fake = bce(tf.zeros_like(pred_fake), pred_fake)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
        grads_d = tape_d.gradient(loss_d, D.trainable_variables)
        opt_d.apply_gradients(zip(grads_d, D.trainable_variables))

        # --- Train G ---
        z = tf.random.normal((bs, latent_dim))
        with tf.GradientTape() as tape_g:
            fake = G(z, training=True)
            pred = D(fake, training=True)
            loss_g = bce(tf.ones_like(pred), pred)
        grads_g = tape_g.gradient(loss_g, G.trainable_variables)
        opt_g.apply_gradients(zip(grads_g, G.trainable_variables))

        return loss_d, loss_g

    # ---- Training loop ----
    steps_per_epoch = max(1, len(ages_norm) // args.batch_size)
    for epoch in range(1, args.epochs + 1):
        ld_acc, lg_acc = 0.0, 0.0
        for step, real in enumerate(ds):
            ld, lg = train_step(real)
            ld_acc += float(ld); lg_acc += float(lg)

        ld_acc /= steps_per_epoch
        lg_acc /= steps_per_epoch
        if epoch % 50 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:4d}/{args.epochs}  D: {ld_acc:.3f}  G: {lg_acc:.3f}")

    # ---- SAVE: generator (.keras) + normalization params (.npz) ----
    gen_path = os.path.join(RES_DIR, "generator.keras")
    G.save(gen_path)
    np.savez(os.path.join(RES_DIR, "norm_params.npz"), mean=mean, std=std)
    print(f"Saved:\n  - {gen_path}\n  - {os.path.join(RES_DIR, 'norm_params.npz')}")

if __name__ == "__main__":
    main()
