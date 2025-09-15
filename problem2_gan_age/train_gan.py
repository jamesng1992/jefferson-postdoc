import os, argparse, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data_loader import maybe_get_data
from models import build_generator, build_discriminator

def plot_training(hist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    g_loss, d_loss = hist['g_loss'], hist['d_loss']
    plt.figure(); plt.plot(g_loss, label='G loss'); plt.plot(d_loss, label='D loss'); plt.legend(); plt.grid(True)
    plt.title('GAN Training Loss'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'gan_losses.png')); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--latent-dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--beta1', type=float, default=0.5)
    ap.add_argument('--out-dir', type=str, default='results')
    args = ap.parse_args()

    real_ages = maybe_get_data('.')
    # Normalize for stability
    mn, sd = real_ages.mean(), real_ages.std() + 1e-6
    ages_n = (real_ages - mn) / sd
    ds = tf.data.Dataset.from_tensor_slices(ages_n.reshape(-1,1)).shuffle(10000).batch(args.batch_size).prefetch(2)

    G = build_generator(args.latent_dim)
    D = build_discriminator()

    g_opt = keras.optimizers.Adam(args.lr, beta_1=args.beta1)
    d_opt = keras.optimizers.Adam(args.lr, beta_1=args.beta1)
    bce = keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(real_batch):
        bs = tf.shape(real_batch)[0]
        # Train D
        z = tf.random.normal((bs, args.latent_dim))
        with tf.GradientTape() as tape_d:
            fake = G(z, training=True)
            d_real = D(real_batch, training=True)
            d_fake = D(fake, training=True)
            real_labels = tf.ones_like(d_real) * 0.9  # label smoothing
            fake_labels = tf.zeros_like(d_fake)
            d_loss = bce(real_labels, d_real) + bce(fake_labels, d_fake)
        d_grads = tape_d.gradient(d_loss, D.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, D.trainable_variables))

        # Train G
        z = tf.random.normal((bs, args.latent_dim))
        with tf.GradientTape() as tape_g:
            fake = G(z, training=True)
            d_fake = D(fake, training=True)
            g_loss = bce(tf.ones_like(d_fake), d_fake)
        g_grads = tape_g.gradient(g_loss, G.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, G.trainable_variables))
        return g_loss, d_loss

    hist = {'g_loss': [], 'd_loss': []}
    for ep in range(args.epochs):
        for real in ds:
            gl, dl = train_step(real)
        hist['g_loss'].append(float(gl.numpy())); hist['d_loss'].append(float(dl.numpy()))
        if (ep+1) % 100 == 0:
            print(f"Epoch {ep+1}/{args.epochs}  G: {hist['g_loss'][-1]:.3f}  D: {hist['d_loss'][-1]:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    G.save(os.path.join(args.out_dir, 'generator.keras'))
    D.save(os.path.join(args.out_dir, 'discriminator.keras'))
    plot_training(hist, 'figures')
    np.savez(os.path.join(args.out_dir, 'norm_params.npz'), mean=mn, std=sd)

if __name__ == '__main__':
    main()
