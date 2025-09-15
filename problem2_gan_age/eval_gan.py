import os, argparse, numpy as np
from tensorflow import keras
from statsmodels.distributions.empirical_distribution import ECDF
from data_loader import maybe_get_data
import matplotlib.pyplot as plt
import seaborn as sns

def qq_plot(data, ref, ax):
    qs = np.linspace(0.01, 0.99, 99)
    qd = np.quantile(data, qs); qr = np.quantile(ref, qs)
    ax.scatter(qr, qd, s=10)
    minv, maxv = min(qr.min(), qd.min()), max(qr.max(), qd.max())
    ax.plot([minv, maxv],[minv, maxv], ls='--')
    ax.set_xlabel('True quantiles'); ax.set_ylabel('Generated quantiles'); ax.set_title('Q–Q plot')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generator', type=str, default='results/generator.keras')
    ap.add_argument('--num-samples', type=int, default=100000)
    args = ap.parse_args()

    ages = maybe_get_data('.')
    params = np.load('results/norm_params.npz')
    mn, sd = float(params['mean']), float(params['std'])

    G = keras.models.load_model(args.generator)
    z = np.random.randn(args.num_samples, G.input_shape[1]).astype('float32')
    gen = G.predict(z, verbose=0).squeeze() * sd + mn

    os.makedirs('figures', exist_ok=True)
    plt.figure(); sns.kdeplot(ages, label='True', bw_adjust=0.8); sns.kdeplot(gen, label='Generated', bw_adjust=0.8)
    plt.legend(); plt.title('Age distribution (KDE overlay)'); plt.tight_layout()
    plt.savefig('figures/hist_overlay.png'); plt.close()

    ecdf_true = ECDF(ages); ecdf_gen = ECDF(gen)
    xs = np.linspace(min(ages.min(), gen.min()), max(ages.max(), gen.max()), 500)
    plt.figure(); plt.plot(xs, ecdf_true(xs), label='True CDF'); plt.plot(xs, ecdf_gen(xs), label='Generated CDF')
    plt.legend(); plt.title('CDF overlay'); plt.tight_layout()
    plt.savefig('figures/cdf_overlay.png'); plt.close()

    fig, ax = plt.subplots(); qq_plot(gen, ages, ax); fig.tight_layout()
    fig.savefig('figures/qq_plot.png'); plt.close(fig)

    print(f"True mean={ages.mean():.2f}, std={ages.std():.2f}")
    print(f"Generated mean={gen.mean():.2f}, std={gen.std():.2f}")

if __name__ == '__main__':
    main()
