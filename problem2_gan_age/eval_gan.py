# problem2_gan_age/eval_gan.py
# Robust, headless plotting that always writes non-empty PNGs on Windows.

import os, argparse, numpy as np
from tensorflow import keras
from statsmodels.distributions.empirical_distribution import ECDF
from data_loader import maybe_get_data

import matplotlib as mpl
mpl.use("Agg")  # force non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths anchored to this file ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(BASE_DIR, "figures")
RES_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

def qq_plot(data, ref, ax):
    qs = np.linspace(0.01, 0.99, 99)
    qd = np.quantile(data, qs); qr = np.quantile(ref, qs)
    ax.scatter(qr, qd, s=10)
    minv, maxv = min(qr.min(), qd.min()), max(qr.max(), qd.max())
    ax.plot([minv, maxv], [minv, maxv], ls="--")
    ax.set_xlabel("True quantiles")
    ax.set_ylabel("Generated quantiles")
    ax.set_title("Q–Q plot")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", type=str, default=os.path.join(RES_DIR, "generator.keras"))
    ap.add_argument("--num-samples", type=int, default=100000)
    args = ap.parse_args()

    # Load real data & normalization params
    ages = maybe_get_data(BASE_DIR)
    params = np.load(os.path.join(RES_DIR, "norm_params.npz"))
    mn, sd = float(params["mean"]), float(params["std"])

    # Load generator & sample
    G = keras.models.load_model(args.generator)
    z = np.random.randn(args.num_samples, G.input_shape[1]).astype("float32")
    gen = G.predict(z, verbose=0).squeeze() * sd + mn

    # 1) KDE / histogram overlay
    fig1, ax1 = plt.subplots()
    sns.kdeplot(ages, label="True", bw_adjust=0.8, ax=ax1)
    sns.kdeplot(gen,  label="Generated", bw_adjust=0.8, ax=ax1)
    ax1.set_title("Age distribution (KDE overlay)")
    ax1.legend(); ax1.grid(True, alpha=0.25)
    fig1.tight_layout()
    out1 = os.path.join(FIG_DIR, "hist_overlay.png")
    fig1.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print("Wrote:", out1)

    # 2) CDF overlay
    ecdf_true = ECDF(ages); ecdf_gen = ECDF(gen)
    xs = np.linspace(min(ages.min(), gen.min()), max(ages.max(), gen.max()), 500)
    fig2, ax2 = plt.subplots()
    ax2.plot(xs, ecdf_true(xs), label="True CDF")
    ax2.plot(xs, ecdf_gen(xs),  label="Generated CDF")
    ax2.set_title("CDF overlay")
    ax2.legend(); ax2.grid(True, alpha=0.25)
    fig2.tight_layout()
    out2 = os.path.join(FIG_DIR, "cdf_overlay.png")
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print("Wrote:", out2)

    # 3) Q–Q plot
    fig3, ax3 = plt.subplots()
    qq_plot(gen, ages, ax3)
    fig3.tight_layout()
    out3 = os.path.join(FIG_DIR, "qq_plot.png")
    fig3.savefig(out3, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print("Wrote:", out3)

    print(f"True mean={ages.mean():.2f}, std={ages.std():.2f}")
    print(f"Generated mean={gen.mean():.2f}, std={gen.std():.2f}")

if __name__ == "__main__":
    main()
