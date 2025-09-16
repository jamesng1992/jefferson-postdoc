# problem1_td3_pendulum/make_action_stats.py
import os, argparse, numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf

BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to td3_actor.keras")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    # load actor
    actor = tf.keras.models.load_model(args.checkpoint)

    # env
    env = gym.make("Pendulum-v1")
    all_actions = []
    per_ep_abs_mean = []

    for ep in range(args.episodes):
        o, _ = env.reset()
        done = False
        actions = []
        while not done:
            a = actor.predict(o[None,:], verbose=0)[0]
            actions.append(a)
            o, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
        actions = np.array(actions)               # [T, act_dim]
        all_actions.append(actions)
        per_ep_abs_mean.append(np.abs(actions).mean())

    all_actions = np.concatenate(all_actions, axis=0)  # [sumT, act_dim]

    # 1) Histogram of action magnitudes
    plt.figure()
    plt.hist(all_actions.flatten(), bins=50, alpha=0.9)
    plt.title("TD3 Action Distribution (Pendulum-v1)")
    plt.xlabel("Action value"); plt.ylabel("Frequency"); plt.grid(True)
    out1 = os.path.join(PLOTS_DIR, "action_stats.png")
    plt.tight_layout(); plt.savefig(out1); plt.close()

    # 2) Per-episode mean |action| (sanity trend)
    plt.figure()
    plt.plot(per_ep_abs_mean, marker="o")
    plt.title("Per-episode mean absolute action")
    plt.xlabel("Episode"); plt.ylabel("Mean |action|"); plt.grid(True)
    out2 = os.path.join(PLOTS_DIR, "action_episode_means.png")
    plt.tight_layout(); plt.savefig(out2); plt.close()

    print("Wrote:", out1)
    print("Wrote:", out2)

if __name__ == "__main__":
    main()

