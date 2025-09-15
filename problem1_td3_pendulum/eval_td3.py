import argparse, os, numpy as np, gymnasium as gym, tensorflow as tf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default='results/td3_actor.keras')
    ap.add_argument('--episodes', type=int, default=10)
    args = ap.parse_args()

    env = gym.make('Pendulum-v1', render_mode=None)
    actor = tf.keras.models.load_model(args.checkpoint)

    returns = []
    for ep in range(args.episodes):
        o, _ = env.reset()
        ep_ret = 0.0; done = False
        while not done:
            a = actor.predict(o[None,:], verbose=0)[0]
            o, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            done = terminated or truncated
        returns.append(ep_ret)
        print(f"Episode {ep+1}: return {ep_ret:.1f}")
    print(f"Avg return over {args.episodes} eps: {np.mean(returns):.1f}")

if __name__ == '__main__':
    main()
