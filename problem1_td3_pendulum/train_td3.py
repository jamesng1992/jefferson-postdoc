import os, argparse, numpy as np, matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf
from replay_buffer import ReplayBuffer
from td3_agent import TD3Agent

def plot_curves(steps, ep_returns, critic_losses, actor_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # Learning returns
    plt.figure()
    plt.plot(steps, ep_returns)
    plt.xlabel('Env steps'); plt.ylabel('Episode return'); plt.title('TD3: Return vs Steps'); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'learning_returns.png')); plt.close()

    # Losses
    c1, c2 = zip(*critic_losses) if critic_losses else ([], [])
    plt.figure()
    if c1: plt.plot(c1, label='Critic1 MSE')
    if c2: plt.plot(c2, label='Critic2 MSE')
    if actor_losses: plt.plot(actor_losses, label='Actor (−Q)')
    plt.legend(); plt.grid(True); plt.title('Losses')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'losses.png')); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--total-steps', type=int, default=150000)
    ap.add_argument('--start-steps', type=int, default=5000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--tau', type=float, default=0.005)
    ap.add_argument('--policy-noise', type=float, default=0.2)
    ap.add_argument('--noise-clip', type=float, default=0.5)
    ap.add_argument('--policy-delay', type=int, default=2)
    ap.add_argument('--exploration-noise', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--log-every', type=int, default=1000)
    ap.add_argument('--out-dir', type=str, default='results')
    args = ap.parse_args()

    env = gym.make('Pendulum-v1')
    env.reset(seed=args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = TD3Agent(obs_dim, act_dim, act_limit, gamma=args.gamma, tau=args.tau,
                     policy_noise=args.policy_noise, noise_clip=args.noise_clip,
                     policy_delay=args.policy_delay)

    buf = ReplayBuffer(obs_dim, act_dim, size=int(1e6))

    o, info = env.reset()
    ep_ret, ep_len, total_steps = 0.0, 0, 0
    ep_returns, critic_losses, actor_losses, steps_track = [], [], [], []

    while total_steps < args.total-steps if False else total_steps < args.total_steps:  # guard against hyphen typo
        if total_steps < args.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.select_action(o, noise_scale=args.exploration_noise)

        o2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        buf.store(o, a, r, o2, float(done))
        o = o2; ep_ret += r; ep_len += 1; total_steps += 1

        if done:
            ep_returns.append(ep_ret); steps_track.append(total_steps)
            o, info = env.reset()
            ep_ret, ep_len = 0.0, 0

        if total_steps >= args.start_steps:
            batch = buf.sample_batch(args.batch_size)
            obs = tf.convert_to_tensor(batch['obs'], dtype=tf.float32)
            acts = tf.convert_to_tensor(batch['acts'], dtype=tf.float32)
            rews = tf.convert_to_tensor(batch['rews'][:, None], dtype=tf.float32)
            next_obs = tf.convert_to_tensor(batch['next_obs'], dtype=tf.float32)
            dones = tf.convert_to_tensor(batch['done'][:, None], dtype=tf.float32)

            c1_loss, c2_loss = agent._train_critics(
                obs, acts, rews, next_obs, dones, act_limit,
                args.policy_noise, args.noise_clip, args.gamma
            )
            critic_losses.append((float(c1_loss.numpy()), float(c2_loss.numpy())))

            if int(agent.total_it.numpy()) % args.policy_delay == 0:
                a_loss = agent._train_actor_and_update_targets(obs, args.tau)
                actor_losses.append(float(a_loss.numpy()))
            agent.total_it.assign_add(1)

        if total_steps % args.log_every == 0 and ep_returns:
            print(f"Step {total_steps}: last ep return {ep_returns[-1]:.1f}")

        # --- periodic checkpoint save every 5k steps ---
        if total_steps % 5000 == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"td3_actor_step{total_steps}.keras")
            agent.actor.save(ckpt_path)
            print(f"[SAVE] Actor checkpoint -> {ckpt_path}")

    # --- FINAL SAVE FIRST (so plotting issues won't lose models) ---
    os.makedirs(args.out_dir, exist_ok=True)
    actor_path   = os.path.join(args.out_dir, 'td3_actor.keras')
    critic1_path = os.path.join(args.out_dir, 'td3_critic1.keras')
    critic2_path = os.path.join(args.out_dir, 'td3_critic2.keras')
    agent.actor.save(actor_path)
    agent.critic1.save(critic1_path)
    agent.critic2.save(critic2_path)
    print(f"[FINAL SAVE] Actor  -> {actor_path}")
    print(f"[FINAL SAVE] Critic1-> {critic1_path}")
    print(f"[FINAL SAVE] Critic2-> {critic2_path}")

    # --- Plot, but never crash training if backend has issues ---
    try:
        plot_curves(steps_track, ep_returns, critic_losses, actor_losses, os.path.join('plots'))
        print("[PLOT] Wrote learning_returns.png and losses.png")
    except Exception as e:
        print(f"[PLOT] Skipped due to error: {e}")


if __name__ == '__main__':
    main()
