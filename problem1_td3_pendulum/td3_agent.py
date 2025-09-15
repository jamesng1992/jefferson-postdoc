import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def polyak_update(source_vars, target_vars, tau=0.005):
    for s, t in zip(source_vars, target_vars):
        t.assign(tau * s + (1.0 - tau) * t)

class TD3Agent:
    def __init__(self, obs_dim, act_dim, act_limit, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_delay=2, lr=3e-4):
        self.act_limit = act_limit
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = tf.Variable(0, dtype=tf.int64, trainable=False)

        # Actor
        inputs = keras.Input(shape=(obs_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        out = layers.Dense(act_dim, activation='tanh')(x)
        outputs = out * act_limit
        self.actor = keras.Model(inputs, outputs)
        self.actor_target = keras.models.clone_model(self.actor)
        self.actor_target.set_weights(self.actor.get_weights())

        # Critics (twin)
        def make_critic():
            obs_in = keras.Input(shape=(obs_dim,))
            act_in = keras.Input(shape=(act_dim,))
            x = layers.Concatenate()([obs_in, act_in])
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dense(256, activation='relu')(x)
            q = layers.Dense(1, activation=None)(x)
            return keras.Model([obs_in, act_in], q)

        self.critic1 = make_critic()
        self.critic2 = make_critic()
        self.critic1_target = keras.models.clone_model(self.critic1)
        self.critic2_target = keras.models.clone_model(self.critic2)
        self.critic1_target.set_weights(self.critic1.get_weights())
        self.critic2_target.set_weights(self.critic2.get_weights())

        self.actor_opt = keras.optimizers.Adam(learning_rate=lr)
        self.critic1_opt = keras.optimizers.Adam(learning_rate=lr)
        self.critic2_opt = keras.optimizers.Adam(learning_rate=lr)

    def select_action(self, obs, noise_scale=0.1):
        a = self.actor.predict(obs[None, :], verbose=0)[0]
        if noise_scale > 0:
            a = a + noise_scale * np.random.randn(*a.shape)
        return np.clip(a, -self.act_limit, self.act_limit)

    @tf.function
    def _train_critics(self, obs, acts, rews, next_obs, done, act_limit, policy_noise, noise_clip, gamma):
        # target actions with smoothing noise
        noise = tf.clip_by_value(
            tf.random.normal(shape=tf.shape(acts), stddev=policy_noise),
            -noise_clip, noise_clip
        )
        next_act = self.actor_target(next_obs) + noise
        next_act = tf.clip_by_value(next_act, -act_limit, act_limit)

        # target Q
        target_q1 = self.critic1_target([next_obs, next_act])
        target_q2 = self.critic2_target([next_obs, next_act])
        target_q = tf.minimum(target_q1, target_q2)
        target_q = rews + gamma * (1.0 - done) * tf.stop_gradient(target_q)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            q1 = self.critic1([obs, acts])
            q2 = self.critic2([obs, acts])
            critic1_loss = tf.reduce_mean((q1 - target_q)**2)
            critic2_loss = tf.reduce_mean((q2 - target_q)**2)
        grads1 = tape1.gradient(critic1_loss, self.critic1.trainable_variables)
        grads2 = tape2.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic1_opt.apply_gradients(zip(grads1, self.critic1.trainable_variables))
        self.critic2_opt.apply_gradients(zip(grads2, self.critic2.trainable_variables))
        return critic1_loss, critic2_loss

    @tf.function
    def _train_actor_and_update_targets(self, obs, tau):
        with tf.GradientTape() as tape:
            actions = self.actor(obs)
            q = self.critic1([obs, actions])
            actor_loss = -tf.reduce_mean(q)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

        # Polyak update
        for s, t in zip(self.actor.variables, self.actor_target.variables):
            t.assign(tau * s + (1.0 - tau) * t)
        for s, t in zip(self.critic1.variables, self.critic1_target.variables):
            t.assign(tau * s + (1.0 - tau) * t)
        for s, t in zip(self.critic2.variables, self.critic2_target.variables):
            t.assign(tau * s + (1.0 - tau) * t)
        return actor_loss
