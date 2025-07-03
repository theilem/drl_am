from dataclasses import dataclass, field

import tensorflow as tf
from tqdm import tqdm

from drl_am.agent.amppo import AMPPOAgent
from drl_am.trainer.feasibility import FeasibilityTrainer
from drl_am.utils.misc import DecayParams, get_decay_curve
from drl_am.utils.rollout import RolloutTF


class AMPPOTrainer(FeasibilityTrainer):
    @dataclass
    class Params:
        # Learning Rate
        batch_size: int = 128
        actor_lr: DecayParams = field(default_factory=lambda: DecayParams("exp", 3e-5, 0.1, 1_000_000))
        critic_lr: DecayParams = field(default_factory=lambda: DecayParams("exp", 1e-4, 0.1, 1_000_000))

        training_steps: int = 2_000_000
        num_gyms: int = 20
        rollout: RolloutTF.Params = field(default_factory=RolloutTF.Params)
        rollout_epochs: int = 5

        normalize_advantage: bool = False
        gamma: float = 0.97
        lmbda: float = 0.9

        epsilon: float = 0.2
        entropy: float = 0.01
        am: FeasibilityTrainer.Params = field(default_factory=FeasibilityTrainer.Params)

    def __init__(self, params: Params, gym, agent: AMPPOAgent, logger):
        FeasibilityTrainer.__init__(self, params.am, gym, agent, logger)
        self.params = params
        self.am_params = params.am
        self.agent = agent
        self.gym = gym
        self.logger = logger

        self.rollout = RolloutTF(params.rollout, num_par_agents=params.num_gyms)
        self.values = tf.Variable(tf.zeros((params.num_gyms, 1), dtype=tf.float32), trainable=False)
        self.advantages = tf.Variable(tf.zeros((self.rollout.size + 1, params.num_gyms, 1), dtype=tf.float32),
                                      trainable=False)

        optimizer_to_step_ratio = params.rollout_epochs / params.batch_size
        actor_lr = get_decay_curve(self.params.actor_lr, optimizer_to_step_ratio)
        critic_lr = get_decay_curve(self.params.critic_lr, optimizer_to_step_ratio)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

    def train(self, skip_feas=False):
        if not skip_feas:
            print("Training Feasibility and then Objective")
            print("Training Feasibility")
            self.ftrain()
        print("Training Objective")
        state = self.gym.reset_tf(self.params.num_gyms)
        obs = self.gym.get_obs(state)
        action, _ = self.agent.get_actor_log_prob_tf(obs)
        self.rollout.init_memory(obs, action)

        self.assign_values_tf(obs)

        progress = tqdm(total=self.params.training_steps, smoothing=1)
        num_steps = self.params.training_steps // self.params.num_gyms
        info = self.gym.get_info(state)
        rollout_infos = {key: tf.zeros_like(value[0], tf.float32) for key, value in info.items()}
        rollout_ep_counter = tf.constant(0, dtype=tf.int64)

        for step in range(num_steps):
            state, reset, info = self.collect_step_tf(state)
            progress.update(self.params.num_gyms)

            for key, value in rollout_infos.items():
                rollout_infos[key] += tf.reduce_sum(tf.where(reset[:, 0], tf.cast(info[key], tf.float32), 0), 0)
            rollout_ep_counter += tf.math.count_nonzero(reset)

            for k in range(self.params.num_gyms):
                self.logger.log_step()

            if (step + 1) % self.rollout.size == 0:
                infos = {key: value / tf.cast(rollout_ep_counter, tf.float32) for key, value in rollout_infos.items()}
                self.logger.log_rollout(infos)
                rollout_infos = {key: tf.zeros_like(value[0], tf.float32) for key, value in info.items()}
                rollout_ep_counter = tf.constant(0, dtype=tf.int64)
                train_log = self.train_rollout_tf()
                self.logger.log_train(train_log)
                obs = self.gym.get_obs(state)
                self.assign_values_tf(obs)

    @tf.function
    def assign_values_tf(self, obs):
        self.values.assign(self.agent.get_value_tf(obs))

    @tf.function
    def collect_step_tf(self, state):
        obs = self.gym.get_obs(state)
        action, latent, log_prob, x_t = self.agent.get_action_latent_log_prob_tf(obs, return_xt=True)

        next_state, reward, terminated, truncated, info = self.gym.step_tf(state, action)

        next_obs = self.gym.get_obs(next_state)
        next_value = self.agent.get_value_tf(next_obs)
        delta = reward + self.params.gamma * next_value * (1.0 - tf.cast(terminated, tf.float32)) - self.values

        reset = terminated | truncated

        self.rollout.store_transition(observation=obs, action=x_t, done=reset, value=self.values, log_prob=log_prob,
                                      delta=delta)

        next_state = self.gym.reset_state(next_state, reset)

        next_value = self.update_values(next_value, next_state, reset)
        self.values.assign(next_value)

        return next_state, reset, info

    @tf.function
    def update_values(self, values, state, reset):
        if tf.math.count_nonzero(reset) == 0:
            return values

        obs = self.gym.get_obs(state)
        reset = tf.squeeze(reset, -1)
        obs = {key: tf.boolean_mask(value, reset) for key, value in obs.items()}

        new_values = self.agent.get_value_tf(obs)

        where = tf.where(reset)
        values = tf.tensor_scatter_nd_update(values, where, new_values)
        return values

    @tf.function
    def train_rollout_tf(self):
        memory = self.rollout.memory
        rev_range = tf.range(self.rollout.size, dtype=tf.int32)[::-1]
        for idx in rev_range:
            next_idx = idx + 1
            self.advantages[idx].assign(
                memory["delta"][idx] + self.params.gamma * self.params.lmbda * self.advantages[next_idx] * (
                        1.0 - tf.cast(memory["done"][idx], dtype=tf.float32)))

        advantage = tf.reshape(self.advantages[:-1], (-1, 1))
        returns = advantage + tf.reshape(memory["value"], (-1, 1))

        if self.params.normalize_advantage:
            std = tf.math.reduce_std(advantage)
            if std > 1e-3:
                advantage = (advantage - tf.reduce_mean(advantage)) / std
        log_prob = tf.reshape(memory["log_prob"], (-1, 1))
        action = tf.reshape(memory["action"], (-1, memory["action"].shape[-1]))
        obs = {key: tf.reshape(value, (-1, *value.shape[2:])) for key, value in memory["observation"].items()}

        idx = tf.random.shuffle(tf.tile(tf.range(self.params.rollout.size), [self.params.rollout_epochs]))
        remainder = idx.shape[0] % self.params.batch_size
        if remainder > 0:
            idx = idx[:-remainder]
        idx_batched = tf.reshape(idx, (-1, self.params.batch_size))

        logs = {"critic_loss": tf.constant(0., dtype=tf.float32), "actor_loss": tf.constant(0., dtype=tf.float32),
                "entropy": tf.constant(0., dtype=tf.float32), "critic_lr": tf.constant(0., dtype=tf.float32),
                "actor_lr": tf.constant(0., dtype=tf.float32), "peak_loss": tf.constant(0., dtype=tf.float32)}

        for batch in idx_batched:
            batch_obs = {key: tf.gather(value, batch) for key, value in obs.items()}
            batch_action = tf.gather(action, batch)
            batch_returns = tf.gather(returns, batch)
            batch_advantage = tf.gather(advantage, batch)
            batch_log_prob = tf.gather(log_prob, batch)

            step_logs = self.train_step_tf(batch_obs, batch_action, batch_returns, batch_advantage, batch_log_prob)

            logs = {key: logs[key] + value for key, value in step_logs.items()}
        logs = {key: value / tf.cast(idx_batched.shape[0], tf.float32) for key, value in logs.items()}
        logs["average_return"] = tf.reduce_mean(returns)
        logs["average_advantage"] = tf.reduce_mean(tf.abs(advantage))
        return logs

    @tf.function
    def train_step_tf(self, obs, action, returns, advantages, old_log_prob):
        with tf.GradientTape() as critic_tape:
            value = self.agent.get_value_tf(obs)
            critic_loss = tf.keras.losses.Huber()(returns, value)
        critic_grads = critic_tape.gradient(critic_loss, self.agent.critic.trainable_variables)
        critic_grads = [tf.clip_by_norm(grad, 0.5) for grad in critic_grads]
        # critic_grads = tf.clip_by_global_norm(critic_grads, 1.0)[0]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.agent.critic.trainable_variables))

        with tf.GradientTape() as tape:
            log_prob, entropy = self.agent.get_log_prob_tf(obs, action, action_is_unsquashed=True)
            log_prob = log_prob[:, None]
            entropy = entropy[:, None]
            r = tf.math.exp(log_prob - old_log_prob)
            epsilon = self.params.epsilon

            loss = tf.minimum(r * advantages,
                              tf.clip_by_value(r, 1. - epsilon, 1. + epsilon) * advantages
                              ) + self.params.entropy * entropy

            actor_loss = -tf.reduce_mean(loss)

        actor_grads = tape.gradient(actor_loss, self.agent.actor.trainable_variables)
        actor_grads = [tf.clip_by_norm(grad, 0.5) for grad in actor_grads]
        # actor_grads = tf.clip_by_global_norm(actor_grads, 1.0)[0]
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.agent.actor.trainable_variables))

        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "entropy": tf.reduce_mean(entropy),
                "critic_lr": self.critic_optimizer.lr, "actor_lr": self.actor_optimizer.lr,
                "peak_loss": tf.reduce_max(tf.abs(loss))}