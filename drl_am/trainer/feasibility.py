from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

import tensorflow as tf

from drl_am.utils.misc import DecayParams


@dataclass
class DivergenceParams:
    n: int = 1024
    sigma: float = 0.1
    sigma_prime_factor: float = 2.0
    epsilon: float = 1e-18
    regularization: float = 0.0


@dataclass
class LearningParams:
    training_steps: int = 1_000_000
    batch_size_actor: int = 16
    lr: DecayParams = field(default_factory=lambda: DecayParams("exp", 5e-5, 0.1, 1_000_000))


class FeasibilityTrainer:
    @dataclass
    class Params:
        divergence: DivergenceParams = field(default_factory=DivergenceParams)
        learning: LearningParams = field(default_factory=LearningParams)

    def __init__(self, params, gym, agent, logger):
        self.params = params
        self.am_params = params
        self.gym = gym
        self.agent = agent
        self.logger = logger

        self.sigma = tf.Variable(self.am_params.divergence.sigma, trainable=False)
        self.action_dim = self.gym.action_dim
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            self.am_params.learning.lr.base,
            self.am_params.learning.lr.decay_steps,
            self.am_params.learning.lr.decay_rate
        )
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.step = 0

    def ftrain(self):
        for self.step in tqdm(range(self.am_params.learning.training_steps)):
            logs = self.ftrain_step_tf()
            if self.logger is not None:
                self.logger.log_fstep(None)
                self.logger.log_ftrain(logs)

    @tf.function
    def ftrain_step_tf(self):
        obs = self.gym.generate_states_tf(self.am_params.learning.batch_size_actor)
        actor_train_log = self._ftrain_step_tf(obs)
        return actor_train_log

    @tf.function
    def _ftrain_step_tf(self, obs):
        if self.am_params.divergence.sigma_prime_factor == 1.0:
            return self._ftrain_step_simple_tf(obs)
        n = self.am_params.divergence.n

        sigma = tf.cast(self.sigma, dtype=tf.float32)
        sigma_prime = tf.cast(self.am_params.divergence.sigma_prime_factor * sigma, dtype=tf.float32)
        epsilon = tf.cast(self.am_params.divergence.epsilon, dtype=tf.float32)

        log = tf.math.log
        logs = {}

        with tf.GradientTape() as tape:
            # sample from q
            supports, latents = self.agent.get_random_feasible_actions_tf(obs, n)

            # sample from q_hat
            noise = tf.random.normal(supports.shape, stddev=sigma_prime, dtype=tf.float32)
            samples = tf.stop_gradient(supports + noise)

            distance_extra_samples = tf.reduce_sum((supports[:, :, None] - samples[:, None, :]) ** 2, axis=-1)

            # Calculate scores for the new samples
            scores_actions_samples = self.gym.action_feasibility_tf(obs=obs, actions=samples)

            scores_samples = tf.cast(tf.stop_gradient(scores_actions_samples), dtype=tf.float32)

            q_hat_samples = self.kernel_density_estimate(distance_extra_samples, sigma, s=self.action_dim)
            q_hat_prime_samples = self.kernel_density_estimate(distance_extra_samples, sigma_prime,
                                                               s=self.action_dim)
            q_hat_clipped = tf.maximum(q_hat_samples, epsilon)
            q_hat_prime_clipped = tf.maximum(q_hat_prime_samples, epsilon)

            # Not self normalized importance sampling
            rs = scores_samples / q_hat_prime_clipped
            volume = tf.maximum(tf.reduce_mean(rs, axis=1, keepdims=True), epsilon)

            p_samples = scores_samples / volume

            loss = 1 / 2 * tf.reduce_mean(tf.stop_gradient(
                q_hat_samples / q_hat_prime_clipped * log(2 * q_hat_clipped / (p_samples + q_hat_clipped))) * log(
                q_hat_samples + epsilon))

            logs.update({
                "js_loss": loss,
            })

            if self.am_params.divergence.regularization > 0:
                dist_latent_action = tf.reduce_sum((supports - tf.stop_gradient(latents)) ** 2, axis=-1) ** 0.5
                regularization_loss = self.am_params.divergence.regularization * tf.reduce_mean(dist_latent_action)
                loss = loss + regularization_loss
                logs.update({
                    "regularization_loss": regularization_loss,
                    "total_loss": loss,
                })

        grad = tape.gradient(loss, self.agent.feasibility_variables)
        self.optimizer_actor.apply_gradients(zip(grad, self.agent.feasibility_variables))
        return logs

    @tf.function
    def _ftrain_step_simple_tf(self, obs):
        n = self.am_params.divergence.n

        sigma = tf.cast(self.sigma, dtype=tf.float32)
        epsilon = tf.cast(self.am_params.divergence.epsilon, dtype=tf.float32)

        log = tf.math.log
        logs = {}

        with tf.GradientTape() as tape:
            # sample from q
            supports, latents = self.agent.get_random_feasible_actions_tf(obs, n)

            # sample from q_hat
            noise = tf.random.normal(supports.shape, stddev=sigma, dtype=tf.float32)
            samples = tf.stop_gradient(supports + noise)

            distance_extra_samples = tf.reduce_sum((supports[:, :, None] - samples[:, None, :]) ** 2, axis=-1)

            # Calculate scores for the new samples
            scores_actions_samples = self.gym.action_feasibility_tf(obs=obs, actions=samples)

            scores_samples = tf.cast(tf.stop_gradient(scores_actions_samples), dtype=tf.float32)

            q_hat_samples = self.kernel_density_estimate(distance_extra_samples, sigma, s=self.action_dim)
            q_hat_clipped = tf.maximum(q_hat_samples, epsilon)

            # Not self normalized importance sampling
            rs = scores_samples / q_hat_clipped
            volume = tf.maximum(tf.reduce_mean(rs, axis=1, keepdims=True), epsilon)

            p_samples = scores_samples / volume

            loss = 1 / 2 * tf.reduce_mean(tf.stop_gradient(log(2 * q_hat_clipped / (p_samples + q_hat_clipped))) * log(
                q_hat_samples + epsilon))

            logs.update({
                "js_loss": loss,
            })

            if self.am_params.divergence.regularization > 0:
                dist_latent_action = tf.reduce_sum((supports - tf.stop_gradient(latents)) ** 2, axis=-1) ** 0.5
                regularization_loss = self.am_params.divergence.regularization * tf.reduce_mean(dist_latent_action)
                loss = loss + regularization_loss
                logs.update({
                    "regularization_loss": regularization_loss,
                    "total_loss": loss,
                })

        grad = tape.gradient(loss, self.agent.feasibility_variables)
        self.optimizer_actor.apply_gradients(zip(grad, self.agent.feasibility_variables))
        return logs

    @staticmethod
    @tf.function
    def kernel_density_estimate(distance_squared, sigma, s):
        """
        distance_squared [B, n, m]
        """
        scale = 1 / (tf.sqrt(2 * np.pi) ** s * sigma ** s)
        g_support = scale * tf.math.exp(-1 / 2 * (distance_squared / sigma ** 2))
        # [B, n, m]

        q_hat_support = tf.reduce_mean(g_support, axis=1)
        return q_hat_support
