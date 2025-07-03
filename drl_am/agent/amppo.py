from dataclasses import dataclass

import tensorflow as tf
import tensorflow_probability as tfp

from drl_am.agent.am import AMAgent
from drl_am.agent.model import create_model
from drl_am.utils.misc import dict_to_tensor


class AMPPOAgent(AMAgent):
    @dataclass
    class Params():
        layer_size: int = 256
        num_layers_pre_attention: int = 1
        num_self_attention_layers: int = 0
        num_attention_layers: int = 3
        num_heads: int = 16
        key_dim: int = 16
        num_layers: int = 3
        layer_norm_before_skip: bool = False

        num_critics: int = 2

    def __init__(self, params: Params, obs_space, action_dim):
        AMAgent.__init__(self, params, obs_space, action_dim)
        self.params = params
        self.action_dim = action_dim

        self.actor = create_model(input_space=obs_space, output_space={"mean": action_dim, "log_std": action_dim},
                                       layer_size=params.layer_size,
                                       num_attention_layers=params.num_attention_layers,
                                       num_heads=params.num_heads,
                                       key_dim=params.key_dim,
                                       num_layers=params.num_layers,
                                       num_layers_pre_attention=params.num_layers_pre_attention,
                                       num_self_attention_layers=params.num_self_attention_layers,
                                       layer_norm_before_skip=params.layer_norm_before_skip,
                                       output_activation='linear', model_name="actor")

        self.critic = create_model(input_space=obs_space, output_space=1, output_activation='linear',
                                         layer_size=params.layer_size,
                                         num_attention_layers=params.num_attention_layers,
                                         num_heads=params.num_heads,
                                         key_dim=params.key_dim,
                                         num_layers=params.num_layers,
                                         num_layers_pre_attention=params.num_layers_pre_attention,
                                         num_self_attention_layers=params.num_self_attention_layers,
                                         layer_norm_before_skip=params.layer_norm_before_skip,
                                         model_name=f"critic")


    def get_action(self, obs, exploit=False):
        obs = dict_to_tensor(obs)
        action, latent, log_prob = self.get_action_latent_log_prob_tf(obs, exploit)
        if action.shape[0] == 1:
            return action.numpy()[0], latent.numpy()[0], log_prob.numpy()[0] if log_prob is not None else None
        return action.numpy(), latent.numpy(), log_prob.numpy()

    def get_actions(self, obs, num_actions=1):
        obs = dict_to_tensor(obs)
        action, latent = self.get_actions_tf(obs, num_actions=num_actions)
        return action.numpy()[0], latent.numpy()[0]

    @tf.function
    def get_action_tf(self, obs, exploit=False):
        action, _, _ = self.get_action_latent_log_prob_tf(obs, exploit)
        return action

    @tf.function
    def actor_exploit_tf(self, actor_input):
        return tf.math.tanh(self.actor(actor_input)["mean"])

    @tf.function
    def get_action_latent_log_prob_tf(self, obs, exploit=False, return_xt=False):
        if exploit:
            latent = self.actor_exploit_tf(obs)
            log_prob = None
            x_t = None
        else:
            latent, log_prob, x_t = self.get_actor_log_prob_tf(obs, return_xt=True)

        feas_input = self.create_feas_input_tf(latent[:, None, :], obs)

        action = self.feasibility_policy(feas_input)
        if return_xt:
            return tf.squeeze(action, axis=1), latent, log_prob, x_t
        return tf.squeeze(action, axis=1), latent, log_prob

    @tf.function
    def get_actor_log_prob_tf(self, actor_input, num_actions=1, return_xt=False):
        dist_params = self.actor(actor_input)

        sample_shape = (num_actions,) if num_actions > 1 else ()

        action, log_prob, x_t = self.sample_actions_tf(dist_params, sample_shape)

        if num_actions > 1:
            action = tf.transpose(action, perm=[1, 0, 2])
            log_prob = tf.transpose(log_prob, perm=[1, 0])
            x_t = tf.transpose(x_t, perm=[1, 0, 2])

        if return_xt:
            return action, tf.expand_dims(log_prob, -1), x_t
        return action, tf.expand_dims(log_prob, -1)

    @tf.function
    def get_log_prob_tf(self, obs, action, action_is_unsquashed=False):
        dist_params = self.actor(obs)

        if not action_is_unsquashed:
            unsquashed_action = tf.math.atanh(tf.clip_by_value(action, -0.999999, 0.999999))
        else:
            unsquashed_action = action
            action = tf.math.tanh(action)

        std = self.get_std_tf(dist_params)
        dist = tfp.distributions.MultivariateNormalDiag(dist_params["mean"], std)
        log_prob = dist.log_prob(unsquashed_action) - tf.reduce_sum(tf.math.log(1 - action ** 2 + 1e-6),
                                                                    axis=-1, keepdims=False)
        action_samp = dist.sample()
        log_prob_samp = dist.log_prob(action_samp) - tf.reduce_sum(
            tf.math.log(1 - tf.math.tanh(action_samp) ** 2 + 1e-6),
            axis=-1, keepdims=False)
        entropy = -log_prob_samp
        return log_prob, entropy

    @tf.function
    def sample_actions_tf(self, dist_params, sample_shape=()):
        std = self.get_std_tf(dist_params)
        dist = tfp.distributions.MultivariateNormalDiag(dist_params["mean"], std)
        x_t = dist.sample(sample_shape)
        y_t = tf.math.tanh(x_t)
        log_prob = dist.log_prob(x_t) - tf.reduce_sum(tf.math.log(1 - y_t ** 2 + 1e-6), axis=-1, keepdims=False)
        action = y_t
        return action, log_prob, x_t

    @tf.function
    def get_std_tf(self, dist_params):
        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        log_std = tf.math.tanh(dist_params["log_std"])
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        std = tf.math.exp(log_std)
        return std

    @tf.function
    def get_actions_tf(self, obs, num_actions):
        latents, _ = self.get_actor_log_prob_tf(obs, num_actions=num_actions)
        latents = tf.reshape(latents, (-1, num_actions, self.latent_dim))
        feas_input = self.create_feas_input_tf(latents, obs)

        action = self.feasibility_policy(feas_input)
        return action, latents

    @tf.function
    def get_value_tf(self, obs) -> tf.Tensor:
        return self.critic(obs)

    def save_weights(self, path, name="weights_latest"):
        self.actor.save_weights(f"{path}/actor_{name}")
        self.critic.save_weights(f"{path}/critic_{name}")
        AMAgent.save_weights(self, path, name)

    def load_weights(self, path, name="weights_latest"):
        self.actor.load_weights(f"{path}/actor_{name}")
        self.critic.load_weights(f"{path}/critic_{name}")
        AMAgent.load_weights(self, path, name)
