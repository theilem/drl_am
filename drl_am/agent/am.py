import numpy as np

import tensorflow as tf

from drl_am.agent.model import create_model
from drl_am.utils.misc import dict_to_tensor


class AMAgent:
    def __init__(self, params, obs_space, action_dim):
        self.action_dim = action_dim
        self.latent_dim = action_dim

        feas_input = obs_space.copy()
        feas_input["latent"] = (None, self.latent_dim)
        feas_input.pop("goal", None)
        feas_input.pop("mult_goals", None)
        feas_input.pop("num_goals", None)
        self.feasibility_policy = create_model(input_space=feas_input, output_space=action_dim,
                                               output_activation="tanh",
                                               mix_state_into_constraints=False,
                                               layer_size=params.layer_size,
                                               num_attention_layers=params.num_attention_layers,
                                               num_heads=params.num_heads,
                                               key_dim=params.key_dim,
                                               num_layers=params.num_layers,
                                               num_layers_pre_attention=params.num_layers_pre_attention,
                                               num_self_attention_layers=params.num_self_attention_layers,
                                               layer_norm_before_skip=params.layer_norm_before_skip,
                                               model_name="feasibility_policy")

    def get_feasible_action(self, obs):
        batch_size = obs["internal_state"].shape[0]
        obs = dict_to_tensor(obs)
        latent = np.random.uniform(-1, 1, size=(batch_size, self.action_dim))
        action = self.get_feasible_action_tf(obs, latent).numpy()
        if action.shape[0] == 1:
            return action[0], latent[0]
        return action, latent

    def get_feasible_actions(self, obs, num_actions=1):
        obs = dict_to_tensor(obs)
        sample = np.random.uniform(-1, 1, size=(1, num_actions, self.action_dim))
        return self.get_feasible_actions_tf(obs, sample).numpy()[0], sample[0]

    def get_specific_feasible_action(self, obs, latent):
        obs = dict_to_tensor(obs)
        latent = np.reshape(latent, (1, self.action_dim))
        return self.get_feasible_action_tf(obs, latent).numpy()[0]

    @tf.function
    def create_feas_input_tf(self, latent, obs):
        feas_input = {k: v for k, v in obs.items()}
        feas_input["latent"] = latent
        feas_input.pop("goal", None)
        feas_input.pop("mult_goals", None)
        feas_input.pop("num_goals", None)
        return feas_input

    @tf.function
    def get_feasible_action_tf(self, obs, latent):
        return tf.squeeze(self.get_feasible_actions_tf(obs, latent[:, None, :]), axis=1)

    @tf.function
    def get_random_feasible_actions_tf(self, obs, num_actions):
        batch_size = tf.shape(obs["internal_state"])[0]
        latent = tf.random.uniform(shape=(batch_size, num_actions, self.latent_dim), minval=-1, maxval=1)
        return self.get_feasible_actions_tf(obs, latent), latent

    @tf.function
    def get_feasible_actions_tf(self, obs, latent):
        feas_input = self.create_feas_input_tf(latent, obs)
        actions = self.feasibility_policy(feas_input)
        return actions

    @property
    def feasibility_variables(self):
        return self.feasibility_policy.trainable_variables

    def save_weights(self, path, name="weights_latest"):
        self.feasibility_policy.save_weights(f"{path}/feas_pol_{name}")

    def load_weights(self, path, name="weights_latest"):
        self.feasibility_policy.load_weights(f"{path}/feas_pol_{name}")

    def load_weights_feasibility(self, path, name="weights_latest"):
        self.feasibility_policy.load_weights(f"{path}/feas_pol_{name}")
