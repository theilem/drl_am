from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tqdm import tqdm


class Evaluator:
    @dataclass
    class Params:
        num_states: int = 64
        num_actions: int = 1024

        num_parallel: int = 50
        num_episodes: int = 1000

    def __init__(self, params, gym, agent, trainer=None):
        self.params = params
        self.gym = gym
        self.agent = agent
        self.trainer = trainer

    def evaluate_episode(self):
        return {}

    def fevaluate(self):
        results, actions, random_results, random_actions, info = self.fevaluate_tf()

        dist_good = 0
        for actions_for_state, result in zip(actions, results):
            res_num = result.numpy()
            if not np.any(res_num):
                continue
            good_actions = actions_for_state.numpy()[res_num]
            dist_good += np.mean(np.sum((good_actions[:, None] - good_actions[None, :]) ** 2, axis=-1) ** 0.5)

        dist_good /= len(actions)

        action_dists = np.array(
            [tf.reduce_mean(tf.reduce_sum((actions_for_state[:, None] - actions_for_state[None, :]) ** 2,
                                          axis=-1) ** 0.5).numpy() for actions_for_state in actions])

        action_dists_random = np.array(
            [tf.reduce_mean(tf.reduce_sum((actions_for_state[:, None] - actions_for_state[None, :]) ** 2,
                                          axis=-1) ** 0.5).numpy() for actions_for_state in random_actions])

        success_rate_per_state = tf.reduce_mean(tf.cast(results, tf.float32), axis=1)
        success_rate_per_state_rand = tf.reduce_mean(tf.cast(random_results, tf.float32), axis=1)

        success_ratio_to_rand = success_rate_per_state / (success_rate_per_state_rand + 1e-3)

        infos = {key: tf.reduce_mean(tf.cast(value, tf.float32)).numpy().item() for key, value in info.items()}

        return {
            "success_rate": tf.reduce_mean(success_rate_per_state).numpy().item(),
            "success_rate_rand": tf.reduce_mean(success_rate_per_state_rand).numpy().item(),
            "success_ratio_to_rand": tf.reduce_mean(success_ratio_to_rand).numpy().item(),
            "mean_dist": np.mean(action_dists),
            "dist_good": dist_good,
            "mean_dist_rand": np.mean(action_dists_random),
            **infos
        }

    @tf.function
    def fevaluate_tf(self):
        obs = self.gym.generate_states_tf(self.params.num_states)
        actions, latents = self.agent.get_random_feasible_actions_tf(obs, self.params.num_actions)
        results, info = self.gym.action_feasibility_tf(obs, actions, return_info=True)
        random_actions = tf.random.uniform((self.params.num_states, self.params.num_actions, self.gym.action_dim),
                                           -1.0, 1.0)
        random_results = self.gym.action_feasibility_tf(obs, random_actions)
        return results, actions, random_results, random_actions, info

    def evaluate_tf(self):
        state = self.gym.reset_tf(self.params.num_parallel)

        info = self.gym.get_info(state)
        eval_infos = {key: tf.zeros_like(value[0], tf.float32) for key, value in info.items()}
        eval_ep_counter = tf.constant(0, dtype=tf.int64)

        progress = tqdm(range(self.params.num_episodes))

        while True:
            state, reset, info = self.advance_gym_tf(state)

            if tf.reduce_any(reset):
                for key, value in eval_infos.items():
                    eval_infos[key] += tf.reduce_sum(tf.where(reset[:, 0], tf.cast(info[key], tf.float32), 0), 0)
                num_finished = tf.math.count_nonzero(reset)
                eval_ep_counter += num_finished
                progress.update(num_finished.numpy().item())
            if eval_ep_counter >= self.params.num_episodes:
                break

        eval_infos = {key: value / tf.cast(eval_ep_counter, tf.float32) for key, value in eval_infos.items()}
        return eval_infos

    @tf.function
    def advance_gym_tf(self, state):
        obs = self.gym.get_obs(state)
        action = self.agent.get_action_tf(obs, exploit=True)
        if self.trainer.params.project_action:
            action = self.trainer.project_actions_tf(obs, action[:, None, :])[:, 0, :]
        next_state, reward, terminated, truncated, info = self.gym.step_tf(state, action)
        reset = terminated | truncated
        next_state = self.gym.reset_state(next_state, reset)
        return next_state, reset, info
