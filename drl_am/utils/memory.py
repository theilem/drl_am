from dataclasses import dataclass

import numpy as np
import tensorflow as tf

class ReplayMemory:
    @dataclass
    class Params:
        size: int = 1000000

    def __init__(self, params: Params):
        self.params = params

        self.memory = {}

        # Flags
        self.obs_dict = False
        self.head = 0
        self.full = False
        self.initialized = False

    def init_memory(self, observation, action):
        assert not isinstance(action, dict), "actions cannot be dictionaries"

        shape = list(action.shape)
        if len(action.shape) == 0:
            shape = [self.params.size]
        else:
            shape = [self.params.size] + shape
        self.memory["action"] = np.zeros(shape, dtype=action.dtype)

        self.memory["reward"] = np.zeros((self.params.size, 1), dtype=np.float32)
        self.memory["done"] = np.zeros((self.params.size, 1), dtype=bool)
        self.memory["failed"] = np.zeros((self.params.size, 1), dtype=bool)

        if isinstance(observation, dict):
            self.obs_dict = True
            for k, o in {"observation": observation, "next_observation": observation}.items():
                obs_mem = {}
                for key, value in o.items():
                    shape = list(value.shape)
                    shape[0] = self.params.size
                    obs_mem[key] = np.zeros(shape, dtype=value.dtype)
                self.memory[k] = obs_mem
        else:
            assert isinstance(observation, np.ndarray), "Observation should be a numpy array then"

            shape = list(observation.shape)
            shape = [self.params.size] + shape
            self.memory["observation"] = np.zeros(shape, dtype=observation.dtype)
            self.memory["next_observation"] = np.zeros(shape, dtype=observation.dtype)
        self.initialized = True

    def store_transition(self, observation, action, reward, next_observation, failed, done):
        if not self.initialized:
            self.init_memory(observation, action)

        idx = self.head

        if self.obs_dict:
            for k, o in {"observation": observation, "next_observation": next_observation}.items():
                for key, value in o.items():
                    self.memory[k][key][idx] = value
        else:
            self.memory["observation"][idx] = observation
            self.memory["next_observation"][idx] = next_observation

        self.memory["action"][idx] = action
        self.memory["reward"][idx] = reward
        self.memory["done"][idx] = done
        self.memory["failed"][idx] = failed

        self.head += 1
        if self.head >= self.params.size:
            self.full = True
            self.head = 0

    def store_transitions(self, observations, actions, rewards, next_observations, failed, dones):
        if not self.initialized:
            self.init_memory(observations, actions[0])

        num = actions.shape[0]
        idx = self.head

        if idx + num > self.params.size:
            part = self.params.size - idx - 1
            if self.obs_dict:
                obs1 = {key: obs[:part] for key, obs in observations.items()}
                obs2 = {key: obs[part:] for key, obs in observations.items()}
                nobs1 = {key: obs[:part] for key, obs in next_observations.items()}
                nobs2 = {key: obs[part:] for key, obs in next_observations.items()}
            else:
                obs1 = observations[:part]
                obs2 = observations[part:]
                nobs1 = next_observations[:part]
                nobs2 = next_observations[part:]
            self.store_transitions(obs1, actions[:part], rewards[:part], nobs1, failed[:part], dones[:part])
            self.store_transitions(obs2, actions[part:], rewards[part:], nobs2, failed[part:], dones[part:])
            return

        if self.obs_dict:
            for k, o in {"observation": observations, "next_observation": next_observations}.items():
                for key, value in o.items():
                    self.memory[k][key][idx:idx + num] = value
        else:
            self.memory["observation"][idx:idx + num] = observations
            self.memory["next_observation"][idx:idx + num] = next_observations

        self.memory["action"][idx:idx + num] = actions
        self.memory["reward"][idx:idx + num, 0] = rewards
        self.memory["done"][idx:idx + num, 0] = dones
        self.memory["failed"][idx:idx + num, 0] = failed

        self.head += num
        if self.head >= self.params.size:
            self.full = True
            self.head = 0

    def sample(self, batch_size, replace=False):
        max_val = self.params.size if self.full else self.head
        idx = np.random.choice(max_val, size=batch_size, replace=replace)
        # idx = np.random.randint(low=0, high=max_val, size=batch_size)

        sample = {}
        if self.obs_dict:
            sample["observation"] = {}
            sample["next_observation"] = {}
            for key, value in self.memory["observation"].items():
                sample["observation"][key] = value[idx]
            for key, value in self.memory["next_observation"].items():
                sample["next_observation"][key] = value[idx]
        else:
            sample["observation"] = self.memory["observation"][idx]
            sample["next_observation"] = self.memory["next_observation"][idx]

        sample["action"] = self.memory["action"][idx]
        sample["reward"] = self.memory["reward"][idx]
        sample["done"] = self.memory["done"][idx]
        sample["failed"] = self.memory["failed"][idx]

        return sample
