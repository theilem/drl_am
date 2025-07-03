from dataclasses import dataclass
import tensorflow as tf


class RolloutTF:
    @dataclass
    class Params:
        size: int = 100_000

    def __init__(self, params: Params, num_par_agents=20):
        self.params = params

        self.memory = {}

        assert params.size % num_par_agents == 0, "Size should be divisible by the number of parallel agents."
        self.size = params.size // num_par_agents
        self.num_par_agents = num_par_agents

        # Flags
        self.initialized = False
        self.obs_dict = False
        self.head = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.full = tf.Variable(False, dtype=tf.bool, trainable=False)

    def init_memory(self, observation, action):
        assert not isinstance(action, dict), "actions cannot be dictionaries"

        shape = list(action.shape)
        shape = [self.size, self.num_par_agents] + shape[1:]
        self.memory["action"] = tf.Variable(initial_value=tf.zeros(shape, dtype=action.dtype), trainable=False)

        self.memory["reward"] = tf.Variable(initial_value=tf.zeros((self.size, self.num_par_agents, 1),
                                                                   dtype=tf.float32), trainable=False)
        self.memory["done"] = tf.Variable(initial_value=tf.zeros((self.size, self.num_par_agents, 1), dtype=tf.bool),
                                          trainable=False)
        self.memory["value"] = tf.Variable(
            initial_value=tf.zeros((self.size, self.num_par_agents, 1), dtype=tf.float32),
            trainable=False)
        self.memory["c_value"] = tf.Variable(
            initial_value=tf.zeros((self.size, self.num_par_agents, 1), dtype=tf.float32),
            trainable=False)
        self.memory["log_prob"] = tf.Variable(
            initial_value=tf.zeros((self.size, self.num_par_agents, 1), dtype=tf.float32),
            trainable=False)
        self.memory["delta"] = tf.Variable(
            initial_value=tf.zeros((self.size, self.num_par_agents, 1), dtype=tf.float32),
            trainable=False)
        self.memory["c_delta"] = tf.Variable(
            initial_value=tf.zeros((self.size, self.num_par_agents, 1), dtype=tf.float32),
            trainable=False)

        if isinstance(observation, dict):
            self.obs_dict = True
            obs_mem = {}
            for key, value in observation.items():
                shape = list(value.shape)
                shape = [self.size, self.num_par_agents] + shape[1:]
                obs_mem[key] = tf.Variable(initial_value=tf.zeros(shape, dtype=value.dtype), trainable=False)
            self.memory["observation"] = obs_mem
        else:
            assert isinstance(observation, tf.Tensor), "Observation should be a tensor then"

            shape = list(observation.shape)
            shape = [self.size, self.num_par_agents] + shape[1:]
            self.memory["observation"] = tf.Variable(initial_value=tf.zeros(shape,
                                                                            dtype=observation.dtype), trainable=False)
        self.initialized = True

    @tf.function
    def store_transition(self, observation, action, log_prob, value, done, delta, c_delta=None, c_value=None):
        if not self.initialized:
            raise ValueError("Initialization missing. Call init_memory first.")

        idx = self.head

        if self.obs_dict:
            for key, val in observation.items():
                self.memory["observation"][key][idx].assign(val)
        else:
            self.memory["observation"][idx].assign(observation)

        self.memory["action"][idx].assign(action)
        self.memory["done"][idx].assign(done)
        self.memory["value"][idx].assign(value)
        self.memory["log_prob"][idx].assign(log_prob)
        self.memory["delta"][idx].assign(delta)
        if c_delta is not None:
            self.memory["c_delta"][idx].assign(c_delta)
        if c_value is not None:
            self.memory["c_value"][idx].assign(c_value)

        self.head.assign_add(1)
        if self.head >= self.size:
            self.full.assign(True)
            self.head.assign(0)
