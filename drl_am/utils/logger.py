import json
import os
from collections import deque
from dataclasses import dataclass

import tensorflow as tf

from drl_am.utils.misc import dict_mean


class Logger:
    @dataclass
    class Params:
        loss_period: int = 100
        floss_period: int = 1000
        evaluation_period: int = 1000
        evaluation_start: int = 1000
        evaluation_episodes: int = 1
        save_weights: int = 1000
        log_episodes: bool = True
        batched_episode_period: int = 10000

    def __init__(self, params: Params, log_dir, agent, evaluator=None):
        self.params = params
        self.log_dir = log_dir
        self.evaluator = evaluator
        self.agent = agent
        # self.agent.save_network(f"{self.log_dir}/models/")
        self.agent.save_weights(f"{self.log_dir}/models/")

        self.log_writer = tf.summary.create_file_writer(self.log_dir + '/training')
        print(f"Logging to: \n{os.getcwd()}/{self.log_dir}")

        self.train_steps = 0
        self.ftrain_steps = 0
        self.steps = 0
        self.fsteps = 0

        self.train_logs = deque(maxlen=self.params.loss_period)
        self.ftrain_logs = deque(maxlen=self.params.loss_period)
        self.action_probs = []
        self.episode_batch = {}
        self.episode_counter = 0
        self.last_episode_log = 0

    def log_train(self, train_log):
        self.train_steps += 1
        self.train_logs.append(train_log)

        if self.train_steps % self.params.loss_period == 0:
            logs = dict_mean(self.train_logs)
            with self.log_writer.as_default():
                for name, value in logs.items():
                    self.log_scalar(f'training/{name}', value, self.train_steps)

    def log_ftrain(self, train_log):
        self.ftrain_steps += 1
        self.ftrain_logs.append(train_log)

        if self.ftrain_steps % self.params.floss_period == 0:
            logs = dict_mean(self.ftrain_logs)
            with self.log_writer.as_default():
                for name, value in logs.items():
                    self.log_scalar(f'ftraining/{name}', value, self.ftrain_steps)

    def log_step(self, step_info=None):
        self.steps += 1

        if self.steps % self.params.evaluation_period == 0 and self.steps >= self.params.evaluation_start:
            if self.evaluator is not None:
                if self.params.evaluation_episodes == 1:
                    info = self.evaluator.evaluate_episode()
                else:
                    infos = [self.evaluator.evaluate_episode() for _ in range(self.params.evaluation_episodes)]
                    info = dict_mean(infos)
                with self.log_writer.as_default():
                    for name, value in info.items():
                        self.log_scalar(f'evaluation/{name}', value, self.steps)

        if self.steps % self.params.save_weights == 0:
            self.agent.save_weights(f"{self.log_dir}/models/")

    def log_fstep(self, step_info):

        if self.fsteps % self.params.evaluation_period == 0 and self.fsteps >= self.params.evaluation_start:
            if self.evaluator is not None:
                info = self.evaluator.fevaluate()
                with self.log_writer.as_default():
                    for name, value in info.items():
                        self.log_scalar(f'fevaluation/{name}', value, self.fsteps)

        if self.fsteps % self.params.save_weights == 0:
            self.agent.save_weights(f"{self.log_dir}/models/")

        self.fsteps += 1

    def log_batched_episode_tf(self, reset, info):
        if self.episode_batch == {}:
            self.episode_batch = {key: tf.zeros_like(value[0], tf.float32) for key, value in info.items()}
            self.episode_counter = tf.constant(0, dtype=tf.int64)

        if tf.reduce_any(reset):
            for key, value in self.episode_batch.items():
                self.episode_batch[key] += tf.reduce_sum(tf.where(reset[:, 0], tf.cast(info[key], tf.float32), 0), 0)
            self.episode_counter += tf.math.count_nonzero(reset)
        if self.steps % self.params.batched_episode_period < tf.shape(reset)[0]:
            for key, value in self.episode_batch.items():
                self.episode_batch[key] /= tf.cast(self.episode_counter, tf.float32)
            self._log_episodic(self.episode_batch)
            self.episode_batch = {}
            self.episode_counter = tf.constant(0, dtype=tf.int64)


    def log_episode(self, info):
        if not self.params.log_episodes:
            return

        if self.episode_batch == {}:
            self.episode_batch = info.copy()
            self.episode_counter = 1
        else:
            for key, value in info.items():
                self.episode_batch[key] += value
            self.episode_counter += 1

        if self.steps - self.last_episode_log > self.params.batched_episode_period:
            batched_info = {key: value / self.episode_counter for key, value in self.episode_batch.items()}
            self._log_episodic(batched_info)
            self.episode_batch = {}
            self.episode_counter = 0
            self.last_episode_log += self.params.batched_episode_period


    def _log_episodic(self, info):
        with self.log_writer.as_default():
            for name, value in info.items():
                self.log_scalar(f'episodic/{name}', value, self.steps)
            self.log_writer.flush()

    def log_rollout(self, infos):
        with self.log_writer.as_default():
            for name, value in infos.items():
                self.log_scalar(f'rollout/{name}', value, self.steps)

    def log_scalar(self, name, value, step):
        if isinstance(value, str):
            return
        tf.summary.scalar(name, value, step)

    def log_params(self, params):
        # Create formatted json string from params nested dict
        js = params.model_dump()
        params_str = json.dumps(js, indent=4)
        pretty_str = "".join("\t" + line for line in params_str.splitlines(True))
        with self.log_writer.as_default():
            tf.summary.text("params", pretty_str, step=0)

