from dataclasses import field, dataclass
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import tensorflow as tf

from drl_am.agent.amsac import AMSACAgent
from drl_am.trainer.feasibility import FeasibilityTrainer
from drl_am.trainer.worker_tasks import step_process, reset_process
from drl_am.utils.memory import ReplayMemory
from drl_am.utils.misc import get_decay_curve, DecayParams, dict_slice_set, dict_to_tensor


class AMSACTrainer(FeasibilityTrainer):
    @dataclass
    class Params:
        # Learning Rate
        batch_size: int = 128
        actor_lr: DecayParams = field(default_factory=lambda: DecayParams("exp", 3e-5, 0.1, 1_000_000))
        critic_lr: DecayParams = field(default_factory=lambda: DecayParams("exp", 1e-4, 0.1, 1_000_000))

        memory: ReplayMemory.Params = field(default_factory=ReplayMemory.Params)
        training_steps: int = 2000000

        gamma: float = 0.97
        tau: float = 0.005

        policy_update_delay: int = 2048
        train_per_step: int = 1

        num_gyms: int = 1
        multi_proc_gym: bool = True
        entropy: float = 0.0002
        am: FeasibilityTrainer.Params = field(default_factory=FeasibilityTrainer.Params)

    def __init__(self, params: Params, gym, agent: AMSACAgent, logger):
        FeasibilityTrainer.__init__(self, params.am, gym, agent, logger)
        self.params = params
        self.am_params = params.am
        self.agent = agent
        self.params = params

        self.memory = ReplayMemory(self.params.memory)
        self.agent = agent
        self.gym = gym
        self.logger = logger

        actor_lr = get_decay_curve(self.params.actor_lr)
        critic_lr = get_decay_curve(self.params.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_steps = tf.Variable(0, dtype=tf.int32)

        self.gym_processes_started = False

    def __del__(self):
        if self.gym_processes_started:
            self.stop_reset_event.set()
            for step_queue in self.step_queues:
                step_queue.put(("stop", None))
            self.reset_process.join()
            for step_proc in self.step_processes:
                step_proc.join()

    def train(self, skip_feas=False):
        if not skip_feas:
            print("Training Feasibility and then Objective")
            print("Training Feasibility")
            self.ftrain()
        print("Training Objective")
        self.otrain()

    def start_gym_processes(self):
        self.reset_queue = mp.JoinableQueue(maxsize=100)
        self.stop_reset_event = mp.Event()
        self.reset_process = mp.Process(target=reset_process,
                                        args=(self.gym, self.reset_queue, self.stop_reset_event))
        self.reset_process.start()
        self.step_queues = [mp.JoinableQueue(2) for _ in range(self.params.num_gyms)]
        self.step_result_queues = [mp.JoinableQueue(1) for _ in range(self.params.num_gyms)]
        self.step_processes = [
            mp.Process(target=step_process, args=(self.gym, self.step_queues[i], self.step_result_queues[i])) for i
            in
            range(self.params.num_gyms)
        ]
        for step_proc in self.step_processes:
            step_proc.start()

        self.gym_processes_started = True

    def reset_gym(self, env_id, return_state=False):
        if not self.reset_queue.empty():
            state, obs = self.reset_queue.get()
        else:
            state, obs = self.gym.reset()
        self.step_queues[env_id].put(("reset", state))
        if return_state:
            return state, obs
        return obs

    def otrain(self):
        if not self.gym_processes_started:
            self.start_gym_processes()
        else:
            raise ValueError("Processes already started. Behavior would be undefined.")
        observes = [self.reset_gym(k) for k in range(self.params.num_gyms)]
        obs = {key: np.concatenate([o[key] for o in observes], axis=0) for key in observes[0].keys()}

        progress = tqdm(range(self.params.training_steps))
        num_steps = self.params.training_steps // self.params.num_gyms

        for step in range(num_steps):
            obs, reset, infos = self.collect_step_multi(obs)
            for i in range(self.params.num_gyms):
                self.logger.log_step(infos[i])
                if reset[i]:
                    self.logger.log_episode(infos[i])
                    o = self.reset_gym(i)
                    dict_slice_set(obs, i, o)

            progress.update(self.params.num_gyms)
            if step * self.params.num_gyms >= self.params.batch_size:
                for _ in range(self.params.train_per_step):
                    self.train_step()

    def train_step(self):
        sample = self.memory.sample(self.params.batch_size)
        logs = self._train_step_tf(dict_to_tensor(sample))
        self.logger.log_train(logs)

    def step_gym_multi(self, actions):
        for i, action in enumerate(actions):
            self.step_queues[i].put(("step", action))
        results = [self.step_result_queues[i].get() for i in range(self.params.num_gyms)]
        obs, rewards, dones, truncated, info = zip(*results)

        obs = {
            key: np.concatenate([o[key] for o in obs], axis=0) for key in obs[0].keys()
        }
        return obs, np.array(rewards), np.array(dones), np.array(truncated), info

    def collect_step_multi(self, obs):
        actions, latents = self.get_action(obs)
        next_obs, rewards, terminated, truncated, infos = self.step_gym_multi(actions)
        failed = np.array([info["failed"] for info in infos])
        self.memory.store_transitions(obs, latents, rewards, next_obs, failed, terminated)
        reset = terminated | truncated
        return next_obs, reset, infos

    def get_action(self, obs):
        use_rand_action = self.critic_steps < self.params.policy_update_delay
        if use_rand_action:
            return self.agent.get_feasible_action(obs)
        return self.agent.get_action(obs)

    @tf.function
    def get_action_tf(self, obs, exploit=False):
        action, _ = self.get_action_latent_tf(obs, exploit)
        return action

    @tf.function
    def get_action_latent_tf(self, obs, exploit=False):
        if self.critic_steps > self.params.policy_update_delay:
            action, latent = self.agent.get_action_latent_tf(obs, exploit)
        else:
            action, latent = self.agent.get_random_feasible_actions_tf(obs, num_actions=1)
            action = tf.squeeze(action, axis=1)
            latent = tf.squeeze(latent, axis=1)

        return action, latent

    @tf.function
    def _train_step_tf(self, sample):
        obs = sample["observation"]
        next_obs = sample["next_observation"]
        action = sample["action"]
        reward = sample["reward"]
        done = tf.cast(sample["done"], tf.float32)

        # Compute target values
        critic_input, next_log_prob = self.agent.get_actor_log_prob_tf(next_obs)

        next_q_values = self.agent.get_q_values_tf(next_obs, critic_input, self.agent.critic_target)

        next_q_val = tf.reduce_min(next_q_values, axis=1, keepdims=True)  # [B, 1]
        next_q_val = next_q_val - self.params.entropy * next_log_prob
        gamma = self.params.gamma
        q_target = reward + gamma * (1.0 - done) * next_q_val  # [B, 1]

        with tf.GradientTape() as tape:
            q_values = self.agent.get_q_values_tf(obs, action, self.agent.critic)  # [B, num_critic]
            critic_loss = tf.reduce_mean((q_values - q_target) ** 2)
        gradients = tape.gradient(critic_loss, self.agent.critic_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.agent.critic_variables))

        self.agent.soft_update_critic(self.params.tau)
        self.critic_steps.assign_add(1)

        actor_loss = 0.0
        log_prob = 0.0
        if self.critic_steps > self.params.policy_update_delay:
            with tf.GradientTape() as tape:
                critic_input, log_prob = self.agent.get_actor_log_prob_tf(obs)
                q_value = self.agent.get_q_values_tf(obs, critic_input, self.agent.critic)
                min_q = tf.reduce_min(q_value, axis=-1, keepdims=True)
                actor_loss = tf.reduce_mean(self.params.entropy * log_prob - min_q)

            gradients = tape.gradient(actor_loss, self.agent.actor_variables)
            self.actor_optimizer.apply_gradients(zip(gradients, self.agent.actor_variables))

        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "predicted_q": tf.reduce_mean(q_values),
                "target_q": tf.reduce_mean(q_target), "log_prob": tf.reduce_mean(log_prob), "gamma": gamma}
