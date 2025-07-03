from typing import Union, NamedTuple
from dataclasses import dataclass
import tensorflow as tf
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p

"""
state:
    - n,4: batch, n spheres with xyzr; if r<0 then masked out (does not exist)
    - 7,2: current joint angles + velocities
action: m,7,1: joint angle offsets to be applied to the current joint angles

g( s[[b,n,4], [b,7,2]], a[b,m,7,1] ) -> bool[b,n,m]:

robot params:
https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
"""


class State(NamedTuple):
    joint_states: tf.Tensor
    obstacles: tf.Tensor
    target: tf.Tensor
    ee_pose: tf.Tensor = None
    is_collision: tf.Tensor = None
    is_inside_joint_limits: tf.Tensor = None
    is_in_speed_limits: tf.Tensor = None
    failed: tf.Tensor = None
    collision_obstacle: tf.Tensor = None
    cumulative_reward: tf.Tensor = None
    steps: tf.Tensor = None


class RobotArmGym:
    @dataclass
    class Params:
        timeout_steps: int = 100
        delta_t: float = 0.5
        angle_target: bool = True

        # --- observation configuration ---
        observe_ee_pose: bool = False

        # --- obstacles configuration ---
        use_obstacles: bool = True
        max_n_obstacles: int = 20
        max_n_obstacles_sample: int = 30
        min_r_obstacle: float = 0.2
        max_r_obstacle: float = 0.5

        # --- reward configuration ---
        reward_on_progress: bool = True
        reward_dist: float = 1.0
        reward_rot: float = 0.1
        penalty_action: float = 0.0
        penalty_failed: float = 0.0

        # --- cost configuration ---
        cost_mult_penetration: float = 1.0
        cost_mult_joint_limits: float = 1.0
        cost_mult_speed_limits: float = 1.0
        cost_mult_action_bounds: float = 1.0

        # --- robot configuration ---
        # Franka Panda parameters
        cartesian_linear_limit: float = 0.3
        cartesian_rot_limit: float = 2.5

        l0: float = 0.333
        l2: float = 0.316
        l4: float = 0.384
        a4: float = 0.0825
        l6: float = 0.088
        l7: float = 0.107

        max_joint_action: float = np.deg2rad(90.0)
        joint_limits: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] = (
            (-2.7437, 2.7437),  # 0
            (-1.7837, 1.7837),  # 1
            (-2.9007, 2.9007),  # 2
            (-3.0421, -0.1518),  # 3
            (-2.8065, 2.8065),  # 4
            (0.5445, 4.5169),  # 5
            (-3.0159, 3.0159),  # 6
        )

    def __init__(self, params: Params = Params(), physicsClientId: int = 0):
        self.params = params
        self.physicsClientId = physicsClientId
        self._debug_ids = {}

        self.observation_space = {
            "internal_state": (7 + 12,) if self.params.observe_ee_pose else (7,),
            "goal": (12,) if self.params.angle_target else (3,),
        }
        if self.params.use_obstacles:
            self.observation_space.update({"mult_constraints": (None, 4), "num_constraints": (1,)})
        self.action_dim = 7

        self.cartesian_space_extent: float = params.l2 + params.l4 + params.l7

        self.joint_limits: tf.Tensor = tf.constant(params.joint_limits, dtype=tf.float32)
        self.DHs: tf.Tensor = tf.constant(
            [
                [0.0, params.l0, 0.0],
                [0.0, 0.0, -np.pi / 2],
                [0.0, params.l2, np.pi / 2],
                [params.a4, 0.0, np.pi / 2],
                [-params.a4, params.l4, -np.pi / 2],
                [0.0, 0.0, np.pi / 2],
                [params.l6, 0.0, np.pi / 2],
                [0.0, params.l7, 0.0],  # flange DH with constant theta
            ],
            dtype=tf.float32,
        )

        self.collision_capsules_A: tf.Tensor = tf.constant(
            [
                [0.0, 0.0, 0.0],  # 0
                [0.0, 0.0, 0.0],  # 1
                [0.0, 0.0, 0.0],  # 2
                [0.0, 0.0, 0.0],  # 3
                [-params.a4, 0.0, 0.0],  # 4
                [0.0, 0.0, 0.0],  # 5
                [0.0, 0.0, 0.0],  # 6
                [0.0, 0.0, 0.0],  # 7
                [0.0, 0.0, 0.0],  # F
            ],
            dtype=tf.float32,
        )
        self.collision_capsules_B: tf.Tensor = tf.constant(
            [
                [0.0, 0.0, params.l0],  # 0
                [0.0, 0.0, 0.0],  # 1
                [0.0, -params.l2, 0.0],  # 2
                [params.a4, 0.0, 0.0],  # 3
                [-params.a4, params.l4, 0.0],  # 4
                [0.0, 0.0, 0.0],  # 5
                [params.l6, 0.0, 0.0],  # 6
                [0.0, 0.0, params.l7],  # 7
                [0.0, 0.0, 0.0],  # F
            ],
            dtype=tf.float32,
        )

        self.collision_capsules_R: tf.Tensor = tf.constant(
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=tf.float32
        )

    def get_obs_action_feasible_dataset(self, batch_size: int):
        def gen():
            N_ACTIONS = 512

            # the smaller the lower the acceptance ratio, but faster
            # 256: 94-95% feasible
            # 512: 98.5% feasible

            @tf.function
            def generate_action():
                obs = self.generate_states_tf(num_states=batch_size)
                rand_action = tf.random.uniform(
                    (batch_size, N_ACTIONS, self.action_dim), -1.0, 1.0
                )  # [B, 1024, 7]
                y = self.action_feasibility_tf(obs, rand_action)  # [B, 1024]

                # if no feasible action -> any non-feasible action is chosen
                inds = tf.argsort(
                    tf.cast(y, tf.float32), axis=1, direction="DESCENDING"
                )  # [B, 1024]
                inds = inds[:, 0:1]

                is_feas = tf.gather(y, inds, axis=1, batch_dims=1)

                feasible = tf.gather(rand_action, inds, axis=1, batch_dims=1)
                feasible.set_shape((batch_size, 1, self.action_dim))

                noisy_feasible = feasible + tf.random.normal(feasible.shape, stddev=0.11)

                is_feas = self.action_feasibility_tf(obs, noisy_feasible)

                return obs, noisy_feasible[:, 0], is_feas[:, 0]  # remove action dim

            while True:
                yield generate_action()

        output_obs = {
            "internal_state": tf.TensorSpec(
                shape=(None, *self.observation_space["internal_state"]), dtype=tf.float32
            ),
            "goal": tf.TensorSpec(shape=(None, *self.observation_space["goal"]), dtype=tf.float32),
        }

        if self.params.use_obstacles:
            output_obs.update(
                {
                    "mult_constraints": tf.TensorSpec(
                        shape=(None, *self.observation_space["mult_constraints"]), dtype=tf.float32
                    ),
                    "num_constraints": tf.TensorSpec(
                        shape=(None, *self.observation_space["num_constraints"]), dtype=tf.int32
                    ),
                }
            )

        output_spec = (
            output_obs,
            tf.TensorSpec(shape=(None, 7), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )

        return tf.data.Dataset.from_generator(gen, output_signature=output_spec)

    @tf.function(reduce_retracing=True, jit_compile=None)
    def reset_tf(self, batch_size: int = 1) -> State:
        """generates a random start state, target and obstacles"""

        rng = tf.random.get_global_generator()

        joint_states = rng.uniform(
            (batch_size, 7, 1),
            self.joint_limits[:, 0, None],
            self.joint_limits[:, 1, None],
        )

        # generate random target
        joint_states_target = rng.uniform(
            (batch_size, 7, 1),
            self.joint_limits[:, 0, None],
            self.joint_limits[:, 1, None],
        )
        RTs = self._get_RTs(joint_states[:, tf.newaxis, ...])
        target_RTs = self._get_RTs(joint_states_target[:, tf.newaxis, ...])

        ee_pose = RTs[:, 0, -1, :, :]
        target = target_RTs[:, 0, -1, :, :]  # [b,4,4]

        if self.params.use_obstacles:

            # generate random obstacles
            r = rng.uniform(
                (batch_size, self.params.max_n_obstacles_sample),
                self.params.min_r_obstacle,
                self.params.max_r_obstacle,
            )
            centers = rng.uniform(
                (batch_size, self.params.max_n_obstacles_sample, 3),
                -self.cartesian_space_extent,
                self.cartesian_space_extent,
            )
            spheres = tf.concat([centers, r[..., tf.newaxis]], axis=-1)  # [b,n,4]

            # calculate collisions with sampled spheres
            start_caps = self._get_robot_capsules(RTs)  # [...,9,7]
            collision_start = RobotArmGym._sphere_capsule_collision(
                spheres[:, tf.newaxis], start_caps
            )
            collision_start = tf.reduce_any(collision_start, axis=[-1])  # [b,n]

            target_caps = self._get_robot_capsules(target_RTs)  # [...,9,7]
            collision_target = RobotArmGym._sphere_capsule_collision(
                spheres[:, tf.newaxis], target_caps
            )
            collision_target = tf.reduce_any(collision_target, axis=[-1])  # [b,n]

            # mask out obstacles that collide with start or target
            is_collision = tf.logical_or(collision_start, collision_target)
            is_collision = tf.squeeze(is_collision, axis=1)
            spheres = tf.where(is_collision[..., tf.newaxis], -1.0, spheres)  # [b,n,4]

            sorted_inds = tf.argsort(spheres[..., -1], direction="DESCENDING", axis=-1)
            spheres = tf.gather(spheres, sorted_inds, axis=-2, batch_dims=1)

            # truncate to max_n_obstacles
            spheres = spheres[:, : self.params.max_n_obstacles, :]  # [b,n,4]
        else:
            spheres = tf.zeros((batch_size, 1, 4), dtype=tf.float32)

        return State(
            joint_states=joint_states,
            obstacles=spheres,
            target=target,
            ee_pose=ee_pose,
            is_collision=tf.zeros((batch_size,), dtype=tf.bool),
            is_inside_joint_limits=tf.ones((batch_size,), dtype=tf.bool),
            is_in_speed_limits=tf.ones((batch_size,), dtype=tf.bool),
            failed=tf.zeros((batch_size, 1), dtype=tf.bool),
            collision_obstacle=tf.zeros((batch_size, self.params.max_n_obstacles), dtype=tf.bool),
            cumulative_reward=tf.zeros((batch_size,), dtype=tf.float32),
            steps=tf.zeros((batch_size,), dtype=tf.int32),
        )

    @tf.function
    def reset_state(self, state: State, mask: tf.Tensor) -> State:
        """resets the states where the mask is True"""

        n_resets = tf.math.count_nonzero(mask, dtype=tf.int32)
        if n_resets == 0:
            return state

        batch_size = tf.shape(state.joint_states)[0]

        reset_state: State = self.reset_tf(n_resets)
        if n_resets == batch_size:
            return reset_state

        where = tf.where(tf.squeeze(mask, -1))
        return State(
            *tuple((tf.tensor_scatter_nd_update(s, where, r) for s, r in zip(state, reset_state)))
        )

    def reset(self, batch_size: int = 1) -> State:
        state: State = self.reset_tf(batch_size)

        if "sliders" in self._debug_ids:
            self._refresh_sliders(state.joint_states)

        return state

    @tf.function
    def generate_states_tf(self, num_states: int):
        states = self.reset(num_states)
        obs = self.get_obs(states)
        return obs

    @tf.function
    def get_feasible_action_tf(self, obs):
        batch_size = tf.shape(obs["internal_state"])[0]
        action = tf.zeros((batch_size, self.action_dim), dtype=tf.float32)
        return action

    @tf.function(jit_compile=None)
    def step_tf(self, state: State, action: tf.Tensor) -> tuple:
        """
        Performs a step in the environment

        Args:
            state:
                joint_states [b,7,1]: The current state of the robot.
                obstacles [b,n,4]: The obstacles in the environment (xyzr).
                target [b,4,4]: The target pose of the robot.

            action [b,7]: The action to be applied to the robot. [-1,1]

        Returns:
            next_state: State: The new state of the robot, as by get_obs
            reward [b,]: The reward for the action.
            done [b,]: True if the episode is done.
            info: Additional information about the step.
        """

        scaled_action = self.scale_action_to_environment(action)

        joint_states = state.joint_states[:, tf.newaxis, :, :]  # add action dim
        scaled_action = scaled_action[:, tf.newaxis, :, :]  # add batch dim

        prev_RTs = self._get_RTs(joint_states)

        # apply the action to the joint states
        new_joint_states = joint_states + scaled_action

        RTs = self._get_RTs(new_joint_states)  # [...,9,4,4]
        capsules = self._get_robot_capsules(RTs)  # [...,9,7]

        # check if the new joint states are within the joint limits
        is_inside_joint_limits = self._is_in_joint_limits(new_joint_states)  # [...,]
        is_inside_joint_limits = tf.squeeze(is_inside_joint_limits, axis=1)

        if self.params.use_obstacles:
            # check for obstacle collisions
            spheres = state.obstacles[:, tf.newaxis, ...]  # [b,1,n,4]
            collision_obstacle = RobotArmGym._sphere_capsule_collision(
                spheres, capsules
            )  # [..., n, 9]
            is_collision = tf.reduce_any(collision_obstacle, axis=[-1])  # [..., n]
            collision_obstacle = tf.squeeze(is_collision, axis=1)
            is_collision = tf.reduce_any(collision_obstacle, axis=-1)  # any sphere collision
        else:
            collision_obstacle = state.collision_obstacle
            is_collision = tf.zeros_like(is_inside_joint_limits)

        # check cartesian speed limit
        is_in_speed_limits = self._is_in_speed_limits(prev_RTs, RTs)

        ee_pose = RTs[..., -1, :, :]

        # remove action dim
        is_in_speed_limits = tf.squeeze(is_in_speed_limits, axis=1)
        new_joint_states = tf.squeeze(new_joint_states, axis=1)
        ee_pose = tf.squeeze(ee_pose, axis=1)

        failed = is_collision | ~is_in_speed_limits | ~is_inside_joint_limits

        prev_dist, prev_angle = self.difference_poses(state.ee_pose, state.target)
        dist, angle = self.difference_poses(ee_pose, state.target)

        rot_mul = self.params.reward_rot if self.params.angle_target else 0.0
        if self.params.reward_on_progress:
            reward = self.params.reward_dist * (prev_dist - dist) + rot_mul * (prev_angle - angle)
        else:
            reward = tf.math.exp(-self.params.reward_dist * dist - rot_mul * angle)
        neg_only_reward = tf.minimum(0.0, reward)
        reward = tf.where(failed, neg_only_reward, reward)
        reward = reward - self.params.penalty_action * tf.reduce_sum(tf.abs(action), axis=-1)
        reward -= tf.cast(failed, tf.float32) * self.params.penalty_failed
        failed = failed[:, None]

        next_state = State(
            joint_states=new_joint_states,
            obstacles=state.obstacles,
            target=state.target,
            ee_pose=ee_pose,
            collision_obstacle=collision_obstacle,
            is_collision=is_collision,
            is_inside_joint_limits=is_inside_joint_limits,
            is_in_speed_limits=is_in_speed_limits,
            failed=failed,
            cumulative_reward=state.cumulative_reward + reward,
            steps=state.steps + 1,
        )
        done = next_state.failed

        return (
            next_state,
            reward[..., None],
            done,
            (next_state.steps > self.params.timeout_steps)[..., None],
            self.get_info(next_state),
        )

    def step(self, state: State, action: tf.Tensor) -> tuple:
        return self.step_tf(state, action)

    @tf.function(jit_compile=None)
    def action_feasibility(
            self, state: State, action, return_info=False
    ) -> Union[tf.Tensor, tuple[tf.Tensor, dict]]:
        """
        Determines if actions are feasible based on a state [batched]

        Args:
            state:
                joint_states [b,7,1]: The current state of the robot.
                obstacles [b,n,4]: The obstacles in the environment (xyzr).
            action [b,m,7]: The actions to be checked for feasibility. [-1,1]

        Returns:
            bool tensor [b,m]: True if the action is feasible, False otherwise.
        """
        action_in_bounds = tf.reduce_all(tf.abs(action) <= 1, axis=2)
        scaled_action = self.scale_action_to_environment(action)

        joint_states = state.joint_states[:, tf.newaxis, :, :]  # add action dim

        prev_RTs = self._get_RTs(joint_states)

        # apply the action to the joint states
        new_joint_states = joint_states + scaled_action

        RTs = self._get_RTs(new_joint_states)  # [...,9,4,4]
        capsules = self._get_robot_capsules(RTs)  # [...,9,7]

        # check if the new joint states are within the joint limits
        is_inside_limits = self._is_in_joint_limits(new_joint_states)  # [...,]

        if self.params.use_obstacles:

            # check for obstacle collisions
            spheres = state.obstacles[:, tf.newaxis, ...]  # [b,1,n,4]
            collision_obstacle = RobotArmGym._sphere_capsule_collision(
                spheres, capsules
            )  # [..., n, 9]
            is_collision = tf.reduce_any(collision_obstacle, axis=[-1])  # [..., n]

            is_collision = tf.reduce_any(is_collision, axis=-1)  # any sphere collision
        else:
            is_collision = tf.zeros_like(is_inside_limits)

        # check the cartesian speed limits
        is_in_speed_limits = self._is_in_speed_limits(prev_RTs, RTs)

        # res = self._step(state, scaled_action)

        if return_info:
            info = {
                "violate_collision": is_collision,
                "violate_joint_limits": ~is_inside_limits,
                "violate_cart_speed_lim": ~is_in_speed_limits,
            }
            return ~is_collision & is_inside_limits & is_in_speed_limits & action_in_bounds, info

        return ~is_collision & is_inside_limits & is_in_speed_limits & action_in_bounds

    @tf.function
    def action_cost_tf(self, obs: dict, actions) -> tf.Tensor:
        state = State(
            joint_states=obs["internal_state"][..., :7, None],
            obstacles=obs["mult_constraints"],
            target=None
        )
        return self.action_cost(state, actions)

    @tf.function
    def action_cost(self, state: State, action) -> tf.Tensor:
        """
        Determines if actions are feasible based on a state [batched]

        Args:
            state:
                joint_states [b,7,1]: The current state of the robot.
                obstacles [b,n,4]: The obstacles in the environment (xyzr).
            action [b,m,7]: The actions to be checked for feasibility. [-1,1]

        Returns:
            float32 tensor [b,m]: continuous and differentiable action cost
        """
        # TODO to be tested and verified
        action_bound_cost = tf.maximum(tf.reduce_sum(tf.abs(action) - 1, axis=-1), 0.0)  # [b,m]

        scaled_action = self.scale_action_to_environment(action)

        joint_states = state.joint_states[:, tf.newaxis, :, :]  # add action dim

        prev_RTs = self._get_RTs(joint_states)

        # apply the action to the joint states
        new_joint_states = joint_states + scaled_action

        RTs = self._get_RTs(new_joint_states)  # [...,9,4,4]
        capsules = self._get_robot_capsules(RTs)  # [...,9,7]

        # joint limits
        upper_cost = new_joint_states - self.joint_limits[None, None, :, 1, None]  # [b,m,7,1]
        lower_cost = self.joint_limits[None, None, :, 0, None] - new_joint_states
        upper_cost = tf.maximum(upper_cost, 0.0)
        lower_cost = tf.maximum(lower_cost, 0.0)
        upper_cost = tf.reduce_sum(upper_cost, axis=[-2, -1])  # sum over robot joints
        lower_cost = tf.reduce_sum(lower_cost, axis=[-2, -1])
        joint_limit_cost = upper_cost + lower_cost  # [b,m]

        if self.params.use_obstacles:

            # broadcast for action dim m
            spheres = state.obstacles[:, tf.newaxis, ...]  # [b,1,n,4]
            radii = spheres[..., tf.newaxis, 3]  # [b,1,n,1]
            #  [b, n, m] m==robot joints
            dist = RobotArmGym._dist_to_capsule(spheres[..., :3], capsules)  # [b, m, n, 9]
            penetration_cost = radii - dist  # [b, n, 9]
            # if negative -> no collision
            penetration_cost = tf.maximum(penetration_cost, 0.0)  # [b,m,n,9]
            # [b, m] max penetration for each robot joint
            # penetration_cost = tf.reduce_max(penetration_cost, axis=-2)  # [b,m,9]
            # sum over robot joints and obstacles
            penetration_cost = tf.reduce_sum(penetration_cost, axis=[-2, -1])  # [b,m]

        else:
            penetration_cost = tf.zeros_like(joint_limit_cost)

        # check the cartesian speed limits
        delta_t = self.params.delta_t
        limit = self.params.cartesian_linear_limit
        prev_pos = prev_RTs[..., :3, 3]  # [b,m,9,3]
        pos = RTs[..., :3, 3]  # [b,m,9,3]
        dpos = pos - prev_pos  # [b,m,9,3]
        speeds = tf.linalg.norm(dpos, axis=-1) / delta_t  # [b,m,9]
        # sum over robot joints
        speed_cost = tf.reduce_sum(tf.maximum(speeds - limit, 0.0), axis=-1)  # [b,m]

        cost = self.params.cost_mult_penetration * penetration_cost \
               + self.params.cost_mult_joint_limits * joint_limit_cost \
               + self.params.cost_mult_speed_limits * speed_cost \
               + self.params.cost_mult_action_bounds * action_bound_cost

        return cost

    @tf.function
    def action_feasibility_tf(
            self, obs, actions, return_info=False
    ) -> Union[tf.Tensor, tuple[tf.Tensor, dict]]:
        if self.params.angle_target:
            target = tf.reshape(obs["goal"], (-1, 3, 4))
        else:
            target = tf.concat(
                (tf.eye(3, batch_shape=[tf.shape(obs["goal"])[0]]), obs["goal"][:, :, None]),
                axis=-1,
            )
        line = tf.reshape(tf.constant([0, 0, 0, 1], dtype=tf.float32), (1, 1, 4))
        line = tf.broadcast_to(line, (tf.shape(actions)[0], 1, 4))
        target = tf.concat([target, line], axis=1)
        state = State(
            joint_states=obs["internal_state"][..., :7, None],
            obstacles=obs["mult_constraints"],
            target=target,
        )
        return self.action_feasibility(state, actions, return_info=return_info)

    @tf.function
    def get_obs(self, state: State):
        internal_state = tf.squeeze(state.joint_states, axis=2)
        if self.params.observe_ee_pose:
            ee_pose = tf.reshape(state.ee_pose[:, :3], (-1, 12))
            internal_state = tf.concat([internal_state, ee_pose], axis=-1)
        obs = {
            "internal_state": internal_state,
            "goal": (
                tf.reshape(state.target[:, :3], (-1, 12))
                if self.params.angle_target
                else state.target[:, :3, 3]
            ),
        }
        if self.params.use_obstacles:
            obs.update(
                {
                    "mult_constraints": state.obstacles,
                    "num_constraints": tf.math.count_nonzero(
                        state.obstacles[..., -1] > 0, axis=-1, keepdims=True
                    ),
                }
            )
        return obs

    @tf.function
    def scale_action_to_environment(self, action: tf.Tensor):
        """scales action (defined as [-1, 1]) to a joint action for the robot and adds dim"""
        return action[..., None] * self.params.max_joint_action * self.params.delta_t

    @tf.function
    def scale_action_to_agent(self, action: tf.Tensor):
        """scales joint action for the robot to [-1, 1]"""
        return action / self.params.max_joint_action / self.params.delta_t

    def get_info(self, state: State):
        dist, angle = self.difference_poses(state.ee_pose, state.target)
        return {
            "cumulative_reward": state.cumulative_reward,
            "steps": state.steps,
            "failed": state.failed[:, 0],
            "distance": dist,
            "angle": angle,
            "weighted_distance": self.params.reward_dist * dist + self.params.reward_rot * angle,
        }

    @tf.function
    def _is_in_joint_limits(self, joint_states):
        feasible_joint_limits = tf.logical_and(
            joint_states >= self.joint_limits[None, None, :, 0, None],
            joint_states <= self.joint_limits[None, None, :, 1, None],
        )  # [...,7,1]

        return tf.reduce_all(feasible_joint_limits, axis=[-2, -1])  # [...,]

    @tf.function
    def _is_in_speed_limits(self, prev_RTs, RTs):
        """check¿ if all joints are within the speed limits

        Args:
            prev_RTs [...,9,4,4]: The previous transformation matrices for each of the links
            RTs [...,9,4,4]: The transformation matrices for each of the links

        Returns:
            bool [...,]: True if all joints are within the speed limits
        """
        delta_t = self.params.delta_t
        limit = self.params.cartesian_linear_limit

        prev_pos = prev_RTs[..., :3, 3]  # [..., 9, 3]
        pos = RTs[..., :3, 3]  # [..., 9, 3]
        dpos = pos - prev_pos  # [..., 9, 3]
        speeds = tf.linalg.norm(dpos, axis=-1) / delta_t  # [..., 9]
        return tf.reduce_all(speeds < limit, axis=-1)

    @tf.function(jit_compile=None)
    def _get_RTs(self, joint_states: tf.Tensor) -> tf.Tensor:
        """
        Return the 4x4 transformation matrix for each of the links of the robot.

        Args:
            joint_states [b,n, 7,1]: The joint states of the robot.

        Returns:
            RTs [b,n,9,4,4]: The transformation matrices for each of the links
        """
        dhs = self._get_DHs(joint_states)
        return self._get_RT_from_DHs(dhs)

    @tf.function
    def _get_DHs(self, joint_states: tf.Tensor) -> tf.Tensor:
        """
        Return the modified Denavit-Hartenberg parameters for the given joint states.

        Args:
            joint_states [b,n, 7,1]: The joint states of the robot.

        Returns:
            dh [b,n, 8,4]: The modified Denavit-Hartenberg parameters for the given joint states.
        """

        b, n = tf.shape(joint_states)[0], tf.shape(joint_states)[1]

        constants = tf.broadcast_to(self.DHs, (b, n, 8, 3))

        # add a 0 for the last flange joint
        joint_states = tf.concat((joint_states, tf.zeros((b, n, 1, 1))), axis=-2)

        return tf.concat((constants, joint_states), axis=-1)  # [b,n,8,4]

    @staticmethod
    @tf.function
    def _get_RT_from_DHs(DHs: tf.Tensor) -> tf.Tensor:
        """
        Return the 4x4 transformation matrix from the modified Denavit-Hartenberg parameters.

        Args:
            DHs [b,n,9,4]: The modified Denavit-Hartenberg parameters.

        Returns:
            T [b,n,9,4,4]: The transformation matrices for each of the links
        """
        b, n = tf.shape(DHs)[0], tf.shape(DHs)[1]
        a, d, alpha, theta = tf.unstack(DHs, axis=-1)
        # [b, 8]

        # [b,8,4,1]
        first_rows = tf.stack([tf.cos(theta), -tf.sin(theta), tf.zeros_like(a), a], axis=-1)
        second_rows = tf.stack(
            [
                tf.sin(theta) * tf.cos(alpha),
                tf.cos(theta) * tf.cos(alpha),
                -tf.sin(alpha),
                -d * tf.sin(alpha),
            ],
            axis=-1,
        )
        third_rows = tf.stack(
            [
                tf.sin(theta) * tf.sin(alpha),
                tf.cos(theta) * tf.sin(alpha),
                tf.cos(alpha),
                d * tf.cos(alpha),
            ],
            axis=-1,
        )
        fourth_rows = tf.stack(
            [tf.zeros_like(a), tf.zeros_like(a), tf.zeros_like(a), tf.ones_like(a)], axis=-1
        )
        rel_RTs = tf.stack([first_rows, second_rows, third_rows, fourth_rows], axis=-2)
        # [b,8,4,4]

        # this looks so stupid (cumulative matrix multiplication)
        return tf.stack(
            [
                tf.eye(4, batch_shape=(b, n)),  # base == link 0
                rel_RTs[..., 0, :, :],  # link 1
                rel_RTs[..., 0, :, :] @ rel_RTs[..., 1, :, :],  # link 2
                rel_RTs[..., 0, :, :] @ rel_RTs[..., 1, :, :] @ rel_RTs[..., 2, :, :],  # link 3
                rel_RTs[..., 0, :, :]
                @ rel_RTs[..., 1, :, :]
                @ rel_RTs[..., 2, :, :]
                @ rel_RTs[..., 3, :, :],  # link 4
                rel_RTs[..., 0, :, :]
                @ rel_RTs[..., 1, :, :]
                @ rel_RTs[..., 2, :, :]
                @ rel_RTs[..., 3, :, :]
                @ rel_RTs[..., 4, :, :],  # link 5
                rel_RTs[..., 0, :, :]
                @ rel_RTs[..., 1, :, :]
                @ rel_RTs[..., 2, :, :]
                @ rel_RTs[..., 3, :, :]
                @ rel_RTs[..., 4, :, :]
                @ rel_RTs[..., 5, :, :],  # link 6
                rel_RTs[..., 0, :, :]
                @ rel_RTs[..., 1, :, :]
                @ rel_RTs[..., 2, :, :]
                @ rel_RTs[..., 3, :, :]
                @ rel_RTs[..., 4, :, :]
                @ rel_RTs[..., 5, :, :]
                @ rel_RTs[..., 6, :, :],  # link 7
                rel_RTs[..., 0, :, :]
                @ rel_RTs[..., 1, :, :]
                @ rel_RTs[..., 2, :, :]
                @ rel_RTs[..., 3, :, :]
                @ rel_RTs[..., 4, :, :]
                @ rel_RTs[..., 5, :, :]
                @ rel_RTs[..., 6, :, :]
                @ rel_RTs[..., 7, :, :],  # link F
            ],
            axis=-3,
        )

    @tf.function
    def _get_robot_capsules(self, RTs: tf.Tensor) -> tf.Tensor:
        """applies the link transformation to get the capsules of the current robot configuration

        Args:
            RTs [b,n,9,4,4]: The transformation matrices for each of the links

        Returns:
            capsules [b,n,9,7]: The capsules for each of the links (xyz(start), xyz(end), r)
        """
        starts = self.collision_capsules_A
        ends = self.collision_capsules_B
        radii = self.collision_capsules_R

        b, n = tf.shape(RTs)[0], tf.shape(RTs)[1]

        R = RTs[..., :3, :3]
        t = RTs[..., :3, 3]

        starts = tf.einsum("...ij,...j->...i", R, starts) + t
        ends = tf.einsum("...ij,...j->...i", R, ends) + t

        radii = tf.broadcast_to(radii, (b, n, 9))[..., tf.newaxis]

        robot_caps = tf.concat([starts, ends, radii], axis=-1)  # [b,9,7]
        return robot_caps

    def render(self, state: State, update_obstacles: bool = True) -> State:
        """
        Renders the robot and obstacles in the environment.

        Args:
            joint_states [b,7,1]: The joint states of the robot.
            obstacles [b,n,4]: The obstacles in the environment (xyzr).
        """

        RTs = self._get_RTs(state.joint_states[:, tf.newaxis, ...])  # add action dim
        capsules = self._get_robot_capsules(RTs)[0, 0]  # remove batch & action dim

        RTs = RTs[0, 0].numpy()  # remove batch & action dim

        if not "sliders" in self._debug_ids:
            self._refresh_sliders(state.joint_states)

        self._debug_ids.setdefault("robot", {})
        for i, RT in enumerate(RTs):
            cap_t = RT[:3, 3]
            cap_R = R.from_matrix(RT[:3, :3]).as_quat()

            if i not in self._debug_ids["robot"]:
                a = self.collision_capsules_A[i]
                b = self.collision_capsules_B[i]
                r = self.collision_capsules_R[i]
                cap_id = self._create_pbcap_from_A_to_B(a, b, r)

                if i == 8:
                    self._add_axis(cap_id, -1)
                self._debug_ids["robot"][i] = cap_id

            p.resetBasePositionAndOrientation(
                self._debug_ids["robot"][i],
                posObj=cap_t,
                ornObj=cap_R,
                physicsClientId=self.physicsClientId,
            )

        if "target" not in self._debug_ids:
            self._debug_ids["target"] = self._create_target()

        target_pos = state.target[0, :3, 3].numpy()
        target_id = self._debug_ids["target"]
        p.resetBasePositionAndOrientation(
            target_id,
            posObj=target_pos,
            ornObj=R.from_matrix(state.target[0, :3, :3]).as_quat(),
            physicsClientId=self.physicsClientId,
        )

        if self.params.use_obstacles:
            if update_obstacles:
                self._update_obstacles(state.obstacles[0])

            if tf.reduce_any(state.collision_obstacle[0]):
                collided = state.collision_obstacle[0].numpy()  # [n]
                for i, col in enumerate(collided):
                    if col:
                        p.changeVisualShape(
                            self._debug_ids["obstacles"][i],
                            -1,
                            rgbaColor=[1, 0, 0, 0.5],
                            physicsClientId=self.physicsClientId,
                        )

        # add floor
        if self._debug_ids.get("floor", None) is None:
            self._debug_ids["floor"] = self._create_floor()

        try:
            joint_commands = self._read_sliders_from_pb_GUI()
        except Exception as e:
            joint_commands = state.joint_states

        state = State(
            joint_states=joint_commands,
            obstacles=state.obstacles,
            target=state.target,
            ee_pose=state.ee_pose,
            collision_obstacle=state.collision_obstacle,
            is_collision=state.is_collision,
            is_inside_joint_limits=state.is_inside_joint_limits,
            is_in_speed_limits=state.is_in_speed_limits,
            failed=state.failed,
            cumulative_reward=state.cumulative_reward,
            steps=state.steps,
        )

        return state

    @staticmethod
    @tf.function
    def difference_poses(pose1: tf.Tensor, pose2: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        Returns the distance and angle between two poses

        Args:
            pose1 [b,4,4]: The first pose
            pose2 [b,4,4]: The second pose

        Returns:
            dist [b,]: The distance between the two poses
            angle [b,]: The angle between the two poses
        """
        dist = tf.linalg.norm(pose1[..., :3, 3] - pose2[..., :3, 3], axis=-1)
        rot1 = pose1[..., :3, :3]
        rot2 = pose2[..., :3, :3]
        trace = tf.linalg.trace(rot1 @ tf.transpose(rot2, perm=[0, 2, 1]))
        angle = tf.acos(tf.clip_by_value((trace - 1) / 2, -1, 1))
        return dist, angle

    @staticmethod
    @tf.function
    def _dist_to_capsule(p: tf.Tensor, capsules: tf.Tensor):
        """
        determins distances from points p to capsules

        Args:
            p: query points      [..., n, 3]
            capsules: capsules   [..., m, 7]
        Returns:
            d: distances         [..., n, m]
        """
        a = capsules[..., :3]
        b = capsules[..., 3:6]
        r = capsules[..., 6]

        ap = p[..., tf.newaxis, :] - a[..., tf.newaxis, :, :]
        ab = b[..., tf.newaxis, :, :] - a[..., tf.newaxis, :, :]

        e = tf.math.divide_no_nan(tf.reduce_sum(ap * ab, axis=-1, keepdims=True), tf.reduce_sum(
            ab * ab, axis=-1, keepdims=True
        ))

        e = tf.clip_by_value(e, 0.0, 1.0)
        d = tf.linalg.norm(ap - ab * e, axis=-1) - r[..., tf.newaxis, :]
        return d

    @staticmethod
    @tf.function
    def _sphere_capsule_collision(spheres: tf.Tensor, capsules: tf.Tensor) -> tf.Tensor:
        """
        Returns [...,n,m] True if n spheres collide with m capsules. Spheres with negative radii are masked

        Args:
            spheres: sphere centers and radii [..., n, 4]
            capsules: capsules [..., m, 7]

        Returns:
            res: collision check [..., n, m] bool
        """
        d = RobotArmGym._dist_to_capsule(spheres[..., :3], capsules)  # d: [b, n, m]
        return (spheres[..., tf.newaxis, 3] > 0.0) & (
                d <= spheres[..., tf.newaxis, 3]
        )  # [b, n, m]

    def _create_pbcap_from_A_to_B(self, A: np.ndarray, B: np.ndarray, r: float) -> int:
        l = np.linalg.norm(B - A)
        cap_R = R.identity() if l <= 0 else R.align_vectors(B - A, [0, 0, 1])[0]
        offset = cap_R.apply([0, 0, l / 2])

        cap_col = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=r,
            height=l,
            collisionFramePosition=offset,
            collisionFrameOrientation=cap_R.as_quat(),
        )
        cap_vis = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            radius=r,
            length=l,
            visualFramePosition=offset,
            visualFrameOrientation=cap_R.as_quat(),
            rgbaColor=[0, 0, 1, 1],
        )
        cap_id = p.createMultiBody(
            baseCollisionShapeIndex=cap_col,
            baseVisualShapeIndex=cap_vis,
            basePosition=A,
        )
        return cap_id

    def _update_obstacles(self, obstacles):
        # remove old obstacles
        for obstacle_id in self._debug_ids.get("obstacles", []):
            p.removeBody(obstacle_id, physicsClientId=self.physicsClientId)
        self._debug_ids["obstacles"] = []

        # add obstacles
        for i, obs in enumerate(obstacles):
            if obs[3] < 0:
                continue

            sphere = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=obs[3],
            )
            sphere_vis = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=obs[3],
                rgbaColor=[0.3, 0.3, 0.3, 0.5],
            )
            sphere_id = p.createMultiBody(
                basePosition=obs[:3],
                baseCollisionShapeIndex=sphere,
                baseVisualShapeIndex=sphere_vis,
                physicsClientId=self.physicsClientId,
            )
            self._debug_ids["obstacles"].append(sphere_id)

    def _create_floor(self):

        size = 0.1
        floor_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size, size, 0.01],
            physicsClientId=self.physicsClientId,
        )
        floor_vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[size, size, 0.01],
            rgbaColor=[0.5, 0.5, 0.5, 1],
            physicsClientId=self.physicsClientId,
        )
        floor_id = p.createMultiBody(
            baseCollisionShapeIndex=floor_col,
            baseVisualShapeIndex=floor_vis,
            basePosition=[0, 0, -0.005],
            physicsClientId=self.physicsClientId,
        )
        return floor_id

    def _create_target(self):
        r = 0.01
        target_col = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=r,
            physicsClientId=self.physicsClientId,
        )
        target_vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=r,
            rgbaColor=[0, 1, 0, 1],
            physicsClientId=self.physicsClientId,
        )
        target_id = p.createMultiBody(
            baseCollisionShapeIndex=target_col,
            baseVisualShapeIndex=target_vis,
            physicsClientId=self.physicsClientId,
        )

        self._add_axis(target_id, -1)

        return target_id

    def _add_axis(self, object_id, link_index, length=0.1):
        x_axis = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0.1, 0, 0],
            lineColorRGB=[1, 0, 0],
            lineWidth=length,
            lifeTime=0,
            parentObjectUniqueId=object_id,
            parentLinkIndex=link_index,
        )

        y_axis = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0.1, 0],
            lineColorRGB=[0, 1, 0],
            lineWidth=length,
            lifeTime=0,
            parentObjectUniqueId=object_id,
            parentLinkIndex=link_index,
        )

        z_axis = p.addUserDebugLine(
            lineFromXYZ=[0, 0, 0],
            lineToXYZ=[0, 0, 0.1],
            lineColorRGB=[0, 0, 1],
            lineWidth=length,
            lifeTime=0,
            parentObjectUniqueId=object_id,
            parentLinkIndex=link_index,
        )

    def _refresh_sliders(self, joint_states: np.ndarray | None = None) -> list[int]:

        # remove old sliders
        p.removeAllUserParameters(physicsClientId=self.physicsClientId)

        self._debug_ids["sliders"] = []
        # add new
        joint_limits = self.joint_limits
        for i in range(7):
            val = 0 if joint_states is None else joint_states[0, i, 0]
            self._debug_ids["sliders"].append(
                p.addUserDebugParameter(
                    f"j{i + 1}",
                    joint_limits[i, 0],
                    joint_limits[i, 1],
                    val,
                    physicsClientId=self.physicsClientId,
                )
            )
        return self._debug_ids["sliders"]

    def _read_sliders_from_pb_GUI(self) -> np.ndarray:
        sliders = self._debug_ids["sliders"]
        joint_states = []
        for i in range(7):
            joint_states.append(
                p.readUserDebugParameter(sliders[i], physicsClientId=self.physicsClientId)
            )
        return np.array(joint_states)[np.newaxis, :, np.newaxis].astype(np.float32)  # [1,7,1]


def main():
    physicsId = p.connect(p.GUI)

    gym = RobotArmGym(physicsClientId=physicsId)

    gym.reset(batch_size=3)
    state = gym.reset()

    update_obstacles = True

    while True:
        # action = tf.random.uniform((1, 7, 1), -0.1, 0.1)

        state = gym.render(state, update_obstacles=update_obstacles)
        update_obstacles = False

        # 0 step, just to check the state as return by render (user command)
        action = tf.zeros((1, 7))

        state, reward, terminated, truncated, info = gym.step(state, action)

        # can be called continuously (conditional reset)
        state = gym.reset_state(state, terminated)  # | truncated) # no timeout in demo mode

        if terminated[0]:
            print("TERMINATED")
            for k, v in info.items():
                print(f"\t{k}:{v}")

            time.sleep(2)
            state = gym.reset()  # to reset sliders
            update_obstacles = True


if __name__ == "__main__":
    main()
