import copy
from dataclasses import dataclass

import numpy as np

import pygame

import tensorflow as tf

from drl_am.utils.misc import dict_to_tensor
from drl_am.utils.spline2d import Spline, QuadraticSpline, CubicSpline


class PathPlanningGym:
    @dataclass
    class Params:
        velocity: float = 0.15
        max_curvature: float = 30.0
        timeout_dist: float = 30.0
        substeps: int = 20
        num_obstacles: int = 30
        num_goals: int = 10
        goal_radius: float = 0.05
        obst_radius: float = 0.05
        obst_shape: str = "circle"  # circle or rectangle

        # Spline
        spline_degree: int = 3
        spline_following_time: float = 0.4

        # Feasibility
        checkpoints: int = 64
        min_length: float = 1.0
        max_length: float = 1.5
        max_curvature_tol: float = 0.95

        # Cost
        cost_tol_factor: float = 0.9
        cost_mult_action_range: float = 1.0
        cost_mult_curvature: float = 1.0
        cost_mult_length: float = 1000.0
        cost_mult_obstacle: float = 10000.0
        cost_mult_outside: float = 1.0


        # Reward
        reward_goal: float = 0.1
        crash_penalty: float = 0.0
        finish_reward: float = 1.0
        move_penalty: float = 0.0

    @dataclass
    class State:
        position: np.ndarray
        velocity: np.ndarray

        goal_positions: np.ndarray
        goal_radius: np.ndarray

        obstacle_positions: np.ndarray
        obstacle_shape: np.ndarray

        steps: int = 0
        failed: bool = False
        success: bool = False
        cumulative_reward: float = 0.0
        trajectory: list = None
        collected_goal_positions: list = None
        collected_goal_radius: list = None

    def __init__(self, params: Params, win=None, offset=(0, 0), win_shape=(1024, 1024)):
        self.params = params
        self.win = win
        self.offset = offset
        self.win_shape = win_shape

        self.action_dim = self.params.spline_degree * 2 - 1

        self.u_s = tf.reshape(
            np.linspace(0, 1, num=self.params.checkpoints, endpoint=True).astype(np.float32),
            (1, 1, -1, 1),
        )
        self.u_squared = tf.square(self.u_s)
        self.u_cubed = self.u_s ** 3
        self.u_back_squared = tf.square(1 - self.u_s)
        self.u_back_cubed = (1 - self.u_s) ** 3

        assert self.params.obst_shape in [
            "circle",
            "rectangle",
        ], f"Invalid obstacle shape {self.params.obst_shape}"

        self.observation_space = {
            "internal_state": (4,),
            "mult_goals": (None, 3),
            "num_goals": (1,),
            "mult_constraints": (None, 3 if self.params.obst_shape == "circle" else 4),
            "num_constraints": (1,),
        }

        self.pool = None
        self.airplane_image = pygame.image.load("drl_am/gym/airplane.png")

    def check_intersection_circle_circle(self, c1, r1, c2, r2):
        """
        :param c1: [A, 2]
        :param r1: [A, 1]
        :param c2: [B, 2]
        :param r2: [B, 1]
        :return: [A, B]
        """
        return np.linalg.norm(c1[:, None] - c2[None, :], axis=-1) < r1[:, None] + r2[None, :]

    def check_intersection_circle_rectangle(self, cc, rc, cr, rr):
        """
        :param cc: Center of circle [A, 2]
        :param rc: Radius of circle [A, 1]
        :param cr: Center of rectangle [B, 2]
        :param rr: half width and height of rectangle [B, 2]
        :return: [A, B]
        """
        circle_distance = np.abs(cc[:, None] - cr[None, :]) - rr[None, :]  # [A, B, 2]
        outside = np.any(circle_distance > rc[:, None], axis=-1)

        inside = np.any(circle_distance <= 0, axis=-1)  # [A, B]

        corner_dist = np.linalg.norm(circle_distance, axis=-1)  # [A, B]
        corner_inside = corner_dist <= rc[:, None]

        return (corner_inside | inside) & ~outside

    def reset(self):
        velocity = np.random.uniform(-1, 1, (2,))
        velocity /= np.linalg.norm(velocity)
        velocity *= self.params.velocity

        obstacle_positions = np.random.uniform(0, 1, (self.params.num_obstacles, 2))
        if self.params.obst_shape == "circle":
            shape_dim = 1
        elif self.params.obst_shape == "rectangle":
            shape_dim = 2
        else:
            raise ValueError("Invalid shape")
        obstacle_shape = np.random.uniform(
            0.5 * self.params.obst_radius,
            self.params.obst_radius,
            (self.params.num_obstacles, shape_dim),
        )

        goal_positions = np.zeros((self.params.num_goals, 2))
        goal_radii = np.zeros((self.params.num_goals, 1))
        for k in range(self.params.num_goals):

            while True:

                goal_position = np.random.uniform(0, 1, (2,))
                goal_radius = np.random.uniform(
                    0.5 * self.params.goal_radius, self.params.goal_radius, (1,)
                )
                goal_position = goal_position * (1 - 2 * goal_radius) + goal_radius

                if self.params.obst_shape == "rectangle":
                    intersection = self.check_intersection_circle_rectangle(
                        goal_position[None, :],
                        goal_radius[None, :],
                        obstacle_positions,
                        obstacle_shape,
                    )
                else:
                    intersection = self.check_intersection_circle_circle(
                        goal_position[None, :],
                        goal_radius[None, :],
                        obstacle_positions,
                        obstacle_shape,
                    )

                if not np.any(intersection):
                    break

            goal_positions[k] = goal_position
            goal_radii[k] = goal_radius

        while True:
            position = np.random.uniform(0.2, 0.8, (2,))
            if self.params.obst_shape == "rectangle":
                intersection = self.check_intersection_circle_rectangle(
                    position[None, :],
                    np.array([[self.params.obst_radius]]),
                    obstacle_positions,
                    obstacle_shape,
                )
            else:
                intersection = self.check_intersection_circle_circle(
                    position[None, :],
                    np.array([[self.params.obst_radius]]),
                    obstacle_positions,
                    obstacle_shape,
                )

            if not np.any(intersection):
                break

        state = PathPlanningGym.State(
            position=position,
            velocity=velocity,
            goal_positions=goal_positions,
            goal_radius=goal_radii,
            obstacle_positions=obstacle_positions,
            obstacle_shape=obstacle_shape,
        )
        state.trajectory = []
        state.collected_goal_positions = []
        state.collected_goal_radius = []
        return state, self.get_obs(state)

    def step(self, state, action, render=False, log_trajectory=False):
        if isinstance(action, Spline):
            spline = action
        else:
            spline = self.action_to_spline(state, action)
        vel_t = self.params.velocity
        fault = False
        num_goals = len(state.goal_positions)
        for _ in range(self.params.substeps):
            vel_u = spline.get_velocity()
            speed = np.linalg.norm(vel_u)
            if speed == 0.0:
                fault = True
                break
            curvature = spline.get_curvature()
            if np.abs(curvature) > self.params.max_curvature:
                fault = True
                break
            scale = vel_t / speed / self.params.substeps * self.params.spline_following_time
            spline.advance_u(scale)
            if spline.u >= 1.0:
                fault = True
                break
            position = spline.get_position()
            if np.any(position < 0) or np.any(position > 1):
                fault = True
                break

            if self.params.obst_shape == "circle":
                dist = np.linalg.norm(state.obstacle_positions - position[None, :], axis=-1)
                obstacle_radius = state.obstacle_shape[..., 0]
                if np.any(dist < obstacle_radius):
                    fault = True
                    break
            else:
                diff = np.abs(state.obstacle_positions - position[None, :]) - state.obstacle_shape
                if np.any(np.all(diff < 0, axis=-1)):
                    fault = True
                    break

            goal_dist = np.linalg.norm(position[None, :] - state.goal_positions, axis=-1)
            in_goal = goal_dist < state.goal_radius[..., 0]
            if log_trajectory and np.any(in_goal):
                for idx in np.where(in_goal)[0]:
                    state.collected_goal_positions.append(state.goal_positions[idx])
                    state.collected_goal_radius.append(state.goal_radius[idx])
            state.goal_positions = state.goal_positions[~in_goal]
            state.goal_radius = state.goal_radius[~in_goal]

            if fault:
                break
            if render:
                inter_state = PathPlanningGym.State(
                    position,
                    spline.get_velocity() / speed * vel_t,
                    state.goal_positions,
                    state.goal_radius,
                    state.obstacle_positions,
                    state.obstacle_shape,
                )
                self.render(inter_state, spline)
                pygame.time.delay(10)

        state.position = spline.get_position()
        vel_u = spline.get_velocity()
        speed = np.linalg.norm(vel_u)
        if speed == 0.0:
            state.velocity = np.zeros_like(state.velocity)
            fault = True
        else:
            state.velocity = vel_u / speed * vel_t
        state.steps += 1
        state.failed = fault

        terminal = fault
        reward = self.params.reward_goal * (num_goals - len(state.goal_positions))
        # Check if in goal
        if not fault and len(state.goal_positions) == 0:
            terminal = True
            reward += self.params.finish_reward
            state.success = True
        else:
            state.success = False

        if fault:
            reward -= self.params.crash_penalty

        reward -= self.params.move_penalty

        state.cumulative_reward += reward

        if log_trajectory:
            state.trajectory.append(spline)

        obs = self.get_obs(state)

        if render:
            self.render(state, spline)

        distance = state.steps * self.params.velocity * self.params.spline_following_time

        return obs, reward, terminal, distance > self.params.timeout_dist, self.get_info(state)

    def get_info(self, state):
        distance = state.steps * self.params.velocity * self.params.spline_following_time
        timeout = (distance > self.params.timeout_dist) and not state.success and not state.failed
        info = {
            "steps": state.steps,
            "failed": state.failed,
            "success": state.success,
            "goals_remaining": len(state.goal_positions),
            "timeout": timeout,
            "cumulative_reward": state.cumulative_reward,
            "total_distance": distance,
        }
        return info

    def render(self, state, spline=None, show_trajectory=False):
        shape = np.array(self.win_shape)
        canvas = pygame.Surface(shape, pygame.SRCALPHA, 32)
        canvas.fill((255, 255, 255))

        # Draw goals
        for goal_position, goal_radius in zip(state.goal_positions, state.goal_radius):
            pygame.draw.circle(
                canvas,
                (100, 100, 255),
                (goal_position * shape).astype(int),
                int(goal_radius * shape[0]),
            )
            pygame.draw.circle(
                canvas,
                (0, 0, 180),
                (goal_position * shape).astype(int),
                int(goal_radius * shape[0]),
                6
            )

        # Draw obstacles
        for obstacle_position, obstacle_shape in zip(
                state.obstacle_positions, state.obstacle_shape
        ):
            if self.params.obst_shape == "circle":
                pygame.draw.circle(
                    canvas,
                    (50, 50, 50),
                    (obstacle_position * shape).astype(int),
                    int(obstacle_shape * shape[0]),
                )
            else:
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    pygame.Rect(
                        (obstacle_position - obstacle_shape) * shape, obstacle_shape * shape * 2
                    ),
                )

        # Draw obstacles
        for obstacle_position, obstacle_shape in zip(
                state.obstacle_positions, state.obstacle_shape
        ):
            if self.params.obst_shape == "circle":
                pygame.draw.circle(
                    canvas,
                    (50, 50, 50),
                    (obstacle_position * shape).astype(int),
                    int(obstacle_shape * shape[0]),
                )
            else:
                pygame.draw.rect(
                    canvas,
                    (100, 100, 100),
                    pygame.Rect(
                        ((obstacle_position - obstacle_shape) * shape) + 6, (obstacle_shape * shape * 2) - 12
                    ),
                )

        # Draw image frame
        pygame.draw.rect(
            canvas,
            (0, 0, 0),
            pygame.Rect(
                (0, 0),
                shape.astype(int)
            ),
            6
        )

        if spline is not None:
            if isinstance(spline, list):
                splines = spline
            else:
                splines = [spline]
            for spline in splines:
                us = np.linspace(0, 1, 16)
                positions = [spline.get_position(u=u) for u in us]
                color = (0, 200, 0) if spline.feasible else (200, 0, 0)

                for k, position in enumerate(positions[:-1]):
                    pygame.draw.line(
                        canvas,
                        color,
                        (position * shape).astype(int),
                        (positions[k + 1] * shape).astype(int),
                        4,
                    )

        if show_trajectory:
            for k, spline in enumerate(state.trajectory):

                # Draw collected goals as empty blue circles
                for goal_position, goal_radius in zip(
                        state.collected_goal_positions, state.collected_goal_radius
                ):
                    pygame.draw.circle(
                        canvas,
                        (0, 0, 180),
                        (goal_position * shape).astype(int),
                        int(goal_radius * shape[0]),
                        6
                    )

                us = np.linspace(0, 1, 16)
                us_followed = np.linspace(0, spline.u, 16)
                positions = [spline.get_position(u=u) for u in us]
                positions_followed = [spline.get_position(u=u) for u in us_followed]
                color = (0, 0, 0)
                for k, position in enumerate(positions[:-1]):
                    pygame.draw.line(
                        canvas,
                        color,
                        (position * shape).astype(int),
                        (positions[k + 1] * shape).astype(int),
                        2,
                    )
                for k, position in enumerate(positions_followed[:-1]):
                    pygame.draw.line(
                        canvas,
                        (255, 102, 0),
                        (position * shape).astype(int),
                        (positions_followed[k + 1] * shape).astype(int),
                        4,
                    )

        # Draw airplane
        airplane = pygame.transform.scale(self.airplane_image, (shape / 15).astype(int))
        airplane = pygame.transform.rotate(airplane, np.degrees(np.arctan2(state.velocity[0], state.velocity[1])) - 180)
        canvas.blit(airplane, (state.position * shape - np.array(airplane.get_size()) / 2).astype(int))

        self.win.blit(canvas, self.offset)
        pygame.display.update()

    def action_to_spline(self, state, action, check_feasibility=False):
        b0 = state.position
        b1 = state.position + state.velocity * (action[0] + 1.0) / 2
        b2 = state.position - action[1:3] * self.params.velocity * 1.5
        feasible = (
            self.action_feasibility(self.get_obs(state), action) if check_feasibility else True
        )
        if self.params.spline_degree == 2:
            return QuadraticSpline(p0=b0, p1=b1, p2=b2, feasible=feasible)
        elif self.params.spline_degree == 3:
            b3 = state.position - action[3:5] * self.params.velocity * 1.5
            return CubicSpline(p0=b0, p1=b1, p2=b2, p3=b3, feasible=feasible)
        else:
            raise ValueError("Spline degree must be 2 or 3")

    def get_obs(self, state):
        goal_positions = np.pad(
            state.goal_positions,
            ((0, self.params.num_goals - len(state.goal_positions)), (0, 0)),
            mode="constant",
        )
        goal_radius = np.pad(
            state.goal_radius,
            ((0, self.params.num_goals - len(state.goal_radius)), (0, 0)),
            mode="constant",
        )

        internal_state = np.concatenate((state.position, state.velocity))
        mult_goals = np.concatenate((goal_positions, goal_radius), axis=-1)
        num_goals = np.array([len(state.goal_positions)], dtype=np.int32)
        mult_constraints = np.concatenate(
            (state.obstacle_positions, state.obstacle_shape), axis=-1
        )
        num_constraints = np.array([len(state.obstacle_positions)], dtype=np.int32)

        obs = {
            "internal_state": internal_state[None, ...].astype(np.float32),
            "mult_goals": mult_goals[None, ...].astype(np.float32),
            "num_goals": num_goals[None, ...].astype(np.int32),
            "mult_constraints": mult_constraints[None, ...].astype(np.float32),
            "num_constraints": num_constraints[None, ...].astype(np.int32),
        }
        return obs

    def action_feasibility(self, obs, action):
        obs = dict_to_tensor(obs)
        action = tf.convert_to_tensor(
            np.reshape(action, (1, -1, self.action_dim)), dtype=tf.float32
        )
        result = self.action_feasibility_tf(obs, action).numpy()[0]  # (N,)
        return result.item() if len(result) == 1 else result

    @tf.function
    def generate_states_tf(self, num_states):
        # TODO: Make number of position candidates a parameter
        position_candidates = tf.random.uniform(
            (
                num_states,
                10,
                1,
                2,
            ),
            0.0,
            1.0,
            dtype=tf.float32,
        )
        velocity = tf.random.uniform(
            (
                num_states,
                2,
            ),
            -1.0,
            1.0,
            dtype=tf.float32,
        )
        velocity /= tf.linalg.norm(velocity, axis=-1, keepdims=True)
        velocity *= self.params.velocity

        obstacles_position = tf.random.uniform(
            (num_states, self.params.num_obstacles, 2), 0.0, 1.0, dtype=tf.float32
        )
        if self.params.obst_shape == "circle":
            shape_dim = 1
        elif self.params.obst_shape == "rectangle":
            shape_dim = 2
        else:
            raise ValueError("Invalid shape")
        obstacles_shape = tf.random.uniform(
            (num_states, self.params.num_obstacles, shape_dim),
            0.5 * self.params.obst_radius,
            self.params.obst_radius,
            dtype=tf.float32,
        )

        position_collision = self.collision_obstacle_tf(
            position_candidates, obstacles_position, obstacles_shape
        )  # (B, 10, 1, O)
        position_collision = tf.squeeze(tf.reduce_any(position_collision, axis=-1), -1)  # (B, 10)
        collision_sort = tf.argsort(
            tf.cast(position_collision, tf.int32), axis=1, direction="ASCENDING"
        )[
                         :, 0
                         ]  # (B,)
        position_candidates = tf.gather(
            position_candidates, collision_sort, axis=1, batch_dims=1
        )  # (B, 1, 2)
        position = tf.squeeze(position_candidates, axis=1)  # (B, 2)

        obs = {
            "internal_state": tf.concat([position, velocity], axis=-1),
            "mult_constraints": tf.concat([obstacles_position, obstacles_shape], axis=-1),
            "num_constraints": tf.ones((num_states, 1), dtype=tf.int32)
                               * self.params.num_obstacles,
        }

        return obs

    @tf.function
    def collision_obstacle_tf(self, pos, obstacles_pos, obstacles_shape):
        """

        :param pos: [B, N, T, 2]
        :param obstacles_pos: [B, O, 2]
        :param obstacles_shape:  [B, O, 1/2]
        :return: [B, N, T, O]
        """
        if self.params.obst_shape == "circle":
            obstacle_dist = tf.reduce_sum(
                tf.square(pos[:, :, :, None, :] - obstacles_pos[:, None, None, :, :]), axis=-1
            )
            return obstacle_dist < obstacles_shape[:, None, None, :, 0] ** 2
        else:
            diff = tf.abs(pos[:, :, :, None, :] - obstacles_pos[:, None, None, :, :])
            diff = diff - obstacles_shape[:, None, None, :, :]
            return tf.reduce_all(diff < 0, axis=-1)

    @tf.function
    def action_feasibility_tf(self, obs, actions, return_info=False):
        if self.params.spline_degree == 2:
            return self.action_feasibility2_tf(obs, actions, return_info=return_info)
        elif self.params.spline_degree == 3:
            return self.action_feasibility3_tf(obs, actions, return_info=return_info)
        else:
            raise ValueError("Spline degree must be 2 or 3")

    @tf.function
    def action_feasibility2_tf(self, obs, actions, return_info=False):
        """
        obs: dict
        actions: (B, N, 3)
        returns: (B, N)
        """
        internal_state = obs["internal_state"]
        positions = internal_state[..., :2]
        velocities = internal_state[..., 2:]
        mult_constraints = obs["mult_constraints"]
        obstacles_pos = mult_constraints[..., :2]  # (B, O, 2)
        obstacles_shape = mult_constraints[..., 2:]  # (B, O, 1/2)

        b0 = positions[:, None, :]  # (B, 1, 2)
        b1 = b0 + velocities[:, None, :] * (actions[..., :1] + 1.0) / 2  # (B, N, 2)
        b2 = b0 - actions[..., 1:] * self.params.velocity * 1.5  # (B, N, 2)

        b0 = b0[:, :, None, :]  # (B, 1, 1, 2)
        b1 = b1[:, :, None, :]  # (B, N, 1, 2)
        b2 = b2[:, :, None, :]  # (B, N, 1, 2)

        pos_on_spline = (
                b1 + (b0 - b1) * self.u_back_squared + (b2 - b1) * self.u_squared
        )  # (B, N, T, 2)

        vel_on_spline = 2 * (b1 - b0) * (1 - self.u_s) + 2 * (b2 - b1) * self.u_s  # (B, N, T, 2)
        acc_on_spline = 2 * (b2 - 2 * b1 + b0)  # (B, N, 1, 2)

        outside = tf.reduce_any((pos_on_spline < 0) | (pos_on_spline > 1), axis=-1)  # (B, N, T)
        outside_violation = tf.reduce_any(outside, axis=-1)  # (B, N)

        obstacle_violations = self.collision_obstacle_tf(
            pos_on_spline, obstacles_pos, obstacles_shape
        )  # (B, N, T, O)
        obstacle_violation = tf.reduce_any(obstacle_violations, axis=[-2, -1])  # (B, N)

        distances = tf.reduce_sum(
            tf.sqrt(
                tf.reduce_sum(
                    tf.square(pos_on_spline[:, :, 1:, :] - pos_on_spline[:, :, :-1, :]), axis=-1
                )
            ),
            axis=-1,
        )  # (B, N)

        length_good = (distances > self.params.velocity * self.params.min_length) & (
                distances < self.params.velocity * self.params.max_length
        )  # (B, N)

        curvature = (
                            vel_on_spline[..., 0] * acc_on_spline[..., 1]
                            - vel_on_spline[..., 1] * acc_on_spline[..., 0]
                    ) / (tf.reduce_sum(tf.square(vel_on_spline), axis=-1) ** (3 / 2))
        good_curve = tf.reduce_all(
            tf.abs(curvature) <= self.params.max_curvature * self.params.max_curvature_tol, axis=-1
        )  # (B, N)

        spline_good = ~outside_violation & ~obstacle_violation & length_good & good_curve
        if return_info:
            info = {
                "violate_outside": outside_violation,
                "violate_collision": obstacle_violation,
                "violate_length": ~length_good,
                "violate_curvature": ~good_curve,
            }
            return spline_good, info

        return spline_good

    @tf.function
    def action_feasibility3_tf(self, obs, actions, return_info=False):
        """
        obs: dict
        actions: (B, N, 5)
        returns: (B, N)
        """
        internal_state = obs["internal_state"]
        positions = internal_state[..., :2]
        velocities = internal_state[..., 2:]
        mult_constraints = obs["mult_constraints"]
        obstacles_pos = mult_constraints[..., :2]  # (B, O, 2)
        obstacles_shape = mult_constraints[..., 2:]  # (B, O, 1/2)

        b0 = positions[:, None, :]  # (B, 1, 2)
        b1 = b0 + velocities[:, None, :] * (actions[..., :1] + 1.0) / 2  # (B, N, 2)
        b2 = b0 - actions[..., 1:3] * self.params.velocity * 1.5  # (B, N, 2)
        b3 = b0 - actions[..., 3:] * self.params.velocity * 1.5  # (B, N, 2)

        b0 = b0[:, :, None, :]  # (B, 1, 1, 2)
        b1 = b1[:, :, None, :]  # (B, N, 1, 2)
        b2 = b2[:, :, None, :]  # (B, N, 1, 2)
        b3 = b3[:, :, None, :]  # (B, N, 1, 2)

        pos_on_spline = (
                self.u_back_cubed * b0
                + 3 * self.u_back_squared * self.u_s * b1
                + 3 * (1 - self.u_s) * self.u_squared * b2
                + self.u_cubed * b3
        )  # (B, N, T, 2)

        vel_on_spline = (
                3 * self.u_back_squared * (b1 - b0)
                + 6 * (1 - self.u_s) * self.u_s * (b2 - b1)
                + 3 * self.u_squared * (b3 - b2)
        )  # (B, N, T, 2)
        acc_on_spline = 6 * self.u_back_squared * (b2 - 2 * b1 + b0) + 6 * self.u_s * (
                b3 - 2 * b2 + b1
        )  # (B, N, 1, 2)

        action_bound_violation = tf.reduce_any(
            (tf.abs(actions) > 1.0), axis=-1
        )

        outside = tf.reduce_any((pos_on_spline < 0) | (pos_on_spline > 1), axis=-1)  # (B, N, T)
        outside_violation = tf.reduce_any(outside, axis=-1)  # (B, N)

        obstacle_violations = self.collision_obstacle_tf(
            pos_on_spline, obstacles_pos, obstacles_shape
        )  # (B, N, T, O)
        obstacle_violation = tf.reduce_any(obstacle_violations, axis=[-2, -1])  # (B, N)

        distances = tf.reduce_sum(
            tf.sqrt(
                tf.reduce_sum(
                    tf.square(pos_on_spline[:, :, 1:, :] - pos_on_spline[:, :, :-1, :]), axis=-1
                )
            ),
            axis=-1,
        )  # (B, N)

        length_good = (distances > self.params.velocity * self.params.min_length) & (
                distances < self.params.velocity * self.params.max_length
        )  # (B, N)

        curvature = (
                            vel_on_spline[..., 0] * acc_on_spline[..., 1]
                            - vel_on_spline[..., 1] * acc_on_spline[..., 0]
                    ) / (tf.reduce_sum(tf.square(vel_on_spline), axis=-1) ** (3 / 2))
        good_curve = tf.reduce_all(
            tf.abs(curvature) <= self.params.max_curvature * self.params.max_curvature_tol, axis=-1
        )  # (B, N)

        spline_good = ~outside_violation & ~obstacle_violation & length_good & good_curve & ~action_bound_violation
        if return_info:
            violate_init_pos = self.collision_obstacle_tf(
                b0, obstacles_pos, obstacles_shape
            )  # (B, 1, 1, O)
            info = {
                "violate_init_pos": tf.reduce_any(violate_init_pos, axis=[1, 2, 3]),
                "violate_outside": outside_violation,
                "violate_collision": obstacle_violation,
                "violate_length": ~length_good,
                "violate_curvature": ~good_curve,
                "violate_action_bound": action_bound_violation,
            }
            return spline_good, info

        return spline_good

    def animate_trajectory(self, state):
        if state.trajectory is None:
            print("No trajectory to animate")
            return
        spline0 = state.trajectory[0]
        pos = spline0.get_position(u=0)
        vel = spline0.get_velocity(u=0)
        goal_positions = np.concatenate(
            (state.goal_positions, np.stack(state.collected_goal_positions, axis=0)), axis=0
        )
        goal_radius = np.concatenate(
            (state.goal_radius, np.stack(state.collected_goal_radius, axis=0)), axis=0
        )

        s = PathPlanningGym.State(
            pos, vel, goal_positions, goal_radius, state.obstacle_positions, state.obstacle_shape
        )

        for spline in state.trajectory:
            sp = copy.deepcopy(spline)
            sp.u = 0.0
            self.step(s, sp, render=True)
