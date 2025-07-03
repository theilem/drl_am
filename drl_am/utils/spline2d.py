from dataclasses import dataclass

import numpy as np


@dataclass
class Spline:
    p0: np.ndarray
    p1: np.ndarray

    u: float = 0.0
    feasible: bool = True

    def get_position(self, u=None):
        raise NotImplementedError

    def get_velocity(self, u=None):
        raise NotImplementedError

    def get_curvature(self):
        raise NotImplementedError

    def advance_u(self, du):
        self.u += du
        self.u = np.clip(self.u, 0.0, 1.0)


@dataclass
class QuadraticSpline(Spline):
    p2: np.ndarray = None

    def get_position(self, u=None):
        if u is None:
            u = self.u
        return self.p0 * (1 - u) ** 2 + self.p1 * 2 * u * (1 - u) + self.p2 * u ** 2

    def get_velocity(self, u=None):
        if u is None:
            u = self.u
        return 2 * (self.p1 - self.p0) * (1 - u) + 2 * (self.p2 - self.p1) * u

    def get_curvature(self, u=None):
        b_ddot = 2 * (self.p2 - 2 * self.p1 + self.p0)
        vel = self.get_velocity(u)
        return np.cross(vel, b_ddot) / np.linalg.norm(vel) ** 3


@dataclass
class CubicSpline(QuadraticSpline):
    p3: np.ndarray = None

    def get_position(self, u=None):
        if u is None:
            u = self.u
        return self.p0 * (1 - u) ** 3 + self.p1 * 3 * u * (1 - u) ** 2 + self.p2 * 3 * u ** 2 * (
                1 - u) + self.p3 * u ** 3

    def get_velocity(self, u=None):
        if u is None:
            u = self.u
        return 3 * (self.p1 - self.p0) * (1 - u) ** 2 + 6 * (self.p2 - self.p1) * u * (1 - u) + 3 * (
                    self.p3 - self.p2) * u ** 2

    def get_curvature(self, u=None):
        if u is None:
            u = self.u
        b_ddot = 6 * ((1 - u) * (self.p2 - 2 * self.p1 + self.p0) + u * (self.p3 - 2 * self.p2 + self.p1))
        vel = self.get_velocity(u)
        return np.cross(vel, b_ddot) / np.linalg.norm(vel) ** 3
