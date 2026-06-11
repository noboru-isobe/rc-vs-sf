"""Classical autonomous double pendulum, RK4 integration.

Deviation from the paper (which uses a Brax cart + double pendulum with linear
feedback control): we use the standard frictionless planar double pendulum and
observe smooth coordinates [sin θ1, cos θ1, sin θ2, cos θ2, ω1, ω2] (d=6),
mirroring Brax's angle encoding. Documented in the README.
"""

from __future__ import annotations

import numpy as np

from .base import Trajectory, random_projection

G = 9.81


def _deriv(s: np.ndarray, m1=1.0, m2=1.0, l1=1.0, l2=1.0) -> np.ndarray:
    th1, w1, th2, w2 = s
    delta = th1 - th2
    denom = 2 * m1 + m2 - m2 * np.cos(2 * delta)
    a1 = (
        -G * (2 * m1 + m2) * np.sin(th1)
        - m2 * G * np.sin(th1 - 2 * th2)
        - 2 * np.sin(delta) * m2 * (w2**2 * l2 + w1**2 * l1 * np.cos(delta))
    ) / (l1 * denom)
    a2 = (
        2 * np.sin(delta)
        * (w1**2 * l1 * (m1 + m2) + G * (m1 + m2) * np.cos(th1) + w2**2 * l2 * m2 * np.cos(delta))
    ) / (l2 * denom)
    return np.array([w1, a1, w2, a2])


def double_pendulum(seed: int, T: int = 1024, dt: float = 0.01,
                    partial: bool = False) -> Trajectory:
    rng = np.random.default_rng(seed)
    s = np.array([
        rng.uniform(-np.pi / 2, np.pi / 2),
        rng.normal(scale=0.5),
        rng.uniform(-np.pi / 2, np.pi / 2),
        rng.normal(scale=0.5),
    ])
    obs = np.empty((T, 6))
    for t in range(T):
        obs[t] = [np.sin(s[0]), np.cos(s[0]), np.sin(s[2]), np.cos(s[2]), s[1], s[3]]
        k1 = _deriv(s)
        k2 = _deriv(s + 0.5 * dt * k1)
        k3 = _deriv(s + 0.5 * dt * k2)
        k4 = _deriv(s + dt * k3)
        s = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    if partial:
        C = random_projection(rng, 6)
        return Trajectory(y=obs @ C.T, u=None, name="double_pendulum_partial")
    return Trajectory(y=obs, u=None, name="double_pendulum")
