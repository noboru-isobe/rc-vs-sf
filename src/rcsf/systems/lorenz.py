"""Lorenz system (paper Section 7.2): explicit Euler, dt=0.01, x0 ~ N(0, I_3)."""

from __future__ import annotations

import numpy as np

from .base import Trajectory, random_projection


def lorenz(seed: int, T: int = 1024, dt: float = 0.01, partial: bool = False,
           sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> Trajectory:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=3)
    states = np.empty((T, 3))
    for t in range(T):
        states[t] = x
        dx = np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ])
        x = x + dt * dx
    if partial:
        C = random_projection(rng, 3)
        y = states @ C.T
        return Trajectory(y=y, u=None, name="lorenz_partial")
    return Trajectory(y=states, u=None, name="lorenz")
