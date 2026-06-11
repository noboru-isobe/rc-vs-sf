"""Mackey–Glass delay differential equation (tau=17), RK4 dt=0.1 subsampled to 1.0.

dx/dt = 0.2 x(t - tau) / (1 + x(t - tau)^10) - 0.1 x(t)
"""

from __future__ import annotations

import numpy as np

from .base import Trajectory


def mackey_glass(seed: int, T: int = 10000, tau: float = 17.0, dt: float = 0.1,
                 subsample: int = 10, transient: int = 1000) -> Trajectory:
    rng = np.random.default_rng(seed)
    delay_steps = int(round(tau / dt))
    n_total = (T + transient) * subsample
    # History buffer initialized near the attractor with a seeded perturbation.
    x = np.empty(n_total + delay_steps + 1)
    x[: delay_steps + 1] = 1.2 + 0.05 * rng.standard_normal(delay_steps + 1)

    def f(xt, xd):
        return 0.2 * xd / (1.0 + xd**10) - 0.1 * xt

    for n in range(delay_steps, delay_steps + n_total):
        xd = x[n - delay_steps]          # delayed value (held constant over the step)
        xt = x[n]
        k1 = f(xt, xd)
        k2 = f(xt + 0.5 * dt * k1, xd)
        k3 = f(xt + 0.5 * dt * k2, xd)
        k4 = f(xt + dt * k3, xd)
        x[n + 1] = xt + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    series = x[delay_steps:][::subsample][transient:transient + T]
    return Trajectory(y=series[:, None], u=None, name="mackey_glass")
