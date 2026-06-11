"""NARMA10 benchmark (classic reservoir computing task)."""

from __future__ import annotations

import numpy as np

from .base import Trajectory


def narma10(seed: int, T: int = 4000) -> Trajectory:
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, size=T + 10)
    y = np.zeros(T + 10)
    for t in range(9, T + 9):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * y[t - 9:t + 1].sum()
            + 1.5 * u[t - 9] * u[t]
            + 0.1
        )
    return Trajectory(y=y[10:, None], u=u[10:, None], name="narma10")
