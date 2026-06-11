"""NARMA10 benchmark (classic reservoir computing task)."""

from __future__ import annotations

import numpy as np

from .base import Trajectory


def narma10(seed: int, T: int = 4000, max_attempts: int = 50) -> Trajectory:
    """NARMA10 with divergence guard.

    The recursion has a positive feedback term (0.05 y_t * sum y) and is known
    to blow up for some input realizations on long horizons; the standard
    remedy is to redraw the input sequence. Deterministic per seed: attempts
    use rng streams (seed, attempt).
    """
    for attempt in range(max_attempts):
        rng = np.random.default_rng([seed, attempt])
        u = rng.uniform(0, 0.5, size=T + 10)
        y = np.zeros(T + 10)
        diverged = False
        for t in range(9, T + 9):
            y[t + 1] = (
                0.3 * y[t]
                + 0.05 * y[t] * y[t - 9:t + 1].sum()
                + 1.5 * u[t - 9] * u[t]
                + 0.1
            )
            if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 10.0:
                diverged = True
                break
        if not diverged:
            return Trajectory(y=y[10:, None], u=u[10:, None], name="narma10")
    raise RuntimeError(f"narma10(seed={seed}, T={T}) diverged {max_attempts} times")
