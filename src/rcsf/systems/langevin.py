"""Langevin dynamics (paper Section 7.2): Euler–Maruyama on a quartic potential.

X_{t+1} = X_t - eta * grad V(X_t) + sqrt(2 eta) w_t,  w_t ~ N(0, I).

V(x) = sum_j (0.05 x_j^4 - x_j^2 + 0.1 x_j) - c * sum_{i<j} x_i^2 x_j^2.

Deviation from the paper: with the paper's c = 0.2 and d = 64 the potential is
unbounded below along the diagonal (the coupling quartic dominates 0.05 sum x^4)
and Euler–Maruyama blows up almost immediately. Boundedness along the diagonal
requires c <= 0.1/(d-1); we default to c = 0.05/(d-1) for margin. See README.
"""

from __future__ import annotations

import numpy as np

from .base import Trajectory


def _grad_v(x: np.ndarray, coupling: float) -> np.ndarray:
    sq = x**2
    other = sq.sum() - sq
    return 0.2 * x**3 - 2.0 * x + 0.1 - 2.0 * coupling * x * other


def langevin(seed: int, T: int = 1024, d: int = 64, eta: float = 0.01,
             coupling: float | None = None) -> Trajectory:
    if coupling is None:
        coupling = 0.05 / (d - 1)
    rng = np.random.default_rng(seed)
    x = np.zeros(d)
    y = np.empty((T, d))
    for t in range(T):
        y[t] = x
        x = x - eta * _grad_v(x, coupling) + np.sqrt(2 * eta) * rng.normal(size=d)
    return Trajectory(y=y, u=None, name="langevin")
