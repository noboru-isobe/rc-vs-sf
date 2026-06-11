"""The single online evaluation loop shared by every method."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .methods.base import OnlinePredictor
from .systems.base import Trajectory

WASHOUT = 100


@dataclass(frozen=True)
class RunResult:
    errors: np.ndarray       # (T,) instantaneous squared error ||ŷ_t - y_t||²
    predictions: np.ndarray  # (T, d_y)
    washout: int

    def mean_error(self) -> float:
        return float(self.errors[self.washout:].mean())

    def final_error(self, fraction: float = 0.1) -> float:
        n = max(1, int(len(self.errors) * fraction))
        return float(self.errors[-n:].mean())

    def nrmse(self, y: np.ndarray) -> float:
        e = self.errors[self.washout:]
        return float(np.sqrt(e.mean()) / y[self.washout:].std())


def run_online(method: OnlinePredictor, traj: Trajectory, seed: int = 0,
               washout: int = WASHOUT) -> RunResult:
    """Strict predict-then-update loop. The method never sees y_t before predicting it."""
    method.reset(traj.d_y, traj.d_u, traj.T, seed=seed)
    errors = np.empty(traj.T)
    preds = np.empty_like(traj.y, dtype=float)
    for t in range(traj.T):
        u_t = None if traj.u is None else traj.u[t]
        y_hat = np.asarray(method.predict(u_t), dtype=float).reshape(-1)
        y_t = traj.y[t]
        errors[t] = float(np.sum((y_hat - y_t) ** 2))
        preds[t] = y_hat
        method.update(y_t)
    return RunResult(errors=errors, predictions=preds, washout=washout)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average used for the log-scale loss plots (paper style)."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")
