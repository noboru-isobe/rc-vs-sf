"""Common interface every online predictor implements.

The contract is strict predict-then-update:
  1. reset(d_y, d_u, T) is called once before the stream starts.
  2. At each step t, predict(u_t) must return ŷ_t using only u_{<=t} and y_{<t}.
  3. update(y_t) then reveals the truth.
Enforced by the spy test in tests/test_fairness.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class OnlinePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def reset(self, d_y: int, d_u: int, T: int, seed: int = 0) -> None:
        """Prepare for a fresh stream. T is the horizon (filter length budget)."""

    @abstractmethod
    def predict(self, u_t: np.ndarray | None) -> np.ndarray:
        """Return ŷ_t (shape (d_y,)). u_t is None for autonomous systems."""

    @abstractmethod
    def update(self, y_t: np.ndarray) -> None:
        """Observe the true y_t and adapt."""

    def num_trainable_params(self) -> int:
        return 0


class Persistence(OnlinePredictor):
    """ŷ_{t} = y_{t-1}: the floor baseline shown in every plot."""

    name = "persistence"

    def reset(self, d_y, d_u, T, seed=0):
        self._last = np.zeros(d_y)

    def predict(self, u_t):
        return self._last.copy()

    def update(self, y_t):
        self._last = np.asarray(y_t, dtype=float)
