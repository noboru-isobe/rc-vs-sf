"""Trajectory container shared by all systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Trajectory:
    """One realization of a dynamical system.

    y: (T, d_y) observations to predict.
    u: (T, d_u) exogenous inputs, or None for autonomous systems.
    name: system identifier (used for caching/plot labels).
    """

    y: np.ndarray
    u: np.ndarray | None
    name: str

    def __post_init__(self):
        assert self.y.ndim == 2
        if self.u is not None:
            assert self.u.ndim == 2 and len(self.u) == len(self.y)

    @property
    def T(self) -> int:
        return len(self.y)

    @property
    def d_y(self) -> int:
        return self.y.shape[1]

    @property
    def d_u(self) -> int:
        return 0 if self.u is None else self.u.shape[1]


def random_projection(rng: np.random.Generator, d_in: int, d_out: int = 1) -> np.ndarray:
    """Gaussian observation matrix C in R^{d_out x d_in} for partial observation."""
    return rng.normal(size=(d_out, d_in))
