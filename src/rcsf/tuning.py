"""Hyperparameter selection protocol (fairness rule 4 of the README).

Grid search on validation seeds that are disjoint from the test seeds; the
selected config is then run once per test seed. Scoring uses the mean
post-washout error over the second half of each validation trajectory.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .evaluation import run_online
from .systems.base import Trajectory

VALIDATION_SEEDS = (1000, 1001, 1002)


@dataclass
class TuneResult:
    best_params: dict
    best_score: float
    all_scores: list  # (params, score) pairs


def expand_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def tune(method_factory: Callable[..., object], grid: dict,
         make_traj: Callable[[int], Trajectory],
         seeds=VALIDATION_SEEDS) -> TuneResult:
    """method_factory(**params) -> OnlinePredictor; make_traj(seed) -> Trajectory."""
    trajs = [make_traj(s) for s in seeds]
    scores = []
    for params in expand_grid(grid):
        vals = []
        for s, traj in zip(seeds, trajs):
            res = run_online(method_factory(**params), traj, seed=s)
            second_half = res.errors[max(res.washout, traj.T // 2):]
            vals.append(second_half.mean())
        score = float(np.mean(vals))
        if not np.isfinite(score):
            score = np.inf
        scores.append((params, score))
    best_params, best_score = min(scores, key=lambda ps: ps[1])
    return TuneResult(best_params=best_params, best_score=best_score, all_scores=scores)


def validation_stats(make_traj: Callable[[int], Trajectory],
                     seeds=VALIDATION_SEEDS) -> tuple[np.ndarray, np.ndarray]:
    """Standardization statistics from the validation seeds (causal w.r.t. test)."""
    ys = np.concatenate([make_traj(s).y for s in seeds])
    mu, sd = ys.mean(axis=0), ys.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd
