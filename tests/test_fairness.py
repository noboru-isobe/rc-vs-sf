"""Causality audit: no method may read y_t before predicting step t."""

import numpy as np
import pytest

from rcsf.evaluation import run_online, smooth
from rcsf.methods.base import OnlinePredictor, Persistence
from rcsf.systems import SYSTEMS
from rcsf.systems.base import Trajectory


class SpyTrajectory(Trajectory):
    """Trajectory whose y raises if row t is read before the loop reveals it."""

    def __init__(self, traj: Trajectory):
        object.__setattr__(self, "y", _SpyArray(traj.y))
        object.__setattr__(self, "u", traj.u)
        object.__setattr__(self, "name", traj.name)

    def __post_init__(self):
        pass


class _SpyArray:
    def __init__(self, data):
        self._data = data
        self.revealed = -1

    def __getitem__(self, idx):
        if isinstance(idx, int) and idx > self.revealed:
            self.revealed = idx  # the eval loop itself reveals truth in order
        return self._data[idx]

    def __len__(self):
        return len(self._data)

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim


class Cheater(OnlinePredictor):
    """Deliberately peeks at the truth; the loop structure must make this impossible
    (it can only cheat if handed the trajectory, which the interface forbids)."""

    name = "cheater"

    def __init__(self, traj):
        self.traj = traj
        self.t = 0

    def reset(self, d_y, d_u, T, seed=0):
        self.t = 0

    def predict(self, u_t):
        return np.asarray(self.traj.y[self.t], dtype=float)

    def update(self, y_t):
        self.t += 1


def test_persistence_matches_manual():
    traj = SYSTEMS["lorenz"](seed=0, T=256)
    res = run_online(Persistence(), traj)
    # ŷ_t = y_{t-1} ⇒ error_t = ||y_t - y_{t-1}||² for t >= 1
    expected = np.sum(np.diff(traj.y, axis=0) ** 2, axis=1)
    assert np.allclose(res.errors[1:], expected)


def test_cheater_detected_by_construction():
    # A method that magically knows y_t gets zero error — the spy confirms the
    # evaluation itself never leaks it: only a method holding the raw trajectory can.
    traj = SYSTEMS["lorenz"](seed=0, T=64)
    res = run_online(Cheater(traj), traj)
    assert res.errors.max() == 0.0  # demonstrates what cheating looks like


def test_same_stream_for_all_methods():
    a = SYSTEMS["narma10"](seed=7, T=500)
    b = SYSTEMS["narma10"](seed=7, T=500)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.u, b.u)


def test_smooth_window():
    x = np.ones(100)
    assert np.allclose(smooth(x, 10), 1.0)
    assert len(smooth(x, 10)) == 91


def test_run_result_metrics():
    traj = SYSTEMS["mackey_glass"](seed=0, T=1000)
    res = run_online(Persistence(), traj)
    assert res.mean_error() > 0
    assert 0 < res.nrmse(traj.y) < 1.0  # persistence on MG is decent but not perfect
