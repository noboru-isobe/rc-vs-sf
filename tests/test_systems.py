import numpy as np
import pytest

from rcsf.systems import SYSTEMS


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_shapes_and_finiteness(name):
    traj = SYSTEMS[name](seed=0, T=512)
    assert traj.T == 512
    assert np.all(np.isfinite(traj.y)), f"{name} produced non-finite observations"
    if traj.u is not None:
        assert np.all(np.isfinite(traj.u))


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_deterministic_given_seed(name):
    a = SYSTEMS[name](seed=3, T=128)
    b = SYSTEMS[name](seed=3, T=128)
    assert np.array_equal(a.y, b.y)
    c = SYSTEMS[name](seed=4, T=128)
    assert not np.array_equal(a.y, c.y)


def test_marginal_stability_no_blowup():
    # Marginally stable linear systems must not blow up over long horizons.
    for name in ["gaussian_lds", "permutation_lds"]:
        traj = SYSTEMS[name](seed=1, T=4096)
        assert np.abs(traj.y).max() < 1e4, name


def test_langevin_bounded():
    traj = SYSTEMS["langevin"](seed=2, T=4096)
    assert np.abs(traj.y).max() < 50


def test_lorenz_on_attractor():
    traj = SYSTEMS["lorenz"](seed=0, T=2048)
    # z stays positive on the attractor after the transient; amplitude is O(10).
    z = traj.y[500:, 2]
    assert z.mean() > 10 and z.max() < 60


def test_pendulum_energy_drift_small():
    # RK4 at dt=0.01 should nearly conserve the angle-encoding norms.
    traj = SYSTEMS["double_pendulum"](seed=0, T=2048)
    s1 = traj.y[:, 0] ** 2 + traj.y[:, 1] ** 2
    assert np.allclose(s1, 1.0, atol=1e-8)


def test_narma10_bounded():
    traj = SYSTEMS["narma10"](seed=0, T=4000)
    assert np.abs(traj.y).max() < 2.0


def test_narma10_bounded_long_horizon_all_seeds():
    # The divergence guard must hold at the full experiment length for every
    # test and validation seed (the unguarded recursion blows up here).
    for seed in [*range(12), 1000, 1001, 1002]:
        traj = SYSTEMS["narma10"](seed=seed, T=10000)
        assert np.abs(traj.y).max() < 10.0, f"seed {seed}"


def test_mackey_glass_chaotic_band():
    traj = SYSTEMS["mackey_glass"](seed=0, T=4000)
    y = traj.y[:, 0]
    assert 0.2 < y.min() and y.max() < 1.5
    assert y.std() > 0.1  # oscillating, not collapsed to a fixed point
