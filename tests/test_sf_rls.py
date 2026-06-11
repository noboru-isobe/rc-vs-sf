import numpy as np

from rcsf.evaluation import run_online
from rcsf.methods import SpectralFilter
from rcsf.systems import SYSTEMS


def test_sf_rls_converges_fast_on_diagonal_lds():
    traj = SYSTEMS["diagonal_lds"](seed=0, T=2000)
    res = run_online(SpectralFilter(use_obs_filters=False, optimizer="rls"),
                     traj, seed=0)
    # Second-order updates should reach near-noiseless prediction quickly.
    assert res.errors[-500:].mean() < 1e-3 * traj.y.var()


def test_sf_rls_obs_filters_solve_permutation():
    # The paper's headline result at full strength: observation filtering with
    # second-order updates learns the asymmetric marginally stable system.
    traj = SYSTEMS["permutation_lds"](seed=0, T=4000)
    y_var = traj.y.var()
    obs = run_online(SpectralFilter(optimizer="rls"), traj, seed=0)
    van = run_online(SpectralFilter(optimizer="rls", use_obs_filters=False,
                                    name="sf"), traj, seed=0)
    assert obs.errors[-1000:].mean() < 0.2 * y_var
    assert obs.errors[-1000:].mean() < 0.3 * van.errors[-1000:].mean()


def test_sf_rls_runs_on_multidim_lorenz():
    traj = SYSTEMS["lorenz"](seed=0, T=1024)
    res = run_online(SpectralFilter(optimizer="rls"), traj, seed=0)
    assert np.all(np.isfinite(res.errors))
