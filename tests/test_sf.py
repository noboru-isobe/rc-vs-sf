import numpy as np

from rcsf.evaluation import run_online
from rcsf.methods import EDMD, Persistence, SFeDMD, SpectralFilter
from rcsf.systems import SYSTEMS


def _phase_means(errors, washout=100):
    early = errors[washout:washout + 200].mean()
    late = errors[-200:].mean()
    return early, late


def test_sf_learns_diagonal_lds():
    # Vanilla SF provably learns symmetric PSD LDS: loss must collapse,
    # judged relative to the output scale (COCOB converges but not instantly).
    traj = SYSTEMS["diagonal_lds"](seed=0, T=4000)
    res = run_online(SpectralFilter(use_obs_filters=False), traj, seed=0)
    early, late = _phase_means(res.errors)
    assert late < early * 0.1
    assert late < 0.01 * traj.y.var()
    # Vanishing-error behavior: each 1000-step chunk must improve on the last;
    # a plateau would indicate a bias bug rather than slow convergence.
    chunks = [res.errors[i:i + 1000].mean() for i in range(0, 4000, 1000)]
    assert all(b < a for a, b in zip(chunks, chunks[1:]))


def test_sf_obs_beats_vanilla_on_permutation():
    # Paper's headline qualitative result (Fig. 1): observation filtering is the
    # difference between learning and not learning the asymmetric permutation LDS.
    traj = SYSTEMS["permutation_lds"](seed=0, T=2000)
    vanilla = run_online(SpectralFilter(use_obs_filters=False, name="sf"), traj, seed=0)
    with_obs = run_online(SpectralFilter(use_obs_filters=True), traj, seed=0)
    assert with_obs.errors[-200:].mean() < 0.5 * vanilla.errors[-200:].mean()


def test_sf_runs_on_autonomous_lorenz():
    traj = SYSTEMS["lorenz"](seed=0, T=1024)
    res = run_online(SpectralFilter(), traj, seed=0)
    assert np.all(np.isfinite(res.errors))
    early, late = _phase_means(res.errors)
    assert late < early  # it should at least improve over time


def test_sf_ogd_variant_runs():
    traj = SYSTEMS["diagonal_lds"](seed=1, T=1000)
    res = run_online(SpectralFilter(optimizer="ogd", lr=1e-2, use_obs_filters=False),
                     traj, seed=0)
    assert np.all(np.isfinite(res.errors))


def test_edmd_beats_persistence_on_lorenz_full():
    traj = SYSTEMS["lorenz"](seed=0, T=1024)
    edmd = run_online(EDMD(), traj, seed=0)
    pers = run_online(Persistence(), traj, seed=0)
    assert edmd.errors[-200:].mean() < pers.errors[-200:].mean()


def test_sfedmd_beats_persistence_on_lorenz_full():
    traj = SYSTEMS["lorenz"](seed=0, T=1024)
    sfedmd = run_online(SFeDMD(), traj, seed=0)
    pers = run_online(Persistence(), traj, seed=0)
    assert np.all(np.isfinite(sfedmd.errors))
    assert sfedmd.errors[-200:].mean() < pers.errors[-200:].mean()
