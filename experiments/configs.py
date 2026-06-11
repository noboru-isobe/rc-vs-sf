"""Experiment configuration: systems, methods, and hyperparameter grids.

The grid sizes are deliberately asymmetric — SF with COCOB is close to
parameter-free while the ESN has the classic reservoir knobs. That asymmetry is
itself part of the comparison and is reported alongside the results.
"""

from __future__ import annotations

from rcsf.methods import EDMD, ESN, Persistence, SFeDMD, SpectralFilter

# Trajectory lengths follow the paper's figures (~1e4 steps; the permutation
# LDS converges more slowly and gets 2e4).
SYSTEM_T = {
    "gaussian_lds": 5000,
    "permutation_lds": 20000,
    "lorenz": 10000,
    "lorenz_partial": 10000,
    "double_pendulum": 10000,
    "double_pendulum_partial": 10000,
    "langevin": 10000,
    "narma10": 10000,
    "mackey_glass": 10000,
}

TEST_SEEDS = tuple(range(12))

SMOOTH_WINDOW = {"gaussian_lds": 100}          # default 1000 otherwise


def method_grids(system: str, quick: bool = False) -> dict:
    """{method_name: (factory, grid)} — factory(**params) -> OnlinePredictor.

    quick=True shrinks the grids (preliminary passes); the full grids are for
    the final 12-seed runs.
    """
    autonomous = system in {"permutation_lds", "lorenz", "lorenz_partial",
                            "double_pendulum", "double_pendulum_partial",
                            "langevin", "mackey_glass"}
    grids = {
        "persistence": (lambda: Persistence(), {}),
        "sf": (
            lambda **p: SpectralFilter(**p),
            {"optimizer": ["cocob", "rls"],
             "loss": ["norm", "squared"]},   # loss is ignored by the rls path
        ),
        "edmd": (lambda **p: EDMD(**p), {"n_centers": [20]}),
        "sfedmd": (
            lambda **p: SFeDMD(**p),
            {"ridge": [1e-6, 1e-3, 1e-1]},
        ),
        "esn-rls": (
            lambda **p: ESN(readout="rls", **p),
            {"sr": [0.9, 1.25] if quick else [0.9, 1.1, 1.25],
             "lr_leak": [0.3, 1.0],
             "input_scaling": [0.01, 0.1] if quick else [0.01, 0.1, 1.0],
             "forgetting": [0.999] if quick else [0.999, 1.0]},
        ),
        "esn-lms": (
            lambda **p: ESN(readout="lms", **p),
            {"sr": [0.9] if quick else [0.9, 1.1],
             "lr_leak": [0.3, 1.0],
             "input_scaling": [0.01, 0.1],
             "lms_rate": [1e-3, 1e-2] if quick else [1e-4, 1e-3, 1e-2]},
        ),
    }
    if not autonomous:
        # input-driven systems also get the vanilla (input-filter only) SF
        grids["sf-vanilla"] = (
            lambda **p: SpectralFilter(use_obs_filters=False, name="sf-vanilla", **p),
            {"optimizer": ["cocob", "rls"], "loss": ["squared"]},
        )
    return grids
