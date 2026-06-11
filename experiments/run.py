"""Main experiment runner: tune on validation seeds, evaluate on test seeds.

Results land in results/<system>/<method>.npz with keys:
  errors (n_seeds, T), predictions0 (T, d_y), y0 (T, d_y),
  params (json), n_trainable, washout

Usage:
  uv run python experiments/run.py --system lorenz                # one system
  uv run python experiments/run.py --all --seeds 12               # full sweep
  uv run python experiments/run.py --all --seeds 4 --scale 4      # quick pass
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import SYSTEM_T, TEST_SEEDS, method_grids

from rcsf.evaluation import run_online
from rcsf.systems import SYSTEMS
from rcsf.systems.base import Trajectory
from rcsf.tuning import tune, validation_stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_traj_factory(system: str, T: int):
    """Trajectory generator standardized with validation statistics (causal)."""
    raw = lambda seed: SYSTEMS[system](seed=seed, T=T)
    mu, sd = validation_stats(raw)

    def make(seed: int) -> Trajectory:
        t = raw(seed)
        return Trajectory(y=(t.y - mu) / sd, u=t.u, name=t.name)

    return make


def run_system(system: str, seeds, scale: int = 1, methods: list[str] | None = None,
               quick: bool = False):
    T = SYSTEM_T[system] // scale
    make = make_traj_factory(system, T)
    # Tuning runs on shortened trajectories (validation seeds are disjoint from
    # test seeds anyway); the selected config is evaluated at full length.
    T_tune = max(2000, T // 4)
    make_tune = make_traj_factory(system, T_tune)
    outdir = RESULTS / system
    outdir.mkdir(parents=True, exist_ok=True)
    grids = method_grids(system, quick=quick)
    if methods:
        grids = {k: v for k, v in grids.items() if k in methods}

    for name, (factory, grid) in grids.items():
        if grid:
            tr = tune(factory, grid, make_tune)
            params = _first_stable(factory, tr, make, system, name)
            print(f"[{system}] {name}: tuned {params} (val score {tr.best_score:.4g})",
                  flush=True)
        else:
            params = {}
        runs = [run_online(factory(**params), make(s), seed=s) for s in seeds]
        errors = np.stack([r.errors for r in runs])
        method_obj = factory(**params)
        traj0 = make(seeds[0])
        np.savez_compressed(
            outdir / f"{name}.npz",
            errors=errors,
            predictions0=runs[0].predictions,
            y0=traj0.y,
            params=json.dumps(params),
            n_trainable=_n_params(factory, params, traj0),
            washout=runs[0].washout,
        )
        final = errors[:, -max(1, errors.shape[1] // 10):].mean()
        print(f"[{system}] {name}: final-10% error {final:.4g}", flush=True)


def _first_stable(factory, tune_result, make, system, name):
    """Walk the tuned configs (best first) and return the first whose
    FULL-LENGTH run on a validation seed stays finite.

    Short-horizon tuning can select configs that only diverge later (e.g. RLS
    wind-up: forgetting < 1 on a poorly excited periodic system lets P blow up
    in the unexcited directions). Uses validation data only — no test leakage.
    """
    ranked = sorted(tune_result.all_scores, key=lambda ps: ps[1])
    for params, score in ranked:
        if not np.isfinite(score):
            continue
        res = run_online(factory(**params), make(1000), seed=1000)
        if np.all(np.isfinite(res.errors)):
            return params
        print(f"[{system}] {name}: {params} diverges at full length, "
              f"falling back", flush=True)
    raise RuntimeError(f"{system}/{name}: no stable config in the grid")


def _n_params(factory, params, traj):
    m = factory(**params)
    m.reset(traj.d_y, traj.d_u, traj.T, seed=0)
    return m.num_trainable_params()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=list(SYSTEM_T))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, default=len(TEST_SEEDS))
    parser.add_argument("--scale", type=int, default=1,
                        help="divide trajectory lengths by this factor")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", help="reduced HP grids")
    args = parser.parse_args()

    systems = list(SYSTEM_T) if args.all else [args.system]
    if not systems or systems == [None]:
        parser.error("specify --system or --all")
    seeds = list(TEST_SEEDS[: args.seeds])
    for system in systems:
        run_system(system, seeds, scale=args.scale, methods=args.methods,
                   quick=args.quick)


if __name__ == "__main__":
    main()
