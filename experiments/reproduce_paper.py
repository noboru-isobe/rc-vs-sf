"""Reproduction gate: qualitative comparison against the figures of arXiv:2508.11990.

Runs the paper's systems with the paper's methods and checks the qualitative
claims numerically. Produces figs/reproduce/*.png for visual comparison with
the paper's figures (downloaded reference: arxiv.org/html/2508.11990v1/figs/).

Settings matched to the paper's figures (not the earlier misread ones):
- trajectory lengths: ~1e4 for nonlinear systems (the "T = 1024" in the paper is
  the Hankel filter horizon, not the trajectory length), 2e4 for the permutation
  LDS (paper runs to 5e4; convergence is visible by ~1e4), 5e3 for the Gaussian LDS
- observations standardized per trajectory (paper loss scales are O(1))
- smoothing windows 100 (Gaussian LDS) / 1000 (everything else)
- linear y-axis for the LDS figures, log for the nonlinear ones (as in the paper)

Usage: uv run python experiments/reproduce_paper.py [--seeds 4] [--fast]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rcsf.evaluation import run_online, smooth
from rcsf.methods import EDMD, Persistence, SFeDMD, SpectralFilter
from rcsf.plotting import style_for
from rcsf.systems import SYSTEMS
from rcsf.systems.base import Trajectory

warnings.filterwarnings("ignore", category=RuntimeWarning)   # thinplate log(0) flood
warnings.filterwarnings("ignore", category=FutureWarning)

FIGDIR = Path(__file__).resolve().parent.parent / "figs" / "reproduce"

# The linear checks use optimizer="rls" (ONS-equivalent second-order updates,
# which the paper lists as an alternative): COCOB converges in the same
# direction but several times slower than the paper's curves (see README);
# both variants are plotted for transparency.
LINEAR_METHODS = {
    "sf (vanilla, rls)": lambda: SpectralFilter(use_obs_filters=False, optimizer="rls", name="sf"),
    "sf+obs (rls)": lambda: SpectralFilter(use_obs_filters=True, optimizer="rls"),
    "sf (vanilla, cocob)": lambda: SpectralFilter(use_obs_filters=False, loss="squared", name="sf"),
    "sf+obs (cocob)": lambda: SpectralFilter(use_obs_filters=True, loss="squared"),
}
NONLINEAR_METHODS = {
    "persistence": Persistence,
    "sf": lambda: SpectralFilter(loss="squared"),   # autonomous: P/N blocks only
    "edmd": EDMD,
    "sfedmd": SFeDMD,
}
# Langevin is noise-dominated: the lifted ridge readout needs strong
# regularization there (the per-system tuning protocol handles this in the
# main experiments; the gate pins the tuned value).
LANGEVIN_METHODS = {**NONLINEAR_METHODS, "sfedmd": lambda: SFeDMD(ridge=0.1)}


def standardize(traj: Trajectory) -> Trajectory:
    """Per-trajectory standardization of y, applied identically to all methods.

    (Slightly acausal — full-trajectory statistics — but uniform across methods;
    the paper's O(1) loss scales indicate the same normalization.)
    """
    mu, sd = traj.y.mean(axis=0), traj.y.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return Trajectory(y=(traj.y - mu) / sd, u=traj.u, name=traj.name)


def run_suite(system: str, methods: dict, seeds: range, T: int):
    """Mean error curves over seeds plus seed-0 predictions for overlay plots."""
    curves, preds = {}, {}
    traj0 = standardize(SYSTEMS[system](seed=seeds[0], T=T))
    for label, factory in methods.items():
        results = [
            run_online(factory(), standardize(SYSTEMS[system](seed=s, T=T)), seed=s)
            for s in seeds
        ]
        curves[label] = np.stack([r.errors for r in results]).mean(axis=0)
        preds[label] = results[0].predictions
    return curves, preds, traj0


def final_mean(curve: np.ndarray, fraction: float = 0.1) -> float:
    n = max(1, int(len(curve) * fraction))
    return float(curve[-n:].mean())


def plot_curves(curves: dict, title: str, window: int, path: Path, log: bool = True):
    plt.figure(figsize=(7, 4.5))
    plot = plt.semilogy if log else plt.plot
    for label, curve in curves.items():
        color, ls = style_for(label)
        plot(smooth(curve, window), label=label, color=color, linestyle=ls, lw=1.8)
    plt.xlabel("step")
    plt.ylabel(f"loss (smoothed, window {window})")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def plot_overlay(preds: dict, traj, title: str, path: Path, last: int = 100, dim: int = 0):
    """Ground truth vs one-step predictions, final `last` steps (paper Fig. 6 style)."""
    t = np.arange(traj.T - last, traj.T)
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, traj.y[-last:, dim], "k-", lw=2, label="ground truth")
    for label, p in preds.items():
        if label == "persistence":
            continue
        color, ls = style_for(label)
        plt.plot(t, p[-last:, dim], linestyle=ls, alpha=0.85, label=label, color=color)
    plt.xlabel("step")
    plt.ylabel(f"y[{dim}] (standardized)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--fast", action="store_true",
                        help="divide trajectory lengths by 10 (smoke test only)")
    args = parser.parse_args()
    seeds = range(args.seeds)
    div = 10 if args.fast else 1
    results = []

    print("== Linear systems (paper Fig. 'Random LDS' / 'Permutation LDS') ==", flush=True)
    perm, _, _ = run_suite("permutation_lds", LINEAR_METHODS, seeds, 20000 // div)
    plot_curves(perm, "Permutation LDS (16x16, autonomous)", 1000 // div,
                FIGDIR / "permutation_lds.png", log=False)
    v, o = final_mean(perm["sf (vanilla, rls)"]), final_mean(perm["sf+obs (rls)"])
    results.append(check("permutation: SF+obs -> ~0, vanilla stalls O(1)",
                         o < 0.1 * v and o < 0.1, f"vanilla={v:.4g} obs={o:.4g}"))

    gauss, _, _ = run_suite("gaussian_lds", LINEAR_METHODS, seeds, 5000 // div)
    plot_curves(gauss, "Random LDS (128x128) w/ disturbances", 100,
                FIGDIR / "gaussian_lds.png", log=False)
    v, o = final_mean(gauss["sf (vanilla, rls)"]), final_mean(gauss["sf+obs (rls)"])
    results.append(check("gaussian+noise: SF+obs floor below vanilla floor",
                         o < 0.5 * v, f"vanilla={v:.4g} obs={o:.4g}"))

    print("== Nonlinear systems (paper Figs. Lorenz / pendulum / Langevin) ==", flush=True)
    T_nl = 10000 // div
    for system in ["lorenz", "double_pendulum"]:
        full, fpreds, ftraj = run_suite(system, NONLINEAR_METHODS, seeds, T_nl)
        plot_curves(full, f"{system} (full obs)", 1000 // div, FIGDIR / f"{system}_full.png")
        plot_overlay(fpreds, ftraj, f"{system} (full obs): truth vs predictions",
                     FIGDIR / f"{system}_full_overlay.png")
        f = {k: final_mean(c) for k, c in full.items()}
        results.append(check(f"{system} full: SFeDMD best",
                             f["sfedmd"] <= 1.2 * min(f["sf"], f["edmd"], f["persistence"]),
                             str({k: round(v, 6) for k, v in f.items()})))
        results.append(check(f"{system} full: eDMD beats persistence",
                             f["edmd"] < f["persistence"],
                             f"edmd={f['edmd']:.4g} pers={f['persistence']:.4g}"))

        part, ppreds, ptraj = run_suite(f"{system}_partial", NONLINEAR_METHODS, seeds, T_nl)
        plot_curves(part, f"{system} (partial obs)", 1000 // div,
                    FIGDIR / f"{system}_partial.png")
        plot_overlay(ppreds, ptraj, f"{system} (partial obs): truth vs predictions",
                     FIGDIR / f"{system}_partial_overlay.png")
        p = {k: final_mean(c) for k, c in part.items()}
        results.append(check(f"{system} partial: eDMD degrades toward persistence",
                             p["edmd"] > 0.3 * p["persistence"],
                             f"edmd={p['edmd']:.4g} pers={p['persistence']:.4g}"))
        results.append(check(f"{system} partial: filtered methods beat eDMD",
                             min(p["sf"], p["sfedmd"]) < p["edmd"],
                             str({k: round(v, 6) for k, v in p.items()})))

    print("== Langevin (noise floor) ==", flush=True)
    raw = SYSTEMS["langevin"](seed=0, T=T_nl)
    eta = 0.01
    sd = raw.y.std(axis=0)
    floor = float(np.sum(2 * eta / sd**2))   # E||noise||^2 after standardization
    lang, _, _ = run_suite("langevin", LANGEVIN_METHODS, seeds, T_nl)
    plot_curves(lang, "Langevin (d=64)", 1000 // div, FIGDIR / "langevin.png")
    fl = {k: final_mean(c) for k, c in lang.items()}
    results.append(check("langevin: all methods sit at the noise floor",
                         all(0.8 * floor < v < 3 * floor for v in fl.values()),
                         f"floor={floor:.3g} " + str({k: round(v, 3) for k, v in fl.items()})))

    print(f"\n{sum(results)}/{len(results)} checks passed; figures in {FIGDIR}", flush=True)
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
