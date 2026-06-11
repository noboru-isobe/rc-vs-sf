"""Figures and summary table from results/<system>/<method>.npz files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import smooth

# Color families: SF-side warm (reds), ESN-side cool (blues), baselines gray.
STYLES = {
    "sf": ("#d62728", "-"),          # red, solid
    "sf-vanilla": ("#ff9896", "--"),  # light red, dashed
    "sfedmd": ("#7f0000", "-."),      # dark red, dash-dot
    "esn-rls": ("#1f77b4", "-"),      # blue, solid
    "esn-lms": ("#9ecae1", "--"),     # light blue, dashed
    "edmd": ("#636363", "-"),         # dark gray, solid
    "persistence": ("#bdbdbd", ":"),  # light gray, dotted
}


def style_for(name: str) -> tuple[str, str]:
    """(color, linestyle) for a method label; tolerant to label variants
    like 'sf+obs (cocob)' or 'sf (vanilla, rls)'."""
    key = name.lower()
    if key in STYLES:
        return STYLES[key]
    if "vanilla" in key:
        return STYLES["sf-vanilla"][0], ("--" if "rls" in key or "(" not in key else ":")
    if key.startswith("sfedmd"):
        return STYLES["sfedmd"]
    if key.startswith("sf"):
        color = STYLES["sf"][0]
        return color, ("-" if "cocob" not in key else "--")
    if "lms" in key:
        return STYLES["esn-lms"]
    if "esn" in key or "rls" in key:
        return STYLES["esn-rls"]
    if "edmd" in key:
        return STYLES["edmd"]
    return "#999999", "-"


def load_results(system_dir: Path) -> dict:
    out = {}
    for f in sorted(system_dir.glob("*.npz")):
        data = np.load(f, allow_pickle=False)
        out[f.stem] = {
            "errors": data["errors"],
            "predictions0": data["predictions0"],
            "y0": data["y0"],
            "params": json.loads(str(data["params"])),
            "n_trainable": int(data["n_trainable"]),
            "washout": int(data["washout"]),
        }
    return out


def plot_loss_curves(system: str, results: dict, window: int, outpath: Path,
                     log: bool = True):
    plt.figure(figsize=(8, 5))
    for name, r in results.items():
        curve = smooth(r["errors"].mean(axis=0), window)
        color, ls = style_for(name)
        (plt.semilogy if log else plt.plot)(
            curve, label=name, color=color, linestyle=ls, lw=1.8)
    plt.xlabel("step")
    plt.ylabel(f"squared error (seed mean, window {window})")
    plt.title(system)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_overlay(system: str, results: dict, outpath: Path, last: int = 100,
                 dim: int = 0):
    y0 = next(iter(results.values()))["y0"]
    t = np.arange(len(y0) - last, len(y0))
    plt.figure(figsize=(8, 4.5))
    plt.plot(t, y0[-last:, dim], "k-", lw=2, label="ground truth")
    for name, r in results.items():
        if name == "persistence":
            continue
        color, ls = style_for(name)
        plt.plot(t, r["predictions0"][-last:, dim], linestyle=ls, alpha=0.85,
                 label=name, color=color)
    plt.xlabel("step")
    plt.ylabel(f"y[{dim}]")
    plt.title(f"{system}: truth vs one-step predictions (seed 0)")
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def summary_rows(system: str, results: dict) -> list[dict]:
    rows = []
    for name, r in results.items():
        errs = r["errors"]
        n_final = max(1, errs.shape[1] // 10)
        finals = errs[:, -n_final:].mean(axis=1)          # per-seed final means
        post = errs[:, r["washout"]:]
        y_var = np.var(r["y0"][r["washout"]:])
        rows.append({
            "system": system,
            "method": name,
            "final_mean": float(finals.mean()),
            "final_std": float(finals.std()),
            "nrmse": float(np.sqrt(post[:, -n_final:].mean()) / np.sqrt(y_var)),
            "n_trainable": r["n_trainable"],
            "params": r["params"],
        })
    return rows


def format_table(rows: list[dict]) -> str:
    lines = [
        "| system | method | final 10% error (mean±std) | NRMSE | trainable params | tuned config |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['method']} | {r['final_mean']:.3g} ± {r['final_std']:.2g} "
            f"| {r['nrmse']:.3g} | {r['n_trainable']} | {json.dumps(r['params'])} |"
        )
    return "\n".join(lines)
