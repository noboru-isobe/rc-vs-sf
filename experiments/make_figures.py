"""Render all figures and the summary table from results/.

Usage: uv run python experiments/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import SMOOTH_WINDOW, SYSTEM_T

from rcsf.plotting import (format_table, load_results, plot_loss_curves,
                           plot_overlay, summary_rows)

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = Path(__file__).resolve().parent.parent / "figs" / "main"

OVERLAY_SYSTEMS = {"lorenz_partial", "double_pendulum_partial"}


def main():
    all_rows = []
    for system in SYSTEM_T:
        sysdir = RESULTS / system
        if not sysdir.exists():
            continue
        results = load_results(sysdir)
        if not results:
            continue
        T = next(iter(results.values()))["errors"].shape[1]
        window = min(SMOOTH_WINDOW.get(system, 1000), max(10, T // 10))
        log = system not in {"gaussian_lds", "permutation_lds"}
        plot_loss_curves(system, results, window, FIGS / f"{system}_loss.png", log=log)
        if system in OVERLAY_SYSTEMS:
            plot_overlay(system, results, FIGS / f"{system}_overlay.png")
        all_rows.extend(summary_rows(system, results))
    table = format_table(all_rows)
    out = FIGS / "summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n")
    print(table)
    print(f"\nfigures in {FIGS}")


if __name__ == "__main__":
    main()
