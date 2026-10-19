"""
replot_complete_by_config.py - redraw figures/complete_by_config.png with corrected footnote wording.

Added after the fresh_v1 run (see EXPERIMENT_PROTOCOL.md section 10, operational log). The drawing code
is copied from src/reporting.py (_figures, first figure), which is frozen and left unchanged. The only
difference is the footnote: "Completable ceiling" becomes "Completable (chunk-fit upper bound)". It reads
aggregated/summary.csv and computes nothing new. The original figure is kept; the new one is written next
to it as complete_by_config_upper_bound.png.

With --footnote-original it draws the original footnote instead, so the output can be compared with the
figure the analysis step wrote.

Usage (from experiment/):
  python replot_complete_by_config.py --run results/fresh_v1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

NEW_NOTE = "Points: mean with 95% paper-bootstrap CI. Dashes: Completable (chunk-fit upper bound)."
OLD_NOTE = "Points: mean with 95% paper-bootstrap CI. Dashes: Completable ceiling."


def draw(run_dir: Path, out: Path, note: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "aggregated" / "summary.csv")
    colors = {0: "#4a4a4a", 10: "#2a78d6", 20: "#eb6834"}
    fig, axes = plt.subplots(1, len(cfg["modes"]), figsize=(5 * len(cfg["modes"]), 4), sharey=True)
    axes = np.atleast_1d(axes)
    sizes = cfg["chunking"]["sizes"]
    for ax, mode in zip(axes, cfg["modes"]):
        s = summary[summary["mode"] == mode]
        for j, pct in enumerate(cfg["chunking"]["overlap_pct"]):
            g = s[s.overlap_pct == pct].set_index("chunk_size").loc[sizes]
            x = np.arange(len(sizes)) + (j - 1) * 0.12
            ax.errorbar(x, g.complete, yerr=[g.complete - g.complete_ci_low, g.complete_ci_high - g.complete],
                        fmt="o", color=colors.get(int(pct), "k"), capsize=3, label=f"{pct:.0f}% overlap")
            ax.plot(x, g.completable, "_", color=colors.get(int(pct), "k"), ms=12, alpha=0.6)
        ax.set_xticks(np.arange(len(sizes)), [str(z) for z in sizes])
        ax.set_xlabel("Chunk size (tokens)")
        m = cfg["modes"][mode]
        desc = (f"nominal {m['budget']:,}-token budget, K = " + " / ".join(str(m['budget'] // z) for z in sizes)
                if m["type"] == "token_budget" else f"fixed K = {m['k']}")
        ax.set_title(f"{mode.capitalize()}: {desc}", fontsize=9, loc="left")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Complete@K (mean of per-paper means)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.text(0.01, 0.01, note, fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out")
    ap.add_argument("--footnote-original", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run)
    out = Path(args.out) if args.out else run_dir / "figures" / "complete_by_config_upper_bound.png"
    draw(run_dir, out, OLD_NOTE if args.footnote_original else NEW_NOTE)
    print(out)


if __name__ == "__main__":
    main()
