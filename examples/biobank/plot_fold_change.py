#!/usr/bin/env python3
"""Same bars as `plot_all_results.py` but on a fold-change scale.

C-statistic = 0.5 is the no-skill floor, so the meaningful quantity to
compare is (C - 0.5) -- the lift above chance. We plot the ratio

    fold = (GAM_C - 0.5) / (Baseline_C - 0.5)

so 1x = baseline, 2x = the GAM doubled the above-chance discrimination,
etc. Same layout, same colors, same banding as the absolute-delta plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plot_all_results import DISEASE_COLOR, REGIME_SHADE, ROWS

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
})

C_FLOOR = 0.5


def row_fold(row: tuple) -> float:
    gam, base = float(row[3]), float(row[4])
    return (gam - C_FLOOR) / (base - C_FLOOR)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 10.8), dpi=160)
    n = len(ROWS)
    y = np.arange(n)[::-1]
    folds = np.array([row_fold(r) for r in ROWS])
    colors = [DISEASE_COLOR[r[0]] for r in ROWS]

    # 1) regime bands behind everything
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ROWS[j + 1][0] == ROWS[i][0] and ROWS[j + 1][1] == ROWS[i][1]:
            j += 1
        top = n - i - 0.5
        bot = n - j - 1.5
        ax.axhspan(bot, top, color=REGIME_SHADE[ROWS[i][1]], zorder=-2)
        i = j + 1

    # 2) thin divider between diseases
    for i, r in enumerate(ROWS):
        if i > 0 and r[0] != ROWS[i - 1][0]:
            ax.axhline(n - i - 0.5, color="#888", lw=0.9, zorder=1)

    # 3) bars drawn relative to the 1.0x baseline line
    bar_left = np.full(n, 1.0)
    bar_width = folds - 1.0
    ax.barh(y, bar_width, left=bar_left, color=colors, edgecolor="white",
            linewidth=1.0, height=0.72, zorder=2)

    # 4) baseline reference line at 1.0x
    ax.axvline(1.0, color="#222", lw=1.0, zorder=3)

    # 5) fold value at the bar's outer end
    xmax = max(folds.max() * 1.18, 1.6)
    for i, f in enumerate(folds):
        ax.text(f + (xmax - 1.0) * 0.012, n - 1 - i,
                f"{f:.2f}x",
                fontsize=9.5, color="#1a1a1a", va="center", ha="left")

    # 6) y-tick labels = fold name only
    ax.set_yticks(y)
    ax.set_yticklabels([r[2] for r in ROWS], fontsize=10)
    ax.tick_params(axis="y", length=0)

    # 7) disease labels on the right
    diseases_in_order: list[str] = []
    for r in ROWS:
        if not diseases_in_order or diseases_in_order[-1] != r[0]:
            diseases_in_order.append(r[0])
    for d in diseases_in_order:
        idxs = [i for i, r in enumerate(ROWS) if r[0] == d]
        y_mid = n - 1 - (idxs[0] + idxs[-1]) / 2
        ax.text(1.025, y_mid, d, transform=ax.get_yaxis_transform(),
                fontsize=14, color=DISEASE_COLOR[d], ha="left", va="center",
                weight="bold", rotation=-90)

    ax.set_xlim(0.92, xmax)
    ax.set_xlabel(
        "Fold change in above-chance C-statistic   "
        "((GAM − 0.5) / (Z_norm2 Cox baseline − 0.5))",
        fontsize=11.5, labelpad=10,
    )
    tick_lo = 1.0
    tick_hi = np.ceil(xmax * 10) / 10
    ax.set_xticks(np.arange(tick_lo, tick_hi + 0.001, 0.2))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.grid(axis="x", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, n - 0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "all_results_fold.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
