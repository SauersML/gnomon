#!/usr/bin/env python3
"""Single-panel bar plot of GAM-vs-baseline Δ test-C across every biobank fold.

One row per fold. Bars colored by disease (Tol-vibrant, colorblind-safe).
Background banded by evaluation regime (random / leave-one-site-out /
leave-one-region-out). Drop folds with events < 1000 (noise floor).

Update the `ROWS` table after each `examples/biobank/run.sh` cycle and
re-run. If CI columns are present, horizontal intervals are drawn around the
delta. Output: ./all_results.png next to this script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
})

# (disease, regime, fold_label, GAM_test_C, base_test_C, test_events,
#  optional_delta_ci_low, optional_delta_ci_high)
# Events>=1000 only -- folds below that are noise (SE on C ~ 0.5/sqrt(events)).
ROWS = [
    ("COPD",         "Random", "Random",    0.5805, 0.5430,  5317),
    ("COPD",         "Site",   "site 321",  0.5781, 0.5659,  5003),
    ("COPD",         "Site",   "site 699",  0.6353, 0.5695,  3313),
    ("COPD",         "Site",   "site 305",  0.6121, 0.5858,  1791),
    ("COPD",         "Site",   "site 638",  0.5792, 0.5651,  2282),
    ("COPD",         "Region", "West",      0.6055, 0.5867,  6945),
    ("COPD",         "Region", "Northeast", 0.6245, 0.5704,  8624),
    ("COPD",         "Region", "Midwest",   0.6394, 0.5663,  6366),
    ("COPD",         "Region", "South",     0.6008, 0.5747,  4626),

    ("Hypertension", "Random", "Random",    0.5937, 0.5393, 26161),
    ("Hypertension", "Site",   "site 321",  0.6075, 0.5402, 19831),
    ("Hypertension", "Site",   "site 699",  0.6024, 0.5477, 13383),
    ("Hypertension", "Site",   "site 305",  0.5996, 0.5484,  9974),
    ("Hypertension", "Site",   "site 195",  0.6485, 0.5549,  3988),
    ("Hypertension", "Site",   "site 638",  0.5870, 0.5338,  7976),
    ("Hypertension", "Region", "West",      0.6329, 0.5527, 33550),
    ("Hypertension", "Region", "Northeast", 0.6173, 0.5428, 39455),
    ("Hypertension", "Region", "Midwest",   0.5745, 0.5434, 32722),
    ("Hypertension", "Region", "South",     0.6278, 0.5355, 25007),

    ("Obesity",      "Random", "Random",    0.6341, 0.5901, 18486),
    ("Obesity",      "Site",   "site 321",  0.6693, 0.6202, 13042),
    ("Obesity",      "Site",   "site 699",  0.6647, 0.6323,  9280),
    ("Obesity",      "Site",   "site 305",  0.6896, 0.6202,  6392),
    ("Obesity",      "Site",   "site 195",  0.7048, 0.6213,  3050),
    ("Obesity",      "Site",   "site 638",  0.6328, 0.5873,  5572),
    ("Obesity",      "Region", "West",      0.7086, 0.6472, 21517),
    ("Obesity",      "Region", "Northeast", 0.6709, 0.6138, 27918),
    ("Obesity",      "Region", "Midwest",   0.6353, 0.6158, 24599),
    ("Obesity",      "Region", "South",     0.6523, 0.5942, 18347),
]

# Tol "vibrant" -- colorblind-safe; only used for the bar (not background).
DISEASE_COLOR = {
    "COPD":         "#0077BB",
    "Hypertension": "#EE7733",
    "Obesity":      "#009988",
}
# Neutral grey shades per regime: subtle background stripes that don't fight
# with the bar color.
REGIME_SHADE = {
    "Random": "#ffffff",
    "Site":   "#f1f1f1",
    "Region": "#e2e2e2",
}


def row_delta(row: tuple) -> float:
    return float(row[3] - row[4])


def row_delta_ci(row: tuple) -> tuple[float, float] | None:
    if len(row) < 8 or row[6] is None or row[7] is None:
        return None
    return float(row[6]), float(row[7])


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 10.8), dpi=160)
    n = len(ROWS)
    y = np.arange(n)[::-1]
    deltas = np.array([row_delta(r) for r in ROWS])
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

    # 3) bars
    ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=1.0,
            height=0.72, zorder=2)
    for i, r in enumerate(ROWS):
        ci = row_delta_ci(r)
        if ci is None:
            continue
        lo, hi = ci
        center = row_delta(r)
        ax.errorbar(
            center,
            n - 1 - i,
            xerr=[[center - lo], [hi - center]],
            fmt="none",
            ecolor="#111",
            elinewidth=1.3,
            capsize=3.0,
            capthick=1.3,
            zorder=4,
        )

    # 4) zero reference line
    ax.axvline(0, color="#222", lw=1.0, zorder=3)

    # 5) Δ value at the right end of each bar
    xmax = max(0.10, deltas.max() * 1.18)
    for i, d in enumerate(deltas):
        ax.text(d + xmax * 0.012, n - 1 - i,
                f"+{d:.3f}" if d >= 0 else f"{d:.3f}",
                fontsize=9.5, color="#1a1a1a", va="center", ha="left")

    # 6) y-tick labels = fold name only
    ax.set_yticks(y)
    ax.set_yticklabels([r[2] for r in ROWS], fontsize=10)
    ax.tick_params(axis="y", length=0)

    # 7) sample size just left of the zero line, in dark grey
    for i, r in enumerate(ROWS):
        ax.text(-0.003, n - 1 - i, f"n = {r[5]:,}",
                fontsize=9, color="#444", ha="right", va="center")

    # 8) disease labels on the right, vertical (top-to-bottom), color-coded
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

    ax.set_xlim(-0.018, xmax * 1.20)
    ax.set_xlabel(
        "Δ held-out C-statistic   "
        "(Marginal-slope GAM  −  Z_norm2 + sex Cox PH baseline)",
        fontsize=11.5, labelpad=10,
    )
    ax.set_xticks(np.arange(0.00, xmax + 0.001, 0.02))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{x:+.2f}" if x else "0"
    ))
    ax.grid(axis="x", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, n - 0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "all_results.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
