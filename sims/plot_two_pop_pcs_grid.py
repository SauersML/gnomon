"""
Build a mega-plot of PC plots for two-pop simulations.

Usage:
  python sims/plot_two_pop_pcs_grid.py
"""
from __future__ import annotations

from pathlib import Path
import re
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

GENS_LEVELS = [0, 20, 50, 100, 500, 1000, 5000, 10_000]
SIMS = ["divergence", "bottleneck"]


def _pick_seeded_plot(sim: str, gens: int) -> Path | None:
    pattern = f"{sim}_g{gens}_s*_pcs.png"
    candidates = sorted(Path(".").glob(pattern))
    if not candidates:
        legacy = Path(f"{sim}_g{gens}_pcs.png")
        return legacy if legacy.exists() else None
    seed_re = re.compile(r"_s(\d+)_pcs\.png$")
    def _seed_key(p: Path) -> int:
        match = seed_re.search(p.name)
        return int(match.group(1)) if match else 1_000_000
    return sorted(candidates, key=_seed_key)[0]


def _collect_images() -> dict[tuple[str, int], Path]:
    items: dict[tuple[str, int], Path] = {}
    for sim in SIMS:
        for gens in GENS_LEVELS:
            path = _pick_seeded_plot(sim, gens)
            if path is not None:
                items[(sim, gens)] = path
    return items


def main() -> None:
    items = _collect_images()
    if not items:
        raise SystemExit("No two-pop PC plots found to combine.")

    # Fixed grid: 2 rows (sims) x 8 columns (gens)
    nrows = len(SIMS)
    ncols = len(GENS_LEVELS)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.0 * ncols, 3.0 * nrows))

    # axes can be 2D or 1D depending on layout
    if nrows == 1:
        axes = [axes]

    for r, sim in enumerate(SIMS):
        for c, gens in enumerate(GENS_LEVELS):
            ax = axes[r][c]
            path = items.get((sim, gens))
            if path is not None:
                img = mpimg.imread(path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
            ax.axis("off")

    # Column labels (gens)
    for c, gens in enumerate(GENS_LEVELS):
        axes[0][c].set_title(f"g{gens}", fontsize=11, fontweight="bold", pad=6)

    fig.suptitle("Two-Population PC Plots (Seeded Example)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0.04, 0.02, 1, 0.93])

    # Row labels (scenarios) after layout for accurate positioning
    for r, sim in enumerate(SIMS):
        pos = axes[r][0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(0.015, y_center, sim, fontsize=12, rotation=90, va="center")
    out_path = Path("two_pop_pcs_mega.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
