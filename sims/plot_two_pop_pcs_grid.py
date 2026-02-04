"""
Build a mega-plot of PC plots for two-pop simulations.

Usage:
  python sims/plot_two_pop_pcs_grid.py
"""
from __future__ import annotations

from pathlib import Path
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

GENS_LEVELS = [0, 20, 50, 100, 500, 1000, 5000, 10_000]
SIMS = ["divergence", "bottleneck"]


def _collect_images() -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for sim in SIMS:
        for gens in GENS_LEVELS:
            name = f"{sim}_g{gens}_pcs.png"
            path = Path(name)
            if path.exists():
                items.append((f"{sim} g{gens}", path))
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

    # Build a lookup for quick access
    img_map = {p.name: (label, p) for label, p in items}

    for r, sim in enumerate(SIMS):
        for c, gens in enumerate(GENS_LEVELS):
            ax = axes[r][c]
            name = f"{sim}_g{gens}_pcs.png"
            if name in img_map:
                _, path = img_map[name]
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.set_title(f"{sim} g{gens}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=10)
                ax.set_title(f"{sim} g{gens}", fontsize=10)
            ax.axis("off")

    fig.suptitle("Two-Population PC Plots", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = Path("two_pop_pcs_mega.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
