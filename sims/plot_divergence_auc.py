"""
Aggregate AUC across divergence levels and plot divergence vs bottleneck.

Usage:
  python sims/plot_divergence_auc.py --method "Raw PGS"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

GENS_LEVELS = [0, 20, 50, 100, 500, 1000, 5000, 10_000]


def _load_auc(sim_prefix: str, method: str) -> float | None:
    metrics_path = Path(f"{sim_prefix}_metrics.csv")
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    row = df[df["Method"] == method]
    if row.empty:
        return None
    return float(row.iloc[0]["AUC_overall"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="Raw PGS", help="Method name from metrics table")
    parser.add_argument("--out-prefix", default="divergence_bottleneck_auc", help="Output prefix")
    args = parser.parse_args()

    rows = []
    for sim_name in ("divergence", "bottleneck"):
        for gens in GENS_LEVELS:
            sim_prefix = f"{sim_name}_g{gens}"
            auc = _load_auc(sim_prefix, args.method)
            if auc is None:
                continue
            rows.append({"scenario": sim_name, "gens": gens, "auc": auc})

    if not rows:
        raise SystemExit("No metrics found for divergence/bottleneck sims.")

    df = pd.DataFrame(rows).sort_values(["scenario", "gens"])
    csv_path = Path(f"{args.out_prefix}.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario].sort_values("gens")
        ax.plot(sub["gens"], sub["auc"], marker="o", linewidth=2, label=scenario)

    ax.set_xlabel("Divergence (generations)")
    ax.set_ylabel(f"AUC (method: {args.method})")
    ax.set_title("AUC vs Divergence for Two-Population Sims")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = Path(f"{args.out_prefix}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
