"""
Aggregate mean AUC across divergence levels and plot divergence vs bottleneck.

Usage:
  python sims/plot_divergence_auc.py
  python sims/plot_divergence_auc.py --method "BayesR + Raw"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

GENS_LEVELS = [0, 20, 50, 100, 500, 1000, 5000, 10_000]
SCENARIOS = ("divergence", "bottleneck")


def _normalize_method(method: str) -> str:
    # Backward-compat alias from old CLI usage.
    if method.strip().lower() == "raw pgs":
        return "BayesR + Raw"
    return method


def _set_gens_scale(ax: plt.Axes) -> None:
    if 0 in GENS_LEVELS:
        ax.set_xscale("symlog", linthresh=1)
    else:
        ax.set_xscale("log")
    ax.set_xticks(GENS_LEVELS)
    ax.set_xticklabels([str(g) for g in GENS_LEVELS])


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "grid.color": "#d0d0d0",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 8,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default=None,
        help="Exact Method name from metrics table. Omit to plot all methods.",
    )
    parser.add_argument("--out-prefix", default="divergence_bottleneck_auc", help="Output prefix")
    args = parser.parse_args()
    method = _normalize_method(args.method) if args.method else None

    rows: list[dict[str, object]] = []
    missing_files: list[str] = []
    missing_methods: dict[str, list[str]] = {}
    for sim_name in SCENARIOS:
        for gens in GENS_LEVELS:
            sim_prefix = f"{sim_name}_g{gens}"
            pattern = f"{sim_prefix}_s*_metrics.csv"
            metrics_paths = sorted(Path(".").glob(pattern))
            if not metrics_paths:
                legacy = Path(f"{sim_prefix}_metrics.csv")
                if legacy.exists():
                    metrics_paths = [legacy]
            if not metrics_paths:
                missing_files.append(f"{sim_prefix}_metrics.csv")
                continue

            seed_re = re.compile(r"_s(\d+)_metrics\.csv$")
            for metrics_path in metrics_paths:
                df = pd.read_csv(metrics_path)
                if "Method" not in df.columns or "AUC_overall" not in df.columns:
                    missing_methods[metrics_path.name] = sorted(set(df.get("Method", [])))
                    continue
                seed_match = seed_re.search(metrics_path.name)
                seed_val = int(seed_match.group(1)) if seed_match else None
                for _, row in df.iterrows():
                    rows.append(
                        {
                            "scenario": sim_name,
                            "gens": gens,
                            "seed": seed_val,
                            "method": row["Method"],
                            "auc": row["AUC_overall"],
                        }
                    )

    if not rows:
        details = []
        if missing_files:
            details.append(
                "Missing metrics files: " + ", ".join(sorted(missing_files))
            )
        if missing_methods:
            sample = list(missing_methods.items())[:3]
            method_lines = [
                f"{name}: {', '.join(methods)}" for name, methods in sample
            ]
            details.append(
                "Method not found. Available methods (sample): " + " | ".join(method_lines)
            )
        detail_msg = "\n".join(details) if details else "No metrics files or methods found."
        raise SystemExit(
            "No metrics found for divergence/bottleneck sims.\n" + detail_msg
        )

    df = pd.DataFrame(rows)
    if method:
        df = df[df["method"] == method]
        if df.empty:
            available = sorted(set(pd.DataFrame(rows)["method"].tolist()))
            raise SystemExit(
                f"Method '{method}' not found. Available methods: {', '.join(available)}"
            )
    df = df.sort_values(["method", "scenario", "gens", "seed"])

    agg = (
        df.groupby(["scenario", "gens", "method"], as_index=False)
        .agg(
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            n_seeds=("auc", "count"),
        )
        .sort_values(["method", "scenario", "gens"])
    )

    _apply_style()
    csv_path = Path(f"{args.out_prefix}.csv")
    agg.to_csv(csv_path, index=False)

    methods = sorted(agg["method"].unique())
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.get_cmap("tab10")
    color_map = {method_name: colors(i % 10) for i, method_name in enumerate(methods)}
    line_styles = {"divergence": "-", "bottleneck": "--"}

    for method_name in methods:
        sub_df = agg[agg["method"] == method_name]
        for scenario in SCENARIOS:
            sub = sub_df[sub_df["scenario"] == scenario].sort_values("gens")
            if sub.empty:
                continue
            std = sub["std_auc"].fillna(0.0).to_numpy()
            ax.plot(
                sub["gens"],
                sub["mean_auc"],
                marker="o",
                linewidth=2,
                color=color_map[method_name],
                linestyle=line_styles.get(scenario, "-"),
                label=f"{method_name} ({scenario})",
            )
            if (std > 0).any():
                ax.fill_between(
                    sub["gens"],
                    sub["mean_auc"] - std,
                    sub["mean_auc"] + std,
                    color=color_map[method_name],
                    alpha=0.12,
                    linewidth=0,
                )

    ax.set_xlabel("Divergence (generations, log scale)")
    ax.set_ylabel("Mean AUC (overall)")
    ax.set_title("Mean AUC vs Divergence for Two-Population Sims")
    _set_gens_scale(ax)
    ax.grid(True, alpha=0.6)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    plot_path = Path(f"{args.out_prefix}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
