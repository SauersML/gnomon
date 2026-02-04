"""
Aggregate AUC across divergence levels and plot divergence vs bottleneck.

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
            metrics_path = Path(f"{sim_prefix}_metrics.csv")
            if not metrics_path.exists():
                missing_files.append(metrics_path.name)
                continue
            df = pd.read_csv(metrics_path)
            if "Method" not in df.columns or "AUC_overall" not in df.columns:
                missing_methods[metrics_path.name] = sorted(set(df.get("Method", [])))
                continue
            for _, row in df.iterrows():
                rows.append(
                    {
                        "scenario": sim_name,
                        "gens": gens,
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
    df = df.sort_values(["method", "scenario", "gens"])
    csv_path = Path(f"{args.out_prefix}.csv")
    df.to_csv(csv_path, index=False)

    methods = sorted(df["method"].unique())
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.get_cmap("tab10")
    color_map = {method_name: colors(i % 10) for i, method_name in enumerate(methods)}
    line_styles = {"divergence": "-", "bottleneck": "--"}

    for method_name in methods:
        sub_df = df[df["method"] == method_name]
        for scenario in SCENARIOS:
            sub = sub_df[sub_df["scenario"] == scenario].sort_values("gens")
            if sub.empty:
                continue
            ax.plot(
                sub["gens"],
                sub["auc"],
                marker="o",
                linewidth=2,
                color=color_map[method_name],
                linestyle=line_styles.get(scenario, "-"),
                label=f"{method_name} ({scenario})",
            )

    ax.set_xlabel("Divergence (generations, log scale)")
    ax.set_ylabel("AUC (overall)")
    ax.set_title("AUC vs Divergence for Two-Population Sims")
    _set_gens_scale(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    plot_path = Path(f"{args.out_prefix}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
