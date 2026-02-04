"""
Aggregate AUC across divergence levels and plot divergence vs bottleneck.

Usage:
  python sims/plot_divergence_auc.py --method "BayesR + Raw"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

GENS_LEVELS = [0, 20, 50, 100, 500, 1000, 5000, 10_000]


def _normalize_method(method: str) -> str:
    # Backward-compat alias from old CLI usage.
    if method.strip().lower() == "raw pgs":
        return "BayesR + Raw"
    return method


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
    parser.add_argument(
        "--method",
        default="BayesR + Raw",
        help="Exact Method name from metrics table, e.g. 'BayesR + Raw'.",
    )
    parser.add_argument("--out-prefix", default="divergence_bottleneck_auc", help="Output prefix")
    args = parser.parse_args()
    method = _normalize_method(args.method)

    rows = []
    missing_files: list[str] = []
    missing_methods: dict[str, list[str]] = {}
    for sim_name in ("divergence", "bottleneck"):
        for gens in GENS_LEVELS:
            sim_prefix = f"{sim_name}_g{gens}"
            metrics_path = Path(f"{sim_prefix}_metrics.csv")
            if not metrics_path.exists():
                missing_files.append(metrics_path.name)
                continue
            auc = _load_auc(sim_prefix, method)
            if auc is None:
                df = pd.read_csv(metrics_path)
                available = sorted(set(df.get("Method", [])))
                if available:
                    missing_methods[metrics_path.name] = available
                continue
            rows.append({"scenario": sim_name, "gens": gens, "auc": auc})

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
            f"No metrics found for divergence/bottleneck sims using method '{method}'.\n{detail_msg}"
        )

    df = pd.DataFrame(rows).sort_values(["scenario", "gens"])
    csv_path = Path(f"{args.out_prefix}.csv")
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for scenario in df["scenario"].unique():
        sub = df[df["scenario"] == scenario].sort_values("gens")
        ax.plot(sub["gens"], sub["auc"], marker="o", linewidth=2, label=scenario)

    ax.set_xlabel("Divergence (generations)")
    ax.set_ylabel(f"AUC (method: {method})")
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
