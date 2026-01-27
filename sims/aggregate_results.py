"""
Aggregate per-simulation metrics and significance tests into a single TSV
and generate a combined p-value plot.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SIM_NAMES = ["confounding", "portability", "sample_imbalance"]


def load_metrics() -> pd.DataFrame:
    frames = []
    for sim in SIM_NAMES:
        path = Path(f"{sim}_metrics.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "Simulation", sim)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_pvalues() -> pd.DataFrame:
    frames = []
    for sim in SIM_NAMES:
        path = Path(f"{sim}_significance_tests.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "Simulation", sim)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_pvalues(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    # Keep AUC p-values only
    df = df.copy()
    df["Comparison"] = df["Method_1"] + " vs " + df["Method_2"]
    # Sort for stable plotting
    df = df.sort_values(["Simulation", "Comparison"])

    sims = df["Simulation"].unique().tolist()
    comparisons = df["Comparison"].unique().tolist()

    fig, ax = plt.subplots(figsize=(12, max(6, len(comparisons) * 0.35)))

    y_positions = np.arange(len(comparisons))
    width = 0.8 / max(1, len(sims))

    for i, sim in enumerate(sims):
        subset = df[df["Simulation"] == sim].set_index("Comparison")
        pvals = [subset.loc[c, "AUC_p_value"] if c in subset.index else np.nan for c in comparisons]
        ax.barh(y_positions + i * width, pvals, height=width, label=sim)

    ax.axvline(0.05, color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax.set_yticks(y_positions + width * (len(sims) - 1) / 2)
    ax.set_yticklabels(comparisons)
    ax.set_xlabel("AUC p-value")
    ax.set_title("Pairwise AUC p-values (all methods, all simulations)")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    metrics = load_metrics()
    if not metrics.empty:
        metrics.to_csv("all_methods_metrics.tsv", sep="\t", index=False)

    pvals = load_pvalues()
    plot_pvalues(pvals, Path("all_methods_pvalues.png"))


if __name__ == "__main__":
    main()
