"""
Aggregate per-simulation metrics and significance tests into a single TSV
and generate a combined AUC plot.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SIM_NAMES = ["confounding", "portability"]


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


def plot_auc_summary(df: pd.DataFrame, pvals: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    df = df.copy()
    df = df.sort_values(["Method", "Simulation"])

    df["TrainingMethod"] = df["Method"].str.split(" + ", n=1, regex=False).str[0]
    df["ApplicationMethod"] = df["Method"].str.split(" + ", n=1, regex=False).str[1]

    training_order = (
        df.groupby("TrainingMethod", as_index=False)["AUC_overall"]
        .mean()
        .sort_values("AUC_overall", ascending=False)["TrainingMethod"]
        .tolist()
    )
    sims = SIM_NAMES
    app_preferred = ["Raw", "Linear", "Normalization", "GAM-mgcv", "GAM-gnomon"]
    app_methods = [m for m in app_preferred if m in df["ApplicationMethod"].unique()]
    app_methods += [m for m in df["ApplicationMethod"].unique() if m not in app_methods]

    bar_width = 0.25
    gap_sim = 0.4
    gap_train = 0.8

    positions = []
    x = 0.0
    for train in training_order:
        for sim in sims:
            for app in app_methods:
                positions.append((x, train, sim, app))
                x += bar_width
            x += gap_sim
        x += gap_train

    fig, ax = plt.subplots(figsize=(14, max(6, len(training_order) * len(sims) * 0.5)))
    colors = plt.get_cmap("tab10")
    app_color = {app: colors(i % 10) for i, app in enumerate(app_methods)}

    position_map: dict[tuple[str, str, str], float] = {}
    height_map: dict[tuple[str, str, str], float] = {}

    for x_pos, train, sim, app in positions:
        row = df[
            (df["TrainingMethod"] == train)
            & (df["Simulation"] == sim)
            & (df["ApplicationMethod"] == app)
        ]
        if row.empty:
            continue
        auc = row["AUC_overall"].iloc[0]
        ax.bar(x_pos, auc, width=bar_width, color=app_color[app], label=app)
        position_map[(train, sim, app)] = x_pos
        height_map[(train, sim, app)] = auc

    # Build x tick labels at simulation group centers
    tick_positions = []
    tick_labels = []
    x = 0.0
    for train in training_order:
        for sim in sims:
            group_center = x + (bar_width * len(app_methods) - bar_width) / 2
            tick_positions.append(group_center)
            tick_labels.append(f"{train}\n{sim}")
            x += bar_width * len(app_methods) + gap_sim
        x += gap_train

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylabel("AUC (overall)")
    max_bar = df["AUC_overall"].max() if not df.empty else 1.0
    max_bracket_y = max_bar
    ax.set_ylim(0, 1)
    ax.set_title("Overall AUC by Training Method and Simulation")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    if not pvals.empty:
        pvals = pvals.copy()
        pvals["Training_1"] = pvals["Method_1"].str.split(" + ", n=1, regex=False).str[0]
        pvals["App_1"] = pvals["Method_1"].str.split(" + ", n=1, regex=False).str[1]
        pvals["Training_2"] = pvals["Method_2"].str.split(" + ", n=1, regex=False).str[0]
        pvals["App_2"] = pvals["Method_2"].str.split(" + ", n=1, regex=False).str[1]

        bracket_height = 0.012
        text_offset = 0.006
        level_step = 0.04

        grouped: dict[tuple[str, str], list[tuple[tuple[str, str, str], tuple[str, str, str], float]]] = {}
        for _, row in pvals.iterrows():
            sim = row["Simulation"]
            train1 = row["Training_1"]
            train2 = row["Training_2"]
            if train1 != train2:
                continue
            app1 = row["App_1"]
            app2 = row["App_2"]
            key1 = (train1, sim, app1)
            key2 = (train2, sim, app2)
            if key1 not in position_map or key2 not in position_map:
                continue
            grouped.setdefault((train1, sim), []).append((key1, key2, row["AUC_p_value"]))

        for (train, sim), pairs in grouped.items():
            # Use a shared baseline per training+simulation group to avoid uneven stacking
            group_keys = [(k1, k2) for k1, k2, _ in pairs]
            group_max = max(
                max(height_map.get(k1, 0), height_map.get(k2, 0)) for k1, k2 in group_keys
            )
            pairs_sorted = []
            for key1, key2, pval in pairs:
                x1 = position_map[key1]
                x2 = position_map[key2]
                if x1 > x2:
                    x1, x2 = x2, x1
                span = x2 - x1
                pairs_sorted.append((span, x1, x2, key1, key2, pval))
            pairs_sorted.sort(reverse=True)

            levels: list[list[tuple[float, float]]] = []
            for _, x1, x2, key1, key2, pval in pairs_sorted:
                base = group_max + bracket_height
                placed = False
                for level_idx, intervals in enumerate(levels):
                    if all(x2 < a or x1 > b for a, b in intervals):
                        levels[level_idx].append((x1, x2))
                        y = base + level_idx * level_step
                        placed = True
                        break
                if not placed:
                    levels.append([(x1, x2)])
                    y = base + (len(levels) - 1) * level_step

                ax.plot([x1, x1, x2, x2], [y, y + bracket_height, y + bracket_height, y], color="black", linewidth=0.8)
                ax.text(
                    (x1 + x2) / 2,
                    y + bracket_height + text_offset,
                    f"p={pval:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold" if pval < 0.05 else "normal",
                )
                max_bracket_y = max(max_bracket_y, y + bracket_height + text_offset)

        ax.set_ylim(0, max(1.0, max_bracket_y + 0.08))

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    metrics = load_metrics()
    if not metrics.empty:
        metrics.to_csv("all_methods_metrics.tsv", sep="\t", index=False)

    pvals = load_pvalues()
    plot_auc_summary(metrics, pvals, Path("all_methods_pvalues.png"))


if __name__ == "__main__":
    main()
