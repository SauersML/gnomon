"""
Aggregate per-simulation metrics and significance tests into a single TSV
and generate a combined AUC plot.
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SIM_NAMES = ["confounding", "portability"]


def _apply_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "axes.facecolor": "#f7f6f2",
            "figure.facecolor": "white",
            "axes.edgecolor": "#2f2f2f",
            "axes.linewidth": 0.9,
            "grid.color": "#c9c4b8",
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11.5,
            "axes.titlesize": 14,
            "font.size": 11,
            "legend.fontsize": 9,
        }
    )


def _style_axes(ax):
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)


def _metric_pval_column(metric: str) -> str | None:
    if metric.startswith("AUC"):
        return "AUC_p_value"
    if metric.startswith("Brier"):
        return "Brier_p_value"
    return None


def _best_method(df: pd.DataFrame, metric: str) -> str:
    ascending = metric.startswith("Brier")
    return df.sort_values(metric, ascending=ascending).iloc[0]["Method"]


def _load_seeded_metrics(sim: str) -> pd.DataFrame:
    seed_re = re.compile(rf"^{sim}_s(\d+)_metrics\.csv$")
    paths = sorted(Path(".").glob(f"{sim}_s*_metrics.csv"))
    if not paths:
        legacy = Path(f"{sim}_metrics.csv")
        if legacy.exists():
            paths = [legacy]
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        match = seed_re.match(path.name)
        seed_val = int(match.group(1)) if match else None
        df["Seed"] = seed_val
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ("Seed",)]
    agg = df.groupby("Method", as_index=False)[numeric_cols].mean()
    if "Seed" in df.columns:
        seed_counts = df.groupby("Method")["Seed"].nunique().reset_index(name="n_seeds")
        agg = agg.merge(seed_counts, on="Method", how="left")
    return agg


def load_metrics() -> pd.DataFrame:
    frames = []
    for sim in SIM_NAMES:
        df = _load_seeded_metrics(sim)
        if df.empty:
            continue
        agg = _aggregate_metrics(df)
        if agg.empty:
            continue
        agg.insert(0, "Simulation", sim)
        frames.append(agg)
        # Write per-simulation mean metrics for downstream summary
        agg.to_csv(f"{sim}_metrics.csv", index=False)
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


def _plot_metric_summary(
    df: pd.DataFrame,
    raw_df: pd.DataFrame,
    pvals_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    if df.empty or metric not in df.columns:
        return
    _apply_plot_style()
    df = df.sort_values(metric, ascending=metric.startswith("Brier"))
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.6)))
    palette = [
        "#3e6b8a",
        "#d87b5a",
        "#8b6f93",
        "#4d8c57",
        "#d0a44b",
        "#6a88a5",
        "#b95c7f",
        "#7f9f6b",
        "#c06e4d",
        "#5b7d8a",
    ]
    colors = [palette[i % len(palette)] for i in range(len(df))]
    bars = ax.bar(df["Method"], df[metric], color=colors, edgecolor="#2a2a2a", linewidth=0.7)
    method_order = df["Method"].tolist()
    if not raw_df.empty and metric in raw_df.columns:
        rng = np.random.default_rng(42)
        for idx, method in enumerate(method_order):
            vals = raw_df.loc[raw_df["Method"] == method, metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.18, 0.18, size=vals.size)
            ax.scatter(
                np.full(vals.size, idx) + jitter,
                vals,
                color="#141414",
                alpha=0.55,
                s=26,
                linewidth=0.4,
                edgecolors="#f7f6f2",
                zorder=3,
            )
    best_method = _best_method(df, metric)
    if not pvals_df.empty:
        col = _metric_pval_column(metric)
        if col in pvals_df.columns:
            for idx, method in enumerate(method_order):
                if method == best_method:
                    continue
                p_row = pvals_df[
                    ((pvals_df["Method_1"] == best_method) & (pvals_df["Method_2"] == method))
                    | ((pvals_df["Method_2"] == best_method) & (pvals_df["Method_1"] == method))
                ]
                if p_row.empty:
                    continue
                pval = p_row.iloc[0][col]
                if pd.isna(pval):
                    continue
                bar_height = bars[idx].get_height()
                ax.text(
                    idx,
                    bar_height + (0.02 if metric.startswith("AUC") else 0.002),
                    f"p={pval:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold" if pval < 0.05 else "normal",
                    color="#2a2a2a",
                )
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, fontweight="bold")
    _style_axes(ax)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)


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

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(14, max(6, len(training_order) * len(sims) * 0.5)))
    palette = [
        "#3e6b8a",
        "#d87b5a",
        "#8b6f93",
        "#4d8c57",
        "#d0a44b",
        "#6a88a5",
        "#b95c7f",
        "#7f9f6b",
    ]
    app_color = {app: palette[i % len(palette)] for i, app in enumerate(app_methods)}

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
    # Overlay per-seed points (jittered)
    raw_frames = []
    for sim in SIM_NAMES:
        raw = _load_seeded_metrics(sim)
        if raw.empty:
            continue
        raw = raw.copy()
        raw["Simulation"] = sim
        raw_frames.append(raw)
    if raw_frames:
        raw_all = pd.concat(raw_frames, ignore_index=True)
        raw_all["TrainingMethod"] = raw_all["Method"].str.split(" + ", n=1, regex=False).str[0]
        raw_all["ApplicationMethod"] = raw_all["Method"].str.split(" + ", n=1, regex=False).str[1]
        rng = np.random.default_rng(42)
        for (train, sim, app), x_pos in position_map.items():
            vals = raw_all.loc[
                (raw_all["TrainingMethod"] == train)
                & (raw_all["Simulation"] == sim)
                & (raw_all["ApplicationMethod"] == app),
                "AUC_overall",
            ].dropna().to_numpy()
            if vals.size == 0:
                continue
            jitter = rng.uniform(-bar_width * 0.35, bar_width * 0.35, size=vals.size)
            ax.scatter(
                np.full(vals.size, x_pos) + jitter,
                vals,
                color="#141414",
                alpha=0.55,
                s=18,
                linewidth=0.4,
                edgecolors="#f7f6f2",
                zorder=3,
            )

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
    _style_axes(ax)

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
    ax.legend(dedup.values(), dedup.keys(), loc="upper right", frameon=False, title="Calibration")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def main() -> None:
    metrics = load_metrics()
    if not metrics.empty:
        metrics.to_csv("all_methods_metrics.tsv", sep="\t", index=False)
    pvals = load_pvalues()

    # Per-simulation mean plots
    for sim in SIM_NAMES:
        sim_df = metrics[metrics["Simulation"] == sim] if not metrics.empty else pd.DataFrame()
        if sim_df.empty:
            continue
        raw_df = _load_seeded_metrics(sim)
        pvals_df = pvals[pvals["Simulation"] == sim] if not pvals.empty else pd.DataFrame()
        _plot_metric_summary(
            sim_df,
            raw_df,
            pvals_df,
            "AUC_overall",
            Path(f"{sim}_comparison_auc.png"),
            f"AUC Summary - Simulation {sim} (mean over seeds)",
        )
        _plot_metric_summary(
            sim_df,
            raw_df,
            pvals_df,
            "Brier_overall",
            Path(f"{sim}_comparison_brier.png"),
            f"Brier Summary - Simulation {sim} (mean over seeds)",
        )

    # Aggregate overview plot (no p-values when averaging across seeds)
    if not pvals.empty:
        plot_auc_summary(metrics, pvals, Path("all_methods_pvalues.png"))
    else:
        plot_auc_summary(metrics, pd.DataFrame(), Path("all_methods_pvalues.png"))


if __name__ == "__main__":
    main()
