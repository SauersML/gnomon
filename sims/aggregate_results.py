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
from sklearn.metrics import roc_curve
from scipy.stats import norm


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


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n = y_true.size
    if n == 0:
        return np.nan
    y_true_int = y_true.astype(np.int8, copy=False)
    n_pos = int(y_true_int.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sum_ranks_pos = ranks[y_true_int == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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
        paths = sorted(Path(".").glob(f"{sim}_s*_significance_tests.csv"))
        if not paths:
            legacy = Path(f"{sim}_significance_tests.csv")
            if legacy.exists():
                paths = [legacy]
        if not paths:
            continue
        seed_re = re.compile(rf"^{sim}_s(\d+)_significance_tests\.csv$")
        seeded = []
        for path in paths:
            df = pd.read_csv(path)
            m = seed_re.match(path.name)
            df["Seed"] = int(m.group(1)) if m else np.nan
            seeded.append(df)
        df = pd.concat(seeded, ignore_index=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Seed"]
        if numeric_cols:
            df = (
                df.groupby(["Method_1", "Method_2"], as_index=False)[numeric_cols]
                .mean()
            )
            df["n_seeds"] = len(paths)
            df.to_csv(f"{sim}_significance_tests.csv", index=False)
        df.insert(0, "Simulation", sim)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_seeded_probabilities(sim: str) -> pd.DataFrame:
    seed_re = re.compile(rf"^{sim}_s(\d+)_probabilities\.csv$")
    paths = sorted(Path(".").glob(f"{sim}_s*_probabilities.csv"))
    if not paths:
        legacy = Path(f"{sim}_probabilities.csv")
        if legacy.exists():
            paths = [legacy]
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        m = seed_re.match(path.name)
        df["Seed"] = int(m.group(1)) if m else np.nan
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _plot_seed_aggregated_roc(prob_df: pd.DataFrame, sim: str, out_path: Path) -> None:
    if prob_df.empty:
        return
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 7))
    grid = np.linspace(0, 1, 201)
    methods = sorted(prob_df["Method"].unique())
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
    for i, method in enumerate(methods):
        method_df = prob_df[prob_df["Method"] == method]
        curves = []
        aucs = []
        for seed in sorted(method_df["Seed"].dropna().unique()):
            sdf = method_df[method_df["Seed"] == seed]
            y_true = sdf["y_true"].to_numpy()
            y_prob = sdf["y_prob"].to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            interp_tpr = np.interp(grid, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tpr[-1] = 1.0
            curves.append(interp_tpr)
            aucs.append(fast_auc(y_true, y_prob))
        if not curves:
            continue
        arr = np.vstack(curves)
        mean_tpr = np.nanmean(arr, axis=0)
        std_tpr = np.nanstd(arr, axis=0)
        lo = np.clip(mean_tpr - std_tpr, 0, 1)
        hi = np.clip(mean_tpr + std_tpr, 0, 1)
        color = palette[i % len(palette)]
        mean_auc = float(np.nanmean(aucs)) if aucs else np.nan
        ax.plot(grid, mean_tpr, color=color, linewidth=2.1, label=f"{method} (mean AUC={mean_auc:.3f})")
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - Simulation {sim} (mean ± SD across seeds)", fontweight="bold")
    _style_axes(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def _plot_seed_aggregated_calibration(prob_df: pd.DataFrame, sim: str, out_path: Path) -> None:
    if prob_df.empty:
        return
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 7))
    bins = np.linspace(0.0, 1.0, 11)
    methods = sorted(prob_df["Method"].unique())
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
    for i, method in enumerate(methods):
        method_df = prob_df[prob_df["Method"] == method]
        per_seed_obs = []
        per_seed_pred = []
        for seed in sorted(method_df["Seed"].dropna().unique()):
            sdf = method_df[method_df["Seed"] == seed]
            y_true = sdf["y_true"].to_numpy()
            y_prob = sdf["y_prob"].to_numpy()
            obs = np.full(len(bins) - 1, np.nan)
            pred = np.full(len(bins) - 1, np.nan)
            for j in range(len(bins) - 1):
                if j == len(bins) - 2:
                    mask = (y_prob >= bins[j]) & (y_prob <= bins[j + 1])
                else:
                    mask = (y_prob >= bins[j]) & (y_prob < bins[j + 1])
                if np.any(mask):
                    obs[j] = np.mean(y_true[mask])
                    pred[j] = np.mean(y_prob[mask])
            per_seed_obs.append(obs)
            per_seed_pred.append(pred)
        if not per_seed_obs:
            continue
        obs_arr = np.vstack(per_seed_obs)
        pred_arr = np.vstack(per_seed_pred)
        mean_obs = np.nanmean(obs_arr, axis=0)
        std_obs = np.nanstd(obs_arr, axis=0)
        mean_pred = np.nanmean(pred_arr, axis=0)
        valid = ~np.isnan(mean_obs) & ~np.isnan(mean_pred)
        if not np.any(valid):
            continue
        color = palette[i % len(palette)]
        ax.plot(mean_pred[valid], mean_obs[valid], "o-", color=color, linewidth=2, markersize=4, label=method)
        lo = np.clip(mean_obs[valid] - std_obs[valid], 0, 1)
        hi = np.clip(mean_obs[valid] + std_obs[valid], 0, 1)
        ax.fill_between(mean_pred[valid], lo, hi, color=color, alpha=0.15)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(f"Calibration Curves - Simulation {sim} (mean ± SD across seeds)", fontweight="bold")
    _style_axes(ax)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def _auc_var_hanley_mcneil(auc: float, n_pos: int, n_neg: int) -> float:
    """
    Approximate sampling variance of AUC from class counts.
    """
    if not np.isfinite(auc) or n_pos <= 0 or n_neg <= 0:
        return np.nan
    auc = float(np.clip(auc, 1e-6, 1 - 1e-6))
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (
        auc * (1 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    return float(max(var, 0.0))


def _random_effects_meta(y: np.ndarray, v: np.ndarray) -> dict[str, float]:
    """
    DerSimonian-Laird random-effects meta-analysis against 0.
    """
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)
    mask = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y = y[mask]
    v = v[mask]
    k = int(len(y))
    if k == 0:
        return {"k": 0, "mu": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "tau2": np.nan}

    w = 1.0 / v
    mu_fe = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mu_fe) ** 2))
    c = float(np.sum(w) - np.sum(w * w) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_re = 1.0 / (v + tau2)
    mu = float(np.sum(w_re * y) / np.sum(w_re))
    se = float(np.sqrt(1.0 / np.sum(w_re)))
    z = mu / se if se > 0 else np.nan
    p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan
    return {"k": k, "mu": mu, "se": se, "z": z, "p": p, "tau2": float(tau2)}


def _plot_portability_eur_minus_non_eur_proper(
    metrics_df: pd.DataFrame,
    prob_df: pd.DataFrame,
    out_plot: Path,
) -> None:
    """
    Portability-only proper test:
    - Within-seed AUC variance (Hanley-McNeil) for EUR and NON-EUR mean.
    - Between-seed random-effects pooling.
    - Always annotate method p-values.
    """
    if metrics_df.empty or prob_df.empty:
        return

    req_cols = {"Method", "Seed", "AUC_EUR"}
    if not req_cols.issubset(metrics_df.columns):
        return

    base = prob_df[["Seed", "IID", "pop_label", "y_true"]].drop_duplicates()
    pop_counts = (
        base.groupby(["Seed", "pop_label"], as_index=False)["y_true"]
        .agg(n="count", n_pos="sum")
    )
    pop_counts["n_neg"] = pop_counts["n"] - pop_counts["n_pos"]

    count_wide = (
        pop_counts.pivot(index="Seed", columns="pop_label", values=["n_pos", "n_neg"])
        .sort_index(axis=1)
    )
    count_wide.columns = [f"{a}_{b}" for a, b in count_wide.columns]
    count_wide = count_wide.reset_index()
    count_wide.to_csv("portability_test_group_counts.csv", index=False)

    per_seed_rows = []
    non_cols_pref = ["AUC_AFR", "AUC_ASIA", "AUC_ADMIX"]
    for row in metrics_df.itertuples(index=False):
        seed = int(row.Seed)
        method = row.Method
        eur_auc = getattr(row, "AUC_EUR", np.nan)
        eur_n_pos = pop_counts.loc[
            (pop_counts["Seed"] == seed) & (pop_counts["pop_label"] == "EUR"),
            "n_pos",
        ]
        eur_n_neg = pop_counts.loc[
            (pop_counts["Seed"] == seed) & (pop_counts["pop_label"] == "EUR"),
            "n_neg",
        ]
        if eur_n_pos.empty or eur_n_neg.empty:
            continue
        eur_var = _auc_var_hanley_mcneil(float(eur_auc), int(eur_n_pos.iloc[0]), int(eur_n_neg.iloc[0]))

        non_aucs = []
        non_vars = []
        used = []
        for col in non_cols_pref:
            if not hasattr(row, col):
                continue
            auc_val = getattr(row, col)
            if pd.isna(auc_val):
                continue
            pop = col.replace("AUC_", "")
            n_pos_s = pop_counts.loc[
                (pop_counts["Seed"] == seed) & (pop_counts["pop_label"] == pop),
                "n_pos",
            ]
            n_neg_s = pop_counts.loc[
                (pop_counts["Seed"] == seed) & (pop_counts["pop_label"] == pop),
                "n_neg",
            ]
            if n_pos_s.empty or n_neg_s.empty:
                continue
            v = _auc_var_hanley_mcneil(float(auc_val), int(n_pos_s.iloc[0]), int(n_neg_s.iloc[0]))
            if not np.isfinite(v):
                continue
            non_aucs.append(float(auc_val))
            non_vars.append(float(v))
            used.append(pop)

        if not non_aucs or not np.isfinite(eur_var):
            continue

        k_non = len(non_aucs)
        auc_non = float(np.mean(non_aucs))
        var_non = float(np.sum(non_vars) / (k_non * k_non))
        delta = float(eur_auc - auc_non)
        var_delta = float(eur_var + var_non)
        per_seed_rows.append(
            {
                "Seed": seed,
                "Method": method,
                "AUC_EUR": float(eur_auc),
                "AUC_NON_EUR": auc_non,
                "EUR_MINUS_NON_EUR": delta,
                "var_within": var_delta,
                "se_within": float(np.sqrt(var_delta)) if np.isfinite(var_delta) else np.nan,
                "n_non_eur_groups": k_non,
                "non_eur_groups_used": ",".join(used),
            }
        )

    if not per_seed_rows:
        return

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["Method", "Seed"])
    per_seed_df.to_csv("portability_eur_minus_non_eur_proper_per_seed.csv", index=False)

    summary_rows = []
    for method, g in per_seed_df.groupby("Method"):
        meta = _random_effects_meta(g["EUR_MINUS_NON_EUR"].to_numpy(), g["var_within"].to_numpy())
        summary_rows.append(
            {
                "Method": method,
                "delta_random_effects": meta["mu"],
                "se_random_effects": meta["se"],
                "z_vs_0": meta["z"],
                "p_value_vs_0": meta["p"],
                "tau2_between_seed": meta["tau2"],
                "k_seeds_used": meta["k"],
                "delta_seed_mean_unweighted": float(g["EUR_MINUS_NON_EUR"].mean()),
                "delta_seed_sd_unweighted": float(g["EUR_MINUS_NON_EUR"].std(ddof=1))
                if len(g) > 1
                else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("delta_random_effects", ascending=False)
    summary_df.to_csv("portability_eur_minus_non_eur_proper_summary.csv", index=False)

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    order = summary_df["Method"].tolist()
    idx = summary_df.set_index("Method")
    means = idx.loc[order, "delta_random_effects"].to_numpy()
    ses = idx.loc[order, "se_random_effects"].to_numpy()
    pvals = idx.loc[order, "p_value_vs_0"].to_numpy()

    palette = ["#3e6b8a", "#d87b5a", "#8b6f93", "#4d8c57", "#d0a44b", "#6a88a5"]
    x = np.arange(len(order))
    colors = [palette[i % len(palette)] for i in range(len(order))]
    ax.bar(x, means, color=colors, edgecolor="#2a2a2a", linewidth=0.7, yerr=1.96 * ses, capsize=4)

    rng = np.random.default_rng(42)
    for i, method in enumerate(order):
        vals = per_seed_df.loc[per_seed_df["Method"] == method, "EUR_MINUS_NON_EUR"].dropna().to_numpy()
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(
            np.full(vals.size, i) + jitter,
            vals,
            color="#141414",
            alpha=0.65,
            s=26,
            linewidth=0.35,
            edgecolors="#f7f6f2",
            zorder=3,
        )

    # Always annotate p-values (requested).
    span = np.nanmax(np.abs(means)) + np.nanmax(1.96 * ses) if len(means) else 0.0
    y_off = max(0.004, float(span) * 0.08 if np.isfinite(span) else 0.004)
    for i, p in enumerate(pvals):
        if not np.isfinite(p):
            continue
        y = means[i]
        y_txt = y + y_off if y >= 0 else y - y_off
        va = "bottom" if y >= 0 else "top"
        ax.text(i, y_txt, f"p={p:.3g}", ha="center", va=va, fontsize=9, color="#2a2a2a")

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.split(" + ", 1)[1] if " + " in m else m for m in order],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("AUC_EUR - AUC_NON_EUR")
    ax.set_title(
        "Portability: EUR vs NON-EUR AUC Gap\n"
        "Random-effects model (within-seed + between-seed variance), 95% CI",
        fontweight="bold",
    )
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=220)


def _plot_portability_auc_by_ancestry_hierarchical(
    metrics_df: pd.DataFrame,
    prob_df: pd.DataFrame,
) -> None:
    """
    Portability ancestry-specific AUC with hierarchical 95% CI:
    - within-seed variance from Hanley-McNeil AUC variance
    - between-seed variance from random-effects pooling
    """
    if metrics_df.empty or prob_df.empty:
        return
    req = {"Method", "Seed", "AUC_AFR", "AUC_EUR", "AUC_ASIA", "AUC_ADMIX"}
    if not req.issubset(metrics_df.columns):
        return

    base = prob_df[["Seed", "IID", "pop_label", "y_true"]].drop_duplicates()
    pop_counts = (
        base.groupby(["Seed", "pop_label"], as_index=False)["y_true"]
        .agg(n="count", n_pos="sum")
    )
    pop_counts["n_neg"] = pop_counts["n"] - pop_counts["n_pos"]

    count_wide = (
        pop_counts.pivot(index="Seed", columns="pop_label", values=["n_pos", "n_neg"])
        .sort_index(axis=1)
    )
    count_wide.columns = [f"{a}_{b}" for a, b in count_wide.columns]
    count_wide = count_wide.reset_index()
    count_wide.to_csv("portability_test_group_counts.csv", index=False)

    ancestries = ["AFR", "EUR", "ASIA", "ADMIX"]
    per_seed_rows = []
    for row in metrics_df.itertuples(index=False):
        seed = int(row.Seed)
        method = row.Method
        for anc in ancestries:
            col = f"AUC_{anc}"
            if not hasattr(row, col):
                continue
            auc_val = getattr(row, col)
            if pd.isna(auc_val):
                continue
            n_pos_s = pop_counts.loc[
                (pop_counts["Seed"] == seed) & (pop_counts["pop_label"] == anc),
                "n_pos",
            ]
            n_neg_s = pop_counts.loc[
                (pop_counts["Seed"] == seed) & (pop_counts["pop_label"] == anc),
                "n_neg",
            ]
            if n_pos_s.empty or n_neg_s.empty:
                continue
            v = _auc_var_hanley_mcneil(float(auc_val), int(n_pos_s.iloc[0]), int(n_neg_s.iloc[0]))
            if not np.isfinite(v):
                continue
            per_seed_rows.append(
                {
                    "Seed": seed,
                    "Method": method,
                    "Ancestry": anc,
                    "AUC": float(auc_val),
                    "var_within": float(v),
                    "se_within": float(np.sqrt(v)),
                    "n_pos": int(n_pos_s.iloc[0]),
                    "n_neg": int(n_neg_s.iloc[0]),
                }
            )

    if not per_seed_rows:
        return

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["Method", "Ancestry", "Seed"])
    per_seed_df.to_csv("portability_auc_by_ancestry_hierarchical_per_seed.csv", index=False)

    summary_rows = []
    for (method, anc), g in per_seed_df.groupby(["Method", "Ancestry"]):
        meta = _random_effects_meta(g["AUC"].to_numpy(), g["var_within"].to_numpy())
        summary_rows.append(
            {
                "Method": method,
                "Ancestry": anc,
                "mean_auc_re": meta["mu"],
                "se_re": meta["se"],
                "ci95_half": 1.96 * meta["se"] if np.isfinite(meta["se"]) else np.nan,
                "tau2_between_seed": meta["tau2"],
                "k_seeds": meta["k"],
                "mean_auc_unweighted": float(g["AUC"].mean()),
                "sd_auc_unweighted": float(g["AUC"].std(ddof=1)) if len(g) > 1 else np.nan,
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["Method", "Ancestry"])
    summary_df.to_csv("portability_auc_by_ancestry_95ci_hierarchical.csv", index=False)

    _apply_plot_style()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 6))
    method_order = (
        summary_df.groupby("Method", as_index=False)["mean_auc_re"]
        .mean()
        .sort_values("mean_auc_re", ascending=False)["Method"]
        .tolist()
    )
    anc_order = ["AFR", "EUR", "ASIA", "ADMIX"]
    colors = {"AFR": "#d87b5a", "EUR": "#3e6b8a", "ASIA": "#4d8c57", "ADMIX": "#8b6f93"}
    x = np.arange(len(method_order))
    offsets = np.linspace(-0.27, 0.27, len(anc_order))
    for off, anc in zip(offsets, anc_order):
        sdf = summary_df[summary_df["Ancestry"] == anc].set_index("Method").reindex(method_order)
        y = sdf["mean_auc_re"].to_numpy()
        e = sdf["ci95_half"].to_numpy()
        ax.errorbar(
            x + off,
            y,
            yerr=e,
            fmt="o",
            capsize=4,
            markersize=6,
            linewidth=1.7,
            color=colors[anc],
            label=anc,
        )

    rng = np.random.default_rng(42)
    for i, method in enumerate(method_order):
        for off, anc in zip(offsets, anc_order):
            vals = per_seed_df.loc[
                (per_seed_df["Method"] == method) & (per_seed_df["Ancestry"] == anc),
                "AUC",
            ].dropna().to_numpy()
            if vals.size == 0:
                continue
            jitter = rng.uniform(-0.04, 0.04, size=vals.size)
            ax.scatter(
                np.full(vals.size, i + off) + jitter,
                vals,
                s=18,
                color=colors[anc],
                alpha=0.35,
                edgecolors="none",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.split(" + ", 1)[1] if " + " in m else m for m in method_order],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel("AUC")
    ax.set_ylim(0.35, 0.70)
    _style_axes(ax)
    ax.legend(title="Ancestry", frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig("portability_auc_by_ancestry_95ci_hierarchical.png", dpi=220)
    # Keep legacy/expected filename in sync for downstream use.
    fig.savefig("portability_auc_by_ancestry_95ci.png", dpi=220)


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
        prob_df = _load_seeded_probabilities(sim)
        pvals_df = pvals[pvals["Simulation"] == sim] if not pvals.empty else pd.DataFrame()
        if not prob_df.empty:
            _plot_seed_aggregated_roc(prob_df, sim, Path(f"{sim}_comparison_roc.png"))
            _plot_seed_aggregated_calibration(prob_df, sim, Path(f"{sim}_comparison_calibration.png"))
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

    portability_metrics = _load_seeded_metrics("portability")
    portability_probs = _load_seeded_probabilities("portability")
    _plot_portability_eur_minus_non_eur_proper(
        portability_metrics,
        portability_probs,
        Path("portability_eur_minus_non_eur_gap_proper_test_all_pvals.png"),
    )
    _plot_portability_auc_by_ancestry_hierarchical(
        portability_metrics,
        portability_probs,
    )


if __name__ == "__main__":
    main()
