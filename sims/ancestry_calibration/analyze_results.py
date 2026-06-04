"""Statistical analysis for the real-P+T ancestry-calibration study."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "sims" / "results_hpc" / "ancestry_calibration" / "results"
REPORTS_DIR = REPO_ROOT / "sims" / "results_hpc" / "ancestry_calibration" / "reports"

METHOD_LABELS = {
    "linpc": "PGS + PCs",
    "znorm": "z-norm",
}

METRICS = {
    "auc": {"label": "AUC", "higher_better": True},
    "liability_pseudo_r2": {"label": "Liability-scale pseudo-R2", "higher_better": True},
    "bss": {"label": "Brier Skill Score (BSS)", "higher_better": True},
    "mae_true_risk": {"label": "MAE vs known true risk", "higher_better": False},
    "rmse_true_risk": {"label": "RMSE vs known true risk", "higher_better": False},
    "abs_slope_error": {"label": "Abs. true-slope error", "higher_better": False},
    "abs_prevalence_error": {"label": "Abs. prevalence error", "higher_better": False},
}


def add_bss_column(df: pd.DataFrame, prevalence: pd.Series) -> pd.DataFrame:
    """Derive Brier Skill Score from raw Brier and the true base rate.

    BSS = 1 - Brier / [p_bar * (1 - p_bar)], where p_bar is the TRUE prevalence
    (base rate) of the evaluation unit. Higher is better. Where the reference
    Brier p_bar*(1-p_bar) is zero or the prevalence is unavailable, BSS is left
    as NaN so downstream tests simply skip those (never fabricated).
    """
    out = df.copy()
    if "brier" not in out.columns:
        out["bss"] = np.nan
        return out
    p = pd.to_numeric(prevalence, errors="coerce").to_numpy(float)
    ref = p * (1.0 - p)
    brier = pd.to_numeric(out["brier"], errors="coerce").to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bss = np.where((ref > 0) & np.isfinite(brier), 1.0 - brier / ref, np.nan)
    out["bss"] = bss
    return out


def cell_true_prevalence(group: pd.DataFrame) -> pd.DataFrame:
    """n-weighted mean of per-bin known_true_prevalence for each (dem, pheno, seed).

    The global accuracy file carries no prevalence column, so the true base rate
    of each (dem, pheno, seed) cell is reconstructed from the distance-bin rows
    (prevalence is the simulated truth and is method-invariant, so any method's
    bins suffice; we average over one method to avoid double counting)."""
    bins = group[["dem", "pheno", "seed", "method", "dist_from_train", "n", "known_true_prevalence"]].copy()
    bins = bins.dropna(subset=["known_true_prevalence", "n"])
    # prevalence/n are method-invariant per bin; pick a single method per cell.
    bins = bins.sort_values("method")
    first_method = bins.groupby(["dem", "pheno", "seed"])["method"].transform("first")
    bins = bins[bins["method"] == first_method]
    bins["n"] = pd.to_numeric(bins["n"], errors="coerce")
    bins["wp"] = bins["n"] * pd.to_numeric(bins["known_true_prevalence"], errors="coerce")
    agg = bins.groupby(["dem", "pheno", "seed"], as_index=False).agg(wp=("wp", "sum"), n=("n", "sum"))
    agg["cell_true_prevalence"] = agg["wp"] / agg["n"]
    return agg[["dem", "pheno", "seed", "cell_true_prevalence"]]


def unique_cols(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def paired_advantage(gamfit: pd.Series, baseline: pd.Series, metric: str) -> np.ndarray:
    if METRICS[metric]["higher_better"]:
        return (gamfit - baseline).to_numpy(float)
    return (baseline - gamfit).to_numpy(float)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2 or np.allclose(values, values[0]):
        return float(np.mean(values)), float(np.mean(values))
    res = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=20000,
        method="BCa",
        random_state=20260604,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def signed_rank(values: np.ndarray, alternative: str = "greater") -> float:
    nonzero = values[~np.isclose(values, 0.0)]
    if nonzero.size == 0:
        return 1.0
    return float(stats.wilcoxon(nonzero, alternative=alternative, zero_method="wilcox").pvalue)


def paired_t(values: np.ndarray, alternative: str = "greater") -> float:
    if values.size < 2 or np.allclose(values, values[0]):
        return 1.0
    return float(stats.ttest_1samp(values, popmean=0.0, alternative=alternative).pvalue)


def sign_test(values: np.ndarray, alternative: str = "greater") -> tuple[int, int, float]:
    nonzero = values[~np.isclose(values, 0.0)]
    wins = int(np.sum(nonzero > 0))
    n = int(nonzero.size)
    if n == 0:
        return wins, n, 1.0
    return wins, n, float(stats.binomtest(wins, n, p=0.5, alternative=alternative).pvalue)


def collect_tests(
    df: pd.DataFrame,
    source: str,
    unit_cols: list[str],
    metric_cols: list[str],
    scope_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        keep = unique_cols(unit_cols + scope_cols + ["method", metric])
        data = df[keep].dropna(subset=[metric]).copy()
        for scope_key, scope_df in data.groupby(scope_cols, dropna=False):
            if not isinstance(scope_key, tuple):
                scope_key = (scope_key,)
            for baseline in METHOD_LABELS:
                wide = (
                    scope_df[scope_df["method"].isin(["gamfit", baseline])]
                    .pivot_table(index=unit_cols, columns="method", values=metric, aggfunc="mean")
                    .dropna(subset=["gamfit", baseline])
                )
                if wide.empty:
                    continue
                advantage = paired_advantage(wide["gamfit"], wide[baseline], metric)
                wins, nonzero_n, p_sign = sign_test(advantage)
                ci_low, ci_high = bootstrap_ci(advantage)
                row = {
                    "source": source,
                    "baseline": baseline,
                    "baseline_label": METHOD_LABELS[baseline],
                    "metric": metric,
                    "metric_label": METRICS[metric]["label"],
                    "higher_better": METRICS[metric]["higher_better"],
                    "n_pairs": int(advantage.size),
                    "wins": wins,
                    "nonzero_pairs": nonzero_n,
                    "win_rate": wins / nonzero_n if nonzero_n else np.nan,
                    "gamfit_advantage_mean": float(np.mean(advantage)),
                    "gamfit_advantage_median": float(np.median(advantage)),
                    "advantage_ci95_low": ci_low,
                    "advantage_ci95_high": ci_high,
                    "p_paired_t_win": paired_t(advantage, alternative="greater"),
                    "p_wilcoxon_win": signed_rank(advantage, alternative="greater"),
                    "p_sign_win": p_sign,
                    "p_paired_t_loss": paired_t(advantage, alternative="less"),
                    "p_wilcoxon_loss": signed_rank(advantage, alternative="less"),
                    "p_sign_loss": sign_test(advantage, alternative="less")[2],
                    "p_paired_t_two_sided": paired_t(advantage, alternative="two-sided"),
                    "p_wilcoxon_two_sided": signed_rank(advantage, alternative="two-sided"),
                }
                row.update(dict(zip(scope_cols, scope_key, strict=True)))
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for p_col in [
        "p_paired_t_win",
        "p_wilcoxon_win",
        "p_sign_win",
        "p_paired_t_loss",
        "p_wilcoxon_loss",
        "p_sign_loss",
        "p_paired_t_two_sided",
        "p_wilcoxon_two_sided",
    ]:
        out[f"q_bh_{p_col[2:]}"] = multipletests(out[p_col].to_numpy(float), method="fdr_bh")[1]
    return out


def add_value_summaries(
    df: pd.DataFrame,
    unit_cols: list[str],
    metric_cols: list[str],
    scope_cols: list[str],
    source: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        data = df[unique_cols(unit_cols + scope_cols + ["method", metric])].dropna(subset=[metric])
        grouped = data.groupby(scope_cols + ["method"], dropna=False)[metric]
        for key, vals in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            scope_key = key[:-1]
            method = key[-1]
            row = {
                "source": source,
                "metric": metric,
                "metric_label": METRICS[metric]["label"],
                "method": method,
                "n": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "sd": float(vals.std(ddof=1)) if vals.shape[0] > 1 else 0.0,
            }
            row.update(dict(zip(scope_cols, scope_key, strict=True)))
            rows.append(row)
    return pd.DataFrame(rows)


def long_accuracy_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["dem", "pheno", "seed", "pgs_mode", "centers", "method"],
        columns="metric",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    return wide


def write_markdown(tests: pd.DataFrame, summaries: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Real P+T Significance Analysis\n")
    lines.append("Positive gamfit advantage means gamfit is better. For AUC and liability pseudo-R2 this is gamfit minus baseline; for error metrics it is baseline minus gamfit.\n")
    lines.append("Tests are paired on the same simulated unit: global metrics pair by demography, phenotype, and seed; distance-bin metrics pair by demography, phenotype, seed, and distance bin. P-values are one-sided for gamfit advantage > 0. BH q-values correct across all reported contrasts per test family.\n")

    def top_table(title: str, data: pd.DataFrame) -> None:
        lines.append(f"\n## {title}\n")
        cols = [
            "source",
            "dem",
            "pheno",
            "baseline_label",
            "metric_label",
            "n_pairs",
            "win_rate",
            "gamfit_advantage_mean",
            "advantage_ci95_low",
            "advantage_ci95_high",
            "p_wilcoxon_win",
            "q_bh_wilcoxon_win",
            "p_wilcoxon_loss",
            "q_bh_wilcoxon_loss",
        ]
        cols = [c for c in cols if c in data.columns]
        show = data[cols].copy()
        for c in [
            "win_rate",
            "gamfit_advantage_mean",
            "advantage_ci95_low",
            "advantage_ci95_high",
            "p_wilcoxon_win",
            "q_bh_wilcoxon_win",
            "p_wilcoxon_loss",
            "q_bh_wilcoxon_loss",
        ]:
            if c in show.columns:
                show[c] = show[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
        lines.append(show.to_markdown(index=False))
        lines.append("")

    main = tests[(tests["pheno"] == "phenoA") & (tests["baseline"].isin(["linpc", "znorm"]))].copy()
    main = main.sort_values(["source", "dem", "metric", "baseline"])
    top_table("Main phenotype: deme-varying environmental baseline risk", main)

    all_pheno = tests[tests["baseline"].isin(["linpc", "znorm"])].copy()
    all_pheno = all_pheno.sort_values(["source", "dem", "pheno", "metric", "baseline"])
    top_table("All phenotypes", all_pheno)

    lines.append("\n## Method-value summaries\n")
    show = summaries.copy()
    show = show.sort_values([c for c in ["source", "dem", "pheno", "metric", "method"] if c in show.columns])
    for c in ["mean", "median", "sd"]:
        show[c] = show[c].map(lambda x: f"{x:.4g}" if pd.notna(x) else "")
    lines.append(show.to_markdown(index=False))
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    accuracy = long_accuracy_to_wide(pd.read_csv(RESULTS_DIR / "accuracy_realpt_binary.csv"))
    group = pd.read_csv(RESULTS_DIR / "group_metrics_realpt_binary.csv")
    group = group[(group["pgs_mode"] == "realpt") & (group["group_kind"] == "distance")]

    # Brier Skill Score is derived from the raw Brier and the TRUE base rate.
    # Distance bins use each bin's own known_true_prevalence; the global cell uses
    # the n-weighted true prevalence reconstructed from that cell's distance bins.
    group = add_bss_column(group, group["known_true_prevalence"])
    cell_prev = cell_true_prevalence(group)
    accuracy = accuracy.merge(cell_prev, on=["dem", "pheno", "seed"], how="left")
    accuracy = add_bss_column(accuracy, accuracy["cell_true_prevalence"])

    global_metrics = ["auc", "liability_pseudo_r2", "bss", "mae_true_risk", "rmse_true_risk"]
    group_metrics = ["auc", "liability_pseudo_r2", "bss", "abs_slope_error", "abs_prevalence_error", "mae_true_risk", "rmse_true_risk"]

    tests = pd.concat(
        [
            collect_tests(
                accuracy,
                source="global_test",
                unit_cols=["dem", "pheno", "seed"],
                metric_cols=global_metrics,
                scope_cols=["dem", "pheno"],
            ),
            collect_tests(
                group,
                source="distance_bins",
                unit_cols=["dem", "pheno", "seed", "dist_from_train"],
                metric_cols=group_metrics,
                scope_cols=["dem", "pheno"],
            ),
        ],
        ignore_index=True,
    )
    summaries = pd.concat(
        [
            add_value_summaries(
                accuracy,
                unit_cols=["dem", "pheno", "seed"],
                metric_cols=global_metrics,
                scope_cols=["dem", "pheno"],
                source="global_test",
            ),
            add_value_summaries(
                group,
                unit_cols=["dem", "pheno", "seed", "dist_from_train"],
                metric_cols=group_metrics,
                scope_cols=["dem", "pheno"],
                source="distance_bins",
            ),
        ],
        ignore_index=True,
    )

    tests.to_csv(REPORTS_DIR / "significance_tests_realpt_binary.csv", index=False)
    summaries.to_csv(REPORTS_DIR / "method_summaries_realpt_binary.csv", index=False)
    write_markdown(tests, summaries, REPORTS_DIR / "significance_report_realpt_binary.md")
    print(f"wrote {REPORTS_DIR / 'significance_tests_realpt_binary.csv'}")
    print(f"wrote {REPORTS_DIR / 'method_summaries_realpt_binary.csv'}")
    print(f"wrote {REPORTS_DIR / 'significance_report_realpt_binary.md'}")


if __name__ == "__main__":
    main()
