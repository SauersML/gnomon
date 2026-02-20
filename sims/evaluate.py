
"""
Evaluate PRS models: Calibration, AUC, and Plots.
Usage: python evaluate.py <sim_name>
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from scipy import stats
from itertools import combinations
import multiprocessing as mp

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from methods import (
    RawPGSMethod,
    GAMMethod,
    LinearInteractionMethod,
    NormalizationMethod,
)
from metrics import compute_all_metrics, compute_calibration_curve

N_PERMUTATIONS = 1000
MAX_WORKERS = 8

def _apply_plot_style():
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

def _style_axes(ax):
    ax.grid(True, alpha=0.6)
    ax.set_axisbelow(True)

def load_scores(work_dir, methods):
    """Load scores. At least ONE method must succeed."""
    scores_dict = {}
    missing_methods = []

    for method in methods:
        score_file = work_dir / f"{method}.sscore"
        if not score_file.exists():
            print(f"Score file missing for {method} at {score_file} - method skipped.")
            missing_methods.append(method)
            continue

        try:
            df = pd.read_csv(score_file, sep='\t')
            # Expect IID, PRS
            df['IID'] = df['IID'].astype(str)
            scores_dict[method] = df.set_index('IID')['PRS']
            print(f"Loaded {method} scores")
        except Exception as e:
            print(f"Failed to load {method} scores: {e}")
            missing_methods.append(method)

    if not scores_dict:
        raise RuntimeError(
            f"REQUIRED: No score files found! All methods failed: {methods}. "
            f"At least one method must succeed."
        )

    if missing_methods:
        print(f"\nEvaluation proceeding with {len(scores_dict)} of {len(methods)} methods")
        print(f"   Available: {list(scores_dict.keys())}")
        print(f"   Missing: {missing_methods}\n")

    return pd.DataFrame(scores_dict)

def plot_roc_curves(methods_results, sim_label):
    """Plot ROC curves."""
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(methods_results))))
    
    for (method_name, result), color in zip(methods_results.items(), colors):
        y_true = result['y_true']
        y_pred = result['y_pred']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = result['metrics']['auc']['overall']
        ax.plot(fpr, tpr, label=f"{method_name} (AUC={auc:.3f})", color=color, linewidth=2.2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f'ROC Curves - Simulation {sim_label}', fontweight='bold')
    _style_axes(ax)
    ax.legend(loc='lower right', frameon=False)
    plt.savefig(f'{sim_label}_comparison_roc.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_calibration_curves(methods_results, sim_label):
    """Plot calibration curves."""
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(methods_results))))
    
    for (method_name, result), color in zip(methods_results.items(), colors):
        y_true = result['y_true']
        y_pred = result['y_pred']
        bin_edges, observed, predicted = compute_calibration_curve(y_true, y_pred, n_bins=10)
        mask = ~np.isnan(observed)
        ax.plot(predicted[mask], observed[mask], 'o-', label=method_name, color=color, linewidth=2, markersize=4)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title(f'Calibration Curves - Simulation {sim_label}', fontweight='bold')
    _style_axes(ax)
    ax.legend(loc='upper left', frameon=False)
    plt.savefig(f'{sim_label}_comparison_calibration.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_pcs(pc1, pc2, pop_labels, sim_label):
    """Plot PC1 vs PC2 colored by population."""
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    unique_pops = sorted(set(pop_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pops)))

    for pop, color in zip(unique_pops, colors):
        mask = pop_labels == pop
        ax.scatter(pc1[mask], pc2[mask], s=12, alpha=0.7, label=str(pop), color=color, edgecolors="none")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Population Structure (PC1 vs PC2)", fontweight='bold')
    _style_axes(ax)
    ax.legend(loc="best", frameon=False, title="Population")
    plt.savefig(f"{sim_label}_pcs.png", dpi=200, bbox_inches="tight")
    plt.close()

def create_metrics_table(methods_results, sim_label):
    """Create metrics table."""
    rows = []
    for method_name, result in methods_results.items():
        metrics = result['metrics']
        row = {
            'Method': method_name,
            'AUC_overall': metrics['auc']['overall'],
            'Brier_overall': metrics['brier']['overall'],
        }
        for pop in ['AFR', 'EUR', 'ASIA', 'ADMIX']:
            if pop in metrics['auc']:
                row[f'AUC_{pop}'] = metrics['auc'][pop]
                row[f'Brier_{pop}'] = metrics['brier'][pop]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f'{sim_label}_metrics.csv', index=False)
    return df

def plot_auc_summary(methods_results, pvals_df, sim_label):
    """Plot AUC summary with p-values vs best method."""
    if not methods_results:
        return

    auc_rows = []
    for method_name, result in methods_results.items():
        auc_rows.append((method_name, result["metrics"]["auc"]["overall"]))

    auc_df = pd.DataFrame(auc_rows, columns=["Method", "AUC"]).sort_values("AUC", ascending=False)
    best_method = auc_df.iloc[0]["Method"]

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, max(5, len(auc_df) * 0.6)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(auc_df)))
    ax.bar(auc_df["Method"], auc_df["AUC"], color=colors, edgecolor="#333333", linewidth=0.6)

    max_auc = auc_df["AUC"].max() if not auc_df.empty else 1.0
    y_offset = max(0.01, max_auc * 0.02)

    for i, (method, auc) in enumerate(auc_df.values):
        ax.text(i, auc + y_offset, f"{auc:.3f}", ha="center", va="bottom", fontsize=9)

        if pvals_df is not None and method != best_method:
            p_row = pvals_df[
                ((pvals_df["Method_1"] == best_method) & (pvals_df["Method_2"] == method)) |
                ((pvals_df["Method_2"] == best_method) & (pvals_df["Method_1"] == method))
            ]
            if not p_row.empty:
                pval = p_row.iloc[0]["AUC_p_value"]
                if pd.notna(pval):
                    ax.text(
                        i,
                        auc + y_offset * 2.5,
                        f"p={pval:.3g}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold" if pval < 0.05 else "normal",
                    )

    ax.set_ylim(0, min(1.0, max_auc + y_offset * 6))
    ax.set_ylabel("AUC (overall)")
    ax.set_title(
        f"AUC Summary - Simulation {sim_label} (p-values vs {best_method})",
        fontweight="bold",
    )
    _style_axes(ax)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{sim_label}_comparison_auc.png", dpi=200, bbox_inches="tight")
    plt.close()

def plot_brier_summary(methods_results, pvals_df, sim_label):
    """Plot Brier summary with p-values vs best (lowest) method."""
    if not methods_results:
        return

    brier_rows = []
    for method_name, result in methods_results.items():
        brier_rows.append((method_name, result["metrics"]["brier"]["overall"]))

    brier_df = pd.DataFrame(brier_rows, columns=["Method", "Brier"]).sort_values("Brier", ascending=True)
    best_method = brier_df.iloc[0]["Method"]

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10, max(5, len(brier_df) * 0.6)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(brier_df)))
    ax.bar(brier_df["Method"], brier_df["Brier"], color=colors, edgecolor="#333333", linewidth=0.6)

    min_brier = brier_df["Brier"].min() if not brier_df.empty else 0.0
    y_offset = max(0.001, min_brier * 0.02)

    for i, (method, brier) in enumerate(brier_df.values):
        ax.text(i, brier + y_offset, f"{brier:.3f}", ha="center", va="bottom", fontsize=9)

        if pvals_df is not None and method != best_method:
            p_row = pvals_df[
                ((pvals_df["Method_1"] == best_method) & (pvals_df["Method_2"] == method)) |
                ((pvals_df["Method_2"] == best_method) & (pvals_df["Method_1"] == method))
            ]
            if not p_row.empty:
                pval = p_row.iloc[0]["Brier_p_value"]
                if pd.notna(pval):
                    ax.text(
                        i,
                        brier + y_offset * 2.5,
                        f"p={pval:.3g}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold" if pval < 0.05 else "normal",
                    )

    ax.set_ylabel("Brier (overall)")
    ax.set_title(
        f"Brier Summary - Simulation {sim_label} (p-values vs {best_method})",
        fontweight="bold",
    )
    _style_axes(ax)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{sim_label}_comparison_brier.png", dpi=200, bbox_inches="tight")
    plt.close()

def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong's test for comparing two correlated ROC curves.

    Returns p-value for two-sided test of AUC1 != AUC2.
    Uses the method from DeLong et al. (1988).
    """
    from sklearn.metrics import roc_auc_score

    n = len(y_true)

    # Compute AUCs
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)

    # Structural components
    def structural_components(y_true, y_pred):
        """Compute structural components for AUC variance estimation."""
        order = np.argsort(y_pred)
        y_sorted = y_true[order]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return np.nan, np.nan

        # Rank each observation
        ranks = np.arange(1, n + 1)

        # V10: structural component for positives
        pos_ranks = ranks[order][y_sorted == 1]
        V10 = np.mean(pos_ranks) - (n_pos + 1) / 2
        V10 = V10 / n_neg

        # V01: structural component for negatives
        neg_ranks = ranks[order][y_sorted == 0]
        V01 = np.mean(n + 1 - neg_ranks) - (n_neg + 1) / 2
        V01 = V01 / n_pos

        return V10, V01

    V10_1, V01_1 = structural_components(y_true, y_pred1)
    V10_2, V01_2 = structural_components(y_true, y_pred2)

    if np.isnan(V10_1) or np.isnan(V10_2):
        return np.nan

    # Covariance estimation (simplified - assumes independence for now)
    # Full DeLong requires computing covariance between structural components
    # Using permutation test as fallback for robustness
    p_auc, _ = permutation_test_auc_brier(y_true, y_pred1, y_pred2)
    return p_auc

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

def _permute_chunk(args):
    y_true, base, diff, n_iters, seed = args
    rng = np.random.default_rng(seed)
    n = y_true.size
    y_true_f = y_true.astype(np.float64, copy=False)
    auc_diffs = np.empty(n_iters, dtype=np.float64)
    brier_diffs = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        swap = rng.integers(0, 2, size=n, dtype=np.int8)
        perm_pred1 = base + swap * diff
        perm_pred2 = base + (1 - swap) * diff

        auc1 = fast_auc(y_true, perm_pred1)
        auc2 = fast_auc(y_true, perm_pred2)
        auc_diffs[i] = abs(auc1 - auc2)

        brier1 = np.mean((y_true_f - perm_pred1) ** 2)
        brier2 = np.mean((y_true_f - perm_pred2) ** 2)
        brier_diffs[i] = abs(brier1 - brier2)

    return auc_diffs, brier_diffs

def permutation_test_auc_brier(y_true, y_pred1, y_pred2, n_permutations=N_PERMUTATIONS):
    """
    Joint permutation test for AUC and Brier using matched-pairs swapping.
    Returns (p_auc, p_brier).
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    auc1 = fast_auc(y_true, y_pred1)
    auc2 = fast_auc(y_true, y_pred2)
    obs_auc_diff = abs(auc1 - auc2)

    brier1 = np.mean((y_true - y_pred1) ** 2)
    brier2 = np.mean((y_true - y_pred2) ** 2)
    obs_brier_diff = abs(brier1 - brier2)

    base = y_pred2
    diff = y_pred1 - y_pred2

    n = y_true.size
    use_parallel = n_permutations >= 200 and n >= 1000
    if use_parallel:
        workers = min(MAX_WORKERS, max(1, mp.cpu_count() or 1))
        chunk = (n_permutations + workers - 1) // workers
        seeds = np.random.SeedSequence(42).spawn(workers)
        args = []
        for i in range(workers):
            start = i * chunk
            end = min(n_permutations, start + chunk)
            if start >= end:
                break
            args.append((y_true, base, diff, end - start, seeds[i].entropy))

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        with ctx.Pool(processes=len(args)) as pool:
            results = pool.map(_permute_chunk, args)

        auc_diffs = np.concatenate([r[0] for r in results])
        brier_diffs = np.concatenate([r[1] for r in results])
    else:
        auc_diffs, brier_diffs = _permute_chunk((y_true, base, diff, n_permutations, 42))

    p_auc = np.mean(auc_diffs >= obs_auc_diff) if auc_diffs.size else np.nan
    p_brier = np.mean(brier_diffs >= obs_brier_diff) if brier_diffs.size else np.nan
    return p_auc, p_brier

def compute_significance_tests(methods_results, sim_label):
    """
    Compute pairwise significance tests for all method comparisons.

    Tests:
    - AUC: Permutation test
    - Brier: Permutation test

    Returns DataFrame with p-values for all pairwise comparisons.
    """
    method_names = list(methods_results.keys())

    # All pairwise combinations
    pairs = list(combinations(method_names, 2))

    rows = []
    for method1, method2 in pairs:
        y_true1 = methods_results[method1]['y_true']
        y_pred1 = methods_results[method1]['y_pred']
        y_true2 = methods_results[method2]['y_true']
        y_pred2 = methods_results[method2]['y_pred']

        # Ensure same samples
        assert np.array_equal(y_true1, y_true2), "Methods must have same test set"
        y_true = y_true1

        # Compute p-values
        print(f"  Testing {method1} vs {method2}...")
        p_auc, p_brier = permutation_test_auc_brier(y_true, y_pred1, y_pred2, n_permutations=N_PERMUTATIONS)

        # Get observed metrics
        auc1 = methods_results[method1]['metrics']['auc']['overall']
        auc2 = methods_results[method2]['metrics']['auc']['overall']
        brier1 = methods_results[method1]['metrics']['brier']['overall']
        brier2 = methods_results[method2]['metrics']['brier']['overall']

        rows.append({
            'Method_1': method1,
            'Method_2': method2,
            'AUC_1': auc1,
            'AUC_2': auc2,
            'AUC_diff': auc1 - auc2,
            'AUC_p_value': p_auc,
            'Brier_1': brier1,
            'Brier_2': brier2,
            'Brier_diff': brier1 - brier2,
            'Brier_p_value': p_brier,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{sim_label}_significance_tests.csv', index=False)
    return df


def write_probability_outputs(methods_results, sim_label):
    """
    Persist per-individual probabilities for downstream seed aggregation.
    """
    long_rows = []
    for method_name, result in methods_results.items():
        iids = result.get("iid", [])
        pops = result.get("pop_label", [])
        y_true = result["y_true"]
        y_pred = result["y_pred"]
        for iid, pop, y_val, p_val in zip(iids, pops, y_true, y_pred):
            long_rows.append(
                {
                    "Method": method_name,
                    "IID": str(iid),
                    "pop_label": str(pop),
                    "y_true": int(y_val),
                    "y_prob": float(p_val),
                }
            )

    if not long_rows:
        return

    probs_df = pd.DataFrame(long_rows)
    probs_df.to_csv(f"{sim_label}_probabilities.csv", index=False)

    summary = (
        probs_df.groupby(["Method", "pop_label"], as_index=False)
        .agg(
            n=("IID", "count"),
            observed_prevalence=("y_true", "mean"),
            mean_predicted_probability=("y_prob", "mean"),
            auc=("y_prob", lambda s: np.nan),
        )
    )
    for method_name in summary["Method"].unique():
        method_mask = probs_df["Method"] == method_name
        method_df = probs_df.loc[method_mask]
        try:
            auc_val = fast_auc(method_df["y_true"].to_numpy(), method_df["y_prob"].to_numpy())
        except Exception:
            auc_val = np.nan
        summary.loc[summary["Method"] == method_name, "auc"] = auc_val
    summary.to_csv(f"{sim_label}_probability_summary.csv", index=False)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate PRS calibration and metrics.")
    parser.add_argument("sim_name", help="Simulation name/prefix (e.g., confounding, portability)")
    parser.add_argument(
        "--enable-gnomon",
        action="store_true",
        help="Include GAM-gnomon in calibration methods",
    )
    args = parser.parse_args()

    sim_prefix = args.sim_name.strip()
    work_dir = Path(f"{sim_prefix}_work")
    
    # 1. Load Data
    # We need phenotypes and population labels. Best source is the test metadata from split_data?
    # Or reload the TSV? TSV has everything.
    # split_data saves test.phen but that doesn't have pop_label.
    # We will reload sim{id}.tsv and filter for test set.
    # But how do we know who is in test set?
    # Dictionary/List in work_dir/test.keep
    
    kfile = work_dir / "test.keep"
    if not kfile.exists():
        raise FileNotFoundError(f"Missing test.keep at {kfile}")
        
    test_ids = pd.read_csv(kfile, sep='\t', header=None, names=['FID', 'IID'])
    test_iids = test_ids['IID'].astype(str).tolist()
    
    tsv_path = f"{sim_prefix}.tsv"
    full_df = pd.read_csv(tsv_path, sep='\t')
    full_df['individual_id'] = full_df['individual_id'].astype(str)

    if {"pc1", "pc2", "pop_label"}.issubset(full_df.columns):
        plot_pcs(
            full_df["pc1"].to_numpy(),
            full_df["pc2"].to_numpy(),
            full_df["pop_label"].to_numpy(),
            sim_prefix,
        )
    
    # Filter to test set
    test_metadata = full_df[full_df['individual_id'].isin(test_iids)].copy()
    test_metadata = test_metadata.set_index('individual_id')
    
    # Re-order to match keep file if needed, but inner join with scores handles it.
    
    # 2. Load Scores
    default_methods = ['BayesR']
    available_methods = sorted(p.stem for p in work_dir.glob("*.sscore"))
    # Only evaluate known high-level methods; ignore intermediate artifacts.
    known_methods = {'BayesR'}
    methods = [m for m in available_methods if m in known_methods]
    if not methods:
        methods = default_methods
    print(f"Loading scores for: {methods}...")
    prs_scores_df = load_scores(work_dir, methods)
    
    if prs_scores_df.empty:
        raise RuntimeError("No scores loaded. Aborting.")
        
    # Join
    analysis_df = test_metadata.join(prs_scores_df, how='inner')
    print(f"Analysis set size: {len(analysis_df)}")
    
    y_test = analysis_df['y'].values
    pop_test = analysis_df['pop_label'].values
    PC_test = analysis_df[[f'pc{i+1}' for i in range(5)]].values
    
    # 3. Calibration & Metrics
    results = {}
    calibration_methods = [
        RawPGSMethod(),
        LinearInteractionMethod(),
        NormalizationMethod(n_pcs=5),
        GAMMethod(n_pcs=5, k_pgs=10, k_pc=10, use_ti=True),
    ]

    def _method_label(method) -> str:
        if isinstance(method, GAMMethod):
            return "GAM-mgcv"
        if isinstance(method, RawPGSMethod):
            return "Raw"
        if isinstance(method, LinearInteractionMethod):
            return "Linear"
        if isinstance(method, NormalizationMethod):
            return "Normalization"
        return method.name

    for prs_name in prs_scores_df.columns:
        P_raw = analysis_df[prs_name].values
        # Standardize
        if np.std(P_raw) == 0:
            raise RuntimeError(
                f"REQUIRED: {prs_name} has zero variance. "
                f"This indicates a critical failure in PGS calculation. Cannot proceed."
            )
        P_raw = (P_raw - np.mean(P_raw)) / np.std(P_raw)
        
        for cal_method in calibration_methods:
            name = f"{prs_name} + {_method_label(cal_method)}"
            print(f"Evaluating: {name}...")
            
            try:
                # Split Test into Calib/Eval halves
                # Stratify by pop to ensure diversity in both
                idx_cal, idx_val = train_test_split(np.arange(len(y_test)), test_size=0.5, random_state=42, stratify=pop_test)
                
                if isinstance(cal_method, NormalizationMethod):
                    cal_method.set_pop_labels(pop_test[idx_cal])
                cal_method.fit(P_raw[idx_cal], PC_test[idx_cal], y_test[idx_cal])
                if isinstance(cal_method, NormalizationMethod):
                    cal_method.set_pop_labels(pop_test[idx_val])
                y_prob = cal_method.predict_proba(P_raw[idx_val], PC_test[idx_val])
                
                metrics = compute_all_metrics(y_test[idx_val], y_prob, pop_test[idx_val])
                
                results[name] = {
                    'y_true': y_test[idx_val],
                    'y_pred': y_prob,
                    'metrics': metrics,
                    'iid': analysis_df.index.to_numpy()[idx_val],
                    'pop_label': pop_test[idx_val],
                }
            except Exception as e:
                print(f"  Failed to evaluate {name}: {e}")
                # We do NOT fail the whole job here if one calibration fails, 
                # but if ALL fail, that's bad.
                
    if not results:
        raise RuntimeError("No results generated!")
        
    # 4. Significance Testing
    print(f"\nComputing significance tests (permutation tests, {N_PERMUTATIONS} iterations)...")
    df_significance = compute_significance_tests(results, sim_prefix)
    print("\nPairwise Significance Tests:")
    print(df_significance.to_string(index=False))

    # 5. Plots and Table
    print("Generating plots and tables...")
    plot_roc_curves(results, sim_prefix)
    plot_calibration_curves(results, sim_prefix)
    plot_auc_summary(results, df_significance, sim_prefix)
    plot_brier_summary(results, df_significance, sim_prefix)
    df_metrics = create_metrics_table(results, sim_prefix)
    write_probability_outputs(results, sim_prefix)
    print(df_metrics.to_string(index=False))

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()
