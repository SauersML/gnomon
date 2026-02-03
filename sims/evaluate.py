
"""
Evaluate PRS models: Calibration, AUC, and Plots.
Usage: python evaluate.py <sim_id>
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import stats
from itertools import combinations

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from methods import (
    RawPGSMethod,
    GAMMethod,
    GnomonGAMMethod,
    LinearInteractionMethod,
    NormalizationMethod,
)
from metrics import compute_all_metrics, compute_calibration_curve

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
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_results)))
    
    for (method_name, result), color in zip(methods_results.items(), colors):
        y_true = result['y_true']
        y_pred = result['y_pred']
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = result['metrics']['auc']['overall']
        ax.plot(fpr, tpr, label=f"{method_name} (AUC={auc:.3f})", color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_title(f'ROC Curves - Simulation {sim_label}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{sim_label}_comparison_roc.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_calibration_curves(methods_results, sim_label):
    """Plot calibration curves."""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_results)))
    
    for (method_name, result), color in zip(methods_results.items(), colors):
        y_true = result['y_true']
        y_pred = result['y_pred']
        bin_edges, observed, predicted = compute_calibration_curve(y_true, y_pred, n_bins=10)
        mask = ~np.isnan(observed)
        ax.plot(predicted[mask], observed[mask], 'o-', label=method_name, color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_title(f'Calibration Curves - Simulation {sim_label}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{sim_label}_comparison_calibration.png', dpi=150, bbox_inches='tight')
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

    fig, ax = plt.subplots(figsize=(10, max(5, len(auc_df) * 0.6)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(auc_df)))
    ax.bar(auc_df["Method"], auc_df["AUC"], color=colors, edgecolor="black", linewidth=0.5)

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
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(f"{sim_label}_comparison_auc.png", dpi=150, bbox_inches="tight")
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
    return permutation_test_auc(y_true, y_pred1, y_pred2)

def permutation_test_auc(y_true, y_pred1, y_pred2, n_permutations=2000):
    """
    Permutation test for comparing AUCs.

    Two-sided test: H0: AUC1 = AUC2
    """
    from sklearn.metrics import roc_auc_score

    # Observed difference
    try:
        auc1 = roc_auc_score(y_true, y_pred1)
        auc2 = roc_auc_score(y_true, y_pred2)
    except:
        return np.nan

    obs_diff = abs(auc1 - auc2)

    # Permutation distribution
    null_diffs = []
    for _ in range(n_permutations):
        # Randomly swap predictions for each individual
        swap = np.random.binomial(1, 0.5, size=len(y_true)).astype(bool)
        perm_pred1 = np.where(swap, y_pred2, y_pred1)
        perm_pred2 = np.where(swap, y_pred1, y_pred2)

        try:
            perm_auc1 = roc_auc_score(y_true, perm_pred1)
            perm_auc2 = roc_auc_score(y_true, perm_pred2)
            null_diffs.append(abs(perm_auc1 - perm_auc2))
        except:
            continue

    if len(null_diffs) == 0:
        return np.nan

    # Two-sided p-value
    p_value = np.mean(np.array(null_diffs) >= obs_diff)
    return p_value

def permutation_test_brier(y_true, y_pred1, y_pred2, n_permutations=2000):
    """
    Permutation test for comparing Brier scores.

    Two-sided test: H0: Brier1 = Brier2
    """
    from sklearn.metrics import brier_score_loss

    # Observed difference
    brier1 = brier_score_loss(y_true, y_pred1)
    brier2 = brier_score_loss(y_true, y_pred2)
    obs_diff = abs(brier1 - brier2)

    # Permutation distribution
    null_diffs = []
    for _ in range(n_permutations):
        # Randomly swap predictions
        swap = np.random.binomial(1, 0.5, size=len(y_true)).astype(bool)
        perm_pred1 = np.where(swap, y_pred2, y_pred1)
        perm_pred2 = np.where(swap, y_pred1, y_pred2)

        perm_brier1 = brier_score_loss(y_true, perm_pred1)
        perm_brier2 = brier_score_loss(y_true, perm_pred2)
        null_diffs.append(abs(perm_brier1 - perm_brier2))

    # Two-sided p-value
    p_value = np.mean(np.array(null_diffs) >= obs_diff)
    return p_value

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
        p_auc = permutation_test_auc(y_true, y_pred1, y_pred2, n_permutations=2000)
        p_brier = permutation_test_brier(y_true, y_pred1, y_pred2, n_permutations=2000)

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

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <sim_name|sim_id>")
        sys.exit(1)
        
    sim_arg = sys.argv[1]
    try:
        sim_id = int(sim_arg)
    except ValueError:
        sim_id = None

    sim_prefix = f"sim{sim_id}" if sim_id is not None else sim_arg
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
    
    # Filter to test set
    test_metadata = full_df[full_df['individual_id'].isin(test_iids)].copy()
    test_metadata = test_metadata.set_index('individual_id')
    
    # Re-order to match keep file if needed, but inner join with scores handles it.
    
    # 2. Load Scores
    default_methods = ['BayesR', 'LDpred2', 'PRS-CSx']
    available_methods = sorted(p.stem for p in work_dir.glob("*.sscore"))
    # Only evaluate known high-level methods; ignore intermediate artifacts.
    known_methods = {'BayesR', 'BayesR-Mix', 'LDpred2', 'PRS-CSx'}
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
    try:
        calibration_methods.append(
            GnomonGAMMethod(n_pcs=5, pgs_knots=10, pc_knots=10, no_calibration=True)
        )
    except FileNotFoundError as e:
        print(f"Skipping Gnomon GAM: {e}")
    
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
            name = f"{prs_name} + {cal_method.name.split(' ')[0]}"
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
                    'metrics': metrics
                }
            except Exception as e:
                print(f"  Failed to evaluate {name}: {e}")
                # We do NOT fail the whole job here if one calibration fails, 
                # but if ALL fail, that's bad.
                
    if not results:
        raise RuntimeError("No results generated!")
        
    # 4. Significance Testing
    print("\nComputing significance tests (permutation tests, 2000 iterations)...")
    df_significance = compute_significance_tests(results, sim_prefix)
    print("\nPairwise Significance Tests:")
    print(df_significance.to_string(index=False))

    # 5. Plots and Table
    print("Generating plots and tables...")
    plot_roc_curves(results, sim_prefix)
    plot_calibration_curves(results, sim_prefix)
    plot_auc_summary(results, df_significance, sim_prefix)
    df_metrics = create_metrics_table(results, sim_prefix)
    print(df_metrics.to_string(index=False))

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()
