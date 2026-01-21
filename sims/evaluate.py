
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

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from methods import (
    RawPGSMethod,
    GAMMethod,
)
from metrics import compute_all_metrics, compute_calibration_curve

def load_scores(work_dir, methods):
    scores_dict = {}
    for method in methods:
        score_file = work_dir / f"{method}.sscore"
        if not score_file.exists():
            raise FileNotFoundError(f"CRITICAL: Missing score file for {method} at {score_file}. Did the training job fail?")
        
        df = pd.read_csv(score_file, sep='\t')
        # Expect IID, PRS
        df['IID'] = df['IID'].astype(str)
        scores_dict[method] = df.set_index('IID')['PRS']
        
    return pd.DataFrame(scores_dict)

def plot_roc_curves(methods_results, sim_id):
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
    ax.set_title(f'ROC Curves - Simulation {sim_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'sim{sim_id}_comparison_roc.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_calibration_curves(methods_results, sim_id):
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
    ax.set_title(f'Calibration Curves - Simulation {sim_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'sim{sim_id}_comparison_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_metrics_table(methods_results, sim_id):
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
    df.to_csv(f'sim{sim_id}_metrics.csv', index=False)
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <sim_id>")
        sys.exit(1)
        
    sim_id = int(sys.argv[1])
    work_dir = Path(f"sim{sim_id}_work")
    
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
    
    tsv_path = f"sim{sim_id}.tsv"
    full_df = pd.read_csv(tsv_path, sep='\t')
    full_df['individual_id'] = full_df['individual_id'].astype(str)
    
    # Filter to test set
    test_metadata = full_df[full_df['individual_id'].isin(test_iids)].copy()
    test_metadata = test_metadata.set_index('individual_id')
    
    # Re-order to match keep file if needed, but inner join with scores handles it.
    
    # 2. Load Scores
    methods = ['BayesR', 'LDpred2', 'PRS-CSx']
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
        GAMMethod(n_pcs=5, k_pgs=10, k_pc=10, use_ti=True)
    ]
    
    for prs_name in prs_scores_df.columns:
        P_raw = analysis_df[prs_name].values
        # Standardize
        if np.std(P_raw) == 0:
            print(f"WARNING: {prs_name} has zero variance. Skipping.")
            continue
        P_raw = (P_raw - np.mean(P_raw)) / np.std(P_raw)
        
        for cal_method in calibration_methods:
            name = f"{prs_name} + {cal_method.name.split(' ')[0]}"
            print(f"Evaluating: {name}...")
            
            try:
                # Split Test into Calib/Eval halves
                # Stratify by pop to ensure diversity in both
                idx_cal, idx_val = train_test_split(np.arange(len(y_test)), test_size=0.5, random_state=42, stratify=pop_test)
                
                cal_method.fit(P_raw[idx_cal], PC_test[idx_cal], y_test[idx_cal])
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
        
    # 4. Plots and Table
    print("Generating plots and tables...")
    plot_roc_curves(results, sim_id)
    plot_calibration_curves(results, sim_id)
    df_metrics = create_metrics_table(results, sim_id)
    print(df_metrics.to_string(index=False))
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()
