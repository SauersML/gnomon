"""
Main analysis script for comparing PGS calibration methods.

Usage:
    python sims/analyze_methods.py <sim_id>
    
Example:
    python sims/analyze_methods.py 1
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

from methods import (
    RawPGSMethod,
    LinearInteractionMethod,
    NormalizationMethod,
    GAMMethod,
)
from metrics import compute_all_metrics, compute_calibration_curve


def load_simulation_data(sim_id: int) -> pd.DataFrame:
    """Load simulation TSV file."""
    path = Path(f"sim{sim_id}.tsv")
    if not path.exists():
        raise FileNotFoundError(f"Simulation file not found: {path}")
    
    df = pd.read_csv(path, sep='\t')
    print(f"Loaded {len(df)} individuals from {path}")
    print(f"Populations: {df['pop_label'].value_counts().to_dict()}")
    print(f"Prevalence: {df['y'].mean():.3f}")
    return df


def prepare_data(df: pd.DataFrame, n_pcs: int = 2):
    """
    Extract P, PC, y arrays from dataframe.
    
    Returns
    -------
    P : np.ndarray
        Polygenic scores
    PC : np.ndarray
        Principal components
    y : np.ndarray
        Binary phenotype
    pop_labels : np.ndarray
        Population labels
    """
    P = df['P_observed'].values
    PC = df[[f'pc{i+1}' for i in range(n_pcs)]].values
    y = df['y'].values
    pop_labels = df['pop_label'].values
    
    return P, PC, y, pop_labels


def plot_roc_curves(methods_results: dict, sim_id: int):
    """Plot ROC curves for all methods."""
    from sklearn.metrics import roc_curve
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_results)))
    
    for (method_name, result), color in zip(methods_results.items(), colors):
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = result['metrics']['auc']['overall']
        
        ax.plot(fpr, tpr, label=f"{method_name} (AUC={auc:.3f})", 
                color=color, linewidth=2)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title(f'ROC Curves - Simulation {sim_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'sim{sim_id}_comparison_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sim{sim_id}_comparison_roc.png")


def plot_calibration_curves(methods_results: dict, sim_id: int):
    """Plot calibration curves for all methods."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods_results)))
    
    for (method_name, result), color in zip(methods_results.items(), colors):
        y_true = result['y_true']
        y_pred = result['y_pred']
        
        bin_edges, observed, predicted = compute_calibration_curve(y_true, y_pred, n_bins=10)
        
        # Remove NaN bins
        mask = ~np.isnan(observed)
        
        ax.plot(predicted[mask], observed[mask], 
                'o-', label=method_name, color=color, linewidth=2, markersize=8)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
    
    ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax.set_ylabel('Observed Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Calibration Curves - Simulation {sim_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'sim{sim_id}_comparison_calibration.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sim{sim_id}_comparison_calibration.png")


def create_metrics_table(methods_results: dict, sim_id: int):
    """Create and save metrics comparison table."""
    rows = []
    
    for method_name, result in methods_results.items():
        metrics = result['metrics']
        
        # Overall metrics
        row = {
            'Method': method_name,
            'AUC_overall': metrics['auc']['overall'],
            'Brier_overall': metrics['brier']['overall'],
        }
        
        # Population-specific AUCs
        for pop in ['AFR', 'EUR', 'ASIA', 'ADMIX']:
            if pop in metrics['auc']:
                row[f'AUC_{pop}'] = metrics['auc'][pop]
            if pop in metrics['brier']:
                row[f'Brier_{pop}'] = metrics['brier'][pop]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(f'sim{sim_id}_metrics.csv', index=False)
    print(f"  Saved: sim{sim_id}_metrics.csv")
    print("\nMetrics Summary:")
    print(df.to_string(index=False))
    return df


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_methods.py <sim_id>")
        print("Example: python analyze_methods.py 1")
        sys.exit(1)
    
    sim_id = int(sys.argv[1])
    print(f"\n{'='*60}")
    print(f"  PGS Method Comparison - Simulation {sim_id}")
    print(f"{'='*60}\n")
    
    # Load data
    print("[1/5] Loading simulation data...")
    df = load_simulation_data(sim_id)
    
    # Prepare arrays
    print("[2/5] Preparing data...")
    P, PC, y, pop_labels = prepare_data(df, n_pcs=2)
    
    # Split train/test
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=pop_labels
    )
    
    P_train, P_test = P[train_idx], P[test_idx]
    PC_train, PC_test = PC[train_idx], PC[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    pop_train, pop_test = pop_labels[train_idx], pop_labels[test_idx]
    
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Test:  {len(test_idx)} samples")
    
    # Define methods to compare
    print("\n[3/5] Fitting methods...")
    methods = [
        RawPGSMethod(),
        LinearInteractionMethod(),
        NormalizationMethod(method='empirical'),
        NormalizationMethod(method='mean'),
        NormalizationMethod(method='mean+var'),
        GAMMethod(
            n_pcs=2,
            k_pgs=10,        # Spline basis dimension for PGS
            k_pc=10,         # Spline basis dimension for PCs
            k_interaction=5, # Interaction term basis dimension
            method='REML',   # Use REML for smoothing parameter selection
            use_ti=True,     # Use decomposed form: s() + ti()
        ),
    ]
    
    results = {}
    
    for method in methods:
        print(f"  Fitting: {method.name}...")
        try:
            method.fit(P_train, PC_train, y_train)
            y_pred = method.predict_proba(P_test, PC_test)
            
            metrics = compute_all_metrics(y_test, y_pred, pop_test)
            
            results[method.name] = {
                'y_true': y_test,
                'y_pred': y_pred,
                'pop_labels': pop_test,
                'metrics': metrics,
                'method': method,  # Store method for diagnostics
            }
            
            print(f"    AUC: {metrics['auc']['overall']:.3f}, "
                  f"Brier: {metrics['brier']['overall']:.3f}")
            
            # Print GAM diagnostics if applicable
            if hasattr(method, 'get_edf'):
                try:
                    edf = method.get_edf()
                    print(f"    GAM effective degrees of freedom:")
                    for term, edf_val in edf.items():
                        print(f"      {term}: {edf_val:.2f}")
                except Exception:
                    pass  # Skip if EDF extraction fails
                    
        except Exception as e:
            print(f"    ERROR: {e}")
    
    if not results:
        print("\nNo methods successfully fitted!")
        sys.exit(1)
    
    # Create visualizations
    print("\n[4/5] Creating visualizations...")
    plot_roc_curves(results, sim_id)
    plot_calibration_curves(results, sim_id)
    
    # Create metrics table
    print("\n[5/5] Generating metrics table...")
    create_metrics_table(results, sim_id)
    
    print(f"\n{'='*60}")
    print("  Analysis complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
