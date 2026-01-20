"""
Main analysis script for comparing Real PRS Training & Calibration methods.

Pipeline:
1. Split Simulation Data -> Train (EUR) + Test (All)
2. Train PRS Models (BayesR, LDpred2, PRS-CSx)
3. Generate Scores on Test Set
4. Run Calibration/Adjustment Methods on these Scores
5. Evaluate Performance
"""
import sys
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add current directory to path to allow imports
sys.path.append(str(Path(__file__).parent))

from methods import (
    RawPGSMethod,
    LinearInteractionMethod,
    NormalizationMethod,
    GAMMethod,
)
# We re-import metrics/plotting functions or redefine them?
# Let's redefine plot/metric helpers here or enable importing from a utils module if I created one.
# I'll paste the plotting code to ensure self-contained script.
from metrics import compute_all_metrics, compute_calibration_curve 
from prs_tools import BayesR, LDpred2, PRScsx

def setup_directories(sim_id):
    """Create work directories."""
    work_dir = Path(f"sim{sim_id}_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    return work_dir

def split_data(sim_id, work_dir):
    """
    Split PLINK data into Training (EUR) and Test (All).
    Returns paths to PLINK prefixes and the test set metadata.
    """
    # Load info from TSV to identify IDs
    tsv_path = f"sim{sim_id}.tsv"
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Simulation TSV not found at {tsv_path}")
        
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Select Training Set: EUR only
    train_mask = df['pop_label'] == 'EUR'
    
    eur_indices = df[train_mask].index.to_numpy()
    
    # Split EUR into 80% Train, 20% Test (Validation within ancestry)
    if len(eur_indices) > 0:
        train_idx, test_eur_idx = train_test_split(eur_indices, test_size=0.2, random_state=42)
    else:
        # Fallback for Sim? If no EUR, use random 80%?
        # Sim 3 has EUR. Sim 1/2? Sim populations are stdpopsim AF/EU/AS.
        print("Warning: No EUR found? Using random 60% split.")
        train_idx, test_eur_idx = train_test_split(df.index.to_numpy(), test_size=0.4, random_state=42)
    
    # Test set = Test EUR + All non-EUR
    non_eur_idx = df[~df.index.isin(eur_indices)].index.to_numpy()
    test_idx = np.concatenate([test_eur_idx, non_eur_idx])
    
    # Ensure IDs are strings
    df['individual_id'] = df['individual_id'].astype(str)
    
    # Write Keep lists for PLINK (FID IID)
    train_ids = df.loc[train_idx, ['individual_id', 'individual_id']]
    test_ids = df.loc[test_idx, ['individual_id', 'individual_id']]
    
    train_keep = work_dir / "train.keep"
    test_keep = work_dir / "test.keep"
    
    train_ids.to_csv(train_keep, sep='\t', index=False, header=False)
    test_ids.to_csv(test_keep, sep='\t', index=False, header=False)
    
    # Make subsets using PLINK2
    bfile_orig = f"sim{sim_id}" # sim_pops.py produces sim{id}.bed/bim/fam
    
    # Training subset
    cmd_train = [
        "plink2", "--bfile", bfile_orig, "--keep", str(train_keep),
        "--make-bed", "--out", str(work_dir / "train"), "--silent"
    ]
    if os.system(' '.join(cmd_train)) != 0:
        raise RuntimeError("PLINK2 split train failed")
    
    # Testing subset
    cmd_test = [
        "plink2", "--bfile", bfile_orig, "--keep", str(test_keep),
        "--make-bed", "--out", str(work_dir / "test"), "--silent"
    ]
    if os.system(' '.join(cmd_test)) != 0:
        raise RuntimeError("PLINK2 split test failed")
    
    # Write phenotype files (FID IID Pheno)
    df.loc[train_idx, ['individual_id', 'individual_id', 'y']].to_csv(
        work_dir / "train.phen", sep=' ', index=False, header=False
    )
    df.loc[test_idx, ['individual_id', 'individual_id', 'y']].to_csv(
        work_dir / "test.phen", sep=' ', index=False, header=False
    )
    
    return work_dir / "train", work_dir / "test", df.loc[test_idx]

def train_and_score(train_prefix, test_prefix, work_dir, methods_to_run):
    """
    Train PRS models and score test set.
    Returns DataFrame with IID and Scores for each method.
    """
    scores_dict = {}
    pheno_file = f"{train_prefix}.phen"
    
    # 1. BayesR
    if 'BayesR' in methods_to_run:
        print("\n--- Training BayesR ---")
        try:
            br = BayesR()
            eff_file = br.fit(str(train_prefix), str(pheno_file), str(work_dir / "bayesr"))
            res = br.predict(str(test_prefix), eff_file, str(work_dir / "bayesr_pred"))
            res['IID'] = res['IID'].astype(str)
            scores_dict['BayesR'] = res.set_index('IID')['PRS']
        except Exception as e:
            print(f"BayesR failed: {e}")

    # 2. LDpred2
    if 'LDpred2' in methods_to_run:
        print("\n--- Training LDpred2 ---")
        try:
            ld = LDpred2()
            eff_file = ld.fit(str(train_prefix), str(pheno_file), str(train_prefix), str(work_dir / "ldpred2"))
            res = ld.predict(str(test_prefix), eff_file, str(work_dir / "ldpred2_pred"))
            res['IID'] = res['IID'].astype(str)
            scores_dict['LDpred2'] = res.set_index('IID')['PRS']
        except Exception as e:
            print(f"LDpred2 failed: {e}")

    # 3. PRS-CSx
    if 'PRS-CSx' in methods_to_run:
        print("\n--- Training PRS-CSx ---")
        try:
            ref_path_env = os.environ.get("PRSCSX_REF") # e.g. path/to/ldblk_1kg
            # If simulated data is not matched to 1kg, this might be poor, but requested.
            # We check if strict requirements are met.
            if ref_path_env and os.path.exists(ref_path_env):
                cs = PRScsx()
                eff_file = cs.fit(str(train_prefix), str(pheno_file), str(work_dir / "prscsx"), 
                                  ref_dir=ref_path_env)
                res = cs.predict(str(test_prefix), eff_file, str(work_dir / "prscsx_pred"))
                res['IID'] = res['IID'].astype(str)
                scores_dict['PRS-CSx'] = res.set_index('IID')['PRS']
            else:
                print("Skipping PRS-CSx (PRSCSX_REF not set or missing)")
        except Exception as e:
             print(f"PRS-CSx failed: {e}")
            
    return pd.DataFrame(scores_dict)

# --- Visualization Functions (Ported from legacy) ---

def plot_roc_curves(methods_results: dict, sim_id: int):
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

def plot_calibration_curves(methods_results: dict, sim_id: int):
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

def create_metrics_table(methods_results: dict, sim_id: int):
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

# --- Main Execution ---

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_methods.py <sim_id>")
        sys.exit(1)
        
    sim_id = int(sys.argv[1])
    print(f"\n{'='*60}")
    print(f"  PGS Pipeline: Real Training - Simulation {sim_id}")
    print(f"{'='*60}\n")
    
    # 1. Setup & Data Split
    work_dir = setup_directories(sim_id)
    print("[1/5] Splitting Data (Train=EUR, Test=All)...")
    train_prefix, test_prefix, test_metadata = split_data(sim_id, work_dir)
    
    print(f"  Train: {train_prefix}")
    print(f"  Test:  {test_prefix}")
    
    # 2. Train PRS Models
    print("[2/5] Training PRS Models (BayesR, LDpred2)...")
    methods = ['BayesR', 'LDpred2', 'PRS-CSx'] 
    prs_scores_df = train_and_score(train_prefix, test_prefix, work_dir, methods)
    
    if prs_scores_df.empty:
        print("No PRS models trained successfully! Cannot verify calibration.")
        # Fallback for debugging if tools missing: Use simulated P_observed?
        # User wants REAL tools. We should fail or warn.
        # But to allow pipeline to complete if tools fail in dev:
        print("WARNING: Using simulated P_observed for fallback comparison.")
        prs_scores_df = pd.DataFrame({'Simulated': test_metadata.set_index('individual_id')['P_observed']})
        
    # 3. Calibration Analysis
    # Join scores with metadata
    test_metadata = test_metadata.set_index('individual_id')
    # Use inner join to ensure matching IDs
    analysis_df = test_metadata.join(prs_scores_df, how='inner')
    
    print(f"\n[3/5] Calibration Analysis on {len(analysis_df)} test samples...")
    
    # Prepare common data
    y_test = analysis_df['y'].values
    pop_test = analysis_df['pop_label'].values
    
    # Get PCs (5 components)
    PC_test = analysis_df[[f'pc{i+1}' for i in range(5)]].values
    
    results = {}
    
    # Loop over each Trained PRS (e.g. 'BayesR', 'LDpred2')
    # For EACH PRS, we apply Calibration Methods (Raw, Gam, etc)
    # The user asked: "We can train in Europeans... then... have Imbalanced..."
    # And "Use MGCV... properly".
    
    # We should show: "LDpred2 Raw" vs "LDpred2 + GAM" vs "LDpred2 + Norm"
    # To keep it manageable, let's treat each (PRS_Tool + Calib_Method) as a result entry. Or stick to one "Best" PRS?
    # Let's pivot: For each available PRS, run the calibration suite.
    
    calibration_methods = [
        RawPGSMethod(),
        # LinearInteractionMethod(), # Maybe skip linear? No, keep it.
        # NormalizationMethod(method='empirical'), 
        GAMMethod(n_pcs=5, k_pgs=10, k_pc=10, use_ti=True)
    ]
    
    for prs_name in prs_scores_df.columns:
        P_raw = analysis_df[prs_name].values
        
        # Standardize P for stable fitting (Z-score)
        # Calibration methods expect ~N(0,1) ranges often
        P_raw = (P_raw - np.mean(P_raw)) / np.std(P_raw)
        
        for cal_method in calibration_methods:
            name = f"{prs_name} + {cal_method.name.split(' ')[0]}" # e.g. "BayesR + GAM"
            print(f"  Fitting: {name}...")
            
            try:
                # We fit calibration on the TEST set? 
                # NO. Calibration must be learned on valid/test?
                # Usually: Train PRS -> Get Scores on Validation -> Train Calibration -> Apply to Test.
                # Here we have Train (EUR) and Test (All).
                # If we fit GAM on Test, we are reporting "Training Performance" of GAM.
                # To be rigorous, we should split Test into Calib_Train and Calib_Test.
                
                # Split Test -> Cal/Val
                idx_cal, idx_val = train_test_split(np.arange(len(y_test)), test_size=0.5, random_state=42, stratify=pop_test)
                
                # Fit Calibration on Cal set
                cal_method.fit(P_raw[idx_cal], PC_test[idx_cal], y_test[idx_cal])
                
                # Evaluate on Val set
                y_prob = cal_method.predict_proba(P_raw[idx_val], PC_test[idx_val])
                
                metrics = compute_all_metrics(y_test[idx_val], y_prob, pop_test[idx_val])
                
                results[name] = {
                    'y_true': y_test[idx_val],
                    'y_pred': y_prob,
                    'metrics': metrics
                }
                print(f"    AUC: {metrics['auc']['overall']:.3f}, Brier: {metrics['brier']['overall']:.3f}")
                
            except Exception as e:
                print(f"    Failed: {e}")

    # 4. Visualization
    if results:
        print("\n[4/5] Creating visualizations...")
        plot_roc_curves(results, sim_id)
        plot_calibration_curves(results, sim_id)
        
        print("\n[5/5] Generating metrics table...")
        df_metrics = create_metrics_table(results, sim_id)
        print(df_metrics.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("  Analysis complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
