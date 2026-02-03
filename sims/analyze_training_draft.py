"""
Main analysis script for comparing Real PRS Training & Calibration methods.
"""
import sys
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

from methods import (
    RawPGSMethod,
    LinearInteractionMethod,
    NormalizationMethod,
    GAMMethod,
)
from metrics import compute_all_metrics, compute_calibration_curve
from prs_tools import BayesR

def setup_directories(sim_name):
    """Create work directories."""
    work_dir = Path(f"{sim_name}_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    return work_dir

def split_data(sim_name, work_dir):
    """
    Split PLINK data into Training (EUR) and Test (All).
    Returns paths to PLINK prefixes.
    """
    # Load info from TSV to identify IDs
    df = pd.read_csv(f"{sim_name}.tsv", sep='\t')
    
    # Select Training Set: EUR only? Or Imbalanced?
    # User said: "Train in Europeans SPECIFICALLY"
    train_mask = df['pop_label'] == 'EUR'
    
    # We also want to hold out some EUR for testing calibration within EUR?
    # Actually, usually we have a discovery cohort (Train) and a Target cohort (Test).
    # Let's split EUR into 80% Train, 20% Test.
    # And keep all non-EUR for Test.
    
    eur_indices = df[train_mask].index
    # Stratified split of EUR
    train_idx, test_eur_idx = train_test_split(eur_indices, test_size=0.2, random_state=42)
    
    # Test set = Test EUR + All non-EUR
    non_eur_idx = df[~train_mask].index
    test_idx = np.concatenate([test_eur_idx, non_eur_idx])
    
    # Write Keep lists for PLINK
    # FID IID
    train_ids = df.loc[train_idx, ['individual_id', 'individual_id']]
    test_ids = df.loc[test_idx, ['individual_id', 'individual_id']]
    
    train_keep = work_dir / "train.keep"
    test_keep = work_dir / "test.keep"
    
    train_ids.to_csv(train_keep, sep='\t', index=False, header=False)
    test_ids.to_csv(test_keep, sep='\t', index=False, header=False)
    
    # Make subsets using PLINK2
    bfile_orig = sim_name
    
    # Training
    cmd_train = [
        "plink2", "--bfile", bfile_orig, "--keep", str(train_keep),
        "--make-bed", "--out", str(work_dir / "train"), "--silent"
    ]
    os.system(' '.join(cmd_train))
    
    # Testing
    cmd_test = [
        "plink2", "--bfile", bfile_orig, "--keep", str(test_keep),
        "--make-bed", "--out", str(work_dir / "test"), "--silent"
    ]
    os.system(' '.join(cmd_test))
    
    # Write phenotype files
    # FID IID Pheno
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
    Returns DataFrame with IID and Scores.
    """
    scores = {}
    pheno_file = f"{train_prefix}.phen"
    
    # 1. BayesR
    if 'BayesR' in methods_to_run:
        print("Training BayesR...")
        try:
            br = BayesR()
            eff_file = br.fit(str(train_prefix), str(pheno_file), str(work_dir / "bayesr"))
            res = br.predict(str(test_prefix), eff_file, str(work_dir / "bayesr_pred"))
            scores['BayesR'] = res.set_index('IID')['PRS']
        except Exception as e:
            print(f"BayesR failed: {e}")

    # Only BayesR is supported now.
            
    return pd.DataFrame(scores)

# Recycled Plotting/Metrics functions... (omitted for brevity, will import or redefine)
# For now, just focus on the flow.

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_methods.py <sim_name>")
        sys.exit(1)
        
    sim_name = sys.argv[1].strip()
    print(f"\nPGS Pipeline: Real Training - Simulation {sim_name}\n")
    
    # Setup
    work_dir = setup_directories(sim_name)
    
    # Data Split
    print("[1/4] Splitting Data (Train=EUR, Test=All)...")
    train_prefix, test_prefix, test_meta = split_data(sim_name, work_dir)
    
    # Train PRS
    print("[2/4] Training PRS Models (BayesR)...")
    methods = ['BayesR']
    prs_scores = train_and_score(train_prefix, test_prefix, work_dir, methods)
    
    if prs_scores.empty:
        print("No PRS models trained successfully!")
        sys.exit(1)
        
    # Merge scores with metadata (PC, y, pop)
    # test_meta has index from original DF, but 'individual_id' matches IID
    test_meta = test_meta.set_index('individual_id')
    
    analysis_df = test_meta.join(prs_scores, how='inner')
    print(f"Test Set: {len(analysis_df)} samples")
    
    # For each PRS Method, Run Calibration Comparison
    from analyze_methods_legacy import (
          plot_roc_curves, plot_calibration_curves, 
          create_metrics_table, compute_all_metrics
    ) # We will repurpose the old script functions or copy them
    
    # Wait, I am overwriting analyze_methods.py. I should copy the helper functions first.
    pass 
