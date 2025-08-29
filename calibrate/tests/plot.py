#!/usr/bin/env python3
"""
A test script to simulate data, train a GAM model using the 'gnomon' Rust
executable, evaluate its performance against a baseline, and generate
a comprehensive set of analysis plots.
"""

import subprocess
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, log_loss, average_precision_score

# --- Path Configuration ---
# Get the absolute path of the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up to the project root directory (assumes tests/ is under calibrate/)
PROJECT_ROOT = SCRIPT_DIR.parent
# Get the workspace root (one level up from the calibrate directory)
WORKSPACE_ROOT = PROJECT_ROOT.parent

# Define all paths relative to the project root
EXECUTABLE_NAME = "gnomon"
EXECUTABLE_PATH = WORKSPACE_ROOT / "target" / "release" / EXECUTABLE_NAME
MODEL_PATH = PROJECT_ROOT / "model.toml"
PREDICTIONS_PATH = PROJECT_ROOT / "predictions.tsv"  # The fixed output file for the tool
TRAIN_DATA_PATH = PROJECT_ROOT / "training_data.tsv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.tsv"
GRID_DATA_PATH = PROJECT_ROOT / "grid_data_for_surface.tsv"


# --- Simulation Configuration ---
N_SAMPLES_TRAIN = 5000
N_SAMPLES_TEST = 1000
NUM_PCS = 1  # Simulate 1 PC for easy 2D visualization
SIMULATION_SIGNAL_STRENGTH = 0.6 # The coefficient for the true logit function


# --- Helper Functions ---

def build_rust_project():
    """Checks for the executable and compiles the Rust project if not found."""
    if not EXECUTABLE_PATH.is_file():
        print("--- Executable not found. Compiling Rust Project... ---")
        try:
            # Assumes the script is run from a location where `cargo` is in the PATH
            # and the project structure is correct.
            subprocess.run(
                ["cargo", "build", "--release", "--bin", EXECUTABLE_NAME],
                check=True, text=True, cwd=WORKSPACE_ROOT
            )
            print("--- Compilation successful. ---")
            
            # Verify the executable was actually created
            if not EXECUTABLE_PATH.is_file():
                raise FileNotFoundError(f"Compilation succeeded but executable not found at {EXECUTABLE_PATH}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n--- ERROR: Rust compilation failed: {e} ---")
            print("Please ensure Rust/Cargo is installed and in your PATH,")
            print(f"and that the workspace root is correctly set to: {WORKSPACE_ROOT}")
            sys.exit(1)
    else:
        print("--- Found existing executable. Skipping build. ---")


def simulate_data(n_samples: int, seed: int):
    """
    Simulates a dataset with heteroscedastic noise.

    Args:
        n_samples: The number of samples to generate.
        seed: The random seed for reproducibility.

    Returns:
        A pandas DataFrame with columns for 'phenotype', 'score', 'PC1',
        the pure 'signal_prob', and the final 'oracle_prob'.
    """
    np.random.seed(seed)
    score = np.random.uniform(-3, 3, n_samples)
    pc1 = np.random.normal(0, 1.2, n_samples)

    # The true, noise-free relationship between features and the outcome logit
    true_logit = SIMULATION_SIGNAL_STRENGTH * (
        0.8 * np.cos(score * 2) - 1.2 * np.tanh(pc1) +
        1.5 * np.sin(score) * pc1 - 0.5 * (score**2 + pc1**2)
    )

    # Calculate the "signal" probability (the ideal, noise-free probability)
    signal_probability = 1 / (1 + np.exp(-true_logit))

    # --- HETEROSCEDASTIC NOISE MODEL ---
    # The noise level depends on the value of PC1
    base_noise_std = 2.0
    pc1_noise_factor = 0.75
    dynamic_noise_std = base_noise_std + pc1_noise_factor * np.maximum(0, pc1)
    noise = np.random.normal(0, dynamic_noise_std, n_samples)

    # Final "oracle" probability (signal + instance-specific noise) for generating the outcome
    oracle_probability = 1 / (1 + np.exp(-(true_logit + noise)))
    phenotype = np.random.binomial(1, oracle_probability, n_samples)

    df = pd.DataFrame({"phenotype": phenotype, "score": score, "PC1": pc1})
    # Add both the pure signal and the final oracle probabilities for evaluation
    df['signal_prob'] = signal_probability
    df['oracle_prob'] = oracle_probability

    return df


def run_subprocess(command, cwd):
    """Runs a subprocess and handles common errors cleanly."""
    try:
        print(f"Executing: {' '.join(map(str, command))}\n")
        subprocess.run(command, check=True, text=True, cwd=cwd)
    except KeyboardInterrupt:
        print("\n--- Process interrupted by user (Ctrl+C). Cleaning up. ---")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n--- A command FAILED (Exit Code: {e.returncode}) ---")
        print(f"Failed command: {' '.join(map(str, command))}")
        sys.exit(1)


def tjurs_r2(y_true, y_prob):
    """Calculates Tjur's Coefficient of Discrimination (R-squared)."""
    y_true = pd.Series(y_true)
    y_prob = pd.Series(y_prob, index=y_true.index)
    mean_prob_cases = y_prob[y_true == 1].mean()
    mean_prob_controls = y_prob[y_true == 0].mean()
    return mean_prob_cases - mean_prob_controls


def nagelkerkes_r2(y_true, y_prob):
    """Calculates Nagelkerke's Pseudo R-squared."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    epsilon = 1e-15
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)

    # Log-likelihood of the model with only an intercept (null model)
    p_mean = np.mean(y_true)
    log_likelihood_null = np.sum(y_true * np.log(p_mean) + (1 - y_true) * np.log(1 - p_mean))

    # Log-likelihood of the full model
    log_likelihood_model = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    n = len(y_true)
    r2_cs = 1 - np.exp((2/n) * (log_likelihood_null - log_likelihood_model))
    max_r2_cs = 1 - np.exp((2/n) * log_likelihood_null)
    
    return r2_cs / max_r2_cs if max_r2_cs > 0 else 0.0


# --- Robust Calibration Metrics ---

# Small constant to prevent numerical issues
_EPS = 1e-15

def _prep_vec(y, p):
    """Prepare data vectors by converting to proper types and handling NaNs."""
    y = pd.Series(y).astype(float)
    p = pd.Series(p).astype(float).clip(_EPS, 1-_EPS)
    m = (~y.isna()) & (~p.isna())
    return y[m].values, p[m].values

def _ece_from_edges(y, p, edges):
    """Calculate ECE given bin edges, using bin-mass weighting."""
    # Assign points to bins formed by 'edges' in probability space
    ids = np.digitize(p, edges, right=True) - 1
    ids = np.clip(ids, 0, len(edges)-2)
    N = len(p)
    ece = 0.0
    for b in range(len(edges)-1):
        idx = (ids == b)
        n_b = int(idx.sum())
        if n_b == 0:
            continue
        conf = p[idx].mean()
        acc  = y[idx].mean()
        ece += (n_b / N) * abs(acc - conf)
    return ece

def ece_quantile(y_true, y_prob, n_bins=20):
    """Mass-weighted ECE using equal-frequency (quantile) bins.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins to use
        
    Returns:
        Bin-mass-weighted ECE value
    """
    y, p = _prep_vec(y_true, y_prob)
    if len(y) == 0:
        return np.nan
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(p, qs)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    if len(edges) < 2:
        return abs(y.mean() - p.mean())  # degenerate case
    return _ece_from_edges(y, p, edges)

def ece_randomized_quantile(y_true, y_prob, bin_counts=(10,20,40), repeats=50, min_per_bin=20, rng=None):
    """Averaged, bin-insensitive ECE: quantile bins with random offsets, multi-resolution.
    
    This method creates multiple random binnings in rank space and
    averages the ECE across them, making the metric more stable and
    less sensitive to bin boundaries.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        bin_counts: Tuple of bin counts to try for multi-resolution averaging
        repeats: Number of random offsets per bin count
        min_per_bin: Minimum examples per bin (auto-reduces bin count if needed)
        rng: Random number generator or seed
        
    Returns:
        Dict with 'ece_mean', 'ece_std', and other details
    """
    rng = np.random.default_rng(rng)
    y, p = _prep_vec(y_true, y_prob)
    if len(y) == 0:
        return {'ece_mean': np.nan, 'ece_std': np.nan, 'details': {}}
    eces = []
    for M in bin_counts:
        # Ensure enough examples per bin
        M_eff = min(M, max(1, len(y)//max(1, min_per_bin)))
        if M_eff < 2:
            eces.append(abs(y.mean() - p.mean()))
            continue
        for _ in range(repeats):
            delta = rng.uniform(0, 1.0/M_eff)
            qs = (delta + np.arange(M_eff+1)/M_eff).clip(0, 1)
            edges = np.quantile(p, qs)
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.unique(edges)
            if len(edges) < 2:
                eces.append(abs(y.mean() - p.mean()))
                continue
            eces.append(_ece_from_edges(y, p, edges))
    eces = np.array(eces, dtype=float)
    return {
        'ece_mean': float(np.mean(eces)),
        'ece_std':  float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0,
        'details':  {'bin_counts': list(bin_counts), 'repeats': repeats}
    }

def wilson_ci(k, n, z=1.959963984540054):  # 95% CI
    """Calculate Wilson score confidence interval for a binomial proportion."""
    if n == 0: 
        return (np.nan, np.nan)
    p = k/n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half = z*np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return center - half, center + half

def shared_quantile_edges(pred_dict, n_bins=20, rng=123):
    """Build shared bin edges across all models using quantiles."""
    rng = np.random.default_rng(rng)
    all_p = np.concatenate([v for v in pred_dict.values()])
    delta = rng.uniform(0, 1.0/n_bins)
    qs = (delta + np.arange(n_bins+1)/n_bins).clip(0,1)
    edges = np.quantile(all_p, qs)
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)

def bin_stats(y, p, edges):
    """Calculate per-bin statistics given bin edges."""
    ids = np.digitize(p, edges, right=True) - 1
    ids = np.clip(ids, 0, len(edges)-2)
    rows = []
    N = len(p)
    for b in range(len(edges)-1):
        idx = (ids == b)
        n_b = int(idx.sum())
        if n_b == 0: 
            continue
        conf = p[idx].mean()
        acc  = y[idx].mean()
        k = int(y[idx].sum())  # Exact count of positives in bin
        lo, hi = wilson_ci(k, n_b)
        rows.append({
            'bin': b, 'left': edges[b], 'right': edges[b+1], 'n': n_b,
            'mass': n_b/N, 'conf': conf, 'acc': acc, 'lo': lo, 'hi': hi
        })
    return pd.DataFrame(rows)

from sklearn.isotonic import IsotonicRegression

def ici_isotonic(y_true, y_prob):
    """Calculate Integrated Calibration Index using isotonic regression.
    
    This is a bin-free calibration metric measuring the mean absolute
    difference between predictions and an isotonic fit of outcomes.
    """
    y, p = _prep_vec(y_true, y_prob)
    if len(y) < 2:
        return np.nan
    iso = IsotonicRegression(out_of_bounds='clip')
    # Fit expected outcome as function of predicted probability
    m = iso.fit_transform(p, y)
    return float(np.mean(np.abs(m - p)))


def ece_rq_shared_edges(preds_dict, y_true, bin_counts=(10,20,40), repeats=50, min_per_bin=20, rng=None):
    """Calculate randomized quantile ECE using shared bin edges across all models.
    
    This ensures that all models are evaluated on the exact same bin edges,
    making the ECE values strictly comparable across models.
    
    Args:
        preds_dict: Dictionary of model_name -> predicted_probabilities
        y_true: True binary labels
        bin_counts: Tuple of bin counts to try for multi-resolution averaging
        repeats: Number of random offsets per bin count
        min_per_bin: Minimum examples per bin (auto-reduces bin count if needed)
        rng: Random number generator or seed
        
    Returns:
        Dictionary of model_name -> {'ece_mean': float, 'ece_std': float}
    """
    rng = np.random.default_rng(rng)
    y, _ = _prep_vec(y_true, next(iter(preds_dict.values())))
    
    # Pool predictions for edges
    pooled = np.concatenate([_prep_vec(y_true, p)[1] for p in preds_dict.values()])
    results = {}
    
    # Pre-generate all edge sets from pooled predictions
    edge_sets = []
    for M in bin_counts:
        M_eff = min(M, max(1, len(y)//max(1, min_per_bin)))
        if M_eff < 2:
            edge_sets.append([(None, M_eff)])  # marker for degenerate case
            continue
        sets_M = []
        for _ in range(repeats):
            delta = rng.uniform(0, 1.0/M_eff)
            qs = (delta + np.arange(M_eff+1)/M_eff).clip(0,1)
            edges = np.quantile(pooled, qs)
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.unique(edges)
            sets_M.append((edges, M_eff))
        edge_sets.append(sets_M)

    # Evaluate each model on the same edge sets
    for name, p_raw in preds_dict.items():
        y_m, p_m = _prep_vec(y_true, p_raw)
        eces = []
        for sets_M in edge_sets:
            for edges, M_eff in sets_M:
                if edges is None:  # degenerate case
                    eces.append(abs(y_m.mean() - p_m.mean()))
                else:
                    eces.append(_ece_from_edges(y_m, p_m, edges))
        eces = np.array(eces, dtype=float)
        results[name] = {
            'ece_mean': float(np.mean(eces)),
            'ece_std': float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0
        }
    return results


def ece_rq_bootstrap_ci(y_true, y_prob, n_boot=500, alpha=0.05, rng=None):
    """Calculate bootstrap confidence intervals for randomized quantile ECE.
    
    This measures sampling uncertainty in the ECE estimate by bootstrapping
    the data and averaging randomized-quantile ECE for each bootstrap sample.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_boot: Number of bootstrap samples
        alpha: Alpha level for confidence interval (e.g., 0.05 for 95% CI)
        rng: Random number generator or seed
        
    Returns:
        Tuple of (mean_ece, lower_ci, upper_ci)
    """
    rng = np.random.default_rng(rng)
    y, p = _prep_vec(y_true, y_prob)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # Bootstrap sampling with replacement
        e = ece_randomized_quantile(
            y[idx], p[idx], 
            bin_counts=(10,20,40), repeats=30, min_per_bin=20, 
            rng=rng.integers(1<<32)
        )['ece_mean']
        stats.append(e)
    stats = np.array(stats, dtype=float)
    stats = stats[~np.isnan(stats)]  # Remove NaNs
    if len(stats) == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = np.quantile(stats, [alpha/2, 1-alpha/2])
    return (float(np.mean(stats)), float(lo), float(hi))


def brier_decomposition(y_true, y_prob, n_bins=20):
    """Decompose Brier score into reliability, resolution, and uncertainty components.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for grouping predictions
        
    Returns:
        Tuple of (reliability, resolution, uncertainty)
    """
    y, p = _prep_vec(y_true, y_prob)
    
    # Use quantile binning for more balanced bins
    qs = np.linspace(0, 1, n_bins+1)
    edges = np.quantile(p, qs)
    edges[0], edges[-1] = 0.0, 1.0
    edges = np.unique(edges)
    
    ids = np.digitize(p, edges, right=True) - 1
    ids = np.clip(ids, 0, len(edges)-2)
    
    # Overall mean outcome (prevalence)
    p_bar = y.mean()
    reliability = 0.0
    resolution = 0.0
    
    for b in range(len(edges)-1):
        idx = (ids == b)
        n_b = idx.sum()
        if n_b == 0:
            continue
        
        # Mean prediction and outcome in this bin
        conf = p[idx].mean()
        acc = y[idx].mean()
        
        # Bin weight
        w = n_b/len(y)
        
        # Accumulate components
        reliability += w * (conf - acc)**2     # Squared calibration error in bin
        resolution += w * (acc - p_bar)**2     # How far bin's outcomes deviate from overall mean
    
    # Inherent uncertainty in the data
    uncertainty = p_bar * (1 - p_bar)
    
    return reliability, resolution, uncertainty


def generate_performance_report(df_results):
    """Calculates and prints a side-by-side comparison of model metrics."""
    y_true = df_results['phenotype']

    models = {
        "GAM (gnomon)": df_results['prediction'],
        "Baseline (Logistic)": df_results['baseline_prediction'],
        "Signal Model (Noise-Free)": df_results['signal_prob'],
        "Oracle (Instance Best)": df_results['oracle_prob']
    }

    print("\n" + "="*60)
    print("      Model Performance Comparison on Test Set")
    print("="*60)

    print("\n[Discrimination: AUC (Area Under ROC Curve) & PR-AUC]")
    print("Higher is better. ROC-AUC for overall discrimination, PR-AUC for imbalanced data.")
    for name, y_prob in models.items():
        auc = roc_auc_score(y_true, y_prob)
        pr_auc_val = average_precision_score(y_true, y_prob)
        print(f"  - {name:<28}: AUC={auc:.4f} | PR-AUC={pr_auc_val:.4f}")

    print("\n[Probabilistic Accuracy: Brier Score & Log Loss]")
    print("Lower is better. Brier for squared error, Log Loss for likelihood-based assessment.")
    for name, y_prob in models.items():
        brier = brier_score_loss(y_true, y_prob)
        logloss = log_loss(y_true, np.clip(y_prob, _EPS, 1-_EPS))
        print(f"  - {name:<28}: Brier={brier:.4f} | LogLoss={logloss:.4f}")

    print("\n[Fit: Nagelkerke's R-squared]")
    print("Higher is better (0-1). Likelihood-based, common in PGS literature.")
    for name, y_prob in models.items():
        r2_n = nagelkerkes_r2(y_true, y_prob)
        print(f"  - {name:<28}: {r2_n:.4f}")

    print("\n[Separation: Tjur's R-squared]")
    print("Higher is better (0-1). The mean difference in prediction between classes.")
    for name, y_prob in models.items():
        r2_t = tjurs_r2(y_true, y_prob)
        print(f"  - {name:<28}: {r2_t:.4f}")

    print("\n[Calibration: Expected Calibration Error (ECE)]")
    print("Lower is better. Mass-weighted; shared-edges for fair comparison.")
    
    # Calculate shared-edges ECE across all models (strictly comparable)
    shared_ece = ece_rq_shared_edges(models, y_true, 
                                   bin_counts=(10, 20, 40),
                                   repeats=50, min_per_bin=20, rng=123)
    
    # Add bootstrap CI for GAM model
    gam_name = "GAM (gnomon)"
    mean_e, lo_e, hi_e = ece_rq_bootstrap_ci(y_true, models[gam_name], 
                                          n_boot=200, alpha=0.05, rng=123)
    
    for name, y_prob in models.items():
        # Calculate individual metrics
        ece_q = ece_quantile(y_true, y_prob, n_bins=20)
        ici = ici_isotonic(y_true, y_prob)
        shared_result = shared_ece[name]
        
        # Print metrics
        print(f"  - {name:<28}: ECE_q20={ece_q:.4f} | ECE_shared={shared_result['ece_mean']:.4f} ±{shared_result['ece_std']:.4f} | ICI={ici:.4f}")
        
        # Add bootstrap CI for GAM
        if name == gam_name:
            print(f"      Bootstrap 95% CI for {name}: [{lo_e:.4f}, {hi_e:.4f}]")
            
    # Add Brier decomposition
    print("\n[Brier Score Decomposition]")
    print("reliability ↓, resolution ↑, uncertainty (data)")
    for name, y_prob in models.items():
        rel, res, unc = brier_decomposition(y_true, y_prob, n_bins=20)
        brier = brier_score_loss(y_true, y_prob)
        # Verify: brier ≈ rel - res + unc
        print(f"  - {name:<28}: rel={rel:.4f}, res={res:.4f}, unc={unc:.4f} (brier={brier:.4f})")

    print("\n[Confusion Matrices (at threshold=0.5)]")
    for name, y_prob in models.items():
        y_pred_class = (y_prob > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_class)
        cm_proportions = cm / cm.sum()
        print(f"\n  --- {name} ---")
        print("  (Proportions of total data)")
        print(f"             Predicted 0   Predicted 1")
        print(f"    True 0     {cm_proportions[0,0]:<12.3f}  {cm_proportions[0,1]:<12.3f}")
        print(f"    True 1     {cm_proportions[1,0]:<12.3f}  {cm_proportions[1,1]:<12.3f}")

    print("\n" + "="*60)


def create_analysis_plots(results_df, gam_surface, signal_surface, score_grid, pc1_grid):
    """Generates a 2x2 grid of plots for model analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle("Comprehensive GAM Model Analysis vs. Baseline and Oracle", fontsize=22, y=0.98)

    vmin = min(signal_surface.min(), gam_surface.min())
    vmax = max(signal_surface.max(), gam_surface.max())

    # --- 1. Top-Left: Ground Truth (Signal) Surface ---
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(score_grid, pc1_grid, signal_surface, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(contour1, ax=ax1, label="True Signal Probability")
    ax1.set_title("1. Ground Truth (Noise-Free) Signal Surface", fontsize=16)
    ax1.set_xlabel("Polygenic Score (score)")
    ax1.set_ylabel("Principal Component 1 (PC1)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 2. Top-Right: GAM's Learned Surface with Test Points Overlay ---
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(score_grid, pc1_grid, gam_surface, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(contour2, ax=ax2, label="GAM Predicted Probability")
    ax2.scatter(
        results_df["score"], results_df["PC1"], c=results_df["prediction"],
        cmap="viridis", edgecolor="k", linewidth=0.5, s=40, vmin=vmin, vmax=vmax,
        alpha=0.8
    )
    ax2.set_title("2. GAM's Learned Surface with Predictions Overlay", fontsize=16)
    ax2.set_xlabel("Polygenic Score (score)")
    ax2.set_ylabel("Principal Component 1 (PC1)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- 3. Bottom-Left: Improved Calibration Curve Comparison ---
    ax3 = axes[1, 0]
    y_true = results_df['phenotype'].values
    
    # Prepare predictions dictionary for all models
    preds = {
        "Oracle":   results_df['oracle_prob'].values,
        "Signal":   results_df['signal_prob'].values,
        "GAM":      results_df['prediction'].values,
        "Baseline": results_df['baseline_prediction'].values,
    }
    
    # Calculate shared quantile bin edges across all models
    edges = shared_quantile_edges(preds, n_bins=20, rng=123)
    
    # Define colors and z-order for consistent appearance
    colors = {"Oracle":"gold", "Signal":"darkorange", "GAM":"blue", "Baseline":"red"}
    zord = {"Oracle":5, "Signal":4, "GAM":3, "Baseline":2}
    
    # Plot each model's calibration curve with error bars
    for name, p in preds.items():
        # Calculate bin statistics with Wilson CIs
        dfb = bin_stats(y_true, p, edges)
        
        # Calculate metrics for label
        ece_rq = ece_randomized_quantile(y_true, p, bin_counts=(10,20,40), repeats=50, rng=123)
        ici = ici_isotonic(y_true, p)
        
        # Plot with error bars
        ax3.errorbar(
            dfb['conf'], dfb['acc'], 
            yerr=[dfb['acc']-dfb['lo'], dfb['hi']-dfb['acc']],
            fmt='o-', capsize=3, lw=1.5, 
            label=f"{name} (ECE={ece_rq['ece_mean']:.3f}, ICI={ici:.3f})", 
            color=colors[name], 
            zorder=zord[name]
        )
    
    # Add perfect calibration line
    ax3.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", zorder=1)
    
    # Set title and labels
    ax3.set_title("3. Reliability Diagram (shared quantile bins, 95% CI)", fontsize=16)
    ax3.set_xlabel("Mean predicted probability (per bin)")
    ax3.set_ylabel("Empirical frequency (per bin)")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper left')
    
    # Add bin-mass bars using twin axis (using GAM as reference model)
    df_mass = bin_stats(y_true, preds['GAM'], edges)
    ax3b = ax3.twinx()
    centers = (df_mass['left'] + df_mass['right'])/2
    widths = df_mass['right'] - df_mass['left']
    ax3b.bar(centers, df_mass['mass'], width=widths, alpha=0.15, color='gray', edgecolor='none')
    ax3b.set_ylabel("Bin mass (GAM)")
    ax3b.set_ylim(0, max(df_mass['mass'])*1.6 if len(df_mass)>0 else 1.0)

    # --- 4. Bottom-Right: Prediction Distribution by Class (GAM Model) ---
    ax4 = axes[1, 1]
    cases = results_df[results_df['phenotype'] == 1]
    controls = results_df[results_df['phenotype'] == 0]
    ax4.hist(controls['prediction'], bins=30, density=True, alpha=0.7, label="Controls (Phenotype=0)", color='orange')
    ax4.hist(cases['prediction'], bins=30, density=True, alpha=0.7, label="Cases (Phenotype=1)", color='green')
    ax4.set_title("4. GAM Prediction Distribution by True Class", fontsize=16)
    ax4.set_xlabel("Predicted Probability")
    ax4.set_ylabel("Density")
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save or show plot based on environment settings
    if save_plots:
        output_file = output_dir / "gam_analysis.png"
        plt.savefig(output_file, dpi=150)
        print(f"\nPlot saved to {output_file}")
    else:
        print("\nPlot generated. Close the plot window to finish the script.")
        plt.show()


def main():
    """Main function to run the end-to-end simulation and plotting."""
    # Check if running in a headless environment and set appropriate backend
    # This allows the script to run on servers without a display
    try:
        if 'DISPLAY' not in os.environ or not os.environ['DISPLAY']:
            print("Running in headless mode. Setting non-interactive backend.")
            matplotlib.use('Agg')
    except Exception as e:
        print(f"Note: {e}. Will attempt to use default backend.")
        
    # Set default figure format and output path for saved plots
    output_dir = Path(os.environ.get('PLOT_OUTPUT_DIR', './'))
    output_dir.mkdir(exist_ok=True)
    save_plots = os.environ.get('SAVE_PLOTS', 'false').lower() in ('true', '1', 't')
        
    try:
        print(f"Project Root Detected: {PROJECT_ROOT}")
        build_rust_project()
        print(f"Using executable: {EXECUTABLE_PATH}\n")

        # 1. Simulate and prepare training data
        print(f"--- Simulating {N_SAMPLES_TRAIN} samples for training ---")
        train_df = simulate_data(N_SAMPLES_TRAIN, seed=42)
        train_df[['phenotype', 'score', 'PC1']].to_csv(TRAIN_DATA_PATH, sep="\t", index=False)
        print(f"Saved training data to '{TRAIN_DATA_PATH}'")

        # 2. Train baseline Logistic Regression model
        print("\n--- Training Baseline Logistic Regression Model ---")
        baseline_model = LogisticRegression(solver='liblinear')
        baseline_model.fit(train_df[['score', 'PC1']], train_df['phenotype'])
        print("Baseline model trained.")

        # 3. Run the gnomon 'train' command
        print("\n--- Running 'train' command for GAM model ---")
        train_command = [
            str(EXECUTABLE_PATH), "train", "--num-pcs", str(NUM_PCS),
            "--pgs-knots", "3", "--pc-knots", "3", "--pgs-degree", "2", "--pc-degree", "2",
            str(TRAIN_DATA_PATH)
        ]
        run_subprocess(train_command, cwd=PROJECT_ROOT)
        print(f"\nGAM model training complete. Model saved to '{MODEL_PATH}'")

        # 4. Simulate and prepare test data
        print(f"\n--- Simulating {N_SAMPLES_TEST} samples for inference ---")
        test_df_full = simulate_data(N_SAMPLES_TEST, seed=101)
        test_df_full[['score', 'PC1']].to_csv(TEST_DATA_PATH, sep="\t", index=False)
        print(f"Saved test data to '{TEST_DATA_PATH}'")

        # 5. Get predictions from baseline model
        print("\n--- Generating predictions with Baseline Model ---")
        baseline_probs = baseline_model.predict_proba(test_df_full[['score', 'PC1']])[:, 1]
        
        # 6. Run 'infer' for GAM on test data and load results
        print("\n--- Running 'infer' command for GAM model on test data ---")
        infer_command = [str(EXECUTABLE_PATH), "infer", "--model", str(MODEL_PATH), str(TEST_DATA_PATH)]
        run_subprocess(infer_command, cwd=PROJECT_ROOT)
        print(f"Inference complete. Predictions saved to '{PREDICTIONS_PATH}'")
        gam_test_predictions = pd.read_csv(PREDICTIONS_PATH, sep="\t")

        # 7. Combine all results for reporting
        print("\n--- Analyzing Results ---")
        results_df = pd.concat([test_df_full.reset_index(drop=True), gam_test_predictions], axis=1)
        results_df['baseline_prediction'] = baseline_probs
        generate_performance_report(results_df)
        
        # 8. Generate a grid of data points to visualize the model's learned surface
        print("\n--- Generating GAM surface plot data ---")
        score_grid, pc1_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3.5, 3.5, 100))
        grid_df = pd.DataFrame({'score': score_grid.ravel(), 'PC1': pc1_grid.ravel()})
        grid_df.to_csv(GRID_DATA_PATH, sep='\t', index=False)
        
        # NOTE: The 'gnomon' tool uses a fixed output file. The following command
        # will overwrite 'predictions.tsv' with the results from the grid data.
        print("\n--- Running 'infer' on grid data for surface plot ---")
        infer_grid_command = [str(EXECUTABLE_PATH), "infer", "--model", str(MODEL_PATH), str(GRID_DATA_PATH)]
        run_subprocess(infer_grid_command, cwd=PROJECT_ROOT)
        
        # Load the newly-overwritten 'predictions.tsv' file for the surface plot
        grid_preds_df = pd.read_csv(PREDICTIONS_PATH, sep="\t")
        gam_surface = grid_preds_df['prediction'].values.reshape(score_grid.shape)
        print("GAM surface data generated.")

        # 9. Calculate the true signal surface for comparison in the plot
        true_logit_surface = SIMULATION_SIGNAL_STRENGTH * (
            0.8 * np.cos(score_grid * 2) - 1.2 * np.tanh(pc1_grid) +
            1.5 * np.sin(score_grid) * pc1_grid - 0.5 * (score_grid**2 + pc1_grid**2)
        )
        signal_surface = 1 / (1 + np.exp(-true_logit_surface))

        # 10. Generate and display the final analysis plots
        create_analysis_plots(results_df, gam_surface, signal_surface, score_grid, pc1_grid)

    finally:
        # Clean up temporary files created during the run
        print("\n--- Cleaning up temporary files ---")
        files_to_clean = [TRAIN_DATA_PATH, TEST_DATA_PATH, GRID_DATA_PATH, PREDICTIONS_PATH]
        for f in files_to_clean:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Removed {f.name}")
                except OSError as e:
                    print(f"Error removing file {f.name}: {e}")

if __name__ == "__main__":
    main()
