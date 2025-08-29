import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
import joblib
from pygam import LogisticGAM, s, te

# --- 1. Define Paths and Parameters ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# --- Model and Data Paths ---
GNOMON_EXECUTABLE = WORKSPACE_ROOT / "target" / "release" / "gnomon"
R_MODEL_PATH = SCRIPT_DIR / 'gam_model_fit.rds'
PYGAM_MODEL_PATH = SCRIPT_DIR / 'gam_model_fit.joblib'
RUST_MODEL_CONFIG_PATH = PROJECT_ROOT / 'model.toml'
TEST_DATA_CSV = SCRIPT_DIR / 'test_data.csv' # Use test data for all evaluation

# --- Temporary File Paths for Inference ---
R_PREDICTIONS_CSV = SCRIPT_DIR / 'r_model_predictions.csv'
PYGAM_PREDICTIONS_CSV = SCRIPT_DIR / 'pygam_model_predictions.csv'
RUST_PREDICTIONS_CSV = SCRIPT_DIR / 'rust_model_predictions.csv'
RUST_FORMATTED_INFERENCE_DATA_TSV = SCRIPT_DIR / 'rust_formatted_inference_data.tsv'

# --- Plotting Parameters ---
GRID_POINTS = 300 # Reduced for faster plotting
PLOT_RANGE_EXPANSION_FACTOR = 1.1 # Expand plot boundaries by 10%

# --- 2. Re-usable Metric Functions ---

# Constant for small values to prevent numerical issues
EPS = 1e-15

def _prep(y_true, y_prob):
    """Prepare data by converting to float, clipping probabilities, and handling NaNs."""
    y = pd.Series(y_true).astype(float)
    p = pd.Series(y_prob).astype(float).clip(EPS, 1-EPS)
    mask = (~y.isna()) & (~p.isna())
    y, p = y[mask], p[mask]
    return y.values, p.values

def safe_auc(y_true, y_prob):
    """Safely calculates ROC AUC with proper edge case handling."""
    y, p = _prep(y_true, y_prob)
    if len(np.unique(y)) < 2:  # Check if we have both classes
        return np.nan
    return roc_auc_score(y, p)

def safe_brier(y_true, y_prob):
    """Safely calculates Brier score with proper edge case handling."""
    y, p = _prep(y_true, y_prob)
    return brier_score_loss(y, p)

def safe_logloss(y_true, y_prob):
    """Safely calculates log loss (cross-entropy) with proper edge case handling."""
    y, p = _prep(y_true, y_prob)
    return log_loss(y, p)

def tjurs_r2(y_true, y_prob):
    """Calculates Tjur's R-squared for model performance with proper edge case handling."""
    y, p = _prep(y_true, y_prob)
    if (y==1).sum()==0 or (y==0).sum()==0:  # Check if we have both classes
        return np.nan
    return p[y==1].mean() - p[y==0].mean()

def nagelkerkes_r2(y_true, y_prob):
    """Calculates Nagelkerke's R-squared for model performance with proper edge case handling."""
    y, p = _prep(y_true, y_prob)
    p_mean = y.mean()
    if p_mean == 0 or p_mean == 1:  # Check for edge cases
        return np.nan
    ll_null = (y*np.log(p_mean) + (1-y)*np.log(1-p_mean)).sum()
    ll_model = (y*np.log(p) + (1-y)*np.log(1-p)).sum()
    n = len(y)
    r2_cs = 1 - np.exp((2/n)*(ll_null - ll_model))
    max_r2_cs = 1 - np.exp((2/n)*ll_null)
    return r2_cs / max_r2_cs if max_r2_cs > 0 else np.nan

def brier_skill_score(y_true, y_prob):
    """Calculates Brier Skill Score compared to a constant-prevalence baseline."""
    y, p = _prep(y_true, y_prob)
    bs_model = ((p - y)**2).mean()
    bs_ref = ((y.mean() - y)**2).mean()  # Baseline model always predicting prevalence
    return 1 - bs_model/bs_ref if bs_ref > 0 else np.nan

def pr_auc(y_true, y_prob):
    """Calculates PR-AUC (Average Precision) with proper edge case handling."""
    y, p = _prep(y_true, y_prob)
    if (y==1).sum()==0:  # Need positive examples for PR-AUC
        return np.nan
    return average_precision_score(y, p)

def calibration_intercept_slope(y_true, y_prob):
    """Calculate calibration intercept and slope via logistic regression."""
    y, p = _prep(y_true, y_prob)
    logit_p = np.log(p/(1-p)).reshape(-1,1)  # logit transform
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000).fit(logit_p, y)
        slope = lr.coef_[0,0]
        intercept = lr.intercept_[0]
        return intercept, slope
    except Exception:
        return np.nan, np.nan

def expected_calibration_error(y_true, y_prob, n_bins=20, strategy='uniform'):
    """Calculate Expected Calibration Error using binning.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
    strategy : str
        Binning strategy: 'uniform' (equal-width) or 'quantile' (equal-frequency)
        
    Returns:
    --------
    float: Expected Calibration Error
    """
    y, p = _prep(y_true, y_prob)
    
    if strategy == 'quantile':
        # Quantile binning (equal-frequency)
        qs = np.linspace(0, 1, n_bins+1)
        edges = np.quantile(p, qs)
        edges[0], edges[-1] = 0.0, 1.0
        # Monotone fix (in case of heavy ties)
        edges = np.maximum.accumulate(edges)
        # Add tiny jitter if edges collapse
        edges = np.unique(edges)
        if len(edges) <= 2:  # degenerate: all p nearly identical
            edges = np.array([0.0, 1.0])
            n_bins = 1
    else:
        # Uniform binning (equal-width)
        edges = np.linspace(0.0, 1.0, n_bins+1)
    
    # Digitize into bins
    bin_ids = np.digitize(p, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges)-2)
    
    # Calculate ECE
    ece = 0.0
    n = len(p)
    for b in range(len(edges)-1):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue
        conf = p[idx].mean()  # avg confidence in bin
        acc = y[idx].mean()   # avg accuracy in bin
        w = idx.sum() / n     # bin weight
        ece += w * abs(acc - conf)
        
    return ece

def ece_randomized_quantile(y_true, y_prob, 
                          bin_counts=(10, 20, 40),  # multi-resolution
                          repeats=50,              # random offsets per resolution
                          min_per_bin=20,          # auto-downshift if sample is small
                          rng=None):
    """Calculate ECE using randomized quantile bins for stability.
    
    This method creates multiple random binnings in rank space and
    averages the ECE across them, making the metric more stable and
    less sensitive to bin boundaries.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    bin_counts : tuple
        Number of bins to try for multi-resolution averaging
    repeats : int
        Number of random offsets per bin count
    min_per_bin : int
        Minimum examples per bin (auto-reduces bin count if needed)
    rng : np.random.Generator
        Random number generator for reproducibility
        
    Returns:
    --------
    dict: Contains 'ece_mean', 'ece_std', sample size 'n', and metadata
    """
    rng = np.random.default_rng(rng)
    y, p = _prep(y_true, y_prob)
    n = len(y)
    eces = []

    for M in bin_counts:
        # Ensure enough examples per bin
        M_eff = min(M, max(1, n // max(1, min_per_bin)))
        if M_eff < 2:
            # Degenerate: treat everything as one bin
            eces.append(abs(y.mean() - p.mean()))
            continue

        # Randomize bin *offset* in rank space: δ ∈ [0, 1/M_eff)
        for _ in range(repeats):
            delta = rng.uniform(0, 1.0/M_eff)
            # Rank-space cutpoints -> map back to probability via quantiles
            qs = (delta + np.arange(M_eff+1)/M_eff).clip(0, 1)
            edges = np.quantile(p, qs)
            # Guard against duplicate edges (ties)
            edges[0], edges[-1] = 0.0, 1.0
            edges = np.unique(edges)
            if len(edges) < 2:
                # All predictions nearly identical
                eces.append(abs(y.mean() - p.mean()))
                continue
                
            # Calculate ECE with these edges
            bin_ids = np.digitize(p, edges, right=True) - 1
            bin_ids = np.clip(bin_ids, 0, len(edges)-2)
            ece = 0.0
            for b in range(len(edges)-1):
                idx = (bin_ids == b)
                if not idx.any():
                    continue
                conf = p[idx].mean()
                acc = y[idx].mean()
                w = idx.mean()  # bin mass
                ece += w * abs(acc - conf)
            eces.append(ece)

    eces = np.array(eces, dtype=float)
    return {
        "ece_mean": float(np.mean(eces)),
        "ece_std": float(np.std(eces, ddof=1)) if len(eces) > 1 else 0.0,
        "n": n,
        "details": {"bin_counts": list(bin_counts), "repeats": repeats}
    }

def brier_decomposition(y_true, y_prob, n_bins=20):
    """Decompose Brier score into reliability, resolution, and uncertainty components."""
    y, p = _prep(y_true, y_prob)
    # Reliability diagram bins
    bins = np.linspace(0,1,n_bins+1)
    bin_ids = np.digitize(p, bins) - 1
    p_bar = y.mean()  # overall prevalence
    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        idx = bin_ids == b
        if idx.sum() == 0: 
            continue
        p_b = p[idx].mean()  # mean prediction in bin
        o_b = y[idx].mean()  # mean outcome in bin
        w = idx.mean()       # bin weight
        reliability += w * (p_b - o_b)**2
        resolution += w * (o_b - p_bar)**2
    uncertainty = p_bar*(1-p_bar)  # inherent uncertainty in the data
    return reliability, resolution, uncertainty

def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_boot=1000, alpha=0.05, rng=None):
    """Calculate bootstrap confidence intervals for any metric.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    metric_fn : function
        Metric function that takes (y_true, y_prob) and returns a scalar
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Alpha level for confidence interval (default: 0.05 for 95% CI)
    rng : np.random.Generator, optional
        Random number generator for reproducibility
        
    Returns:
    --------
    tuple: (estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(rng)
    y, p = _prep(y_true, y_prob)
    n = len(y)
    stats = []
    
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # Bootstrap sampling with replacement
        try:
            stats.append(metric_fn(y[idx], p[idx]))
        except Exception:
            continue
            
    stats = np.array(stats, dtype=float)
    stats = stats[~np.isnan(stats)]  # Remove NaNs
    
    if len(stats) == 0:
        return (np.nan, np.nan, np.nan)
        
    lo, hi = np.quantile(stats, [alpha/2, 1-alpha/2])
    return (metric_fn(y, p), lo, hi)


def wilson_ci(k, n, alpha=0.05):
    """Calculate Wilson score confidence interval for a binomial proportion.
    
    Parameters:
    -----------
    k : int
        Number of successes
    n : int
        Number of trials
    alpha : float
        Alpha level for confidence interval (default: 0.05 for 95% CI)
        
    Returns:
    --------
    tuple: (lower_bound, upper_bound)
    """
    if n == 0:
        return (np.nan, np.nan)
        
    from scipy.stats import norm
    z = norm.ppf(1 - alpha/2)  # Two-tailed z-score
    p = k/n
    
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    half_width = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    
    return (center - half_width, center + half_width)


def compute_calibration_bins(y_true, y_prob, n_bins=20, strategy='quantile'):
    """Compute calibration bins for reliability diagram.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
    strategy : str
        Binning strategy: 'uniform' (equal-width) or 'quantile' (equal-frequency)
        
    Returns:
    --------
    DataFrame: Per-bin statistics for plotting
    """
    y, p = _prep(y_true, y_prob)
    
    if strategy == 'quantile':
        # Quantile binning (equal-frequency)
        qs = np.linspace(0, 1, n_bins+1)
        edges = np.quantile(p, qs)
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.maximum.accumulate(edges)  # Monotone fix
        edges = np.unique(edges)  # Handle duplicates
        if len(edges) <= 2:  # Degenerate case
            edges = np.array([0.0, 1.0])
    else:
        # Uniform binning (equal-width)
        edges = np.linspace(0.0, 1.0, n_bins+1)
    
    # Digitize into bins
    bin_ids = np.digitize(p, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges)-2)
    
    # Collect per-bin statistics
    rows = []
    n_total = len(p)
    for b in range(len(edges)-1):
        idx = bin_ids == b
        n_bin = idx.sum()
        if n_bin == 0:
            continue
            
        mean_pred = p[idx].mean()  # Mean prediction in bin (x-axis)
        obs_freq = y[idx].mean()   # Observed frequency in bin (y-axis)
        n_pos = (y[idx] == 1).sum()  # Positives for Wilson CI
        lo, hi = wilson_ci(n_pos, n_bin)
        
        rows.append({
            'bin': b,
            'left_edge': edges[b], 
            'right_edge': edges[b+1],
            'mean_pred': mean_pred,
            'obs_freq': obs_freq,
            'n': n_bin,
            'bin_mass': n_bin / n_total,
            'lo_ci': lo,
            'hi_ci': hi
        })
    
    return pd.DataFrame(rows)


def plot_reliability_diagram(y_true, y_prob, n_bins=20, strategy='quantile', title=None, ax=None):
    """Plot a reliability diagram with proper bin mass visualization.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins
    strategy : str
        Binning strategy: 'uniform' (equal-width) or 'quantile' (equal-frequency)
    title : str, optional
        Plot title
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    tuple: (fig, ax, ax2) - Figure and both axes (calibration and bin mass)
    """
    # Compute calibration bins
    bins_df = compute_calibration_bins(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    
    # Calculate ECE using both standard and randomized methods
    ece_std = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    ece_rand = ece_randomized_quantile(y_true, y_prob, bin_counts=(10, 20, 40), repeats=20)
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
        
    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
    
    # Plot calibration points with Wilson CIs
    xs = bins_df['mean_pred'].values
    ys = bins_df['obs_freq'].values
    los = bins_df['lo_ci'].values
    his = bins_df['hi_ci'].values
    
    ax.errorbar(xs, ys, yerr=[ys - los, his - ys], 
                fmt='o', capsize=3, linewidth=1.5, 
                color='#1f77b4', label='Bin accuracy with 95% CI')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability (per bin)', fontsize=12)
    ax.set_ylabel('Observed frequency with Wilson 95% CI', fontsize=12)
    
    # Title with ECE information
    plot_title = title or 'Reliability diagram'
    ax.set_title(f"{plot_title}\nECE={ece_std:.4f} | Randomized ECE={ece_rand['ece_mean']:.4f} ± {ece_rand['ece_std']:.4f}\n({strategy} bins, n={len(bins_df)} non-empty)")
    
    # Add bin mass visualization on twin axis
    ax2 = ax.twinx()
    bin_width = 0.8 * (bins_df['right_edge'] - bins_df['left_edge']).mean()
    bin_centers = (bins_df['left_edge'] + bins_df['right_edge']) / 2
    
    ax2.bar(bin_centers, bins_df['bin_mass'], width=bin_width, alpha=0.15, 
            color='#1f77b4', edgecolor='none', label='Bin mass')
    
    max_mass = bins_df['bin_mass'].max() if len(bins_df) > 0 else 0.1
    ax2.set_ylim(0, max_mass * 1.6)  # Leave some headroom
    ax2.set_ylabel('Bin mass (fraction of samples)', fontsize=12)
    
    # Legend for both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return fig, ax, ax2

# --- 3. Analysis and Plotting Functions ---

def run_r_inference(input_csv, output_csv):
    """Generic function to get predictions from the R/mgcv model."""
    print(f"--- Running R/mgcv inference on '{input_csv.name}'")
    r_script = f"suppressPackageStartupMessages(library(mgcv)); model<-readRDS('{R_MODEL_PATH.name}');" \
               f"d<-read.csv('{input_csv.name}'); p<-predict(model,d,type='response');" \
               f"write.csv(data.frame(r_prediction=p),'{output_csv.name}',row.names=F)"
    try:
        subprocess.run(["Rscript", "-e", r_script], check=True, text=True, capture_output=True, cwd=SCRIPT_DIR)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nERROR: R script failed. Is R installed? Error:\n{getattr(e, 'stderr', e)}"); sys.exit(1)

def run_rust_inference(input_csv, temp_tsv, output_csv):
    """Generic function to get predictions from the Rust/gnomon model."""
    print(f"--- Running Rust/gnomon inference on '{input_csv.name}'")
    pd.read_csv(input_csv).rename(columns={'variable_one':'score','variable_two':'PC1'}).to_csv(temp_tsv,sep='\t',index=False)
    cmd = [str(GNOMON_EXECUTABLE), "infer", "--model", RUST_MODEL_CONFIG_PATH.name, str(temp_tsv.relative_to(PROJECT_ROOT))]
    try:
        subprocess.run(cmd, check=True, text=True, cwd=PROJECT_ROOT)
        rust_output_path = PROJECT_ROOT / 'predictions.tsv'
        pd.read_csv(rust_output_path, sep='\t').rename(columns={'prediction':'rust_prediction'}).to_csv(output_csv, index=False)
        rust_output_path.unlink()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nERROR: gnomon failed. Is it built? Error:\n{e}"); sys.exit(1)
        
def run_python_inference(input_csv, output_csv):
    """Generic function to get predictions from the Python/PyGAM model.
    Returns True if successful, False if PyGAM model is not available."""
    print(f"--- Running Python/PyGAM inference on '{input_csv.name}'")
    
    # Check if model file exists first
    if not PYGAM_MODEL_PATH.is_file():
        print(f"INFO: PyGAM model file not found at {PYGAM_MODEL_PATH}")
        print("PyGAM will be omitted from comparison. This is not an error.")
        return False
        
    try:
        # Try to import required modules
        try:
            import joblib
            from pygam import LogisticGAM
        except ImportError as e:
            print(f"INFO: Required PyGAM dependencies not available: {e}")
            print("PyGAM will be omitted from comparison. This is not an error.")
            return False
            
        # Load the model and make predictions
        model = joblib.load(PYGAM_MODEL_PATH)
        data = pd.read_csv(input_csv)
        X = data[['variable_one', 'variable_two']].values
        predictions = model.predict_proba(X)
        
        # Save predictions to CSV
        pd.DataFrame({'pygam_prediction': predictions}).to_csv(output_csv, index=False)
        return True
        
    except Exception as e:
        print(f"INFO: PyGAM inference failed: {e}")
        print("PyGAM will be omitted from comparison. This is not an error.")
        return False

def print_performance_report(df, bootstrap_ci=True, n_boot=1000, seed=42):
    """Calculates and prints all performance metrics based on test data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with outcome and model predictions
    bootstrap_ci : bool
        Whether to compute bootstrap confidence intervals
    n_boot : int
        Number of bootstrap samples
    seed : int
        Random seed for reproducibility
    """
    y_true = df['outcome'].values
    models = {
        "R / mgcv": df['r_prediction'].values,
        "Rust / gnomon": df['rust_prediction'].values
    }
    
    # Add PyGAM if available
    if 'pygam_prediction' in df.columns:
        models["Python / PyGAM"] = df['pygam_prediction'].values
    print("\n" + "="*60)
    print("      Model Performance on TEST Data")
    print("="*60)
    
    rng = np.random.default_rng(seed) if bootstrap_ci else None

    print("\n[Discrimination: ROC AUC] (higher is better)")
    for n, p in models.items():
        if bootstrap_ci:
            est, lo, hi = bootstrap_metric_ci(y_true, p, safe_auc, n_boot=n_boot, rng=rng)
            print(f"  - {n:<20}: {est:.4f} [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"  - {n:<20}: {safe_auc(y_true, p):.4f}")

    print("\n[Proper scoring: Brier, Log Loss] (lower is better)")
    for n, p in models.items():
        if bootstrap_ci:
            b_est, b_lo, b_hi = bootstrap_metric_ci(y_true, p, safe_brier, n_boot=n_boot, rng=rng)
            l_est, l_lo, l_hi = bootstrap_metric_ci(y_true, p, safe_logloss, n_boot=n_boot, rng=rng)
            print(f"  - {n:<20}: Brier={b_est:.4f} [{b_lo:.4f}, {b_hi:.4f}] | "  
                  f"LogLoss={l_est:.4f} [{l_lo:.4f}, {l_hi:.4f}]")
        else:
            print(f"  - {n:<20}: Brier={safe_brier(y_true, p):.4f} | LogLoss={safe_logloss(y_true, p):.4f}")

    print("\n[Skill vs baseline: Brier Skill Score] (higher is better)")
    for n, p in models.items():
        if bootstrap_ci:
            est, lo, hi = bootstrap_metric_ci(y_true, p, brier_skill_score, n_boot=n_boot, rng=rng)
            print(f"  - {n:<20}: {est:.4f} [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"  - {n:<20}: {brier_skill_score(y_true, p):.4f}")

    print("\n[Fit proxies: Nagelkerke R², Tjur R²] (higher is better)")
    for n, p in models.items():
        if bootstrap_ci:
            n_est, n_lo, n_hi = bootstrap_metric_ci(y_true, p, nagelkerkes_r2, n_boot=n_boot, rng=rng)
            t_est, t_lo, t_hi = bootstrap_metric_ci(y_true, p, tjurs_r2, n_boot=n_boot, rng=rng)
            print(f"  - {n:<20}: Nagelkerke={n_est:.4f} [{n_lo:.4f}, {n_hi:.4f}] | "
                  f"Tjur={t_est:.4f} [{t_lo:.4f}, {t_hi:.4f}]")
        else:
            print(f"  - {n:<20}: Nagelkerke={nagelkerkes_r2(y_true, p):.4f} | Tjur={tjurs_r2(y_true, p):.4f}")

    print("\n[Imbalanced-case view: PR-AUC / Avg Precision] (higher is better)")
    for n, p in models.items():
        if bootstrap_ci:
            est, lo, hi = bootstrap_metric_ci(y_true, p, pr_auc, n_boot=n_boot, rng=rng)
            print(f"  - {n:<20}: {est:.4f} [{lo:.4f}, {hi:.4f}]")
        else:
            print(f"  - {n:<20}: {pr_auc(y_true, p):.4f}")

    print("\n[Calibration: intercept, slope, ECE]")
    for n, p in models.items():
        ci, cs = calibration_intercept_slope(y_true, p)
        
        # Standard ECE with quantile binning
        ece_quant = expected_calibration_error(y_true, p, n_bins=20, strategy='quantile')
        
        # Randomized quantile ECE for stability
        seed = rng.integers(0, 1000) if rng is not None else 42
        ece_rand = ece_randomized_quantile(
            y_true, p, 
            bin_counts=(10, 20, 40), 
            repeats=20,  # reduced for speed
            rng=np.random.default_rng(seed)
        )
        
        print(f"  - {n:<20}: intercept={ci:+.3f}, slope={cs:.3f},")
        print(f"                       ECE (quantile)={ece_quant:.4f}, ECE (randomized)={ece_rand['ece_mean']:.4f} ±{ece_rand['ece_std']:.4f}")


    print("\n[Brier decomposition: reliability ↓, resolution ↑, uncertainty (data)]")
    for n, p in models.items():
        rel, res, unc = brier_decomposition(y_true, p, n_bins=20)
        print(f"  - {n:<20}: rel={rel:.4f}, res={res:.4f}, unc={unc:.4f}")

    print("\n" + "="*60)

def plot_prediction_comparisons(df):
    """
    Generates plots centered around Rust comparisons with other models and ground truth.
    """
    print("\n--- Generating Model Comparison Plots ---")
    
    # Define consistent colors across all visualizations
    COLORS = {
        'rust': '#2ca02c',      # Green for Rust
        'r': '#1f77b4',         # Blue for R
        'pygam': '#ff7f0e',     # Orange for PyGAM
        'perfect': '#d62728',   # Red for perfect prediction lines
        'grid': '#cccccc'       # Light gray for grids
    }
    
    # Check which models we have
    has_pygam = 'pygam_prediction' in df.columns
    
    # Create custom colormaps with model-specific colors for density plots
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define separate colormaps for each model
    # Each colormap transitions from white to the model's color
    r_cmap = LinearSegmentedColormap.from_list('r_cmap', ['#ffffff', COLORS['r']], N=100)
    rust_cmap = LinearSegmentedColormap.from_list('rust_cmap', ['#ffffff', COLORS['rust']], N=100)
    
    if has_pygam:
        pygam_cmap = LinearSegmentedColormap.from_list('pygam_cmap', ['#ffffff', COLORS['pygam']], N=100)
    
    # Define models and their visual attributes
    models = {
        'R / mgcv': {
            'predictions': 'r_prediction',
            'color': COLORS['r'],
            'marker': 'o',       # Circle
            'label': 'R / mgcv',
            'zorder': 2
        },
        'Rust / gnomon': {
            'predictions': 'rust_prediction',
            'color': COLORS['rust'],
            'marker': 'o',       # Circle
            'label': 'Rust / gnomon',
            'zorder': 3  # Higher zorder to ensure Rust is on top
        }
    }
    
    # Add PyGAM model if available
    if has_pygam:
        models['Python / PyGAM'] = {
            'predictions': 'pygam_prediction',
            'color': COLORS['pygam'],
            'marker': 'o',       # Circle
            'label': 'Python / PyGAM',
            'zorder': 2
        }
    
    # Function to add density-based coloring to a scatter plot
    def density_scatter(ax, x, y, color, marker, label, zorder=1):
        from scipy.stats import gaussian_kde
        xy = np.vstack([x, y])
        try:
            density = gaussian_kde(xy)(xy)
            idx = np.argsort(density)
            x, y, density = x[idx], y[idx], density[idx]
            # Use a custom colormap based on the model's main color
            from matplotlib.colors import LinearSegmentedColormap
            light_color = '#ffffff'  # White for low density
            cmap_name = f"density_{color.replace('#', '')}"
            custom_cmap = LinearSegmentedColormap.from_list(cmap_name, [light_color, color], N=100)
            
            scatter = ax.scatter(x, y, c=density, cmap=custom_cmap, 
                      s=30, marker=marker, edgecolor='none',
                      alpha=0.8, zorder=zorder)
            return scatter
        except Exception:
            # Fallback if KDE fails
            return ax.scatter(x, y, c=color, s=30, marker=marker, alpha=0.5, 
                           label=label, zorder=zorder)
    
    # Define plot layout based on PyGAM availability
    if has_pygam:
        # Create a 2x2 figure for comparing all models
        fig = plt.figure(figsize=(28, 13))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.8, 1.8], hspace=0.4, wspace=0.1)
        
        # Plot 1: Rust vs Ground Truth (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
    else:
        # Create a 1x3 figure when PyGAM is not available
        fig = plt.figure(figsize=(28, 6.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.8, 1.8, 1.8], wspace=0.1)
        
        # Plot 1: Rust vs Ground Truth (Left)
        ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate MAE for Rust vs ground truth
    rust_mae = np.mean(np.abs(df['final_probability'] - df['rust_prediction']))
    
    # Create Rust vs Truth scatter plot with model-specific colormap
    rust_x = df['final_probability'].values
    rust_y = df['rust_prediction'].values
    
    # Use direct colormap approach for better color consistency
    from scipy.stats import gaussian_kde
    xy = np.vstack([rust_x, rust_y])
    try:
        density = gaussian_kde(xy)(xy)
        idx = np.argsort(density)
        x, y, density = rust_x[idx], rust_y[idx], density[idx]
        rust_scatter = ax1.scatter(x, y, c=density, cmap=rust_cmap,
                               s=30, marker='o', edgecolor='none',
                               alpha=0.8, zorder=3, label='Rust vs Ground Truth')
    except Exception:
        # Fallback if KDE fails
        rust_scatter = ax1.scatter(rust_x, rust_y, c=COLORS['rust'],
                               s=30, marker='o', alpha=0.5, 
                               label='Rust vs Ground Truth', zorder=3)
    
    # Add reference line and styling
    ax1.plot([0, 1], [0, 1], '--', color=COLORS['perfect'], linewidth=2, label='Perfect Prediction', zorder=4)
    ax1.set_title(f"Rust / gnomon vs. Ground Truth\nMAE = {rust_mae:.4f}", fontsize=16)
    ax1.set_xlabel("True Probability", fontsize=14)
    ax1.set_ylabel("Rust / gnomon Prediction", fontsize=14)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Add colorbar
    if hasattr(rust_scatter, 'get_cmap'):
        cbar = fig.colorbar(rust_scatter, ax=ax1, shrink=0.8)
        cbar.set_label('Density', rotation=270, labelpad=20)
    
    # Plot 2: Rust vs R (Top Right or Middle)
    if has_pygam:
        ax2 = fig.add_subplot(gs[0, 1])
    else:
        ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate metrics for Rust vs R
    mae_rust_r = np.mean(np.abs(df['rust_prediction'] - df['r_prediction']))
    
    # Create scatter plot for Rust vs R using R's color
    ax2.scatter(df['r_prediction'], df['rust_prediction'], 
               c=COLORS['r'], s=30, marker='o', 
               alpha=0.5, label='Rust vs R')
    
    # Add reference line and styling
    ax2.plot([0, 1], [0, 1], '--', color=COLORS['perfect'], linewidth=2, label='Perfect Agreement', zorder=4)
    ax2.set_title(f"Rust / gnomon vs. R / mgcv\nMAE = {mae_rust_r:.4f}", fontsize=16)
    ax2.set_xlabel("R / mgcv Prediction", fontsize=14)
    ax2.set_ylabel("Rust / gnomon Prediction", fontsize=14)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Plot PyGAM comparison if available
    if has_pygam:
        # Plot 3: Rust vs PyGAM (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate metrics for Rust vs PyGAM
        mae_rust_pygam = np.mean(np.abs(df['rust_prediction'] - df['pygam_prediction']))
        
        # Create scatter plot for Rust vs PyGAM using PyGAM's color
        ax3.scatter(df['pygam_prediction'], df['rust_prediction'], 
                   c=COLORS['pygam'], s=30, marker='o', 
                   alpha=0.5, label='Rust vs PyGAM')
        
        # Add reference line and styling
        ax3.plot([0, 1], [0, 1], '--', color=COLORS['perfect'], linewidth=2, label='Perfect Agreement', zorder=4)
        ax3.set_title(f"Rust / gnomon vs. Python / PyGAM\nMAE = {mae_rust_pygam:.4f}", fontsize=16)
        ax3.set_xlabel("Python / PyGAM Prediction", fontsize=14)
        ax3.set_ylabel("Rust / gnomon Prediction", fontsize=14)
        ax3.set_xlim(-0.05, 1.05)
        ax3.set_ylim(-0.05, 1.05)
        ax3.grid(True, linestyle='--', alpha=0.3)
        ax3.legend(loc='upper left')
        
        # Plot 4: All Models vs Ground Truth (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
    else:
        # Plot 3: All Models vs Ground Truth (Right)
        ax4 = fig.add_subplot(gs[0, 2])
    
    # Calculate model MAEs for legend
    mae_values = {name: np.mean(np.abs(df['final_probability'] - df[model['predictions']]))
                 for name, model in models.items()}
    
    # Add density-colored scatter plots for each model with consistent colormaps
    from scipy.stats import gaussian_kde
    
    # R model
    r_x = df['final_probability'].values
    r_y = df['r_prediction'].values
    try:
        xy = np.vstack([r_x, r_y])
        density = gaussian_kde(xy)(xy)
        idx = np.argsort(density)
        x, y, density = r_x[idx], r_y[idx], density[idx]
        ax4.scatter(x, y, c=density, cmap=r_cmap,
                  s=25, marker='o', edgecolor='none', alpha=0.5, 
                  label=f"R / mgcv (MAE={mae_values['R / mgcv']:.4f})")
    except Exception:
        ax4.scatter(r_x, r_y, c=COLORS['r'], s=25, marker='o', alpha=0.3,
                  label=f"R / mgcv (MAE={mae_values['R / mgcv']:.4f})")
    
    # PyGAM model (if available)
    if has_pygam:
        pygam_x = df['final_probability'].values
        pygam_y = df['pygam_prediction'].values
        try:
            xy = np.vstack([pygam_x, pygam_y])
            density = gaussian_kde(xy)(xy)
            idx = np.argsort(density)
            x, y, density = pygam_x[idx], pygam_y[idx], density[idx]
            ax4.scatter(x, y, c=density, cmap=pygam_cmap,
                      s=25, marker='o', edgecolor='none', alpha=0.5,
                      label=f"PyGAM (MAE={mae_values['Python / PyGAM']:.4f})")
        except Exception:
            ax4.scatter(pygam_x, pygam_y, c=COLORS['pygam'], s=25, marker='o', alpha=0.3,
                      label=f"PyGAM (MAE={mae_values['Python / PyGAM']:.4f})")
    
    # Make Rust more prominent
    rust_x = df['final_probability'].values
    rust_y = df['rust_prediction'].values
    try:
        xy = np.vstack([rust_x, rust_y])
        density = gaussian_kde(xy)(xy)
        idx = np.argsort(density)
        x, y, density = rust_x[idx], rust_y[idx], density[idx]
        ax4.scatter(x, y, c=density, cmap=rust_cmap,
                  s=30, marker='o', edgecolor='none', alpha=0.7, zorder=3,
                  label=f"Rust / gnomon (MAE={mae_values['Rust / gnomon']:.4f})")
    except Exception:
        ax4.scatter(rust_x, rust_y, c=COLORS['rust'], s=30, marker='o', alpha=0.7, zorder=3,
                  label=f"Rust / gnomon (MAE={mae_values['Rust / gnomon']:.4f})")
    
    # Add reference line and styling
    ax4.plot([0, 1], [0, 1], '--', color=COLORS['perfect'], linewidth=2, label='Perfect Prediction', zorder=4)
    ax4.set_title("All Models vs. Ground Truth\n(Rust highlighted)", fontsize=16)
    ax4.set_xlabel("True Probability", fontsize=14)
    ax4.set_ylabel("Model Predictions", fontsize=14)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, linestyle='--', alpha=0.3)
    ax4.legend(loc='upper left')
    
    # Add overall title
    fig.suptitle("Rust-Centered Model Comparisons", fontsize=22, y=0.98)
    
    # Use more compatible padding approach instead of tight_layout
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.4, wspace=0.2)
    plt.show()

def plot_model_surfaces(test_df):
    """Generates model surface plots for all available models and empirical data."""
    print("\n" + "#"*70); print("### PLOTTING MODEL SURFACES COMPARISON ###"); print("#"*70)
    
    # Check if PyGAM is available (by looking for the file)
    has_pygam = PYGAM_MODEL_PATH.is_file()
    if has_pygam:
        try:
            import joblib
            from pygam import LogisticGAM
        except ImportError:
            has_pygam = False
    
    # Define consistent colors across all visualizations
    COLORS = {
        'rust': '#2ca02c',      # Green for Rust
        'r': '#1f77b4',         # Blue for R
        'pygam': '#ff7f0e',     # Orange for PyGAM
        'perfect': '#d62728',   # Red for decision boundaries
        'grid': '#cccccc'       # Light gray for grids
    }
    
    # Define custom colormaps for each model to ensure color consistency
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom colormaps with model-specific colors
    r_cmap = LinearSegmentedColormap.from_list('r_cmap', ['#ffffff', COLORS['r']], N=100)
    rust_cmap = LinearSegmentedColormap.from_list('rust_cmap', ['#ffffff', COLORS['rust']], N=100)
    
    # Only create PyGAM colormap if it's available
    if has_pygam:
        pygam_cmap = LinearSegmentedColormap.from_list('pygam_cmap', ['#ffffff', COLORS['pygam']], N=100)
    
    # For empirical data, use a neutral colormap
    emp_cmap = 'viridis'

    # A. Dynamically calculate plot boundaries based on the test data range
    print(f"\n--- Dynamically calculating plot ranges from '{TEST_DATA_CSV.name}' ---")
    def get_expanded_range(data_series, factor):
        min_val, max_val = data_series.min(), data_series.max()
        center, half_width = (min_val + max_val) / 2, (max_val - min_val) / 2
        expanded_half_width = half_width * factor
        return center - expanded_half_width, center + expanded_half_width

    v1_min, v1_max = get_expanded_range(test_df['variable_one'], PLOT_RANGE_EXPANSION_FACTOR)
    v2_min, v2_max = get_expanded_range(test_df['variable_two'], PLOT_RANGE_EXPANSION_FACTOR)

    # B. Create the grid for querying models
    v1_range = np.linspace(v1_min, v1_max, GRID_POINTS)
    v2_range = np.linspace(v2_min, v2_max, GRID_POINTS)
    v1_grid, v2_grid = np.meshgrid(v1_range, v2_range)

    # C. Get predictions from the models over the generated grid
    grid_df = pd.DataFrame({'variable_one': v1_grid.flatten(), 'variable_two': v2_grid.flatten()})
    grid_csv_path = SCRIPT_DIR / 'temp_grid_data.csv'; grid_df.to_csv(grid_csv_path, index=False)
    grid_tsv_path = SCRIPT_DIR / 'temp_rust_grid_data.tsv'
    r_preds_path = SCRIPT_DIR / 'temp_r_surface_preds.csv'
    rust_preds_path = SCRIPT_DIR / 'temp_rust_surface_preds.csv'

    run_r_inference(grid_csv_path, r_preds_path)
    run_rust_inference(grid_csv_path, grid_tsv_path, rust_preds_path)

    r_preds = pd.read_csv(r_preds_path)['r_prediction'].values.reshape(GRID_POINTS, GRID_POINTS)
    rust_preds = pd.read_csv(rust_preds_path)['rust_prediction'].values.reshape(GRID_POINTS, GRID_POINTS)
    
    # Run PyGAM inference only if available
    if has_pygam:
        pygam_preds_path = SCRIPT_DIR / 'temp_pygam_surface_preds.csv'
        pygam_available = run_python_inference(grid_csv_path, pygam_preds_path)
        
        if pygam_available and pygam_preds_path.is_file():
            pygam_preds = pd.read_csv(pygam_preds_path)['pygam_prediction'].values.reshape(GRID_POINTS, GRID_POINTS)
            # Calculate differences between models, with Rust as the reference
            rust_r_diff = np.abs(rust_preds - r_preds)
            rust_pygam_diff = np.abs(rust_preds - pygam_preds)
            max_diff = max(rust_r_diff.max(), rust_pygam_diff.max())
        else:
            has_pygam = False
    
    if not has_pygam:
        # Just calculate difference between Rust and R
        rust_r_diff = np.abs(rust_preds - r_preds)
        max_diff = rust_r_diff.max()
    
    # D. Create plot layout based on available models
    print("\n--- Generating Model Surface Plots ---")
    if has_pygam:
        # 2x2 layout with all three models + empirical data
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.2)
        
        # Model predictions in a clear layout
        ax1 = fig.add_subplot(gs[0, 0])  # R model (top left)
        ax2 = fig.add_subplot(gs[0, 1])  # PyGAM model (top right)
        ax3 = fig.add_subplot(gs[1, 0])  # Rust model (bottom left)
        ax4 = fig.add_subplot(gs[1, 1])  # Empirical data (bottom right)
    else:
        # 1x3 layout with R, Rust, and empirical data
        fig = plt.figure(figsize=(20, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.2)
        
        # Model predictions in a clear layout
        ax1 = fig.add_subplot(gs[0, 0])  # R model (left)
        ax3 = fig.add_subplot(gs[0, 1])  # Rust model (middle)
        ax4 = fig.add_subplot(gs[0, 2])  # Empirical data (right)
    
    fig.suptitle("Model Surfaces Comparison", fontsize=24, y=0.98)
    
    # Define consistent colormaps for each model
    from matplotlib.colors import LinearSegmentedColormap
    
    levels = np.linspace(0, 1, 31)  # Increase levels for smoother contours
    
    # Top left: R model
    cf_r = ax1.contourf(v1_grid, v2_grid, r_preds, levels=levels, cmap=r_cmap, alpha=0.9)
    ax1.contour(v1_grid, v2_grid, r_preds, levels=[0.25, 0.5, 0.75], colors='white', linewidths=1.0)
    ax1.contour(v1_grid, v2_grid, r_preds, levels=[0.5], colors=COLORS['r'], linewidths=2.5)
    ax1.set_title("R / mgcv Model Surface", fontsize=18, pad=10)
    ax1.set_xlabel("variable_one", fontsize=14)
    ax1.set_ylabel("variable_two", fontsize=14)
    ax1.grid(color='white', linestyle=':', alpha=0.5)
    fig.colorbar(cf_r, ax=ax1, orientation='vertical', shrink=0.8, 
                label='Probability (R / mgcv)')
    
    # Top right: PyGAM model (if available)
    if has_pygam:
        cf_pygam = ax2.contourf(v1_grid, v2_grid, pygam_preds, levels=levels, cmap=pygam_cmap, alpha=0.9)
        ax2.contour(v1_grid, v2_grid, pygam_preds, levels=[0.25, 0.5, 0.75], colors='white', linewidths=1.0)
        ax2.contour(v1_grid, v2_grid, pygam_preds, levels=[0.5], colors=COLORS['pygam'], linewidths=2.5)
        ax2.set_title("Python / PyGAM Model Surface", fontsize=18, pad=10)
        ax2.set_xlabel("variable_one", fontsize=14)
        ax2.grid(color='white', linestyle=':', alpha=0.5)
        fig.colorbar(cf_pygam, ax=ax2, orientation='vertical', shrink=0.8, 
                    label='Probability (Python / PyGAM)')
    
    # Bottom left: Rust model
    cf_rust = ax3.contourf(v1_grid, v2_grid, rust_preds, levels=levels, cmap=rust_cmap, alpha=0.9)
    ax3.contour(v1_grid, v2_grid, rust_preds, levels=[0.25, 0.5, 0.75], colors='white', linewidths=1.0)
    ax3.contour(v1_grid, v2_grid, rust_preds, levels=[0.5], colors=COLORS['rust'], linewidths=2.5)
    ax3.set_title("Rust / gnomon Model Surface", fontsize=18, pad=10)
    ax3.set_xlabel("variable_one", fontsize=14)
    ax3.set_ylabel("variable_two", fontsize=14)
    ax3.grid(color='white', linestyle=':', alpha=0.5)
    fig.colorbar(cf_rust, ax=ax3, orientation='vertical', shrink=0.8, 
                label='Probability (Rust / gnomon)')
    
    # Bottom right: Empirical data with all decision boundaries
    hb = ax4.hexbin(x=test_df['variable_one'], y=test_df['variable_two'], 
                  C=test_df['outcome'], gridsize=40, cmap=emp_cmap,
                  reduce_C_function=np.mean, mincnt=1, vmin=0, vmax=1,
                  edgecolors='gray', linewidths=0.1)
    
    # Add decision boundaries with consistent colors and styles
    r_line = ax4.contour(v1_grid, v2_grid, r_preds, levels=[0.5], colors=COLORS['r'], 
                        linewidths=2.0, linestyles='-', alpha=0.9)
    rust_line = ax4.contour(v1_grid, v2_grid, rust_preds, levels=[0.5], colors=COLORS['rust'], 
                          linewidths=2.0, linestyles='-', alpha=0.9)
    
    # Add PyGAM decision boundary if available
    if has_pygam:
        pygam_line = ax4.contour(v1_grid, v2_grid, pygam_preds, levels=[0.5], colors=COLORS['pygam'], 
                               linewidths=2.0, linestyles='-', alpha=0.9)
    
    # Create proxy artists for the legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['r'], lw=2, label='R / mgcv Decision Boundary'),
        Line2D([0], [0], color=COLORS['rust'], lw=2, label='Rust / gnomon Decision Boundary')
    ]
    
    # Add PyGAM to legend if available
    if has_pygam:
        legend_elements.insert(1, Line2D([0], [0], color=COLORS['pygam'], lw=2, label='Python / PyGAM Decision Boundary'))
    
    ax4.set_title("Empirical Data with All Decision Boundaries", fontsize=18, pad=10)
    ax4.set_xlabel("variable_one", fontsize=14)
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    fig.colorbar(hb, ax=ax4, orientation='vertical', shrink=0.8, 
                label='Probability (Mean outcome)')
    
    # Ensure all subplots share the same axes limits
    if has_pygam:
        # When we have all four axes (with PyGAM)
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xlim(v1_min, v1_max)
            ax.set_ylim(v2_min, v2_max)
    else:
        # When we only have three axes (without PyGAM)
        for ax in [ax1, ax3, ax4]:
            ax.set_xlim(v1_min, v1_max)
            ax.set_ylim(v2_min, v2_max)
    
    # Use more compatible padding approach instead of tight_layout
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.4, wspace=0.2)
    plt.show()

    # E. Cleanup temporary files
    for file_path in [grid_csv_path, grid_tsv_path, r_preds_path, rust_preds_path]:
        if file_path.is_file():
            file_path.unlink()
    
    # Clean up PyGAM predictions file if it exists
    if has_pygam and 'pygam_preds_path' in locals() and pygam_preds_path.is_file():
        pygam_preds_path.unlink()

# --- 4. Main Execution Block ---

def plot_model_calibration_comparison(df):
    """Plot reliability diagrams for all models."""
    print("\n--- Generating Calibration Comparison Plots ---")
    
    # Define consistent colors across all visualizations (same as other plots)
    COLORS = {
        'rust': '#2ca02c',      # Green for Rust
        'r': '#1f77b4',         # Blue for R
        'pygam': '#ff7f0e',     # Orange for PyGAM
        'perfect': '#d62728',   # Red for perfect prediction lines
    }
    
    # Check which models we have
    has_pygam = 'pygam_prediction' in df.columns
    
    # Set up the figure - adjust grid based on available models
    if has_pygam:
        # 2x2 grid with all 3 models
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        fig.suptitle("Model Calibration Comparison", fontsize=20, y=0.98)
        # Extract data for all models
        y_true = df['outcome'].values
        models = {
            "R / mgcv": {'preds': df['r_prediction'].values, 'color': COLORS['r'], 'ax': axes[0, 0]},
            "Python / PyGAM": {'preds': df['pygam_prediction'].values, 'color': COLORS['pygam'], 'ax': axes[0, 1]},
            "Rust / gnomon": {'preds': df['rust_prediction'].values, 'color': COLORS['rust'], 'ax': axes[1, 0]},
        }
        comparison_ax = axes[1, 1]  # Last plot is for comparison
    else:
        # 1x3 grid with just R and Rust
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Model Calibration Comparison", fontsize=20, y=0.98)
        # Extract data for just R and Rust models
        y_true = df['outcome'].values
        models = {
            "R / mgcv": {'preds': df['r_prediction'].values, 'color': COLORS['r'], 'ax': axes[0]},
            "Rust / gnomon": {'preds': df['rust_prediction'].values, 'color': COLORS['rust'], 'ax': axes[1]},
        }
        comparison_ax = axes[2]  # Last plot is for comparison
    
    # Plot individual model calibration
    for name, model_data in models.items():
        ax = model_data['ax']
        color = model_data['color']
        predictions = model_data['preds']
        
        # Get bin data with quantile binning
        bins_df = compute_calibration_bins(y_true, predictions, n_bins=15, strategy='quantile')
        
        # Calculate ECEs
        ece_std = expected_calibration_error(y_true, predictions, n_bins=15, strategy='quantile')
        ece_rand = ece_randomized_quantile(y_true, predictions, bin_counts=(10, 15, 20), repeats=20)
        
        # Perfect calibration reference line
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
        
        # Plot calibration points with Wilson CIs
        xs = bins_df['mean_pred'].values
        ys = bins_df['obs_freq'].values
        los = bins_df['lo_ci'].values
        his = bins_df['hi_ci'].values
        
        # Plot error bars with model color
        ax.errorbar(xs, ys, yerr=[ys - los, his - ys], 
                    fmt='o', capsize=3, linewidth=1.5, 
                    color=color, label=f'Bin accuracy with 95% CI')
        
        # Add bin mass visualization
        ax2 = ax.twinx()
        bin_width = 0.8 * (bins_df['right_edge'] - bins_df['left_edge']).mean()
        bin_centers = (bins_df['left_edge'] + bins_df['right_edge']) / 2
        
        ax2.bar(bin_centers, bins_df['bin_mass'], width=bin_width, alpha=0.15, 
                color=color, edgecolor='none', label='Bin mass')
        
        max_mass = bins_df['bin_mass'].max() if len(bins_df) > 0 else 0.1
        ax2.set_ylim(0, max_mass * 1.6)
        ax2.set_ylabel('Bin mass', fontsize=10)
        
        # Set title and labels
        ax.set_title(f"{name} Calibration\nECE={ece_std:.4f} | Randomized ECE={ece_rand['ece_mean']:.4f} ± {ece_rand['ece_std']:.4f}", fontsize=14)
        ax.set_xlabel('Mean predicted probability', fontsize=12)
        ax.set_ylabel('Observed frequency', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3, linestyle=':')
    
    # Plot all models together on the comparison subplot
    comparison_ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
    
    # Plot isotonic regression curves for each model
    for name, model_data in models.items():
        color = model_data['color']
        predictions = model_data['preds']
        
        # Sort predictions and outcomes for isotonic regression visualization
        idx = np.argsort(predictions)
        p_sorted = predictions[idx]
        y_sorted = y_true[idx]
        
        # Use moving average to smooth the curve
        window = max(30, len(p_sorted) // 40)  # Dynamic window size
        p_smooth = []
        y_smooth = []
        
        for i in range(0, len(p_sorted), window // 2):
            if i + window > len(p_sorted):
                break
            p_smooth.append(np.mean(p_sorted[i:i+window]))
            y_smooth.append(np.mean(y_sorted[i:i+window]))
            
        # Plot the smoothed isotonic calibration curve
        comparison_ax.plot(p_smooth, y_smooth, '-', linewidth=2.5, color=model_data['color'], label=name)
    
    comparison_ax.set_title("All Models Calibration Comparison", fontsize=14)
    comparison_ax.set_xlabel('Predicted probability', fontsize=12)
    comparison_ax.set_ylabel('Observed frequency (smoothed)', fontsize=12)
    comparison_ax.set_xlim(0, 1)
    comparison_ax.set_ylim(0, 1)
    comparison_ax.grid(alpha=0.3, linestyle=':')
    comparison_ax.legend(loc='upper left')
    
    # Use more compatible padding approach instead of tight_layout
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.4, wspace=0.2)
    plt.show()
    

def main():
    # --- 1. Check for required files ---
    # PyGAM is optional, don't check for its existence
    required_files = [R_MODEL_PATH, RUST_MODEL_CONFIG_PATH, TEST_DATA_CSV]
    for f in required_files:
        if not f.is_file():
            print(f"FATAL ERROR: Required file not found: {f}")
            print("Please run the training and data generation scripts first.")
            sys.exit(1)

    # --- 2. Load the test data ---
    print(f"--- Loading test data from '{TEST_DATA_CSV.name}' ---")
    test_df = pd.read_csv(TEST_DATA_CSV)

    # --- 3. Run inference on the test data to get predictions ---
    run_r_inference(TEST_DATA_CSV, R_PREDICTIONS_CSV)
    pygam_available = run_python_inference(TEST_DATA_CSV, PYGAM_PREDICTIONS_CSV)
    run_rust_inference(TEST_DATA_CSV, RUST_FORMATTED_INFERENCE_DATA_TSV, RUST_PREDICTIONS_CSV)

    # --- 4. Combine data and predictions ---
    r_preds_df = pd.read_csv(R_PREDICTIONS_CSV)
    rust_preds_df = pd.read_csv(RUST_PREDICTIONS_CSV)
    
    # Add PyGAM predictions only if available
    if pygam_available and PYGAM_PREDICTIONS_CSV.is_file():
        pygam_preds_df = pd.read_csv(PYGAM_PREDICTIONS_CSV)
        combined_df = pd.concat([test_df, r_preds_df, pygam_preds_df, rust_preds_df], axis=1)
    else:
        combined_df = pd.concat([test_df, r_preds_df, rust_preds_df], axis=1)

    # --- 5. Report metrics and plot comparisons on test data ---
    # Use bootstrap_ci=True for confidence intervals (can be slow with n_boot=1000)
    # For faster results without confidence intervals, use bootstrap_ci=False
    print_performance_report(combined_df, bootstrap_ci=True, n_boot=200, seed=42)
    plot_prediction_comparisons(combined_df)
    plot_model_calibration_comparison(combined_df)  # New calibration plots
    plot_model_surfaces(test_df)

    # --- 6. Final cleanup ---
    for file_path in [R_PREDICTIONS_CSV, RUST_PREDICTIONS_CSV, RUST_FORMATTED_INFERENCE_DATA_TSV]:
        if file_path.is_file():
            file_path.unlink()
    
    # Clean up PyGAM predictions file if it exists
    if PYGAM_PREDICTIONS_CSV.is_file():
        PYGAM_PREDICTIONS_CSV.unlink()
        
    print("\n--- Analysis script finished successfully. ---")

if __name__ == "__main__":
    main()
