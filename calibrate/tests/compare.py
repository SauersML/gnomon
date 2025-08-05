import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

# --- 1. Define Paths and Parameters ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# --- Model and Data Paths ---
GNOMON_EXECUTABLE = WORKSPACE_ROOT / "target" / "release" / "gnomon"
R_MODEL_PATH = SCRIPT_DIR / 'gam_model_fit.rds'
RUST_MODEL_CONFIG_PATH = PROJECT_ROOT / 'model.toml'
TRAINING_DATA_CSV = SCRIPT_DIR / 'synthetic_classification_data.csv'

# --- Temporary File Paths for Inference ---
R_PREDICTIONS_CSV = SCRIPT_DIR / 'r_model_predictions.csv'
RUST_PREDICTIONS_CSV = SCRIPT_DIR / 'rust_model_predictions.csv'
RUST_FORMATTED_TRAINING_DATA_TSV = SCRIPT_DIR / 'rust_formatted_training_data.tsv'

# --- Plotting Parameters ---
GRID_POINTS = 500
PLOT_RANGE_EXPANSION_FACTOR = 1.2 # Expand plot boundaries by 20%

# --- 2. Re-usable Metric Functions (Unchanged) ---

def tjurs_r2(y_true, y_prob):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)
    mean_prob_cases = y_prob[y_true == 1].mean()
    mean_prob_controls = y_prob[y_true == 0].mean()
    return mean_prob_cases - mean_prob_controls

def nagelkerkes_r2(y_true, y_prob):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    p_mean = np.mean(y_true)
    if p_mean == 0 or p_mean == 1: return 0.0
    ll_null = np.sum(y_true * np.log(p_mean) + (1 - y_true) * np.log(1 - p_mean))
    ll_model = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    n = len(y_true)
    r2_cs = 1 - np.exp((2/n) * (ll_null - ll_model))
    max_r2_cs = 1 - np.exp((2/n) * ll_null)
    return r2_cs / max_r2_cs if max_r2_cs > 0 else 0.0

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

def print_performance_report(df):
    """Calculates and prints all performance metrics."""
    y_true, models = df['outcome'], {"R / mgcv": df['r_prediction'], "Rust / gnomon": df['rust_prediction']}
    print("\n" + "="*60); print("      Model Performance on Training Data"); print("="*60)
    print("\n[Discrimination: AUC]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {roc_auc_score(y_true, p):.4f}")
    print("\n[Accuracy: Brier Score]"); print("Lower is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {brier_score_loss(y_true, p):.4f}")
    print("\n[Fit: Nagelkerke's R-squared]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {nagelkerkes_r2(y_true, p):.4f}")
    print("\n[Separation: Tjur's R-squared]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {tjurs_r2(y_true, p):.4f}")
    print("\n" + "="*60)

def plot_prediction_scatter(df):
    """Plots the scatter plot comparing R and Rust predictions."""
    mae = np.mean(np.abs(df['r_prediction'] - df['rust_prediction']))
    print(f"\n[Inter-Model Agreement: MAE]\n  - MAE: {mae:.6f}")
    fig, ax = plt.subplots(figsize=(8, 8)); ax.scatter(df['r_prediction'], df['rust_prediction'], alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement'); ax.set(xlabel="R / mgcv Predictions", ylabel="Rust / gnomon Predictions", title=f"Prediction Comparison (MAE = {mae:.6f})")
    ax.grid(True, linestyle='--'); ax.set_aspect('equal', 'box'); ax.legend(); plt.tight_layout(); plt.show()

def plot_all_surfaces(empirical_df):
    """Generates the 2x2 plot of model, true, and empirical surfaces using a dynamic range from the data."""
    print("\n" + "#"*70); print("### PLOTTING LEARNED SURFACE COMPARISON ###"); print("#"*70)

    # A. Dynamically calculate plot boundaries based on empirical data range
    # This makes the plot robust to different datasets.
    print(f"\n--- Dynamically calculating plot ranges from '{TRAINING_DATA_CSV.name}' ---")

    def get_expanded_range(data_series, factor):
        """Calculates an expanded range around the data's min and max."""
        min_val, max_val = data_series.min(), data_series.max()
        center = (min_val + max_val) / 2
        half_width = (max_val - min_val) / 2
        expanded_half_width = half_width * factor
        return center - expanded_half_width, center + expanded_half_width

    v1_min, v1_max = get_expanded_range(empirical_df['variable_one'], PLOT_RANGE_EXPANSION_FACTOR)
    v2_min, v2_max = get_expanded_range(empirical_df['variable_two'], PLOT_RANGE_EXPANSION_FACTOR)
    
    print(f"  - variable_one plot range: [{v1_min:.2f}, {v1_max:.2f}]")
    print(f"  - variable_two plot range: [{v2_min:.2f}, {v2_max:.2f}]")

    # B. Create the grid for querying models using the new dynamic ranges
    v1_range = np.linspace(v1_min, v1_max, GRID_POINTS)
    v2_range = np.linspace(v2_min, v2_max, GRID_POINTS)
    v1_grid, v2_grid = np.meshgrid(v1_range, v2_range)

    # C. Calculate the "True Optimal Surface" over the new dynamic grid
    print("--- Calculating the fixed 'True Optimal Surface' (sine wave benchmark) on the dynamic grid ---")
    true_logit = np.sin(v1_grid) + v2_grid
    true_surface_prob = 1 / (1 + np.exp(-true_logit))

    # D. Get predictions from the models over the generated grid
    grid_df = pd.DataFrame({'variable_one': v1_grid.flatten(), 'variable_two': v2_grid.flatten()})
    grid_csv_path = SCRIPT_DIR / 'temp_grid_data.csv'; grid_df.to_csv(grid_csv_path, index=False)
    grid_tsv_path = SCRIPT_DIR / 'temp_rust_grid_data.tsv'
    r_preds_path = SCRIPT_DIR / 'temp_r_surface_preds.csv'
    rust_preds_path = SCRIPT_DIR / 'temp_rust_surface_preds.csv'

    run_r_inference(grid_csv_path, r_preds_path)
    run_rust_inference(grid_csv_path, grid_tsv_path, rust_preds_path)

    r_preds = pd.read_csv(r_preds_path)['r_prediction'].values.reshape(GRID_POINTS, GRID_POINTS)
    rust_preds = pd.read_csv(rust_preds_path)['rust_prediction'].values.reshape(GRID_POINTS, GRID_POINTS)

    # E. Create the 2x2 plot
    print("\n--- Generating 2x2 Surface Plot ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True, sharex=True, sharey=True)
    fig.suptitle("Comparison of Learned Surfaces: P(outcome=1)", fontsize=20)
    levels = np.linspace(0, 1, 21)

    axes[0, 0].contourf(v1_grid, v2_grid, r_preds, levels=levels, cmap='viridis'); axes[0, 0].set_title("R / mgcv Model", fontsize=14); axes[0, 0].set_ylabel("variable_two")
    axes[0, 1].contourf(v1_grid, v2_grid, rust_preds, levels=levels, cmap='viridis'); axes[0, 1].set_title("Rust / gnomon Model", fontsize=14)
    axes[1, 0].contourf(v1_grid, v2_grid, true_surface_prob, levels=levels, cmap='viridis'); axes[1, 0].set_title("True Optimal Surface (Fixed Benchmark)", fontsize=14); axes[1, 0].set_xlabel("variable_one"); axes[1, 0].set_ylabel("variable_two")

    cf = axes[1, 1].hexbin(x=empirical_df['variable_one'], y=empirical_df['variable_two'], C=empirical_df['outcome'], gridsize=40, cmap='viridis', reduce_C_function=np.mean, vmin=0, vmax=1)
    axes[1, 1].set_title(f"Empirical Surface (Hexbin on Training Data)", fontsize=14); axes[1, 1].set_xlabel("variable_one")

    fig.colorbar(cf, ax=axes, orientation='vertical', shrink=0.8, label='Predicted Probability')
    plt.show()

    # F. Cleanup temporary files
    grid_csv_path.unlink(); grid_tsv_path.unlink(); r_preds_path.unlink(); rust_preds_path.unlink();

# --- 4. Main Execution Block ---

def main():
    # --- 1. Check for required files ---
    required_files = [R_MODEL_PATH, RUST_MODEL_CONFIG_PATH, TRAINING_DATA_CSV]
    for f in required_files:
        if not f.is_file():
            print(f"FATAL ERROR: Required file not found: {f}")
            print("Please run the training script first.")
            sys.exit(1)

    # --- 2. Load the original training data ---
    print(f"--- Loading training data from '{TRAINING_DATA_CSV.name}' ---")
    training_df = pd.read_csv(TRAINING_DATA_CSV)

    # --- 3. Run inference on the training data to get predictions ---
    run_r_inference(TRAINING_DATA_CSV, R_PREDICTIONS_CSV)
    run_rust_inference(TRAINING_DATA_CSV, RUST_FORMATTED_TRAINING_DATA_TSV, RUST_PREDICTIONS_CSV)

    # --- 4. Combine data and predictions ---
    r_preds_df = pd.read_csv(R_PREDICTIONS_CSV)
    rust_preds_df = pd.read_csv(RUST_PREDICTIONS_CSV)
    combined_df = pd.concat([training_df, r_preds_df, rust_preds_df], axis=1)

    # --- 5. Report metrics and plot scatter ---
    print_performance_report(combined_df)
    plot_prediction_scatter(combined_df)

    # --- 6. Plot all surfaces ---
    plot_all_surfaces(training_df)

    # --- 7. Final cleanup ---
    R_PREDICTIONS_CSV.unlink(); RUST_PREDICTIONS_CSV.unlink(); RUST_FORMATTED_TRAINING_DATA_TSV.unlink()
    print("\n--- Analysis script finished successfully. ---")

if __name__ == "__main__":
    main()
