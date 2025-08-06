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
TEST_DATA_CSV = SCRIPT_DIR / 'test_data.csv' # Use test data for all evaluation

# --- Temporary File Paths for Inference ---
R_PREDICTIONS_CSV = SCRIPT_DIR / 'r_model_predictions.csv'
RUST_PREDICTIONS_CSV = SCRIPT_DIR / 'rust_model_predictions.csv'
RUST_FORMATTED_INFERENCE_DATA_TSV = SCRIPT_DIR / 'rust_formatted_inference_data.tsv'

# --- Plotting Parameters ---
GRID_POINTS = 300 # Reduced for faster plotting
PLOT_RANGE_EXPANSION_FACTOR = 1.1 # Expand plot boundaries by 10%

# --- 2. Re-usable Metric Functions (Unchanged) ---

def tjurs_r2(y_true, y_prob):
    """Calculates Tjur's R-squared for model performance."""
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_prob = pd.Series(y_prob).reset_index(drop=True)
    mean_prob_cases = y_prob[y_true == 1].mean()
    mean_prob_controls = y_prob[y_true == 0].mean()
    return mean_prob_cases - mean_prob_controls

def nagelkerkes_r2(y_true, y_prob):
    """Calculates Nagelkerke's R-squared for model performance."""
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
    """Calculates and prints all performance metrics based on test data."""
    y_true, models = df['outcome'], {"R / mgcv": df['r_prediction'], "Rust / gnomon": df['rust_prediction']}
    print("\n" + "="*60); print("      Model Performance on TEST Data"); print("="*60)
    print("\n[Discrimination: AUC]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {roc_auc_score(y_true, p):.4f}")
    print("\n[Accuracy: Brier Score]"); print("Lower is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {brier_score_loss(y_true, p):.4f}")
    print("\n[Fit: Nagelkerke's R-squared]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {nagelkerkes_r2(y_true, p):.4f}")
    print("\n[Separation: Tjur's R-squared]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {tjurs_r2(y_true, p):.4f}")
    print("\n" + "="*60)

def plot_prediction_comparisons(df):
    """Generates a 1x3 plot comparing model predictions to each other and to the true outcomes."""
    print("\n--- Generating 1x3 Prediction Comparison Scatter Plot ---")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    fig.suptitle("Prediction Comparisons on Test Data", fontsize=20)

    # --- Plot 1: R vs. Rust ---
    mae = np.mean(np.abs(df['r_prediction'] - df['rust_prediction']))
    axes[0].scatter(df['r_prediction'], df['rust_prediction'], alpha=0.2, s=10, rasterized=True)
    axes[0].plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
    axes[0].set(xlabel="R / mgcv Prediction", ylabel="Rust / gnomon Prediction", title=f"Model vs. Model (MAE = {mae:.4f})")
    axes[0].grid(True, linestyle='--'); axes[0].set_aspect('equal', 'box'); axes[0].legend()

    # --- Plot 2: R vs. True Outcome ---
    tjur_r = tjurs_r2(df['outcome'], df['r_prediction'])
    axes[1].scatter(df['r_prediction'], df['outcome'], alpha=0.1, s=10, rasterized=True)
    axes[1].set(xlabel="R / mgcv Prediction", ylabel="True Outcome", title=f"R vs. Outcome (Tjur's R² = {tjur_r:.3f})")
    axes[1].grid(True, linestyle='--'); axes[1].set_yticks([0, 1])

    # --- Plot 3: Rust vs. True Outcome ---
    tjur_rust = tjurs_r2(df['outcome'], df['rust_prediction'])
    axes[2].scatter(df['rust_prediction'], df['outcome'], alpha=0.1, s=10, rasterized=True)
    axes[2].set(xlabel="Rust / gnomon Prediction", ylabel="True Outcome", title=f"Rust vs. Outcome (Tjur's R² = {tjur_rust:.3f})")
    axes[2].grid(True, linestyle='--'); axes[2].set_yticks([0, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

def plot_model_surfaces(test_df):
    """Generates a 1x3 plot of model surfaces and the empirical test data surface."""
    print("\n" + "#"*70); print("### PLOTTING LEARNED SURFACE COMPARISON ###"); print("#"*70)

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

    # D. Create the 1x3 plot
    print("\n--- Generating 1x3 Surface Plot ---")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True, sharex=True, sharey=True)
    fig.suptitle("Comparison of Learned Surfaces vs. Empirical Test Data", fontsize=20)
    levels = np.linspace(0, 1, 21)

    axes[0].contourf(v1_grid, v2_grid, r_preds, levels=levels, cmap='viridis'); axes[0].set_title("R / mgcv Model", fontsize=14); axes[0].set_ylabel("variable_two")
    axes[1].contourf(v1_grid, v2_grid, rust_preds, levels=levels, cmap='viridis'); axes[1].set_title("Rust / gnomon Model", fontsize=14)
    cf = axes[2].hexbin(x=test_df['variable_one'], y=test_df['variable_two'], C=test_df['outcome'], gridsize=40, cmap='viridis', reduce_C_function=np.mean, vmin=0, vmax=1)
    axes[2].set_title(f"Empirical Surface (Test Data)", fontsize=14)

    for ax in axes: ax.set_xlabel("variable_one")
    fig.colorbar(cf, ax=axes, orientation='vertical', shrink=0.8, label='P(outcome=1)')
    plt.show()

    # E. Cleanup temporary files
    grid_csv_path.unlink(); grid_tsv_path.unlink(); r_preds_path.unlink(); rust_preds_path.unlink();

# --- 4. Main Execution Block ---

def main():
    # --- 1. Check for required files ---
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
    run_rust_inference(TEST_DATA_CSV, RUST_FORMATTED_INFERENCE_DATA_TSV, RUST_PREDICTIONS_CSV)

    # --- 4. Combine data and predictions ---
    r_preds_df = pd.read_csv(R_PREDICTIONS_CSV)
    rust_preds_df = pd.read_csv(RUST_PREDICTIONS_CSV)
    combined_df = pd.concat([test_df, r_preds_df, rust_preds_df], axis=1)

    # --- 5. Report metrics and plot comparisons on test data ---
    print_performance_report(combined_df)
    plot_prediction_comparisons(combined_df)
    plot_model_surfaces(test_df)

    # --- 6. Final cleanup ---
    R_PREDICTIONS_CSV.unlink()
    RUST_PREDICTIONS_CSV.unlink()
    RUST_FORMATTED_INFERENCE_DATA_TSV.unlink()
    print("\n--- Analysis script finished successfully. ---")

if __name__ == "__main__":
    main()
