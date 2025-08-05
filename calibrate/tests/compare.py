import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

# --- 1. Import the data generation function from the existing script ---
from mgcv import generate_data
# --- 2. Define Paths and Parameters ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

GNOMON_EXECUTABLE = WORKSPACE_ROOT / "target" / "release" / "gnomon"
R_MODEL_PATH = SCRIPT_DIR / 'gam_model_fit.rds'
RUST_MODEL_CONFIG_PATH = PROJECT_ROOT / 'model.toml'

# Permanent files for scatter plot comparison
INFERENCE_DATA_CSV = SCRIPT_DIR / 'new_inference_data.csv'
RUST_FORMATTED_INFERENCE_DATA_TSV = SCRIPT_DIR / 'rust_formatted_inference_data.tsv'
R_PREDICTIONS_CSV = SCRIPT_DIR / 'r_model_predictions.csv'
RUST_PREDICTIONS_CSV = SCRIPT_DIR / 'rust_model_predictions.csv'

N_INFERENCE_SAMPLES = 1000
NOISE_STD_DEV = 0.5
# Increased resolution for smoother surface plots
GRID_POINTS = 100

# --- 3. Re-usable Metric Functions ---

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

# --- 4. Main Execution Blocks ---

def generate_and_run_scatter_comparison():
    """Generates random data, runs inference, prints metrics, and plots scatter."""
    print("\n" + "#"*70); print("### STEP 1: SCATTER PLOT AND METRIC COMPARISON ON RANDOM DATA ###"); print("#"*70)
    print("\n--- 1A. Generating New Data for Inference ---")
    generate_data(N_INFERENCE_SAMPLES, NOISE_STD_DEV).to_csv(INFERENCE_DATA_CSV, index=False)
    print(f"Successfully generated {N_INFERENCE_SAMPLES} samples.")
    run_r_inference(INFERENCE_DATA_CSV, R_PREDICTIONS_CSV)
    run_rust_inference(INFERENCE_DATA_CSV, RUST_FORMATTED_INFERENCE_DATA_TSV, RUST_PREDICTIONS_CSV)
    compare_and_plot_scatter()

def generate_and_plot_surfaces():
    """Generates grid data, runs inference, and plots all four surfaces."""
    print("\n" + "#"*70); print("### STEP 2: LEARNED SURFACE COMPARISON PLOT ###"); print("#"*70)

    # A. Create grid and get data for benchmark surfaces
    print("\n--- 2A. Generating Grid and Calculating Benchmark Surfaces ---")
    v1_range = np.linspace(0, 2 * np.pi, GRID_POINTS)
    v2_range = np.linspace(-1.5, 1.5, GRID_POINTS)
    v1_grid, v2_grid = np.meshgrid(v1_range, v2_range)

    # Surface 1: The "True" optimal surface (noise-free ground truth)
    true_logit = np.sin(v1_grid) + v2_grid
    true_surface_prob = 1 / (1 + np.exp(-true_logit))

    # Data for Surface 2: Load the actual CSV data used for metrics. This will
    # be plotted directly using a hexbin plot.
    empirical_df = pd.read_csv(INFERENCE_DATA_CSV)

    # B. Get predictions from the models for the grid
    grid_df = pd.DataFrame({'variable_one': v1_grid.flatten(), 'variable_two': v2_grid.flatten()})
    grid_csv_path = SCRIPT_DIR / 'temp_grid_data.csv'; grid_df.to_csv(grid_csv_path, index=False)
    grid_tsv_path = SCRIPT_DIR / 'temp_rust_grid_data.tsv'
    r_preds_path = SCRIPT_DIR / 'temp_r_surface_preds.csv'
    rust_preds_path = SCRIPT_DIR / 'temp_rust_surface_preds.csv'

    run_r_inference(grid_csv_path, r_preds_path)
    run_rust_inference(grid_csv_path, grid_tsv_path, rust_preds_path)

    r_preds = pd.read_csv(r_preds_path)['r_prediction'].values.reshape(GRID_POINTS, GRID_POINTS)
    rust_preds = pd.read_csv(rust_preds_path)['rust_prediction'].values.reshape(GRID_POINTS, GRID_POINTS)

    # C. Create the 2x2 plot
    print("\n--- 2B. Generating 2x2 Surface Plot ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True, sharex=True, sharey=True)
    fig.suptitle("Comparison of Learned Surfaces: P(outcome=1)", fontsize=20)
    levels = np.linspace(0, 1, 21)

    # Top-left: R/mgcv Model
    axes[0, 0].contourf(v1_grid, v2_grid, r_preds, levels=levels, cmap='viridis')
    axes[0, 0].set_title("R / mgcv Model", fontsize=14)
    axes[0, 0].set_ylabel("variable_two")

    # Top-right: Rust/gnomon Model
    axes[0, 1].contourf(v1_grid, v2_grid, rust_preds, levels=levels, cmap='viridis')
    axes[0, 1].set_title("Rust / gnomon Model", fontsize=14)

    # Bottom-left: True Optimal Surface
    axes[1, 0].contourf(v1_grid, v2_grid, true_surface_prob, levels=levels, cmap='viridis')
    axes[1, 0].set_title("True Optimal Surface (Noise-Free)", fontsize=14)
    axes[1, 0].set_xlabel("variable_one"); axes[1, 0].set_ylabel("variable_two")

    # Bottom-right: Empirical Surface using a Hexagonal Binning plot from the CSV data
    # The 'C' argument takes the 'outcome' values. 'reduce_C_function=np.mean' calculates
    # the average outcome (i.e., empirical probability) for all points in each hexagon.
    cf = axes[1, 1].hexbin(
        x=empirical_df['variable_one'],
        y=empirical_df['variable_two'],
        C=empirical_df['outcome'],
        gridsize=40,  # Number of hexagons across the x-axis. A key tuning parameter.
        cmap='viridis',
        reduce_C_function=np.mean,
        vmin=0, vmax=1  # Ensure color scale is consistent with other plots
    )
    axes[1, 1].set_title(f"Empirical Surface (Hexbin on {N_INFERENCE_SAMPLES} CSV points)", fontsize=14)
    axes[1, 1].set_xlabel("variable_one")

    fig.colorbar(cf, ax=axes, orientation='vertical', shrink=0.8, label='Predicted Probability')
    plt.show()

    # D. Cleanup
    grid_csv_path.unlink(); grid_tsv_path.unlink(); r_preds_path.unlink(); rust_preds_path.unlink()


# --- 5. Helper Functions ---

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
    y_true, models = df['outcome'], {"R / mgcv": df['r_prediction'], "Rust / gnomon": df['rust_prediction']}
    print("\n" + "="*60); print("      Model Performance Comparison"); print("="*60)
    print("\n[Discrimination: AUC]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {roc_auc_score(y_true, p):.4f}")
    print("\n[Accuracy: Brier Score]"); print("Lower is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {brier_score_loss(y_true, p):.4f}")
    print("\n[Fit: Nagelkerke's R-squared]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {nagelkerkes_r2(y_true, p):.4f}")
    print("\n[Separation: Tjur's R-squared]"); print("Higher is better.")
    for n, p in models.items(): print(f"  - {n:<20}: {tjurs_r2(y_true, p):.4f}")
    print("\n" + "="*60)

def compare_and_plot_scatter():
    df = pd.concat([pd.read_csv(INFERENCE_DATA_CSV)[['outcome']], pd.read_csv(R_PREDICTIONS_CSV), pd.read_csv(RUST_PREDICTIONS_CSV)], axis=1)
    print_performance_report(df)
    mae = np.mean(np.abs(df['r_prediction'] - df['rust_prediction']))
    print(f"\n[Inter-Model Agreement: MAE]\n  - MAE: {mae:.6f}")
    fig, ax = plt.subplots(figsize=(8, 8)); ax.scatter(df['r_prediction'], df['rust_prediction'], alpha=0.5, s=20)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement'); ax.set(xlabel="R / mgcv Predictions", ylabel="Rust / gnomon Predictions", title=f"Prediction Comparison (MAE = {mae:.6f})")
    ax.grid(True, linestyle='--'); ax.set_aspect('equal', 'box'); ax.legend(); plt.tight_layout(); plt.show()

def main():
    if not R_MODEL_PATH.is_file() or not RUST_MODEL_CONFIG_PATH.is_file():
        print("ERROR: Required model files not found."); sys.exit(1)
    generate_and_run_scatter_comparison()
    generate_and_plot_surfaces()
    print("\n--- Script finished successfully. ---")

if __name__ == "__main__":
    main()
