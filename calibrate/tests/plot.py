import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.calibration import calibration_curve

# --- Path Configuration ---
# Get the absolute path of the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up to the project root directory
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Define all paths relative to the project root
EXECUTABLE_NAME = "gnomon"
EXECUTABLE_PATH = PROJECT_ROOT / "target" / "release" / EXECUTABLE_NAME
MODEL_PATH = PROJECT_ROOT / "model.toml"
PREDICTIONS_PATH = PROJECT_ROOT / "predictions.tsv" # The fixed output file for the tool
TRAIN_DATA_PATH = PROJECT_ROOT / "training_data.tsv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.tsv"
GRID_DATA_PATH = PROJECT_ROOT / "grid_data_for_surface.tsv"


# --- Simulation Configuration ---
N_SAMPLES_TRAIN = 5000
N_SAMPLES_TEST = 1000
NUM_PCS = 1  # Simulate 1 PC for easy 2D visualization

# --- Helper Functions ---

def build_rust_project():
    """Checks for the executable and compiles the Rust project if not found."""
    if not EXECUTABLE_PATH.is_file():
        print("--- Executable not found. Compiling Rust Project... ---")
        try:
            subprocess.run(
                ["cargo", "build", "--release", "--bin", EXECUTABLE_NAME],
                check=True, text=True, cwd=PROJECT_ROOT
            )
            print("--- Compilation successful. ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n--- ERROR: Rust compilation failed: {e} ---")
            print("Please ensure Rust/Cargo is installed and in your PATH.")
            sys.exit(1)
    else:
        print(f"--- Found existing executable. Skipping build. ---")


def simulate_data(n_samples: int, seed: int):
    """
    Simulates a dataset with heteroscedastic noise.
    Returns phenotype, covariates, the pure signal probability, and the final oracle probability.
    """
    """
    np.random.seed(seed)
    score = np.random.uniform(-3, 3, n_samples)
    pc1 = np.random.normal(0, 1.2, n_samples)

    signal_strength = 0.6
    true_logit = signal_strength * (
        0.8 * np.cos(score * 2) - 1.2 * np.tanh(pc1) +
        1.5 * np.sin(score) * pc1 - 0.5 * (score**2 + pc1**2)
    )
    
    # Calculate the "signal" probability (noise-free)
    signal_probability = 1 / (1 + np.exp(-true_logit))

    # --- HETEROSCEDASTIC NOISE MODEL ---
    base_noise_std = 2.0
    pc1_noise_factor = 0.75
    dynamic_noise_std = base_noise_std + pc1_noise_factor * np.maximum(0, pc1)
    noise = np.random.normal(0, dynamic_noise_std, n_samples)

    # Final "oracle" probability for generating the observable outcome
    oracle_probability = 1 / (1 + np.exp(-(true_logit + noise)))
    phenotype = np.random.binomial(1, oracle_probability, n_samples)
    
    df = pd.DataFrame({"phenotype": phenotype, "score": score, "PC1": pc1})
    # Add both the pure signal and the final oracle probabilities
    df['signal_prob'] = signal_probability
    df['oracle_prob'] = oracle_probability
    
    return df


def run_subprocess(command, cwd):
    """Runs a subprocess and handles KeyboardInterrupt cleanly."""
    try:
        print(f"Executing: {' '.join(map(str, command))}\n")
        subprocess.run(command, check=True, text=True, cwd=cwd)
    except KeyboardInterrupt:
        print("\n--- Process interrupted by user (Ctrl+C). ---")
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
    p_mean = np.mean(y_true)
    log_likelihood_null = np.sum(y_true * np.log(p_mean) + (1 - y_true) * np.log(1 - p_mean))
    log_likelihood_model = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    n = len(y_true)
    r2_cs = 1 - np.exp((2/n) * (log_likelihood_null - log_likelihood_model))
    max_r2_cs = 1 - np.exp((2/n) * log_likelihood_null)
    if max_r2_cs == 0:
        return 0.0
    return r2_cs / max_r2_cs


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
    
    print("\n[Discrimination: AUC (Area Under ROC Curve)]")
    print("Higher is better. Measures ability to distinguish classes.")
    for name, y_prob in models.items():
        auc = roc_auc_score(y_true, y_prob)
        print(f"  - {name:<28}: {auc:.4f}")
        
    print("\n[Probabilistic Accuracy: Brier Score]")
    print("Lower is better. Measures accuracy of the predicted probabilities.")
    for name, y_prob in models.items():
        brier = brier_score_loss(y_true, y_prob)
        print(f"  - {name:<28}: {brier:.4f}")
        
    print("\n[Fit: Nagelkerke's R-squared]")
    print("Higher is better (0-1). Likelihood-based, common in PGS literature.")
    for name, y_prob in models.items():
        stable_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
        r2_n = nagelkerkes_r2(y_true, stable_prob)
        print(f"  - {name:<28}: {r2_n:.4f}")
        
    print("\n[Separation: Tjur's R-squared]")
    print("Higher is better (0-1). The mean difference in prediction between classes.")
    for name, y_prob in models.items():
        r2_t = tjurs_r2(y_true, y_prob)
        print(f"  - {name:<28}: {r2_t:.4f}")
        
    print("\n[Calibration: Expected Calibration Error (ECE)]")
    print("Lower is better. Measures how trustworthy the probabilities are.")
    for name, y_prob in models.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        ece = np.mean(np.abs(prob_true - prob_pred))
        print(f"  - {name:<28}: {ece:.4f}")


    print("\n[Confusion Matrices (at threshold=0.5)]")
    for name, y_prob in models.items():
        y_pred_class = (y_prob > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred_class)
        cm_proportions = cm / cm.sum()
        print(f"\n  --- {name} ---")
        print(f"  (Proportions of total data)")
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
        cmap="viridis", edgecolor="k", linewidth=0.5, s=40, vmin=vmin, vmax=vmax
    )
    ax2.set_title("2. GAM's Learned Surface with Predictions Overlay", fontsize=16)
    ax2.set_xlabel("Polygenic Score (score)")
    ax2.set_ylabel("Principal Component 1 (PC1)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- 3. Bottom-Left: Calibration Curve Comparison ---
    ax3 = axes[1, 0]
    y_true = results_df['phenotype']
    models_for_calib = {
        "Oracle": (results_df['oracle_prob'], 'gold', 'D-', 5),
        "Signal Model": (results_df['signal_prob'], 'darkorange', 'x--', 4),
        "GAM": (results_df['prediction'], 'blue', 's-', 3),
        "Baseline": (results_df['baseline_prediction'], 'red', 'o--', 2)
    }
    for name, (y_prob, color, style, z) in models_for_calib.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        ece = np.mean(np.abs(prob_true - prob_pred))
        ax3.plot(prob_pred, prob_true, style, label=f"{name} (ECE = {ece:.3f})", color=color, zorder=z)
    ax3.plot([0, 1], [0, 1], "k:", label="Perfect Calibration", zorder=1)
    ax3.set_title("3. Calibration Curve (Lower ECE is Better)", fontsize=16)
    ax3.set_xlabel("Mean Predicted Probability (per bin)")
    ax3.set_ylabel("Observed Frequency of Positives (per bin)")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

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
    print("\nPlot generated. Close the plot window to finish the script.")
    plt.show()

# --- Main Script Logic ---

def main():
    """Main function to run the end-to-end simulation and plotting."""
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
        print(f"\nGAM model training complete. Model saved to '{MODEL_PATH}'\n")

        # 4. Simulate and prepare test data
        print(f"--- Simulating {N_SAMPLES_TEST} samples for inference ---")
        test_df_full = simulate_data(N_SAMPLES_TEST, seed=101)
        test_df_full[['score', 'PC1']].to_csv(TEST_DATA_PATH, sep="\t", index=False)
        print(f"Saved test data to '{TEST_DATA_PATH}'")

        # 5. Get predictions from baseline model
        print("\n--- Generating predictions with Baseline Model ---")
        baseline_probs = baseline_model.predict_proba(test_df_full[['score', 'PC1']])[:, 1]
        
        # 6. Run 'infer' for GAM on TEST DATA and load results
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
        
        # 8. Generate GAM surface plot data
        print("\n--- Generating GAM surface plot data ---")
        score_grid, pc1_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3.5, 3.5, 100))
        grid_df = pd.DataFrame({'score': score_grid.ravel(), 'PC1': pc1_grid.ravel()})
        grid_df.to_csv(GRID_DATA_PATH, sep='\t', index=False)
        
        # FIX: Run 'infer' on the grid data. This will OVERWRITE 'predictions.tsv'.
        # This is the correct behavior as the tool does not support a custom output path.
        print("\n--- Running 'infer' command on grid data for surface plot ---")
        infer_grid_command = [str(EXECUTABLE_PATH), "infer", "--model", str(MODEL_PATH), str(GRID_DATA_PATH)]
        run_subprocess(infer_grid_command, cwd=PROJECT_ROOT)
        
        # FIX: Load the newly-overwritten 'predictions.tsv' file.
        grid_preds_df = pd.read_csv(PREDICTIONS_PATH, sep="\t")
        gam_surface = grid_preds_df['prediction'].values.reshape(score_grid.shape)
        print("GAM surface data generated.")

        # 9. Calculate the true signal surface for plotting
        signal_strength = 0.6
        true_logit_surface = signal_strength * (
            0.8 * np.cos(score_grid * 2) - 1.2 * np.tanh(pc1_grid) +
            1.5 * np.sin(score_grid) * pc1_grid - 0.5 * (score_grid**2 + pc1_grid**2)
        )
        signal_surface = 1 / (1 + np.exp(-true_logit_surface))

        # 10. Call the master plotting function
        create_analysis_plots(results_df, gam_surface, signal_surface, score_grid, pc1_grid)

    finally:
        # Clean up temporary files
        print("\n--- Cleaning up temporary files ---")
        # FIX: Removed GRID_PREDICTIONS_PATH as it's never created.
        for f in [TRAIN_DATA_PATH, TEST_DATA_PATH, GRID_DATA_PATH, PREDICTIONS_PATH]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Removed {f.name}")
                except OSError as e:
                    print(f"Error removing file {f.name}: {e}")

if __name__ == "__main__":
    main()
