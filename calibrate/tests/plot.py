import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, r2_score
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
PREDICTIONS_PATH = PROJECT_ROOT / "predictions.tsv"
TRAIN_DATA_PATH = PROJECT_ROOT / "training_data.tsv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.tsv"


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
    Simulates a dataset with heteroscedastic noise (noise level varies with PC1).
    Returns both the final phenotype and the true, noise-free underlying probability.
    """
    np.random.seed(seed)
    score = np.random.uniform(-3, 3, n_samples)
    pc1 = np.random.normal(0, 1.2, n_samples)

    signal_strength = 0.6
    true_logit = signal_strength * (
        0.8 * np.cos(score * 2) - 1.2 * np.tanh(pc1) +
        1.5 * np.sin(score) * pc1 - 0.5 * (score**2 + pc1**2)
    )
    
    # Calculate the "true" probability before adding noise
    true_probability = 1 / (1 + np.exp(-true_logit))

    # --- HETEROSCEDASTIC NOISE MODEL ---
    # The noise level is no longer constant. It has a base level and increases
    # as the value of PC1 increases, making predictions more uncertain at high PC1 values.
    base_noise_std = 2.0
    pc1_noise_factor = 0.75 # How much PC1 affects noise
    # Noise standard deviation is now a vector, different for each sample
    dynamic_noise_std = base_noise_std + pc1_noise_factor * np.maximum(0, pc1)
    
    noise = np.random.normal(0, dynamic_noise_std, n_samples)

    # Final probability for generating the observable outcome
    final_probability = 1 / (1 + np.exp(-(true_logit + noise)))
    phenotype = np.random.binomial(1, final_probability, n_samples)
    
    df = pd.DataFrame({"phenotype": phenotype, "score": score, "PC1": pc1})
    # Add the true probability for Oracle model calculation
    df['true_prob'] = true_probability
    
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
        sys.exit(1)

def generate_performance_report(df_results):
    """Calculates and prints a side-by-side comparison of model metrics."""
    
    y_true = df_results['phenotype']
    
    models = {
        "GAM (gnomon)": df_results['prediction'],
        "Baseline (Logistic)": df_results['baseline_prediction'],
        "Oracle (Theoretical Best)": df_results['true_prob']
    }
    
    print("\n" + "="*60)
    print("      Model Performance Comparison on Test Set")
    print("="*60)
    
    # --- AUC ---
    print("\n[Discrimination: AUC (Area Under ROC Curve)]")
    print("Higher is better. Measures ability to distinguish classes.")
    for name, y_prob in models.items():
        auc = roc_auc_score(y_true, y_prob)
        print(f"  - {name:<28}: {auc:.4f}")
        
    # --- Brier Score ---
    print("\n[Probabilistic Accuracy: Brier Score]")
    print("Lower is better. Measures accuracy of the predicted probabilities.")
    for name, y_prob in models.items():
        brier = brier_score_loss(y_true, y_prob)
        print(f"  - {name:<28}: {brier:.4f}")
        
    # --- R-squared ---
    print("\n[Goodness of Fit: Pseudo R-squared]")
    print("Measures variance in the binary outcome explained by the probabilities.")
    for name, y_prob in models.items():
        r2 = r2_score(y_true, y_prob)
        print(f"  - {name:<28}: {r2:.4f}")
        
    # --- ECE ---
    print("\n[Calibration: Expected Calibration Error (ECE)]")
    print("Lower is better. Measures how trustworthy the probabilities are.")
    for name, y_prob in models.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        ece = np.mean(np.abs(prob_true - prob_pred))
        print(f"  - {name:<28}: {ece:.4f}")

    # --- Confusion Matrices ---
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
    

def create_analysis_plots(results_df, baseline_probs, oracle_probs, true_prob_surface, score_grid, pc1_grid):
    """Generates a 2x2 grid of plots for model analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle("Comprehensive GAM Model Analysis vs. Baseline and Oracle", fontsize=22, y=0.98)
    
    # --- Define a consistent color range for the first two plots ---
    # Find the global min/max across the true surface and GAM predictions to unify the scale
    vmin = min(true_prob_surface.min(), results_df["prediction"].min())
    vmax = max(true_prob_surface.max(), results_df["prediction"].max())

    # --- 1. Top-Left: Ground Truth Surface ---
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(score_grid, pc1_grid, true_prob_surface, levels=20, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(contour1, ax=ax1, label="True Probability")
    ax1.set_title("1. Ground Truth Probability Surface", fontsize=16)
    ax1.set_xlabel("Polygenic Score (score)")
    ax1.set_ylabel("Principal Component 1 (PC1)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- 2. Top-Right: GAM Model's Predictions ---
    ax2 = axes[0, 1]
    # Use the same colormap and vmin/vmax to ensure the color scales are identical
    scatter = ax2.scatter(
        results_df["score"], results_df["PC1"], c=results_df["prediction"],
        cmap="viridis", edgecolor="k", linewidth=0.5, s=40, vmin=vmin, vmax=vmax
    )
    fig.colorbar(scatter, ax=ax2, label="GAM Predicted Probability")
    ax2.set_title("2. GAM Predictions on Test Data", fontsize=16)
    ax2.set_xlabel("Polygenic Score (score)")
    ax2.set_ylabel("Principal Component 1 (PC1)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- 3. Bottom-Left: Calibration Curve Comparison ---
    ax3 = axes[1, 0]
    y_true = results_df['phenotype']
    
    # Oracle Calibration
    prob_true_oracle, prob_pred_oracle = calibration_curve(y_true, oracle_probs, n_bins=10, strategy='uniform')
    ece_oracle = np.mean(np.abs(prob_true_oracle - prob_pred_oracle))
    
    # GAM Calibration
    prob_true_gam, prob_pred_gam = calibration_curve(y_true, results_df['prediction'], n_bins=10, strategy='uniform')
    ece_gam = np.mean(np.abs(prob_true_gam - prob_pred_gam))
    
    # Baseline Calibration
    prob_true_base, prob_pred_base = calibration_curve(y_true, baseline_probs, n_bins=10, strategy='uniform')
    ece_base = np.mean(np.abs(prob_true_base - prob_pred_base))

    ax3.plot(prob_pred_oracle, prob_true_oracle, "D-", label=f"Oracle (ECE = {ece_oracle:.3f})", color='gold', zorder=4)
    ax3.plot(prob_pred_gam, prob_true_gam, "s-", label=f"GAM (ECE = {ece_gam:.3f})", color='blue', zorder=3)
    ax3.plot(prob_pred_base, prob_true_base, "o--", label=f"Baseline (ECE = {ece_base:.3f})", color='red', zorder=2)
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
    print(f"Project Root Detected: {PROJECT_ROOT}")
    build_rust_project()
    print(f"Using executable: {EXECUTABLE_PATH}\n")

    # 1. Simulate training data
    print(f"--- Simulating {N_SAMPLES_TRAIN} samples for training ---")
    train_df = simulate_data(N_SAMPLES_TRAIN, seed=42)
    
    # 2. Train baseline Logistic Regression model
    print("\n--- Training Baseline Logistic Regression Model ---")
    X_train = train_df[['score', 'PC1']]
    y_train = train_df['phenotype']
    baseline_model = LogisticRegression(solver='liblinear')
    baseline_model.fit(X_train, y_train)
    print("Baseline model trained.")

    # 3. Save training data for gnomon
    train_df[['phenotype', 'score', 'PC1']].to_csv(TRAIN_DATA_PATH, sep="\t", index=False)
    print(f"Saved training data to '{TRAIN_DATA_PATH}'")

    # 4. Run the gnomon 'train' command
    print("\n--- Running 'train' command for GAM model ---")
    train_command = [
        str(EXECUTABLE_PATH), "train",
        "--num-pcs", str(NUM_PCS),
        "--pgs-knots", "3", "--pc-knots", "3",
        "--pgs-degree", "2", "--pc-degree", "2",
        str(TRAIN_DATA_PATH)
    ]
    run_subprocess(train_command, cwd=PROJECT_ROOT)
    print(f"\nGAM model training complete. Model saved to '{MODEL_PATH}'\n")

    # 5. Simulate test data
    print(f"--- Simulating {N_SAMPLES_TEST} samples for inference ---")
    test_df_full = simulate_data(N_SAMPLES_TEST, seed=101)
    
    # 6. Make predictions with the baseline model
    print("\n--- Generating predictions with Baseline Model ---")
    X_test = test_df_full[['score', 'PC1']]
    baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
    
    # 7. Save test data for gnomon
    test_df_full[['score', 'PC1']].to_csv(TEST_DATA_PATH, sep="\t", index=False)
    print(f"Saved test data to '{TEST_DATA_PATH}'")

    # 8. Run the gnomon 'infer' command
    print("\n--- Running 'infer' command for GAM model ---")
    infer_command = [
        str(EXECUTABLE_PATH), "infer",
        "--model", str(MODEL_PATH),
        str(TEST_DATA_PATH)
    ]
    run_subprocess(infer_command, cwd=PROJECT_ROOT)
    print(f"\nInference complete. Predictions saved to '{PREDICTIONS_PATH}'\n")

    # 9. Load results and analyze
    print("\n--- Analyzing Results ---")
    if not PREDICTIONS_PATH.is_file():
        print(f"Error: Predictions file not found at '{PREDICTIONS_PATH}'. Cannot plot results.")
        return

    predictions_df = pd.read_csv(PREDICTIONS_PATH, sep="\t")
    # Combine the full test data with predictions from all models
    results_df = pd.concat([test_df_full.reset_index(drop=True), predictions_df], axis=1)
    results_df['baseline_prediction'] = baseline_probs

    # Print the new side-by-side performance report
    generate_performance_report(results_df)
    
    # Create the grid for the surface plot
    score_grid, pc1_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3.5, 3.5, 100))
    signal_strength = 0.6
    true_logit_surface = signal_strength * (
        0.8 * np.cos(score_grid * 2) - 1.2 * np.tanh(pc1_grid) +
        1.5 * np.sin(score_grid) * pc1_grid - 0.5 * (score_grid**2 + pc1_grid**2)
    )
    true_prob_surface = 1 / (1 + np.exp(-true_logit_surface))

    # Call the plotting function
    create_analysis_plots(
        results_df, 
        results_df['baseline_prediction'], 
        results_df['true_prob'], 
        true_prob_surface, 
        score_grid, 
        pc1_grid
    )

if __name__ == "__main__":
    main()
