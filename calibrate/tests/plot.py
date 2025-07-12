import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- Path Configuration ---
# Get the absolute path of the directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent
# Navigate up to the project root directory
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Define all paths relative to the project root. These are now absolute Path objects.
EXECUTABLE_NAME = "gnomon"
EXECUTABLE_PATH = PROJECT_ROOT / "target" / "release" / EXECUTABLE_NAME
MODEL_PATH = PROJECT_ROOT / "model.toml"
PREDICTIONS_PATH = PROJECT_ROOT / "predictions.tsv"
TRAIN_DATA_PATH = PROJECT_ROOT / "training_data.tsv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.tsv"


# --- Simulation Configuration ---
N_SAMPLES_TRAIN = 2000
N_SAMPLES_TEST = 500
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
    Simulates a dataset with a complex, non-linear relationship and added noise.
    """
    np.random.seed(seed)
    score = np.random.uniform(-3, 3, n_samples)
    pc1 = np.random.normal(0, 1.2, n_samples)
    true_logit = (
        0.8 * np.cos(score * 2) - 1.2 * np.tanh(pc1) +
        1.5 * np.sin(score) * pc1 - 0.5 * (score**2 + pc1**2)
    )
    noise = np.random.normal(0, 0.5, n_samples)
    probability = 1 / (1 + np.exp(-(true_logit + noise)))
    phenotype = np.random.binomial(1, probability, n_samples)
    return pd.DataFrame({"phenotype": phenotype, "score": score, "PC1": pc1})


def run_subprocess(command, cwd):
    """Runs a subprocess and handles KeyboardInterrupt cleanly."""
    try:
        # Using str() on the Path objects ensures they are passed as absolute paths
        # to the command.
        print(f"Executing: {' '.join(map(str, command))}\n")
        subprocess.run(command, check=True, text=True, cwd=cwd)
    except KeyboardInterrupt:
        print("\n--- Process interrupted by user (Ctrl+C). ---")
        print("The script will now exit. Any files created before the interruption (like training_data.tsv) will remain.")
        sys.exit(1) # Exit immediately
    except subprocess.CalledProcessError as e:
        print(f"\n--- A command FAILED (Exit Code: {e.returncode}) ---")
        sys.exit(1)

# --- Main Script Logic ---

def main():
    """Main function to run the end-to-end simulation and plotting."""
    print(f"Project Root Detected: {PROJECT_ROOT}")
    build_rust_project()
    print(f"Using executable: {EXECUTABLE_PATH}\n")

    # 1. Simulate and save training data
    print(f"--- Simulating {N_SAMPLES_TRAIN} samples for training ---")
    train_df = simulate_data(N_SAMPLES_TRAIN, seed=42)
    train_df.to_csv(TRAIN_DATA_PATH, sep="\t", index=False)
    print(f"Saved training data to '{TRAIN_DATA_PATH}'")

    # 2. Run the 'train' command using absolute paths
    print("\n--- Running 'train' command ---")
    train_command = [
        str(EXECUTABLE_PATH), "train",
        "--num-pcs", str(NUM_PCS),
        "--pgs-knots", "12", "--pc-knots", "8",
        "--pgs-degree", "3", "--pc-degree", "3",
        str(TRAIN_DATA_PATH)  # Use absolute path
    ]
    run_subprocess(train_command, cwd=PROJECT_ROOT)
    print(f"\nModel training complete. Model saved to '{MODEL_PATH}'\n")

    # 3. Simulate and save test data
    print(f"--- Simulating {N_SAMPLES_TEST} samples for inference ---")
    test_df_full = simulate_data(N_SAMPLES_TEST, seed=101)
    test_df_full.drop(columns=["phenotype"]).to_csv(TEST_DATA_PATH, sep="\t", index=False)
    print(f"Saved test data to '{TEST_DATA_PATH}'")

    # 4. Run the 'infer' command using absolute paths
    print("\n--- Running 'infer' command ---")
    infer_command = [
        str(EXECUTABLE_PATH), "infer",
        "--model", str(MODEL_PATH),       # Use absolute path
        str(TEST_DATA_PATH)              # Use absolute path
    ]
    run_subprocess(infer_command, cwd=PROJECT_ROOT)
    print(f"\nInference complete. Predictions saved to '{PREDICTIONS_PATH}'\n")

    # 5. Load results and plot
    print("--- Loading results and plotting ---")
    if not PREDICTIONS_PATH.is_file():
        print(f"Error: Predictions file not found at '{PREDICTIONS_PATH}'. Cannot plot results.")
        return

    predictions_df = pd.read_csv(PREDICTIONS_PATH, sep="\t")
    results_df = pd.concat([test_df_full.reset_index(drop=True), predictions_df], axis=1)

    # Create a grid to plot the true probability surface
    score_grid, pc1_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3.5, 3.5, 100))
    true_logit_surface = (
        0.8 * np.cos(score_grid * 2) - 1.2 * np.tanh(pc1_grid) +
        1.5 * np.sin(score_grid) * pc1_grid - 0.5 * (score_grid**2 + pc1_grid**2)
    )
    true_prob_surface = 1 / (1 + np.exp(-true_logit_surface))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(score_grid, pc1_grid, true_prob_surface, levels=20, cmap="viridis", alpha=0.8)
    fig.colorbar(contour, ax=ax, label="True Probability of Phenotype=1")
    scatter = ax.scatter(
        results_df["score"], results_df["PC1"], c=results_df["prediction"],
        cmap="plasma", edgecolor="black", linewidth=0.5, s=50,
    )
    fig.colorbar(scatter, ax=ax, label="Model Predicted Probability")
    ax.set_title("GAM Model Predictions vs. Complex Ground Truth", fontsize=16)
    ax.set_xlabel("Polygenic Score (score)", fontsize=12)
    ax.set_ylabel("Principal Component 1 (PC1)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    print("\nPlot generated. Close the plot window to finish the script.")
    plt.show()

if __name__ == "__main__":
    main()
