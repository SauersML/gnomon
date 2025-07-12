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

# Define all paths relative to the project root
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
    if EXECUTABLE_PATH.exists():
        print(f"--- Found existing executable at '{EXECUTABLE_PATH}'. Skipping build. ---")
        return

    print("--- Executable not found. Compiling Rust Project from root directory... ---")
    try:
        # Check if Rust/Cargo is installed
        subprocess.run(["cargo", "--version"], check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: Rust/Cargo is not installed or not in your PATH.")
        print("Please install Rust from https://rustup.rs/ to continue.")
        sys.exit(1)

    # Stream compilation output in real-time.
    # By removing 'capture_output', subprocess output goes directly to the console.
    # 'check=True' will raise an error and print stderr if compilation fails.
    build_command = ["cargo", "build", "--release", "--bin", EXECUTABLE_NAME]
    print(f"Executing: {' '.join(build_command)}")
    try:
        subprocess.run(
            build_command,
            check=True,
            text=True,    # Ensures output is decoded as text
            cwd=PROJECT_ROOT,
        )
    except subprocess.CalledProcessError as e:
        print(f"\n--- Rust Compilation FAILED (Exit Code: {e.returncode}) ---")
        sys.exit(1)

    print("--- Compilation successful. ---")
    if not EXECUTABLE_PATH.exists():
        print(f"\nERROR: Could not find compiled executable at '{EXECUTABLE_PATH}' after build.")
        sys.exit(1)


def simulate_data(n_samples: int, seed: int):
    """
    REVISED: Simulates a dataset with a more complex, non-linear relationship and added noise.
    This function defines a "ground truth" that the GAM will try to learn.
    """
    np.random.seed(seed)
    # Widen the distributions slightly for a more varied dataset
    score = np.random.uniform(-3, 3, n_samples)
    pc1 = np.random.normal(0, 1.2, n_samples)

    # The true, non-linear log-odds function (the "secret" the GAM must discover)
    # This relationship is now more complex, featuring multiple interactions.
    true_logit = (
        0.8 * np.cos(score * 2)                  # A wave-like pattern along the score axis
        - 1.2 * np.tanh(pc1)                     # A saturating effect for PC1
        + 1.5 * np.sin(score) * pc1              # A twisted, non-linear interaction term
        - 0.5 * (score**2 + pc1**2)              # A radial dampening effect (pulls probabilities to center)
    )

    # Add random Gaussian noise to the logit. This makes the relationship stochastic,
    # meaning the features don't perfectly determine the outcome. This is more realistic.
    noise = np.random.normal(0, 0.5, n_samples)  # Increase standard deviation for more noise
    logit_with_noise = true_logit + noise

    # Convert the final log-odds to a probability
    probability = 1 / (1 + np.exp(-logit_with_noise))

    # Generate the binary phenotype based on the calculated probability
    phenotype = np.random.binomial(1, probability, n_samples)

    return pd.DataFrame({"phenotype": phenotype, "score": score, "PC1": pc1})

# --- Main Script Logic ---

def main():
    """Main function to run the end-to-end simulation and plotting."""
    print(f"Project Root Detected: {PROJECT_ROOT}")

    build_rust_project()
    print(f"Using executable: {EXECUTABLE_PATH}\n")

    # 1. Simulate and save training data
    # This data is intentionally not deleted to adhere to the requirements.
    # Overwriting is the default and desired behavior.
    print(f"--- Simulating {N_SAMPLES_TRAIN} samples for training ---")
    train_df = simulate_data(N_SAMPLES_TRAIN, seed=42)
    train_df.to_csv(TRAIN_DATA_PATH, sep="\t", index=False)
    print(f"Saved training data to '{TRAIN_DATA_PATH}'. This file will be overwritten on next run.\n")

    # 2. Run the 'train' command
    print("--- Running 'train' command ---")
    train_command = [
        str(EXECUTABLE_PATH),
        "train",
        "--num-pcs", str(NUM_PCS),
        "--pgs-knots", "12",
        "--pc-knots", "8",
        "--pgs-degree", "3",
        "--pc-degree", "3",
        TRAIN_DATA_PATH.name,
    ]
    print(f"Executing: {' '.join(map(str, train_command))}")
    try:
        # Stream gnomon output in real-time
        subprocess.run(
            train_command,
            check=True,
            text=True,
            cwd=PROJECT_ROOT
        )
    except subprocess.CalledProcessError as e:
        print(f"\n--- Model Training FAILED (Exit Code: {e.returncode}) ---")
        return
    print(f"\nModel training complete. Model saved to '{MODEL_PATH}'\n")

    # 3. Simulate and save test data
    print(f"--- Simulating {N_SAMPLES_TEST} samples for inference ---")
    test_df_full = simulate_data(N_SAMPLES_TEST, seed=101)
    # The phenotype column is dropped as it's what we want to predict
    test_df_full.drop(columns=["phenotype"]).to_csv(TEST_DATA_PATH, sep="\t", index=False)
    print(f"Saved test data to '{TEST_DATA_PATH}'\n")

    # 4. Run the 'infer' command
    print("--- Running 'infer' command ---")
    infer_command = [
        str(EXECUTABLE_PATH),
        "infer",
        "--model", MODEL_PATH.name,
        TEST_DATA_PATH.name,
    ]
    print(f"Executing: {' '.join(map(str, infer_command))}")
    try:
        # Stream gnomon output in real-time
        subprocess.run(
            infer_command,
            check=True,
            text=True,
            cwd=PROJECT_ROOT
        )
    except subprocess.CalledProcessError as e:
        print(f"\n--- Model Inference FAILED (Exit Code: {e.returncode}) ---")
        return
    print(f"\nInference complete. Predictions saved to '{PREDICTIONS_PATH}'\n")

    # 5. Load results and plot
    print("--- Loading results and plotting ---")
    predictions_df = pd.read_csv(PREDICTIONS_PATH, sep="\t")
    results_df = pd.concat([test_df_full.reset_index(drop=True), predictions_df], axis=1)

    # REVISED: Create a grid to plot the new, more complex probability surface
    score_grid, pc1_grid = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3.5, 3.5, 100))
    # The true, underlying log-odds surface (without the random noise component)
    # This must match the formula in `simulate_data` for an accurate comparison.
    true_logit_surface = (
        0.8 * np.cos(score_grid * 2)
        - 1.2 * np.tanh(pc1_grid)
        + 1.5 * np.sin(score_grid) * pc1_grid
        - 0.5 * (score_grid**2 + pc1_grid**2)
    )
    true_prob_surface = 1 / (1 + np.exp(-true_logit_surface))

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(score_grid, pc1_grid, true_prob_surface, levels=20, cmap="viridis", alpha=0.8)
    fig.colorbar(contour, ax=ax, label="True Probability of Phenotype=1")

    # Scatter plot of the model's predictions on the test set
    scatter = ax.scatter(
        results_df["score"],
        results_df["PC1"],
        c=results_df["prediction"],
        cmap="plasma",
        edgecolor="black",
        linewidth=0.5,
        s=50,
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
