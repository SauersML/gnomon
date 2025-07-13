import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- Configuration ---
EXECUTABLE_NAME = "gnomon-calibrate"
N_SAMPLES_TRAIN = 2000
N_SAMPLES_TEST = 500
NUM_PCS = 1 # We'll simulate 1 PC for easy visualization

# --- Helper Functions ---

def build_rust_project():
    """Compiles the Rust project in release mode using Cargo."""
    print("--- Compiling Rust Project (cargo build --release) ---")
    
    # Check if Rust/Cargo is installed
    try:
        subprocess.run(["cargo", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\nERROR: Rust/Cargo is not installed or not in your PATH.")
        print("Please install Rust from https://rustup.rs/ to continue.")
        sys.exit(1)

    # Compile the project
    process = subprocess.run(
        ["cargo", "build", "--release"],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        print("\n--- Rust Compilation FAILED ---")
        print("Stdout:\n", process.stdout)
        print("Stderr:\n", process.stderr)
        sys.exit(1)
        
    print("Compilation successful.")
    executable_path = Path(f"./target/release/{EXECUTABLE_NAME}")
    if not executable_path.exists():
        print(f"\nERROR: Could not find compiled executable at {executable_path}")
        sys.exit(1)
        
    return executable_path

def simulate_data(n_samples: int, seed: int):
    """
    Simulates a dataset with a non-linear relationship between predictors and a binary outcome.
    
    This function defines a "ground truth" that the GAM will try to learn.
    """
    np.random.seed(seed)
    
    # 1. Generate predictors: 'score' and 'PC1'
    # Use different distributions to make it more realistic
    score = np.random.uniform(-2.5, 2.5, n_samples)
    pc1 = np.random.normal(0, 1.0, n_samples)
    
    # 2. Define the true, non-linear log-odds function
    # This includes non-linear main effects and an interaction term
    # It's the "secret" function we want the GAM to discover.
    logit = (
        -1.0                                # Intercept (low base probability)
        + 1.5 * np.sin(score * 1.2)         # Non-linear effect of the score
        + 0.8 * pc1**2                      # Quadratic effect of PC1
        - 1.0 * (score * pc1)               # Interaction effect
    )
    
    # 3. Convert log-odds to probabilities
    probability = 1 / (1 + np.exp(-logit))
    
    # 4. Generate the binary 'phenotype' based on the probability
    phenotype = np.random.binomial(1, probability, n_samples)
    
    # 5. Assemble into a pandas DataFrame
    df = pd.DataFrame({
        "phenotype": phenotype,
        "score": score,
        "PC1": pc1
    })
    
    return df

# --- Main Script Logic ---

def main():
    """Main function to run the end-to-end simulation and plotting."""
    
    # 1. Compile the Rust executable
    executable_path = build_rust_project()
    print(f"Using executable: {executable_path}\n")

    # 2. Generate and save training data
    print(f"--- Simulating {N_SAMPLES_TRAIN} samples for training ---")
    train_df = simulate_data(N_SAMPLES_TRAIN, seed=42)
    train_data_path = Path("training_data.tsv")
    train_df.to_csv(train_data_path, sep="\t", index=False)
    print(f"Saved training data to '{train_data_path}'\n")

    # 3. Run the 'train' command
    print("--- Running 'train' command ---")
    model_path = Path("model.toml")
    train_command = [
        str(executable_path),
        "train",
        "--training-data", str(train_data_path),
        "--output-path", str(model_path),
        "--num-pcs", str(NUM_PCS),
        "--pgs-knots", "12",  # More knots to capture the sine wave
        "--pc-knots", "8",   # Fewer knots for the quadratic PC effect
        "--pgs-degree", "3",
        "--pc-degree", "3",
    ]
    
    print(f"Executing: {' '.join(train_command)}")
    train_process = subprocess.run(train_command, capture_output=True, text=True)
    
    if train_process.returncode != 0:
        print("\n--- Model Training FAILED ---")
        print("Stdout:\n", train_process.stdout)
        print("Stderr:\n", train_process.stderr)
        return

    print(train_process.stdout)
    print(f"Model training complete. Model saved to '{model_path}'\n")

    # 4. Generate and save test data for inference
    print(f"--- Simulating {N_SAMPLES_TEST} samples for inference ---")
    test_df_full = simulate_data(N_SAMPLES_TEST, seed=101)
    
    # For inference, the input file only needs the predictors ('score', 'PC1', etc.)
    test_df_inference = test_df_full.drop(columns=["phenotype"])
    test_data_path = Path("test_data.tsv")
    test_df_inference.to_csv(test_data_path, sep="\t", index=False)
    print(f"Saved test data to '{test_data_path}'\n")
    
    # 5. Run the 'infer' command
    print("--- Running 'infer' command ---")
    predictions_path = Path("predictions.tsv")
    infer_command = [
        str(executable_path),
        "infer",
        "--test-data", str(test_data_path),
        "--model", str(model_path),
        "--output-path", str(predictions_path),
    ]

    print(f"Executing: {' '.join(infer_command)}")
    infer_process = subprocess.run(infer_command, capture_output=True, text=True)

    if infer_process.returncode != 0:
        print("\n--- Model Inference FAILED ---")
        print("Stdout:\n", infer_process.stdout)
        print("Stderr:\n", infer_process.stderr)
        return

    print(infer_process.stdout)
    print(f"Inference complete. Predictions saved to '{predictions_path}'\n")

    # 6. Load results and plot
    print("--- Loading results and plotting ---")
    predictions_df = pd.read_csv(predictions_path, sep="\t")
    
    # Combine test data with predictions for plotting
    results_df = pd.concat([test_df_full.reset_index(drop=True), predictions_df], axis=1)

    # Create a grid for plotting the true and predicted surfaces
    score_grid, pc1_grid = np.meshgrid(
        np.linspace(-2.5, 2.5, 100),
        np.linspace(-3, 3, 100)
    )

    # Calculate the "ground truth" probability surface
    true_logit_surface = (
        -1.0
        + 1.5 * np.sin(score_grid * 1.2)
        + 0.8 * pc1_grid**2
        - 1.0 * (score_grid * pc1_grid)
    )
    true_prob_surface = 1 / (1 + np.exp(-true_logit_surface))

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the true probability surface as a background contour map
    contour = ax.contourf(score_grid, pc1_grid, true_prob_surface, levels=20, cmap="viridis", alpha=0.8)
    fig.colorbar(contour, ax=ax, label="True Probability of Phenotype=1")

    # Overlay the model's predictions as a scatter plot
    # The color of each point will represent the predicted probability
    scatter = ax.scatter(
        results_df["score"],
        results_df["PC1"],
        c=results_df["prediction"],
        cmap="plasma",
        edgecolor="black",
        linewidth=0.5,
        s=50,
        label="Model Predictions"
    )
    fig.colorbar(scatter, ax=ax, label="Model Predicted Probability")
    
    ax.set_title("GAM Model Predictions vs. Ground Truth", fontsize=16)
    ax.set_xlabel("Polygenic Score (score)", fontsize=12)
    ax.set_ylabel("Principal Component 1 (PC1)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    print("Plot generated. Close the plot window to exit.")
    plt.show()

if __name__ == "__main__":
    main()
