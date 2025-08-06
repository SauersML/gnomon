import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- Constants for Data Generation ---
N_SAMPLES_TRAIN = 2000
N_SAMPLES_TEST = 20000
NOISE_STD_DEV = 0.5
TRAIN_OUTPUT_FILENAME = 'synthetic_classification_data.csv'
TEST_OUTPUT_FILENAME = 'test_data.csv'
N_BINS = 20


def generate_data(n_samples, noise_std, linear_mode=False, noise_mode=False):
    # Generate common components first
    noise = np.random.normal(0, noise_std, n_samples)

    # Note: The 'print' statements inside this function will now run for both
    # training and test set generation, which is expected.
    if noise_mode:
        print(f"--- Running in PURE NOISE mode ---")
        print(f"Generating {n_samples} samples with ZERO signal...")
        # Use standard ranges for variables, but they won't affect the outcome
        var1 = np.random.uniform(-3, 3, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        # The logit is ONLY noise, severing any link to the variables.
        logit = noise
    elif linear_mode:
        print(f"--- Running in LINEAR mode ---")
        print(f"Generating {n_samples} samples...")
        var1 = np.random.uniform(-3, 3, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        logit = var1 + noise  # var2 is intentionally omitted
    else: # Default non-linear mode
        print(f"--- Running in NON-LINEAR mode (default) ---")
        print(f"Generating {n_samples} samples...")
        var1 = np.random.uniform(0, 2 * np.pi, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        logit = np.sin(var1) + var2 + noise

    prob_of_one = 1 / (1 + np.exp(-logit))
    random_chance = np.random.uniform(0, 1, n_samples)
    outcome = (random_chance < prob_of_one).astype(int)

    df = pd.DataFrame({
        'variable_one': var1,
        'variable_two': var2,
        'outcome': outcome
    })
    return df

def create_binned_plots(df, linear_mode=False, noise_mode=False):
    """
    Creates plots showing the binned probability of outcome=1 against each
    independent variable, adapted for the generation mode.
    """
    print("\nGenerating plots (based on the large test set)...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Determine the main title based on the mode
    if noise_mode:
        fig.suptitle('Binned Probability (Pure Noise Mode)', fontsize=16)
    elif linear_mode:
        fig.suptitle('Binned Probability (Linear Logit Mode)', fontsize=16)
    else:
        fig.suptitle('Binned Probability (Non-Linear Mode)', fontsize=16)


    # --- Subplot 1: variable 1 vs. P(1) ---
    df['v1_bin'] = pd.cut(df['variable_one'], bins=N_BINS)
    binned_prob_v1 = df.groupby('v1_bin', observed=True)['outcome'].mean()
    bin_centers_v1 = [b.mid for b in binned_prob_v1.index]
    axes[0].plot(bin_centers_v1, binned_prob_v1.values, 'o-', label='Binned Empirical P(1)')

    if noise_mode:
        axes[0].axhline(0.5, color='r', linestyle='--', label='Theoretical P(1) = 0.5\n(No relationship)')
        axes[0].set_title('variable 1 vs. P(1) (Pure Noise)')
        axes[0].set_xlabel('variable 1')
    elif linear_mode:
        x_theory = np.linspace(df['variable_one'].min(), df['variable_one'].max(), 200)
        y_theory = 1 / (1 + np.exp(-x_theory))
        axes[0].plot(x_theory, y_theory, 'r--', label='Theoretical P(1) = sigmoid(var1)')
        axes[0].set_title('variable 1 vs. P(1)')
        axes[0].set_xlabel('variable 1')
    else: # Non-linear mode
        x_theory = np.linspace(0, 2 * np.pi, 200)
        y_theory = 1 / (1 + np.exp(-np.sin(x_theory)))
        axes[0].plot(x_theory, y_theory, 'r--', label='Theoretical P(1) = sigmoid(sin(var1))')
        axes[0].set_title('variable 1 vs. P(1)')
        axes[0].set_xlabel('variable 1 (angle in radians)')

    axes[0].set_ylabel('Proportion of "1"s in Bin')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- Subplot 2: variable 2 vs. P(1) ---
    df['v2_bin'] = pd.cut(df['variable_two'], bins=N_BINS)
    binned_prob_v2 = df.groupby('v2_bin', observed=True)['outcome'].mean()
    bin_centers_v2 = [b.mid for b in binned_prob_v2.index]
    axes[1].plot(bin_centers_v2, binned_prob_v2.values, 'o-', label='Binned Empirical P(1)')

    # In both linear and pure noise modes, var2 has no relationship to the outcome.
    if linear_mode or noise_mode:
        axes[1].axhline(0.5, color='r', linestyle='--', label='Theoretical P(1) = 0.5\n(No relationship)')
        axes[1].set_title('variable 2 vs. P(1) (Pure Noise)')
    else: # Non-linear mode
        x_theory = np.linspace(df['variable_two'].min(), df['variable_two'].max(), 200)
        y_theory = 1 / (1 + np.exp(-x_theory))
        axes[1].plot(x_theory, y_theory, 'r--', label='Theoretical P(1) = sigmoid(var2)')
        axes[1].set_title('variable 2 vs. P(1)')

    axes[1].set_xlabel('variable 2')
    axes[1].set_ylabel('Proportion of "1"s in Bin')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.show()


def main():
    """
    Parses command-line arguments and runs the data generation and plotting.
    """
    parser = argparse.ArgumentParser(
        description="Generate separate TRAINING and TESTING synthetic datasets for binary classification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Group for mutually exclusive modes. Default is non-linear.
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '--linear',
        action='store_true',
        help="""Run in 100%% linear mode:
  - 'variable_one' is linearly related to the log-odds of the outcome.
  - 'variable_two' is pure noise with no relationship to the outcome."""
    )
    mode_group.add_argument(
        '--noise',
        action='store_true',
        help="""Run in 100%% pure noise mode:
  - 'variable_one' has NO relationship to the outcome.
  - 'variable_two' has NO relationship to the outcome.
  - Both variables are pure noise (zero signal)."""
    )
    args = parser.parse_args()

    # --- 1. Generate TRAINING Data ---
    print("\n" + "="*50)
    print("### GENERATING TRAINING DATA ###")
    print("="*50)
    training_data = generate_data(
        N_SAMPLES_TRAIN,
        NOISE_STD_DEV,
        linear_mode=args.linear,
        noise_mode=args.noise
    )
    training_data.to_csv(TRAIN_OUTPUT_FILENAME, index=False)
    print(f"\nTraining data ({N_SAMPLES_TRAIN} rows) saved to '{TRAIN_OUTPUT_FILENAME}'")
    print("\nTraining Data Head:")
    print(training_data.head())

    # --- 2. Generate TESTING Data ---
    print("\n" + "="*50)
    print("### GENERATING TESTING DATA ###")
    print("="*50)
    test_data = generate_data(
        N_SAMPLES_TEST,
        NOISE_STD_DEV,
        linear_mode=args.linear,
        noise_mode=args.noise
    )
    test_data.to_csv(TEST_OUTPUT_FILENAME, index=False)
    print(f"\nTest data ({N_SAMPLES_TEST} rows) saved to '{TEST_OUTPUT_FILENAME}'")
    print("\nTest Data Outcome Distribution (should be ~50/50):")
    print(test_data['outcome'].value_counts(normalize=True))

    # --- 3. Create Plots from the large, clean test set ---
    create_binned_plots(
        test_data.copy(), # Pass a copy to avoid SettingWithCopyWarning
        linear_mode=args.linear,
        noise_mode=args.noise
    )


if __name__ == "__main__":
    main()
