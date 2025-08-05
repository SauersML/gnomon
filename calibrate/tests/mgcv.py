import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- 1. Parameters (Global Constants) ---
# These are safe to define globally as they are just configuration values.
N_SAMPLES = 2000
NOISE_STD_DEV = 0.5
OUTPUT_FILENAME = 'synthetic_classification_data.csv'
N_BINS = 20

# --- 2. Function Definitions ---
# These functions are the script's "tools." They are defined but not run here.

def generate_data(n_samples, noise_std, linear_mode=False):
    """
    Generates synthetic data for a binary classification problem.
    The script supports two modes controlled by the `linear_mode` flag.

    --- Non-Linear Mode (default) ---
    The probability of the outcome is a non-linear function of both variables.
    - logit = sin(var1) + var2 + noise

    --- Linear Mode (--linear flag) ---
    The probability of the outcome is a linear function of only variable_one.
    - logit = var1 + noise
    - var2 is generated but is pure noise with no relation to the outcome.
    """
    if linear_mode:
        print(f"--- Running in LINEAR mode ---")
        print(f"Generating {n_samples} samples...")
        var1 = np.random.uniform(-3, 3, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        noise = np.random.normal(0, noise_std, n_samples)
        logit = var1 + noise  # var2 is intentionally omitted
    else:
        print(f"--- Running in NON-LINEAR mode (default) ---")
        print(f"Generating {n_samples} samples...")
        var1 = np.random.uniform(0, 2 * np.pi, n_samples)
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        noise = np.random.normal(0, noise_std, n_samples)
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

def create_binned_plots(df, linear_mode=False):
    """
    Creates plots showing the binned probability of outcome=1 against each
    independent variable, adapted for the generation mode.
    """
    print("\nGenerating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # --- Subplot 1: variable 1 vs. P(1) ---
    df['v1_bin'] = pd.cut(df['variable_one'], bins=N_BINS)
    binned_prob_v1 = df.groupby('v1_bin', observed=True)['outcome'].mean()
    bin_centers_v1 = [b.mid for b in binned_prob_v1.index]
    axes[0].plot(bin_centers_v1, binned_prob_v1.values, 'o-', label='Binned Empirical P(1)')

    if linear_mode:
        fig.suptitle('Binned Probability (Linear Logit Mode)', fontsize=16)
        x_theory = np.linspace(df['variable_one'].min(), df['variable_one'].max(), 200)
        y_theory = 1 / (1 + np.exp(-x_theory))
        axes[0].plot(x_theory, y_theory, 'r--', label='Theoretical P(1) = sigmoid(var1)')
        axes[0].set_xlabel('variable 1')
        axes[0].set_title('variable 1 vs. P(1)')
    else:
        fig.suptitle('Binned Probability (Non-Linear Mode)', fontsize=16)
        x_theory = np.linspace(0, 2 * np.pi, 200)
        y_theory = 1 / (1 + np.exp(-np.sin(x_theory)))
        axes[0].plot(x_theory, y_theory, 'r--', label='Theoretical P(1) = sigmoid(sin(var1))')
        axes[0].set_xlabel('variable 1 (angle in radians)')
        axes[0].set_title('variable 1 vs. P(1)')

    axes[0].set_ylabel('Proportion of "1"s in Bin')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- Subplot 2: variable 2 vs. P(1) ---
    df['v2_bin'] = pd.cut(df['variable_two'], bins=N_BINS)
    binned_prob_v2 = df.groupby('v2_bin', observed=True)['outcome'].mean()
    bin_centers_v2 = [b.mid for b in binned_prob_v2.index]
    axes[1].plot(bin_centers_v2, binned_prob_v2.values, 'o-', label='Binned Empirical P(1)')

    if linear_mode:
        axes[1].axhline(0.5, color='r', linestyle='--', label='Theoretical P(1) = 0.5\n(No relationship)')
        axes[1].set_title('variable 2 vs. P(1) (Pure Noise)')
    else:
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

# --- 3. Main Execution Block ---
# This is where the script's actions are orchestrated.

def main():
    """
    Parses command-line arguments and runs the data generation and plotting.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for binary classification.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--linear',
        action='store_true',
        help="""Run in 100%% linear mode:
  - 'variable_one' is linearly related to the log-odds of the outcome.
  - 'variable_two' is pure noise with no relationship to the outcome.
  - The sine wave is removed."""
    )
    args = parser.parse_args()

    # --- Run the logic ---
    # The 'linear_mode' flag is now correctly passed to the functions.
    final_data = generate_data(N_SAMPLES, NOISE_STD_DEV, linear_mode=args.linear)

    final_data.to_csv(OUTPUT_FILENAME, index=False)

    print(f"\nData successfully generated and saved to '{OUTPUT_FILENAME}'")
    print("\nData Head:")
    print(final_data.head())
    print("\nOutcome Distribution (should be ~50/50):")
    print(final_data['outcome'].value_counts(normalize=True))

    create_binned_plots(final_data, linear_mode=args.linear)

# --- 4. Script Entry Point ---
# This special block ensures that the main() function is called only when
# the script is executed directly (e.g., `python your_script.py`).
# It prevents the code from running if it's imported as a module elsewhere.
if __name__ == "__main__":
    main()
