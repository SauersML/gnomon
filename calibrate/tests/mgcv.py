import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

# --- 1. Parameters ---
N_SAMPLES = 2000
NOISE_STD_DEV = 0.5  # Controls how "fuzzy" the relationship is
OUTPUT_FILENAME = 'synthetic_classification_data.csv'
N_BINS = 20  # Number of bins for plotting

# --- 2. Data Generation ---

def generate_data(n_samples, noise_std, linear_mode=False):
    """
    Generates synthetic data for a binary classification problem.
    The script supports two modes controlled by the `linear_mode` flag.

    --- Non-Linear Mode (default) ---
    The probability of the outcome is a non-linear function of both variables.
    1.  Generate two independent variables, var1 (angle) and var2.
    2.  Create a 'logit' score: logit = sin(var1) + var2 + noise.
    3.  Map the logit to a probability via the sigmoid function.

    --- Linear Mode (--linear flag) ---
    The probability of the outcome is a linear function of only variable_one.
    1.  Generate var1, which is linearly related to the outcome's log-odds.
    2.  Generate var2 as PURE NOISE, with no relation to the outcome.
    3.  Create a 'logit' score: logit = var1 + noise.
    4.  Map the logit to a probability via the sigmoid function.
    """
    if linear_mode:
        print(f"--- Running in LINEAR mode ---")
        print(f"Generating {n_samples} samples...")

        # Variable 1: Linearly related to the logit. Uniformly distributed.
        var1 = np.random.uniform(-3, 3, n_samples)
        
        # Variable 2: PURE NOISE. Uniformly distributed, but not used in logit.
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        
        # Gaussian noise with mean 0
        noise = np.random.normal(0, noise_std, n_samples)
        
        # Calculate the underlying "logit" score. LINEAR relationship with var1.
        # var2 is intentionally omitted to make it pure noise.
        logit = var1 + noise
        
    else: # Original non-linear behavior
        print(f"--- Running in NON-LINEAR mode (default) ---")
        print(f"Generating {n_samples} samples...")

        # Variable 1: Uniformly distributed angle from 0 to 2*pi for a full sine cycle
        var1 = np.random.uniform(0, 2 * np.pi, n_samples)
        
        # Variable 2: Uniformly distributed from -1.5 to 1.5. Centered at 0.
        var2 = np.random.uniform(-1.5, 1.5, n_samples)
        
        # Gaussian noise with mean 0
        noise = np.random.normal(0, noise_std, n_samples)
        
        # Calculate the underlying "logit" score
        logit = np.sin(var1) + var2 + noise
    
    # Use the sigmoid function to convert the logit into a probability [0, 1]
    # This step is common to both modes.
    prob_of_one = 1 / (1 + np.exp(-logit))
    
    # Generate the final binary outcome (0 or 1) via a Bernoulli trial
    random_chance = np.random.uniform(0, 1, n_samples)
    outcome = (random_chance < prob_of_one).astype(int)
    
    # Create a pandas DataFrame
    df = pd.DataFrame({
        'variable_one': var1,
        'variable_two': var2,
        'outcome': outcome
    })
    
    return df

# --- 3. Plotting ---

def create_binned_plots(df, linear_mode=False):
    """
    Creates plots showing the binned probability of outcome=1 against each
    independent variable, adapted for the generation mode.
    """
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    
    if linear_mode:
        fig.suptitle('Binned Probability (Linear Logit Mode)', fontsize=16)

        # --- Subplot 1: variable 1 vs. P(1) ---
        df['v1_bin'] = pd.cut(df['variable_one'], bins=N_BINS)
        binned_prob_v1 = df.groupby('v1_bin', observed=True)['outcome'].mean()
        bin_centers_v1 = [b.mid for b in binned_prob_v1.index]

        axes[0].plot(bin_centers_v1, binned_prob_v1.values, 'o-', label='Binned Empirical P(1)')
        
        # Theoretical curve: P(1) = sigmoid(var1)
        x_theory_v1 = np.linspace(df['variable_one'].min(), df['variable_one'].max(), 200)
        y_theory_v1 = 1 / (1 + np.exp(-x_theory_v1))
        axes[0].plot(x_theory_v1, y_theory_v1, 'r--', label='Theoretical P(1) = sigmoid(var1)\n(ignoring noise)')

        axes[0].set_xlabel('variable 1')
        axes[0].set_ylabel('Proportion of "1"s in Bin')
        axes[0].set_title('variable 1 vs. P(1)')
        
    else: # Original non-linear plot
        fig.suptitle('Binned Probability (Non-Linear Mode)', fontsize=16)
        
        # --- Subplot 1: variable 1 vs. P(1) ---
        df['v1_bin'] = pd.cut(df['variable_one'], bins=N_BINS)
        binned_prob_v1 = df.groupby('v1_bin', observed=True)['outcome'].mean()
        bin_centers_v1 = [b.mid for b in binned_prob_v1.index]

        axes[0].plot(bin_centers_v1, binned_prob_v1.values, 'o-', label='Binned Empirical P(1)')
        
        # Theoretical curve: P(1) = sigmoid(sin(var1))
        x_theory_v1 = np.linspace(0, 2 * np.pi, 200)
        y_theory_v1 = 1 / (1 + np.exp(-np.sin(x_theory_v1)))
        axes[0].plot(x_theory_v1, y_theory_v1, 'r--', label='Theoretical P(1)\n(ignoring var2 & noise)')

        axes[0].set_xlabel('variable 1 (angle in radians)')
        axes[0].set_ylabel('Proportion of "1"s in Bin')
        axes[0].set_title('variable 1 vs. P(1)')

    # Common settings for subplot 1
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- Subplot 2: variable 2 vs. P(1) ---    
    df['v2_bin'] = pd.cut(df['variable_two'], bins=N_BINS)
    binned_prob_v2 = df.groupby('v2_bin', observed=True)['outcome'].mean()
    bin_centers_v2 = [b.mid for b in binned_prob_v2.index]
    axes[1].plot(bin_centers_v2, binned_prob_v2.values, 'o-', label='Binned Empirical P(1)')
    
    if linear_mode:
        # In linear mode, var2 is pure noise, so the theoretical probability is 0.5
        axes[1].axhline(0.5, color='r', linestyle='--', label='Theoretical P(1) = 0.5\n(No relationship)')
        axes[1].set_title('variable 2 vs. P(1) (Pure Noise)')
    else: # Original non-linear plot
        # Theoretical curve: P(1) = sigmoid(var2)
        x_theory_v2 = np.linspace(df['variable_two'].min(), df['variable_two'].max(), 200)
        y_theory_v2 = 1 / (1 + np.exp(-x_theory_v2))
        axes[1].plot(x_theory_v2, y_theory_v2, 'r--', label='Theoretical P(1)\n(ignoring sin(v1) & noise)')
        axes[1].set_title('variable 2 vs. P(1)')
        
    # Common settings for subplot 2
    axes[1].set_xlabel('variable 2')
    axes[1].set_ylabel('Proportion of "1"s in Bin')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.show()

# --- 4. Main Execution Block ---

def main():
    """
    Parses command-line arguments and runs the data generation and plotting script.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for binary classification.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    parser.add_argument(
        '--linear',
        action='store_true',
        help="""Run in 100%% linear mode:
  - 'variable_one' is linearly related to the log-odds of the outcome.
  - 'variable_two' is pure noise with no relationship to the outcome.
  - The sine wave is removed.
If this flag is not present, the script runs in the original non-linear mode."""
    )
    args = parser.parse_args()

    # --- Generate and Save Data ---
    final_data = generate_data(N_SAMPLES, NOISE_STD_DEV, linear_mode=args.linear)

    final_data.to_csv(OUTPUT_FILENAME, index=False)

    print(f"\nData successfully generated and saved to '{OUTPUT_FILENAME}'")
    print("\nData Head:")
    print(final_data.head())
    print("\nOutcome Distribution (should be ~50/50):")
    print(final_data['outcome'].value_counts(normalize=True))

    # --- Call the plotting function ---
    create_binned_plots(final_data, linear_mode=args.linear)

if __name__ == "__main__":
    main()
