import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Parameters ---
N_SAMPLES = 2000
NOISE_STD_DEV = 0.5  # Controls how "fuzzy" the relationship is
OUTPUT_FILENAME = 'synthetic_classification_data.csv'
N_BINS = 20  # Number of bins for plotting

# --- 2. Data Generation ---

def generate_data(n_samples, noise_std):
    """
    Generates synthetic data where the probability of a binary outcome
    is a function of two variables and noise.

    The key steps are:
    1.  Generate two independent variables, var1 and var2.
    2.  Create a 'logit' score: logit = sin(var1) + var2 + noise. This value
        is not a probability; it's an unbounded score.
    3.  Use the sigmoid (logistic) function to map the logit to a probability
        between 0 and 1. P(1) = 1 / (1 + exp(-logit)).
    4.  Since the mean of sin(var1), var2, and noise are all 0, the mean
        logit is 0. sigmoid(0) = 0.5, ensuring the average probability is 0.5,
        which leads to a balanced number of 0s and 1s.
    5.  Generate the final binary outcome based on this probability.
    """
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

# --- 3. Generate and Save Data ---

# Generate the data
final_data = generate_data(N_SAMPLES, NOISE_STD_DEV)

# Save the required columns to a CSV file
final_data.to_csv(OUTPUT_FILENAME, index=False)

print(f"\nData successfully generated and saved to '{OUTPUT_FILENAME}'")
print("\nData Head:")
print(final_data.head())
print("\nOutcome Distribution (should be ~50/50):")
print(final_data['outcome'].value_counts(normalize=True))

# --- 4. Plotting ---

def create_binned_plots(df):
    """
    Creates a plot with two subplots showing the binned probability of outcome=1
    against each of the independent variables.
    """
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle('Binned Probability of Outcome = 1', fontsize=16)

    # --- Subplot 1: variable 1 vs. P(1) ---
    # To see the sinusoidal relationship, we plot against the original variable 1
    # (the angle), not against sin(variable 1).
    
    # Bin the data and calculate the proportion of '1's in each bin
    df['v1_bin'] = pd.cut(df['variable_one'], bins=N_BINS)
    # Use observed=True to silence the FutureWarning and improve performance
    binned_prob_v1 = df.groupby('v1_bin', observed=True)['outcome'].mean()
    
    # Get the midpoint of each bin for the x-axis
    bin_centers_v1 = [b.mid for b in binned_prob_v1.index]

    # Plot the empirical binned data
    axes[0].plot(bin_centers_v1, binned_prob_v1.values, 'o-', label='Binned Empirical P(1)')
    
    # Plot the theoretical curve (P(1) = sigmoid(sin(var1)))
    x_theory_v1 = np.linspace(0, 2 * np.pi, 200)
    y_theory_v1 = 1 / (1 + np.exp(-np.sin(x_theory_v1)))
    axes[0].plot(x_theory_v1, y_theory_v1, 'r--', label='Theoretical P(1)\n(ignoring var2 & noise)')

    axes[0].set_xlabel('variable 1 (angle in radians)')
    axes[0].set_ylabel('Proportion of "1"s in Bin')
    axes[0].set_title('variable 1 vs. P(1)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    # --- Subplot 2: variable 2 vs. P(1) ---    
    # Bin the data and calculate the proportion of '1's
    df['v2_bin'] = pd.cut(df['variable_two'], bins=N_BINS)
    binned_prob_v2 = df.groupby('v2_bin', observed=True)['outcome'].mean()
    bin_centers_v2 = [b.mid for b in binned_prob_v2.index]

    # Plot the empirical binned data
    axes[1].plot(bin_centers_v2, binned_prob_v2.values, 'o-', label='Binned Empirical P(1)')
    
    # Plot the theoretical curve (P(1) = sigmoid(var2))
    x_theory_v2 = np.linspace(df['variable_two'].min(), df['variable_two'].max(), 200)
    y_theory_v2 = 1 / (1 + np.exp(-x_theory_v2))
    axes[1].plot(x_theory_v2, y_theory_v2, 'r--', label='Theoretical P(1)\n(ignoring sin(v1) & noise)')

    axes[1].set_xlabel('variable 2')
    axes[1].set_ylabel('Proportion of "1"s in Bin')
    axes[1].set_title('variable 2 vs. P(1)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    plt.show()

# Call the plotting function
create_binned_plots(final_data)
