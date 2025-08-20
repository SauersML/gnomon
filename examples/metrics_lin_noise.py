import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse
from pygam import LogisticGAM, s, te
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

# --- PyGam Patches for Compatibility ---
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(scipy.sparse.csc_matrix, 'A'):
    scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())
if not hasattr(scipy.sparse.csr_matrix, 'A'):
    scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())

def nagelkerkes_r2(y_true, y_prob):
    """Calculates Nagelkerke's R-squared for model performance."""
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

# --- 1. Style, Colors, and Data Generation ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.weight': 'bold', 'axes.labelweight': 'bold',
    'font.size': 20, 'axes.labelsize': 22, 'axes.titlesize': 26,
    'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 18,
    'figure.titlesize': 32
})
COLOR_GAM_FIT = '#333333'      # Dark Gray
COLOR_EMPIRICAL = '#0072B2'   # Vibrant Blue
COLOR_LINEAR_FAIL = '#D55E00' # Burnt Orange
np.random.seed(42)

# --- REVISED DATA GENERATION with STEEPER NOISE ---
print("--- Generating Simulated Data ---")
N = 50000
beta = 0.8
gamma = 0.4
sigma_g_sq = 1.0

# MODIFIED: Making the noise gradient steeper
# Less noise at low PC, more noise at high PC
c0_std, c1_std = 0.05, 5.0

PC = np.random.uniform(0, 1, N)
G = np.random.normal(0, np.sqrt(sigma_g_sq), N)
error_std = c0_std + c1_std * PC
P = G + np.random.normal(0, error_std, N)

linear_predictor = beta * G + gamma * PC
prob_Y = 1 / (1 + np.exp(-linear_predictor))
Y = np.random.binomial(1, prob_Y)

print(f"Generated {N} samples with a STEEP noise gradient.")
print(f"Overall prevalence of the outcome: {np.mean(Y):.2%}\n")


# --- 2. Data Splitting ---
(P_train, P_test,
 PC_train, PC_test,
 Y_train, Y_test) = train_test_split(P, PC, Y, test_size=0.2, random_state=42)
print(f"Data split into {len(Y_train)} training and {len(Y_test)} testing samples.\n")


# --- 3. Model Implementation and Metric Calculation ---
results = []

# --- MODEL 1: Basic Model (PGS Only) ---
print("--- Training Model 1: Basic PGS Model ---")
model1 = LogisticRegression(solver='liblinear').fit(P_train.reshape(-1, 1), Y_train)
probs1 = model1.predict_proba(P_test.reshape(-1, 1))[:, 1]
results.append({'Model': '1: Basic (PGS Only)',
                'AUC': roc_auc_score(Y_test, probs1),
                'Brier Score': brier_score_loss(Y_test, probs1),
                'Nagelkerke\'s R²': nagelkerkes_r2(Y_test, probs1)})
print("Model 1 complete.\n")

# --- MODEL 2: Regress Out PCs (Normalized PGS) ---
print("--- Training Model 2: Normalized PGS (Regress Out PCs) ---")
pc_regressor = LinearRegression().fit(PC_train.reshape(-1, 1), P_train)
P_adj_train = P_train - pc_regressor.predict(PC_train.reshape(-1, 1))
P_adj_test = P_test - pc_regressor.predict(PC_test.reshape(-1, 1))
model2 = LogisticRegression(solver='liblinear').fit(P_adj_train.reshape(-1, 1), Y_train)
probs2 = model2.predict_proba(P_adj_test.reshape(-1, 1))[:, 1]
results.append({'Model': '2: Normalized PGS',
                'AUC': roc_auc_score(Y_test, probs2),
                'Brier Score': brier_score_loss(Y_test, probs2),
                'Nagelkerke\'s R²': nagelkerkes_r2(Y_test, probs2)})
print("Model 2 complete.\n")

# --- MODEL 3: Our Proposed GAM ---
print("--- Training Model 3: Our GAM ---")
X_train_gam = np.c_[P_train, PC_train]
X_test_gam = np.c_[P_test, PC_test]
model3 = LogisticGAM(s(0, n_splines=15) + s(1, n_splines=15) + te(0, 1, n_splines=[15, 15]))
model3.gridsearch(X_train_gam, Y_train)
probs3 = model3.predict_proba(X_test_gam)
results.append({'Model': '3: Our GAM',
                'AUC': roc_auc_score(Y_test, probs3),
                'Brier Score': brier_score_loss(Y_test, probs3),
                'Nagelkerke\'s R²': nagelkerkes_r2(Y_test, probs3)})
print("Model 3 complete.\n")


# --- 4. Display Final Metric Results ---
results_df = pd.DataFrame(results)
print("--- Final Model Comparison ---")
print("AUC: Higher is better (better discrimination).")
print("Brier Score: Lower is better (better calibration).")
print("Nagelkerke's R²: Higher is better (better model fit).")
print("-" * 55)
print(results_df.to_string(index=False))
print("-" * 55)


# --- 5. Plot Generation ---
print("\n--- Generating Plots ---")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), sharex=True)
fig.suptitle('GAMs Correct for Ancestry-Dependent PGS Error', fontweight='bold', y=0.98)
fig.subplots_adjust(hspace=0.35)

# PLOT 1: The Problem
ax1.set_title('PGS Error Increases Systematically with PC Value', fontweight='bold')
prediction_error = P_test - Y_test
plot_idx = np.random.choice(len(P_test), 1000, replace=False)
ax1.scatter(PC_test[plot_idx], prediction_error[plot_idx],
            c=PC_test[plot_idx], cmap='plasma', alpha=0.7, s=50)
ax1.set_ylabel('Prediction Error (P - Y)')
ax1.grid(True, which='both', linestyle=':', linewidth=1.0)

# PLOT 2: The Solution
ax2.set_title('GAM Recovers the Optimal PGS Calibration Curve', fontweight='bold')

# A) GAM-derived Coefficient
pc_smooth = np.linspace(PC.min(), PC.max(), 200)
grid_p0 = np.c_[np.zeros_like(pc_smooth), pc_smooth]
grid_p1 = np.c_[np.ones_like(pc_smooth), pc_smooth]
epsilon = 1e-10
probs_p0 = model3.predict_mu(grid_p0)
probs_p1 = model3.predict_mu(grid_p1)
pred_p0_link = np.log((probs_p0 + epsilon) / (1 - probs_p0 + epsilon))
pred_p1_link = np.log((probs_p1 + epsilon) / (1 - probs_p1 + epsilon))
gam_derived_coefficient = pred_p1_link - pred_p0_link
ax2.plot(pc_smooth, gam_derived_coefficient, color=COLOR_GAM_FIT, linewidth=8,
         label='GAM-derived Coefficient', zorder=10)

# B) Misspecified Linear Interaction Model
X_linear_train = np.c_[P_train, PC_train, P_train * PC_train]
lin_model = LogisticRegression(solver='liblinear').fit(X_linear_train, Y_train)
coef_p, _, coef_int = lin_model.coef_[0]
linear_fit_coefficient = coef_p + coef_int * pc_smooth
ax2.plot(pc_smooth, linear_fit_coefficient, color=COLOR_LINEAR_FAIL, linestyle='--', linewidth=6,
         label='Misspecified Linear Fit', zorder=8)

# C) Empirical Binned Coefficient
n_bins = 10
bin_edges = np.linspace(PC_train.min(), PC_train.max(), n_bins + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
empirical_coeffs = []
for i in range(n_bins):
    mask = (PC_train >= bin_edges[i]) & (PC_train < bin_edges[i+1])
    if np.sum(mask) > 50:
        binned_model = LogisticRegression(solver='liblinear').fit(P_train[mask].reshape(-1, 1), Y_train[mask])
        empirical_coeffs.append(binned_model.coef_[0][0])
    else:
        empirical_coeffs.append(np.nan)
ax2.plot(bin_centers, empirical_coeffs, 'o', color=COLOR_EMPIRICAL, markersize=18,
         markeredgecolor='white', markeredgewidth=2.0, label='Empirical Optimum (Binned)', zorder=12)

ax2.set_xlabel('Principal Component Value')
ax2.set_ylabel('Effective PGS Coefficient\n(on Log-Odds Scale)')
ax2.legend(loc='upper right')
ax2.grid(True, which='both', linestyle=':', linewidth=1.0)
ax2.set_ylim(0, max(np.nanmax(empirical_coeffs), np.nanmax(gam_derived_coefficient)) * 1.2)

# --- 6. Final Touches ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("gam_metrics_and_plot_figure.png", dpi=300, bbox_inches='tight')
print("Plot saved to gam_metrics_and_plot_figure.png")
plt.show()
