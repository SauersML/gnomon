"""
Example of how to use the enhanced GAM method for PGS calibration.

This demonstrates:
1. Fitting the decomposed GAM model (main effects + interactions)
2. Inspecting effective degrees of freedom (EDF)
3. Getting model summary
4. Making predictions
"""
import numpy as np
import pandas as pd
from methods import GAMMethod

# Example: simulate some data
np.random.seed(42)
n = 500

# Simulate PGS and PCs
P = np.random.randn(n)
PC1 = np.random.randn(n)
PC2 = np.random.randn(n)
PC = np.column_stack([PC1, PC2])

# Simulate outcome with non-linear P×PC interaction
# True model: logit(p) = 0.5*P + 0.3*PC1 + 0.2*P*PC1^2
logit_p = 0.5 * P + 0.3 * PC1 + 0.2 * P * (PC1**2)
p = 1 / (1 + np.exp(-logit_p))
y = np.random.binomial(1, p)

print(f"Simulated {n} observations")
print(f"Prevalence: {y.mean():.3f}")
print()

# Fit GAM
print("=" * 60)
print("Fitting GAM with decomposed structure:")
print("  y ~ s(P) + s(PC1) + s(PC2) + ti(P,PC1) + ti(P,PC2)")
print("=" * 60)

gam = GAMMethod(
    n_pcs=2,
    k_pgs=10,
    k_pc=10,
    k_interaction=5,
    method='REML',
    use_ti=True,
)

gam.fit(P, PC, y)
print("\n✅ Model fitted successfully")

# Get effective degrees of freedom
print("\n" + "=" * 60)
print("Effective Degrees of Freedom (EDF):")
print("=" * 60)
edf = gam.get_edf()
for term, edf_val in edf.items():
    interpretation = "linear" if edf_val < 2 else "non-linear"
    print(f"  {term:20s}: {edf_val:6.2f}  ({interpretation})")

print("\nInterpretation:")
print("  - EDF ≈ 1: Linear relationship")
print("  - EDF > 2: Non-linear relationship detected")
print("  - Higher EDF: More flexible/wiggly smooth")

# Make predictions
print("\n" + "=" * 60)
print("Making predictions on new data:")
print("=" * 60)

# New data points
P_new = np.array([0.0, 1.0, -1.0])
PC_new = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])

probs = gam.predict_proba(P_new, PC_new)
print("\nPredictions:")
for i, (p_val, pc_vals, prob) in enumerate(zip(P_new, PC_new, probs)):
    print(f"  P={p_val:5.1f}, PC1={pc_vals[0]:5.1f}, PC2={pc_vals[1]:5.1f} → Pr(Y=1)={prob:.3f}")

# Get model summary
print("\n" + "=" * 60)
print("Full R GAM Summary:")
print("=" * 60)
print(gam.get_summary())
