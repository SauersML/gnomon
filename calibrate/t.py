# SCRIPT PURPOSE:
# To perform a deep, "glass-box" validation of the Rust/gnomon GAM implementation against
# R/mgcv. This is achieved by reconstructing the internal mathematical components (constrained
# basis functions, weighted bases, and final smooth curves) from the saved model artifacts
# of both systems and comparing them visually in a single comprehensive plot.
# It is intentionally extremely verbose to show every step of the data transformation.

import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Library Import Checks ---
try:
    import tomli
except ImportError:
    print("--- FATAL ERROR: 'tomli' package not found. Please install it: pip install tomli ---")
    sys.exit(1)
try:
    from scipy.interpolate import BSpline
except ImportError:
    print("--- FATAL ERROR: 'scipy' package not found. Please install it: pip install scipy ---")
    sys.exit(1)

# --- 1. Define Paths and Parameters ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
R_MODEL_PATH = SCRIPT_DIR / 'gam_model_fit.rds'
RUST_MODEL_CONFIG_PATH = PROJECT_ROOT / 'model.toml'
N_POINTS_PLOT = 400

def print_array_summary(name, arr):
    """Helper to print detailed diagnostics for a numpy array."""
    if arr.ndim == 1:
        print(f"  [DIAGNOSTIC] {name} | Shape: {arr.shape} | Min: {np.min(arr):.4f} | Max: {np.max(arr):.4f} | Mean: {np.mean(arr):.4f} | Std: {np.std(arr):.4f}")
        print(f"    -> First 5 elements: {arr[:5]}")
    else:
        print(f"  [DIAGNOSTIC] {name} | Shape: {arr.shape} | Min: {np.min(arr):.4f} | Max: {np.max(arr):.4f} | Mean: {np.mean(arr):.4f}")
        col_stds = np.std(arr, axis=0)
        print(f"    -> Stds of first 5 columns: {col_stds[:5]}")
        print(f"    -> First 2x5 slice:\n{arr[:2, :5]}")


def get_mgcv_basis_data():
    """
    Runs a robust R script to extract the final, constrained basis functions
    and coefficients for the main smooth term of 'variable_one'.
    """
    print("\n" + "="*80)
    print("--- STAGE 1: Extracting CONSTRAINED basis from R/mgcv model ---")
    print("="*80)
    
    x_axis_file, basis_file, coeffs_file = [SCRIPT_DIR / f for f in ["t_x.csv", "t_b.csv", "t_c.csv"]]
    
    r_script = f"""
    suppressPackageStartupMessages(library(mgcv))
    model <- readRDS('{R_MODEL_PATH}')
    var_range <- range(model$model$variable_one)
    x_seq <- seq(var_range[1], var_range[2], length.out = {N_POINTS_PLOT})
    newdata <- data.frame(variable_one = x_seq, variable_two = 0)
    lp_matrix <- predict(model, newdata = newdata, type = "lpmatrix")
    smooth_info <- model$smooth[[1]]
    start_index <- smooth_info$first.para
    end_index <- smooth_info$last.para
    basis_coeffs <- coef(model)[start_index:end_index]
    constrained_basis_functions <- lp_matrix[, start_index:end_index]
    write.csv(data.frame(x=x_seq), '{x_axis_file.name}', row.names=FALSE)
    write.csv(constrained_basis_functions, '{basis_file.name}', row.names=FALSE)
    write.csv(data.frame(coeffs=basis_coeffs), '{coeffs_file.name}', row.names=FALSE)
    cat("R: Extracted main effect basis for '", smooth_info$label, "'.\\n", sep="")
    """
    
    try:
        result = subprocess.run(["Rscript", "-e", r_script], check=True, text=True, cwd=SCRIPT_DIR, capture_output=True)
        print(f"  [INFO] R stdout: {result.stdout.strip()}")
        if result.stderr: print(f"  [INFO] R stderr: {result.stderr.strip()}")

        x_axis = pd.read_csv(x_axis_file)['x'].values
        basis_matrix = pd.read_csv(basis_file).values
        coeffs = pd.read_csv(coeffs_file)['coeffs'].values
        
        print(f"  [PRINT] mgcv: Loaded x-axis vector.")
        print_array_summary("mgcv x_axis", x_axis)
        print(f"  [PRINT] mgcv: Loaded constrained basis matrix.")
        print_array_summary("mgcv basis_matrix", basis_matrix)
        print(f"  [PRINT] mgcv: Loaded coefficients.")
        print_array_summary("mgcv coeffs", coeffs)
        
        return {"x_axis": x_axis, "basis_matrix": basis_matrix, "coeffs": coeffs}
    except subprocess.CalledProcessError as e:
        print(f"\n--- FATAL ERROR: R script execution failed. ---")
        print(f"R stdout:\n{e.stdout}\n R stderr:\n{e.stderr}")
        sys.exit(1)
    finally:
        for f in [x_axis_file, basis_file, coeffs_file]:
            if f.exists(): f.unlink()

def get_gnomon_basis_data():
    """
    Correctly reconstructs the gnomon constrained basis using BSpline.design_matrix
    and applying constraints ONLY to the non-constant raw basis functions.
    """
    print("\n" + "="*80)
    print("--- STAGE 2: Reconstructing CONSTRAINED basis from Rust/gnomon model ---")
    print("="*80)
    
    with open(RUST_MODEL_CONFIG_PATH, "rb") as f:
        toml_data = tomli.load(f)

    # 1. Extract B-spline recipe and coefficients for the 'pgs' MAIN EFFECT
    knots = np.array(toml_data['config']['knot_vectors']['pgs']['data'])
    degree = toml_data['config']['pgs_basis_config']['degree']
    coeffs = np.array(toml_data['coefficients']['main_effects']['pgs'])
    print(f"  [PRINT] gnomon: Loaded `pgs` knot vector with {len(knots)} knots.")
    print_array_summary("Knot Vector", knots)
    print(f"  [PRINT] gnomon: Loaded spline degree {degree}.")
    print(f"  [PRINT] gnomon: Loaded {len(coeffs)} coefficients for `pgs` main effect.")
    print_array_summary("Coefficients", coeffs)
    
    # 2. Derive number of raw basis functions from the knot vector (num_knots = num_basis + degree + 1)
    num_raw_bases = len(knots) - degree - 1
    print(f"  [INFO] gnomon: Derived k={num_raw_bases} total raw B-spline bases from knot vector.")

    # 3. Extract the reparameterization matrix ('z_transform')
    constraint_info = toml_data['config']['constraints']['pgs_main']['z_transform']
    z_dims, z_data = constraint_info['dim'], constraint_info['data']
    z_transform = np.array(z_data).reshape(z_dims, order='F')
    print(f"  [PRINT] gnomon: Loaded 'pgs_main' constraint matrix.")
    print_array_summary("z_transform", z_transform)

    # 4. *** THE DEFINITIVE FIX ***
    # Reconstruct the FULL RAW basis using the robust BSpline.design_matrix function.
    # This is the canonical and correct way to generate the basis matrix in SciPy.
    x_range = toml_data['config']['pgs_range']
    x_axis = np.linspace(x_range[0], x_range[1], N_POINTS_PLOT)
    raw_basis_matrix = BSpline.design_matrix(x_axis, knots, degree, extrapolate=False).toarray()
    print(f"  [PRINT] gnomon: Reconstructed FULL raw basis matrix using BSpline.design_matrix.")
    print_array_summary("raw_basis_matrix", raw_basis_matrix)

    # 5. Correctly slice the raw basis to isolate the non-constant functions
    raw_main_basis_functions = raw_basis_matrix[:, 1:]
    print(f"  [INFO] gnomon: Sliced raw basis to get the non-constant bases for constraining.")
    print_array_summary("raw_main_basis_functions", raw_main_basis_functions)
    
    # 6. Verify dimensions before applying the constraint
    if raw_main_basis_functions.shape[1] != z_transform.shape[0]:
        print(f"\n--- FATAL ERROR: Dimension mismatch for constraint matrix multiplication! ---")
        print(f"Non-constant raw basis has {raw_main_basis_functions.shape[1]} columns.")
        print(f"Constraint matrix 'z_transform' expects {z_transform.shape[0]} rows.")
        sys.exit(1)

    # 7. Apply transformation to get the FINAL CONSTRAINED basis: B_constrained = B_raw_non_constant @ Z
    constrained_basis_matrix = raw_main_basis_functions @ z_transform
    print(f"  [PRINT] gnomon: Created FINAL constrained basis matrix.")
    print_array_summary("constrained_basis_matrix", constrained_basis_matrix)
    
    # 8. Final dimension verification
    if constrained_basis_matrix.shape[1] != len(coeffs):
        print(f"\n--- FATAL ERROR: Final dimension mismatch! ---")
        print(f"Final constrained basis has {constrained_basis_matrix.shape[1]} columns.")
        print(f"Model file provides {len(coeffs)} coefficients for this term.")
        sys.exit(1)
        
    print("  [INFO] All dimension checks passed successfully.")
    return {"x_axis": x_axis, "basis_matrix": constrained_basis_matrix, "coeffs": coeffs}

def create_comparison_plot(mgcv_data, gnomon_data):
    """
    Creates ONE single 3x2 plot comparing all components of the main smooth term.
    """
    print("\n" + "="*80)
    print("--- STAGE 3: Generating the SINGLE 3x2 Comparison Plot ---")
    print("="*80)

    # --- Pre-calculate all components for plotting ---
    mgcv_basis = mgcv_data['basis_matrix']
    gnomon_basis = gnomon_data['basis_matrix']
    mgcv_weighted = mgcv_basis * mgcv_data['coeffs']
    gnomon_weighted = gnomon_basis * gnomon_data['coeffs']
    mgcv_final_curve = mgcv_weighted.sum(axis=1)
    gnomon_final_curve = gnomon_weighted.sum(axis=1)
    
    print(f"  [PRINT] mgcv: Calculated weighted basis and final curve.")
    print_array_summary("mgcv_weighted", mgcv_weighted)
    print_array_summary("mgcv_final_curve", mgcv_final_curve)
    print(f"  [PRINT] gnomon: Calculated weighted basis and final curve.")
    print_array_summary("gnomon_weighted", gnomon_weighted)
    print_array_summary("gnomon_final_curve", gnomon_final_curve)

    fig, axes = plt.subplots(3, 2, figsize=(15, 18), sharex=True, constrained_layout=True)
    fig.suptitle("Internal Component Comparison: s(variable_one) vs s(pgs)", fontsize=20)

    # --- Column Titles ---
    axes[0, 0].set_title("mgcv Model", fontsize=16)
    axes[0, 1].set_title("gnomon Model", fontsize=16)

    # --- Row 1: Constrained Basis Functions ---
    axes[0, 0].plot(mgcv_data['x_axis'], mgcv_basis, alpha=0.7)
    axes[0, 0].set_ylabel("Constrained Basis Value", fontsize=12)
    axes[0, 1].plot(gnomon_data['x_axis'], gnomon_basis, alpha=0.7)

    # --- Row 2: Weighted Basis Functions ---
    axes[1, 0].plot(mgcv_data['x_axis'], mgcv_weighted, alpha=0.7)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_ylabel("Weighted Basis Value", fontsize=12)
    axes[1, 1].plot(gnomon_data['x_axis'], gnomon_weighted, alpha=0.7)
    axes[1, 1].axhline(0, color='black', linestyle='--', linewidth=1)

    # --- Row 3: Final Smooth Curve and Overlay ---
    axes[2, 0].plot(mgcv_data['x_axis'], mgcv_final_curve, color='crimson', linewidth=3)
    axes[2, 0].set_xlabel("variable_one", fontsize=12)
    axes[2, 0].set_ylabel("Final Smooth Contribution", fontsize=12)
    axes[2, 0].set_title("Sum of Weighted Bases", fontsize=14)
    axes[2, 0].grid(True, linestyle=':', alpha=0.7)
    
    # The final verification plot: BOTH curves overlaid
    axes[2, 1].plot(mgcv_data['x_axis'], mgcv_final_curve, label='mgcv', color='blue', linewidth=6, alpha=0.6)
    axes[2, 1].plot(gnomon_data['x_axis'], gnomon_final_curve, label='gnomon', color='red', linewidth=2.5, linestyle='--')
    axes[2, 1].set_xlabel("pgs", fontsize=12)
    axes[2, 1].legend(title="Model")
    axes[2, 1].set_title("Verification: Overlay of Final Curves", fontsize=14)
    axes[2, 1].grid(True, linestyle=':', alpha=0.7)

    plt.show()

def main():
    """Main script to generate and display the basis function plots."""
    for f in [R_MODEL_PATH, RUST_MODEL_CONFIG_PATH]:
        if not f.is_file():
            print(f"--- FATAL ERROR: Required file not found: '{f}' ---"); sys.exit(1)

    mgcv_data = get_mgcv_basis_data()
    gnomon_data = get_gnomon_basis_data()
    create_comparison_plot(mgcv_data, gnomon_data)
    
    print("\n--- Script finished successfully. ---")

if __name__ == "__main__":
    main()
