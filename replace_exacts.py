import sys

replacements = {
    6163: "exact Calibrator.scenarios_are_distinct k hk_pos",
    6174: "exact _root_.scenarios_are_distinct k hk_pos",
    6187: "exact fun a => necessity_of_phenotype_data",
    6194: "exact h_decay",
    6201: "exact directionalLD_nonzero_implies_slope_ne_one arch c h_genic_pos h_cov_ne",
    6271: "exact selection_variation_implies_nonlinear_slope arch c₁ c₂ h_genic_pos₁ h_genic_pos₂ h_link h_sel_var",
    6292: "exact normalization_erases_heritability arch c h_genic_pos h_cov_pos",
    6297: "exact fun c => neutral_drift_implies_additive_correction mech c",
    6309: "exact h_mono h_dist",
    6329: "exact optimal_slope_trace_variance arch c h_genic_pos",
    6351: "exact integral_mul_fst_snd_eq_zero μ h_indep hP0 hC0",
    6366: "exact integral_mul_fst_snd_eq_zero_proven μ h_indep hP_int hC_int hP0 hC0",
    6414: "exact linear_coeff_zero_of_quadratic_nonneg a b h",
    6457: "exact linear_coeff_zero_of_quadratic_nonneg a b h",
    6472: "exact optimal_intercept_eq_mean_of_zero_mean_p μ Y a b hY hP hP0 h_orth_1",
    6516: "exact evalSmooth_eq_zero_of_raw_gen h_raw l c_val",
    6522: "exact evalSmooth_interaction_eq_zero_of_raw_gen h_raw m l c_val",
    6530: "exact fun p c => linearPredictor_eq_affine_of_raw_gen model_raw h_raw h_lin p c",
    6550: "exact let a := model.γ₀₀; let b := model.γₘ₀ ⟨0, by norm_num⟩; rawOptimal_implies_orthogonality_gen model dgp h_opt h_linear hY_int hP_int hP2_int hYP_int h_resid_sq_int"
}

filepath = "proofs/Calibrator.lean"
with open(filepath, "r") as f:
    lines = f.readlines()

for line_num, replacement in replacements.items():
    idx = line_num - 1
    if idx < len(lines):
        original = lines[idx]
        indent = original[:original.find("exact?")]
        if "exact?" not in original:
             # Handle case where exact? is not alone or indentation is tricky
             # Try to find exact? in the line
             pos = original.find("exact?")
             if pos != -1:
                 indent = original[:pos]
             else:
                 print(f"Warning: exact? not found at line {line_num}: {original.strip()}")
                 continue

        # Keep any prefix before exact? (like '· ') if it was part of indent logic
        # But my indent calculation above is simple.

        # Check if line has suffix (e.g. semicolon)
        suffix = ""
        if ";" in original:
             suffix = ";"

        lines[idx] = indent + replacement + suffix + "\n"

with open(filepath, "w") as f:
    f.writelines(lines)
