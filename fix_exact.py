lines_to_replace = {
    6432: "  exact Calibrator.scenarios_are_distinct k hk_pos",
    6443: "  exact Calibrator.scenarios_are_distinct k hk_pos",
    6456: "  exact Calibrator.necessity_of_phenotype_data",
    6463: "  exact Calibrator.drift_implies_attenuation phys c_near c_far h_decay",
    6470: "  exact Calibrator.directionalLD_nonzero_implies_slope_ne_one arch c h_genic_pos h_cov_ne",
    6540: "  exact Calibrator.selection_variation_implies_nonlinear_slope arch c₁ c₂ h_genic_pos₁ h_genic_pos₂ h_link h_sel_var",
    6561: "  exact Calibrator.normalization_erases_heritability arch c h_genic_pos h_cov_pos",
    6566: "  exact Calibrator.neutral_drift_implies_additive_correction mech",
    6578: "  exact Calibrator.ld_decay_implies_shrinkage mech c_near c_far h_dist h_mono",
    6598: "  exact Calibrator.optimal_slope_trace_variance arch c h_genic_pos",
    6620: "  exact Calibrator.integral_mul_fst_snd_eq_zero μ h_indep hP0 hC0",
    6635: "  exact integral_mul_fst_snd_eq_zero_proven μ h_indep hP_int hC_int hP0 hC0",
    6683: "  exact Calibrator.linear_coeff_zero_of_quadratic_nonneg a b h",
    6726: "  exact Calibrator.linear_coeff_zero_of_quadratic_nonneg a b h",
    6741: "  exact Calibrator.optimal_intercept_eq_mean_of_zero_mean_p μ Y a b hY hP hP0 h_orth_1",
    6785: "  exact Calibrator.evalSmooth_eq_zero_of_raw_gen h_raw l c_val",
    6791: "  exact Calibrator.evalSmooth_interaction_eq_zero_of_raw_gen h_raw m l c_val",
    6799: "  exact Calibrator.linearPredictor_eq_affine_of_raw_gen model_raw h_raw h_lin",
    6819: "  convert Calibrator.rawOptimal_implies_orthogonality_gen model dgp h_opt h_linear hY_int hP_int hP2_int hYP_int h_resid_sq_int"
}

with open("proofs/Calibrator.lean", "r") as f:
    lines = f.readlines()

for line_num, replacement in lines_to_replace.items():
    # line_num is 1-based, list is 0-based
    idx = line_num - 1
    # Check if the line is indeed exact?
    if "exact?" in lines[idx]:
        lines[idx] = replacement + "\n"
    else:
        # Search nearby lines if line numbers shifted
        found = False
        for offset in range(-5, 6):
            if 0 <= idx + offset < len(lines) and "exact?" in lines[idx + offset]:
                lines[idx + offset] = replacement + "\n"
                found = True
                break
        if not found:
            print(f"Warning: exact? not found at line {line_num} or nearby")

with open("proofs/Calibrator.lean", "w") as f:
    f.writelines(lines)
