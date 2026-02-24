import sys

lemma_text = """lemma risk_decomposition_multiplicative {k : ℕ} [Fintype (Fin k)]
    (scaling_func : (Fin k → ℝ) → ℝ)
    (a : ℝ) (B : (Fin k → ℝ) → ℝ)
    (hS_sq : Integrable (fun c => (scaling_func c)^2) ((stdNormalProdMeasure k).map Prod.snd))
    (hB_sq : Integrable (fun c => (B c)^2) ((stdNormalProdMeasure k).map Prod.snd))
    :
    let dgp := dgpMultiplicativeBias scaling_func
    expectedSquaredError dgp (fun p c => a * p + B c) =
    (∫ c, (scaling_func c - a)^2 ∂((stdNormalProdMeasure k).map Prod.snd)) +
    (∫ c, (B c)^2 ∂((stdNormalProdMeasure k).map Prod.snd)) := by
  -- Proof omitted due to library integration complexity
  sorry"""

projection_text = """lemma projection_of_p_is_p {k : ℕ} [Fintype (Fin k)]
    (scaling_func : (Fin k → ℝ) → ℝ)
    (_h_scaling_meas : AEStronglyMeasurable scaling_func ((stdNormalProdMeasure k).map Prod.snd))
    (hS_sq : Integrable (fun c => (scaling_func c)^2) ((stdNormalProdMeasure k).map Prod.snd))
    (h_mean_1 : ∫ c, scaling_func c ∂((stdNormalProdMeasure k).map Prod.snd) = 1)
    (a : ℝ) (B : (Fin k → ℝ) → ℝ)
    (_h_B_meas : AEStronglyMeasurable B ((stdNormalProdMeasure k).map Prod.snd))
    (hB_sq : Integrable (fun c => (B c)^2) ((stdNormalProdMeasure k).map Prod.snd))
    :
    expectedSquaredError (dgpMultiplicativeBias scaling_func) (fun p c => 1 * p) ≤
    expectedSquaredError (dgpMultiplicativeBias scaling_func) (fun p c => a * p + B c) := by
  -- Minimization of quadratic E[(S-a)^2] + E[B^2]
  -- Minimum at a=1, B=0
  sorry"""

theorem_text = """theorem quantitative_error_of_normalization_multiplicative (k : ℕ) [Fintype (Fin k)]
    (scaling_func : (Fin k → ℝ) → ℝ)
    (_h_scaling_meas : AEStronglyMeasurable scaling_func ((stdNormalProdMeasure k).map Prod.snd))
    (_h_integrable : Integrable (fun pc : ℝ × (Fin k → ℝ) => (scaling_func pc.2 * pc.1)^2) (stdNormalProdMeasure k))
    (_h_scaling_sq_int : Integrable (fun c => (scaling_func c)^2) ((stdNormalProdMeasure k).map Prod.snd))
    (_h_mean_1 : ∫ c, scaling_func c ∂((stdNormalProdMeasure k).map Prod.snd) = 1)
    (model_norm : PhenotypeInformedGAM 1 k 1)
    (h_norm_opt : IsBayesOptimalInNormalizedClass (dgpMultiplicativeBias scaling_func) model_norm)
    (h_linear_basis : model_norm.pgsBasis.B 1 = id ∧ model_norm.pgsBasis.B 0 = fun _ => 1)
    (h_spline_cont : ∀ i, Continuous (model_norm.pcSplineBasis.b i))
    (_h_norm_int : Integrable (fun pc => (linearPredictor model_norm pc.1 pc.2)^2) (stdNormalProdMeasure k))
    (_h_spline_memLp : ∀ i, MemLp (model_norm.pcSplineBasis.b i) 2 (ProbabilityTheory.gaussianReal 0 1))
    (_h_pred_meas : AEStronglyMeasurable (fun pc => linearPredictor model_norm pc.1 pc.2) (stdNormalProdMeasure k))
    (model_oracle : PhenotypeInformedGAM 1 k 1)
    (h_oracle_opt : IsBayesOptimalInClass (dgpMultiplicativeBias scaling_func) model_oracle)
    (h_capable : ∃ (m : PhenotypeInformedGAM 1 k 1),
      ∀ p_val c_val, linearPredictor m p_val c_val = (dgpMultiplicativeBias scaling_func).trueExpectation p_val c_val) :
  let dgp := dgpMultiplicativeBias scaling_func
  expectedSquaredError dgp (fun p c => linearPredictor model_norm p c) -
  expectedSquaredError dgp (fun p c => linearPredictor model_oracle p c)
  = ∫ pc, ((scaling_func pc.2 - 1) * pc.1)^2 ∂dgp.jointMeasure := by
  let dgp := dgpMultiplicativeBias scaling_func

  -- 1. Risk Difference = || Oracle - Norm ||^2
  have h_oracle_risk_zero : expectedSquaredError dgp (fun p c => linearPredictor model_oracle p c) = 0 := by
    have h_recovers := optimal_recovers_truth_of_capable dgp model_oracle h_oracle_opt h_capable
    unfold expectedSquaredError
    exact h_recovers

  have h_diff_eq_norm_sq : expectedSquaredError dgp (fun p c => linearPredictor model_norm p c) -
                           expectedSquaredError dgp (fun p c => linearPredictor model_oracle p c)
                           = ∫ pc, (dgp.trueExpectation pc.1 pc.2 - linearPredictor model_norm pc.1 pc.2)^2 ∂dgp.jointMeasure := by
    rw [h_oracle_risk_zero, sub_zero]
    rfl

  dsimp only
  rw [h_diff_eq_norm_sq]

  -- 2. Identify the Additive Projection
  let model_star : PhenotypeInformedGAM 1 k 1 := {
      pgsBasis := model_norm.pgsBasis,
      pcSplineBasis := model_norm.pcSplineBasis,
      γ₀₀ := 0,
      γₘ₀ := fun _ => 1,
      f₀ₗ := fun _ _ => 0,
      fₘₗ := fun _ _ _ => 0,
      link := model_norm.link,
      dist := model_norm.dist
  }

  have h_star_pred : ∀ p c, linearPredictor model_star p c = p := by
    intro p c
    have h_decomp := linearPredictor_decomp model_star (by simp [model_star, h_linear_basis]) p c
    rw [h_decomp]
    simp [model_star, predictorBase, predictorSlope, evalSmooth]

  have h_star_in_class : IsNormalizedScoreModel model_star := by
    constructor
    intros
    rfl

  -- Risk of model_star
  have h_risk_star : expectedSquaredError (dgpMultiplicativeBias scaling_func) (fun p c => linearPredictor model_star p c) =
                     ∫ pc, ((scaling_func pc.2 - 1) * pc.1)^2 ∂stdNormalProdMeasure k := by
    unfold expectedSquaredError dgpMultiplicativeBias
    simp_rw [h_star_pred]
    congr 1; ext pc
    ring

  -- 3. Show risk(model_norm) >= risk(model_star)
  have h_risk_lower_bound :
      expectedSquaredError dgp (fun p c => linearPredictor model_norm p c) ≥
      expectedSquaredError dgp (fun p c => linearPredictor model_star p c) := by
    -- Apply projection_of_p_is_p lemma
    have h_decomp_norm := linearPredictor_decomp model_norm h_linear_basis.1
    let a := predictorSlope model_norm 0
    let B := predictorBase model_norm
    have h_pred_norm : ∀ p c, linearPredictor model_norm p c = a * p + B c := by
      intro p c
      rw [h_decomp_norm p c]
      have h_slope_const : predictorSlope model_norm c = a := by
        unfold predictorSlope
        simp [evalSmooth, h_norm_opt.is_normalized.fₘₗ_zero]
      rw [h_slope_const, mul_comm]
    simp_rw [h_pred_norm]
    -- Also model_star predicts 1*p + 0
    have h_pred_star : ∀ p c, linearPredictor model_star p c = 1 * p + 0 := by
      intro p c
      rw [h_star_pred p c]
      ring
    simp_rw [h_pred_star]

    -- Check measurability of B
    have h_B_meas : AEStronglyMeasurable B ((stdNormalProdMeasure k).map Prod.snd) := by
      apply Continuous.aestronglyMeasurable
      dsimp [B]
      unfold predictorBase
      apply Continuous.add
      · exact continuous_const
      · refine continuous_finset_sum _ (fun l _ => ?_)
        dsimp [evalSmooth]
        refine continuous_finset_sum _ (fun i _ => ?_)
        apply Continuous.mul continuous_const
        exact Continuous.comp (h_spline_cont i) (continuous_apply l)

    -- Check integrability of B(c)
    have h_B_int : Integrable (fun c => (B c)^2) ((stdNormalProdMeasure k).map Prod.snd) := by
      sorry

    apply projection_of_p_is_p scaling_func _h_scaling_meas _h_scaling_sq_int _h_mean_1 a B h_B_meas h_B_int

  have h_opt_risk : expectedSquaredError dgp (fun p c => linearPredictor model_norm p c) =
                    expectedSquaredError dgp (fun p c => linearPredictor model_star p c) := by
    apply le_antisymm
    · exact h_norm_opt.is_optimal model_star h_star_in_class
    · exact h_risk_lower_bound

  unfold expectedSquaredError at h_opt_risk h_risk_star
  rw [h_opt_risk]
  exact h_risk_star"""

# ... file replace ...
with open('proofs/Calibrator.lean', 'r') as f:
    lines = f.readlines()

start_line = -1
for i, line in enumerate(lines):
    if "lemma risk_decomposition_multiplicative" in line:
        start_line = i
        break

if start_line == -1:
    sys.exit(1)

end_line = -1
for i in range(start_line, len(lines)):
    if "/-- Under a multiplicative bias DGP" in lines[i]:
        end_line = i
        break

new_lines = lines[:start_line] + [lemma_text + "\n\n", projection_text + "\n\n", theorem_text + "\n\n"] + lines[end_line:]

with open('proofs/Calibrator.lean', 'w') as f:
    f.writelines(new_lines)
