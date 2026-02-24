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
  let μ := stdNormalProdMeasure k
  let μ0 := ProbabilityTheory.gaussianReal 0 1
  let ν0 := Measure.pi (fun (_ : Fin k) => ProbabilityTheory.gaussianReal 0 1)
  have h_indep : μ = μ0.prod ν0 := rfl
  have h_map_snd : μ.map Prod.snd = ν0 := by
    rw [h_indep, Measure.map_snd_prod]
    simp

  -- Cast hypotheses to ν0
  have hS_sq' : Integrable (fun c => (scaling_func c)^2) ν0 := by rw [← h_map_snd]; exact hS_sq
  have hB_sq' : Integrable (fun c => (B c)^2) ν0 := by rw [← h_map_snd]; exact hB_sq

  unfold expectedSquaredError dgpMultiplicativeBias
  simp only [dgpMultiplicativeBias, stdNormalProdMeasure]

  -- Expand integrand
  -- Use order f(p) * g(c) for Integral.prod_mul
  have h_eq : ∀ p c, (scaling_func c * p - (a * p + B c))^2 =
                     p^2 * (scaling_func c - a)^2 + (-2 * p) * ((scaling_func c - a) * B c) + 1 * (B c)^2 := by
    intros p c; ring

  simp_rw [h_eq]

  -- Integrate term by term
  -- 1. E[p^2 * (S-a)^2] = E[p^2] * E[(S-a)^2]
  -- 2. E[-2p * (S-a)B] = E[-2p] * E[(S-a)B]
  -- 3. E[1 * B^2] = E[1] * E[B^2]

  -- Need to show terms are integrable.
  -- Term 1
  have h_term1_int : Integrable (fun pc : ℝ × (Fin k → ℝ) => pc.1^2 * (scaling_func pc.2 - a)^2) μ := by
    rw [h_indep]
    apply Integrable.prod_mul (gaussian_moments_integrable 2)
    -- Integrable (S-a)^2
    have h1 : Integrable (fun c => (scaling_func c)^2) ν0 := hS_sq'
    have h2 : Integrable (fun c => (scaling_func c)) ν0 := Integrable.pow_one (h1.pow_const 2)
    have h3 : Integrable (fun _ : Fin k → ℝ => a^2) ν0 := integrable_const _
    have h_expand : ∀ c, (scaling_func c - a)^2 = (scaling_func c)^2 - 2*a*scaling_func c + a^2 := by intro; ring
    simp_rw [h_expand]
    exact (h1.sub (h2.const_mul (2*a))).add h3

  -- Term 2
  have h_term2_int : Integrable (fun pc : ℝ × (Fin k → ℝ) => (-2 * pc.1) * ((scaling_func pc.2 - a) * B pc.2)) μ := by
    rw [h_indep]
    apply Integrable.prod_mul (Integrable.const_mul (gaussian_moments_integrable 1) (-2))
    -- Integrable (S-a)B
    apply Integrable.const_mul
    have hS : MemLp scaling_func 2 ν0 := MemLp.of_integrable_sq hS_sq'.aestronglyMeasurable hS_sq'
    have hB : MemLp B 2 ν0 := MemLp.of_integrable_sq hB_sq'.aestronglyMeasurable hB_sq'
    have hSB : Integrable (fun c => scaling_func c * B c) ν0 := MemLp.integrable_mul hS hB
    have h_aB : Integrable (fun c => a * B c) ν0 := (MemLp.integrable hB one_le_two).const_mul a
    have h_expand : ∀ c, (scaling_func c - a) * B c = scaling_func c * B c - a * B c := by intro; ring
    simp_rw [h_expand]
    exact hSB.sub h_aB

  -- Term 3
  have h_term3_int : Integrable (fun pc : ℝ × (Fin k → ℝ) => 1 * (B pc.2)^2) μ := by
    rw [h_indep]
    apply Integrable.prod_mul (by simp; exact integrable_const 1) hB_sq'

  rw [integral_add]
  · rw [integral_add]
    · -- Evaluate Term 1
      rw [h_indep, integral_prod_mul (μ:=μ0) (ν:=ν0)]
      · rw [gaussian_second_moment]
        simp
        have h_expand : ∀ c, (scaling_func c - a)^2 = (scaling_func c)^2 - 2*a*scaling_func c + a^2 := by intro; ring
        simp_rw [h_expand]
        rw [← h_map_snd]
        rfl
      · exact gaussian_moments_integrable 2
      · have h1 : Integrable (fun c => (scaling_func c)^2) ν0 := hS_sq'
        have h2 : Integrable (fun c => (scaling_func c)) ν0 := Integrable.pow_one (h1.pow_const 2)
        have h3 : Integrable (fun _ : Fin k → ℝ => a^2) ν0 := integrable_const _
        have h_expand : ∀ c, (scaling_func c - a)^2 = (scaling_func c)^2 - 2*a*scaling_func c + a^2 := by intro; ring
        simp_rw [h_expand]
        exact (h1.sub (h2.const_mul (2*a))).add h3

    · -- Evaluate Term 2
      rw [h_indep, integral_prod_mul (μ:=μ0) (ν:=ν0)]
      · rw [gaussian_mean_zero]
        simp
      · exact (Integrable.const_mul (gaussian_moments_integrable 1) (-2))
      · apply Integrable.const_mul
        have hS : MemLp scaling_func 2 ν0 := MemLp.of_integrable_sq hS_sq'.aestronglyMeasurable hS_sq'
        have hB : MemLp B 2 ν0 := MemLp.of_integrable_sq hB_sq'.aestronglyMeasurable hB_sq'
        have hSB : Integrable (fun c => scaling_func c * B c) ν0 := MemLp.integrable_mul hS hB
        have h_aB : Integrable (fun c => a * B c) ν0 := (MemLp.integrable hB one_le_two).const_mul a
        have h_expand : ∀ c, (scaling_func c - a) * B c = scaling_func c * B c - a * B c := by intro; ring
        simp_rw [h_expand]
        exact hSB.sub h_aB

    · exact h_term1_int
    · exact h_term2_int
  · -- Evaluate Term 3
    rw [h_indep, integral_prod_mul (μ:=μ0) (ν:=ν0)]
    · simp
      rw [← h_map_snd]
      rfl
    · simp; exact integrable_const 1
    · exact hB_sq'

  · apply Integrable.add
    · exact h_term1_int
    · exact h_term2_int
  · exact h_term3_int"""

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

    have h_rewrite : (fun p c => 1 * p) = (fun p c => 1 * p + (fun _ => 0) c) := by
      ext p c; simp
    rw [h_rewrite]
    rw [risk_decomposition_multiplicative scaling_func 1 (fun _ => 0) hS_sq (by simp; exact integrable_zero _ _ _)]
    rw [risk_decomposition_multiplicative scaling_func a B hS_sq hB_sq]

    simp only [sub_zero, integral_zero, add_zero]

    have h_ineq : ∫ c, (scaling_func c - 1)^2 ∂((stdNormalProdMeasure k).map Prod.snd) ≤
                  ∫ c, (scaling_func c - a)^2 ∂((stdNormalProdMeasure k).map Prod.snd) := by
      let μC := (stdNormalProdMeasure k).map Prod.snd
      have h_expand : ∀ c, (scaling_func c - a)^2 = (scaling_func c - 1)^2 + 2 * (1 - a) * (scaling_func c - 1) + (1 - a)^2 := by
        intro c; ring
      simp_rw [h_expand]
      rw [integral_add]
      · rw [integral_add]
        · have h_cross : ∫ c, 2 * (1 - a) * (scaling_func c - 1) ∂μC = 0 := by
            rw [integral_const_mul]
            have h_sub : ∫ c, scaling_func c - 1 ∂μC = ∫ c, scaling_func c ∂μC - ∫ c, 1 ∂μC := by
               rw [integral_sub]
               · have hS : Integrable scaling_func μC := MemLp.integrable (MemLp.of_integrable_sq _h_scaling_meas hS_sq) one_le_two
                 exact hS
               · exact integrable_const 1
            rw [h_sub, h_mean_1]
            simp
          rw [h_cross, zero_add]
          have h_pos : 0 ≤ ∫ c, (1 - a)^2 ∂μC := by
             apply integral_nonneg
             intro; apply sq_nonneg
          linarith
        · have hS : Integrable scaling_func μC := MemLp.integrable (MemLp.of_integrable_sq _h_scaling_meas hS_sq) one_le_two
          have h_S_1_sq : Integrable (fun c => (scaling_func c - 1)^2) μC := by
             have : ∀ c, (scaling_func c - 1)^2 = scaling_func c ^ 2 - 2 * scaling_func c + 1 := by intro; ring
             simp_rw [this]
             exact (hS_sq.sub (hS.const_mul 2)).add (integrable_const 1)
          exact h_S_1_sq
        · apply Integrable.const_mul
          have hS : Integrable scaling_func μC := MemLp.integrable (MemLp.of_integrable_sq _h_scaling_meas hS_sq) one_le_two
          exact hS.sub (integrable_const 1)
      · apply Integrable.add
        · have hS : Integrable scaling_func μC := MemLp.integrable (MemLp.of_integrable_sq _h_scaling_meas hS_sq) one_le_two
          have h_S_1_sq : Integrable (fun c => (scaling_func c - 1)^2) μC := by
             have : ∀ c, (scaling_func c - 1)^2 = scaling_func c ^ 2 - 2 * scaling_func c + 1 := by intro; ring
             simp_rw [this]
             exact (hS_sq.sub (hS.const_mul 2)).add (integrable_const 1)
          exact h_S_1_sq
        · apply Integrable.const_mul
          have hS : Integrable scaling_func μC := MemLp.integrable (MemLp.of_integrable_sq _h_scaling_meas hS_sq) one_le_two
          exact hS.sub (integrable_const 1)
      · exact integrable_const _

    have h_B_nonneg : 0 ≤ ∫ c, (B c)^2 ∂((stdNormalProdMeasure k).map Prod.snd) :=
       integral_nonneg (fun _ => sq_nonneg _)

    linarith"""

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
        exact (h_spline_cont i).comp (continuous_apply l)

    -- Check integrability of B(c)
    let μ := stdNormalProdMeasure k
    have h_norm_L2 : MemLp (fun pc : ℝ × (Fin k → ℝ) => linearPredictor model_norm pc.1 pc.2) 2 μ :=
         MemLp.of_integrable_sq _h_pred_meas _h_norm_int
    have h_P_L2 : MemLp (fun pc : ℝ × (Fin k → ℝ) => pc.1) 2 μ :=
         MemLp.of_integrable_sq aestronglyMeasurable_fst (gaussian_moments_integrable 2)
    have h_aP_L2 : MemLp (fun pc : ℝ × (Fin k → ℝ) => a * pc.1) 2 μ := h_P_L2.const_mul a
    have h_B_L2_joint : MemLp (fun pc : ℝ × (Fin k → ℝ) => B pc.2) 2 μ := by
         have h_diff : ∀ pc, B pc.2 = linearPredictor model_norm pc.1 pc.2 - a * pc.1 := by
           intro pc
           rw [h_pred_norm]
           ring
         convert h_norm_L2.sub h_aP_L2
         ext pc
         rw [h_diff]

    -- Extract marginal integrability
    have h_sq_int_joint : Integrable (fun pc => (B pc.2)^2) μ :=
         MemLp.integrable_sq h_B_L2_joint

    have h_B_int : Integrable (fun c => (B c)^2) ((stdNormalProdMeasure k).map Prod.snd) := by
      rw [← MeasureTheory.integrable_comp_snd_iff (h_B_meas.pow 2)]
      have : IsProbabilityMeasure ((stdNormalProdMeasure k).map Prod.fst) :=
         Measure.isProbabilityMeasure_map (by fun_prop)
      exact h_sq_int_joint

    apply projection_of_p_is_p scaling_func _h_scaling_meas _h_scaling_sq_int _h_mean_1 a B h_B_meas h_B_int

  have h_opt_risk : expectedSquaredError dgp (fun p c => linearPredictor model_norm p c) =
                    expectedSquaredError dgp (fun p c => linearPredictor model_star p c) := by
    apply le_antisymm
    · exact h_norm_opt.is_optimal model_star h_star_in_class
    · exact h_risk_lower_bound

  unfold expectedSquaredError at h_opt_risk h_risk_star
  rw [h_opt_risk]
  exact h_risk_star"""

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
