    relative to the optimal model is the variance of the interaction term.

    Error = || Oracle - Norm ||^2 = E[ ( (scaling(C) - 1) * P )^2 ]

    Assumption: E[scaling(C)] = 1 (centered scaling).
    Then the additive projection of scaling(C)*P is 1*P.
    The residual is (scaling(C) - 1)*P. -/
lemma risk_decomposition_multiplicative {k : ℕ} [Fintype (Fin k)]
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
  let μP := μ.map Prod.fst
  let μC := μ.map Prod.snd
  have h_indep : μ = μP.prod μC := rfl

  unfold expectedSquaredError dgpMultiplicativeBias
  simp only [dgpMultiplicativeBias, stdNormalProdMeasure]

  -- Expand integrand
  have h_eq : ∀ p c, (scaling_func c * p - (a * p + B c))^2 =
                     (scaling_func c - a)^2 * p^2 - 2 * (scaling_func c - a) * B c * p + (B c)^2 := by
    intros p c; ring

  simp_rw [h_eq]

  -- Integrate term by term
  -- 1. E[(S-a)^2 * P^2] = E[(S-a)^2] * E[P^2]
  -- 2. E[-2(S-a)B * P] = -2 E[(S-a)B] * E[P]
  -- 3. E[B^2]

  -- Need to show terms are integrable.
  -- Term 1: (S-a)^2 * P^2
  have h_term1_int : Integrable (fun pc : ℝ × (Fin k → ℝ) => (scaling_func pc.2 - a)^2 * pc.1^2) μ := by
    have h_prod := Integrable.prod_mul (gaussian_moments_integrable 2) (by
        -- Integrable (S-a)^2
        have h1 : Integrable (fun c => (scaling_func c)^2) μC := hS_sq
        have h2 : Integrable (fun c => (scaling_func c)) μC := Integrable.pow_one (h1.pow_const 2)
        have h3 : Integrable (fun _ : Fin k → ℝ => a^2) μC := integrable_const _
        have h_expand : ∀ c, (scaling_func c - a)^2 = (scaling_func c)^2 - 2*a*scaling_func c + a^2 := by intro; ring
        simp_rw [h_expand]
        exact (h1.sub (h2.const_mul (2*a))).add h3
      )
    apply h_prod.congr
    intro x
    ring

  -- Term 2: -2(S-a)B * P
  have h_term2_int : Integrable (fun pc : ℝ × (Fin k → ℝ) => -2 * (scaling_func pc.2 - a) * B pc.2 * pc.1) μ := by
    have h_prod := Integrable.prod_mul (gaussian_moments_integrable 1) (by
        -- Integrable -2(S-a)B
        apply Integrable.const_mul
        have hS : MemLp scaling_func 2 μC := MemLp.of_integrable_sq hS_sq.aestronglyMeasurable hS_sq
        have hB : MemLp B 2 μC := MemLp.of_integrable_sq hB_sq.aestronglyMeasurable hB_sq
        have hSB : Integrable (fun c => scaling_func c * B c) μC := MemLp.integrable_mul hS hB
        have h_aB : Integrable (fun c => a * B c) μC := (MemLp.integrable hB one_le_two).const_mul a
        have h_expand : ∀ c, (scaling_func c - a) * B c = scaling_func c * B c - a * B c := by intro; ring
        simp_rw [h_expand]
        exact hSB.sub h_aB
      )
    apply h_prod.congr
    intro x
    ring

  -- Term 3: B^2
  have h_term3_int : Integrable (fun pc : ℝ × (Fin k → ℝ) => (B pc.2)^2) μ := by
    have h_prod := Integrable.prod_mul (by simp; exact integrable_const 1) hB_sq
    apply h_prod.congr
    intro x
    simp

  rw [integral_add]
  · rw [integral_sub]
    · -- Evaluate Term 1
      rw [integral_prod_mul (μ:=μP) (ν:=μC)]
      · rw [gaussian_second_moment]
        simp
        have h_expand : ∀ c, (scaling_func c - a)^2 = (scaling_func c)^2 - 2*a*scaling_func c + a^2 := by intro; ring
        simp_rw [h_expand]
        -- Re-prove integrability for this step? No, just calculation.
        -- Need to use the values.
        -- Goal is ∫ (S-a)^2.
        rfl
      · exact gaussian_moments_integrable 2
      · have h1 : Integrable (fun c => (scaling_func c)^2) μC := hS_sq
        have h2 : Integrable (fun c => (scaling_func c)) μC := Integrable.pow_one (h1.pow_const 2)
        have h3 : Integrable (fun _ : Fin k → ℝ => a^2) μC := integrable_const _
        have h_expand : ∀ c, (scaling_func c - a)^2 = (scaling_func c)^2 - 2*a*scaling_func c + a^2 := by intro; ring
        simp_rw [h_expand]
        exact (h1.sub (h2.const_mul (2*a))).add h3

    · -- Evaluate Term 2
      rw [integral_prod_mul (μ:=μP) (ν:=μC)]
      · rw [gaussian_mean_zero]
        simp
      · exact gaussian_moments_integrable 1
      · apply Integrable.const_mul
        have hS : MemLp scaling_func 2 μC := MemLp.of_integrable_sq hS_sq.aestronglyMeasurable hS_sq
        have hB : MemLp B 2 μC := MemLp.of_integrable_sq hB_sq.aestronglyMeasurable hB_sq
        have hSB : Integrable (fun c => scaling_func c * B c) μC := MemLp.integrable_mul hS hB
        have h_aB : Integrable (fun c => a * B c) μC := (MemLp.integrable hB one_le_two).const_mul a
        have h_expand : ∀ c, (scaling_func c - a) * B c = scaling_func c * B c - a * B c := by intro; ring
        simp_rw [h_expand]
        exact hSB.sub h_aB

    · exact h_term1_int
    · exact h_term2_int
  · -- Evaluate Term 3
    rw [integral_prod_mul (μ:=μP) (ν:=μC)]
    · simp
    · simp; exact integrable_const 1
    · exact hB_sq

  · exact h_term1_int.sub h_term2_int
  · exact h_term3_int

lemma projection_of_p_is_p {k : ℕ} [Fintype (Fin k)]
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

    -- Goal: ∫ (S-1)^2 ≤ ∫ (S-a)^2 + ∫ B^2
    -- ∫ B^2 ≥ 0, so sufficient to show ∫ (S-1)^2 ≤ ∫ (S-a)^2

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
        · -- Integrability of (S-1)^2
          have hS : Integrable scaling_func μC := MemLp.integrable (MemLp.of_integrable_sq _h_scaling_meas hS_sq) one_le_two
          have h_S_1_sq : Integrable (fun c => (scaling_func c - 1)^2) μC := by
             have : ∀ c, (scaling_func c - 1)^2 = scaling_func c ^ 2 - 2 * scaling_func c + 1 := by intro; ring
             simp_rw [this]
             exact (hS_sq.sub (hS.const_mul 2)).add (integrable_const 1)
          exact h_S_1_sq
        · -- Integrability of 2(1-a)(S-1)
          apply Integrable.const_mul
          have hS : Integrable scaling_func μC := MemLp.integrable (MemLp.of_integrable_sq _h_scaling_meas hS_sq) one_le_two
          exact hS.sub (integrable_const 1)
      · -- Integrability of first two terms
         apply Integrable.add
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

    linarith

theorem quantitative_error_of_normalization_multiplicative (k : ℕ) [Fintype (Fin k)]
    (scaling_func : (Fin k → ℝ) → ℝ)
    (_h_scaling_meas : AEStronglyMeasurable scaling_func ((stdNormalProdMeasure k).map Prod.snd))
    (_h_integrable : Integrable (fun pc : ℝ × (Fin k → ℝ) => (scaling_func pc.2 * pc.1)^2) (stdNormalProdMeasure k))
    (_h_scaling_sq_int : Integrable (fun c => (scaling_func c)^2) ((stdNormalProdMeasure k).map Prod.snd))
    (_h_mean_1 : ∫ c, scaling_func c ∂((stdNormalProdMeasure k).map Prod.snd) = 1)
    (model_norm : PhenotypeInformedGAM 1 k 1)
    (h_norm_opt : IsBayesOptimalInNormalizedClass (dgpMultiplicativeBias scaling_func) model_norm)
    (h_linear_basis : model_norm.pgsBasis.B 1 = id ∧ model_norm.pgsBasis.B 0 = fun _ => 1)
    -- Add Integrability hypothesis for the normalized model to avoid specification gaming
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
    -- Need to show model_norm has form a*p + B(c)
    have h_decomp_norm := linearPredictor_decomp model_norm h_linear_basis.1
    let a := predictorSlope model_norm 0
    let B := predictorBase model_norm
    have h_pred_norm : ∀ p c, linearPredictor model_norm p c = a * p + B c := by
      intro p c
      rw [h_decomp_norm p c]
      -- For normalized model, slope is constant
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
    have h_B_meas_sq : AEStronglyMeasurable (fun c => (B c)^2) ((stdNormalProdMeasure k).map Prod.snd) := by
       have h_joint_meas := h_sq_int_joint.aestronglyMeasurable
       exact AEStronglyMeasurable.snd h_joint_meas

    have h_B_int : Integrable (fun c => (B c)^2) ((stdNormalProdMeasure k).map Prod.snd) := by
      rw [← MeasureTheory.integrable_comp_snd_iff h_B_meas_sq]
      have : IsProbabilityMeasure ((stdNormalProdMeasure k).map Prod.fst) :=
         Measure.isProbabilityMeasure_map (by fun_prop)
      exact h_sq_int_joint

    -- Check measurability of B
    have h_B_meas : AEStronglyMeasurable B ((stdNormalProdMeasure k).map Prod.snd) := by
      -- Since B is integrable L2, it is AEStronglyMeasurable
      exact (MemLp.of_integrable_sq (h_B_int.aestronglyMeasurable) h_B_int).aestronglyMeasurable

    apply projection_of_p_is_p scaling_func _h_scaling_meas _h_scaling_sq_int _h_mean_1 a B h_B_meas h_B_int

  have h_opt_risk : expectedSquaredError dgp (fun p c => linearPredictor model_norm p c) =
                    expectedSquaredError dgp (fun p c => linearPredictor model_star p c) := by
    apply le_antisymm
    · exact h_norm_opt.is_optimal model_star h_star_in_class
    · exact h_risk_lower_bound
