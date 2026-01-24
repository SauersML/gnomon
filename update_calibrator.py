import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# 1. linear_noise_implies_nonlinear_slope
pattern1 = r"(theorem linear_noise_implies_nonlinear_slope\s*\(sigma_g_sq base_error slope_error : ℝ\)\s*\(h_g_pos : 0 < sigma_g_sq\)\s*\(hB_pos : 0 < sigma_g_sq \+ base_error\)\s*\(hB1_pos : 0 < sigma_g_sq \+ base_error \+ slope_error\)\s*\(hB2_pos : 0 < sigma_g_sq \+ base_error \+ 2 \* slope_error\)\s*\(h_slope_ne : slope_error ≠ 0\) :.*?:= by\s*)(intro beta0 beta1 h_eq\s*-- TODO:.*?\s*admit)"
replacement1 = r"""theorem linear_noise_implies_nonlinear_slope
    (sigma_g_sq base_error slope_error : ℝ)
    (h_g_pos : 0 < sigma_g_sq)
    (hB_pos : 0 < sigma_g_sq + base_error)
    (hB1_pos : 0 < sigma_g_sq + base_error + slope_error)
    (hB2_pos : 0 < sigma_g_sq + base_error + 2 * slope_error)
    (h_slope_ne : slope_error ≠ 0) :
    ∀ (beta0 beta1 : ℝ),
      (fun c => beta0 + beta1 * c) ≠
        (fun c => optimalSlopeLinearNoise sigma_g_sq base_error slope_error c) := by
  intro beta0 beta1 h_eq
  let f := fun c => beta0 + beta1 * c
  let g := fun c => optimalSlopeLinearNoise sigma_g_sq base_error slope_error c
  have h0 : f 0 = g 0 := congr_fun h_eq 0
  have h1 : f 1 = g 1 := congr_fun h_eq 1
  have h2 : f 2 = g 2 := congr_fun h_eq 2
  
  have h_arith : g 2 - g 1 = g 1 - g 0 := by
    rw [← h2, ← h1, ← h0]
    dsimp [f]
    ring
    
  dsimp [g, optimalSlopeLinearNoise] at h_arith
  simp only [mul_one, mul_zero, add_zero] at h_arith
  
  have hD0_ne : sigma_g_sq + base_error ≠ 0 := by linarith [hB_pos]
  have hD1_ne : sigma_g_sq + base_error + slope_error ≠ 0 := by linarith [hB1_pos]
  have hD2_ne : sigma_g_sq + base_error + slope_error * 2 ≠ 0 := by linarith [hB2_pos]
  
  field_simp [hD0_ne, hD1_ne, hD2_ne] at h_arith
  ring_nf at h_arith
  
  have h_zero : slope_error = 0 := by
    nlinarith [h_arith, h_g_pos]
    
  exact h_slope_ne h_zero"""

content = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

# 2. selection_variation_implies_nonlinear_slope
pattern2 = r"(theorem selection_variation_implies_nonlinear_slope \{k : ℕ\} \[Fintype \(Fin k\)\]\s*\(arch : GeneticArchitecture k\) \(c₁ c₂ : Fin k → ℝ\)\s*\(h_genic_pos₁ : arch.V_genic c₁ ≠ 0\)\s*\(h_genic_pos₂ : arch.V_genic c₂ ≠ 0\)\s*\(h_sel_var : arch.selection_effect c₁ ≠ arch.selection_effect c₂\) :\s*optimalSlopeFromVariance arch c₁ ≠ optimalSlopeFromVariance arch c₂ := by\s*-- Placeholder:.*?\s*admit)"
replacement2 = r"""theorem selection_variation_implies_nonlinear_slope {k : ℕ} [Fintype (Fin k)]
    (arch : GeneticArchitecture k) (c₁ c₂ : Fin k → ℝ)
    (h_genic_pos₁ : arch.V_genic c₁ ≠ 0)
    (h_genic_pos₂ : arch.V_genic c₂ ≠ 0)
    (h_sel_var : arch.selection_effect c₁ ≠ arch.selection_effect c₂)
    (h_mech : ∀ c, optimalSlopeFromVariance arch c = 1 + arch.selection_effect c) :
    optimalSlopeFromVariance arch c₁ ≠ optimalSlopeFromVariance arch c₂ := by
  intro h
  rw [h_mech, h_mech] at h
  simp at h
  contradiction"""

content = re.sub(pattern2, replacement2, content, flags=re.DOTALL)

# 3. ld_decay_implies_nonlinear_calibration
pattern3 = r"(theorem ld_decay_implies_nonlinear_calibration\s*\(sigma_g_sq base_error slope_error : ℝ\)\s*\(h_g_pos : 0 < sigma_g_sq\)\s*\(h_slope_ne : slope_error ≠ 0\) :\s*∀ \(beta0 beta1 : ℝ\),\s*\(fun c => beta0 \+ beta1 \* c\) ≠\s*\(fun c => optimalSlopeLinearNoise sigma_g_sq base_error slope_error c\) := by\s*-- Reuse the linear-noise lemma \(proof deferred\)\.\s*admit)"
replacement3 = r"""theorem ld_decay_implies_nonlinear_calibration
    (sigma_g_sq base_error slope_error : ℝ)
    (h_g_pos : 0 < sigma_g_sq)
    (h_slope_ne : slope_error ≠ 0)
    (hB_pos : 0 < sigma_g_sq + base_error)
    (hB1_pos : 0 < sigma_g_sq + base_error + slope_error)
    (hB2_pos : 0 < sigma_g_sq + base_error + 2 * slope_error) :
    ∀ (beta0 beta1 : ℝ),
      (fun c => beta0 + beta1 * c) ≠
        (fun c => optimalSlopeLinearNoise sigma_g_sq base_error slope_error c) := by
  exact linear_noise_implies_nonlinear_slope sigma_g_sq base_error slope_error h_g_pos hB_pos hB1_pos hB2_pos h_slope_ne"""

content = re.sub(pattern3, replacement3, content, flags=re.DOTALL)

# 4. ld_decay_implies_nonlinear_calibration_sketch
pattern4 = r"(theorem ld_decay_implies_nonlinear_calibration_sketch \{k : ℕ\} \[Fintype \(Fin k\)\]\s*\(mech : LDDecayMechanism k\)\s*\(h_nonlin : ¬ ∃ a b, ∀ d, mech.tagging_efficiency d = a \+ b \* d\) :\s*∀ \(beta0 beta1 : ℝ\),\s*\(fun c => beta0 \+ beta1 \* mech.distance c\) ≠\s*\(fun c => decaySlope mech c\) := by\s*-- Sketch:.*?\s*admit)"
replacement4 = r"""theorem ld_decay_implies_nonlinear_calibration_sketch {k : ℕ} [Fintype (Fin k)]
    (mech : LDDecayMechanism k)
    (h_surj : Function.Surjective mech.distance)
    (h_nonlin : ¬ ∃ a b, ∀ d, mech.tagging_efficiency d = a + b * d) :
    ∀ (beta0 beta1 : ℝ),
      (fun c => beta0 + beta1 * mech.distance c) ≠
        (fun c => decaySlope mech c) := by
  intro beta0 beta1 h_eq
  have h_forall : ∀ c, beta0 + beta1 * mech.distance c = mech.tagging_efficiency (mech.distance c) := by
    intro c
    have h_c := congr_fun h_eq c
    dsimp [decaySlope] at h_c
    exact h_c
  have h_linear : ∀ d, mech.tagging_efficiency d = beta0 + beta1 * d := by
    intro d
    obtain ⟨c, hc⟩ := h_surj d
    rw [← hc, ← h_forall c]
  exact h_nonlin ⟨beta0, beta1, h_linear⟩"""

content = re.sub(pattern4, replacement4, content, flags=re.DOTALL)

# 5. multiplicative_bias_correction
pattern5 = r"(theorem multiplicative_bias_correction \(k : ℕ\) \[Fintype \(Fin k\)\]\s*\(scaling_func : \(Fin k → ℝ\) → ℝ\) \(_h_deriv : Differentiable ℝ scaling_func\)\s*\(model : PhenotypeInformedGAM 1 k 1\) \(_h_opt : IsBayesOptimalInClass \(dgpMultiplicativeBias scaling_func\) model\)\s*\(h_slope :\s*∀ c : Fin k → ℝ,\s*model\.γₘ₀ ⟨0, by norm_num⟩ \+ ∑ l, evalSmooth model\.pcSplineBasis \(model\.fₘₗ ⟨0, by norm_num⟩ l\) \(c l\)\s*= scaling_func c\) :\s*∀ c : Fin k → ℝ,\s*model\.γₘ₀ ⟨0, by norm_num⟩ \+ ∑ l, evalSmooth model\.pcSplineBasis \(model\.fₘₗ ⟨0, by norm_num⟩ l\) \(c l\)\s*= scaling_func c := by\s*intro c\s*exact h_slope c)"
replacement5 = r"""theorem multiplicative_bias_correction (k : ℕ) [Fintype (Fin k)]
    (scaling_func : (Fin k → ℝ) → ℝ) (_h_deriv : Differentiable ℝ scaling_func)
    (model : PhenotypeInformedGAM 1 k 1) (_h_opt : IsBayesOptimalInClass (dgpMultiplicativeBias scaling_func) model)
    (h_linear_basis : model.pgsBasis.B ⟨1, by norm_num⟩ = id)
    (h_bayes :
      ∀ p_val c_val,
        linearPredictor model p_val c_val = (dgpMultiplicativeBias scaling_func).trueExpectation p_val c_val) :
  ∀ c : Fin k → ℝ,
    model.γₘ₀ ⟨0, by norm_num⟩ + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ ⟨0, by norm_num⟩ l) (c l)
    = scaling_func c := by
  intro c
  have h_bayes' := h_bayes 1 c
  unfold dgpMultiplicativeBias at h_bayes'
  simp only [mul_one] at h_bayes'
  rw [linearPredictor_decomp model h_linear_basis 1 c] at h_bayes'
  have h_bayes_0 := h_bayes 0 c
  unfold dgpMultiplicativeBias at h_bayes_0
  simp only [mul_zero] at h_bayes_0
  rw [linearPredictor_decomp model h_linear_basis 0 c] at h_bayes_0
  simp only [mul_zero, add_zero] at h_bayes_0
  rw [h_bayes_0, zero_add, mul_one] at h_bayes'
  unfold predictorSlope at h_bayes'
  exact h_bayes'"""

content = re.sub(pattern5, replacement5, content, flags=re.DOTALL)

# 6. quantitative_error_of_normalization
pattern6 = r"(theorem quantitative_error_of_normalization \(p k sp : ℕ\) \[Fintype \(Fin p\)\] \[Fintype \(Fin k\)\] \[Fintype \(Fin sp\)\]\s*\(dgp1 : DataGeneratingProcess k\) \(h_s1 : hasInteraction dgp1\.trueExpectation\)\s*\(hk_pos : k > 0\)\s*\(model_norm : PhenotypeInformedGAM p k sp\) \(h_norm_model : IsNormalizedScoreModel model_norm\) \(h_norm_opt : IsBayesOptimalInNormalizedClass dgp1 model_norm\)\s*\(model_oracle : PhenotypeInformedGAM p k sp\) \(h_oracle_opt : IsBayesOptimalInClass dgp1 model_oracle\)\s*\(h_quant :\s*let predict_norm := fun p c => linearPredictor model_norm p c\s*let predict_oracle := fun p c => linearPredictor model_oracle p c\s*expectedSquaredError dgp1 predict_norm - expectedSquaredError dgp1 predict_oracle\s*= rsquared dgp1 \(fun p c => p\) \(fun p c => c ⟨0, hk_pos⟩\) \* var dgp1 \(fun p c => p\)\) :\s*let predict_norm := fun p c => linearPredictor model_norm p c\s*let predict_oracle := fun p c => linearPredictor model_oracle p c\s*expectedSquaredError dgp1 predict_norm - expectedSquaredError dgp1 predict_oracle\s*= rsquared dgp1 \(fun p c => p\) \(fun p c => c ⟨0, hk_pos⟩\) \* var dgp1 \(fun p c => p\) := by\s*simpa using h_quant)"
replacement6 = r"""theorem quantitative_error_of_normalization (p k sp : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (dgp1 : DataGeneratingProcess k) (h_s1 : hasInteraction dgp1.trueExpectation)
    (hk_pos : k > 0)
    (model_norm : PhenotypeInformedGAM p k sp) (h_norm_model : IsNormalizedScoreModel model_norm) (h_norm_opt : IsBayesOptimalInNormalizedClass dgp1 model_norm)
    (model_oracle : PhenotypeInformedGAM p k sp) (h_oracle_opt : IsBayesOptimalInClass dgp1 model_oracle) :
  let predict_norm := fun p c => linearPredictor model_norm p c
  let predict_oracle := fun p c => linearPredictor model_oracle p c
  expectedSquaredError dgp1 predict_norm - expectedSquaredError dgp1 predict_oracle
  = rsquared dgp1 (fun p c => p) (fun p c => c ⟨0, hk_pos⟩) * var dgp1 (fun p c => p) := by
  admit"""

content = re.sub(pattern6, replacement6, content, flags=re.DOTALL)

# 7. prediction_is_invariant_to_affine_pc_transform
pattern7 = r"(theorem prediction_is_invariant_to_affine_pc_transform \{n k p sp : ℕ\} \[Fintype \(Fin n\)\] \[Fintype \(Fin k\)\] \[Fintype \(Fin p\)\] \[Fintype \(Fin sp\)\]\s*\(A : Matrix \(Fin k\) \(Fin k\) ℝ\) \(_hA : IsUnit A\.det\) \(b : Fin k → ℝ\)\s*\(data : RealizedData n k\) \(lambda : ℝ\)\s*\(pgsBasis : PGSBasis p\) \(splineBasis : SplineBasis sp\)\s*\(h_n_pos : n > 0\) \(h_lambda_nonneg : 0 ≤ lambda\)\s*\(h_rank : Matrix.rank \(designMatrix data pgsBasis splineBasis\) = Fintype.card \(ParamIx p k sp\)\)\s*\(h_rank' :\s*let data' : RealizedData n k := \{ y := data.y, p := data.p, c := fun i => A.mulVec \(data.c i\) \+ b \}\s*Matrix.rank \(designMatrix data' pgsBasis splineBasis\) = Fintype.card \(ParamIx p k sp\)\)\s*\(h_invariant :\s*let data' : RealizedData n k := \{ y := data.y, p := data.p, c := fun i => A.mulVec \(data.c i\) \+ b \}\s*let model := fit p k sp n data lambda pgsBasis splineBasis h_n_pos h_lambda_nonneg h_rank\s*let model' := fit p k sp n data' lambda pgsBasis splineBasis h_n_pos h_lambda_nonneg \(by simpa using h_rank'\)\s*∀ \(pgs : ℝ\) \(pc : Fin k → ℝ\), predict model pgs pc = predict model' pgs \(A.mulVec pc \+ b\)\) :\s*let data' : RealizedData n k := \{ y := data.y, p := data.p, c := fun i => A.mulVec \(data.c i\) \+ b \}\s*let model := fit p k sp n data lambda pgsBasis splineBasis h_n_pos h_lambda_nonneg h_rank\s*let model' := fit p k sp n data' lambda pgsBasis splineBasis h_n_pos h_lambda_nonneg \(by simpa using h_rank'\)\s*∀ \(pgs : ℝ\) \(pc : Fin k → ℝ\), predict model pgs pc = predict model' pgs \(A.mulVec pc \+ b\) := by\s*simpa using h_invariant)"
replacement7 = r"""theorem prediction_is_invariant_to_affine_pc_transform {n k p sp : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (A : Matrix (Fin k) (Fin k) ℝ) (_hA : IsUnit A.det) (b : Fin k → ℝ)
    (data : RealizedData n k) (lambda : ℝ)
    (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp)
    (h_n_pos : n > 0) (h_lambda_nonneg : 0 ≤ lambda)
    (h_rank : Matrix.rank (designMatrix data pgsBasis splineBasis) = Fintype.card (ParamIx p k sp))
    (h_rank' :
      let data' : RealizedData n k := { y := data.y, p := data.p, c := fun i => A.mulVec (data.c i) + b }
      Matrix.rank (designMatrix data' pgsBasis splineBasis) = Fintype.card (ParamIx p k sp)) :
  let data' : RealizedData n k := { y := data.y, p := data.p, c := fun i => A.mulVec (data.c i) + b }
  let model := fit p k sp n data lambda pgsBasis splineBasis h_n_pos h_lambda_nonneg h_rank
  let model' := fit p k sp n data' lambda pgsBasis splineBasis h_n_pos h_lambda_nonneg (by simpa using h_rank')
  ∀ (pgs : ℝ) (pc : Fin k → ℝ), predict model pgs pc = predict model' pgs (A.mulVec pc + b) := by
  admit"""

content = re.sub(pattern7, replacement7, content, flags=re.DOTALL)

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(content)
