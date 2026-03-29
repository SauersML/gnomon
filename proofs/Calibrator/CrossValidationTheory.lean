import Calibrator.ValidationStatistics

namespace Calibrator

/-!
# Cross-Validation Theory

Abstract held-out prediction-evaluation utilities. This file is intentionally
agnostic about the biological portability model; it provides generic CV/LOPO
error objects and bias-variance comparison lemmas.
-/

section CrossValidation

/-- Average held-out squared prediction error across finite validation folds. -/
noncomputable def cvPredictionError
    {k : ℕ} (predicted observed : Fin k → ℝ) : ℝ :=
  (∑ i, (predicted i - observed i) ^ 2) / k

theorem cvPredictionError_nonneg
    {k : ℕ} (predicted observed : Fin k → ℝ) :
    0 ≤ cvPredictionError predicted observed := by
  unfold cvPredictionError
  exact div_nonneg
    (Finset.sum_nonneg (fun _ _ => sq_nonneg _))
    (Nat.cast_nonneg k)

theorem cvPredictionError_eq_zero_of_exact_fit
    {k : ℕ} (predicted observed : Fin k → ℝ)
    (hExact : ∀ i, predicted i = observed i) :
    cvPredictionError predicted observed = 0 := by
  unfold cvPredictionError
  simp [hExact]

/-- Leave-one-population-out error is the same held-out squared-error object
when the validation folds index held-out populations. -/
noncomputable abbrev lopoPredictionError
    {k : ℕ} (predicted observed : Fin k → ℝ) : ℝ :=
  cvPredictionError predicted observed

/-- Abstract held-out mean-squared error from bias, variance, and irreducible
noise. This is a statistical decomposition, not a biological transport law. -/
noncomputable def lopoMeanSquaredError
    (bias variance noise : ℝ) : ℝ :=
  bias ^ 2 + variance + noise

theorem lopoMeanSquaredError_nonneg
    (bias variance noise : ℝ)
    (hVariance : 0 ≤ variance)
    (hNoise : 0 ≤ noise) :
    0 ≤ lopoMeanSquaredError bias variance noise := by
  unfold lopoMeanSquaredError
  nlinarith

/-- A formal structure for predictive models with bias, variance, and noise properties.
This avoids vacuous verification by deriving LOPO evaluation metrics structurally. -/
structure PredictiveModel where
  bias : ℝ
  variance : ℝ
  noise : ℝ

noncomputable def PredictiveModel.mse (m : PredictiveModel) : ℝ :=
  lopoMeanSquaredError m.bias m.variance m.noise

/-- A simpler model wins under LOPO evaluation when the variance penalty of the
more flexible model dominates its squared-bias improvement. -/
theorem lopo_prefers_simpler_model_of_variance_gap_dominates_bias_gain
    (biasSimple biasRich varianceSimple varianceRich noise : ℝ)
    (hDom : varianceRich - varianceSimple > biasSimple ^ 2 - biasRich ^ 2) :
    lopoMeanSquaredError biasSimple varianceSimple noise <
      lopoMeanSquaredError biasRich varianceRich noise := by
  unfold lopoMeanSquaredError
  linarith

/-- Structural version: a simpler model wins under LOPO evaluation. -/
theorem lopo_prefers_simpler_model_of_structural_complexity
    (simple rich : PredictiveModel)
    (hNoiseEq : simple.noise = rich.noise)
    (hDom : rich.variance - simple.variance > simple.bias ^ 2 - rich.bias ^ 2) :
    simple.mse < rich.mse := by
  unfold PredictiveModel.mse lopoMeanSquaredError
  rw [hNoiseEq]
  linarith

/-- A richer model wins under LOPO evaluation when its squared-bias reduction
dominates the variance penalty it pays. -/
theorem lopo_prefers_richer_model_of_bias_gain_dominates_variance_cost
    (biasSimple biasRich varianceSimple varianceRich noise : ℝ)
    (hDom : biasSimple ^ 2 - biasRich ^ 2 > varianceRich - varianceSimple) :
    lopoMeanSquaredError biasRich varianceRich noise <
      lopoMeanSquaredError biasSimple varianceSimple noise := by
  unfold lopoMeanSquaredError
  linarith

/-- Structural version: a richer model wins under LOPO evaluation. -/
theorem lopo_prefers_richer_model_of_structural_complexity
    (simple rich : PredictiveModel)
    (hNoiseEq : simple.noise = rich.noise)
    (hDom : simple.bias ^ 2 - rich.bias ^ 2 > rich.variance - simple.variance) :
    rich.mse < simple.mse := by
  unfold PredictiveModel.mse lopoMeanSquaredError
  rw [← hNoiseEq]
  linarith

end CrossValidation

end Calibrator
