import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Calibrator.ValidationStatistics

namespace Calibrator

open scoped BigOperators

/-!
# Cross-Validation Theory

Abstract held-out prediction-evaluation utilities. This file is intentionally
agnostic about the biological portability model; it provides generic CV/LOPO
error objects and bias-variance comparison lemmas.
-/

section CrossValidation

structure BlockCVModel where
  k : ℕ
  hk : 0 < (k : ℝ)
  true_effect : Fin k → ℝ
  confounding_bias : ℝ
  pred_simple : Fin k → ℝ
  h_pred_simple : ∀ i, pred_simple i = true_effect i - confounding_bias
  pred_rich : Fin k → ℝ
  variance_noise : Fin k → ℝ
  h_pred_rich : ∀ i, pred_rich i = true_effect i + variance_noise i
  observed : Fin k → ℝ
  noise : Fin k → ℝ
  h_observed : ∀ i, observed i = true_effect i + noise i
  -- cross terms are 0
  h_cross_simple : ∑ i : Fin k, confounding_bias * noise i = 0
  h_cross_rich : ∑ i : Fin k, variance_noise i * noise i = 0


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

/-- A simpler model wins under LOPO evaluation when the variance penalty of the
more flexible model dominates its squared-bias improvement. -/
theorem lopo_prefers_simpler_model_of_variance_gap_dominates_bias_gain
    (m : BlockCVModel)
    (hDom : (∑ i : Fin m.k, m.variance_noise i ^ 2) > m.k * m.confounding_bias ^ 2) :
    cvPredictionError m.pred_simple m.observed <
      cvPredictionError m.pred_rich m.observed := by
  unfold cvPredictionError
  have h_div : (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2) / m.k < (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2) / m.k ↔ (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2) < (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2) := by
    exact div_lt_div_iff_of_pos_right m.hk
  rw [h_div]

  have h_simple : (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2) = m.k * m.confounding_bias ^ 2 + ∑ i : Fin m.k, m.noise i ^ 2 := by
    calc
      (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2)
        = ∑ i : Fin m.k, (m.true_effect i - m.confounding_bias - (m.true_effect i + m.noise i)) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          rw [m.h_pred_simple, m.h_observed]
      _ = ∑ i : Fin m.k, (-m.confounding_bias - m.noise i) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = ∑ i : Fin m.k, (m.confounding_bias ^ 2 + 2 * (m.confounding_bias * m.noise i) + m.noise i ^ 2) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = (∑ i : Fin m.k, m.confounding_bias ^ 2) + (∑ i : Fin m.k, 2 * (m.confounding_bias * m.noise i)) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
      _ = (∑ i : Fin m.k, m.confounding_bias ^ 2) + 2 * (∑ i : Fin m.k, m.confounding_bias * m.noise i) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [←Finset.mul_sum]
      _ = m.k * m.confounding_bias ^ 2 + ∑ i : Fin m.k, m.noise i ^ 2 := by
          rw [m.h_cross_simple]
          simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring

  have h_rich : (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2) = ∑ i : Fin m.k, m.variance_noise i ^ 2 + ∑ i : Fin m.k, m.noise i ^ 2 := by
    calc
      (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2)
        = ∑ i : Fin m.k, (m.true_effect i + m.variance_noise i - (m.true_effect i + m.noise i)) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          rw [m.h_pred_rich, m.h_observed]
      _ = ∑ i : Fin m.k, (m.variance_noise i - m.noise i) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = ∑ i : Fin m.k, (m.variance_noise i ^ 2 + -2 * (m.variance_noise i * m.noise i) + m.noise i ^ 2) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = (∑ i : Fin m.k, m.variance_noise i ^ 2) + (∑ i : Fin m.k, -2 * (m.variance_noise i * m.noise i)) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
      _ = (∑ i : Fin m.k, m.variance_noise i ^ 2) - 2 * (∑ i : Fin m.k, m.variance_noise i * m.noise i) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [←Finset.mul_sum]
          ring
      _ = (∑ i : Fin m.k, m.variance_noise i ^ 2) + ∑ i : Fin m.k, m.noise i ^ 2 := by
          rw [m.h_cross_rich]
          ring

  rw [h_simple, h_rich]
  linarith

/-- A richer model wins under LOPO evaluation when its squared-bias reduction
dominates the variance penalty it pays. -/
theorem lopo_prefers_richer_model_of_bias_gain_dominates_variance_cost
    (m : BlockCVModel)
    (hDom : m.k * m.confounding_bias ^ 2 > (∑ i : Fin m.k, m.variance_noise i ^ 2)) :
    cvPredictionError m.pred_rich m.observed <
      cvPredictionError m.pred_simple m.observed := by
  unfold cvPredictionError
  have h_div : (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2) / m.k < (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2) / m.k ↔ (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2) < (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2) := by
    exact div_lt_div_iff_of_pos_right m.hk
  rw [h_div]

  have h_simple : (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2) = m.k * m.confounding_bias ^ 2 + ∑ i : Fin m.k, m.noise i ^ 2 := by
    calc
      (∑ i : Fin m.k, (m.pred_simple i - m.observed i) ^ 2)
        = ∑ i : Fin m.k, (m.true_effect i - m.confounding_bias - (m.true_effect i + m.noise i)) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          rw [m.h_pred_simple, m.h_observed]
      _ = ∑ i : Fin m.k, (-m.confounding_bias - m.noise i) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = ∑ i : Fin m.k, (m.confounding_bias ^ 2 + 2 * (m.confounding_bias * m.noise i) + m.noise i ^ 2) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = (∑ i : Fin m.k, m.confounding_bias ^ 2) + (∑ i : Fin m.k, 2 * (m.confounding_bias * m.noise i)) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
      _ = (∑ i : Fin m.k, m.confounding_bias ^ 2) + 2 * (∑ i : Fin m.k, m.confounding_bias * m.noise i) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [←Finset.mul_sum]
      _ = m.k * m.confounding_bias ^ 2 + ∑ i : Fin m.k, m.noise i ^ 2 := by
          rw [m.h_cross_simple]
          simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring

  have h_rich : (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2) = ∑ i : Fin m.k, m.variance_noise i ^ 2 + ∑ i : Fin m.k, m.noise i ^ 2 := by
    calc
      (∑ i : Fin m.k, (m.pred_rich i - m.observed i) ^ 2)
        = ∑ i : Fin m.k, (m.true_effect i + m.variance_noise i - (m.true_effect i + m.noise i)) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          rw [m.h_pred_rich, m.h_observed]
      _ = ∑ i : Fin m.k, (m.variance_noise i - m.noise i) ^ 2 := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = ∑ i : Fin m.k, (m.variance_noise i ^ 2 + -2 * (m.variance_noise i * m.noise i) + m.noise i ^ 2) := by
          apply Finset.sum_congr rfl
          intro i _
          ring
      _ = (∑ i : Fin m.k, m.variance_noise i ^ 2) + (∑ i : Fin m.k, -2 * (m.variance_noise i * m.noise i)) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
      _ = (∑ i : Fin m.k, m.variance_noise i ^ 2) - 2 * (∑ i : Fin m.k, m.variance_noise i * m.noise i) + (∑ i : Fin m.k, m.noise i ^ 2) := by
          rw [←Finset.mul_sum]
          ring
      _ = (∑ i : Fin m.k, m.variance_noise i ^ 2) + ∑ i : Fin m.k, m.noise i ^ 2 := by
          rw [m.h_cross_rich]
          ring

  rw [h_simple, h_rich]
  linarith

end CrossValidation

end Calibrator
