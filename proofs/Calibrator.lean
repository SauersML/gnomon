import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

/-- An explicit 2x2 LD matrix model that decays linearly with divergence and genetic distance. -/
def explicitLDMatrix (fst recombRate arraySparsity : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let cov := 1 - recombRate * arraySparsity * fst
  ![![1, cov], ![cov, 1]]

theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ) :
    let kappa := 2 * (fstTarget - fstSource) * recombRate * arraySparsity
    let sigmaSource := explicitLDMatrix fstSource recombRate arraySparsity
    let sigmaTarget := explicitLDMatrix fstTarget recombRate arraySparsity
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro kappa sigmaSource sigmaTarget
  unfold demographicCovarianceGapLowerBound taggingMismatchScale
  unfold sigmaSource sigmaTarget explicitLDMatrix frobeniusNormSq
  dsimp [kappa]
  simp only [Fin.sum_univ_two, Matrix.sub_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val', Matrix.cons_val', Matrix.cons_val_fin_one]
  ring_nf!
  exact le_refl _

theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity)
    (_h_kappa_pos : 0 < kappa) :
    let sigmaSource := explicitLDMatrix fstSource recombRate arraySparsity
    let sigmaTarget := explicitLDMatrix fstTarget recombRate arraySparsity
    let _kappa2 := 2 * (fstTarget - fstSource) * recombRate * arraySparsity
    0 < frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro sigmaSource sigmaTarget _kappa2
  have h_bound := wrightFisher_covariance_gap_lower_bound_proved fstSource fstTarget recombRate arraySparsity
  exact covariance_mismatch_pos_of_fst_and_sparse_array
    (explicitLDMatrix fstSource recombRate arraySparsity)
    (explicitLDMatrix fstTarget recombRate arraySparsity)
    fstSource fstTarget recombRate arraySparsity
    (2 * (fstTarget - fstSource) * recombRate * arraySparsity)
    h_bound h_fst h_recomb_pos h_sparse_pos
    (by
      have h1 : 0 < fstTarget - fstSource := sub_pos.mpr h_fst
      positivity
    )

theorem expected_abs_mean_shift_bound_proved
    (V_A fstS fstT : ℝ)
    (hVA_pos : 0 < V_A)
    (hfst_sum_nonneg : 0 ≤ fstS + fstT)
    (hfstS_lt_one : fstS < 1) :
    Expected_Abs_Shift V_A fstS fstT / Real.sqrt (presentDayPGSVariance V_A fstS) =
      2 * Real.sqrt ((fstS + fstT) / (Real.pi * (1 - fstS))) := by
  unfold Expected_Abs_Shift presentDayPGSVariance Var_Delta_Mu
  have h_pi_pos : 0 < Real.pi := Real.pi_pos
  have h1 : 2 * (fstS + fstT) * V_A ≥ 0 := by
    have h2 : 0 ≤ 2 * (fstS + fstT) := mul_nonneg (by positivity) hfst_sum_nonneg
    exact mul_nonneg h2 (le_of_lt hVA_pos)
  have h2 : 2 / Real.pi ≥ 0 := by positivity
  have h3 : (1 - fstS) * V_A ≥ 0 := by
    have h4 : 0 ≤ 1 - fstS := sub_nonneg.mpr (le_of_lt hfstS_lt_one)
    exact mul_nonneg h4 (le_of_lt hVA_pos)
  have hVA_ne_zero : V_A ≠ 0 := ne_of_gt hVA_pos
  have h_1_fstS_ne_zero : 1 - fstS ≠ 0 := sub_ne_zero.mpr (ne_of_lt hfstS_lt_one).symm
  have h_pi_ne_zero : Real.pi ≠ 0 := ne_of_gt h_pi_pos

  have h_sqrt_mul : Real.sqrt (2 * (fstS + fstT) * V_A) * Real.sqrt (2 / Real.pi) = Real.sqrt (2 * (fstS + fstT) * V_A * (2 / Real.pi)) := by
    rw [Real.sqrt_mul h1]

  rw [h_sqrt_mul]
  have h_sqrt_div : Real.sqrt (2 * (fstS + fstT) * V_A * (2 / Real.pi)) / Real.sqrt ((1 - fstS) * V_A) = Real.sqrt ((2 * (fstS + fstT) * V_A * (2 / Real.pi)) / ((1 - fstS) * V_A)) := by
    have h_top_nonneg : 0 ≤ 2 * (fstS + fstT) * V_A * (2 / Real.pi) := mul_nonneg h1 h2
    rw [Real.sqrt_div h_top_nonneg]
  rw [h_sqrt_div]

  have h_inside : (2 * (fstS + fstT) * V_A * (2 / Real.pi)) / ((1 - fstS) * V_A) = 2^2 * ((fstS + fstT) / (Real.pi * (1 - fstS))) := by
    have h_cross : (2 * (fstS + fstT) * V_A * (2 / Real.pi)) * ((1 - fstS) * V_A)⁻¹ = 4 * (fstS + fstT) * (Real.pi * (1 - fstS))⁻¹ := by
      calc
        (2 * (fstS + fstT) * V_A * (2 / Real.pi)) * ((1 - fstS) * V_A)⁻¹
          = 2 * (fstS + fstT) * V_A * 2 * Real.pi⁻¹ * ((1 - fstS) * V_A)⁻¹ := by ring_nf
        _ = 2 * (fstS + fstT) * V_A * 2 * Real.pi⁻¹ * (V_A⁻¹ * (1 - fstS)⁻¹) := by
            rw [mul_inv]
            ring_nf
        _ = 4 * (fstS + fstT) * Real.pi⁻¹ * (1 - fstS)⁻¹ * (V_A * V_A⁻¹) := by ring_nf
        _ = 4 * (fstS + fstT) * Real.pi⁻¹ * (1 - fstS)⁻¹ * 1 := by rw [mul_inv_cancel₀ hVA_ne_zero]
        _ = 4 * (fstS + fstT) * (Real.pi * (1 - fstS))⁻¹ := by
            rw [mul_inv]
            ring_nf

    calc
      (2 * (fstS + fstT) * V_A * (2 / Real.pi)) / ((1 - fstS) * V_A)
        = (2 * (fstS + fstT) * V_A * (2 / Real.pi)) * ((1 - fstS) * V_A)⁻¹ := by rfl
      _ = 4 * (fstS + fstT) * (Real.pi * (1 - fstS))⁻¹ := h_cross
      _ = (4 * (fstS + fstT)) / (Real.pi * (1 - fstS)) := by rfl
      _ = 2^2 * ((fstS + fstT) / (Real.pi * (1 - fstS))) := by ring

  rw [h_inside]
  rw [Real.sqrt_mul (by positivity)]
  rw [Real.sqrt_sq (by positivity)]

end Calibrator
