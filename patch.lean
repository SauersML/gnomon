import Mathlib.Tactic
import Mathlib.Data.Matrix.Basic
open Matrix

noncomputable def taggingMismatchScale (r a : ℝ) : ℝ := r * a

noncomputable def demographicCovarianceGapLowerBound (fs ft r a k : ℝ) : ℝ :=
  k * taggingMismatchScale r a * (ft - fs)

noncomputable def wrightFisherLDMatrix (fst recombRate arraySparsity : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let d := taggingMismatchScale recombRate arraySparsity * fst
  ![![1, d], ![d, 1]]

noncomputable def frobeniusNormSq {t : ℕ} (A : Matrix (Fin t) (Fin t) ℝ) : ℝ :=
  ∑ i : Fin t, ∑ j : Fin t, (A i j) ^ 2

theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ)
    (kappa : ℝ)
    (h_kappa : kappa = 2 * taggingMismatchScale recombRate arraySparsity * (fstTarget - fstSource)) :
    let sigmaSource := wrightFisherLDMatrix fstSource recombRate arraySparsity
    let sigmaTarget := wrightFisherLDMatrix fstTarget recombRate arraySparsity
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro sigmaSource sigmaTarget
  unfold demographicCovarianceGapLowerBound sigmaSource sigmaTarget wrightFisherLDMatrix frobeniusNormSq
  simp only [Fin.sum_univ_two, Matrix.sub_apply, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val', Matrix.cons_val', Matrix.cons_val_fin_one]
  ring_nf
  rw [h_kappa]
  ring_nf
  nlinarith

theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity)
    (h_kappa : kappa = 2 * taggingMismatchScale recombRate arraySparsity * (fstTarget - fstSource)) :
    let sigmaSource := wrightFisherLDMatrix fstSource recombRate arraySparsity
    let sigmaTarget := wrightFisherLDMatrix fstTarget recombRate arraySparsity
    0 < frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro sigmaSource sigmaTarget
  have h_bound := wrightFisher_covariance_gap_lower_bound_proved fstSource fstTarget recombRate arraySparsity kappa h_kappa
  have h_lb_pos : 0 < demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa := by
    unfold demographicCovarianceGapLowerBound taggingMismatchScale
    rw [h_kappa]
    unfold taggingMismatchScale
    have h1 : 0 < recombRate * arraySparsity := mul_pos h_recomb_pos h_sparse_pos
    have h2 : 0 < fstTarget - fstSource := sub_pos.mpr h_fst
    have h3 : 0 < 2 * (recombRate * arraySparsity) * (fstTarget - fstSource) := by
      exact mul_pos (mul_pos (by norm_num) h1) h2
    nlinarith
  exact lt_of_lt_of_le h_lb_pos h_bound
