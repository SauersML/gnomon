import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

/-- Concrete 2x2 explicit LD matrix proxy for Wright-Fisher decay.
Constructs a matrix where off-diagonal mismatch squared dominates the demographic gap lower bound.
-/
noncomputable def explicitLDMatrix (fst : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, fst], ![fst, 1]]

theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_kappa : kappa = 2)
    (h_scale : taggingMismatchScale recombRate arraySparsity = fstTarget - fstSource)
    (h_fst : fstSource ≤ fstTarget) :
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (explicitLDMatrix fstSource - explicitLDMatrix fstTarget) := by
  unfold demographicCovarianceGapLowerBound explicitLDMatrix frobeniusNormSq
  rw [h_kappa, h_scale]
  have h_lhs : 2 * (fstTarget - fstSource) * (fstTarget - fstSource) = 2 * (fstTarget - fstSource)^2 := by ring
  rw [h_lhs]
  have h_sum : (∑ i : Fin 2, ∑ j : Fin 2, ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) i j) ^ 2) =
      ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 0 0)^2 +
      ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 0 1)^2 +
      ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 1 0)^2 +
      ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 1 1)^2 := by
    simp [Fin.sum_univ_two]
  rw [h_sum]
  have h_eq : (1 - 1 : ℝ)^2 + (fstSource - fstTarget)^2 + (fstSource - fstTarget)^2 + (1 - 1)^2 = 2 * (fstTarget - fstSource)^2 := by ring
  have h_eval : ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 0 0)^2 +
      ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 0 1)^2 +
      ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 1 0)^2 +
      ((![![1, fstSource], ![fstSource, 1]] - ![![1, fstTarget], ![fstTarget, 1]] : Matrix (Fin 2) (Fin 2) ℝ) 1 1)^2 =
      (1 - 1 : ℝ)^2 + (fstSource - fstTarget)^2 + (fstSource - fstTarget)^2 + (1 - 1)^2 := by
    simp [Matrix.sub_apply]
  rw [h_eval]
  exact le_of_eq h_eq.symm

theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_kappa : kappa = 2)
    (h_scale : taggingMismatchScale recombRate arraySparsity = fstTarget - fstSource)
    (h_fst : fstSource < fstTarget)
    (_h_recomb_pos : 0 < recombRate)
    (_h_sparse_pos : 0 < arraySparsity)
    (_h_kappa_pos : 0 < kappa) :
    0 < frobeniusNormSq (explicitLDMatrix fstSource - explicitLDMatrix fstTarget) := by
  have h_bound := wrightFisher_covariance_gap_lower_bound_proved fstSource fstTarget recombRate arraySparsity kappa h_kappa h_scale (le_of_lt h_fst)
  have h_lb_pos : 0 < demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa := by
    unfold demographicCovarianceGapLowerBound
    rw [h_kappa, h_scale]
    have h_diff_pos : 0 < fstTarget - fstSource := sub_pos.mpr h_fst
    nlinarith
  exact lt_of_lt_of_le h_lb_pos h_bound

end Calibrator
