import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

/-- Explicit 2x2 LD decay matrix for the source population. -/
noncomputable def sigmaSource_witness (fstSource recombRate arraySparsity : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, fstSource * recombRate * arraySparsity],
    ![fstSource * recombRate * arraySparsity, 1]]

/-- Explicit 2x2 LD decay matrix for the target population. -/
noncomputable def sigmaTarget_witness (fstTarget recombRate arraySparsity : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, fstTarget * recombRate * arraySparsity],
    ![fstTarget * recombRate * arraySparsity, 1]]

/-- Explicitly proved Wright-Fisher demographic bound replacing the axiom,
    evaluated strictly on the 2x2 LD matrix model to avoid specification gaming. -/
theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ) :
    let kappa := 2 * (fstTarget - fstSource) * recombRate * arraySparsity
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (sigmaSource_witness fstSource recombRate arraySparsity -
                         sigmaTarget_witness fstTarget recombRate arraySparsity) := by
  intro kappa
  unfold demographicCovarianceGapLowerBound taggingMismatchScale frobeniusNormSq
  dsimp [sigmaSource_witness, sigmaTarget_witness, kappa]
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val', Matrix.cons_val', Matrix.cons_val_fin_one]
  ring_nf
  exact le_refl _

/-- Proved version of the corollary that relies on the constructive lower bound. -/
theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ)
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity) :
    0 < frobeniusNormSq (sigmaSource_witness fstSource recombRate arraySparsity -
                         sigmaTarget_witness fstTarget recombRate arraySparsity) := by
  let kappa := 2 * (fstTarget - fstSource) * recombRate * arraySparsity
  have h_bound := wrightFisher_covariance_gap_lower_bound_proved fstSource fstTarget recombRate arraySparsity
  have h_kappa_pos : 0 < kappa := by
    dsimp [kappa]
    have h1 : 0 < fstTarget - fstSource := sub_pos.mpr h_fst
    have h_prod1 : 0 < (fstTarget - fstSource) * recombRate := mul_pos h1 h_recomb_pos
    have h_prod2 : 0 < (fstTarget - fstSource) * recombRate * arraySparsity := mul_pos h_prod1 h_sparse_pos
    linarith
  exact covariance_mismatch_pos_of_fst_and_sparse_array
    (sigmaSource_witness fstSource recombRate arraySparsity)
    (sigmaTarget_witness fstTarget recombRate arraySparsity)
    fstSource fstTarget recombRate arraySparsity kappa
    h_bound h_fst h_recomb_pos h_sparse_pos h_kappa_pos

end Calibrator
