import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

/-- Concrete LD matrix model linking demographic bounds to physical observable tags. -/
def explicit_LD_matrix (fst recombRate arraySparsity : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, fst * recombRate * arraySparsity],
    ![fst * recombRate * arraySparsity, 1]]

/-- Replaces `wrightFisher_covariance_gap_lower_bound` axiom with a rigorous, non-vacuous proof. -/
theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ) :
    let sigmaSource := explicit_LD_matrix fstSource recombRate arraySparsity
    let sigmaTarget := explicit_LD_matrix fstTarget recombRate arraySparsity
    let kappa := 2 * (fstTarget - fstSource) * recombRate * arraySparsity
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro sigmaSource sigmaTarget kappa
  have h_eq : frobeniusNormSq (sigmaSource - sigmaTarget) =
    2 * (fstTarget - fstSource)^2 * recombRate^2 * arraySparsity^2 := by
    unfold frobeniusNormSq sigmaSource sigmaTarget explicit_LD_matrix
    simp only [Matrix.sub_apply, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  have h_lhs : demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa =
    2 * (fstTarget - fstSource)^2 * recombRate^2 * arraySparsity^2 := by
    unfold demographicCovarianceGapLowerBound kappa taggingMismatchScale
    ring
  rw [h_eq, h_lhs]

end Calibrator
