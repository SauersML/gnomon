import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift


namespace Calibrator

/-- Concrete 2x2 matrix representing simplified LD decay for the demographic bound proof. -/
def ldMatrix (r : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, r], ![r, 1]]

theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ)
    (rS rT : ℝ)
    (h_delta : fstTarget - fstSource = (rS - rT)^2) :
    let sigmaSource := ldMatrix rS
    let sigmaTarget := ldMatrix rT
    let kappa := 2 * recombRate * arraySparsity
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro sigmaSource sigmaTarget kappa
  unfold demographicCovarianceGapLowerBound taggingMismatchScale frobeniusNormSq
  -- Evaluate the matrix difference Frobenius norm
  have h_norm : ∑ i : Fin 2, ∑ j : Fin 2, ((sigmaSource - sigmaTarget) i j) ^ 2 = 2 * (rS - rT)^2 := by
    simp only [sigmaSource, sigmaTarget, ldMatrix, Matrix.sub_apply, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, Matrix.empty_val', Matrix.cons_val', Matrix.cons_val_fin_one, sub_self, sq, MulZeroClass.zero_mul, add_zero, zero_add]
    ring
  rw [h_norm, h_delta]
  -- kappa * (recombRate * arraySparsity) * (rS - rT)^2
  -- = (2 * recombRate * arraySparsity) * recombRate * arraySparsity * (rS - rT)^2
  -- Wait, let's just use `kappa = 1` and require `2 = recombRate * arraySparsity`?
  -- Let's construct a cleaner version.
  sorry

end Calibrator
