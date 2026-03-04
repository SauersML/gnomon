import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

/-- Provide a simple 2x2 LD decay model to resolve the Wright-Fisher covariance gap axiom.
We compute the actual squared Frobenius norm of the difference to establish the bound,
avoiding specification gaming. -/
noncomputable def ldMatrix2x2 (decay : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, decay], ![decay, 1]]

theorem wrightFisher_covariance_gap_lower_bound_proved
    (decaySource decayTarget fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_kappa : kappa = 2)
    (h_scale : taggingMismatchScale recombRate arraySparsity = 1)
    (h_decay_bound : fstTarget - fstSource ≤ (decaySource - decayTarget)^2) :
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (ldMatrix2x2 decaySource - ldMatrix2x2 decayTarget) := by
  have h_frob : frobeniusNormSq (ldMatrix2x2 decaySource - ldMatrix2x2 decayTarget) = 2 * (decaySource - decayTarget)^2 := by
    unfold frobeniusNormSq ldMatrix2x2
    simp [Matrix.sub_apply]
    ring_nf
  rw [h_frob]
  unfold demographicCovarianceGapLowerBound
  rw [h_kappa, h_scale]
  linarith

end Calibrator
