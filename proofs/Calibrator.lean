import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

def LDMatrix (fst recomb sparse : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, fst * recomb * sparse; fst * recomb * sparse, 1]

theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ) :
    let kappa := 2 * (fstTarget - fstSource) * recombRate * arraySparsity
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (LDMatrix fstSource recombRate arraySparsity - LDMatrix fstTarget recombRate arraySparsity) := by
  intro kappa
  unfold demographicCovarianceGapLowerBound taggingMismatchScale LDMatrix frobeniusNormSq
  dsimp [Matrix.sub_apply]
  simp [Fin.sum_univ_two]
  ring_nf!
  exact le_rfl

end Calibrator
