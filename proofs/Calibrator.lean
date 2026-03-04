import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift


namespace Calibrator
namespace PortabilityDrift

-- Fix specification gaming for demographic covariance gap
noncomputable def explicitLDMatrix (fst distance lambda : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, ldCorrelationDecay distance fst lambda],
    ![ldCorrelationDecay distance fst lambda, 1]]

theorem ldCorrelationDecay_strictAnti_fst
    (distance lambda fstSource fstTarget : ℝ)
    (hDist : 0 < distance)
    (hLambda : 0 < lambda)
    (hFst : fstSource < fstTarget) :
    ldCorrelationDecay distance fstTarget lambda < ldCorrelationDecay distance fstSource lambda := by
  unfold ldCorrelationDecay
  apply Real.exp_lt_exp.mpr
  have h_pos : 0 < lambda * distance := mul_pos hLambda hDist
  have h_lt : fstSource * (lambda * distance) < fstTarget * (lambda * distance) :=
    mul_lt_mul_of_pos_right hFst h_pos
  linarith

theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ)
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity) :
    0 < frobeniusNormSq (explicitLDMatrix fstSource arraySparsity recombRate - explicitLDMatrix fstTarget arraySparsity recombRate) := by
  have h_decay := ldCorrelationDecay_strictAnti_fst arraySparsity recombRate fstSource fstTarget h_sparse_pos h_recomb_pos h_fst
  have h_ne : ldCorrelationDecay arraySparsity fstSource recombRate - ldCorrelationDecay arraySparsity fstTarget recombRate ≠ 0 := by
    linarith
  apply frobeniusNormSq_pos_of_exists_ne_zero
  use 0, 1
  dsimp [explicitLDMatrix]
  exact h_ne

end PortabilityDrift
end Calibrator
