import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

theorem expectedBrierScore_deriv_proved (π : ℝ) :
    deriv (fun p => expectedBrierScore p π) = fun p => 2 * (p - π) := by
  ext p
  unfold expectedBrierScore
  have h_expand : (fun p => π * (1 - p) ^ 2 + (1 - π) * (0 - p) ^ 2) = fun p => p ^ 2 - 2 * π * p + π := by
    ext x
    ring
  rw [h_expand]
  have hd1 : DifferentiableAt ℝ (fun p => p ^ 2) p := differentiableAt_id.pow 2
  have hd2 : DifferentiableAt ℝ (fun p => 2 * π * p) p := differentiableAt_id.const_mul (2 * π)
  have hd3 : DifferentiableAt ℝ (fun p => (fun p => p ^ 2) p - (fun p => 2 * π * p) p) p := DifferentiableAt.sub hd1 hd2
  rw [deriv_add hd3 (differentiableAt_const π)]
  rw [deriv_sub hd1 hd2]
  rw [deriv_pow 2 differentiableAt_id]
  rw [deriv_const_mul (2 * π) differentiableAt_id]
  rw [deriv_id, deriv_const]
  ring

end Calibrator
