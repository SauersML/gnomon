import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

theorem expectedBrierScore_deriv_proved (p π : ℝ) :
    deriv (fun x => expectedBrierScore x π) p = 2 * (p - π) := by
  have : (fun x => expectedBrierScore x π) = fun x => π - 2 * π * x + x ^ 2 := by
    ext x
    exact expectedBrierScore_quadratic x π
  rw [this]
  have h_add : deriv (fun x => (π - 2 * π * x) + x ^ 2) p = deriv (fun x => π - 2 * π * x) p + deriv (fun x => x ^ 2) p := by
    apply deriv_add
    · apply DifferentiableAt.sub
      · apply differentiableAt_const
      · exact DifferentiableAt.mul (differentiableAt_const _) differentiableAt_id
    · apply DifferentiableAt.pow; exact differentiableAt_id
  rw [h_add]
  have h_sub : deriv (fun x => π - 2 * π * x) p = deriv (fun x => π) p - deriv (fun x => 2 * π * x) p := by
    apply deriv_sub
    · apply differentiableAt_const
    · exact DifferentiableAt.mul (differentiableAt_const _) differentiableAt_id
  rw [h_sub]
  rw [deriv_const]
  have h_mul : deriv (fun x => 2 * π * x) p = 2 * π * deriv (fun x => x) p := by
    exact deriv_const_mul (2 * π) differentiableAt_id
  rw [h_mul]
  have h_id : deriv (fun x => x) p = 1 := deriv_id p
  rw [h_id]
  have h_pow : deriv (fun x => x ^ 2) p = 2 * p := by
    have h := deriv_pow (n := 2) (differentiableAt_id (x := p))
    change deriv (fun x => x ^ 2) p = 2 * p ^ 1 * deriv (fun x => x) p at h
    rw [h]
    rw [h_id]
    ring
  rw [h_pow]
  ring

end Calibrator
