import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift


namespace Calibrator

/-- The formal derivative of expected Brier score with respect to p.
    This replaces the tautological `expectedBrierScore_deriv` from Conclusions
    which only proved the algebraic equality `2*(p-π) = -2*π + 2*p`. -/
theorem expectedBrierScore_deriv_real (p π : ℝ) :
    deriv (fun p => expectedBrierScore p π) p = 2 * (p - π) := by
  have h_eq : (fun p => expectedBrierScore p π) = (fun p => π - 2 * π * p + p ^ 2) := by
    ext x
    rw [expectedBrierScore_quadratic]
  rw [h_eq]
  have hd1 : DifferentiableAt ℝ (fun p => π - 2 * π * p) p := by fun_prop
  have hd2 : DifferentiableAt ℝ (fun p : ℝ => p ^ 2) p := by fun_prop
  have h_add : deriv (fun p => (π - 2 * π * p) + p ^ 2) p = deriv (fun p => π - 2 * π * p) p + deriv (fun p : ℝ => p ^ 2) p := by
    exact deriv_add hd1 hd2
  rw [h_add]
  have hd1a : DifferentiableAt ℝ (fun p => π) p := by fun_prop
  have hd1b : DifferentiableAt ℝ (fun p => 2 * π * p) p := by fun_prop
  have h_sub : deriv (fun p => π - 2 * π * p) p = deriv (fun p => π) p - deriv (fun p => 2 * π * p) p := by
    exact deriv_sub hd1a hd1b
  rw [h_sub]
  have h_mul : deriv (fun p => 2 * π * p) p = 2 * π := by
    have hm1 : DifferentiableAt ℝ (fun p => 2 * π) p := by fun_prop
    have hm2 : DifferentiableAt ℝ (fun p => p) p := by fun_prop
    rw [deriv_const_mul (2 * π) hm2]
    simp
  rw [h_mul]
  simp
  ring

end Calibrator
