import Mathlib
import proofs.Calibrator

open Matrix
open Calibrator

variable {k p n : Type*} [Fintype k] [Fintype p] [Fintype n] [DecidableEq k] [DecidableEq p] [DecidableEq n]

noncomputable def LAML_fixed_beta_fn
    (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)
    (X : Matrix (Fin n) (Fin p) ℝ)
    (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ)
    (b : Matrix (Fin p) (Fin 1) ℝ)
    (rho : Fin k → ℝ) : ℝ :=
  let H := Hessian_fn S_basis X W rho b
  L_pen_fn log_lik S_basis rho b + 0.5 * Real.log (H.det) - 0.5 * Real.log ((S_lambda_fn S_basis rho).det)

-- We need to prove that the derivative of LAML_fixed_beta_fn with respect to `rho i`
-- is exactly rust_direct_gradient_fn when beta_hat rho = b.
