import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# I want to insert LAML_fixed_beta_fn and laml_fixed_beta_gradient_is_exact
# Before laml_gradient_is_exact

new_content = """
noncomputable def LAML_fixed_beta_fn (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ) (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ) (X : Matrix (Fin n) (Fin p) ℝ) (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ) (b : Matrix (Fin p) (Fin 1) ℝ) (rho : Fin k → ℝ) : ℝ :=
  let H := Hessian_fn S_basis X W rho b
  L_pen_fn log_lik S_basis rho b + 0.5 * Real.log (H.det) - 0.5 * Real.log ((S_lambda_fn S_basis rho).det)

theorem laml_fixed_beta_gradient_is_exact
    (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)
    (X : Matrix (Fin n) (Fin p) ℝ)
    (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ)
    (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ)
    (rho : Fin k → ℝ) (i : Fin k)
    (D : (Fin k → ℝ) →L[ℝ] ℝ)
    (hF : HasFDerivAt (fun r => LAML_fixed_beta_fn log_lik S_basis X W (beta_hat rho) r) D rho)
    (h_split : D (Pi.single i 1) = rust_direct_gradient_fn S_basis X W beta_hat log_lik rho i) :
  deriv (fun r => LAML_fixed_beta_fn log_lik S_basis X W (beta_hat rho) (Function.update rho i r)) (rho i) =
  rust_direct_gradient_fn S_basis X W beta_hat log_lik rho i :=
by
  let g : ℝ → (Fin k → ℝ) := Function.update rho i
  have hg : HasDerivAt g (Pi.single i 1) (rho i) := by
    simpa [g] using (hasDerivAt_update rho i (rho i))
  have h_update : g (rho i) = rho := by
    simpa [g] using (Function.update_eq_self i rho)
  have hF_at_update : HasFDerivAt (fun r => LAML_fixed_beta_fn log_lik S_basis X W (beta_hat rho) r) D (g (rho i)) := by
    simpa [h_update] using hF
  have hcomp : HasDerivAt (fun r => LAML_fixed_beta_fn log_lik S_basis X W (beta_hat rho) (g r))
      (D (Pi.single i 1)) (rho i) := by
    exact hF_at_update.comp_hasDerivAt (rho i) hg
  have h_deriv :
      deriv (fun r => LAML_fixed_beta_fn log_lik S_basis X W (beta_hat rho) (g r)) (rho i) =
      D (Pi.single i 1) := hcomp.deriv
  simpa [g, h_split] using h_deriv

"""

# Insert before theorem laml_gradient_is_exact
split_idx = content.find("theorem laml_gradient_is_exact")
final_content = content[:split_idx] + new_content + content[split_idx:]

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(final_content)
