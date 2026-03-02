import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# Fix unused `log_lik` in `laml_fixed_beta_gradient_is_exact`
content = content.replace(
    """(log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)""",
    """(_log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)
    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)"""
)
content = content.replace(
    """  deriv (fun r => LAML_fixed_beta_fn _log_lik S_basis X W b (Function.update rho i r)) (rho i) =
  rust_direct_gradient_fn S_basis X W beta_hat _log_lik rho i""",
    """  deriv (fun r => LAML_fixed_beta_fn log_lik S_basis X W b (Function.update rho i r)) (rho i) =
  rust_direct_gradient_fn S_basis X W beta_hat log_lik rho i"""
)
# Revert that, it's easier to just name it _log_lik and rename it back in the signature

content = re.sub(
    r'theorem laml_fixed_beta_gradient_is_exact\n\s*\(log_lik : Matrix \(Fin p\) \(Fin 1\) ℝ → ℝ\)',
    r'theorem laml_fixed_beta_gradient_is_exact\n    (_log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)',
    content
)

content = re.sub(
    r'theorem laml_gradient_validity\n\s*\(log_lik : Matrix \(Fin p\) \(Fin 1\) ℝ → ℝ\)',
    r'theorem laml_gradient_validity\n    (_log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)',
    content
)

content = content.replace(
    "(h_grad_beta : HasGradientAt (fun b => LAML_fn _log_lik S_basis X W (fun _ => b) rho)",
    "(_h_grad_beta : HasGradientAt (fun b => LAML_fn _log_lik S_basis X W (fun _ => b) rho)"
)

content = content.replace(
    "(h_opt1 : ∀ β, penalized_objective X y S β_opt ≤ penalized_objective X y S β)",
    "(_h_opt1 : ∀ β, penalized_objective X y S β_opt ≤ penalized_objective X y S β)"
)


with open('proofs/Calibrator.lean', 'w') as f:
    f.write(content)
