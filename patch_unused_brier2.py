import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# Fix unused `log_lik` in `laml_fixed_beta_gradient_is_exact`
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

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(content)
