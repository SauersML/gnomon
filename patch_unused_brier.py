import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# Fix unused `log_lik` in `laml_fixed_beta_gradient_is_exact`
# Actually wait, `laml_fixed_beta_gradient_is_exact` unused variable warning is at 6477
# The warning for `laml_gradient_validity` unused variable `h_grad_beta` is at 6665.

content = re.sub(
    r'theorem laml_fixed_beta_gradient_is_exact\n\s*\(\s*_log_lik\s*:\s*Matrix\s*\(Fin\s*p\)\s*\(Fin\s*1\)\s*ℝ\s*→\s*ℝ\s*\)',
    r'theorem laml_fixed_beta_gradient_is_exact\n    (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)',
    content
)

content = re.sub(
    r'theorem laml_gradient_validity\n\s*\(\s*_log_lik\s*:\s*Matrix\s*\(Fin\s*p\)\s*\(Fin\s*1\)\s*ℝ\s*→\s*ℝ\s*\)',
    r'theorem laml_gradient_validity\n    (log_lik : Matrix (Fin p) (Fin 1) ℝ → ℝ)',
    content
)

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(content)
