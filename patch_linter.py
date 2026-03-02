import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# Fix unused `log_lik` in `inv_mul_self_of_det_ne_zero`
# Actually, the unused log_lik is in `rust_delta_correctness`

content = re.sub(
    r'theorem rust_delta_correctness\s*\(\s*S_basis : Fin k → Matrix \(Fin p\) \(Fin p\) ℝ\s*\)\s*\(\s*X : Matrix \(Fin n\) \(Fin p\) ℝ\s*\)\s*\(\s*W : Matrix \(Fin p\) \(Fin 1\) ℝ → Matrix \(Fin n\) \(Fin n\) ℝ\s*\)\s*\(\s*beta_hat : \(Fin k → ℝ\) → Matrix \(Fin p\) \(Fin 1\) ℝ\s*\)\s*\(\s*rho : Fin k → ℝ\s*\)\s*\(\s*i : Fin k\s*\)',
    r'theorem rust_delta_correctness\n    (S_basis : Fin k → Matrix (Fin p) (Fin p) ℝ)\n    (X : Matrix (Fin n) (Fin p) ℝ)\n    (W : Matrix (Fin p) (Fin 1) ℝ → Matrix (Fin n) (Fin n) ℝ)\n    (beta_hat : (Fin k → ℝ) → Matrix (Fin p) (Fin 1) ℝ)\n    (rho : Fin k → ℝ) (i : Fin k)',
    content
)

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(content)
