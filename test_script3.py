import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# Let's replace _hA : A.PosDef and _hB : B.IsSymm
new_content = content.replace("theorem derivative_log_det_H_matrix (A B : Matrix m m ℝ)\n    (_hA : A.PosDef) (_hB : B.IsSymm)\n    (rho : ℝ) (h_inv : (H_matrix A B rho).det ≠ 0) :", "theorem derivative_log_det_H_matrix (A B : Matrix m m ℝ)\n    (rho : ℝ) (h_inv : (H_matrix A B rho).det ≠ 0) :")

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(new_content)
