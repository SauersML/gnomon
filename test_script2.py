import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

print("S_lambda_fn" in content)
print("Hessian_fn" in content)
print("L_pen_fn" in content)
