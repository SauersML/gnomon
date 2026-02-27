import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# Let's extract the derivative_log_det_H_matrix theorem
match = re.search(r'theorem derivative_log_det_H_matrix.*?:= by', content, re.DOTALL)
if match:
    print(match.group(0))
