import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

match = re.search(r'laml_gradient_is_exact.*?:=', content, re.DOTALL)
if match:
    print(match.group(0))
