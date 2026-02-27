import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

new_content = content.replace("simp [g, Function.update_eq_self i rho]", "simp [g]")

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(new_content)
