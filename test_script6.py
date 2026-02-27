import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

new_content = content.replace("simpa [g] using (Function.update_eq_self i rho)", "simp [g, Function.update_eq_self i rho]")

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(new_content)
