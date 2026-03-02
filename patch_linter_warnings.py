import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# 4704: try 'simp' instead of 'simpa'
content = content.replace("simpa using h_pos p", "simp using h_pos p")
content = content.replace("simpa using h_lt", "simp using h_lt")
content = content.replace("simpa using h", "simp using h")
# Wait, let's just use regex for simpa
content = re.sub(r'simpa using h_pos ([a-z]+)', r'simp [h_pos \1]', content)

# A safer approach for simpa -> simp
content = content.replace("simpa [F_diff] using h_diff", "simp [F_diff, h_diff]")

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(content)
