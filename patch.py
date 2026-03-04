import re

with open("proofs/Calibrator.lean", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.startswith("theorem context_specificity_proved"):
        break
    new_lines.append(line)

with open("proofs/Calibrator.lean", "w") as f:
    f.writelines(new_lines)
