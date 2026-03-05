import re
with open("proofs/Calibrator/Models.lean", "r") as f:
    content = f.read()

lines = content.split("\n")
for i, line in enumerate(lines):
    if "hoeffding_decomposition_exists_unique" in line:
        for j in range(max(0, i-5), min(len(lines), i+15)):
            print(f"{j+1}: {lines[j]}")
