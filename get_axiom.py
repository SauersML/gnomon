import re
with open("proofs/Calibrator/DGP.lean", "r") as f:
    content = f.read()

lines = content.split("\n")
for i, line in enumerate(lines):
    if "axiom wrightFisher_covariance_gap_lower_bound" in line:
        for j in range(max(0, i-5), min(len(lines), i+15)):
            print(f"{j+1}: {lines[j]}")
