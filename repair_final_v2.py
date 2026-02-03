import sys

with open("proofs/Calibrator.lean", "r") as f:
    lines = f.readlines()

# 1. Restore lines in rawOptimal_implies_orthogonality_gen (~1106)
# Find hYP_int line
target_1 = -1
for i, line in enumerate(lines):
    if i < 2000 and "(hYP_int : Integrable" in line:
        # Check if next line is let b := ...
        if "let b :=" in lines[i+1]:
            target_1 = i
            break

if target_1 != -1:
    print(f"Restoring lines at {target_1+1}")
    restored = [
        "    (h_resid_sq_int : Integrable (fun pc => (dgp.trueExpectation pc.1 pc.2 - (model.γ₀₀ + model.γₘ₀ ⟨0, by norm_num⟩ * pc.1))^2) dgp.jointMeasure) :\n",
        "    let a := model.γ₀₀\n"
    ]
    lines[target_1+1:target_1+1] = restored
else:
    print("Could not find insertion point for rawOptimal...gen")

# 2. Remove garbage at 3826-3827 (indices might shift due to insertion above)
# The garbage starts with (h_resid_sq_int ...
# and is inside "have h_coeffs" block.
garbage_idx = -1
for i, line in enumerate(lines):
    if i > 3000 and i < 5000 and "(h_resid_sq_int : Integrable" in line:
        # Check context: inside h_coeffs block?
        # Check indentation: 4 spaces?
        if line.strip().startswith("(h_resid_sq_int"):
             # Also next line model...
             if "model.γ₀₀ = " in lines[i+1]:
                 garbage_idx = i
                 break

if garbage_idx != -1:
    print(f"Removing garbage at {garbage_idx+1}")
    del lines[garbage_idx:garbage_idx+2]
else:
    print("Garbage not found (or index mismatch).")

with open("proofs/Calibrator.lean", "w") as f:
    f.writelines(lines)
