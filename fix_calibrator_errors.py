import sys

# Read the file
filepath = "proofs/Calibrator.lean"
with open(filepath, "r") as f:
    lines = f.readlines()

# Helper to find line index
def find_line_index(content_snippet):
    for i, line in enumerate(lines):
        if content_snippet in line:
            return i
    return -1

# Fix 1: h_gauss_moments le_refl
idx1 = find_line_index("integrable one_le_one")
if idx1 != -1:
    lines[idx1] = lines[idx1].replace("one_le_one", "le_refl 1")

# Fix 2: h_base_L2 rw h_prod
idx2 = find_line_index("exact MeasureTheory.Integrable.integral_prod_left h_norm_int'")
if idx2 != -1:
    # Insert rw before exact
    lines.insert(idx2, "        rw [h_prod] at h_norm_int'\n")

# Fix 3: integral_sq_id_gaussianReal (1st occurrence)
idx3 = find_line_index("have h_var_P : ∫ p, p^2 ∂μP = 1 := ProbabilityTheory.integral_sq_id_gaussianReal")
if idx3 != -1:
    lines[idx3] = """          have h_var_P : ∫ p, p^2 ∂μP = 1 := by
            -- Use variance property
            have h_var := ProbabilityTheory.variance_id_gaussianReal 0 1
            rw [ProbabilityTheory.variance_def' h_p_int h_p2_int] at h_var
            rw [h_mean_P] at h_var
            simp at h_var
            exact h_var
"""

# Fix 4: fun_prop failure in h_base_L2
idx4 = find_line_index("unfold predictorBase; fun_prop")
if idx4 != -1:
    lines[idx4] = "          unfold predictorBase evalSmooth; fun_prop\n"

# Fix 5: fun_prop failure in h_base_L2 second occurrence
idx5 = find_line_index("unfold predictorBase; fun_prop")
if idx5 != -1 and idx5 > idx4: # Make sure we find the next one
    lines[idx5] = "        unfold predictorBase evalSmooth; fun_prop\n"

# Fix 6: linarith failure in h_base_L2
idx6 = find_line_index("linarith [sq_nonneg (predictorSlope m c)]")
if idx6 != -1:
    # Add simplification before linarith
    lines.insert(idx6, "          simp only [Real.norm_eq_abs, sq_abs]\n")

# Fix 7: integral_sq_id_gaussianReal (2nd occurrence in h_term1_zero)
# It was around line 4169 in previous view
idx7 = -1
for i, line in enumerate(lines):
    if "ProbabilityTheory.integral_sq_id_gaussianReal" in line and i != idx3:
        idx7 = i
        break

if idx7 != -1:
    lines[idx7] = """        have h_var : ∫ x, x^2 ∂μP = 1 := by
          have h_var := ProbabilityTheory.variance_id_gaussianReal 0 1
          rw [ProbabilityTheory.variance_def' h_p_int h_p2_int] at h_var
          have h_mean : ∫ x, x ∂μP = 0 := ProbabilityTheory.integral_id_gaussianReal (μ := 0) (v := 1)
          rw [h_mean] at h_var
          simp at h_var
          exact h_var
"""

# Fix 8: h_B_sq_int Integrable.add failure
# The issue was likely missing h_base_sq_int. We need to derive it from h_base_L2.
idx8 = find_line_index("have h_base_L2_sq : Integrable (fun pc : ℝ × (Fin k → ℝ) => (predictorBase m pc.2)^2) (μP.prod μC) := by")
if idx8 != -1:
    # We replace the block inside
    # We want:
    # have h_sq : Integrable (fun c => (predictorBase m c)^2) μC := (memLp_two_iff_integrable_sq h_base_L2.aestronglyMeasurable).mp h_base_L2
    # exact MeasureTheory.Integrable.comp_snd h_sq

    # Locate  which was the old attempt
    idx_refine = -1
    for i in range(idx8, idx8+10):
        if "refine" in lines[i]:
            idx_refine = i
            break

    if idx_refine != -1:
        lines[idx_refine] = """               have h_sq : Integrable (fun c => (predictorBase m c)^2) μC := (memLp_two_iff_integrable_sq h_base_L2.aestronglyMeasurable).mp h_base_L2
               exact MeasureTheory.Integrable.comp_snd h_sq
"""
        # Remove subsequent lines until
        # Actually I need to be careful not to delete  line which follows the  block indentation?
        # The code was:
        #          · have h_base_L2_sq : ... := by
        #               refine ...
        #               rw ...
        #               exact ...
        #            exact h_base_L2_sq

        # I'll just clear the lines between idx_refine and the closing brace/indent drop.
        # But python script is tricky with indents.
        # I'll just comment out the bad lines if I can or overwrite them.
        pass # Handling via exact overwrite is safer.

# I'll use a targeted rewrite for h_B_sq_int logic via a separate tool call if this script is too complex.
# But let's try to do it here.
# I will replace the  block entirely.

# Fix 9: h_cross_int AEStronglyMeasurable failures
# This is likely due to implicit arguments. I will try to make it more robust.

# Write the file back
with open(filepath, "w") as f:
    f.writelines(lines)
