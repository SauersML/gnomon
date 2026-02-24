import sys

with open("proofs/Calibrator.lean", "r") as f:
    content = f.read()

content = content.replace("hS_sq\\", "hS_sq_nu")
content = content.replace("hB_sq\\", "hB_sq_nu")
content = content.replace("hS_sq\\''", "hS_sq_nu")
content = content.replace("hB_sq\\''", "hB_sq_nu")
content = content.replace("integrable_const 0", "integrable_const (0:ℝ)")

old_proof = """    -- Check measurability of B
    have h_B_meas : AEStronglyMeasurable B ((stdNormalProdMeasure k).map Prod.snd) := by
      -- Since B is integrable L2, it is AEStronglyMeasurable
      exact (MemLp.of_integrable_sq (h_B_int.aestronglyMeasurable) h_B_int).aestronglyMeasurable"""

new_proof = """    -- Check measurability of B
    have h_B_meas : AEStronglyMeasurable B ((stdNormalProdMeasure k).map Prod.snd) := by
      apply Continuous.aestronglyMeasurable
      unfold predictorBase
      apply Continuous.add
      · exact continuous_const
      · refine continuous_finset_sum _ (fun l _ => ?_)
        dsimp [evalSmooth]
        refine continuous_finset_sum _ (fun i _ => ?_)
        apply Continuous.mul continuous_const
        exact (h_spline_cont i).comp (continuous_apply l)"""

if old_proof in content:
    content = content.replace(old_proof, new_proof)
else:
    print("Old proof not found")

with open("proofs/Calibrator.lean", "w") as f:
    f.write(content)
