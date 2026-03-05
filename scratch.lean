import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt

open Real

noncomputable def presentDayPGSVariance (V_A fst : ℝ) : ℝ :=
  (1 - fst) * V_A

noncomputable def Var_Delta_Mu (V_A fst : ℝ) : ℝ :=
  2 * fst * V_A

noncomputable def Expected_Abs_Shift (V_A fstS fstT : ℝ) : ℝ :=
  Real.sqrt (Var_Delta_Mu V_A (fstS + fstT)) * Real.sqrt (2 / Real.pi)

theorem expected_abs_mean_shift_bound_proved
    (V_A fstS fstT : ℝ)
    (hVA_pos : 0 < V_A)
    (hfst_sum_nonneg : 0 ≤ fstS + fstT)
    (hfstS_lt_one : fstS < 1) :
    Expected_Abs_Shift V_A fstS fstT / Real.sqrt (presentDayPGSVariance V_A fstS) =
      2 * Real.sqrt ((fstS + fstT) / (Real.pi * (1 - fstS))) := by
  unfold Expected_Abs_Shift presentDayPGSVariance Var_Delta_Mu
  have h_pi_pos : 0 < Real.pi := Real.pi_pos
  have h1 : 2 * (fstS + fstT) * V_A ≥ 0 := by
    have h2 : 0 ≤ 2 * (fstS + fstT) := mul_nonneg (by positivity) hfst_sum_nonneg
    exact mul_nonneg h2 (le_of_lt hVA_pos)
  have h2 : 2 / Real.pi ≥ 0 := by positivity
  have h3 : (1 - fstS) * V_A ≥ 0 := by
    have h4 : 0 ≤ 1 - fstS := sub_nonneg.mpr (le_of_lt hfstS_lt_one)
    exact mul_nonneg h4 (le_of_lt hVA_pos)
  have hVA_ne_zero : V_A ≠ 0 := ne_of_gt hVA_pos
  have h_1_fstS_ne_zero : 1 - fstS ≠ 0 := sub_ne_zero.mpr (ne_of_lt hfstS_lt_one).symm
  have h_pi_ne_zero : Real.pi ≠ 0 := ne_of_gt h_pi_pos

  have h_sqrt_mul : Real.sqrt (2 * (fstS + fstT) * V_A) * Real.sqrt (2 / Real.pi) = Real.sqrt (2 * (fstS + fstT) * V_A * (2 / Real.pi)) := by
    exact (Real.sqrt_mul h1).symm

  rw [h_sqrt_mul]
  rw [← Real.sqrt_div h1]

  congr 1
  calc
    2 * (fstS + fstT) * V_A * (2 / Real.pi) / ((1 - fstS) * V_A)
      = (4 * (fstS + fstT) * V_A / Real.pi) / ((1 - fstS) * V_A) := by ring
    _ = (4 * (fstS + fstT) / (Real.pi * (1 - fstS))) * (V_A / V_A) := by ring
    _ = 4 * (fstS + fstT) / (Real.pi * (1 - fstS)) := by rw [div_self hVA_ne_zero, mul_one]
    _ = 2 ^ 2 * ((fstS + fstT) / (Real.pi * (1 - fstS))) := by ring
