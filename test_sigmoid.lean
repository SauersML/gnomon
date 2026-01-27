import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic

noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

lemma sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  apply div_pos one_pos
  have h : Real.exp (-x) > 0 := Real.exp_pos (-x)
  linarith

lemma deriv_sigmoid (x : ℝ) : deriv sigmoid x = sigmoid x * (1 - sigmoid x) := by
  unfold sigmoid
  rw [deriv_div_const, deriv_inv]
  sorry
  sorry
  sorry
