import Mathlib

example (a b : ℝ) (h : 0 < a * b) : 0 < a * b := by
  have h_inner : a - a = 0 := sub_self _
  have h_zero : |a - a| = 0 := by rw [h_inner, abs_zero]
  exact h
