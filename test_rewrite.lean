import Mathlib

example (a b : ℝ) (h : 0 < a * b) : 0 < a * b := by
  have h_zero : |a - a| = 0 := by exact abs_zero
  exact h
