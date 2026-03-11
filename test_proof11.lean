import Mathlib

example (a b c : ℝ) (hc : 0 < c) (h : a * c < b * c) : a < b := by
  exact (mul_lt_mul_right hc).mp h
