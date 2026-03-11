import Mathlib

example (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a < b) : 1 / b < 1 / a := by
  apply one_div_lt_one_div_of_lt ha h
