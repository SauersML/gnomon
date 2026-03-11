import Mathlib

example (p1_r p2_r p1_t p2_t : ℝ) (h_same : p1_r = p2_r) (h_r_pos : 0 < p1_r) (h_time : p1_t < p2_t) :
  -2 * p2_r * p2_t < -2 * p1_r * p1_t := by
  rw [←h_same]
  apply mul_lt_mul_of_neg_left h_time
  linarith
