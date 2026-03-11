import Mathlib

example (Ne mu : ℝ) (hNe : 0 < Ne) (hmu : 0 ≤ mu) : 0 ≤ 4 * Ne * mu := by
  positivity
