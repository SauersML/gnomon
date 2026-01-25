import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Integral.Bochner

open MeasureTheory ProbabilityTheory

example : ∫ x : ℝ, x ∂(gaussianReal 0 1) = 0 := by
  -- try library_search or check known lemmas
  exact integral_gaussianReal_mul_self_pow_odd 0 0 1 (by norm_num)

example : ∫ x : ℝ, x^2 ∂(gaussianReal 0 1) = 1 := by
  -- try library_search
  sorry
