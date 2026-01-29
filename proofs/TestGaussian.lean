import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Function.L2Space

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} [MeasureSpace Ω]

-- Check if we can prove integrability of x^n for Gaussian
example (n : ℕ) : Integrable (fun x : ℝ => x ^ n) (gaussianReal 0 1) := by
  sorry
