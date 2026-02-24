import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Probability.Notation

open MeasureTheory ProbabilityTheory

noncomputable def stdGaussian : Measure ℝ := gaussianReal 0 1
noncomputable def stdGaussianK (k : ℕ) : Measure (Fin k → ℝ) := Measure.pi (fun _ => stdGaussian)
noncomputable def stdNormalProd (k : ℕ) : Measure (ℝ × (Fin k → ℝ)) := stdGaussian.prod (stdGaussianK k)

variable {k : ℕ} [Fintype (Fin k)]

-- Lemma: E[(S(c)P - (aP + B(c)))^2] = E[(S(c)-a)^2] + E[B(c)^2]
-- Assuming P, C independent, P~N(0,1)
lemma risk_decomp_multiplicative
    (S : (Fin k → ℝ) → ℝ)
    (a : ℝ)
    (B : (Fin k → ℝ) → ℝ)
    (hS : Integrable (fun c => (S c)^2) (stdGaussianK k))
    (hB : Integrable (fun c => (B c)^2) (stdGaussianK k))
    :
    ∫ pc : ℝ × (Fin k → ℝ), (S pc.2 * pc.1 - (a * pc.1 + B pc.2))^2 ∂(stdNormalProd k) =
    (∫ c, (S c - a)^2 ∂(stdGaussianK k)) + (∫ c, (B c)^2 ∂(stdGaussianK k)) := by
  sorry
