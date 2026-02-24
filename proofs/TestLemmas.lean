import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.Notation

open MeasureTheory ProbabilityTheory

noncomputable def stdGaussian : Measure ℝ := gaussianReal 0 1
noncomputable def stdGaussianK (k : ℕ) : Measure (Fin k → ℝ) := Measure.pi (fun _ => stdGaussian)
noncomputable def stdNormalProd (k : ℕ) : Measure (ℝ × (Fin k → ℝ)) := stdGaussian.prod (stdGaussianK k)

variable {k : ℕ} [Fintype (Fin k)]

lemma test_prod_mul_congr (f : ℝ → ℝ) (g : (Fin k → ℝ) → ℝ)
  (hf : Integrable f stdGaussian) (hg : Integrable g (stdGaussianK k)) :
  Integrable (fun pc : ℝ × (Fin k → ℝ) => g pc.2 * f pc.1) (stdNormalProd k) := by
  have h := Integrable.prod_mul hf hg
  apply h.congr
  intro x
  ring

lemma test_snd_meas (f : (Fin k → ℝ) → ℝ)
  (h : AEStronglyMeasurable (fun pc : ℝ × (Fin k → ℝ) => f pc.2) (stdNormalProd k)) :
  AEStronglyMeasurable f (stdGaussianK k) := by
  -- AEStronglyMeasurable.snd ?
  -- There is AEStronglyMeasurable.comp_snd
  -- But going from product to marginal?
  sorry
