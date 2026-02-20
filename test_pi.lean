import Mathlib.MeasureTheory.Constructions.Pi

open MeasureTheory

variable {ι : Type*} [Fintype ι] [DecidableEq ι]
variable {α : ι → Type*} [∀ i, MeasurableSpace (α i)]
variable (μ : ∀ i, Measure (α i)) [∀ i, IsProbabilityMeasure (μ i)]

#check Measure.pi_map_eval
