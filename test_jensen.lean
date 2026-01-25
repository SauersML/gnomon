import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.Convex.Strict
import Mathlib.MeasureTheory.Integral.Bochner

open MeasureTheory

variable {Ω : Type*} [MeasureSpace Ω] [IsProbabilityMeasure (volume : Measure Ω)]
variable (f : ℝ → ℝ) (X : Ω → ℝ)

-- Check if this theorem exists
#check StrictConcaveOn.map_integral_lt

-- Also check strictConcaveOn_of_deriv2_neg
import Mathlib.Analysis.Convex.Deriv
#check strictConcaveOn_of_deriv2_neg
