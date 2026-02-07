import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic

open scoped InnerProductSpace

variable {n : ℕ}

#check (inferInstance : InnerProductSpace ℝ (Fin n → ℝ))
