import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional

open scoped InnerProductSpace

variable {n : ℕ}
variable (K : Submodule ℝ (Fin n → ℝ))

#check orthogonalProjection K
