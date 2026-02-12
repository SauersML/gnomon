import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Submodule

variable {n : ℕ}

-- Check if Fin n -> ℝ has InnerProductSpace instance directly (it shouldn't usually, it's WithLp 2)
-- #check (inferInstance : InnerProductSpace ℝ (Fin n → ℝ))

-- Check with EuclideanSpace
#check (inferInstance : InnerProductSpace ℝ (EuclideanSpace ℝ (Fin n)))

-- Check if we can define projection
noncomputable def myProj (K : Submodule ℝ (EuclideanSpace ℝ (Fin n))) (y : EuclideanSpace ℝ (Fin n)) :=
  orthogonalProjection K y

-- But the code uses Fin n -> ℝ.
-- We might need to convert or assume the instance is there if open scoped PiL2?
open scoped InnerProductSpace
-- #check (inferInstance : InnerProductSpace ℝ (Fin n → ℝ))
