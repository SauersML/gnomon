import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Basic

variable (n : ℕ)

-- Check if Fin n → ℝ is an inner product space
-- #check (inferInstance : InnerProductSpace ℝ (Fin n → ℝ))

-- Check WithLp 2
#check (inferInstance : InnerProductSpace ℝ (WithLp 2 (Fin n → ℝ)))
