import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic

open scoped InnerProductSpace

variable (n : ℕ)

#check (inferInstance : NormedAddCommGroup (Fin n → ℝ))
#check (inferInstance : InnerProductSpace ℝ (Fin n → ℝ))

def dist_l2 (x y : Fin n → ℝ) : ℝ := dist (WithLp.equiv 2 (Fin n → ℝ) x) (WithLp.equiv 2 (Fin n → ℝ) y)

#check dist_l2
