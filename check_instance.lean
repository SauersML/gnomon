import Mathlib.Analysis.InnerProductSpace.PiL2

variable {n : ℕ}

example : InnerProductSpace ℝ (Fin n → ℝ) := inferInstance
