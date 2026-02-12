import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt

open scoped InnerProductSpace

def v : Fin 2 → ℝ := ![1, 1]
def z : Fin 2 → ℝ := ![0, 0]

-- Trying L-infinity
example : dist v z = 1 := by
  dsimp [v, z, dist]
  simp

-- Trying L2
-- example : dist v z = Real.sqrt 2 := by
--   dsimp [v, z, dist]
--   simp
