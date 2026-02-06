import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Topology.MetricSpace.Basic

open Real

#check (dist : (Fin 2 → ℝ) → (Fin 2 → ℝ) → ℝ)

def x : Fin 2 → ℝ := ![0, 0]
def y : Fin 2 → ℝ := ![1, 1]

#eval dist x y
-- If L-infinity, dist is 1.
-- If L2, dist is sqrt(2) ≈ 1.414.
