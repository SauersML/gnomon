import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Topology.MetricSpace.Basic

open InnerProductSpace

def test_dist (n : ℕ) (x y : Fin n → ℝ) : ℝ := dist x y

#print test_dist
