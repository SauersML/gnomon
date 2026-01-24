import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Analysis.Calculus.MeanValue

open Metric

theorem test_lip : True := by
  have : LipschitzWith 1 (id : ℝ → ℝ) := LipschitzWith.id
  trivial
