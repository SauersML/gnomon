import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.DotProduct
import Mathlib.Data.Matrix.Mul
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Topology.Order.Compact
import Mathlib.Topology.MetricSpace.ProperSpace

open scoped InnerProductSpace

theorem penalty_quadratic_tendsto_proof {ι : Type*} [Fintype ι] [DecidableEq ι]
    (S : Matrix ι ι ℝ) (lam : ℝ) (hlam : 0 < lam)
    (hS_posDef : ∀ v : ι → ℝ, v ≠ 0 → 0 < Matrix.dotProduct (S.mulVec v) v) :
    Filter.Tendsto
      (fun β => lam * Finset.univ.sum (fun i => β i * (S.mulVec β) i))
      (Filter.cocompact (ι → ℝ)) Filter.atTop := by
  sorry
