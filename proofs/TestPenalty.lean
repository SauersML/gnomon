import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Topology.Order.Compact
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.DotProduct

-- Define dotProduct' and l2norm_sq as in Calibrator.lean
def dotProduct' {ι : Type*} [Fintype ι] (u v : ι → ℝ) : ℝ :=
  Finset.univ.sum (fun i => u i * v i)

def l2norm_sq {ι : Type*} [Fintype ι] (v : ι → ℝ) : ℝ :=
  Finset.univ.sum (fun i => v i ^ 2)

-- The theorem to prove
theorem penalty_quadratic_tendsto_test {ι : Type*} [Fintype ι] [DecidableEq ι]
    (S : Matrix ι ι ℝ) (lam : ℝ) (hlam : 0 < lam)
    (hS_posDef : ∀ v : ι → ℝ, v ≠ 0 → 0 < dotProduct' (S.mulVec v) v) :
    Filter.Tendsto
      (fun β => lam * Finset.univ.sum (fun i => β i * (S.mulVec β) i))
      (Filter.cocompact (ι → ℝ)) Filter.atTop := by
  sorry
