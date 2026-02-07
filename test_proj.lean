import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped InnerProductSpace
open InnerProductSpace

noncomputable def myProj {n : ℕ} (K : Submodule ℝ (EuclideanSpace ℝ (Fin n))) (y : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  (orthogonalProjection K y : EuclideanSpace ℝ (Fin n))

variable {n : ℕ}
variable (K : Submodule ℝ (Fin n → ℝ))

-- We need to check how to map K to EuclideanSpace
-- EuclideanSpace ℝ (Fin n) is just WithLp 2 (Fin n → ℝ)

def toEuclidean (x : Fin n → ℝ) : EuclideanSpace ℝ (Fin n) := WithLp.equiv 2 (Fin n → ℝ) x
def fromEuclidean (x : EuclideanSpace ℝ (Fin n)) : Fin n → ℝ := WithLp.equiv 2 (Fin n → ℝ) x -- It is an equiv, so use symm or inverse?
-- WithLp.equiv is : (Fin n → ℝ) ≃ WithLp 2 (Fin n → ℝ)

#check WithLp.linearEquiv
