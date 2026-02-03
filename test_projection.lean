import Mathlib.Analysis.InnerProductSpace.Projection
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.Topology.Algebra.Module.FiniteDimension

open scoped InnerProductSpace
open Submodule

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
variable (K : Submodule ℝ E) [CompleteSpace K]

-- Test coercion
#check (orthogonalProjection K (0 : E) : E)

-- Test Uniqueness Lemma
theorem unique_proj (y : E) (z : E) (hz : z ∈ K) (h_min : dist y z ≤ dist y (orthogonalProjection K y)) :
    z = orthogonalProjection K y := by
  let p := orthogonalProjection K y
  have hp : (p : E) ∈ K := coe_mem (orthogonalProjection K y)
  have h_orth : y - p ∈ Kᗮ := by
    rw [← orthogonalProjection_eq_self_iff.mpr hp] -- Just use the property
    exact sub_orthogonalProjection_mem_orthogonal y

  -- We want ||y - z||^2 = ||y - p + p - z||^2 = ||y - p||^2 + ||p - z||^2
  have h_decomp : ‖y - z‖^2 = ‖y - p‖^2 + ‖p - z‖^2 := by
    have h1 : y - z = (y - p) + (p - z) := by ring
    rw [h1]
    apply norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero
    rw [real_inner_comm]
    apply h_orth
    apply Submodule.sub_mem K hp hz

  have h_le_sq : ‖y - z‖^2 ≤ ‖y - p‖^2 := by
    rw [dist_eq_norm, dist_eq_norm] at h_min
    exact sq_le_sq' (norm_nonneg _) h_min

  rw [h_decomp] at h_le_sq
  have h_pz_sq_le_zero : ‖p - z‖^2 ≤ 0 := by linarith
  have h_pz_zero : p - z = 0 := norm_eq_zero.mp (pow_eq_zero (le_antisymm h_pz_sq_le_zero (sq_nonneg _)))
  eq_symm
  exact eq_of_sub_eq_zero h_pz_zero

-- Test toLin' unfolding
open Matrix
variable (n m : Type) [Fintype n] [Fintype m] [DecidableEq n] [DecidableEq m]
variable (A : Matrix m n ℝ)

example : toLin' A = toLin (Pi.basisFun ℝ n) (Pi.basisFun ℝ m) A := by
  rfl
