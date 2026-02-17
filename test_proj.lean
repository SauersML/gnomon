import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

open InnerProductSpace

noncomputable def myProj {n : ℕ} (K : Submodule ℝ (Fin n → ℝ)) (y : Fin n → ℝ) : Fin n → ℝ :=
  K.subtype (orthogonalProjection K y)

lemma myProj_spec {n : ℕ} (K : Submodule ℝ (Fin n → ℝ)) (y p : Fin n → ℝ)
    (h_mem : p ∈ K) (h_min : ∀ w ∈ K, dist y p ≤ dist y w) :
    p = myProj K y := by
  rw [myProj]
  -- We want to prove p = orthogonalProjection K y
  -- We know orthogonalProjection K y is the unique minimizer of distance
  have h_proj_mem : (orthogonalProjection K y : Fin n → ℝ) ∈ K := Submodule.coe_mem (orthogonalProjection K y)
  have h_min_proj : ∀ w : K, dist y (orthogonalProjection K y) ≤ dist y w :=
    orthogonalProjection_is_norm_min K y
  -- We need uniqueness.
  -- Since Fin n -> R is strictly convex (it's Euclidean), the minimizer is unique.
  sorry
