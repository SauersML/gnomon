import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Submodule
import Mathlib.Data.Fin.Basic

open MeasureTheory

variable (n : ℕ)

def l2norm_sq {ι : Type*} [Fintype ι] (v : ι → ℝ) : ℝ :=
  Finset.univ.sum (fun i => v i ^ 2)

noncomputable def orthogonalProjection {n : ℕ} (K : Submodule ℝ (Fin n → ℝ)) (y : Fin n → ℝ) : Fin n → ℝ :=
  let equiv := (WithLp.linearEquiv 2 ℝ (Fin n → ℝ)).symm
  let K_E : Submodule ℝ (WithLp 2 (Fin n → ℝ)) := K.map equiv
  let y_E : WithLp 2 (Fin n → ℝ) := equiv y
  let proj_E := Submodule.orthogonalProjection K_E y_E
  equiv.symm proj_E

lemma orthogonalProjection_eq_of_dist_le {n : ℕ} (K : Submodule ℝ (Fin n → ℝ)) (y p : Fin n → ℝ)
    (h_mem : p ∈ K) (h_min : ∀ w ∈ K, l2norm_sq (y - p) ≤ l2norm_sq (y - w)) :
    p = orthogonalProjection K y := by
  let equiv := (WithLp.linearEquiv 2 ℝ (Fin n → ℝ)).symm
  let K_E : Submodule ℝ (WithLp 2 (Fin n → ℝ)) := K.map equiv
  let y_E : WithLp 2 (Fin n → ℝ) := equiv y
  let p_E : WithLp 2 (Fin n → ℝ) := equiv p

  -- Verify p_E corresponds to p and is in K_E
  have hp_E_mem : p_E ∈ K_E := by
    simp only [K_E, p_E, Submodule.mem_map, equiv]
    use p
    simp [h_mem]

  -- The minimization condition translates to Euclidean distance
  have h_min_E : ∀ w_E ∈ K_E, dist y_E p_E ≤ dist y_E w_E := by
    intro w_E hw_E
    -- Get w back in the original space
    obtain ⟨w, hw_mem, hw_eq⟩ := (Submodule.mem_map).mp hw_E
    rw [← hw_eq]
    -- Translate l2norm_sq to dist^2
    have h_norm_eq : ∀ a b : Fin n → ℝ, l2norm_sq (a - b) = (dist (equiv a) (equiv b))^2 := by
      intro a b
      simp only [l2norm_sq, dist, NormedAddCommGroup.dist_eq_norm, equiv]
      -- WithLp norm is L2 norm.
      -- l2norm_sq is sum of squares.
      have h_pow : ∀ r : ℝ, |r|^2 = r^2 := fun r => sq_abs r
      rw [PiLp.norm_eq_of_nat 2 (by norm_num)]
      simp only [Real.rpow_two, h_pow]
      rw [Real.sq_sqrt (by apply Finset.sum_nonneg; intro _ _; apply sq_nonneg)]
      simp only [LinearEquiv.map_sub]
      rfl

    have h_ineq := h_min w hw_mem
    rw [h_norm_eq y p, h_norm_eq y w] at h_ineq
    apply nonneg_le_nonneg_of_sq_le_sq (dist_nonneg) h_ineq

  -- Apply the uniqueness of orthogonal projection in Euclidean space
  have h_eq_E : p_E = Submodule.orthogonalProjection K_E y_E := by
    apply Submodule.eq_orthogonalProjection_of_mem_of_dist_le hp_E_mem h_min_E

  -- Map back to original space
  simp only [orthogonalProjection]
  rw [← h_eq_E]
  simp only [equiv, LinearEquiv.symm_symm, LinearEquiv.apply_symm_apply]
