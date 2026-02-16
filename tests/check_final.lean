import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Submodule
import Mathlib.Data.Fin.Basic

open MeasureTheory
open scoped InnerProductSpace

variable (n : ℕ)

def l2norm_sq {ι : Type*} [Fintype ι] (v : ι → ℝ) : ℝ :=
  Finset.univ.sum (fun i => v i ^ 2)

lemma linear_coeff_zero_of_quadratic_nonneg_aux (a b : ℝ)
    (h : ∀ ε : ℝ, a * ε + b * ε^2 ≥ 0) : a = 0 := by
  by_contra ha_ne
  by_cases hb : b = 0
  · -- Case b = 0: then a*ε ≥ 0 for all ε, impossible if a ≠ 0
    by_cases ha_pos : 0 < a
    · have h_neg1 := h (-1)
      simp only [hb, zero_mul, add_zero, mul_neg, mul_one] at h_neg1
      linarith
    · push_neg at ha_pos
      have ha_neg : a < 0 := lt_of_le_of_ne ha_pos ha_ne
      have h_1 := h 1
      simp only [hb, zero_mul, add_zero, mul_one] at h_1
      linarith
  · -- Case b ≠ 0: consider the vertex of the parabola
    by_cases hb_pos : 0 < b
    · -- b > 0: minimum at ε = -a/(2b) gives value -a²/(4b) < 0
      let ε := -a / (2 * b)
      have hε := h ε
      have ha_sq_pos : 0 < a^2 := sq_pos_of_ne_zero ha_ne
      have eval : a * ε + b * ε^2 = -a^2 / (4 * b) := by
        simp only [ε]; field_simp; ring
      rw [eval] at hε
      have : -a^2 / (4 * b) < 0 := by
        apply div_neg_of_neg_of_pos
        · linarith
        · linarith
      linarith
    · -- b < 0: quadratic opens downward, eventually negative
      push_neg at hb_pos
      have hb_neg : b < 0 := lt_of_le_of_ne hb_pos hb
      let ε := -2 * a / b
      have hε := h ε
      have ha_sq_pos : 0 < a^2 := sq_pos_of_ne_zero ha_ne
      have eval : a * ε + b * ε^2 = 2 * a^2 / b := by
        simp only [ε]; field_simp; ring
      rw [eval] at hε
      have : 2 * a^2 / b < 0 := by
        apply div_neg_of_pos_of_neg
        · linarith
        · exact hb_neg
      linarith


noncomputable def orthogonalProjection {n : ℕ} (K : Submodule ℝ (Fin n → ℝ)) (y : Fin n → ℝ) : Fin n → ℝ :=
  have : Fact ((1 : ℝ≥0∞) ≤ 2) := ⟨by norm_num⟩
  let equiv := (WithLp.linearEquiv 2 ℝ (Fin n → ℝ)).symm
  let K_E : Submodule ℝ (WithLp 2 (Fin n → ℝ)) := K.map equiv
  let y_E : WithLp 2 (Fin n → ℝ) := equiv y
  let proj_E := Submodule.starProjection K_E y_E
  equiv.symm proj_E

lemma orthogonalProjection_eq_of_dist_le {n : ℕ} (K : Submodule ℝ (Fin n → ℝ)) (y p : Fin n → ℝ)
    (h_mem : p ∈ K) (h_min : ∀ w ∈ K, l2norm_sq (y - p) ≤ l2norm_sq (y - w)) :
    p = orthogonalProjection K y := by
  have : Fact ((1 : ℝ≥0∞) ≤ 2) := ⟨by norm_num⟩
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
      rw [dist_eq_norm, ← equiv.map_sub]
      simp only [l2norm_sq, equiv]
      have h_pow : ∀ r : ℝ, |r|^2 = r^2 := fun r => sq_abs r
      rw [PiLp.norm_eq_of_nat 2 (by norm_num)]
      simp only [Real.rpow_two, h_pow]
      rw [Real.sq_sqrt (by apply Finset.sum_nonneg; intro _ _; apply sq_nonneg)]
      rfl

    have h_ineq := h_min w hw_mem
    rw [h_norm_eq y p, h_norm_eq y w] at h_ineq
    simp only [sq_le_sq, abs_of_nonneg dist_nonneg] at h_ineq
    exact h_ineq

  -- Apply the uniqueness of orthogonal projection in Euclidean space
  have h_eq_E : p_E = Submodule.starProjection K_E y_E := by
    -- Variational characterization: minimizer implies orthogonality
    have h_orth : ∀ w_E ∈ K_E, ⟪y_E - p_E, w_E⟫_ℝ = 0 := by
      intro w_E hw_E
      -- Use the linear_coeff_zero_of_quadratic_nonneg lemma
      apply linear_coeff_zero_of_quadratic_nonneg_aux
      intro ε
      -- dist (y - p)^2 <= dist (y - (p + ε w))^2
      have h_le := h_min_E (p_E + ε • w_E) (Submodule.add_mem K_E hp_E_mem (Submodule.smul_mem K_E ε hw_E))
      rw [dist_eq_norm, dist_eq_norm] at h_le
      simp only [sub_add_eq_sub_sub] at h_le
      -- Expand norm(y - p - ε w)^2 = norm(y-p)^2 - 2ε<y-p, w> + ε^2 norm(w)^2
      -- We want to match form: a*ε + b*ε^2 >= 0
      -- Rearranging h_le: norm(y-p - εw)^2 - norm(y-p)^2 >= 0
      have h_expand : ‖(y_E - p_E) - ε • w_E‖^2 = ‖y_E - p_E‖^2 - 2 * ε * ⟪y_E - p_E, w_E⟫_ℝ + ε^2 * ‖w_E‖^2 := by
        rw [norm_sub_sq_real, inner_smul_right, norm_smul, mul_pow]
        simp only [sq_abs]
        ring
      rw [h_expand] at h_le
      linarith [h_le]

    -- Uniqueness implies p_E is the projection
    exact Eq.symm (Submodule.eq_starProjection_of_mem_of_inner_eq_zero hp_E_mem h_orth)

  -- Map back to original space
  simp only [orthogonalProjection]
  rw [← h_eq_E]
  simp only [equiv, LinearEquiv.symm_symm, LinearEquiv.apply_symm_apply]
