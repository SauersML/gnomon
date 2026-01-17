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

-- We need to make sure we are using the Euclidean norm structure on ι → ℝ
-- This comes from Mathlib.Analysis.InnerProductSpace.PiL2

theorem penalty_quadratic_tendsto_proof {ι : Type*} [Fintype ι] [DecidableEq ι]
    (S : Matrix ι ι ℝ) (lam : ℝ) (hlam : 0 < lam)
    (hS_posDef : ∀ v : ι → ℝ, v ≠ 0 → 0 < Matrix.dotProduct (S.mulVec v) v) :
    Filter.Tendsto
      (fun β => lam * Finset.univ.sum (fun i => β i * (S.mulVec β) i))
      (Filter.cocompact (ι → ℝ)) Filter.atTop := by
  -- Define the quadratic form Q(β) = βᵀSβ
  let Q := fun β => Matrix.dotProduct (S.mulVec β) β
  have hQ_def : ∀ β, Q β = Finset.univ.sum (fun i => β i * (S.mulVec β) i) := by
    intro β
    simp [Q, Matrix.dotProduct, mul_comm]

  -- Q is continuous
  have hQ_cont : Continuous Q := by
    unfold Q Matrix.dotProduct
    -- Matrix.mulVec is linear, hence continuous
    -- dotProduct is bilinear, hence continuous
    -- Or just built from sums and prods
    simp only [Matrix.dotProduct, Matrix.mulVec]
    continuity

  -- Restrict Q to the unit sphere
  let sphere := Metric.sphere (0 : ι → ℝ) 1
  have h_sphere_compact : IsCompact sphere := Metric.isCompact_sphere 0 1

  -- Sphere is nonempty (assuming ι is inhabited? Or handle ι empty case)
  -- If ι is empty, then cocompact is bot? No, the space is unique point 0.
  -- If ι is empty, then β = 0 is the only point. Filter.cocompact is ⊥?
  -- Actually, let's assume Nonempty ι. If empty, the limit is trivial or vacuously true?
  -- Wait, if ι is empty, then cocompact is ⊥, and Tendsto to atTop is true.
  rcases isEmpty_or_nonempty ι with h_empty | h_nonempty
  · simp [Filter.cocompact_eq_bot]

  have h_sphere_nonempty : sphere.Nonempty := by
    -- construct a vector of norm 1.
    -- e.g. basis vector.
    obtain ⟨i⟩ := h_nonempty
    let v := EuclideanSpace.single i (1 : ℝ)
    have hv : ‖v‖ = 1 := by
      simp [EuclideanSpace.norm_single, abs_one]
    use v
    simp [sphere, hv]

  -- Q attains a minimum on the sphere
  obtain ⟨v_min, hv_min_in, h_min_le⟩ := h_sphere_compact.exists_forall_le h_sphere_nonempty hQ_cont.continuousOn

  let c := Q v_min
  have hc_pos : 0 < c := by
    apply hS_posDef
    -- v_min is in sphere, so norm is 1, so v_min ≠ 0
    intro h0
    simp [sphere] at hv_min_in
    rw [h0, norm_zero] at hv_min_in
    linarith

  -- For any β, Q(β) ≥ c * ‖β‖²
  have h_bound : ∀ β, Q β ≥ c * ‖β‖^2 := by
    intro β
    by_cases hβ : β = 0
    · subst hβ
      simp [Q, Matrix.dotProduct, Matrix.mulVec_zero, Matrix.dotProduct_zero, norm_zero]
      linarith
    · let u := (‖β‖⁻¹) • β
      have hu_norm : ‖u‖ = 1 := by
        rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel]
        rwa [norm_eq_zero]
      have hu_in : u ∈ sphere := by simp [sphere, hu_norm]
      have hQu : Q u ≥ c := h_min_le u hu_in
      -- Q(u) = Q( (1/‖β‖) β ) = (1/‖β‖)^2 Q(β)
      -- Matrix.mulVec is linear: S(kβ) = k Sβ
      -- dotProduct is bilinear: (k Sβ) · (k β) = k^2 (Sβ · β)
      have h_scale : Q u = (‖β‖⁻¹)^2 * Q β := by
        simp [Q, Matrix.dotProduct, Matrix.mulVec_smul]
        ring
      rw [h_scale] at hQu
      -- c ≤ ‖β‖⁻² Q β  =>  c ‖β‖² ≤ Q β
      rw [inv_pow] at hQu
      have hnorm_sq_pos : 0 < ‖β‖^2 := pow_pos (norm_pos_iff.mpr hβ) 2
      rw [le_div_iff hnorm_sq_pos] at hQu
      linarith

  -- We want to show lam * Q(β) → ∞
  -- lam * Q(β) ≥ lam * c * ‖β‖²
  -- lam * c > 0
  -- ‖β‖² → ∞ on cocompact
  apply Filter.tendsto_atTop_mono (fun β => lam * c * ‖β‖^2)
  · intro β
    simp only [hQ_def] at h_bound
    -- hQ_def says Q β = sum ...
    -- The target function is lam * sum ...
    rw [← hQ_def]
    apply mul_le_mul_of_nonneg_left (h_bound β) (le_of_lt hlam)
  · -- Show lam * c * ‖β‖^2 → ∞
    have h_coeff_pos : 0 < lam * c := mul_pos hlam hc_pos
    -- ‖x‖ → ∞ on cocompact
    have h_norm_tendsto : Filter.Tendsto (fun β => ‖β‖) (Filter.cocompact (ι → ℝ)) Filter.atTop := by
      rw [Metric.cocompact_eq_cocompact_norm]
      exact Filter.tendsto_norm_cocompact_atTop
    have h_sq_tendsto : Filter.Tendsto (fun x : ℝ => x^2) Filter.atTop Filter.atTop :=
      Filter.tendsto_pow_atTop two_ne_zero
    have h_comp := h_sq_tendsto.comp h_norm_tendsto
    apply Filter.Tendsto.const_mul_atTop h_coeff_pos h_comp
