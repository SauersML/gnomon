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
  let Q := fun β => Matrix.dotProduct (S.mulVec β) β
  have hQ_def : ∀ β, Q β = Finset.univ.sum (fun i => β i * (S.mulVec β) i) := by
    intro β
    simp [Q, Matrix.dotProduct]
  simp_rw [← hQ_def]

  have hQ_cont : Continuous Q := by
    unfold Q
    apply Continuous.sum
    intro i _
    apply Continuous.mul
    · exact continuous_apply i
    · apply Continuous.sum
      intro j _
      apply Continuous.mul
      · exact continuous_const
      · exact continuous_apply j

  cases isEmpty_or_nonempty ι
  case inl =>
    rw [Filter.cocompact_eq_bot]
    · exact Filter.tendsto_bot
    · exact isCompact_univ
  case inr =>
    let sphere := Metric.sphere (0 : ι → ℝ) 1
    have h_sphere_compact : IsCompact sphere := Metric.isCompact_sphere _ _
    have h_sphere_nonempty : sphere.Nonempty := by
      exact NormedSpace.sphere_nonempty.mpr zero_ne_one

    obtain ⟨β_min, h_min_mem, h_min_le⟩ := h_sphere_compact.exists_forall_le h_sphere_nonempty hQ_cont.continuousOn
    let c := Q β_min

    have hc_pos : 0 < c := by
      apply hS_posDef
      intro h_zero
      rw [h_zero] at h_min_mem
      simp at h_min_mem

    have h_bound : ∀ β, c * ‖β‖^2 ≤ Q β := by
      intro β
      by_cases h_zero : β = 0
      · rw [h_zero, norm_zero, zero_pow two_ne_zero, mul_zero]
        simp [Q, Matrix.dotProduct]
      · let u := (‖β‖⁻¹) • β
        have hu_norm : ‖u‖ = 1 := by
          rw [norm_smul, norm_inv, norm_norm, inv_mul_cancel]
          rwa [norm_eq_zero]
        have hu_mem : u ∈ sphere := by simp [sphere, hu_norm]
        have h_scale : Q β = ‖β‖^2 * Q u := by
          simp [Q, Matrix.dotProduct, Matrix.mulVec_smul, Matrix.dotProduct_smul, Matrix.smul_dotProduct]
          ring
        rw [h_scale]
        gcongr
        exact h_min_le u hu_mem

    apply Filter.tendsto_atTop_mono (g := fun β => lam * c * ‖β‖^2)
    · filter_upwards with β
      calc lam * c * ‖β‖^2
          = lam * (c * ‖β‖^2) := by ring
        _ ≤ lam * Q β := by
          gcongr
          exact h_bound β

    have h_coeff_pos : 0 < lam * c := mul_pos hlam hc_pos

    have h_norm_sq_tendsto : Filter.Tendsto (fun β : ι → ℝ => ‖β‖^2) (Filter.cocompact (ι → ℝ)) Filter.atTop := by
      apply Filter.Tendsto.pow_atTop (n := 2) two_ne_zero
      rw [Metric.tendsto_norm_cocompact_atTop]

    exact Filter.Tendsto.const_mul_atTop h_coeff_pos h_norm_sq_tendsto
