import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.DotProduct

open scoped InnerProductSpace
open MeasureTheory

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

theorem continuous_coercive_exists_min_proof
    (f : (ι → ℝ) → ℝ) (h_cont : Continuous f)
    (h_coercive : Filter.Tendsto f (Filter.cocompact _) Filter.atTop) :
    ∃ x : ι → ℝ, ∀ y : ι → ℝ, f x ≤ f y := by
  haveI : Nonempty (ι → ℝ) := Nonempty.intro (fun _ => 0)
  exact Continuous.exists_forall_le h_cont h_coercive

def dotProduct' {ι : Type*} [Fintype ι] (u v : ι → ℝ) : ℝ :=
  Finset.univ.sum (fun i => u i * v i)

theorem penalty_quadratic_tendsto_proof
    (S : Matrix ι ι ℝ) (lam : ℝ) (hlam : 0 < lam)
    (hS_posDef : ∀ v : ι → ℝ, v ≠ 0 → 0 < dotProduct' (S.mulVec v) v) :
    Filter.Tendsto
      (fun β : ι → ℝ => lam * Finset.univ.sum (fun i => β i * (S.mulVec β) i))
      (Filter.cocompact _) Filter.atTop := by
  
  -- Q(β) = βᵀ S β
  let Q := fun β : ι → ℝ => Finset.univ.sum (fun i => β i * (S.mulVec β) i)
  
  have hQ_def : ∀ β, Q β = Finset.univ.sum (fun i => β i * (S.mulVec β) i) := fun _ => rfl
  
  -- Q is continuous (it's a quadratic polynomial)
  have hQ_cont : Continuous Q := by
    apply Continuous.sum
    intro i _
    apply Continuous.mul
    · exact continuous_apply i
    · apply Continuous.finset_sum
      intro j _
      apply Continuous.mul
      · exact continuous_const
      · exact continuous_apply j
  
  -- Use PiLp 2 norm (Euclidean norm)
  letI : Norm (ι → ℝ) := PiLp.norm 2 (fun _ => ℝ)
  haveI : NormedAddCommGroup (ι → ℝ) := PiLp.normedAddCommGroup 2 (fun _ => ℝ)
  haveI : NormedSpace ℝ (ι → ℝ) := PiLp.normedSpace 2 (fun _ => ℝ)
  
  -- Handle the trivial case where ι is empty
  by_cases h_nonempty : Nonempty ι
  · -- Case: ι is nonempty
    -- The unit sphere is compact and nonempty
    let K := Metric.sphere (0 : ι → ℝ) 1
    have hK_compact : IsCompact K := Metric.isCompact_sphere 0 1
    have hK_nonempty : K.Nonempty := by
      exact Metric.sphere_nonempty.mpr zero_le_one
    
    -- Q attains a minimum c on the sphere
    rcases hK_compact.exists_isMinOn hK_nonempty hQ_cont.continuousOn with ⟨v_min, hv_min_mem, h_min⟩
    
    let c := Q v_min
    
    -- c > 0 because S is pos def and v_min is on sphere (so non-zero)
    have hc_pos : 0 < c := by
      apply hS_posDef
      intro h_zero
      rw [h_zero] at hv_min_mem
      simp at hv_min_mem -- ||0|| = 0 != 1
    
    -- Establish lower bound: Q(β) >= c * ||β||^2
    have h_bound : ∀ β : ι → ℝ, c * ‖β‖^2 ≤ Q β := by
      intro β
      by_cases hβ : β = 0
      · rw [hβ]; simp; unfold Q; simp
      · let u := (‖β‖⁻¹) • β
        have hu_norm : ‖u‖ = 1 := by
          simp only [u, norm_smul, Real.norm_eq_abs, abs_inv, abs_norm]
          rw [inv_mul_cancel]
          exact norm_ne_zero_iff.mpr hβ
        have hu_mem : u ∈ K := by simp [K, hu_norm]
        have h_ge : c ≤ Q u := h_min u hu_mem
        
        -- Expand Q u
        -- Q(k*v) = k^2 * Q(v) for bilinear form... wait Q is quadratic.
        -- Q(k*v) = (kv)ᵀ S (kv) = k^2 vᵀ S v = k^2 Q(v)
        have h_homog : Q u = (‖β‖⁻¹)^2 * Q β := by
          -- Q is homogeneous of degree 2
          -- Need to show Q (c • x) = c^2 * Q x
          unfold Q
          simp only [Matrix.mulVec_smul, Pi.smul_apply, smul_eq_mul]
          rw [Finset.sum_mul]
          apply Finset.sum_congr rfl
          intro i _
          ring
        rw [h_homog] at h_ge
        
        -- c <= ||β||^-2 * Q β  =>  c * ||β||^2 <= Q β
        rw [inv_pow] at h_ge
        have h_norm_sq_pos : 0 < ‖β‖^2 := pow_pos (norm_pos_iff.mpr hβ) 2
        rw [le_div_iff h_norm_sq_pos] at h_ge
        exact h_ge
        
    -- Now show limit
    -- We want lim (lam * Q β) = inf
    -- We know lam * Q β >= lam * c * ||β||^2
    -- And ||β|| -> inf implies ||β||^2 -> inf implies lam * c * ||β||^2 -> inf
    
    apply Filter.tendsto_atTop_mono (g := fun β => lam * c * ‖β‖^2)
    · intro β
      apply mul_le_mul_of_nonneg_left (h_bound β) (le_of_lt hlam)
    
    apply Filter.Tendsto.const_mul_atTop (mul_pos hlam hc_pos)
    apply Filter.Tendsto.pow_atTop (n := 2) (by norm_num)
    apply Filter.tendsto_norm_cocompact_atTop
    
  · -- Case: ι is empty
    -- Then ι → ℝ has only 0
    -- Filter.cocompact is ⊥
    simp only [Filter.cocompact_eq_bot]
    exact Filter.tendsto_bot
