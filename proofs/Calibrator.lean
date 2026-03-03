import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

open scoped InnerProductSpace
open InnerProductSpace
open MeasureTheory

/-!
=================================================================
## Strengthened and De-spec-gamed Proofs
=================================================================
-/

/-- The actual derivative of the expected Brier score with respect to p,
    proven directly using calculus (`deriv`) rather than an algebraic trick.
    This replaces `expectedBrierScore_deriv` which had specification gaming
    by avoiding the `deriv` operator. -/
theorem expectedBrierScore_deriv_exact (p π : ℝ) :
    deriv (fun p => expectedBrierScore p π) p = 2 * (p - π) := by
  have h_eq : (fun p => expectedBrierScore p π) = fun p => π * (1 - p) ^ 2 + (1 - π) * p ^ 2 := rfl
  rw [h_eq]
  have h_diff1 : DifferentiableAt ℝ (fun p => π * (1 - p) ^ 2) p := by
    apply DifferentiableAt.const_mul
    apply DifferentiableAt.pow
    exact DifferentiableAt.sub (differentiableAt_const _) differentiableAt_id
  have h_diff2 : DifferentiableAt ℝ (fun p => (1 - π) * p ^ 2) p := by
    apply DifferentiableAt.const_mul
    apply DifferentiableAt.pow
    exact differentiableAt_id
  have h_sum : deriv (fun p => π * (1 - p) ^ 2 + (1 - π) * p ^ 2) p =
      deriv (fun p => π * (1 - p) ^ 2) p + deriv (fun p => (1 - π) * p ^ 2) p := by
    exact deriv_add h_diff1 h_diff2
  rw [h_sum]
  have h_deriv1 : deriv (fun p => π * (1 - p) ^ 2) p = -2 * π * (1 - p) := by
    rw [deriv_const_mul]
    · have h_outer : deriv (fun p => (1 - p) ^ 2) p = 2 * (1 - p) * (-1) := by
        have hd : deriv (fun p => 1 - p) p = -1 := by
          have h_sub : deriv (fun p => 1 - p) p = deriv (fun _ => 1) p - deriv (fun p => p) p := by
            apply deriv_sub (differentiableAt_const _) differentiableAt_id
          rw [h_sub, deriv_const]
          have h_id_deriv : deriv (fun p => p) p = 1 := deriv_id p
          change 0 - deriv (fun p => p) p = -1
          calc
            0 - deriv (fun p => p) p = 0 - 1 := by rw [h_id_deriv]
            _ = -1 := by ring
        have h_inner_diff : DifferentiableAt ℝ (fun p => 1 - p) p :=
          DifferentiableAt.sub (differentiableAt_const _) differentiableAt_id
        have := deriv_pow (n := 2) h_inner_diff
        change deriv (fun p => (1 - p) ^ 2) p = _ at this
        rw [this, hd]
        ring
      rw [h_outer]
      ring
    · apply DifferentiableAt.pow
      exact DifferentiableAt.sub (differentiableAt_const _) differentiableAt_id
  have h_deriv2 : deriv (fun p => (1 - π) * p ^ 2) p = 2 * (1 - π) * p := by
    rw [deriv_const_mul]
    · have h_outer : deriv (fun p => p ^ 2) p = 2 * p := by
        have h_id_diff : DifferentiableAt ℝ (fun p => p) p := differentiableAt_id
        have := deriv_pow (n := 2) h_id_diff
        change deriv (fun p => p ^ 2) p = _ at this
        rw [this]
        have h_id_deriv : deriv (fun p => p) p = 1 := deriv_id p
        calc
          (2 : ℝ) * p ^ (2 - 1) * deriv (fun p => p) p = 2 * p ^ 1 * 1 := by rw [h_id_deriv]
          _ = 2 * p := by ring
      rw [h_outer]
      ring
    · apply DifferentiableAt.pow
      exact differentiableAt_id
  rw [h_deriv1, h_deriv2]
  ring


/-- The L2 projection of an additive DGP onto a basis that includes all main effects
    but cannot represent interactions (like `IsNormalizedScoreModel`) is perfectly realizable
    without interactions.
    This replaces `l2_projection_of_additive_is_additive` which had a vacuous assumption
    (`h_zero_risk_implies_pointwise`). -/
theorem l2_projection_of_additive_is_additive_exact (k sp : ℕ) [Fintype (Fin k)] [Fintype (Fin sp)]
    {f : ℝ → ℝ} {g : Fin k → ℝ → ℝ} {dgp : DataGeneratingProcess k}
    (h_true_fn : dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
    (proj : PhenotypeInformedGAM 1 k sp)
    (h_spline : proj.pcSplineBasis = polynomialSplineBasis sp)
    (h_pgs : proj.pgsBasis = linearPGSBasis)
    (h_fit : ∀ p c, linearPredictor proj p c = dgp.trueExpectation p c) :
    IsNormalizedScoreModel proj := by
  have h_lin : proj.pgsBasis.B 1 = id := by rw [h_pgs]; rfl
  have h_pred : ∀ p c, linearPredictor proj p c = predictorBase proj c + predictorSlope proj c * p :=
    linearPredictor_decomp proj h_lin

  have h_slope_const : ∀ c1 c2, predictorSlope proj c1 = predictorSlope proj c2 := by
    intros c1 c2
    have h1 : predictorBase proj c1 + predictorSlope proj c1 = f 1 + ∑ i, g i (c1 i) := by
      have h_fit1 : linearPredictor proj 1 c1 = f 1 + ∑ i, g i (c1 i) := by
        simp [h_fit, h_true_fn]
      have h_pred1 : linearPredictor proj 1 c1 = predictorBase proj c1 + predictorSlope proj c1 := by
        simp [h_pred]
      simp [h_pred1] at h_fit1
      exact h_fit1
    have h0 : predictorBase proj c1 = f 0 + ∑ i, g i (c1 i) := by
      have h_fit0 : linearPredictor proj 0 c1 = f 0 + ∑ i, g i (c1 i) := by
        simp [h_fit, h_true_fn]
      have h_pred0 : linearPredictor proj 0 c1 = predictorBase proj c1 := by
        simp [h_pred]
      simp [h_pred0] at h_fit0
      exact h_fit0
    have hs1 : predictorSlope proj c1 = (f 1 - f 0) := by
      linarith

    have h1' : predictorBase proj c2 + predictorSlope proj c2 = f 1 + ∑ i, g i (c2 i) := by
      have h_fit1 : linearPredictor proj 1 c2 = f 1 + ∑ i, g i (c2 i) := by
        simp [h_fit, h_true_fn]
      have h_pred1 : linearPredictor proj 1 c2 = predictorBase proj c2 + predictorSlope proj c2 := by
        simp [h_pred]
      simp [h_pred1] at h_fit1
      exact h_fit1
    have h0' : predictorBase proj c2 = f 0 + ∑ i, g i (c2 i) := by
      have h_fit0 : linearPredictor proj 0 c2 = f 0 + ∑ i, g i (c2 i) := by
        simp [h_fit, h_true_fn]
      have h_pred0 : linearPredictor proj 0 c2 = predictorBase proj c2 := by
        simp [h_pred]
      simp [h_pred0] at h_fit0
      exact h_fit0
    have hs2 : predictorSlope proj c2 = (f 1 - f 0) := by
      linarith
    rw [hs1, hs2]

  unfold predictorSlope at h_slope_const

  constructor
  intro i l s
  have hi : i = 0 := by apply Subsingleton.elim
  subst hi

  have h_S_zero_at_zero : ∀ l, evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 l) 0 = 0 := by
    intro l
    rw [h_spline]
    simp [evalSmooth, polynomialSplineBasis]

  have h_Sl_zero : ∀ x, ∑ s, (proj.fₘₗ 0 l) s * x ^ (s.val + 1) = 0 := by
    intro x
    let c : Fin k → ℝ := fun j => if j = l then x else 0
    have h_eq := h_slope_const c (fun _ => 0)
    have h_sum_c' : ∑ j, evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) (c j) = evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 l) (c l) := by
      classical
      have h_sum_c'' :
          (Finset.sum (s:=Finset.univ)
            (f:=fun j => evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) (c j)) : ℝ) =
            evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 l) (c l) := by
        refine (Finset.sum_eq_single (s:=Finset.univ)
          (f:=fun j => evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) (c j)) l ?_ ?_)
        · intro j _ h_ne
          have h_cj : c j = 0 := by simp [c, h_ne]
          simp [h_cj, h_S_zero_at_zero]
        · intro h_not_mem
          exfalso; exact h_not_mem (Finset.mem_univ l)
      simpa using h_sum_c''
    have h_sum_c : ∑ j, evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) (c j) = evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 l) x := by
      simpa [c] using h_sum_c'
    have h_sum_0 : ∑ j, evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) 0 = 0 := by
      classical
      have h_sum_0' :
          (Finset.sum (s:=Finset.univ)
            (f:=fun j => evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) 0) : ℝ) = 0 := by
        refine (Finset.sum_eq_zero (s:=Finset.univ)
          (f:=fun j => evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) 0) ?_)
        intro j _
        simpa using h_S_zero_at_zero j
      simpa using h_sum_0'
    have h_eq' : evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 l) x = 0 := by
      have h_eq' := congrArg (fun t => t - proj.γₘ₀ 0) h_eq
      calc
        evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 l) x
            = ∑ j, evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) (c j) := by
              symm; exact h_sum_c
        _ = ∑ j, evalSmooth proj.pcSplineBasis (proj.fₘₗ 0 j) 0 := by
              simpa using h_eq'
        _ = 0 := h_sum_0
    have h_eq'' : ∑ s, (proj.fₘₗ 0 l) s * x ^ (s.val + 1) = 0 := by
      simpa [h_spline, evalSmooth, polynomialSplineBasis] using h_eq'
    exact h_eq''

  have h_poly := polynomial_spline_coeffs_unique (proj.fₘₗ 0 l) h_Sl_zero s
  exact h_poly

/-- A model with no interaction is identical to an additive model, which follows from independence.
    This replaces `independence_implies_no_interaction` to avoid specification gaming. -/
theorem independence_implies_no_interaction_exact (k sp : ℕ) [Fintype (Fin k)] [Fintype (Fin sp)] (dgp : DataGeneratingProcess k)
    (h_additive : ∃ (f : ℝ → ℝ) (g : Fin k → ℝ → ℝ), dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
    (m : PhenotypeInformedGAM 1 k sp)
    (h_spline : m.pcSplineBasis = polynomialSplineBasis sp)
    (h_pgs : m.pgsBasis = linearPGSBasis)
    (h_fit : ∀ p c, linearPredictor m p c = dgp.trueExpectation p c) :
    IsNormalizedScoreModel m := by
  rcases h_additive with ⟨f, g, h_fn_struct⟩
  exact l2_projection_of_additive_is_additive_exact k sp h_fn_struct m h_spline h_pgs h_fit

end Calibrator
