import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift


namespace Calibrator

theorem l2_projection_of_additive_is_additive_proved (k sp : ℕ) [Fintype (Fin k)] [Fintype (Fin sp)] {f : ℝ → ℝ} {g : Fin k → ℝ → ℝ} {dgp : DataGeneratingProcess k}
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
        simpa [h_pred]
      rw [h_pred1] at h_fit1
      exact h_fit1
    have h0 : predictorBase proj c1 = f 0 + ∑ i, g i (c1 i) := by
      have h_fit0 : linearPredictor proj 0 c1 = f 0 + ∑ i, g i (c1 i) := by
        simp [h_fit, h_true_fn]
      have h_pred0 : linearPredictor proj 0 c1 = predictorBase proj c1 := by
        simpa [h_pred]
      rw [h_pred0] at h_fit0
      exact h_fit0
    have hs1 : predictorSlope proj c1 = (f 1 - f 0) := by
      linarith

    have h1' : predictorBase proj c2 + predictorSlope proj c2 = f 1 + ∑ i, g i (c2 i) := by
      have h_fit1 : linearPredictor proj 1 c2 = f 1 + ∑ i, g i (c2 i) := by
        simp [h_fit, h_true_fn]
      have h_pred1 : linearPredictor proj 1 c2 = predictorBase proj c2 + predictorSlope proj c2 := by
        simpa [h_pred]
      rw [h_pred1] at h_fit1
      exact h_fit1
    have h0' : predictorBase proj c2 = f 0 + ∑ i, g i (c2 i) := by
      have h_fit0 : linearPredictor proj 0 c2 = f 0 + ∑ i, g i (c2 i) := by
        simp [h_fit, h_true_fn]
      have h_pred0 : linearPredictor proj 0 c2 = predictorBase proj c2 := by
        simpa [h_pred]
      rw [h_pred0] at h_fit0
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

theorem independence_implies_no_interaction_proved (k sp : ℕ) [Fintype (Fin k)] [Fintype (Fin sp)] (dgp : DataGeneratingProcess k)
    (h_additive : ∃ (f : ℝ → ℝ) (g : Fin k → ℝ → ℝ), dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
    (m : PhenotypeInformedGAM 1 k sp)
    (h_spline : m.pcSplineBasis = polynomialSplineBasis sp)
    (h_pgs : m.pgsBasis = linearPGSBasis)
    (h_fit : ∀ p c, linearPredictor m p c = dgp.trueExpectation p c) :
    IsNormalizedScoreModel m := by
  rcases h_additive with ⟨f, g, h_fn_struct⟩
  exact l2_projection_of_additive_is_additive_proved k sp h_fn_struct m h_spline h_pgs h_fit

theorem context_specificity_proved {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (dgp1 dgp2 : DGPWithEnvironment k)
    (h_same_genetics : dgp1.trueGeneticEffect = dgp2.trueGeneticEffect ∧ dgp1.to_dgp.jointMeasure = dgp2.to_dgp.jointMeasure)
    (h_diff_env : dgp1.environmentalEffect ≠ dgp2.environmentalEffect)
    (model1 : PhenotypeInformedGAM p k sp)
    (h_fit1 : ∀ p c, linearPredictor model1 p c = dgp1.to_dgp.trueExpectation p c) :
  ¬ (∀ p c, linearPredictor model1 p c = dgp2.to_dgp.trueExpectation p c) := by
  intro h_fit2
  have h_eq_fn : dgp1.to_dgp.trueExpectation = dgp2.to_dgp.trueExpectation := by
    ext p c
    rw [← h_fit1 p c, h_fit2 p c]
  rw [dgp1.is_additive_causal, dgp2.is_additive_causal, h_same_genetics.1] at h_eq_fn
  have : dgp1.environmentalEffect = dgp2.environmentalEffect := by
    ext c
    have := congr_fun (congr_fun h_eq_fn 0) c
    simp at this; exact this
  exact h_diff_env this

end Calibrator
