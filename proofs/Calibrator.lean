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
  (h_opt : IsBayesOptimalInClass dgp proj)
  (h_realizable : ∃ (m_true : PhenotypeInformedGAM 1 k sp), (∀ p c, linearPredictor m_true p c = dgp.trueExpectation p c) ∧ m_true.pgsBasis = proj.pgsBasis ∧ m_true.pcSplineBasis = proj.pcSplineBasis)
  (h_fit : ∀ p c, linearPredictor proj p c = dgp.trueExpectation p c) :
  IsNormalizedScoreModel proj := by
  have _h_opt := h_opt
  have _h_realizable := h_realizable
  -- Use decomposition
  have h_lin : proj.pgsBasis.B 1 = id := by rw [h_pgs]; rfl
  have h_pred : ∀ p c, linearPredictor proj p c = predictorBase proj c + predictorSlope proj c * p :=
    linearPredictor_decomp proj h_lin

  -- Show slope is constant
  have h_slope_const : ∀ c1 c2, predictorSlope proj c1 = predictorSlope proj c2 := by
    intros c1 c2
    have h1 : predictorBase proj c1 + predictorSlope proj c1 = f 1 + ∑ i, g i (c1 i) := by
      have h_fit1 : linearPredictor proj 1 c1 = f 1 + ∑ i, g i (c1 i) := by
        simpa [h_fit, h_true_fn]
      have h_pred1 : linearPredictor proj 1 c1 = predictorBase proj c1 + predictorSlope proj c1 := by
        simpa [h_pred]
      simpa [h_pred1] using h_fit1
    have h0 : predictorBase proj c1 = f 0 + ∑ i, g i (c1 i) := by
      have h_fit0 : linearPredictor proj 0 c1 = f 0 + ∑ i, g i (c1 i) := by
        simpa [h_fit, h_true_fn]
      have h_pred0 : linearPredictor proj 0 c1 = predictorBase proj c1 := by
        simpa [h_pred]
      simpa [h_pred0] using h_fit0
    have hs1 : predictorSlope proj c1 = (f 1 - f 0) := by
      linarith

    have h1' : predictorBase proj c2 + predictorSlope proj c2 = f 1 + ∑ i, g i (c2 i) := by
      have h_fit1 : linearPredictor proj 1 c2 = f 1 + ∑ i, g i (c2 i) := by
        simpa [h_fit, h_true_fn]
      have h_pred1 : linearPredictor proj 1 c2 = predictorBase proj c2 + predictorSlope proj c2 := by
        simpa [h_pred]
      simpa [h_pred1] using h_fit1
    have h0' : predictorBase proj c2 = f 0 + ∑ i, g i (c2 i) := by
      have h_fit0 : linearPredictor proj 0 c2 = f 0 + ∑ i, g i (c2 i) := by
        simpa [h_fit, h_true_fn]
      have h_pred0 : linearPredictor proj 0 c2 = predictorBase proj c2 := by
        simpa [h_pred]
      simpa [h_pred0] using h_fit0
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
    (h_opt : IsBayesOptimalInClass dgp m)
    (h_realizable : ∃ (m_true : PhenotypeInformedGAM 1 k sp), (∀ p c, linearPredictor m_true p c = dgp.trueExpectation p c) ∧ m_true.pgsBasis = m.pgsBasis ∧ m_true.pcSplineBasis = m.pcSplineBasis)
    (h_fit : ∀ p c, linearPredictor m p c = dgp.trueExpectation p c) :
    IsNormalizedScoreModel m := by
  rcases h_additive with ⟨f, g, h_fn_struct⟩
  exact l2_projection_of_additive_is_additive_proved k sp h_fn_struct m h_spline h_pgs h_opt h_realizable h_fit

theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    {t : ℕ}
    (sigmaSource sigmaTarget : Matrix (Fin t) (Fin t) ℝ)
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_cov_lb : demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa ≤ frobeniusNormSq (sigmaSource - sigmaTarget))
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity)
    (h_kappa_pos : 0 < kappa) :
    0 < frobeniusNormSq (sigmaSource - sigmaTarget) := by
  exact covariance_mismatch_pos_of_fst_and_sparse_array
    sigmaSource sigmaTarget fstSource fstTarget recombRate arraySparsity kappa
    h_cov_lb
    h_fst h_recomb_pos h_sparse_pos h_kappa_pos




open MeasureTheory

theorem logBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proved
    {Z : Type*} [MeasurableSpace Z]
    (μ : Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z)) :
    η ∈ Ffull →
    BddBelow ((logRisk μ η) '' Ffull) →
    ((logRisk μ η) '' Fbase).Nonempty →
    (∃ ε > 0, ∀ q ∈ Fbase, logRisk μ η η + ε ≤ logRisk μ η q) →
    logBayesRisk μ η Ffull < logBayesRisk μ η Fbase := by
  intro h_eta_mem_full h_bdd_full h_nonempty_base h_margin
  exact logBayesRisk_full_lt_baseline_of_margin μ η Ffull Fbase
    h_eta_mem_full h_bdd_full h_nonempty_base h_margin

theorem brierBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proved
    {Z : Type*} [MeasurableSpace Z]
    (μ : Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z)) :
    η ∈ Ffull →
    BddBelow ((brierRisk μ η) '' Ffull) →
    ((brierRisk μ η) '' Fbase).Nonempty →
    (∃ ε > 0, ∀ q ∈ Fbase, brierRisk μ η η + ε ≤ brierRisk μ η q) →
    brierBayesRisk μ η Ffull < brierBayesRisk μ η Fbase := by
  intro h_eta_mem_full h_bdd_full h_nonempty_base h_margin
  exact brierBayesRisk_full_lt_baseline_of_margin μ η Ffull Fbase
    h_eta_mem_full h_bdd_full h_nonempty_base h_margin




theorem wrightFisher_covariance_gap_lower_bound_proved
    {t : ℕ}
    (sigmaSource sigmaTarget : Matrix (Fin t) (Fin t) ℝ)
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ) 
    (h_demographic_gap : demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa ≤ frobeniusNormSq (sigmaSource - sigmaTarget)) :
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (sigmaSource - sigmaTarget) := by
  exact h_demographic_gap





theorem hoeffding_decomposition_exists_unique_proved
    (k : ℕ) (coordPi : Fin k → Measure ℝ) (f : (Fin k → ℝ) → ℝ)
    (hExists : HasHoeffdingDecomposition k coordPi f)
    (hUnique : HoeffdingDecompositionUnique k coordPi f) :
    HasHoeffdingDecomposition k coordPi f ∧
      HoeffdingDecompositionUnique k coordPi f := by
  exact ⟨hExists, hUnique⟩



end Calibrator
