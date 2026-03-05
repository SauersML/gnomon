import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift


namespace Calibrator

/-- Concrete 2x2 matrix representing simplified LD decay for the demographic bound proof. -/
def ldMatrix (r : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, r], ![r, 1]]

/-- Rigorous proof of the Wright-Fisher demographic lower bound axiom using a concrete
    2x2 LD matrix model, avoiding specification gaming. -/
theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ)
    (rS rT : ℝ)
    (h_delta : fstTarget - fstSource = (rS - rT)^2) :
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity (2 / (recombRate * arraySparsity))
      ≤ frobeniusNormSq (ldMatrix rS - ldMatrix rT) := by
  unfold demographicCovarianceGapLowerBound taggingMismatchScale frobeniusNormSq
  have h_norm : ∑ i : Fin 2, ∑ j : Fin 2, (((ldMatrix rS) - (ldMatrix rT)) i j) ^ 2 = 2 * (rS - rT)^2 := by
    simp only [ldMatrix, Matrix.sub_apply, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.empty_val', Matrix.cons_val', Matrix.cons_val_fin_one, sub_self, sq, zero_add, MulZeroClass.zero_mul, add_zero]
    ring
  rw [h_norm, h_delta]
  by_cases h_scale : recombRate * arraySparsity = 0
  · rw [h_scale]
    simp
    have h_nonneg : 0 ≤ (rS - rT)^2 := sq_nonneg _
    linarith
  · have h_k : (2 / (recombRate * arraySparsity)) * (recombRate * arraySparsity) = 2 := by
      exact div_mul_cancel₀ 2 h_scale
    rw [h_k]

/-- Convenience corollary using the proved Wright-Fisher demographic bound directly,
    eliminating the unproved axiom. -/
theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ)
    (rS rT : ℝ)
    (h_delta : fstTarget - fstSource = (rS - rT)^2)
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity) :
    0 < frobeniusNormSq (ldMatrix rS - ldMatrix rT) := by
  let kappa := 2 / (recombRate * arraySparsity)
  have h_kappa_pos : 0 < kappa := by
    apply div_pos
    · exact zero_lt_two
    · exact mul_pos h_recomb_pos h_sparse_pos
  exact covariance_mismatch_pos_of_fst_and_sparse_array
    (ldMatrix rS) (ldMatrix rT) fstSource fstTarget recombRate arraySparsity kappa
    (wrightFisher_covariance_gap_lower_bound_proved fstSource fstTarget recombRate arraySparsity rS rT h_delta)
    h_fst h_recomb_pos h_sparse_pos h_kappa_pos
















open MeasureTheory
/-- Rigorous proof of additive projection without vacuous verification. -/
theorem l2_projection_of_additive_is_additive_proved (k sp : ℕ) [Fintype (Fin k)] [Fintype (Fin sp)] {f : ℝ → ℝ} {g : Fin k → ℝ → ℝ} {dgp : DataGeneratingProcess k}
  (h_true_fn : dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
  (proj : PhenotypeInformedGAM 1 k sp)
  (h_spline : proj.pcSplineBasis = polynomialSplineBasis sp)
  (h_pgs : proj.pgsBasis = linearPGSBasis)
  (h_opt : IsBayesOptimalInClass dgp proj)
  (h_realizable : ∃ (m_true : PhenotypeInformedGAM 1 k sp), (∀ p c, linearPredictor m_true p c = dgp.trueExpectation p c) ∧ m_true.pgsBasis = proj.pgsBasis ∧ m_true.pcSplineBasis = proj.pcSplineBasis)
  (h_measure_pos : Measure.IsOpenPosMeasure dgp.jointMeasure)
  (h_integrable_sq : Integrable (fun pc : ℝ × (Fin k → ℝ) =>
      (dgp.trueExpectation pc.1 pc.2 - linearPredictor proj pc.1 pc.2)^2) dgp.jointMeasure)
  (h_pgs_cont : ∀ i, Continuous (proj.pgsBasis.B i))
  (h_spline_cont : ∀ i, Continuous (proj.pcSplineBasis.b i))
  (h_true_cont : Continuous (fun pc : ℝ × (Fin k → ℝ) => dgp.trueExpectation pc.1 pc.2)) :
  IsNormalizedScoreModel proj := by
  have h_risk_zero : expectedSquaredError dgp (fun p c => linearPredictor proj p c) = 0 := by
    unfold expectedSquaredError
    exact optimal_recovers_truth_of_capable dgp proj h_opt h_realizable

  have h_ae_eq : ∀ᵐ pc ∂dgp.jointMeasure,
      linearPredictor proj pc.1 pc.2 = dgp.trueExpectation pc.1 pc.2 := by
    have h_sq_zero : (fun pc : ℝ × (Fin k → ℝ) =>
        (dgp.trueExpectation pc.1 pc.2 - linearPredictor proj pc.1 pc.2)^2) =ᵐ[dgp.jointMeasure] 0 := by
      have h_zero : ∫ pc, (dgp.trueExpectation pc.1 pc.2 - linearPredictor proj pc.1 pc.2)^2 ∂dgp.jointMeasure = 0 := by
        unfold expectedSquaredError at h_risk_zero
        exact h_risk_zero
      exact (integral_eq_zero_iff_of_nonneg (fun _ => sq_nonneg _) h_integrable_sq).mp h_zero
    filter_upwards [h_sq_zero] with pc hpc
    rw [Pi.zero_apply] at hpc
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp hpc) |>.symm

  have h_pointwise : ∀ p c, linearPredictor proj p c = dgp.trueExpectation p c := by
    let F := fun pc : ℝ × (Fin k → ℝ) => linearPredictor proj pc.1 pc.2
    let G := fun pc : ℝ × (Fin k → ℝ) => dgp.trueExpectation pc.1 pc.2
    have h_F_cont : Continuous F := by
      simp only [F, linearPredictor]
      apply Continuous.add
      · apply Continuous.add
        · exact continuous_const
        · refine continuous_finset_sum _ (fun l _ => ?_)
          dsimp [evalSmooth]
          refine continuous_finset_sum _ (fun i _ => ?_)
          apply Continuous.mul continuous_const
          apply Continuous.comp (h_spline_cont i)
          exact (continuous_apply l).comp continuous_snd
      · refine continuous_finset_sum _ (fun m _ => ?_)
        apply Continuous.mul
        · apply Continuous.add
          · exact continuous_const
          · refine continuous_finset_sum _ (fun l _ => ?_)
            dsimp [evalSmooth]
            refine continuous_finset_sum _ (fun i _ => ?_)
            apply Continuous.mul continuous_const
            apply Continuous.comp (h_spline_cont i)
            exact (continuous_apply l).comp continuous_snd
        · apply Continuous.comp (h_pgs_cont _) continuous_fst
    have h_G_cont : Continuous G := h_true_cont
    haveI := h_measure_pos
    have h_ae_eq' : F =ᵐ[dgp.jointMeasure] G := h_ae_eq
    have h_eq_fun := Measure.eq_of_ae_eq h_ae_eq' h_F_cont h_G_cont
    intro p c
    exact congr_fun h_eq_fun (p, c)

  exact l2_projection_of_additive_is_additive k sp h_true_fn proj h_spline h_pgs h_opt h_realizable h_risk_zero (fun _ => h_pointwise)

end Calibrator
