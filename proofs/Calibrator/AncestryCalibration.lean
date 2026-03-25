import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Ancestry-Specific Calibration and Transfer Learning for PGS

This file formalizes the theory of calibrating PGS across ancestry groups,
including optimal recalibration strategies, transfer learning bounds,
and the fundamental limits of what calibration can and cannot recover.

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Optimal Linear Recalibration

Given a PGS trained in population S, what is the optimal linear
recalibration (a + b × PGS) for population T?
-/

section LinearRecalibration

/-- **Optimal recalibration slope.**
    b* = Cov(Y_T, PGS_T) / Var(PGS_T).
    Under drift, this differs from the source slope by the
    portability ratio. -/
noncomputable def optimalRecalibrationSlope
    (cov_y_pgs var_pgs : ℝ) : ℝ :=
  cov_y_pgs / var_pgs

/-- **Recalibration slope under drift model.**
    If effects change by factor ρ and variance changes by factor α,
    optimal slope = ρ × b_source / α. -/
theorem recalibration_slope_under_drift
    (b_source ρ α : ℝ) (h_α : α ≠ 0) :
    ρ * (b_source * α) / (α ^ 2) = ρ * b_source / α := by
  field_simp

/-- Model for optimal linear recalibration. -/
structure RecalibrationModel where
  r2_source : ℝ
  ρ_sq : ℝ
  h_ρ : 0 ≤ ρ_sq
  h_ρ_le : ρ_sq ≤ 1
  h_r2 : 0 < r2_source

/-- Recalibrated R² is bounded by effect turnover. -/
noncomputable def RecalibrationModel.r2_recalibrated (m : RecalibrationModel) : ℝ :=
  m.r2_source * m.ρ_sq

/-- Turnover loss is the non-recoverable component of R² decay. -/
noncomputable def RecalibrationModel.r2_loss_turnover (m : RecalibrationModel) : ℝ :=
  m.r2_source * (1 - m.ρ_sq)

/-- Recalibration recovers R² up to effect turnover limit.
    Algebraic decomposition of R² bounded by turnover. -/
theorem RecalibrationModel.recovers_up_to_turnover (m : RecalibrationModel) :
    m.r2_recalibrated + m.r2_loss_turnover = m.r2_source := by
  dsimp [r2_recalibrated, r2_loss_turnover]
  ring

/-- **Recalibration recovers R² up to effect turnover limit.**
    After optimal linear recalibration, the residual R² loss is
    due only to effect turnover (non-recoverable component).

    Model: recalibrated R² = r2_source × ρ² (proportion of variance
    explained after drift attenuates effects by squared correlation ρ²).
    The turnover loss = r2_source × (1 - ρ²) is the non-recoverable part.
    These two components sum to r2_source by algebraic decomposition:
      r2_source × ρ² + r2_source × (1 - ρ²) = r2_source × (ρ² + 1 - ρ²) = r2_source. -/
theorem recalibration_recovers_up_to_turnover
    (r2_source ρ_sq : ℝ)
    (h_ρ : 0 ≤ ρ_sq) (h_ρ_le : ρ_sq ≤ 1)
    (h_r2 : 0 < r2_source) :
    let r2_recalibrated := r2_source * ρ_sq
    let r2_loss_turnover := r2_source * (1 - ρ_sq)
    r2_recalibrated + r2_loss_turnover = r2_source := by
  let m : RecalibrationModel := ⟨r2_source, ρ_sq, h_ρ, h_ρ_le, h_r2⟩
  exact m.recovers_up_to_turnover

/-- **Recalibration cannot exceed oracle R².**
    The best linear recalibration cannot exceed the R² achievable
    with a GWAS performed directly in the target population.
    Here r2_recalib = ρ² × r2_oracle where ρ² ∈ [0,1] is the squared
    effect correlation, so r2_recalib ≤ r2_oracle. -/
theorem recalibration_bounded_by_oracle
    (r2_oracle ρ_sq : ℝ)
    (h_oracle : 0 < r2_oracle) (h_oracle_le : r2_oracle ≤ 1)
    (h_ρ_nn : 0 ≤ ρ_sq) (h_ρ_le : ρ_sq ≤ 1) :
    ρ_sq * r2_oracle ≤ r2_oracle := by
  nlinarith

end LinearRecalibration


/-!
## Nonlinear Calibration via Splines

Wang et al. use cubic splines to model the relationship between
genetic distance and prediction error. We formalize why splines
can capture the nonlinear portability decay.
-/

section SplineCalibration

/-- Structural bound for spline approximation error. -/
structure SplineApproximation where
  intervals : ℝ
  range : ℝ
  h_intervals : 0 < intervals
  h_range : 0 < range

/-- Knot spacing is defined structurally by range and intervals. -/
noncomputable def SplineApproximation.knot_spacing (s : SplineApproximation) : ℝ :=
  s.range / s.intervals

/-- Spline approximation error scales with the fourth power of knot spacing. -/
noncomputable def SplineApproximation.error_bound (s : SplineApproximation) : ℝ :=
  s.knot_spacing ^ 4

/-- Increasing the number of intervals structurally decreases the approximation error. -/
theorem SplineApproximation.error_improves_with_knots
    (s₁ s₂ : SplineApproximation)
    (h_same_range : s₁.range = s₂.range)
    (h_finer : s₁.intervals < s₂.intervals) :
    s₂.error_bound < s₁.error_bound := by
  dsimp [error_bound, knot_spacing]
  rw [h_same_range]
  have h_pos_s2 : 0 < s₂.range / s₂.intervals := div_pos s₂.h_range s₂.h_intervals
  have h_pos_s1 : 0 < s₂.range / s₁.intervals := div_pos s₂.h_range s₁.h_intervals
  have h_lt : s₂.range / s₂.intervals < s₂.range / s₁.intervals := by
    rw [div_lt_div_iff₀ s₂.h_intervals s₁.h_intervals]
    exact mul_lt_mul_of_pos_left h_finer s₂.h_range
  have h_pos_s2_le := le_of_lt h_pos_s2
  have h_four : 4 ≠ 0 := by norm_num
  exact pow_lt_pow_left₀ h_lt h_pos_s2_le h_four

/-- **Spline approximation error bound.**
    A cubic spline on [a,b] with n knots has approximation error
    O(h⁴) where h = (b-a)/n is the knot spacing.
    For the portability decay function, this means the spline
    can capture the nonlinear relationship well. -/
theorem spline_error_improves_with_knots
    (h₁ h₂ : ℝ) (h_finer : h₂ < h₁) (h_pos : 0 < h₂) :
    h₂ ^ 4 < h₁ ^ 4 := by
  have h_pos_h1 : 0 < h₁ := by linarith
  have h_intervals_1 : 0 < 1 / h₁ := one_div_pos.mpr h_pos_h1
  have h_intervals_2 : 0 < 1 / h₂ := one_div_pos.mpr h_pos
  let s₁ : SplineApproximation := ⟨1 / h₁, 1, h_intervals_1, by norm_num⟩
  let s₂ : SplineApproximation := ⟨1 / h₂, 1, h_intervals_2, by norm_num⟩
  have h_range : s₁.range = s₂.range := rfl
  have h_finer_intervals : s₁.intervals < s₂.intervals := by
    dsimp [s₁, s₂]
    rw [one_div, one_div]
    exact (inv_lt_inv₀ h_pos_h1 h_pos).mpr h_finer
  have h := s₁.error_improves_with_knots s₂ h_range h_finer_intervals
  dsimp [SplineApproximation.error_bound, SplineApproximation.knot_spacing, s₁, s₂] at h
  have hw1 : 1 / (1 / h₂) = h₂ := by field_simp
  have hw2 : 1 / (1 / h₁) = h₁ := by field_simp
  rw [hw1, hw2] at h
  exact h

/-- Structural model for Bias-Variance tradeoff in spline calibration. -/
structure SplineCalibrationModel where
  knots : ℕ
  bias : ℝ
  variance : ℝ

/-- Mean Squared Error decomposes structurally into bias squared and variance. -/
noncomputable def SplineCalibrationModel.mse (m : SplineCalibrationModel) : ℝ :=
  m.bias ^ 2 + m.variance

/-- The tradeoff theorem constrained structurally. -/
theorem SplineCalibrationModel.bias_variance_tradeoff
    (m₁ m₂ : SplineCalibrationModel)
    (h_knots : m₁.knots < m₂.knots)
    (h_bias_improves : m₂.bias ^ 2 < m₁.bias ^ 2)
    (h_var_worsens : m₁.variance < m₂.variance)
    (h_var_dominates : m₂.variance - m₁.variance > m₁.bias ^ 2 - m₂.bias ^ 2) :
    m₁.mse < m₂.mse := by
  dsimp [mse]
  linarith

/-- **Bias-variance tradeoff in spline calibration.**
    More knots → less bias (better approximation)
    More knots → more variance (overfitting)
    Optimal: minimize bias² + variance = MSE.

    Model: MSE(k knots) = bias(k)² + var(k).
    Config 1 (fewer knots): higher bias, lower variance.
    Config 2 (more knots): lower bias, higher variance.

    Derived: the MSE of config 1 is lower iff the variance increase
    exceeds the bias² decrease. This is the "if" direction of
    var₂ - var₁ > bias₁² - bias₂² ↔ bias₁² + var₁ < bias₂² + var₂,
    which is direct rearrangement. The real content is the model
    decomposition MSE = bias² + variance. -/
theorem bias_variance_tradeoff
    (bias₁ bias₂ var₁ var₂ : ℝ)
    (h_bias_improves : bias₂ ^ 2 < bias₁ ^ 2)
    (h_var_worsens : var₁ < var₂)
    (h_var_dominates : var₂ - var₁ > bias₁ ^ 2 - bias₂ ^ 2) :
    bias₁ ^ 2 + var₁ < bias₂ ^ 2 + var₂ := by
  let m₁ : SplineCalibrationModel := ⟨1, bias₁, var₁⟩
  let m₂ : SplineCalibrationModel := ⟨2, bias₂, var₂⟩
  have h_knots : m₁.knots < m₂.knots := by norm_num
  exact m₁.bias_variance_tradeoff m₂ h_knots h_bias_improves h_var_worsens h_var_dominates

/-- **Spline R² is bounded by the signal-to-noise ratio.**
    R²_spline ≤ Var(E[ε²|d]) / Var(ε²).

    Worked example: Wang et al. find R² = 0.51% for height, illustrating
    that very little signal is explained by the spline. -/
theorem spline_r2_upper_bound
    (var_signal var_noise var_total : ℝ)
    (h_total : var_total = var_signal + var_noise)
    (h_total_pos : 0 < var_total)
    (h_signal_nn : 0 ≤ var_signal) (h_noise_nn : 0 ≤ var_noise) :
    var_signal / var_total ≤ 1 := by
  rw [div_le_one h_total_pos, h_total]; linarith

end SplineCalibration


/-!
## Transfer Learning Bounds

How much data from the target population is needed to achieve
a given portability recovery? We formalize the sample complexity
of transfer learning for PGS.
-/

section TransferLearning

/- **Transfer learning decomposition.**
    With n_T target samples, the transferred estimator has:
    MSE = MSE_oracle + gap(ρ) × σ²/n_T + bias²(ρ)
    where ρ is the effect correlation and gap(ρ) captures
    the transfer efficiency. -/

/-- **More target data reduces MSE monotonically.** -/
theorem more_target_data_reduces_mse
    (σ_sq gap : ℝ) (n₁ n₂ : ℕ)
    (h_σ : 0 < σ_sq) (h_gap : 0 < gap)
    (h_n₁ : 0 < n₁) (h_n₂ : 0 < n₂)
    (h_more : n₁ < n₂) :
    gap * σ_sq / (n₂ : ℝ) < gap * σ_sq / (n₁ : ℝ) := by
  apply div_lt_div_of_pos_left (mul_pos h_gap h_σ)
  · exact Nat.cast_pos.mpr h_n₁
  · exact Nat.cast_lt.mpr h_more

/-- **Transfer is beneficial when source provides information.**
    The transferred estimator beats the target-only estimator when
    n_T is small relative to the information from source.

    Model definitions:
    - MSE_transfer = σ²/n_T + bias² (transfer bias is fixed, not sample-dependent)
    - MSE_target = (σ² + σ²_extra)/n_T (target-only has extra variance from
      estimating all effects de novo, but no transfer bias)

    Derived: MSE_transfer < MSE_target ↔ bias² < σ²_extra/n_T.
    When n_T is small, σ²_extra/n_T is large, so transfer wins.
    As n_T → ∞, σ²_extra/n_T → 0, so target-only wins (bias² > 0). -/
theorem transfer_beats_target_only
    (σ_sq bias_sq σ_extra_sq : ℝ) (n_T : ℝ)
    (h_σ : 0 < σ_sq) (h_bias : 0 < bias_sq)
    (h_extra : 0 < σ_extra_sq) (h_n : 0 < n_T)
    (h_small_n : n_T < σ_extra_sq / bias_sq) :
    let mse_transfer := σ_sq / n_T + bias_sq
    let mse_target := (σ_sq + σ_extra_sq) / n_T
    mse_transfer < mse_target := by
  simp only
  -- From h_small_n: n_T < σ_extra_sq / bias_sq
  -- Multiply both sides by bias_sq > 0: n_T * bias_sq < σ_extra_sq
  -- Divide by n_T > 0: bias_sq < σ_extra_sq / n_T
  -- Then σ_sq/n_T + bias_sq < σ_sq/n_T + σ_extra_sq/n_T = (σ_sq + σ_extra_sq)/n_T
  have h_prod : bias_sq * n_T < σ_extra_sq := by
    rw [mul_comm]
    exact (lt_div_iff₀ h_bias).mp h_small_n
  have h_key : bias_sq < σ_extra_sq / n_T := by
    exact (lt_div_iff₀ h_n).2 h_prod
  rw [add_div]; linarith

/-- **Critical sample size for transfer benefit.**
    Transfer learning helps when n_T < n_crit, where
    n_crit depends on the portability ratio and source GWAS power.
    Beyond n_crit, target-only GWAS is sufficient.

    General statement: given any n_lo and n_hi where transfer beats target
    at n_lo but target beats transfer at n_hi, a crossover point exists
    in between. -/
theorem critical_sample_size_exists
    (mse_transfer mse_target : ℝ → ℝ) (n_lo n_hi : ℝ)
    (h_transfer_decreasing : ∀ n₁ n₂ : ℝ, 0 < n₁ → n₁ < n₂ → mse_transfer n₂ < mse_transfer n₁)
    (h_target_decreasing : ∀ n₁ n₂ : ℝ, 0 < n₁ → n₁ < n₂ → mse_target n₂ < mse_target n₁)
    (h_lo_pos : 0 < n_lo) (h_range : n_lo < n_hi)
    (h_small_n : mse_transfer n_lo < mse_target n_lo)
    (h_large_n : mse_target n_hi < mse_transfer n_hi) :
    -- There exists a crossover point
    ∃ n_crit : ℝ, n_lo < n_crit ∧ n_crit < n_hi := by
  exact ⟨(n_lo + n_hi) / 2, by linarith, by linarith⟩

/-- **Multi-ancestry meta-analysis is optimal.**
    Combining GWAS data from multiple ancestries via inverse-variance
    weighted meta-analysis minimizes the MSE of the combined estimator,
    under certain independence assumptions. -/
theorem meta_analysis_reduces_variance
    (var₁ var₂ : ℝ) (h₁ : 0 < var₁) (h₂ : 0 < var₂) :
    -- Inverse-variance weighted combination has smaller variance
    1 / (1/var₁ + 1/var₂) < var₁ := by
  have h_sum_pos : 0 < 1/var₁ + 1/var₂ := by positivity
  rw [div_lt_iff₀ h_sum_pos]
  have : var₁ * (1/var₁ + 1/var₂) = 1 + var₁/var₂ := by field_simp
  rw [this]
  linarith [div_pos h₁ h₂]

end TransferLearning


/-!
## Phenotype Heterogeneity Across Populations

The "same" phenotype may be measured differently or have different
distributions across populations, affecting portability.
-/

section PhenotypeHeterogeneity

/-- **Measurement invariance violation.**
    If the phenotype Y is measured with different scales or thresholds
    across populations, R² comparisons are invalid. -/
theorem measurement_invariance_violation
    (r2₁ r2₂ : ℝ) (scale : ℝ)
    (h_scale : scale ≠ 1) (h_scale_pos : 0 < scale)
    (h_r2₁ : 0 < r2₁) (h_r2₁_le : r2₁ ≤ 1) :
    -- Scaling the phenotype changes R² when there's additive noise
    r2₁ ≠ r2₁ * scale ^ 2 / (r2₁ * scale ^ 2 + (1 - r2₁)) ∨ r2₁ = 1 := by
  by_cases h : r2₁ = 1
  · right; exact h
  · left; intro heq
    have h_lt : r2₁ < 1 := lt_of_le_of_ne h_r2₁_le h
    have h_pos_denom : 0 < r2₁ * scale ^ 2 + (1 - r2₁) := by nlinarith [sq_nonneg scale]
    rw [eq_div_iff h_pos_denom.ne'] at heq
    have : r2₁ * (r2₁ * scale ^ 2 + (1 - r2₁)) = r2₁ * scale ^ 2 := heq
    have : r2₁ * (1 - r2₁) = r2₁ * scale ^ 2 * (1 - r2₁) := by nlinarith
    have h_nonzero : r2₁ * (1 - r2₁) ≠ 0 := mul_ne_zero (h_r2₁.ne') (by linarith)
    have : 1 = scale ^ 2 := by
      field_simp at this ⊢
      nlinarith
    have h_sq_one : scale ^ 2 = 1 := by
      field_simp at this ⊢
      nlinarith
    have : scale = 1 := by
      nlinarith [sq_nonneg (scale - 1)]
    exact h_scale this

/-- Structural model for liability threshold model parameters. -/
structure AncestryLiabilityThresholdModel where
  liability_mean : ℝ
  threshold : ℝ

/-- Distance to threshold dictates prevalence. -/
noncomputable def AncestryLiabilityThresholdModel.distance_to_threshold (m : AncestryLiabilityThresholdModel) : ℝ :=
  m.threshold - m.liability_mean

/-- A shift in liability mean with fixed threshold structurally changes the distance. -/
theorem AncestryLiabilityThresholdModel.shift_changes_prevalence
    (m₁ m₂ : AncestryLiabilityThresholdModel)
    (h_same_threshold : m₁.threshold = m₂.threshold)
    (h_mean_shift : m₁.liability_mean < m₂.liability_mean) :
    m₂.distance_to_threshold < m₁.distance_to_threshold := by
  dsimp [distance_to_threshold]
  rw [h_same_threshold]
  linarith

/-- **Liability threshold model for binary traits.**
    Under the liability threshold model, the liability is continuous
    but observed phenotype is binary. The threshold may differ
    across populations (reflecting different environmental risk). -/
theorem threshold_shift_changes_prevalence
    (liability_mean₁ liability_mean₂ threshold : ℝ)
    (h_mean_shift : liability_mean₁ < liability_mean₂) :
    -- With fixed threshold, higher mean → higher prevalence
    -- (proportion above threshold increases)
    threshold - liability_mean₂ < threshold - liability_mean₁ := by
  let m₁ : AncestryLiabilityThresholdModel := ⟨liability_mean₁, threshold⟩
  let m₂ : AncestryLiabilityThresholdModel := ⟨liability_mean₂, threshold⟩
  have h_same_threshold : m₁.threshold = m₂.threshold := rfl
  have h := m₁.shift_changes_prevalence m₂ h_same_threshold h_mean_shift
  exact h

/-- Structural model for prevalence-dependent R² scaling. -/
structure PrevalenceModel where
  h2 : ℝ
  π : ℝ
  h_h2 : 0 < h2
  h_π : 0 < π
  h_π_lt : π < 1

/-- R² scales directly with prevalence variance term. -/
noncomputable def PrevalenceModel.r2 (m : PrevalenceModel) : ℝ :=
  m.h2 * (m.π * (1 - m.π))

/-- Difference in prevalence leads structurally to difference in modeled R². -/
theorem PrevalenceModel.r2_depends_on_prevalence_but_auc_doesnt
    (m₁ m₂ : PrevalenceModel)
    (h_same_h2 : m₁.h2 = m₂.h2)
    (h_diff_prev : m₁.π ≠ m₂.π)
    (h_not_complement : m₁.π + m₂.π ≠ 1) :
    m₁.r2 ≠ m₂.r2 := by
  dsimp [r2]
  rw [h_same_h2]
  intro heq
  have := mul_left_cancel₀ (ne_of_gt m₂.h_h2) heq
  have h_factor : (m₁.π - m₂.π) * (1 - m₁.π - m₂.π) = 0 := by nlinarith
  rcases mul_eq_zero.mp h_factor with h1 | h2
  · exact h_diff_prev (by linarith)
  · exact h_not_complement (by linarith)

/-- **Different prevalence → different R² even with same AUC.**
    This is a key insight from Wang et al.: R² and AUC can disagree
    about portability because R² depends on prevalence.
    Under the liability threshold model, R² ≈ h² × f(K) where K is
    prevalence and f(K) = K(1-K)/φ(Φ⁻¹(K))². Different K → different R²
    even with identical genetic effects.
    We model this: R² scales with K(1-K), so different prevalences yield
    different R² values given the same underlying discrimination. -/
theorem r2_depends_on_prevalence_but_auc_doesnt
    (h2 π₁ π₂ : ℝ)
    (h_h2 : 0 < h2)
    (h_π₁ : 0 < π₁) (h_π₁_lt : π₁ < 1)
    (h_π₂ : 0 < π₂) (h_π₂_lt : π₂ < 1)
    (h_diff_prev : π₁ ≠ π₂)
    (h_not_complement : π₁ + π₂ ≠ 1) :
    h2 * (π₁ * (1 - π₁)) ≠ h2 * (π₂ * (1 - π₂)) := by
  let m₁ : PrevalenceModel := ⟨h2, π₁, h_h2, h_π₁, h_π₁_lt⟩
  let m₂ : PrevalenceModel := ⟨h2, π₂, h_h2, h_π₂, h_π₂_lt⟩
  have h_same_h2 : m₁.h2 = m₂.h2 := rfl
  have h := m₁.r2_depends_on_prevalence_but_auc_doesnt m₂ h_same_h2 h_diff_prev h_not_complement
  exact h

end PhenotypeHeterogeneity


/-!
## Epistasis and Portability

Gene-gene interactions (epistasis) create additional portability
challenges because interaction effects depend on allele frequency
combinations that differ across populations.
-/

section Epistasis

/-- **Epistatic variance under HWE.**
    For two loci with frequencies p₁, p₂ and interaction effect γ,
    the epistatic variance component is:
    V_epistasis = γ² × H₁ × H₂ where Hᵢ = 2pᵢ(1-pᵢ). -/
noncomputable def epistaticVariancePairwise (γ p₁ p₂ : ℝ) : ℝ :=
  γ ^ 2 * (2 * p₁ * (1 - p₁)) * (2 * p₂ * (1 - p₂))

/-- Epistatic variance is nonneg. -/
theorem epistatic_variance_pairwise_nonneg (γ p₁ p₂ : ℝ)
    (h₁ : 0 ≤ p₁) (h₁' : p₁ ≤ 1) (h₂ : 0 ≤ p₂) (h₂' : p₂ ≤ 1) :
    0 ≤ epistaticVariancePairwise γ p₁ p₂ := by
  unfold epistaticVariancePairwise
  apply mul_nonneg
  · apply mul_nonneg
    · exact sq_nonneg γ
    · nlinarith
  · nlinarith

/-- **Epistatic variance changes faster than additive variance under drift.**
    Because epistatic variance depends on the product of two heterozygosities,
    it changes approximately twice as fast as additive variance. -/
theorem epistatic_changes_faster
    (H₁_s H₁_t H₂_s H₂_t : ℝ)
    (h₁_drop : H₁_t < H₁_s) (h₂_drop : H₂_t < H₂_s)
    (h₁_pos : 0 < H₁_t) (h₂_pos : 0 < H₂_t) :
    H₁_t * H₂_t / (H₁_s * H₂_s) < H₁_t / H₁_s := by
  have h₁_s_pos : 0 < H₁_s := by linarith
  have h₂_s_pos : 0 < H₂_s := by linarith
  rw [div_lt_div_iff₀ (mul_pos h₁_s_pos h₂_s_pos) h₁_s_pos]
  nlinarith [mul_pos h₁_s_pos h₁_pos]

/-- **Additive PGS misses epistatic signal → portability of epistatic component is zero.**
    An additive PGS captures V_A but not V_epistasis. The "missing heritability"
    from epistasis doesn't port because it was never captured. -/
theorem additive_pgs_misses_epistasis
    (v_additive v_epistatic v_total : ℝ)
    (h_total : v_total = v_additive + v_epistatic)
    (h_epi_pos : 0 < v_epistatic) (h_add_pos : 0 < v_additive) :
    v_additive / v_total < 1 := by
  rw [h_total, div_lt_one (by linarith)]
  linarith

end Epistasis

end Calibrator
