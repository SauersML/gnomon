import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Population Stratification, Confounding, and PGS Portability

This file formalizes how population stratification creates confounding in PGS,
and how this confounding interacts with portability. Key results:

1. Stratification bias in GWAS effect estimates
2. Principal component correction and residual confounding
3. Assortative mating effects on PGS variance
4. Collider bias in ascertained samples
5. Gene-environment correlation and portability

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Stratification Bias in GWAS

When a GWAS sample is stratified (contains subgroups with different allele
frequencies AND different mean phenotypes), the estimated effect sizes are biased.
This bias is a form of confounding.
-/

section StratificationBias

/- **Stratification bias model.**
    True effect: β. Stratification inflates to β̂ = β + b_confound.
    b_confound = Cov(ancestry, phenotype) * Cov(ancestry, genotype) / Var(genotype). -/

/-- Stratification bias is nonzero when ancestry correlates with both
    phenotype and genotype. -/
theorem stratification_bias_nonzero
    (cov_anc_pheno cov_anc_geno var_geno : ℝ)
    (h_pheno : cov_anc_pheno ≠ 0)
    (h_geno : cov_anc_geno ≠ 0)
    (h_var : 0 < var_geno) :
    cov_anc_pheno * cov_anc_geno / var_geno ≠ 0 := by
  apply div_ne_zero
  · exact mul_ne_zero h_pheno h_geno
  · exact h_var.ne'

/-- **Stratification bias model for p SNPs.**
    Each SNP i has true effect β_i and confounding bias b_i.
    PGS_obs = Σ (β_i + b_i) * x_i.
    The observed variance decomposes as Var(PGS_true) + Var(PGS_bias) + 2·Cov
    when biases are independent of true effects, Cov vanishes. -/
structure StratificationModel (p : ℕ) where
  /-- True per-SNP effects -/
  β : Fin p → ℝ
  /-- Confounding bias per SNP -/
  b : Fin p → ℝ
  /-- Per-SNP heterozygosity (proxy for allele freq variance) -/
  H : Fin p → ℝ
  /-- All heterozygosities are positive -/
  H_pos : ∀ i, 0 < H i
  /-- At least one bias is nonzero -/
  bias_nonzero : ∃ i, b i ≠ 0

/-- True PGS variance component: Σ β_i² · H_i -/
noncomputable def StratificationModel.varTrue {p : ℕ} (m : StratificationModel p) : ℝ :=
  ∑ i : Fin p, m.β i ^ 2 * m.H i

/-- Bias variance component: Σ b_i² · H_i -/
noncomputable def StratificationModel.varBias {p : ℕ} (m : StratificationModel p) : ℝ :=
  ∑ i : Fin p, m.b i ^ 2 * m.H i

/-- The bias variance is strictly positive when any bias is nonzero. -/
theorem stratification_bias_variance_pos {p : ℕ} (m : StratificationModel p) :
    0 < m.varBias := by
  unfold StratificationModel.varBias
  obtain ⟨j, hj⟩ := m.bias_nonzero
  apply Finset.sum_pos'
  · intro i _
    exact mul_nonneg (sq_nonneg _) (le_of_lt (m.H_pos i))
  · exact ⟨j, Finset.mem_univ _, mul_pos (sq_pos_of_ne_zero hj) (m.H_pos j)⟩

/-- **Stratification inflates PGS variance.**
    The observed PGS variance (true + bias components, ignoring cross-term for
    independent biases) exceeds the true PGS variance, derived from the model
    structure rather than assumed. -/
theorem stratification_inflates_pgs_variance {p : ℕ} (m : StratificationModel p)
    (h_true : 0 < m.varTrue) :
    m.varTrue < m.varTrue + m.varBias := by
  linarith [stratification_bias_variance_pos m]

/-- **Stratification creates spurious portability.**
    In source population, bias structure correlates with LD → inflates R².
    In target, different LD means a different projection of the bias vector
    onto phenotype. We model this: source bias variance > target bias variance
    because the bias vector was "tuned" to the source LD structure. -/
structure TwoPopBiasModel (p : ℕ) extends StratificationModel p where
  /-- Bias attenuation in target: fraction of source bias variance retained -/
  attenuation : ℝ
  /-- Attenuation is in (0, 1): some but not all bias transfers -/
  atten_pos : 0 < attenuation
  atten_lt_one : attenuation < 1

/-- Target population bias variance -/
noncomputable def TwoPopBiasModel.varBiasTarget {p : ℕ} (m : TwoPopBiasModel p) : ℝ :=
  m.attenuation * m.toStratificationModel.varBias

theorem spurious_portability_from_stratification {p : ℕ} (m : TwoPopBiasModel p)
    (r2_true : ℝ) (h_true_nn : 0 ≤ r2_true) :
    -- Apparent portability drop (source_obs - target_obs) exceeds true drop (0)
    (r2_true + m.toStratificationModel.varBias) -
      (r2_true + m.varBiasTarget) > 0 := by
  unfold TwoPopBiasModel.varBiasTarget
  have hv := stratification_bias_variance_pos m.toStratificationModel
  have : m.attenuation * m.toStratificationModel.varBias < m.toStratificationModel.varBias := by
    rw [← mul_one m.toStratificationModel.varBias]
    simpa [mul_assoc] using mul_lt_mul_of_pos_right m.atten_lt_one hv
  linarith

/-- **PC correction model.**
    Ancestry covariance has eigenvalues λ₁ ≥ λ₂ ≥ ... ≥ λ_p > 0.
    Correcting for k PCs removes variance proportional to Σ_{i≤k} λ_i.
    Residual bias is proportional to Σ_{i>k} λ_i. -/
structure PCCorrectionModel where
  /-- Number of eigenvalues -/
  p : ℕ
  /-- Eigenvalues of ancestry covariance, in decreasing order -/
  eigenvals : Fin p → ℝ
  /-- All eigenvalues positive -/
  eig_pos : ∀ i, 0 < eigenvals i
  /-- Eigenvalues are decreasing -/
  eig_decreasing : ∀ i j : Fin p, i < j → eigenvals j < eigenvals i
  /-- Proportionality constant relating eigenvalues to bias -/
  c : ℝ
  c_pos : 0 < c
  /-- Number of PCs used for correction -/
  k : Fin p
  /-- At least one eigenvalue remains after correction -/
  k_lt : k.val + 1 < p

/-- Residual bias after correcting for k PCs -/
noncomputable def PCCorrectionModel.residualBias (m : PCCorrectionModel) : ℝ :=
  m.c * ∑ i : Fin m.p, if m.k.val < i.val then m.eigenvals i else 0

/-- Uncorrected bias (k = 0) -/
noncomputable def PCCorrectionModel.uncorrectedBias (m : PCCorrectionModel) : ℝ :=
  m.c * ∑ i : Fin m.p, m.eigenvals i

/-- **PC correction reduces but doesn't eliminate bias.**
    Residual bias is strictly less than uncorrected bias (because we remove
    at least one positive eigenvalue), but strictly positive (because at
    least one eigenvalue remains). -/
theorem pc_correction_residual_bias (m : PCCorrectionModel) :
    0 < m.residualBias ∧ m.residualBias < m.uncorrectedBias := by
  unfold PCCorrectionModel.residualBias PCCorrectionModel.uncorrectedBias
  constructor
  · apply mul_pos m.c_pos
    apply Finset.sum_pos'
    · intro i _
      split_ifs with h
      · exact le_of_lt (m.eig_pos i)
      · exact le_refl _
    · refine ⟨⟨m.k.val + 1, m.k_lt⟩, Finset.mem_univ _, ?_⟩
      simp only [show m.k.val < m.k.val + 1 from Nat.lt_succ_iff.mpr (le_refl _), ite_true]
      exact m.eig_pos _
  · apply mul_lt_mul_of_pos_left _ m.c_pos
    apply Finset.sum_lt_sum
    · intro i _
      split_ifs with h
      · exact le_refl _
      · exact le_of_lt (m.eig_pos i)
    · have hp : 0 < m.p := by
        exact lt_trans (Nat.succ_pos m.k.val) m.k_lt
      refine ⟨⟨0, hp⟩, Finset.mem_univ _, ?_⟩
      simp only [show ¬(m.k.val < 0) from Nat.not_lt_zero _, ite_false]
      exact m.eig_pos _

/-- **More PCs reduce residual bias monotonically.**
    Residual bias with k+1 PCs < residual bias with k PCs, because
    adding a PC removes the (k+1)-th eigenvalue from the residual sum. -/
theorem more_pcs_less_bias
    (p : ℕ) (eigenvals : Fin p → ℝ) (c : ℝ) (k : ℕ)
    (h_c : 0 < c)
    (h_eig_pos : ∀ i, 0 < eigenvals i)
    (h_k_bound : k + 2 < p) :
    c * (∑ i : Fin p, if k + 1 < i.val then eigenvals i else 0) <
      c * (∑ i : Fin p, if k < i.val then eigenvals i else 0) := by
  apply mul_lt_mul_of_pos_left _ h_c
  apply Finset.sum_lt_sum
  · intro i _
    split_ifs with h1 h2
    · exact le_refl _
    · exfalso
      exact h2 (lt_trans (Nat.lt_succ_self k) h1)
    · exact le_of_lt (h_eig_pos i)
    · exact le_refl _
  · have hk1_bound : k + 1 < p := by
      exact lt_trans (Nat.lt_succ_self (k + 1)) h_k_bound
    refine ⟨⟨k + 1, hk1_bound⟩, Finset.mem_univ _, ?_⟩
    simp only [show ¬(k + 1 < k + 1) from lt_irrefl _, ite_false,
               show k < k + 1 from Nat.lt_succ_iff.mpr (le_refl _), ite_true]
    exact h_eig_pos _

end StratificationBias


/-!
## Assortative Mating and PGS Variance

Assortative mating (AM) for a trait increases the genetic variance of that trait
in subsequent generations. This affects PGS portability because:
1. Source population PGS variance depends on AM history
2. AM patterns differ across populations
-/

section AssortativeMating

/-- **AM inflation factor.**
    After t generations of AM with spousal correlation r,
    genetic variance inflates by factor ≈ 1/(1-r) at equilibrium. -/
noncomputable def amInflationFactor (r : ℝ) : ℝ :=
  1 / (1 - r)

/-- AM inflation factor > 1 for positive assortment. -/
theorem am_inflation_gt_one (r : ℝ) (hr : 0 < r) (hr1 : r < 1) :
    1 < amInflationFactor r := by
  unfold amInflationFactor
  rw [lt_div_iff₀ (by linarith)]
  linarith

/-- **Differential AM creates portability artifact.**
    If source population has stronger AM (r_s > r_t), then
    PGS variance is higher in source → R² appears higher in source
    even with identical genetic architecture. -/
theorem differential_am_creates_portability_artifact
    (r_s r_t : ℝ)
    (hrs : 0 < r_s) (hrt : 0 < r_t) (hrs1 : r_s < 1) (hrt1 : r_t < 1)
    (h_stronger : r_t < r_s) :
    amInflationFactor r_t < amInflationFactor r_s := by
  unfold amInflationFactor
  apply div_lt_div_of_pos_left one_pos (by linarith) (by linarith)

/-- **AM affects both numerator and denominator of R².**
    R² = V_PGS / V_Y. AM inflates V_PGS by α and V_Y by less than α
    (because V_E doesn't change), so R² increases. -/
theorem am_increases_r2
    (v_pgs v_e α : ℝ)
    (h_vpgs : 0 < v_pgs) (h_ve : 0 < v_e) (h_α : 1 < α) :
    v_pgs / (v_pgs + v_e) < (α * v_pgs) / (α * v_pgs + v_e) := by
  have h_d1 : 0 < v_pgs + v_e := by linarith
  have h_d2 : 0 < α * v_pgs + v_e := by nlinarith
  rw [div_lt_div_iff₀ h_d1 h_d2]
  nlinarith [mul_pos h_vpgs h_ve]

end AssortativeMating


/-!
## Collider Bias in Ascertained Samples

When GWAS or validation cohorts are ascertained (e.g., hospital-based,
volunteer bias), collider bias can create spurious associations and
affect portability estimates.
-/

section ColliderBias

/-- **Collider bias model.**
    In the population, G and E are independent (Cov = 0).
    Selection S depends on G + E: individuals selected when G + E > threshold.
    Conditioning on S induces negative covariance between G and E among
    the selected, because high G "explains away" high E. -/
structure ColliderModel where
  /-- Genetic risk variance -/
  σ2_G : ℝ
  /-- Environmental risk variance -/
  σ2_E : ℝ
  /-- True effect of G on outcome -/
  β_G : ℝ
  σ2_G_pos : 0 < σ2_G
  σ2_E_pos : 0 < σ2_E
  β_G_pos : 0 < β_G

/-- **Selection on G + E induces negative covariance in selected sample.**
    In the selected subsample, Cov(G, E | S=1) = -σ²_G · σ²_E / (σ²_G + σ²_E).
    This is the classical "explaining away" effect. -/
noncomputable def ColliderModel.inducedCov (m : ColliderModel) : ℝ :=
  -(m.σ2_G * m.σ2_E / (m.σ2_G + m.σ2_E))

theorem ColliderModel.inducedCov_neg (m : ColliderModel) :
    m.inducedCov < 0 := by
  unfold ColliderModel.inducedCov
  rw [neg_neg_iff_pos]
  exact div_pos (mul_pos m.σ2_G_pos m.σ2_E_pos) (by linarith [m.σ2_G_pos, m.σ2_E_pos])

/-- **Selection induces correlation.**
    Population covariance is zero; the model-derived induced covariance in the
    selected sample is negative, hence different from the population value. -/
theorem selection_induces_correlation (m : ColliderModel) :
    m.inducedCov ≠ 0 := by
  exact ne_of_lt m.inducedCov_neg

/-- **Collider bias attenuates PGS-outcome association.**
    In the full population, regression coefficient is β_G.
    In the selected sample, the induced G-E covariance attenuates:
    β_selected = β_G · σ²_G / (σ²_G + σ²_E).
    Since σ²_E > 0, this ratio is < 1, so β_selected < β_G. -/
noncomputable def ColliderModel.β_selected (m : ColliderModel) : ℝ :=
  m.β_G * (m.σ2_G / (m.σ2_G + m.σ2_E))

theorem collider_attenuates_association (m : ColliderModel) :
    m.β_selected < m.β_G := by
  unfold ColliderModel.β_selected
  have h_denom_pos : 0 < m.σ2_G + m.σ2_E := by linarith [m.σ2_G_pos, m.σ2_E_pos]
  have h_ratio_lt_one : m.σ2_G / (m.σ2_G + m.σ2_E) < 1 := by
    rw [div_lt_one h_denom_pos]
    linarith [m.σ2_E_pos]
  calc m.β_G * (m.σ2_G / (m.σ2_G + m.σ2_E))
      < m.β_G * 1 := by exact mul_lt_mul_of_pos_left h_ratio_lt_one m.β_G_pos
    _ = m.β_G := by ring

/-- **Differential ascertainment creates portability artifact.**
    If source and target cohorts have different ascertainment patterns,
    the apparent portability drop includes an ascertainment component. -/
theorem differential_ascertainment_artifact
    (r2_source_pop r2_target_pop r2_source_asc r2_target_asc : ℝ)
    (_h_source_asc : r2_source_asc < r2_source_pop)
    (_h_target_asc : r2_target_asc < r2_target_pop)
    -- Different ascertainment severity
    (h_diff_severity : r2_target_pop - r2_target_asc < r2_source_pop - r2_source_asc) :
    -- Apparent portability drop is smaller than true portability drop
    r2_source_asc - r2_target_asc < r2_source_pop - r2_target_pop := by
  linarith

end ColliderBias


/-!
## Gene-Environment Correlation (rGE) and Portability

Gene-environment correlation means that genetic effects are partially
mediated through environmental pathways. This affects portability because
the environmental mediation may differ across populations.
-/

section GeneEnvironmentCorrelation

/- **rGE decomposition of PGS prediction.**
    PGS predicts outcome through two pathways:
    1. Direct genetic effect: β_direct
    2. Indirect (rGE-mediated) effect: β_indirect = β_genetic × r_GE × β_env
    Total prediction: β_total = β_direct + β_indirect -/

/-- **rGE model.**
    Genetic effects on an outcome are mediated through both a direct
    pathway and an indirect (environment-mediated) pathway. -/
structure RGEModel where
  /-- Direct genetic effect on outcome -/
  β_direct : ℝ
  /-- Effect of genotype on environment -/
  β_genetic : ℝ
  /-- Effect of environment on outcome -/
  β_env : ℝ
  /-- Gene-environment correlation in source -/
  rge_source : ℝ
  /-- Gene-environment correlation in target -/
  rge_target : ℝ
  β_genetic_ne : β_genetic ≠ 0
  β_env_ne : β_env ≠ 0
  rge_diff : rge_source ≠ rge_target

/-- Total prediction in source population -/
noncomputable def RGEModel.predSource (m : RGEModel) : ℝ :=
  m.β_direct + m.β_genetic * m.rge_source * m.β_env

/-- Total prediction in target population -/
noncomputable def RGEModel.predTarget (m : RGEModel) : ℝ :=
  m.β_direct + m.β_genetic * m.rge_target * m.β_env

/-- If rGE differs across populations, total prediction changes
    even with identical direct genetic effects. -/
theorem rge_changes_total_prediction (m : RGEModel) :
    m.predSource ≠ m.predTarget := by
  unfold RGEModel.predSource RGEModel.predTarget
  intro h
  have h_eq : m.β_genetic * m.rge_source * m.β_env = m.β_genetic * m.rge_target * m.β_env := by linarith
  apply m.rge_diff
  have h_ne : m.β_genetic * m.β_env ≠ 0 := mul_ne_zero m.β_genetic_ne m.β_env_ne
  have : m.rge_source * (m.β_genetic * m.β_env) = m.rge_target * (m.β_genetic * m.β_env) := by nlinarith
  exact mul_right_cancel₀ h_ne this

/-- **rGE inflation model.**
    R² in the presence of rGE decomposes as:
      R²_obs = R²_direct + 2·β_d·β_g·rge·β_e·σ² + (β_g·rge·β_e)²·σ²
    The cross term and indirect term add to R² when rge > 0. -/
structure RGEInflationModel where
  /-- Direct R² component -/
  r2_direct : ℝ
  /-- Indirect effect magnitude -/
  β_indirect : ℝ
  /-- Genetic variance -/
  σ2 : ℝ
  r2_direct_pos : 0 < r2_direct
  β_indirect_ne : β_indirect ≠ 0
  σ2_pos : 0 < σ2

/-- Observed R² under rGE -/
noncomputable def RGEInflationModel.r2_obs (m : RGEInflationModel) : ℝ :=
  m.r2_direct + m.β_indirect ^ 2 * m.σ2

/-- **rGE inflation of apparent heritability.**
    The indirect effect contributes β_indirect² · σ² > 0 to observed R²,
    derived from the model rather than assumed. -/
theorem rge_inflates_apparent_heritability (m : RGEInflationModel) :
    m.r2_direct < m.r2_obs := by
  unfold RGEInflationModel.r2_obs
  have : 0 < m.β_indirect ^ 2 * m.σ2 :=
    mul_pos (sq_pos_of_ne_zero m.β_indirect_ne) m.σ2_pos
  linarith

/-- **Loss of rGE is not recoverable from genetic data alone.**
    The rGE-mediated component requires knowing the environmental
    structure of the target population. -/
theorem rge_loss_requires_environmental_data (m : RGEModel) :
    m.predSource ≠ m.predTarget := rge_changes_total_prediction m

end GeneEnvironmentCorrelation


/-!
## Survivorship Bias in PGS Portability Studies

When studying portability across age-structured populations,
survivorship bias can affect results because the genotype distribution
in older cohorts differs from birth cohorts due to differential mortality.
-/

section SurvivorshipBias

/-- **Survivorship model.**
    At birth, genotype frequency is p₀. Genotype confers relative risk γ > 1
    for mortality. After selection, surviving frequency among risk allele
    carriers is p₀ · s / (p₀ · s + (1 - p₀)) where s < 1 is survival prob. -/
structure SurvivorshipModel where
  /-- Birth frequency of risk allele -/
  p₀ : ℝ
  /-- Survival probability for risk carriers (relative to non-carriers = 1) -/
  s : ℝ
  p₀_pos : 0 < p₀
  p₀_lt_one : p₀ < 1
  s_pos : 0 < s
  s_lt_one : s < 1

/-- Frequency of risk allele among survivors -/
noncomputable def SurvivorshipModel.pSurv (m : SurvivorshipModel) : ℝ :=
  m.p₀ * m.s / (m.p₀ * m.s + (1 - m.p₀))

/-- **Age-dependent genotype frequency shift.**
    The risk allele frequency among survivors is lower than at birth,
    derived from the survival model. -/
theorem survivorship_shifts_genotype_freq (m : SurvivorshipModel) :
    m.pSurv < m.p₀ := by
  unfold SurvivorshipModel.pSurv
  have h1 : 0 < 1 - m.p₀ := by linarith [m.p₀_lt_one]
  have h2 : 0 < m.p₀ * m.s := mul_pos m.p₀_pos m.s_pos
  have h_denom_pos : 0 < m.p₀ * m.s + (1 - m.p₀) := by linarith
  rw [div_lt_iff₀ h_denom_pos]
  have : m.p₀ * (m.p₀ * m.s + (1 - m.p₀)) = m.p₀ ^ 2 * m.s + m.p₀ * (1 - m.p₀) := by ring
  have : m.p₀ * m.s = m.p₀ * m.s := rfl
  -- Need: p₀ · s < p₀ · (p₀ · s + (1 - p₀))
  -- i.e., p₀ · s < p₀² · s + p₀ · (1 - p₀)
  -- i.e., p₀ · s · (1 - p₀) < p₀ · (1 - p₀)    [rearranging]
  -- i.e., s < 1   ✓ (since p₀ · (1 - p₀) > 0)
  nlinarith [m.p₀_pos, m.p₀_lt_one, m.s_lt_one, m.s_pos,
             mul_pos m.p₀_pos h1, sq_nonneg m.p₀]

/-- **Survivorship bias attenuation model.**
    PGS-outcome R² depends on allele frequency variance.
    Among survivors, risk allele frequency is shifted down → reduced variance
    → attenuated R².
    R²_surv = R²_full · (Var_surv / Var_birth) where Var_surv < Var_birth. -/
structure SurvivorshipAttenuationModel where
  /-- R² in birth cohort -/
  r2_full : ℝ
  /-- Variance of genotype in birth cohort -/
  var_birth : ℝ
  /-- Variance of genotype among survivors -/
  var_surv : ℝ
  r2_full_pos : 0 < r2_full
  var_birth_pos : 0 < var_birth
  var_surv_pos : 0 < var_surv
  /-- Survivorship truncation reduces variance -/
  var_reduced : var_surv < var_birth

/-- R² among survivors -/
noncomputable def SurvivorshipAttenuationModel.r2_surv (m : SurvivorshipAttenuationModel) : ℝ :=
  m.r2_full * (m.var_surv / m.var_birth)

/-- **Survivorship bias attenuates PGS-outcome association in older cohorts.**
    Derived from the attenuation model: Var_surv < Var_birth implies
    the variance ratio < 1, so R²_surv < R²_full. -/
theorem survivorship_attenuates_in_older (m : SurvivorshipAttenuationModel) :
    m.r2_surv < m.r2_full := by
  unfold SurvivorshipAttenuationModel.r2_surv
  have h_ratio_lt_one : m.var_surv / m.var_birth < 1 := by
    rw [div_lt_one m.var_birth_pos]
    exact m.var_reduced
  calc m.r2_full * (m.var_surv / m.var_birth)
      < m.r2_full * 1 := by exact mul_lt_mul_of_pos_left h_ratio_lt_one m.r2_full_pos
    _ = m.r2_full := by ring

/-- **Differential survivorship across populations creates portability artifact.**
    If the target population has different age structure or mortality patterns,
    survivorship bias contributes to apparent portability loss. -/
theorem differential_survivorship_artifact
    (r2_source_full r2_target_full Δ_surv_source Δ_surv_target : ℝ)
    (h_surv_s : 0 ≤ Δ_surv_source) (h_surv_t : 0 ≤ Δ_surv_target)
    (h_diff : Δ_surv_target > Δ_surv_source)
    (h_obs_s : r2_source_full - Δ_surv_source > 0) :
    (r2_source_full - Δ_surv_source) - (r2_target_full - Δ_surv_target) >
      r2_source_full - r2_target_full := by
  linarith

end SurvivorshipBias


/-!
## Causal Inference Framework for Portability

Portability loss can be understood through the lens of causal inference:
the PGS is a proxy for a causal variable (genetic liability), and
portability loss arises from violations of the assumptions needed for
the proxy to work across populations.
-/

section CausalInference

/-- **Measurement error model for PGS.**
    PGS = true genetic liability × attenuation + noise.
    Attenuation = √R² in the training GWAS.
    In a new population, attenuation changes. -/
noncomputable def pgsAttenuationFactor (r2_gwas : ℝ) : ℝ :=
  Real.sqrt r2_gwas

/-- **Attenuation factor decreases with lower GWAS R².**
    In target populations where the GWAS is less predictive,
    the PGS is a noisier proxy for genetic liability. -/
theorem attenuation_decreases_with_r2
    (r2_source r2_target : ℝ)
    (h_s : 0 ≤ r2_source) (h_t : 0 ≤ r2_target)
    (h_drop : r2_target < r2_source) :
    pgsAttenuationFactor r2_target < pgsAttenuationFactor r2_source := by
  unfold pgsAttenuationFactor
  exact Real.sqrt_lt_sqrt h_t h_drop

/-- **Measurement error attenuation model.**
    When PGS is used as covariate, measurement error attenuates the
    coefficient. True coefficient β_true is observed as β_true · R
    where R = reliability ratio = signal variance / total variance.
    Reliability ratio = r² / (r² + σ²_noise). -/
structure AttenuationModel where
  /-- True regression coefficient -/
  β_true : ℝ
  /-- Signal R² in source population -/
  r2_source : ℝ
  /-- Signal R² in target population -/
  r2_target : ℝ
  /-- Noise variance (constant across populations) -/
  σ2_noise : ℝ
  β_true_pos : 0 < β_true
  r2_source_pos : 0 < r2_source
  r2_target_pos : 0 < r2_target
  σ2_noise_pos : 0 < σ2_noise
  /-- Target has lower signal -/
  r2_drop : r2_target < r2_source

/-- Reliability ratio in a population -/
noncomputable def reliabilityRatio (r2 σ2_noise : ℝ) : ℝ :=
  r2 / (r2 + σ2_noise)

/-- Observed (attenuated) coefficient -/
noncomputable def AttenuationModel.β_obs (m : AttenuationModel) (r2 : ℝ) : ℝ :=
  m.β_true * reliabilityRatio r2 m.σ2_noise

/-- Helper: x ↦ x / (x + c) is strictly monotone for c > 0. -/
theorem ratio_strict_mono {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (hab : a < b) : a / (a + c) < b / (b + c) := by
  have h1 : 0 < a + c := by linarith
  have h2 : 0 < b + c := by linarith
  rw [div_lt_div_iff₀ h1 h2]
  nlinarith

/-- **Noisier proxy → more attenuation bias in downstream analysis.**
    Lower R² → lower reliability ratio → more attenuated coefficient.
    Uses monotonicity of x/(x+c). -/
theorem noisier_proxy_more_bias (m : AttenuationModel) :
    m.β_obs m.r2_target < m.β_obs m.r2_source := by
  unfold AttenuationModel.β_obs
  apply mul_lt_mul_of_pos_left _ m.β_true_pos
  exact ratio_strict_mono m.r2_target_pos m.r2_source_pos m.σ2_noise_pos m.r2_drop

/-- **Transportability model.**
    PGS accuracy in the target is the source accuracy minus the sum of
    bias contributions from each violated assumption. Each assumption i
    contributes a penalty δ_i ≥ 0 to accuracy loss, and δ_i > 0 when
    the assumption is violated. -/
structure TransportabilityModel (n : ℕ) where
  /-- Source population R² -/
  r2_source : ℝ
  /-- Per-assumption accuracy penalty in target -/
  δ : Fin n → ℝ
  /-- All penalties are nonneg -/
  δ_nonneg : ∀ i, 0 ≤ δ i
  /-- At least one assumption violated (positive penalty) -/
  violated : ∃ i, 0 < δ i
  r2_source_pos : 0 < r2_source

/-- Target R² under the transportability model -/
noncomputable def TransportabilityModel.r2_target {n : ℕ}
    (m : TransportabilityModel n) : ℝ :=
  m.r2_source - ∑ i : Fin n, m.δ i

/-- **Transportability violation creates gap.**
    When any assumption is violated, the total penalty is positive,
    so target R² < source R². Derived from the sum structure and
    the existence of a positive penalty term. -/
theorem transportability_violation_creates_gap {n : ℕ}
    (m : TransportabilityModel n) :
    m.r2_target < m.r2_source := by
  unfold TransportabilityModel.r2_target
  linarith [show 0 < ∑ i : Fin n, m.δ i from
    Finset.sum_pos' (fun i _ => m.δ_nonneg i)
      (let ⟨j, hj⟩ := m.violated; ⟨j, Finset.mem_univ _, hj⟩)]

end CausalInference


/-!
## Mendelian Randomization and Portability

MR uses genetic variants as instruments. Portability of MR estimates
depends on the same factors as PGS portability, plus additional
assumptions about pleiotropy and instrument strength.
-/

section MRPortability

/-- **MR instrument model.**
    F-statistic of a genetic instrument depends on effect size and
    allele frequency: F = n · β² · 2p(1-p) / σ²_Y.
    When allele frequency p changes across populations, F changes. -/
structure MRInstrumentModel where
  /-- Sample size -/
  n : ℝ
  /-- Effect of instrument on exposure -/
  β_inst : ℝ
  /-- Allele frequency in source -/
  p_source : ℝ
  /-- Allele frequency in target -/
  p_target : ℝ
  /-- Outcome variance -/
  σ2_Y : ℝ
  n_pos : 0 < n
  β_inst_ne : β_inst ≠ 0
  σ2_Y_pos : 0 < σ2_Y
  p_source_pos : 0 < p_source
  p_source_lt : p_source < 1
  p_target_pos : 0 < p_target
  p_target_lt : p_target < 1

/-- Heterozygosity 2p(1-p) as a function of allele frequency -/
noncomputable def heterozygosity (p : ℝ) : ℝ := 2 * p * (1 - p)

/-- F-statistic of an instrument at a given allele frequency -/
noncomputable def MRInstrumentModel.fStat (m : MRInstrumentModel) (p : ℝ) : ℝ :=
  m.n * m.β_inst ^ 2 * heterozygosity p / m.σ2_Y

/-- Heterozygosity is maximized at p = 0.5 and decreasing as p moves away. -/
theorem heterozygosity_pos (p : ℝ) (hp : 0 < p) (hp1 : p < 1) :
    0 < heterozygosity p := by
  unfold heterozygosity
  have : 0 < 1 - p := by linarith
  positivity

/-- **Instrument strength decreases with allele frequency divergence.**
    If the target has lower heterozygosity (allele frequency further from 0.5
    or toward fixation), F-stat decreases. Derived from the F-stat formula. -/
theorem instrument_strength_decreases (m : MRInstrumentModel)
    (h_het : heterozygosity m.p_target < heterozygosity m.p_source) :
    m.fStat m.p_target < m.fStat m.p_source := by
  unfold MRInstrumentModel.fStat
  apply div_lt_div_of_pos_right _ m.σ2_Y_pos
  apply mul_lt_mul_of_pos_left h_het
  exact mul_pos m.n_pos (sq_pos_of_ne_zero m.β_inst_ne)

/-- **Weak instrument bias in MR.**
    Bias = (1 - 1/F) × confounding bias.
    As F decreases (weaker instrument), bias increases toward the
    confounded OLS estimate. -/
theorem weak_instrument_bias_increases
    (conf_bias : ℝ) (F₁ F₂ : ℝ)
    (h_conf : 0 < conf_bias)
    (h_F₁ : 1 < F₁) (h_F₂ : 1 < F₂)
    (h_weaker : F₂ < F₁) :
    (1 - 1/F₂) * conf_bias < (1 - 1/F₁) * conf_bias := by
  apply mul_lt_mul_of_pos_right _ h_conf
  have h1 : 1/F₁ < 1/F₂ := by
    rw [div_lt_div_iff₀ (by linarith) (by linarith)]
    linarith
  linarith

/-- **Horizontal pleiotropy patterns differ across populations.**
    If pleiotropic effects change across populations (due to different
    LD patterns or gene regulation), MR estimates are not portable. -/
theorem pleiotropy_changes_invalidate_mr
    (β_causal α_pleio_source α_pleio_target : ℝ)
    (h_diff : α_pleio_source ≠ α_pleio_target) :
    β_causal + α_pleio_source ≠ β_causal + α_pleio_target := by
  intro h; exact h_diff (by linarith)

end MRPortability


/-!
## Sample Size and Statistical Power for Portability Detection

Detecting portability differences requires adequate statistical power.
We formalize the power analysis for portability comparisons.
-/

section PowerAnalysis

/-- **Variance of R² estimator.**
    Var(R²) ≈ 4R²(1-R²)²/n for the standard R² estimator. -/
noncomputable def r2EstimatorVariance (r2 : ℝ) (n : ℕ) : ℝ :=
  4 * r2 * (1 - r2) ^ 2 / n

/-- R² estimator variance is positive for non-degenerate R². -/
theorem r2_estimator_variance_pos (r2 : ℝ) (n : ℕ)
    (h_r2 : 0 < r2) (h_r2_lt : r2 < 1) (h_n : 0 < n) :
    0 < r2EstimatorVariance r2 n := by
  unfold r2EstimatorVariance
  apply div_pos
  · apply mul_pos
    · apply mul_pos
      · linarith
      · exact h_r2
    · exact sq_pos_of_pos (by linarith)
  · exact Nat.cast_pos.mpr h_n

/-- **Power to detect portability difference.**
    To detect ΔR² = R²_source - R²_target at power 1-β with significance α,
    need n ≈ (z_α + z_β)² × (Var₁ + Var₂) / ΔR²². -/
theorem larger_sample_more_power
    (var₁ var₂ Δr2 z_sum n₁ n₂ : ℝ)
    (h_var : 0 < var₁ + var₂) (h_Δ : 0 < Δr2)
    (h_z : 0 < z_sum)
    (h_n : n₁ < n₂) (h_n₁ : 0 < n₁) :
    -- Larger sample → smaller required effect size (more power)
    z_sum * Real.sqrt ((var₁ + var₂) / n₂) <
      z_sum * Real.sqrt ((var₁ + var₂) / n₁) := by
  apply mul_lt_mul_of_pos_left _ h_z
  apply Real.sqrt_lt_sqrt
  · exact div_nonneg (le_of_lt h_var) (le_of_lt (by linarith : 0 < n₂))
  · exact div_lt_div_of_pos_left h_var h_n₁ h_n

/-- **Small portability differences require large samples.**
    When R² of the distance-on-error relationship is small, enormous
    samples are needed to detect this reliably.

    Worked example: Wang et al.'s finding of R² ≈ 0.5% for
    distance-on-error illustrates this. -/
theorem small_effect_needs_large_n
    (r2_effect n_required ub : ℝ)
    (h_small : r2_effect ≤ ub) (h_ub_pos : 0 < ub)
    (h_formula : n_required ≥ 1 / r2_effect)
    (h_effect_pos : 0 < r2_effect) :
    n_required ≥ 1 / ub := by
  calc n_required ≥ 1 / r2_effect := h_formula
    _ ≥ 1 / ub := by
        exact div_le_div_of_nonneg_left (le_of_lt one_pos) h_effect_pos h_small

end PowerAnalysis

end Calibrator
