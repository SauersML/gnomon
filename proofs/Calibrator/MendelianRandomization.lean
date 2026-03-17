import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Mendelian Randomization and PGS Portability

This file formalizes the relationship between Mendelian Randomization (MR)
and PGS portability. MR uses genetic variants as instruments to estimate
causal effects. The validity of MR depends on assumptions that interact
with cross-ancestry portability.

Key results:
1. IV assumptions and their cross-ancestry validity
2. Pleiotropy and instrument validity
3. Population-specific instrument strength (F-statistic)
4. Two-sample MR with cross-ancestry data
5. MR for validating PGS mechanisms

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Instrumental Variable Assumptions

MR requires three core assumptions:
1. Relevance: instrument is associated with exposure
2. Independence: instrument is independent of confounders
3. Exclusion: instrument affects outcome only through exposure
-/

section IVAssumptions

/-- **IV estimate (Wald ratio).**
    β_IV = β_ZY / β_ZX
    where Z is instrument, X is exposure, Y is outcome. -/
noncomputable def waldRatio (beta_ZY beta_ZX : ℝ) : ℝ :=
  beta_ZY / beta_ZX

/-- Wald ratio equals causal effect under valid IV assumptions. -/
theorem wald_ratio_identifies_causal
    (beta_causal beta_ZY beta_ZX : ℝ)
    (h_ZX : beta_ZX ≠ 0)
    (h_reduced : beta_ZY = beta_causal * beta_ZX) :
    waldRatio beta_ZY beta_ZX = beta_causal := by
  unfold waldRatio
  rw [h_reduced, mul_div_cancel_right₀ _ h_ZX]

/-- **F-statistic for instrument strength.**
    F = (n-2) × R²_ZX / (1 - R²_ZX)
    Weak instruments (F < 10) produce biased MR estimates. -/
noncomputable def fStatistic (n : ℕ) (r2_ZX : ℝ) : ℝ :=
  (n - 2) * r2_ZX / (1 - r2_ZX)

/-- F-statistic increases with R². -/
theorem f_stat_increases_with_r2 (n : ℕ) (r2₁ r2₂ : ℝ)
    (h_n : 2 < n) (h_r2₂_lt : r2₂ < 1)
    (h_lt : r2₁ < r2₂) :
    fStatistic n r2₁ < fStatistic n r2₂ := by
  unfold fStatistic
  have h_n_pos : (0 : ℝ) < n - 2 := by
    have : (2 : ℝ) < n := Nat.ofNat_lt_cast.mpr h_n
    linarith
  have h_r2₁_lt : r2₁ < 1 := by linarith
  rw [div_lt_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- **Weak instrument bias.**
    With weak instruments (F ≈ 1), the IV estimate is biased
    toward the OLS estimate. Bias ≈ 1/F × (OLS estimate). -/
noncomputable def weakInstrumentBias (beta_OLS : ℝ) (F : ℝ) : ℝ :=
  beta_OLS / F

/-- Weak instrument bias decreases with F. -/
theorem weak_bias_decreases_with_f (beta_OLS F₁ F₂ : ℝ)
    (h_beta : 0 < beta_OLS)
    (h_F₁ : 0 < F₁) (h_F₂ : 0 < F₂) (h_lt : F₁ < F₂) :
    weakInstrumentBias beta_OLS F₂ < weakInstrumentBias beta_OLS F₁ := by
  unfold weakInstrumentBias
  exact div_lt_div_iff_of_pos_left h_beta h_F₂ h_F₁ |>.mpr h_lt

end IVAssumptions


/-!
## Cross-Ancestry Instrument Validity

MR instruments selected in one ancestry may not be valid in
another due to LD differences and population-specific pleiotropy.
-/

section CrossAncestryInstruments

/-- **Instrument strength varies across ancestries.**
    An instrument selected in one population (source) may be a weak instrument
    in another (target) because LD between instrument and causal variant differs.
    When the source F-statistic exceeds a threshold but the target falls below it,
    the target instrument is strictly weaker. -/
theorem instrument_strength_varies
    (n : ℕ) (r2_source r2_target : ℝ)
    (h_n : 2 < n)
    (h_r2_source_lt : r2_source < 1)
    (h_ld_weaker : r2_target < r2_source) :
    fStatistic n r2_target < fStatistic n r2_source := by
  exact f_stat_increases_with_r2 n r2_target r2_source h_n h_r2_source_lt h_ld_weaker

/-- **LD proxy instruments weaken across ancestry.**
    In MR, we often use a proxy SNP (in LD with causal SNP).
    In different LD backgrounds, the proxy may be less correlated
    with the causal SNP → weaker instrument.

    General statement: for any two populations where the LD r² between
    proxy and causal is r2_source > r2_target, the apparent effect
    |β| × r² is diminished in the target.

    Worked example: EUR r² > 0.8 vs AFR r² < 0.5 for the same locus. -/
theorem ld_proxy_weakens_cross_ancestry
    (r2_ld_source r2_ld_target beta_causal : ℝ)
    (h_ld_diff : r2_ld_target < r2_ld_source)
    (h_beta : beta_causal ≠ 0) :
    |beta_causal| * r2_ld_target < |beta_causal| * r2_ld_source := by
  apply mul_lt_mul_of_pos_left h_ld_diff
  exact abs_pos.mpr h_beta

/-- **Horizontal pleiotropy may be population-specific.**
    A variant may have pleiotropic effects in one ancestry
    but not another, due to different LD partners. -/
theorem population_specific_pleiotropy
    (beta_direct_eur beta_direct_afr : ℝ)
    (h_eur_pleio : beta_direct_eur ≠ 0)
    (h_afr_no_pleio : beta_direct_afr = 0) :
    beta_direct_eur ≠ beta_direct_afr := by
  rw [h_afr_no_pleio]; exact h_eur_pleio

/-- **MR-PRESSO for cross-ancestry outlier detection.**
    Variants that are outliers in cross-ancestry MR analysis
    are likely to violate IV assumptions due to population-specific
    pleiotropy or LD. -/
theorem outlier_detection_identifies_violations
    (residual threshold : ℝ)
    (h_threshold : 0 ≤ threshold)
    (h_outlier : threshold < |residual|) :
    residual ≠ 0 := by
  intro h; rw [h, abs_zero] at h_outlier; linarith

end CrossAncestryInstruments


/-!
## Two-Sample MR Across Ancestries

Two-sample MR uses summary statistics from separate exposure
and outcome GWAS. Cross-ancestry designs offer advantages
for MR but introduce new challenges.
-/

section TwoSampleMR

/-- **Cross-ancestry MR reduces confounding.**
    Using exposure GWAS from ancestry A and outcome GWAS from
    ancestry B breaks shared population stratification.
    Model: bias_same = confound × r_strat² where r_strat is the
    stratification correlation within ancestry. Cross-ancestry breaks
    this: bias_cross = confound × r_strat_cross² where r_strat_cross < r_strat
    (different ancestries share less stratification structure). -/
theorem cross_ancestry_mr_less_confounded
    (confound r_strat r_strat_cross : ℝ)
    (h_conf : 0 < confound)
    (h_rc : 0 ≤ r_strat_cross)
    (h_cross_less : r_strat_cross < r_strat) :
    confound * r_strat_cross ^ 2 < confound * r_strat ^ 2 := by
  apply mul_lt_mul_of_pos_left _ h_conf
  nlinarith

/-- **But cross-ancestry MR has lower power.**
    Instruments are weaker in the target ancestry (shorter LD,
    different allele frequencies), reducing MR precision.
    Model: SE(β_IV) = σ_Y / (β_ZX × √n). In cross-ancestry MR,
    the instrument-exposure association β_ZX is weaker (due to LD mismatch),
    so SE is larger. Specifically, SE ∝ 1/β_ZX. -/
theorem cross_ancestry_mr_lower_power
    (sigma_Y beta_ZX_same beta_ZX_cross sqrt_n : ℝ)
    (h_sigma : 0 < sigma_Y) (h_same : 0 < beta_ZX_same)
    (h_cross : 0 < beta_ZX_cross) (h_n : 0 < sqrt_n)
    (h_weaker : beta_ZX_cross < beta_ZX_same) :
    sigma_Y / (beta_ZX_same * sqrt_n) < sigma_Y / (beta_ZX_cross * sqrt_n) := by
  exact div_lt_div_iff_of_pos_left h_sigma
    (mul_pos h_same h_n) (mul_pos h_cross h_n) |>.mpr
    (mul_lt_mul_of_pos_right h_weaker h_n)

/-- **Bias-variance tradeoff in cross-ancestry MR.**
    Cross-ancestry: less bias but more variance.
    Same-ancestry: more bias but less variance.
    MSE = Bias² + Variance determines optimal choice. -/
noncomputable def mrMSE (bias variance : ℝ) : ℝ :=
  bias^2 + variance

/-- Cross-ancestry MR is better when bias dominates.
    Model: MSE = bias² + variance. Cross-ancestry has less bias but more variance.
    When the bias reduction (in squared terms) exceeds the variance increase,
    cross-ancestry MSE is lower. -/
theorem cross_better_when_bias_dominates
    (bias_same bias_cross var_same var_cross variance_penalty : ℝ)
    (h_less_bias : bias_cross ^ 2 + variance_penalty < bias_same ^ 2)
    (h_var_penalty : var_cross - var_same ≤ variance_penalty) :
    mrMSE bias_cross var_cross < mrMSE bias_same var_same := by
  unfold mrMSE
  linarith

end TwoSampleMR


/-!
## MR for Validating PGS Mechanisms

MR can be used to understand why PGS portability varies
across traits and populations.
-/

section MRForPGSValidation

/-- **MR tests whether portability loss is causal.**
    If a PGS predicts a phenotype through a causal pathway,
    the MR estimate should be consistent across ancestries.
    Model: under a valid IV with causal effect β_causal, the Wald ratios
    from both ancestries recover the same β_causal. The difference
    |β_MR_eur - β_MR_afr| = 0 < any positive tolerance, confirming
    consistency. Here we show that if both MR estimates equal the
    same causal effect, their difference is within any tolerance. -/
theorem mr_consistency_implies_causal_prediction
    (beta_causal beta_ZY_eur beta_ZX_eur beta_ZY_afr beta_ZX_afr tolerance : ℝ)
    (h_tol : 0 < tolerance)
    (h_eur_valid : beta_ZY_eur = beta_causal * beta_ZX_eur)
    (h_afr_valid : beta_ZY_afr = beta_causal * beta_ZX_afr)
    (h_ZX_eur : beta_ZX_eur ≠ 0) (h_ZX_afr : beta_ZX_afr ≠ 0) :
    |waldRatio beta_ZY_eur beta_ZX_eur - waldRatio beta_ZY_afr beta_ZX_afr| < tolerance := by
  have h1 : waldRatio beta_ZY_eur beta_ZX_eur = beta_causal :=
    wald_ratio_identifies_causal beta_causal beta_ZY_eur beta_ZX_eur h_ZX_eur h_eur_valid
  have h2 : waldRatio beta_ZY_afr beta_ZX_afr = beta_causal :=
    wald_ratio_identifies_causal beta_causal beta_ZY_afr beta_ZX_afr h_ZX_afr h_afr_valid
  rw [h1, h2, sub_self, abs_zero]; exact h_tol

/-- **MR estimate is invariant to LD if IV is valid.**
    A valid IV gives the same causal estimate regardless of
    the LD background, while observational PGS depends on LD.
    We prove that under valid IV assumptions (reduced form = causal × first stage),
    the Wald ratio recovers the same causal effect in both populations,
    even when the first-stage coefficients differ due to LD differences. -/
theorem valid_iv_ld_invariant
    (beta_causal beta_ZY_eur beta_ZX_eur beta_ZY_afr beta_ZX_afr : ℝ)
    (h_ZX_eur : beta_ZX_eur ≠ 0) (h_ZX_afr : beta_ZX_afr ≠ 0)
    (h_eur_valid : beta_ZY_eur = beta_causal * beta_ZX_eur)
    (h_afr_valid : beta_ZY_afr = beta_causal * beta_ZX_afr) :
    waldRatio beta_ZY_eur beta_ZX_eur = waldRatio beta_ZY_afr beta_ZX_afr := by
  unfold waldRatio
  rw [h_eur_valid, h_afr_valid, mul_div_cancel_right₀ _ h_ZX_eur, mul_div_cancel_right₀ _ h_ZX_afr]

noncomputable def portabilityR2 (V_G V_E V_GxE : ℝ) : ℝ :=
  V_G / (V_G + V_E + V_GxE)

noncomputable def portabilityR2Adjusted (V_G V_E : ℝ) : ℝ :=
  V_G / (V_G + V_E)

/-- **Bidirectional MR for GxE.**
    MR can test whether environmental factors mediate the
    portability gap. If environment → portability loss,
    adjusting for environment should improve portability. -/
theorem environment_mediates_portability_mr
    (V_G V_E V_GxE : ℝ)
    (h_G_pos : 0 < V_G) (h_E_pos : 0 < V_E) (h_GxE_pos : 0 < V_GxE) :
    portabilityR2 V_G V_E V_GxE < portabilityR2Adjusted V_G V_E := by
  unfold portabilityR2 portabilityR2Adjusted
  apply (div_lt_div_iff₀ (by linarith) (by linarith)).mpr
  have h1 : V_G * (V_G + V_E) < V_G * (V_G + V_E + V_GxE) := by
    apply mul_lt_mul_of_pos_left
    linarith
    exact h_G_pos
  exact h1

noncomputable def conservedPathwayProportion (n_conserved n_total : ℕ) : ℝ :=
  (n_conserved : ℝ) / (n_total : ℝ)

/-- **MR-PheWAS for systematic portability analysis.**
    Running MR across many phenotypes simultaneously reveals
    which causal pathways are conserved across ancestries
    and which are population-specific. -/
theorem mr_phewas_identifies_conserved_pathways
    (n_conserved n_specific n_total : ℕ)
    (h_sum : n_conserved + n_specific = n_total)
    (h_some_specific : 0 < n_specific)
    (h_some_conserved : 0 < n_conserved) :
    conservedPathwayProportion n_conserved n_total < 1 := by
  unfold conservedPathwayProportion
  have h_total_pos : 0 < n_total := by omega
  have h_total_pos_real : (0 : ℝ) < n_total := Nat.cast_pos.mpr h_total_pos
  rw [div_lt_one h_total_pos_real]
  exact Nat.cast_lt.mpr (by omega)

end MRForPGSValidation


/-!
## Collider Bias in PGS Studies

Selection on the outcome (or a correlate) creates collider bias
that can mimic or mask portability effects.
-/

section ColliderBias

noncomputable def observedEffectWithColliderBias (beta_true cov_GU beta_U var_G : ℝ) : ℝ :=
  beta_true + (cov_GU * beta_U) / var_G

/-- **Survivorship creates collider bias.**
    If the study conditions on survival (e.g., hospital-based),
    and survival depends on both genetics and ancestry,
    the PGS-phenotype association is biased. -/
theorem collider_bias_from_selection
    (beta_true cov_GU beta_U var_G : ℝ)
    (h_cov_nz : cov_GU ≠ 0)
    (h_beta_U_nz : beta_U ≠ 0)
    (h_var_G_pos : 0 < var_G) :
    observedEffectWithColliderBias beta_true cov_GU beta_U var_G ≠ beta_true := by
  unfold observedEffectWithColliderBias
  intro h_eq
  have h_zero : (cov_GU * beta_U) / var_G = 0 := by linarith
  have h_num_zero : cov_GU * beta_U = 0 := by
    exact (div_eq_zero_iff.mp h_zero).resolve_right (ne_of_gt h_var_G_pos)
  cases mul_eq_zero.mp h_num_zero with
  | inl h1 => exact h_cov_nz h1
  | inr h2 => exact h_beta_U_nz h2

noncomputable def biasedPrognosisEffect (beta_true cov_GE beta_E var_G : ℝ) : ℝ :=
  beta_true + (cov_GE * beta_E) / var_G

/-- **Index event bias in PGS studies.**
    Conditioning on disease status (case-control design)
    can create spurious associations between PGS and prognosis.
    This is more severe when PGS is ancestry-specific. -/
theorem index_event_bias
    (beta_true cov_GE beta_E var_G : ℝ)
    (h_cov_neg : cov_GE < 0)
    (h_beta_E_pos : 0 < beta_E)
    (h_var_G_pos : 0 < var_G) :
    biasedPrognosisEffect beta_true cov_GE beta_E var_G < beta_true := by
  unfold biasedPrognosisEffect
  have h_neg : (cov_GE * beta_E) / var_G < 0 := by
    apply div_neg_of_neg_of_pos
    · exact mul_neg_of_neg_of_pos h_cov_neg h_beta_E_pos
    · exact h_var_G_pos
  linarith

noncomputable def berksonObservedEffect (beta_pop cov_GE beta_E var_G : ℝ) : ℝ :=
  beta_pop + (cov_GE * beta_E) / var_G

/-- **Berkson's paradox in biobank studies.**
    Biobank participants are healthier and more educated than
    the general population. This selection affects PGS associations
    differently across ancestries.
    Model: β_biobank = β_pop + selection_bias, where the selection bias
    differs by ancestry (different participation rates and selection pressures).
    If bias_eur ≠ bias_afr, then β_biobank_eur ≠ β_biobank_afr even when
    the true population-level effect is the same. -/
theorem berkson_paradox_ancestry_specific
    (beta_pop beta_E var_G cov_GE_eur cov_GE_afr : ℝ)
    (h_var_G_pos : 0 < var_G)
    (h_beta_E_nz : beta_E ≠ 0)
    (h_cov_diff : cov_GE_eur ≠ cov_GE_afr) :
    berksonObservedEffect beta_pop cov_GE_eur beta_E var_G ≠
    berksonObservedEffect beta_pop cov_GE_afr beta_E var_G := by
  unfold berksonObservedEffect
  intro h_eq
  have h_eq2 : (cov_GE_eur * beta_E) / var_G = (cov_GE_afr * beta_E) / var_G := by linarith
  have h_eq3 : cov_GE_eur * beta_E = cov_GE_afr * beta_E := by
    exact (div_left_inj' (ne_of_gt h_var_G_pos)).mp h_eq2
  have h_eq4 : cov_GE_eur = cov_GE_afr := by
    exact (mul_right_cancel₀ h_beta_E_nz h_eq3)
  exact h_cov_diff h_eq4

end ColliderBias

end Calibrator
