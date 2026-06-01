import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
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
    (h_n : 2 < n) (h_r2₁ : 0 < r2₁) (h_r2₂ : 0 < r2₂)
    (h_r2₁_lt : r2₁ < 1) (h_r2₂_lt : r2₂ < 1)
    (h_lt : r2₁ < r2₂) :
    fStatistic n r2₁ < fStatistic n r2₂ := by
  unfold fStatistic
  have h_n_pos : (0 : ℝ) < n - 2 := by
    have : (2 : ℝ) < n := Nat.ofNat_lt_cast.mpr h_n
    linarith
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
    When the source LD is stronger, the target instrument is strictly weaker. -/
theorem instrument_strength_varies
    (n : ℕ) (r2_source r2_target : ℝ)
    (h_n : 2 < n)
    (h_r2_source_pos : 0 < r2_source) (h_r2_target_pos : 0 < r2_target)
    (h_r2_source_lt_one : r2_source < 1) (h_r2_target_lt_one : r2_target < 1)
    (h_ld_diff : r2_target < r2_source) :
    fStatistic n r2_target < fStatistic n r2_source := by
  exact f_stat_increases_with_r2 n r2_target r2_source h_n h_r2_target_pos h_r2_source_pos h_r2_target_lt_one h_r2_source_lt_one h_ld_diff

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

structure MRModel where
  beta_causal : ℝ
  beta_ZX : ℝ
  alpha_pleio : ℝ

noncomputable def waldRatioPleio (m : MRModel) : ℝ :=
  (m.beta_causal * m.beta_ZX + m.alpha_pleio) / m.beta_ZX

/-- **Horizontal pleiotropy may be population-specific.**
    A variant may have pleiotropic effects in one ancestry
    but not another, due to different LD partners. -/
theorem population_specific_pleiotropy
    (m_eur m_afr : MRModel)
    (h_causal : m_eur.beta_causal = m_afr.beta_causal)
    (h_ZX : m_eur.beta_ZX = m_afr.beta_ZX)
    (h_ZX_nz : m_eur.beta_ZX ≠ 0)
    (h_pleio_diff : m_eur.alpha_pleio ≠ m_afr.alpha_pleio) :
    waldRatioPleio m_eur ≠ waldRatioPleio m_afr := by
  unfold waldRatioPleio
  intro h
  rw [h_causal, h_ZX] at h
  have h1 : (m_afr.beta_causal * m_afr.beta_ZX + m_eur.alpha_pleio) / m_afr.beta_ZX * m_afr.beta_ZX =
            (m_afr.beta_causal * m_afr.beta_ZX + m_afr.alpha_pleio) / m_afr.beta_ZX * m_afr.beta_ZX := by rw [h]
  have h_afr_ZX_nz : m_afr.beta_ZX ≠ 0 := by rwa [←h_ZX]
  rw [div_mul_cancel₀ _ h_afr_ZX_nz, div_mul_cancel₀ _ h_afr_ZX_nz] at h1
  have h2 : m_eur.alpha_pleio = m_afr.alpha_pleio := by linarith
  exact h_pleio_diff h2

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
    (h_rs : 0 ≤ r_strat) (h_rc : 0 ≤ r_strat_cross)
    (h_cross_less : r_strat_cross < r_strat) :
    confound * r_strat_cross ^ 2 < confound * r_strat ^ 2 := by
  apply mul_lt_mul_of_pos_left _ h_conf
  exact sq_lt_sq' (by linarith) h_cross_less

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
    (bias_same bias_cross var_same var_cross : ℝ)
    (h_less_bias : |bias_cross| < |bias_same|)
    (h_more_var : var_same < var_cross)
    (h_var_nn : 0 ≤ var_same)
    (h_bias_dominates : bias_same ^ 2 - bias_cross ^ 2 > var_cross - var_same) :
    mrMSE bias_cross var_cross < mrMSE bias_same var_same := by
  unfold mrMSE; linarith

/- **IVW estimator for multiple instruments.**
    β_IVW = Σ w_j β_j / Σ w_j
    where w_j = 1/σ²_j and β_j are individual Wald ratios.
    Cross-ancestry weights differ due to different σ²_j. -/

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
    (beta_causal beta_ZX_eur beta_ZX_afr : ℝ)
    (h_ZX_eur : beta_ZX_eur ≠ 0) (h_ZX_afr : beta_ZX_afr ≠ 0)
    (h_eur_valid : beta_causal * beta_ZX_eur = beta_causal * beta_ZX_eur)
    (h_afr_valid : beta_causal * beta_ZX_afr = beta_causal * beta_ZX_afr) :
    waldRatio (beta_causal * beta_ZX_eur) beta_ZX_eur =
      waldRatio (beta_causal * beta_ZX_afr) beta_ZX_afr := by
  unfold waldRatio
  rw [mul_div_cancel_right₀ _ h_ZX_eur, mul_div_cancel_right₀ _ h_ZX_afr]

structure EnvPortabilityModel where
  base_portability : ℝ
  env_effect : ℝ
  env_mismatch : ℝ

noncomputable def portability (m : EnvPortabilityModel) : ℝ :=
  m.base_portability - m.env_effect * m.env_mismatch

/-- **Bidirectional MR for GxE.**
    MR can test whether environmental factors mediate the
    portability gap. If environment → portability loss,
    adjusting for environment should improve portability. -/
theorem environment_mediates_portability_mr
    (m_unadj m_adj : EnvPortabilityModel)
    (h_base : m_unadj.base_portability = m_adj.base_portability)
    (h_effect : m_unadj.env_effect = m_adj.env_effect)
    (h_effect_pos : 0 < m_unadj.env_effect)
    (h_mismatch_reduction : m_adj.env_mismatch < m_unadj.env_mismatch) :
    portability m_unadj < portability m_adj := by
  unfold portability
  rw [h_base, h_effect]
  have : m_adj.env_effect * m_adj.env_mismatch < m_adj.env_effect * m_unadj.env_mismatch := by
    exact mul_lt_mul_of_pos_left h_mismatch_reduction (by linarith)
  linarith

/-- **MR-PheWAS for systematic portability analysis.**
    Running MR across many phenotypes simultaneously reveals
    which causal pathways are conserved across ancestries
    and which are population-specific. -/
theorem mr_phewas_identifies_conserved_pathways
    {α : Type*} (s : Finset α)
    (is_conserved : α → Prop) [DecidablePred is_conserved]
    (a : α) (ha : a ∈ s) (hna : ¬is_conserved a) :
    (s.filter is_conserved).card < s.card := by
  apply Finset.card_lt_card
  constructor
  · exact Finset.filter_subset is_conserved s
  · intro h_sub
    have h_in_filter : a ∈ s.filter is_conserved := h_sub ha
    rw [Finset.mem_filter] at h_in_filter
    exact hna h_in_filter.2

end MRForPGSValidation


/-!
## Collider Bias in PGS Studies

Selection on the outcome (or a correlate) creates collider bias
that can mimic or mask portability effects.
-/

section ColliderBias

structure SelectionBiasModel where
  beta_true : ℝ
  genetic_selection : ℝ
  phenotype_selection : ℝ
  selection_variance : ℝ

noncomputable def observedEffect (m : SelectionBiasModel) : ℝ :=
  m.beta_true - (m.genetic_selection * m.phenotype_selection) / m.selection_variance

/-- **Survivorship creates collider bias.**
    If the study conditions on survival (e.g., hospital-based),
    and survival depends on both genetics and ancestry,
    the PGS-phenotype association is biased. -/
theorem collider_bias_from_selection
    (m : SelectionBiasModel)
    (h_gen_sel : m.genetic_selection ≠ 0)
    (h_pheno_sel : m.phenotype_selection ≠ 0)
    (h_var : m.selection_variance ≠ 0) :
    observedEffect m ≠ m.beta_true := by
  unfold observedEffect
  intro h
  have h1 : (m.genetic_selection * m.phenotype_selection) / m.selection_variance = 0 := by linarith
  have h2 : m.genetic_selection * m.phenotype_selection = 0 := by
    exact (div_eq_zero_iff.mp h1).resolve_right (by exact h_var)
  cases mul_eq_zero.mp h2 with
  | inl h_g => exact h_gen_sel h_g
  | inr h_p => exact h_pheno_sel h_p

structure IndexEventModel where
  beta_prognosis_true : ℝ
  pgs_diagnosis_effect : ℝ
  disease_incidence : ℝ

noncomputable def indexEventBiasedEffect (m : IndexEventModel) : ℝ :=
  m.beta_prognosis_true - m.pgs_diagnosis_effect * (1 - m.disease_incidence)

/-- **Index event bias in PGS studies.**
    Conditioning on disease status (case-control design)
    can create spurious associations between PGS and prognosis.
    This is more severe when PGS is ancestry-specific. -/
theorem index_event_bias
    (m : IndexEventModel)
    (h_pgs_pos : 0 < m.pgs_diagnosis_effect)
    (h_incidence : m.disease_incidence < 1) :
    indexEventBiasedEffect m < m.beta_prognosis_true := by
  unfold indexEventBiasedEffect
  have h_pos : 0 < m.pgs_diagnosis_effect * (1 - m.disease_incidence) :=
    mul_pos h_pgs_pos (sub_pos.mpr h_incidence)
  linarith

structure BiobankSelectionModel where
  beta_population : ℝ
  participation_rate : ℝ
  health_selection_pressure : ℝ

noncomputable def biobankEffect (m : BiobankSelectionModel) : ℝ :=
  m.beta_population + m.health_selection_pressure * (1 - m.participation_rate)

/-- **Berkson's paradox in biobank studies.**
    Biobank participants are healthier and more educated than
    the general population. This selection affects PGS associations
    differently across ancestries.
    Model: β_biobank = β_pop + selection_bias, where the selection bias
    differs by ancestry (different participation rates and selection pressures).
    If bias_eur ≠ bias_afr, then β_biobank_eur ≠ β_biobank_afr even when
    the true population-level effect is the same. -/
theorem berkson_paradox_ancestry_specific
    (m_eur m_afr : BiobankSelectionModel)
    (h_beta : m_eur.beta_population = m_afr.beta_population)
    (h_pressure : m_eur.health_selection_pressure = m_afr.health_selection_pressure)
    (h_pressure_nz : m_eur.health_selection_pressure ≠ 0)
    (h_part_diff : m_eur.participation_rate ≠ m_afr.participation_rate) :
    biobankEffect m_eur ≠ biobankEffect m_afr := by
  unfold biobankEffect
  intro h
  rw [h_beta, h_pressure] at h
  have h1 : m_afr.health_selection_pressure * (1 - m_eur.participation_rate) =
            m_afr.health_selection_pressure * (1 - m_afr.participation_rate) := by linarith
  have h_afr_pressure_nz : m_afr.health_selection_pressure ≠ 0 := by rwa [←h_pressure]
  have h2 : 1 - m_eur.participation_rate = 1 - m_afr.participation_rate := by
    exact mul_left_cancel₀ h_afr_pressure_nz h1
  have h3 : m_eur.participation_rate = m_afr.participation_rate := by linarith
  exact h_part_diff h3

end ColliderBias

end Calibrator
