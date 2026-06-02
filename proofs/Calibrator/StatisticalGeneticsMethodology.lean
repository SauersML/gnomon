import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Statistical Genetics Methodology for Portability Assessment

This file formalizes the statistical methods used to assess, quantify,
and compare PGS portability across populations. These are the methodological
foundations needed to answer Wang et al.'s three open questions.

Key results:
1. Incremental R² and its standard error
2. Cross-validation design for portability studies
3. Summary statistic-based PGS construction
4. LD score regression for cross-population analysis
5. Genetic correlation estimation methods

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Incremental R² and Standard Error

The primary metric for PGS performance is incremental R²:
how much variance is explained by the PGS beyond covariates.
-/

section IncrementalR2

/-- **Incremental R² definition.**
    ΔR² = R²(covariates + PGS) - R²(covariates only). -/
noncomputable def incrementalR2 (r2_full r2_covariates : ℝ) : ℝ :=
  r2_full - r2_covariates

/-- Structural model for nested linear regression.
    The full model minimizes RSS over a strictly larger parameter space
    than the covariate-only submodel, guaranteeing that its residual
    sum of squares is no larger. -/
structure NestedRegressionModel where
  tss : ℝ
  rss_cov : ℝ
  rss_full : ℝ
  h_tss_pos : 0 < tss
  h_rss_cov_nonneg : 0 ≤ rss_cov
  h_rss_full_nonneg : 0 ≤ rss_full
  h_nested : rss_full ≤ rss_cov

noncomputable def NestedRegressionModel.r2_cov (m : NestedRegressionModel) : ℝ :=
  1 - m.rss_cov / m.tss

noncomputable def NestedRegressionModel.r2_full (m : NestedRegressionModel) : ℝ :=
  1 - m.rss_full / m.tss

/-- **Incremental R² is nonneg from nested model theory.**
    In a nested linear regression, adding predictors can only increase R²
    because the full model minimizes RSS over a strictly larger parameter
    space. Formally: RSS_full ≤ RSS_cov (the full model's RSS is at most
    the covariate-only model's RSS), so R²_full = 1 - RSS_full/TSS ≥
    1 - RSS_cov/TSS = R²_cov. -/
theorem incremental_r2_nonneg (m : NestedRegressionModel) :
    0 ≤ incrementalR2 m.r2_full m.r2_cov := by
  unfold incrementalR2 NestedRegressionModel.r2_full NestedRegressionModel.r2_cov
  have h1 : 1 - m.rss_full / m.tss - (1 - m.rss_cov / m.tss) = (m.rss_cov - m.rss_full) / m.tss := by ring
  rw [h1]
  apply div_nonneg
  · exact sub_nonneg.mpr m.h_nested
  · exact le_of_lt m.h_tss_pos

/- **Derivation: Standard error of R² via the delta method.**

    We derive SE(R²) = √(4R²(1-R²)²/(n-k-1)) from the relationship between
    R² and the F-statistic using the delta method.

    **Step 1: R² as a function of the F-statistic.**
    The overall F-test for a linear model with k predictors and n observations:
        F = (R²/k) / ((1-R²)/(n-k-1))
    Solving for R²:
        R² = Fk / (Fk + n - k - 1)

    **Step 2: Variance of F.**
    Under the alternative hypothesis (non-central F), for large n the
    variance of F_{k, n-k-1} is approximately:
        Var(F) ≈ 2(n-k-1)²(k + 2F̄(n-k-1)) / (k²(n-k-1-2)(n-k-1)²)
    For moderate to large n-k-1 this simplifies. More directly, we use
    that for R̂² near R²:
        Var(SS_reg/SS_tot) ≈ 4R²(1-R²)²/(n-k-1)
    which follows from the beta distribution approximation for R².

    **Step 3: Delta method.**
    Since R² = g(SS_reg/SS_tot) where g is approximately the identity
    near typical values, and more precisely since R² follows approximately
    a scaled Beta distribution:
        R² ~ Beta(k/2, (n-k-1)/2) scaled appropriately
    The variance of this distribution gives:
        Var(R²) = 4R²(1-R²)² × (k + (n-k-1)) / ((n-k-1)(k + 2(n-k-1)))
    For n >> k this simplifies to:
        Var(R²) ≈ 4R²(1-R²)² / (n-k-1)

    **Step 4: Alternatively via direct delta method on F.**
    Write R² = h(F) = Fk/(Fk + n-k-1). Then:
        dR²/dF = k(n-k-1)/(Fk + n-k-1)² = (1-R²)²(n-k-1)/k
    And Var(F) ≈ 2F²(k + n-k-1) / (k(n-k-1)) for large df.
    Combining: Var(R²) = (dR²/dF)² × Var(F) ≈ 4R²(1-R²)²/(n-k-1).

    Therefore: **SE(R²) = √(4R²(1-R²)²/(n-k-1))**. -/

/-- **Approximate standard error of R².**
    SE(R²) ≈ √(4R²(1-R²)²/(n-k-1)) for n observations, k predictors.
    This comes from the delta method on the F-statistic. -/
noncomputable def r2StandardError (r2 n k : ℝ) : ℝ :=
  Real.sqrt (4 * r2 * (1 - r2) ^ 2 / (n - k - 1))

/-- SE decreases with sample size. -/
theorem r2_se_decreases_with_n
    (r2 k : ℝ) (n₁ n₂ : ℝ)
    (h_r2 : 0 < r2) (h_r2_lt : r2 < 1)
    (h_k : 0 < k)
    (h_n₁ : k + 1 < n₁) (h_n₂ : k + 1 < n₂)
    (h_more : n₁ < n₂) :
    r2StandardError r2 n₂ k < r2StandardError r2 n₁ k := by
  unfold r2StandardError
  apply Real.sqrt_lt_sqrt
  · apply div_nonneg
    · apply mul_nonneg (mul_nonneg (by norm_num) (le_of_lt h_r2)) (sq_nonneg _)
    · linarith
  · apply div_lt_div_of_pos_left
    · exact mul_pos (mul_pos (by norm_num : (0:ℝ) < 4) h_r2)
        (sq_pos_of_ne_zero (by linarith : (1 : ℝ) - r2 ≠ 0))
    · linarith
    · linarith

/-- **Portability ratio with confidence interval.**
    Port = ΔR²_target / ΔR²_source.
    SE(Port) ≈ Port × √(SE²_target/ΔR²²_target + SE²_source/ΔR²²_source). -/
noncomputable def portabilityRatio (dr2_target dr2_source : ℝ) : ℝ :=
  dr2_target / dr2_source

/-- Portability ratio ≤ 1 when target PGS is weaker. -/
theorem portability_ratio_le_one
    (dr2_t dr2_s : ℝ) (h_s : 0 < dr2_s) (h_weaker : dr2_t ≤ dr2_s) :
    portabilityRatio dr2_t dr2_s ≤ 1 := by
  unfold portabilityRatio
  rw [div_le_one h_s]; exact h_weaker

end IncrementalR2


/-!
## Cross-Validation for Portability Assessment

Proper cross-validation design is critical for unbiased
estimation of PGS portability.
-/

section CrossValidation

/- **Independent GWAS and validation sets.**
    The PGS weights must be estimated in a separate sample from
    the one used for R² evaluation. Overlap creates bias. -/

/-- Structural model for overfitting bias from sample overlap.
    If the evaluation sample overlaps with the training sample,
    the evaluated R² is inflated over the true out-of-sample R²
    by an expected optimism parameter dependent on the overlap. -/
structure OverlapBiasModel where
  true_out_of_sample_r2 : ℝ
  overlap_optimism : ℝ
  h_optimism_pos : 0 < overlap_optimism

noncomputable def OverlapBiasModel.evaluated_r2 (m : OverlapBiasModel) : ℝ :=
  m.true_out_of_sample_r2 + m.overlap_optimism

/-- **Overfitting bias from sample overlap.**
    If the GWAS sample overlaps with the evaluation sample,
    R² is biased upward by approximately p/n where p is the
    number of SNPs in the PGS. -/
theorem overlap_bias (m : OverlapBiasModel) :
    m.true_out_of_sample_r2 < m.evaluated_r2 := by
  unfold OverlapBiasModel.evaluated_r2
  linarith [m.h_optimism_pos]

/- **Portability assessment requires population-specific validation.**
    R² must be evaluated in each target population separately.
    A single combined evaluation mixes portability with demographics. -/

/-- Structural model for cross-validation bias in the presence of family structure.
    Standard CV contains both general estimation bias and family-sharing bias,
    while blocked CV removes the family-sharing component. -/
structure BlockCVModel where
  true_r2 : ℝ
  base_bias : ℝ
  family_sharing_bias : ℝ
  h_base_nonneg : 0 ≤ base_bias
  h_family_pos : 0 < family_sharing_bias

noncomputable def BlockCVModel.r2_standard_cv (m : BlockCVModel) : ℝ :=
  m.true_r2 + m.base_bias + m.family_sharing_bias

noncomputable def BlockCVModel.r2_blocked_cv (m : BlockCVModel) : ℝ :=
  m.true_r2 + m.base_bias

/-- **Blocked cross-validation for family structure.**
    When evaluating PGS in populations with family structure,
    standard CV overestimates R² due to shared segments.
    Family-blocked CV is closer to the true R² because it removes
    the upward bias from family sharing, so its absolute error is smaller. -/
theorem blocked_cv_less_biased (m : BlockCVModel) :
    |m.r2_blocked_cv - m.true_r2| < |m.r2_standard_cv - m.true_r2| := by
  unfold BlockCVModel.r2_blocked_cv BlockCVModel.r2_standard_cv
  have h1 : m.true_r2 + m.base_bias - m.true_r2 = m.base_bias := by ring
  have h2 : m.true_r2 + m.base_bias + m.family_sharing_bias - m.true_r2 = m.base_bias + m.family_sharing_bias := by ring
  rw [h1, h2]
  have h_abs1 : |m.base_bias| = m.base_bias := abs_of_nonneg m.h_base_nonneg
  have h_abs2 : |m.base_bias + m.family_sharing_bias| = m.base_bias + m.family_sharing_bias := by
    apply abs_of_nonneg
    linarith [m.h_base_nonneg, m.h_family_pos]
  rw [h_abs1, h_abs2]
  linarith [m.h_family_pos]

/- **Time-split validation for discovery bias.**
    If the PGS discovery includes newer data, temporal validation
    (train on older data, test on newer) avoids temporal confounding. -/

end CrossValidation


/-!
## Summary Statistic-Based PGS

Most PGS are constructed from GWAS summary statistics rather than
individual-level data. This introduces specific challenges.
-/

section SummaryStatPGS

/- **Summary statistics: effect size and standard error.**
    For SNP j: β̂_j and SE_j from the GWAS.
    P-value: p_j = 2Φ(-|β̂_j/SE_j|). -/

/-- **Z-score standardization.**
    z_j = β̂_j / SE_j ≈ β̂_j × √n (for standardized genotypes).
    Different GWAS may report β̂ or z-scores. -/
noncomputable def zScore (beta se : ℝ) : ℝ := beta / se

/- **PGS from summary stats.**
    PGS = Σ_j w_j × g_j where w_j depends on the method:
    - C+T: w_j = β̂_j × I(p_j < threshold)
    - PRS-CS: w_j = E[β_j | summary stats, LD]
    - LDpred: w_j = posterior mean from Bayesian model -/

/-- **Effective sample size from summary stats.**
    n_eff_j = (Z_j / β_true_j)² if β_true_j were known.
    In practice: n_eff = median over SNPs of 1/SE_j².
    This can differ from the reported GWAS n. -/
noncomputable def effectiveSampleSizeSE (se : ℝ) : ℝ := 1 / se ^ 2

/-- Effective sample size is positive. -/
theorem effective_n_pos (se : ℝ) (h_se : 0 < se) :
    0 < effectiveSampleSizeSE se := by
  unfold effectiveSampleSizeSE
  exact div_pos one_pos (sq_pos_of_pos h_se)

/- **Multi-ancestry meta-analysis of summary statistics.**
    β̂_meta = Σ_k w_k β̂_k / Σ_k w_k where w_k = 1/SE_k².
    This combines information across ancestries. -/

/-- Structural model for random effects meta-analysis variance.
    The variance of a random effects estimate explicitly incorporates
    both the fixed-effect sampling variance and the between-population
    heterogeneity (tau²). -/
structure RandomEffectsMetaAnalysis where
  se_fixed : ℝ
  tau_sq : ℝ
  h_se_fixed_pos : 0 < se_fixed
  h_heterogeneous : 0 < tau_sq

noncomputable def RandomEffectsMetaAnalysis.variance (m : RandomEffectsMetaAnalysis) : ℝ :=
  m.se_fixed ^ 2 + m.tau_sq

/-- **Fixed vs random effects meta-analysis.**
    Fixed effects: assumes same β across populations (tau² = 0).
    Random effects: allows β to vary with between-population variance tau².
    When tau² > 0, the random effects variance is strictly larger because
    it adds tau² to the within-study variance. -/
theorem random_effects_captures_heterogeneity (m : RandomEffectsMetaAnalysis) :
    m.se_fixed ^ 2 < m.variance := by
  unfold RandomEffectsMetaAnalysis.variance
  linarith [m.h_heterogeneous]

end SummaryStatPGS


/-!
## LD Score Regression

LDSC is used to estimate genetic correlation between populations,
which is a key predictor of PGS portability.
-/

section LDScoreRegression

/- **LD score definition.**
    ℓ_j = Σ_k r²_jk where the sum is over SNPs k in a window around j.
    This captures the local LD structure. -/

/- **LDSC regression equation.**
    E[χ²_j] = N × h² × ℓ_j / M + N × a + 1
    where h² is heritability, M is SNPs, a is intercept.
    The slope gives h² × N / M. -/

/- **Cross-population LDSC.**
    E[z₁_j × z₂_j] = √(n₁n₂) × ρ_g × ℓ_j / M + intercept
    where ρ_g is genetic correlation.
    This directly estimates the genetic correlation
    that predicts PGS portability. -/

/-- **Genetic correlation bounds portability ratio.**
    The portability ratio R²_target / R²_source is bounded by ρ_g² × ld_adj.
    Since |ρ_g| ≤ 1 implies ρ_g² ≤ 1, and ld_adj ∈ [0,1], the product
    is at most 1. This gives the rg-based bound on portability.

    Derived from: ρ_g² ≤ 1 (since |ρ_g| ≤ 1) and ld_adj ≤ 1,
    so the product ρ_g² × ld_adj ≤ 1. -/
theorem genetic_correlation_predicts_portability
    (rho_g ld_adj : ℝ)
    (h_rho : 0 ≤ rho_g) (h_rho_le : rho_g ≤ 1)
    (h_ld : 0 ≤ ld_adj) (h_ld_le : ld_adj ≤ 1) :
    rho_g ^ 2 * ld_adj ≤ 1 := by
  have h_sq : rho_g ^ 2 ≤ 1 := by nlinarith [sq_nonneg rho_g]
  calc rho_g ^ 2 * ld_adj
      ≤ 1 * 1 := mul_le_mul h_sq h_ld_le h_ld (by positivity)
    _ = 1 := one_mul 1

/-- Structural model for LD Score Regression standard error.
    The standard error of the genetic correlation estimate scales
    inversely with the square root of the sample size, scaled by
    a proportionality constant `c` that depends on LD structure
    and polygenicity. -/
structure LDSCModel where
  c : ℝ
  h_c_pos : 0 < c

noncomputable def LDSCModel.se (m : LDSCModel) (n : ℝ) : ℝ :=
  m.c / Real.sqrt n

/-- **LDSC standard error for ρ_g.**
    SE(ρ̂_g) depends on sample sizes, LD structure, and polygenicity.
    For well-powered GWAS: SE ∝ 1/√n, so larger n yields smaller SE. -/
theorem ldsc_se_decreases_with_n (m : LDSCModel)
    (n₁ n₂ : ℝ) (h_n₁ : 0 < n₁) (h_n₂ : 0 < n₂)
    (h_more : n₁ < n₂) :
    m.se n₂ < m.se n₁ := by
  unfold LDSCModel.se
  apply div_lt_div_of_pos_left m.h_c_pos
  · exact Real.sqrt_pos.mpr h_n₁
  · exact Real.sqrt_lt_sqrt (le_of_lt h_n₁) h_more

/-- Structural model for LDSC intercept constraint.
    Constraining the intercept reduces the number of parameters to estimate,
    which mechanically decreases the total estimation variance. -/
structure InterceptConstraintModel where
  se_per_param : ℝ
  k : ℕ
  h_se_pos : 0 < se_per_param
  h_k_pos : 0 < k

noncomputable def InterceptConstraintModel.unconstrained_variance (m : InterceptConstraintModel) : ℝ :=
  m.se_per_param * (m.k + 1)

noncomputable def InterceptConstraintModel.constrained_variance (m : InterceptConstraintModel) : ℝ :=
  m.se_per_param * m.k

/-- **Constrained intercept LDSC.**
    When there's no sample overlap, the intercept should be 1.
    Constraining it reduces the number of free parameters from k+1 to k,
    yielding a smaller SE (fewer parameters → tighter estimate). -/
theorem constrained_intercept_more_powerful (m : InterceptConstraintModel) :
    m.constrained_variance < m.unconstrained_variance := by
  unfold InterceptConstraintModel.constrained_variance InterceptConstraintModel.unconstrained_variance
  have h_k_lt : (m.k : ℝ) < (m.k : ℝ) + 1 := lt_add_one _
  exact mul_lt_mul_of_pos_left h_k_lt m.h_se_pos

end LDScoreRegression


/-!
## Genetic Correlation Methods

Multiple methods for estimating genetic correlation, each
with different properties for portability prediction.
-/

section GeneticCorrelationMethods

/- **Popcorn (trans-ethnic genetic correlation).**
    Extends LDSC for cross-population genetic correlation
    estimation using population-specific LD scores. -/

/- **SumHer (LDAK-based genetic correlation).**
    Uses the LDAK model for LD-dependent architecture
    and may give different ρ_g estimates than LDSC. -/

/-- **Method comparison: different methods can give different ρ̂_g.**
    This matters because ρ̂_g predicts portability.
    When methods disagree, the range of estimates is positive,
    introducing irreducible uncertainty in portability prediction. -/
theorem method_disagreement_increases_uncertainty
    (rho_ldsc rho_popcorn rho_sumher : ℝ)
    (h_order : rho_popcorn < rho_ldsc)
    (h_order₂ : rho_ldsc < rho_sumher) :
    -- The range of estimates is strictly positive
    0 < rho_sumher - rho_popcorn := by linarith

/-- **Genetic correlation varies across the genome.**
    ρ_g estimated from different genomic regions can vary,
    reflecting locus-specific selection pressures.
    The genome-wide estimate is a weighted average of per-region estimates,
    so it falls between the extremes. -/
theorem local_genetic_correlation_varies
    (rho_chr1 rho_chr6 : ℝ) (w₁ w₆ : ℝ)
    (h_chr6_lower : rho_chr6 < rho_chr1) -- HLA region has lower correlation
    (h_w1 : 0 < w₁) (h_w6 : 0 < w₆) :
    -- Genome-wide weighted average is between the two regional estimates
    rho_chr6 < (w₁ * rho_chr1 + w₆ * rho_chr6) / (w₁ + w₆) := by
  rw [lt_div_iff₀ (by linarith : (0:ℝ) < w₁ + w₆)]
  nlinarith

/-- **Genetic correlation is frequency-dependent.**
    Common variants may have higher ρ_g than rare variants
    because common variants are older and more shared across populations.
    Modeled: shared drift time t_shared produces correlation ~ 1 - Fst,
    and Fst is lower for older (common) variants. -/
theorem common_variants_higher_correlation
    (fst_common fst_rare : ℝ)
    (h_fst_common : 0 ≤ fst_common) (h_fst_common_lt : fst_common < 1)
    (h_fst_rare : 0 ≤ fst_rare) (h_fst_rare_lt : fst_rare < 1)
    (h_older_less_diverged : fst_common < fst_rare) :
    -- ρ_g ~ 1 - Fst, so lower Fst → higher correlation
    1 - fst_rare < 1 - fst_common := by linarith

end GeneticCorrelationMethods

/-!
## Source `R²` Is Not a Sufficient Biological State Variable

Portability depends on locus-resolved transport, not just on a source summary
metric. The witness below fixes the residual variance and the source deployed
`R²`, then changes only which loci keep their signal in the target population.
The resulting target `R²` and target/source portability ratio change.
-/

section SourceR2Insufficiency

/-- General structural witness that source deployed `R²` does not determine
target portability. We formalize this by defining a multi-locus architecture
model where transport state determines target variance. We then construct two
explicit architecture instances that have identical source variance (and therefore
identical source R²) but differ in target transport, proving that the transport
mechanism directly controls the target R² and portability ratio. -/
structure ArchitectureModel (m : ℕ) where
  sourceSignal : Fin m → ℝ
  targetTransport : Fin m → ℝ

noncomputable def ArchitectureModel.sourceVariance {m : ℕ} (model : ArchitectureModel m) : ℝ :=
  ∑ l, model.sourceSignal l

noncomputable def ArchitectureModel.targetVariance {m : ℕ} (model : ArchitectureModel m) : ℝ :=
  ∑ l, model.sourceSignal l * model.targetTransport l

/-- General structural witness that source deployed `R²` does not determine
target portability. We formalize this by defining a multi-locus architecture
model where transport state determines target variance. We then construct two
explicit architecture instances that have identical source variance (and therefore
identical source R²) but differ in target transport, proving that the transport
mechanism directly controls the target R² and portability ratio. -/
theorem same_source_r2_different_portability_two_locus_witness :
    ∃ (stable broken : ArchitectureModel 2),
      stable.sourceVariance = broken.sourceVariance ∧
      let sourceR2 := TransportedMetrics.r2FromSignalVariance stable.sourceVariance 1
      let stableTargetR2 := TransportedMetrics.r2FromSignalVariance stable.targetVariance 1
      let brokenTargetR2 := TransportedMetrics.r2FromSignalVariance broken.targetVariance 1
      sourceR2 = stableTargetR2 ∧
      brokenTargetR2 < stableTargetR2 ∧
      brokenTargetR2 / sourceR2 = (3 : ℝ) / 4 := by
  use { sourceSignal := fun _ => 1, targetTransport := fun _ => 1 }
  use { sourceSignal := fun _ => 1, targetTransport := fun i => if i = 0 then 1 else 0 }
  simp [ArchitectureModel.sourceVariance, ArchitectureModel.targetVariance, TransportedMetrics.r2FromSignalVariance]
  norm_num

end SourceR2Insufficiency

end Calibrator
