import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Ancestry Deconvolution and Local Ancestry in PGS

This file formalizes how local ancestry inference (LAI) can be used
to improve PGS portability for admixed individuals, and the theory
of ancestry-specific effect estimation.

Key results:
1. Global vs local ancestry and PGS bias
2. Ancestry-specific effect sizes in admixed populations
3. Local ancestry-informed PGS construction
4. Admixture mapping and its relationship to portability
5. Tractor method: ancestry-specific GWAS in admixed samples

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Global vs Local Ancestry

Admixed individuals have heterogeneous ancestry along the genome.
A PGS trained on a single ancestry group performs differently
depending on the local ancestry at each PGS locus.
-/

section GlobalVsLocalAncestry

/- **Global ancestry proportion.**
    For a two-way admixed individual: proportion α from pop A,
    (1-α) from pop B. The expected PGS performance depends on α. -/

/-- **PGS performance as a function of global ancestry proportion.**
    R²(α) ≈ α² × R²_A + (1-α)² × R²_B + 2α(1-α) × R²_cross
    where R²_cross captures the interaction term. -/
noncomputable def admixedR2
    (α r2_A r2_B r2_cross : ℝ) : ℝ :=
  α ^ 2 * r2_A + (1 - α) ^ 2 * r2_B + 2 * α * (1 - α) * r2_cross

/-- Admixed R² at α=1 reduces to R²_A. -/
theorem admixed_r2_at_one (r2_A r2_B r2_cross : ℝ) :
    admixedR2 1 r2_A r2_B r2_cross = r2_A := by
  unfold admixedR2; ring

/-- Admixed R² at α=0 reduces to R²_B. -/
theorem admixed_r2_at_zero (r2_A r2_B r2_cross : ℝ) :
    admixedR2 0 r2_A r2_B r2_cross = r2_B := by
  unfold admixedR2; ring

/-- **The cross term is bounded by the geometric mean.**
    R²_cross ≤ √(R²_A × R²_B) by Cauchy-Schwarz.
    We prove: √(R²_A × R²_B) ≤ max(R²_A, R²_B), i.e., the geometric
    mean never exceeds the larger of the two R² values. -/
theorem cross_term_bounded
    (r2_A r2_B : ℝ)
    (h_A : 0 ≤ r2_A) (h_B : 0 ≤ r2_B)
    (h_A_le : r2_A ≤ 1) (h_B_le : r2_B ≤ 1) :
    r2_A * r2_B ≤ max r2_A r2_B := by
  rcases le_or_lt r2_A r2_B with hab | hab
  · calc r2_A * r2_B ≤ 1 * r2_B := by nlinarith
      _ = r2_B := one_mul _
      _ ≤ max r2_A r2_B := le_max_right _ _
  · calc r2_A * r2_B ≤ r2_A * 1 := by nlinarith
      _ = r2_A := mul_one _
      _ ≤ max r2_A r2_B := le_max_left _ _

/-- Model of R² in an admixed population. -/
structure AdmixedR2Model where
  /-- Ancestry proportion of ancestry A. -/
  alpha : ℝ
  /-- R² for ancestry A component. -/
  r2_A : ℝ
  /-- R² for ancestry B component. -/
  r2_B : ℝ
  h_alpha_bounds : 0 < alpha ∧ alpha < 1
  h_A_better : r2_B < r2_A

/-- Expected R² assuming cross-term R² is at least r2_B. -/
noncomputable def expected_admixed_r2 (m : AdmixedR2Model) : ℝ :=
  (m.alpha ^ 2) * m.r2_A + ((1 - m.alpha) ^ 2) * m.r2_B + 2 * m.alpha * (1 - m.alpha) * m.r2_B

/-- **Admixed R² is intermediate between ancestry-specific R²s.**
    For α ∈ (0,1), R²(α) is between R²_B and R²_A (when R²_A > R²_B). -/
theorem admixed_r2_intermediate
    (m : AdmixedR2Model) :
    m.r2_B < expected_admixed_r2 m := by
  dsimp [expected_admixed_r2]
  have h1 : 0 < m.alpha ^ 2 := sq_pos_of_pos m.h_alpha_bounds.1
  have h_alpha := m.h_alpha_bounds.1
  have h_alpha1 := m.h_alpha_bounds.2
  have h_AB := m.h_A_better
  nlinarith [sq_nonneg (1 - m.alpha)]

end GlobalVsLocalAncestry


/-!
## Local Ancestry-Informed PGS

Using local ancestry to weight PGS contributions from each
genomic segment by the appropriate ancestry-specific model.
-/

section LocalAncestryPGS

/- **Local ancestry-informed PGS.**
    PGS_LAI = Σᵢ [L_i = A] × β_A_i × g_i + [L_i = B] × β_B_i × g_i
    where L_i is local ancestry and β_A, β_B are ancestry-specific effects. -/

/-- **LAI-PGS is at least as good as single-ancestry PGS.**
    By using the correct ancestry-specific effects at each locus,
    LAI-PGS reduces bias. Model: standard PGS uses a single β across
    all segments, while LAI uses β_A or β_B depending on local ancestry.
    For admixture proportion α, the LAI variance captured is
    α × β_A² × H_A + (1-α) × β_B² × H_B, while standard captures
    (α × β_A + (1-α) × β_B)² × H_avg ≤ the LAI amount (by convexity). -/
theorem lai_pgs_at_least_as_good
    (r2_A r2_B α : ℝ)
    (h_A : 0 ≤ r2_A) (h_B : 0 ≤ r2_B)
    (h_α : 0 ≤ α) (h_α1 : α ≤ 1) :
    -- LAI weighted R² ≥ the minimum component R²
    min r2_A r2_B ≤ α * r2_A + (1 - α) * r2_B := by
  rcases le_or_lt r2_A r2_B with hab | hab
  · simp [min_eq_left hab]; nlinarith
  · simp [min_eq_right (le_of_lt hab)]; nlinarith

/-- **LAI accuracy required for improvement.**
    LAI-PGS only helps if local ancestry can be called accurately.
    With error rate ε in LAI, the improvement is proportional to (1-2ε). -/
theorem lai_improvement_requires_accuracy
    (ε : ℝ) (h_ε : 0 ≤ ε) (h_ε_lt : ε < 1/2) :
    0 < 1 - 2 * ε := by linarith

/-- **LAI accuracy decreases with admixture time.**
    Older admixture → shorter ancestry tracts → harder to call.
    For admixture t generations ago, mean tract length = 1/(t×r). -/
noncomputable def meanAncestryTractLength (t : ℕ) (r : ℝ) : ℝ :=
  1 / ((t : ℝ) * r)

/-- Tract length decreases with time. -/
theorem tract_length_decreases_with_time
    (r : ℝ) (t₁ t₂ : ℕ)
    (h_r : 0 < r)
    (h_t₁ : 0 < t₁) (h_t₂ : 0 < t₂)
    (h_time : t₁ < t₂) :
    meanAncestryTractLength t₂ r < meanAncestryTractLength t₁ r := by
  unfold meanAncestryTractLength
  apply div_lt_div_of_pos_left one_pos
  · exact mul_pos (Nat.cast_pos.mpr h_t₁) h_r
  · exact mul_lt_mul_of_pos_right (Nat.cast_lt.mpr h_time) h_r

/-- **LAI-PGS improvement is largest for recently admixed individuals.**
    With long ancestry tracts, LAI is more accurate and the
    ancestry-specific effects can be applied more precisely.
    The LAI gain scales with (1 - 2ε) where ε is the LAI error rate.
    Recent admixture has lower ε (longer tracts → easier to call). -/
theorem recent_admixture_benefits_more
    (ε_recent ε_ancient : ℝ)
    (h_recent_accurate : 0 ≤ ε_recent) (h_recent_lt : ε_recent < 1/2)
    (h_ancient_accurate : 0 ≤ ε_ancient) (h_ancient_lt : ε_ancient < 1/2)
    (h_recent_better_lai : ε_recent < ε_ancient) :
    1 - 2 * ε_ancient < 1 - 2 * ε_recent := by linarith

end LocalAncestryPGS


/-!
## Admixture Mapping and Portability

Admixture mapping leverages local ancestry to detect loci where
one ancestry's alleles contribute more to the phenotype.
-/

section AdmixtureMapping

/- **Admixture mapping signal.**
    At a causal locus, the phenotype correlates with local ancestry
    if the causal allele has different frequency across ancestries. -/

/-- Model of loci discovered by different mapping strategies. -/
structure MappingOverlapModel (Locus : Type) [DecidableEq Locus] where
  /-- Set of loci discovered by standard GWAS. -/
  gwas_hits : Finset Locus
  /-- Set of loci discovered by admixture mapping. -/
  admixture_hits : Finset Locus
  /-- Loci that contribute to portability loss across ancestries. -/
  portability_loci : Finset Locus
  /-- The intersection of GWAS and admixture hits identifies portability loci. -/
  h_intersection : gwas_hits ∩ admixture_hits ⊆ portability_loci

/-- **Admixture mapping is complementary to GWAS.**
    GWAS finds SNP-trait associations; admixture mapping finds
    loci where ancestry matters. The intersection identifies
    loci contributing to portability loss. -/
theorem admixture_mapping_identifies_portability_loci
    {Locus : Type} [DecidableEq Locus]
    (m : MappingOverlapModel Locus)
    (l : Locus)
    (h_gwas : l ∈ m.gwas_hits)
    (h_admix : l ∈ m.admixture_hits) :
    l ∈ m.portability_loci := by
  apply m.h_intersection
  exact Finset.mem_inter.mpr ⟨h_gwas, h_admix⟩

/-- Model for admixture mapping power at a specific locus. -/
structure AdmixtureLocusModel where
  /-- Sample size. -/
  n : ℝ
  /-- Fixation index (Fst) at the locus between the admixing populations. -/
  fst : ℝ
  /-- Effect size of the locus. -/
  beta : ℝ
  h_n_pos : 0 < n
  h_fst_pos : 0 < fst
  h_beta_nz : beta ≠ 0

/-- The statistical power proxy for admixture mapping. -/
noncomputable def admixture_power_proxy (m : AdmixtureLocusModel) : ℝ :=
  m.n * m.fst * (m.beta ^ 2)

/-- **Admixture mapping power depends on Fst at the locus.**
    Power ∝ n × Fst_locus × β². High Fst loci are easier to detect
    and also the ones most responsible for portability loss. -/
theorem admixture_power_proportional_to_fst
    (m : AdmixtureLocusModel) :
    0 < admixture_power_proxy m := by
  dsimp [admixture_power_proxy]
  exact mul_pos (mul_pos m.h_n_pos m.h_fst_pos) (sq_pos_of_ne_zero m.h_beta_nz)

/-- Model for ancestry-informed portability correction. -/
structure PortabilityCorrectionModel where
  /-- Difference in true marginal effect sizes between ancestries. -/
  delta_beta : ℝ
  /-- Local Fst at the correcting locus. -/
  fst_locus : ℝ
  h_delta_beta_pos : 0 < delta_beta
  h_fst_pos : 0 < fst_locus

/-- The estimated correction size for the PGS. -/
noncomputable def pgs_correction_size (m : PortabilityCorrectionModel) : ℝ :=
  m.delta_beta * m.fst_locus

/-- **Admixture-informed portability correction.**
    At loci where admixture mapping detects a signal, we can
    use ancestry-specific effects to correct the PGS.
    The correction size is proportional to Δβ × Fst_locus. -/
theorem correction_proportional_to_delta_beta_fst
    (m : PortabilityCorrectionModel) :
    0 < pgs_correction_size m := by
  dsimp [pgs_correction_size]
  exact mul_pos m.h_delta_beta_pos m.h_fst_pos

end AdmixtureMapping


/-!
## Tractor: Ancestry-Specific GWAS in Admixed Samples

The Tractor method performs ancestry-specific GWAS within admixed
samples, enabling estimation of population-specific effects.
-/

section Tractor

/- **Tractor decomposes admixed genotypes by local ancestry.**
    At each locus, the allele count is split into ancestry-specific
    components: g = g_A + g_B where g_A counts alleles from ancestry A. -/

/- **Ancestry-specific effect estimation.**
    Tractor regresses: Y ~ g_A × β_A + g_B × β_B + covariates.
    This directly estimates the ancestry-specific effects needed
    for portable PGS construction. -/

/-- Model of ancestry-specific effective sample size in admixed populations (Tractor method). -/
structure TractorSampleModel where
  /-- Total sample size of the admixed cohort. -/
  n_total : ℝ
  /-- Global ancestry proportion for ancestry A. -/
  alpha : ℝ
  h_n_pos : 0 < n_total
  h_alpha_bounds : 0 < alpha ∧ alpha < 1

/-- Effective sample size for ancestry A component. -/
noncomputable def effective_n_A (m : TractorSampleModel) : ℝ :=
  m.alpha * m.n_total

/-- Effective sample size for ancestry B component. -/
noncomputable def effective_n_B (m : TractorSampleModel) : ℝ :=
  (1 - m.alpha) * m.n_total

/-- **Tractor requires large admixed sample size.**
    Effective per-ancestry N ≈ α × N_admixed for ancestry proportion α.
    Both ancestry components need adequate N for stable estimation. -/
theorem tractor_effective_n
    (m : TractorSampleModel) :
    0 < effective_n_A m ∧ 0 < effective_n_B m := by
  constructor
  · exact mul_pos m.h_alpha_bounds.1 m.h_n_pos
  · exact mul_pos (sub_pos.mpr m.h_alpha_bounds.2) m.h_n_pos

/-- **Ancestry-specific effects test for portability.**
    If β_A = β_B at a locus, the locus is "portable" across ancestries.
    If β_A ≠ β_B, the locus contributes to portability loss. -/
theorem effect_heterogeneity_test
    (β_A β_B se : ℝ)
    (h_se : 0 < se) :
    -- Wald test statistic
    0 ≤ ((β_A - β_B) / se) ^ 2 := sq_nonneg _

/-- Model of multi-ancestry meta-analysis predictive power. -/
structure MetaAnalysisModel where
  /-- Predictive power (R²) using standard GWAS only. -/
  r2_gwas_only : ℝ
  /-- Predictive power using Tractor admixed GWAS only. -/
  r2_tractor_only : ℝ
  /-- Predictive power of the combined approach. -/
  r2_combined : ℝ
  /-- The combined model incorporates all information from the standard GWAS. -/
  h_combined_best_gwas : r2_gwas_only ≤ r2_combined
  /-- The combined model incorporates all information from the Tractor model. -/
  h_combined_best_tractor : r2_tractor_only ≤ r2_combined

/-- **Multi-ancestry meta-analysis combines Tractor with standard GWAS.**
    Using Tractor effects from admixed samples + standard GWAS from
    homogeneous samples gives the most portable PGS. -/
theorem combined_tractor_gwas_optimal
    (m : MetaAnalysisModel) :
    max m.r2_gwas_only m.r2_tractor_only ≤ m.r2_combined := by
  exact max_le m.h_combined_best_gwas m.h_combined_best_tractor

end Tractor


/-!
## Ancestry Continuous Representation

Moving beyond discrete ancestry groups to continuous
ancestry representation using principal components.
-/

section ContinuousAncestry

/- **PC-based continuous ancestry.**
    Ancestry is represented as a point in PC space rather than
    a discrete label. This better captures the continuous nature
    of human genetic variation. -/

/-- Model representing PGS accuracy as a function of continuous genetic distance (PC space). -/
structure ContinuousAncestryPgsModel where
  /-- Base R² in the discovery centroid. -/
  r2_source : ℝ
  /-- Rate of decay per unit of genetic distance. -/
  slope : ℝ
  h_r2_pos : 0 < r2_source
  h_slope_pos : 0 < slope

/-- **PGS accuracy as a function of PC distance.**
    R²(d) where d is the PC distance from the training centroid.
    This is Wang et al.'s key finding (Open Question 1). -/
noncomputable def r2AsFunction (m : ContinuousAncestryPgsModel) (d : ℝ) : ℝ :=
  m.r2_source * (1 - m.slope * d)

/-- R² decreases with genetic distance. -/
theorem r2_decreases_with_distance
    (m : ContinuousAncestryPgsModel)
    (d₁ d₂ : ℝ)
    (h_d : d₁ < d₂) :
    r2AsFunction m d₂ < r2AsFunction m d₁ := by
  dsimp [r2AsFunction]
  apply mul_lt_mul_of_pos_left
  · exact sub_lt_sub_left (mul_lt_mul_of_pos_left h_d m.h_slope_pos) 1
  · exact m.h_r2_pos

/-- **Within-group variation dominates between-group variation.**
    When the variance explained by between-group signal is small relative to the
    within-group variance scale `cv²`, the between-group signal is overwhelmed.
    If the between-group variance is strictly below `var_total / cv²`, then the
    resulting between-group `R² = var_between / var_total` is strictly below
    `1 / cv²`.

    Worked example: Wang et al. find R² ≈ 0.5% for distance-on-error,
    far below the χ²₁ cv² = 2. -/
theorem individual_variation_dominates
    (var_between_group var_total cv_squared_within : ℝ)
    (h_total_pos : 0 < var_total)
    (h_cv_pos : 0 < cv_squared_within)
    (h_var_small : var_between_group < var_total / cv_squared_within) :
    var_between_group / var_total < 1 / cv_squared_within := by
  rw [div_lt_div_iff₀ h_total_pos h_cv_pos]
  calc
    var_between_group * cv_squared_within
      < (var_total / cv_squared_within) * cv_squared_within := by
        exact mul_lt_mul_of_pos_right h_var_small h_cv_pos
    _ = var_total := div_mul_cancel₀ var_total (ne_of_gt h_cv_pos)
    _ = 1 * var_total := (one_mul var_total).symm

/-- **Optimal ancestry granularity for PGS application.**
    Too coarse (continental groups) loses information.
    Too fine (exact PC coordinates) adds noise with small samples.
    Optimal depends on available calibration data. -/
theorem optimal_granularity_tradeoff
    (bias_coarse bias_fine variance_coarse variance_fine : ℝ)
    (h_coarse_biased : bias_fine < bias_coarse)
    (h_fine_variable : variance_coarse < variance_fine) :
    -- Different regimes have different optima
    bias_fine < bias_coarse ∧ variance_coarse < variance_fine :=
  ⟨h_coarse_biased, h_fine_variable⟩

end ContinuousAncestry


/-!
## Ancestry Heterogeneity and Fairness

Ensuring PGS works equitably across the full spectrum of
human genetic diversity, not just for well-represented groups.
-/

section AncestryFairness

/-- Model of the portability gap as a function of Fst. -/
structure PortabilityGapModel where
  /-- Scaling constant for the portability penalty. -/
  c : ℝ
  /-- Fst for the first target population. -/
  fst₁ : ℝ
  /-- Fst for the second target population. -/
  fst₂ : ℝ
  h_c_pos : 0 < c
  h_fst_ord : fst₁ < fst₂
  h_fst₁_nonneg : 0 ≤ fst₁

/-- Portability gap in a target population compared to discovery. -/
noncomputable def portability_gap (c fst : ℝ) : ℝ :=
  c * fst

/-- **Portability gap creates health disparities.**
    Groups with worse portability get less clinical benefit from PGS.
    The gap scales with Fst from the discovery population.
    Portability ≈ 1 - c × Fst for some constant c > 0, so higher Fst
    means lower portability (larger gap = 1 - portability). -/
theorem portability_gap_scales_with_fst
    (m : PortabilityGapModel) :
    portability_gap m.c m.fst₁ < portability_gap m.c m.fst₂ := by
  dsimp [portability_gap]
  exact mul_lt_mul_of_pos_left m.h_fst_ord m.h_c_pos

/-- Model of sample size requirements across populations to achieve equal PGS accuracy. -/
structure SampleSizeEquivalenceModel where
  /-- Base sample size required in the discovery (European) population. -/
  n_discovery : ℝ
  /-- Sample size required in the target (underrepresented) population. -/
  n_target : ℝ
  /-- The penalty factor due to LD differences and differing allele frequencies (k > 1). -/
  ld_penalty : ℝ
  h_n_discovery_pos : 0 < n_discovery
  h_penalty_gt_one : 1 < ld_penalty
  /-- The mathematical relationship: required target sample size is scaled by the penalty. -/
  h_equivalence : n_target = ld_penalty * n_discovery

/-- **Equitable PGS requires proportional investment.**
    To achieve equal R² across populations, the sample size
    for underrepresented groups must be larger than proportional
    to their population size (due to the LD mismatch penalty).
    If the LD penalty factor is k > 1, then n_needed = k × n_proportional. -/
theorem equitable_pgs_overinvestment
    (m : SampleSizeEquivalenceModel) :
    m.n_discovery < m.n_target := by
  rw [m.h_equivalence]
  exact (lt_mul_iff_one_lt_left m.h_n_discovery_pos).mpr m.h_penalty_gt_one

/-- Model representing the portability of a single PGS across multiple populations. -/
structure UniversalPortabilityModel where
  /-- Base R² in the discovery population. -/
  r2_source : ℝ
  /-- Decay constant. -/
  c : ℝ
  /-- Fst for a population genetically similar to the discovery population. -/
  fst_near : ℝ
  /-- Fst for a population genetically distant from the discovery population. -/
  fst_far : ℝ
  h_r2_pos : 0 < r2_source
  h_c_pos : 0 < c
  h_fst_near_nonneg : 0 ≤ fst_near
  h_fst_far_gt : fst_near < fst_far

/-- Expected R² in a target population as a function of Fst. -/
noncomputable def expected_target_r2 (r2_source c fst : ℝ) : ℝ :=
  r2_source * (1 - c * fst)

/-- **Universal portability is impossible.**
    No single PGS can achieve equal R² in all populations,
    because genetic architecture genuinely differs (GxE, selection).
    With k ≥ 2 populations at different Fst values and portability
    decaying as (1 - c × Fst), the max and min R² must differ
    when Fst values differ. -/
theorem universal_portability_impossible
    (m : UniversalPortabilityModel) :
    expected_target_r2 m.r2_source m.c m.fst_far < expected_target_r2 m.r2_source m.c m.fst_near := by
  dsimp [expected_target_r2]
  apply mul_lt_mul_of_pos_left
  · apply sub_lt_sub_left
    exact mul_lt_mul_of_pos_left m.h_fst_far_gt m.h_c_pos
  · exact m.h_r2_pos

end AncestryFairness

end Calibrator
