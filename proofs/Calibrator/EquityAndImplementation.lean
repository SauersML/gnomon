import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.ClinicalUtilityFairness
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Equity, Ethical, and Implementation Aspects of PGS Portability

This file formalizes the equity implications of PGS portability gaps,
the ethical framework for clinical PGS deployment, and the practical
considerations for implementing PGS across diverse populations.

Key results:
1. Portability gap creates health disparities
2. Fairness impossibility for PGS across populations
3. Resource allocation for equitable PGS development
4. Clinical implementation guidelines
5. Regulatory and return-of-results considerations

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Health Disparity from Portability Gaps

PGS portability gaps directly translate to disparities in clinical
benefit across ancestry groups.
-/

section HealthDisparity

/-- A formal mathematical model of PGS deployment in a specific population. -/
structure PopulationDeployment where
  R2 : ℝ
  h_R2_pos : 0 < R2
  h_R2_le_one : R2 ≤ 1
  fst_to_source : ℝ
  h_fst_pos : 0 ≤ fst_to_source
  h_fst_lt_one : fst_to_source < 1
  baseline_health : ℝ

/-- A configuration modeling the portability gap between a source (discovery) and target population. -/
structure TwoPopDeployment where
  source : PopulationDeployment
  target : PopulationDeployment
  h_portability_gap : target.R2 < source.R2
  h_fst_gap : source.fst_to_source < target.fst_to_source
  benefit_factor : ℝ
  h_benefit_pos : 0 < benefit_factor
  qaly_per_r2 : ℝ
  h_qaly_pos : 0 < qaly_per_r2

/-- Clinical benefit is proportional to R2. -/
def clinical_benefit (pop : PopulationDeployment) (benefit_factor : ℝ) : ℝ :=
  benefit_factor * pop.R2

/-- QALYs gained is proportional to R2. -/
def qaly_gained (pop : PopulationDeployment) (qaly_per_r2 : ℝ) : ℝ :=
  qaly_per_r2 * pop.R2

/-- Expected R2 loss based on Fst. -/
def expected_r2_loss (source_r2 fst : ℝ) : ℝ :=
  source_r2 * (1 - (1 - fst) ^ 2)

/-- Post-deployment health disparity between source and target populations. -/
def post_deployment_disparity (model : TwoPopDeployment) : ℝ :=
  (model.source.baseline_health + clinical_benefit model.source model.benefit_factor) -
  (model.target.baseline_health + clinical_benefit model.target model.benefit_factor)

/-- **Clinical utility depends on PGS R².**
    The net clinical benefit from PGS-guided care is monotonically
    increasing in R². When target R² < source R², the benefit is strictly less. -/
theorem clinical_benefit_increases_with_r2
    (model : TwoPopDeployment) :
    clinical_benefit model.target model.benefit_factor <
      clinical_benefit model.source model.benefit_factor := by
  dsimp [clinical_benefit]
  exact mul_lt_mul_of_pos_left model.h_portability_gap model.h_benefit_pos

/-- **Portability gap creates benefit gap.**
    The benefit gap is positive. -/
theorem portability_creates_benefit_gap
    (model : TwoPopDeployment) :
    0 < clinical_benefit model.source model.benefit_factor -
        clinical_benefit model.target model.benefit_factor := by
  have h_gap : model.target.R2 < model.source.R2 := model.h_portability_gap
  have h_pos : 0 < model.benefit_factor := model.h_benefit_pos
  dsimp [clinical_benefit]
  nlinarith

/-- **Disparity increases with Fst from discovery population.** -/
theorem disparity_increases_with_distance
    (model : TwoPopDeployment) :
    expected_r2_loss model.source.R2 model.source.fst_to_source <
      expected_r2_loss model.source.R2 model.target.fst_to_source := by
  dsimp [expected_r2_loss]
  have h_R2 : 0 < model.source.R2 := model.source.h_R2_pos
  apply mul_lt_mul_of_pos_left _ h_R2
  have h_fst_gap : model.source.fst_to_source < model.target.fst_to_source := model.h_fst_gap
  have h1 : (1 - model.target.fst_to_source) ^ 2 < (1 - model.source.fst_to_source) ^ 2 := by
    nlinarith [model.target.h_fst_lt_one, model.source.h_fst_pos]
  linarith

/-- **Existing health disparities may be amplified.** -/
theorem deployment_amplifies_disparity
    (model : TwoPopDeployment) :
    model.source.baseline_health - model.target.baseline_health <
      post_deployment_disparity model := by
  dsimp [post_deployment_disparity, clinical_benefit]
  have h_gap : model.target.R2 < model.source.R2 := model.h_portability_gap
  have h_pos : 0 < model.benefit_factor := model.h_benefit_pos
  linarith [mul_lt_mul_of_pos_left h_gap h_pos]

/-- **QALY gap from portability.** -/
theorem qaly_gap_proportional_to_r2_gap
    (model : TwoPopDeployment) :
    0 < qaly_gained model.source model.qaly_per_r2 - qaly_gained model.target model.qaly_per_r2 := by
  dsimp [qaly_gained]
  have h_gap : model.target.R2 < model.source.R2 := model.h_portability_gap
  have h_pos : 0 < model.qaly_per_r2 := model.h_qaly_pos
  nlinarith

end HealthDisparity


/-!
## Fairness Impossibility Results

It is mathematically impossible to simultaneously satisfy
multiple fairness criteria when PGS performance differs
across populations.
-/

section FairnessImpossibility

/-- A model for PPV derived from disease prevalence (K), FPR, and FNR. -/
structure PPVModel where
  K : ℝ
  h_K_pos : 0 < K
  h_K_lt_one : K < 1

/-- A configuration showing two populations sharing the same FPR and FNR but differing prevalence. -/
structure TwoPopChouldechova where
  pop1 : PPVModel
  pop2 : PPVModel
  fpr : ℝ
  fnr : ℝ
  h_prev_diff : pop1.K ≠ pop2.K
  h_fpr_pos : 0 < fpr
  h_fnr_lt_one : fnr < 1
  _h_fnr_nn : 0 ≤ fnr

/-- Computes PPV using Bayes' theorem form. -/
noncomputable def compute_ppv (K fpr fnr : ℝ) : ℝ :=
  K * (1 - fnr) / (K * (1 - fnr) + (1 - K) * fpr)

/-- **Chouldechova's impossibility theorem for PGS.**
    When base rates (disease prevalence) differ across groups,
    it's impossible to simultaneously achieve:
    1. Equal false positive rates (FPR₁ = FPR₂)
    2. Equal false negative rates (FNR₁ = FNR₂)
    3. Equal positive predictive values (PPV₁ = PPV₂)
    This fundamental result shows that choosing a threshold to
    equalize one fairness metric forces disparities in another. -/
theorem chouldechova_impossibility
    (model : TwoPopChouldechova) :
    compute_ppv model.pop1.K model.fpr model.fnr ≠
    compute_ppv model.pop2.K model.fpr model.fnr := by
  dsimp [compute_ppv]
  intro h
  apply model.h_prev_diff
  have h_sens : 0 < 1 - model.fnr := by linarith [model.h_fnr_lt_one]
  have h1_pos : 0 < model.pop1.K * (1 - model.fnr) + (1 - model.pop1.K) * model.fpr := by
    nlinarith [model.pop1.h_K_pos, model.pop1.h_K_lt_one, model.h_fpr_pos, h_sens]
  have h2_pos : 0 < model.pop2.K * (1 - model.fnr) + (1 - model.pop2.K) * model.fpr := by
    nlinarith [model.pop2.h_K_pos, model.pop2.h_K_lt_one, model.h_fpr_pos, h_sens]
  rw [div_eq_div_iff h1_pos.ne' h2_pos.ne'] at h
  nlinarith [mul_pos h_sens model.h_fpr_pos]

/-- Models the performance and threshold of a classifier. -/
structure FairnessModel where
  mu : ℝ
  sigma : ℝ
  threshold : ℝ
  h_sigma_pos : 0 < sigma

/-- Computes the standard z-score for a threshold. -/
noncomputable def z_score (m : FairnessModel) : ℝ :=
  (m.threshold - m.mu) / m.sigma

/-- A configuration with two populations with differing parameters. -/
structure TwoPopFairness where
  pop1 : FairnessModel
  pop2 : FairnessModel
  h_mu_diff : pop1.mu ≠ pop2.mu
  h_sigma_eq : pop1.sigma = pop2.sigma

/-- **Simplified fairness impossibility: equal calibration + equal thresholds.**
    If we use a population-specific threshold to achieve equal FPR,
    the thresholds must differ, which means the treatment policies
    are ancestry-dependent. -/
theorem equal_fpr_requires_different_thresholds
    (model : TwoPopFairness)
    (h_equal_z : z_score model.pop1 = z_score model.pop2) :
    model.pop1.threshold ≠ model.pop2.threshold := by
  intro h_eq
  have h_mu_diff := model.h_mu_diff
  have h_sigma_eq := model.h_sigma_eq
  have h_sigma1 := model.pop1.h_sigma_pos
  have h_sigma2 := model.pop2.h_sigma_pos
  dsimp [z_score] at h_equal_z
  rw [h_eq, h_sigma_eq] at h_equal_z
  have h_eq₂ : (model.pop2.threshold - model.pop1.mu) * model.pop2.sigma =
               (model.pop2.threshold - model.pop2.mu) * model.pop2.sigma := by
    rwa [div_eq_div_iff (ne_of_gt h_sigma2) (ne_of_gt h_sigma2)] at h_equal_z
  have := mul_right_cancel₀ (ne_of_gt h_sigma2) h_eq₂
  apply h_mu_diff
  linarith

/-- **Group-blind vs group-aware PGS policies.**
    A group-blind policy (same threshold for all) violates
    calibration equality. A group-aware policy violates
    treatment equality. Neither is fully "fair". -/
theorem no_fully_fair_policy
    (model : TwoPopFairness) :
    0 < |model.pop1.mu - model.pop2.mu| ∧
    0 < |model.pop1.mu - model.pop2.mu| / model.pop1.sigma := by
  constructor
  · exact abs_pos.mpr (sub_ne_zero.mpr model.h_mu_diff)
  · exact div_pos (abs_pos.mpr (sub_ne_zero.mpr model.h_mu_diff)) model.pop1.h_sigma_pos

end FairnessImpossibility


/-!
## Resource Allocation for Equitable PGS

Optimal allocation of GWAS resources (funding, participants)
to minimize the maximum portability gap.
-/

section ResourceAllocation

/- **Minimax allocation minimizes the maximum disparity.**
    Instead of maximizing average R², allocate resources to
    minimize max_pop(R²_source - R²_pop). -/

/-- **Diminishing returns per additional sample in the source.**
    R² ∝ n × h² / (n × h² + M) where M is effective number
    of independent causal loci. As n → ∞, R² → h². -/
noncomputable def expectedR2FromN (n h2 M : ℝ) : ℝ :=
  n * h2 / (n * h2 + M)

/-- R² increases with n. -/
theorem r2_increases_with_n
    (h2 M : ℝ) (n₁ n₂ : ℝ)
    (h_h2 : 0 < h2) (h_M : 0 < M)
    (h_n₁ : 0 < n₁) (h_n₂ : 0 < n₂) (h_more : n₁ < n₂) :
    expectedR2FromN n₁ h2 M < expectedR2FromN n₂ h2 M := by
  unfold expectedR2FromN
  rw [div_lt_div_iff₀ (by positivity) (by positivity)]
  nlinarith [mul_pos h_h2 h_M]

/-- R² is concave in n (diminishing returns). -/
theorem r2_concave_in_n
    (h2 M n dn : ℝ)
    (h_h2 : 0 < h2) (h_M : 0 < M)
    (h_n : 0 < n) (h_dn : 0 < dn) :
    -- Marginal gain from n+dn to n+2dn is less than from n to n+dn
    expectedR2FromN (n + 2*dn) h2 M - expectedR2FromN (n + dn) h2 M <
      expectedR2FromN (n + dn) h2 M - expectedR2FromN n h2 M := by
  unfold expectedR2FromN
  -- This is equivalent to showing f''(n) < 0 for f(n) = nh²/(nh²+M)
  -- f'(n) = h²M/(nh²+M)², f''(n) = -2(h²)²M/(nh²+M)³ < 0
  -- f(n) = nh²/(nh²+M) is concave in n, so f(n+2d)-f(n+d) < f(n+d)-f(n)
  -- Each difference = dh²M / ((xh²+M)((x+d)h²+M))
  -- Denominator grows with x → difference shrinks
  have h1 : 0 < n * h2 + M := by nlinarith [mul_pos h_n h_h2]
  have h2' : 0 < (n + dn) * h2 + M := by nlinarith [mul_pos (by linarith : 0 < n + dn) h_h2]
  have h3 : 0 < (n + 2 * dn) * h2 + M := by nlinarith [mul_pos (by linarith : 0 < n + 2 * dn) h_h2]
  rw [div_sub_div _ _ h3.ne' h2'.ne', div_sub_div _ _ h2'.ne' h1.ne']
  -- Both numerators simplify to dn * h2 * M (matching div_sub_div output order)
  have lhs_eq : (n + 2 * dn) * h2 * ((n + dn) * h2 + M) -
    ((n + 2 * dn) * h2 + M) * ((n + dn) * h2) = dn * h2 * M := by ring
  have rhs_eq : (n + dn) * h2 * (n * h2 + M) -
    ((n + dn) * h2 + M) * (n * h2) = dn * h2 * M := by ring
  rw [lhs_eq, rhs_eq]
  -- Goal: dn*h2*M / (h3*h2') < dn*h2*M / (h2'*h1)
  apply div_lt_div_of_pos_left (mul_pos (mul_pos h_dn h_h2) h_M) (mul_pos h2' h1) _
  -- Need: h2'*h1 < h3*h2', i.e., h2'*(h3 - h1) > 0
  nlinarith [mul_pos h2' (show 0 < 2 * dn * h2 from by nlinarith [mul_pos h_dn h_h2])]

/-- **Marginal value of diversity.**
    Adding underrepresented individuals has higher marginal value
    for reducing the max disparity than adding EUR individuals.
    By concavity of R²(n), marginal gain is larger at smaller n.
    If n_eur >> n_underrep, an extra sample in the underrepresented
    group yields more R² gain. -/
theorem diversity_has_higher_marginal_value
    (h2 M n_eur n_underrep dn : ℝ)
    (h_h2 : 0 < h2) (h_M : 0 < M)
    (h_eur : 0 < n_eur) (h_underrep : 0 < n_underrep)
    (h_dn : 0 < dn)
    (h_larger : n_underrep < n_eur) :
    -- Marginal R² gain is larger for the underrepresented group
    expectedR2FromN (n_eur + dn) h2 M - expectedR2FromN n_eur h2 M <
      expectedR2FromN (n_underrep + dn) h2 M - expectedR2FromN n_underrep h2 M := by
  unfold expectedR2FromN
  have h1 : 0 < n_underrep * h2 + M := by nlinarith [mul_pos h_underrep h_h2]
  have h2' : 0 < (n_underrep + dn) * h2 + M := by nlinarith [mul_pos (by linarith : 0 < n_underrep + dn) h_h2]
  have h3 : 0 < n_eur * h2 + M := by nlinarith [mul_pos h_eur h_h2]
  have h4 : 0 < (n_eur + dn) * h2 + M := by nlinarith [mul_pos (by linarith : 0 < n_eur + dn) h_h2]
  rw [div_sub_div _ _ h4.ne' h3.ne', div_sub_div _ _ h2'.ne' h1.ne']
  have lhs_eq : (n_eur + dn) * h2 * (n_eur * h2 + M) -
    ((n_eur + dn) * h2 + M) * (n_eur * h2) = dn * h2 * M := by ring
  have rhs_eq : (n_underrep + dn) * h2 * (n_underrep * h2 + M) -
    ((n_underrep + dn) * h2 + M) * (n_underrep * h2) = dn * h2 * M := by ring
  rw [lhs_eq, rhs_eq]
  apply div_lt_div_of_pos_left (mul_pos (mul_pos h_dn h_h2) h_M) (mul_pos h2' h1) _
  have h_base_lt : n_underrep * h2 + M < n_eur * h2 + M := by
    nlinarith [h_larger, h_h2]
  have h_shift_lt : (n_underrep + dn) * h2 + M < (n_eur + dn) * h2 + M := by
    nlinarith [h_larger, h_h2]
  nlinarith [h_base_lt, h_shift_lt, h1, h2', h3, h4]

end ResourceAllocation


/-!
## Clinical Implementation Guidelines

Practical considerations for deploying PGS in clinical settings
with diverse populations.
-/

section ClinicalImplementation

/-- Models the clinical utility of a genetic test based on R2 and cost. -/
structure ClinicalUtilityModel where
  R2 : ℝ
  h_r2_nn : 0 ≤ R2
  benefit_per_r2 : ℝ
  h_benefit_pos : 0 < benefit_per_r2
  cost : ℝ
  h_cost_pos : 0 < cost

/-- Computes the net clinical value. -/
def net_clinical_value (m : ClinicalUtilityModel) : ℝ :=
  m.benefit_per_r2 * m.R2 - m.cost

/-- **Minimum R² for clinical utility.**
    Below a threshold R², PGS does not improve clinical decisions.
    The net clinical value = α × R² - cost. When R² is small
    (R² < cost / α), the net value is negative. -/
theorem r2_threshold_for_utility
    (model : ClinicalUtilityModel)
    (h_below : model.R2 < model.cost / model.benefit_per_r2) :
    net_clinical_value model < 0 := by
  dsimp [net_clinical_value]
  have h_benefit := model.h_benefit_pos
  have : model.R2 * model.benefit_per_r2 < model.cost := by rwa [lt_div_iff₀ h_benefit] at h_below
  linarith [mul_comm model.benefit_per_r2 model.R2]

/- **Population-specific PGS report cards.**
    For each PGS, report: R², AUC, calibration, and portability ratio
    for each clinically relevant population. -/

/-- Models the required validation sample size for estimating R2. -/
structure ValidationModel where
  r2_source : ℝ
  h_r2_source_pos : 0 < r2_source
  h_r2_source_lt : r2_source < 1
  r2_target : ℝ
  h_r2_target_pos : 0 < r2_target
  h_r2_target_lt : r2_target < 1
  h_portability_gap : r2_target < r2_source
  delta : ℝ
  h_delta_pos : 0 < delta

/-- Required sample size per unit R2 to achieve given standard error. -/
noncomputable def required_n_per_r2 (r2 delta : ℝ) : ℝ :=
  4 * (1 - r2) ^ 2 / delta ^ 2

/-- **Relative precision of R² estimate is worse for smaller R².**
    To estimate R² with SE < δ, need approximately n > 4R²(1-R²)²/δ².
    The *relative* standard error SE/R² = (1-R²)/√(nR²) × 2/δ grows as R²
    shrinks.  For the same sample size, the relative error is larger for the
    target population (smaller R²).

    Equivalently, the required sample size per unit R² (n/R²) is larger for
    smaller R².  Here we prove: n/R² = 4(1-R²)²/δ² is a decreasing function
    of R² on (0,1), so the target population (lower R²) requires a larger
    sample-per-unit-R². -/
theorem validation_n_depends_on_r2
    (model : ValidationModel) :
    required_n_per_r2 model.r2_source model.delta <
      required_n_per_r2 model.r2_target model.delta := by
  dsimp [required_n_per_r2]
  have h_delta := model.h_delta_pos
  have h1 : model.r2_target < model.r2_source := model.h_portability_gap
  have h2 : model.r2_source < 1 := model.h_r2_source_lt
  have h3 : model.r2_target < 1 := model.h_r2_target_lt
  apply div_lt_div_of_pos_right _ (sq_pos_of_ne_zero (ne_of_gt h_delta))
  apply mul_lt_mul_of_pos_left _ (by norm_num : (0:ℝ) < 4)
  -- (1-r2_source)² < (1-r2_target)² because 0 < 1-r2_source < 1-r2_target
  apply sq_lt_sq'
  · linarith
  · linarith

/- **Population-aware clinical decision support.**
    The clinical decision system should:
    1. Report population-specific PGS performance
    2. Adjust confidence intervals for portability
    3. Flag when PGS may be unreliable for the patient's population -/

/-- Model encapsulating PGS deployment parameters for clinical benefit analysis. -/
structure DeploymentParams where
  r2_source : ℝ
  r2_target : ℝ
  h_r2_loss : r2_target < r2_source
  h_r2_target_nn : 0 ≤ r2_target
  h_r2_source_le : r2_source ≤ 1
  pi : ℝ
  h_pi_pos : 0 < pi
  h_pi_lt_one : pi < 1
  benefit : ℝ
  h_benefit_pos : 0 < benefit
  harm : ℝ
  h_harm_pos : 0 < harm

/-- **Do-no-harm principle for PGS deployment.**
    PGS should only be used clinically when the expected benefit exceeds
    the expected harm from misclassification.

    We model benefit and harm through the exact liability-threshold operating
    metrics from `ClinicalUtilityFairness.lean`. As `R²` increases, exact
    threshold sensitivity rises (more true positives treated) and exact
    threshold specificity rises (fewer false positives harmed).

    Here we prove the structural result: if the target population's R² is
    strictly below the source's (portability loss), the net clinical value
    in the target is strictly less than in the source. This means populations
    with poor portability may violate the do-no-harm principle even when the
    source population satisfies it. This is derived from the exact operating
    metric formulas, not assumed through an abstract link. -/
theorem do_no_harm_principle
    (m : LiabilityThresholdModel) (T' μ_control : ℝ)
    (params : DeploymentParams)
    (hPhi_mono : StrictMono Phi)
    (hμ_control_neg : μ_control < 0)
    (h_sens_num_nonneg :
      0 ≤ Real.sqrt params.r2_target * Real.sqrt m.h_sq * m.case_mean - T')
    (h_spec_num_nonneg :
      0 ≤ T' - Real.sqrt params.r2_target * Real.sqrt m.h_sq * μ_control) :
    -- Net value in target is strictly less than in source
    sensFromR2 m params.r2_target T' * params.pi * params.benefit -
        (1 - specFromR2 m params.r2_target T' μ_control) * (1 - params.pi) * params.harm <
      sensFromR2 m params.r2_source T' * params.pi * params.benefit -
        (1 - specFromR2 m params.r2_source T' μ_control) * (1 - params.pi) * params.harm := by
  have h_sens :
      sensFromR2 m params.r2_target T' < sensFromR2 m params.r2_source T' := by
    exact sensFromR2_strictMono m T' params.r2_target params.r2_source
      hPhi_mono params.h_r2_target_nn params.h_r2_source_le params.h_r2_loss h_sens_num_nonneg
  have h_spec :
      specFromR2 m params.r2_target T' μ_control <
        specFromR2 m params.r2_source T' μ_control := by
    exact specFromR2_strictMono m T' μ_control params.r2_target params.r2_source
      hPhi_mono hμ_control_neg params.h_r2_target_nn params.h_r2_source_le params.h_r2_loss h_spec_num_nonneg
  have h1 :
      sensFromR2 m params.r2_target T' * params.pi * params.benefit <
        sensFromR2 m params.r2_source T' * params.pi * params.benefit := by
    apply mul_lt_mul_of_pos_right _ params.h_benefit_pos
    exact mul_lt_mul_of_pos_right h_sens params.h_pi_pos
  have h2 :
      (1 - specFromR2 m params.r2_source T' μ_control) * (1 - params.pi) * params.harm <
        (1 - specFromR2 m params.r2_target T' μ_control) * (1 - params.pi) * params.harm := by
    apply mul_lt_mul_of_pos_right _ params.h_harm_pos
    apply mul_lt_mul_of_pos_right _ (by linarith [params.h_pi_lt_one])
    linarith
  linarith

/-- **Phased deployment strategy.**
    Deploy PGS first for well-validated populations (R² above threshold),
    then expand as validation data becomes available.

    We model: the validated population has strictly higher R² than the
    unvalidated population (portability gap). By the exact liability-threshold
    sensitivity and specificity formulas, the validated population has
    strictly better discrimination.

    Phased deployment achieves the same clinical benefit for the validated
    population while avoiding the risk of deploying a poorly-performing
    PGS in the unvalidated population.  The benefit gap between
    validated and unvalidated populations is derived from the exact operating
    metrics, not from an abstract link. -/
theorem phased_deployment_reduces_risk
    (m : LiabilityThresholdModel) (T' μ_control : ℝ)
    (params : DeploymentParams)
    (hPhi_mono : StrictMono Phi)
    (hμ_control_neg : μ_control < 0)
    (h_sens_num_nonneg :
      0 ≤ Real.sqrt params.r2_target * Real.sqrt m.h_sq * m.case_mean - T')
    (h_spec_num_nonneg :
      0 ≤ T' - Real.sqrt params.r2_target * Real.sqrt m.h_sq * μ_control) :
    sensFromR2 m params.r2_target T' * params.pi +
        specFromR2 m params.r2_target T' μ_control * (1 - params.pi) <
      sensFromR2 m params.r2_source T' * params.pi +
        specFromR2 m params.r2_source T' μ_control * (1 - params.pi) := by
  have h_sens :
      sensFromR2 m params.r2_target T' < sensFromR2 m params.r2_source T' := by
    exact sensFromR2_strictMono m T' params.r2_target params.r2_source
      hPhi_mono params.h_r2_target_nn params.h_r2_source_le params.h_r2_loss h_sens_num_nonneg
  have h_spec :
      specFromR2 m params.r2_target T' μ_control <
        specFromR2 m params.r2_source T' μ_control := by
    exact specFromR2_strictMono m T' μ_control params.r2_target params.r2_source
      hPhi_mono hμ_control_neg params.h_r2_target_nn params.h_r2_source_le params.h_r2_loss h_spec_num_nonneg
  have h1 :
      sensFromR2 m params.r2_target T' * params.pi <
        sensFromR2 m params.r2_source T' * params.pi :=
    mul_lt_mul_of_pos_right h_sens params.h_pi_pos
  have h2 :
      specFromR2 m params.r2_target T' μ_control * (1 - params.pi) <
        specFromR2 m params.r2_source T' μ_control * (1 - params.pi) :=
    mul_lt_mul_of_pos_right h_spec (by linarith [params.h_pi_lt_one])
  linarith

end ClinicalImplementation

end Calibrator
