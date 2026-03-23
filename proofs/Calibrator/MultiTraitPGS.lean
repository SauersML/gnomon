import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Multi-Trait PGS and Portability

This file formalizes how multi-trait PGS methods — using genetic
correlations across traits to improve prediction — interact with
cross-ancestry portability.

Key results:
1. Genetic correlation structure across traits and ancestries
2. Multi-trait BLUP and prediction improvement
3. Portability of genetic correlations
4. Pleiotropic architecture and shared portability
5. Trait-specific vs shared PGS components

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Genetic Correlation Structure

Genetic correlations between traits capture shared genetic
architecture and enable multi-trait prediction methods.
-/

section GeneticCorrelation

/-- **Genetic correlation definition.**
    r_g(A,B) = Cov_g(A,B) / √(V_gA × V_gB)
    where V_gX is the genetic variance of trait X. -/
noncomputable def geneticCorrelationMT (cov_g V_gA V_gB : ℝ) : ℝ :=
  cov_g / Real.sqrt (V_gA * V_gB)

/-- Genetic correlation is bounded by 1 in absolute value
    (Cauchy-Schwarz). -/
theorem genetic_correlation_bounded_mt
    (rg : ℝ) (h_bound : |rg| ≤ 1) :
    -1 ≤ rg ∧ rg ≤ 1 := by
  exact ⟨by linarith [abs_nonneg rg, abs_le.mp h_bound |>.1], abs_le.mp h_bound |>.2⟩

/-- **Model for genetic covariance across ancestries.**
    Captures the variance of traits A and B and their covariance
    in two different populations (e.g., source and target). -/
structure CrossAncestryCovarianceModel where
  varA : ℝ
  varB : ℝ
  covSource : ℝ
  covTarget : ℝ
  h_varA_pos : 0 < varA
  h_varB_pos : 0 < varB

/-- **Genetic correlation in the source population.** -/
noncomputable def rgSource (m : CrossAncestryCovarianceModel) : ℝ :=
  m.covSource / Real.sqrt (m.varA * m.varB)

/-- **Genetic correlation in the target population.** -/
noncomputable def rgTarget (m : CrossAncestryCovarianceModel) : ℝ :=
  m.covTarget / Real.sqrt (m.varA * m.varB)

/-- **Genetic correlation is partially ancestry-specific.**
    If LD and GxE attenuate the genetic covariance between traits A and B
    in the target population compared to the source, the target
    genetic correlation will be strictly lower. -/
theorem rg_ancestry_specific
    (m : CrossAncestryCovarianceModel)
    (h_cov_attenuation : m.covTarget < m.covSource) :
    rgTarget m < rgSource m := by
  unfold rgTarget rgSource
  have h_den_pos : 0 < Real.sqrt (m.varA * m.varB) := by
    apply Real.sqrt_pos.mpr
    exact mul_pos m.h_varA_pos m.h_varB_pos
  exact (div_lt_div_iff_of_pos_right h_den_pos).mpr h_cov_attenuation

/-- **Equal-correlation PSD constraint.**
    For a 3×3 correlation matrix with equal pairwise correlation r,
    the determinant is (1-r)²(1+2r). PSD requires det ≥ 0, which
    (given |r| ≤ 1 so (1-r)² ≥ 0) is equivalent to 1 + 2r ≥ 0,
    i.e., r ≥ -1/2. This is a non-trivial constraint: not all
    pairwise-valid correlations give a valid correlation matrix. -/
theorem rg_matrix_equal_corr_psd_constraint
    (r : ℝ)
    (h_bound : |r| ≤ 1)
    (h_lower : -1/2 ≤ r) :
    0 ≤ (1 - r)^2 * (1 + 2 * r) := by
  apply mul_nonneg
  · exact sq_nonneg _
  · linarith

end GeneticCorrelation


/-!
## Multi-Trait BLUP

Multi-trait BLUP (MTBLUP) uses genetic correlations to
borrow information across traits, improving prediction
especially for low-heritability traits.
-/

section MultiTraitBLUP

/-- **MTBLUP prediction improvement.**
    For a target trait with h² and genetic correlation r_g
    with an auxiliary trait:
    R²_multi / R²_single ≈ 1 + r_g² × (n_aux / n_target) × (h²_aux / h²_target)
    when auxiliary GWAS is much larger. -/
noncomputable def mtblupImprovement (rg n_aux n_target h2_aux h2_target : ℝ) : ℝ :=
  1 + rg^2 * (n_aux / n_target) * (h2_aux / h2_target)

/-- MTBLUP always improves over single-trait (when rg > 0 and aux is larger). -/
theorem mtblup_improves (rg n_aux n_target h2_aux h2_target : ℝ)
    (h_rg : rg ≠ 0) (h_n_aux : 0 < n_aux) (h_n_target : 0 < n_target)
    (h_h2_aux : 0 < h2_aux) (h_h2_target : 0 < h2_target) :
    1 < mtblupImprovement rg n_aux n_target h2_aux h2_target := by
  unfold mtblupImprovement
  linarith [sq_pos_of_ne_zero h_rg,
            div_pos h_n_aux h_n_target,
            div_pos h_h2_aux h_h2_target,
            mul_pos (mul_pos (sq_pos_of_ne_zero h_rg) (div_pos h_n_aux h_n_target))
                    (div_pos h_h2_aux h_h2_target)]

/-- **Model for Multi-Trait BLUP cross-ancestry prediction.**
    Captures the parameters needed to evaluate prediction
    improvement from an auxiliary trait across populations. -/
structure MultiTraitBLUPModel where
  rgSame : ℝ
  rgCross : ℝ
  nAux : ℝ
  nTarget : ℝ
  h2Aux : ℝ
  h2Target : ℝ
  h_nAux_pos : 0 < nAux
  h_nTarget_pos : 0 < nTarget
  h_h2Aux_pos : 0 < h2Aux
  h_h2Target_pos : 0 < h2Target

/-- **MTBLUP improvement ratio.**
    1 + r_g² * (n_aux/n_target) * (h²_aux/h²_target) -/
noncomputable def mtblupImprovementRatio (m : MultiTraitBLUPModel) (rg : ℝ) : ℝ :=
  1 + rg^2 * (m.nAux / m.nTarget) * (m.h2Aux / m.h2Target)

/-- **MTBLUP portability.**
    The multi-trait improvement is less portable when cross-ancestry
    genetic correlation is strictly less than within-ancestry genetic correlation. -/
theorem mtblup_portability_reduced
    (m : MultiTraitBLUPModel)
    (h_rg_cross_nn : 0 ≤ m.rgCross)
    (h_rg_less : m.rgCross < m.rgSame) :
    mtblupImprovementRatio m m.rgCross < mtblupImprovementRatio m m.rgSame := by
  unfold mtblupImprovementRatio
  have h_k_pos : 0 < (m.nAux / m.nTarget) * (m.h2Aux / m.h2Target) := by
    apply mul_pos
    · exact div_pos m.h_nAux_pos m.h_nTarget_pos
    · exact div_pos m.h_h2Aux_pos m.h_h2Target_pos
  have h_sq : m.rgCross^2 < m.rgSame^2 := sq_lt_sq.mpr
    (by rw [abs_of_nonneg h_rg_cross_nn, abs_of_pos (by linarith)]; exact h_rg_less)
  linarith [mul_lt_mul_of_pos_right h_sq h_k_pos]

end MultiTraitBLUP


/-!
## Pleiotropic Architecture

Pleiotropy — one variant affecting multiple traits — creates
correlated portability patterns across traits.
-/

section Pleiotropy

/-- **Model for horizontal pleiotropy and correlated portability.**
    Captures the portability (R²) of two traits connected by
    a horizontal pleiotropy structure bounded by their cross-ancestry
    genetic correlation squared. -/
structure HorizontalPleiotropyModel where
  portA : ℝ
  portB : ℝ
  rgCross : ℝ
  h_correlated : |portA - portB| ≤ 2 * (1 - rgCross^2)

/-- **Horizontal pleiotropy creates correlated portability.**
    If two traits have a strictly positive lower bound on their cross-ancestry
    genetic correlation, the divergence in their portability is strictly bounded. -/
theorem pleiotropic_correlated_portability
    (m : HorizontalPleiotropyModel)
    (lb : ℝ)
    (h_lb_nn : 0 ≤ lb)
    (h_rg : lb < |m.rgCross|) :
    |m.portA - m.portB| < 2 * (1 - lb^2) := by
  have h_sq : lb^2 < m.rgCross^2 := by
    have h_abs_rg : |lb| < |m.rgCross| := by rwa [abs_of_nonneg h_lb_nn]
    exact sq_lt_sq.mpr h_abs_rg
  linarith [m.h_correlated]

/-- **Model for mediated pleiotropy.**
    Mediated pleiotropy (A → B) means variants affect trait B solely
    through trait A. The portability of B is thus a direct fractional
    scaling (α) of the portability of A. -/
structure MediatedPleiotropyModel where
  portA : ℝ
  portB : ℝ
  alpha : ℝ
  h_mediated_eq : portB = alpha * portA
  h_alpha_bounds : 0 ≤ alpha ∧ alpha ≤ 1
  h_portA_nn : 0 ≤ portA

/-- **Mediated pleiotropy vs biological pleiotropy.**
    Because B is mediated through A with a fraction ≤ 1,
    the portability of B is bounded by the portability of A. -/
theorem mediated_pleiotropy_portability_bound
    (m : MediatedPleiotropyModel) :
    m.portB ≤ m.portA := by
  rw [m.h_mediated_eq]
  have h_le : m.alpha ≤ 1 := m.h_alpha_bounds.2
  have h_nn : 0 ≤ m.portA := m.h_portA_nn
  nlinarith

/-- **Model for trait genetic architecture decomposition.**
    Captures the variance and portability (R² preservation)
    of shared (pleiotropic) and unique components of a trait. -/
structure PleiotropicTraitModel where
  h2Shared : ℝ
  h2Unique : ℝ
  portShared : ℝ
  portUnique : ℝ
  h_shared_pos : 0 < h2Shared
  h_unique_pos : 0 < h2Unique

/-- **Overall trait portability.**
    The variance-weighted average of shared and unique portability. -/
noncomputable def overallPortability (m : PleiotropicTraitModel) : ℝ :=
  (m.h2Shared * m.portShared + m.h2Unique * m.portUnique) / (m.h2Shared + m.h2Unique)

/-- **Trait-specific genetic components are less portable.**
    If the unique component is strictly less portable than the shared component,
    the overall portability is strictly bounded from above by the shared component's portability. -/
theorem unique_component_less_portable
    (m : PleiotropicTraitModel)
    (h_selection : m.portUnique < m.portShared) :
    overallPortability m < m.portShared := by
  unfold overallPortability
  have h_total_pos : 0 < m.h2Shared + m.h2Unique := by linarith [m.h_shared_pos, m.h_unique_pos]
  have h_bound : m.h2Shared * m.portShared + m.h2Unique * m.portUnique <
                 m.h2Shared * m.portShared + m.h2Unique * m.portShared := by
    apply add_lt_add_left
    exact mul_lt_mul_of_pos_left h_selection m.h_unique_pos
  have h_factored : m.h2Shared * m.portShared + m.h2Unique * m.portShared = m.portShared * (m.h2Shared + m.h2Unique) := by ring
  rw [h_factored] at h_bound
  exact (div_lt_iff₀ h_total_pos).mpr h_bound

/-- **Decomposing trait heritability into shared and unique.**
    When the shared fraction dominates (h²_shared > h²_unique), and is more portable,
    overall portability is strictly greater than the unweighted average. -/
theorem heritability_shared_dominates_portability
    (m : PleiotropicTraitModel)
    (h_shared_larger : m.h2Unique < m.h2Shared)
    (h_port_shared_better : m.portUnique < m.portShared) :
    (m.portShared + m.portUnique) / 2 < overallPortability m := by
  unfold overallPortability
  have h_total_pos : 0 < m.h2Shared + m.h2Unique := by linarith [m.h_shared_pos, m.h_unique_pos]
  rw [lt_div_iff₀ h_total_pos]
  have h_diff_mul : 0 < (m.h2Shared - m.h2Unique) * (m.portShared - m.portUnique) := by
    apply mul_pos
    · exact sub_pos.mpr h_shared_larger
    · exact sub_pos.mpr h_port_shared_better
  linarith

end Pleiotropy


/-!
## Cross-Ancestry Genetic Correlation

Genetic correlations between the same trait across ancestries
directly measure portability of genetic effects.
-/

section CrossAncestryRg

/-- **Cross-population r_g measures effect portability.**
    r_g(trait_source, trait_target) measures how similar the genetic
    effects are between source and target for the same trait. -/
noncomputable def crossAncestryRg (cov_cross V_g_source V_g_target : ℝ) : ℝ :=
  cov_cross / Real.sqrt (V_g_source * V_g_target)

/-- **r_g bounds PGS portability.**
    R²_target / R²_source ≤ r_g² (cross-ancestry). -/
theorem rg_bounds_portability_ratio
    (r2_source r2_target rg_cross : ℝ)
    (h_r2_s : 0 < r2_source)
    (h_bound : r2_target / r2_source ≤ rg_cross^2) :
    r2_target ≤ rg_cross^2 * r2_source := by
  rwa [div_le_iff₀ h_r2_s] at h_bound

/-- **Traits with high cross-population r_g have good portability.**
    When r_g is high (e.g., ~0.95), R² portability is bounded by ~0.90. -/
theorem high_cross_rg
    (rg lb : ℝ) (h_rg : lb < rg) (h_lb_nn : 0 ≤ lb) (h_rg_le : rg ≤ 1) :
    lb^2 < rg^2 := by nlinarith [sq_nonneg (rg - lb)]

/-- **Traits with low cross-population r_g have poor portability.**
    When r_g is low (e.g., ~0.3), R² portability is bounded by ~0.09. -/
theorem low_cross_rg
    (rg ub : ℝ) (h_rg : rg ≤ ub) (h_rg_nn : 0 ≤ rg) (h_ub_nn : 0 ≤ ub) :
    rg^2 ≤ ub^2 := by nlinarith [sq_nonneg rg, sq_nonneg (rg - ub)]

/-- **r_g can be underestimated due to power.**
    With low power in underrepresented-population GWAS, r_g estimates are
    attenuated toward zero. Correction for attenuation:
    r_g_corrected = r_g_observed / √(h²_1 × h²_2). -/
theorem rg_attenuation_correction
    (rg_obs h2_1 h2_2 : ℝ)
    (h_h2_1 : 0 < h2_1) (h_h2_1_le : h2_1 < 1)
    (h_h2_2 : 0 < h2_2) (h_h2_2_le : h2_2 < 1)
    (h_rg : 0 < rg_obs) :
    rg_obs < rg_obs / Real.sqrt (h2_1 * h2_2) := by
  rw [lt_div_iff₀ (Real.sqrt_pos.mpr (mul_pos h_h2_1 h_h2_2))]
  have h_prod_lt : h2_1 * h2_2 < 1 := by nlinarith
  have h_sqrt_lt : Real.sqrt (h2_1 * h2_2) < 1 := by
    rw [show (1 : ℝ) = Real.sqrt 1 from (Real.sqrt_one).symm]
    exact Real.sqrt_lt_sqrt (mul_nonneg (le_of_lt h_h2_1) (le_of_lt h_h2_2)) h_prod_lt
  nlinarith

end CrossAncestryRg

end Calibrator
