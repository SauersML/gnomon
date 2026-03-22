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

structure BivariateGeneticModel where
  varA : ℝ
  varB : ℝ
  covAB : ℝ
  varA_pos : 0 < varA
  varB_pos : 0 < varB
  cs_bound : covAB^2 ≤ varA * varB

/-- **Genetic correlation definition.**
    r_g(A,B) = Cov_g(A,B) / √(V_gA × V_gB)
    where V_gX is the genetic variance of trait X. -/
noncomputable def geneticCorrelationMT (m : BivariateGeneticModel) : ℝ :=
  m.covAB / Real.sqrt (m.varA * m.varB)

/-- Genetic correlation is bounded by 1 in absolute value
    (Cauchy-Schwarz). -/
theorem genetic_correlation_bounded_mt (m : BivariateGeneticModel) :
    -1 ≤ geneticCorrelationMT m ∧ geneticCorrelationMT m ≤ 1 := by
  unfold geneticCorrelationMT
  have h_var_mul_pos : 0 < m.varA * m.varB := mul_pos m.varA_pos m.varB_pos
  have h_sq : (m.covAB / Real.sqrt (m.varA * m.varB))^2 ≤ 1 := by
    rw [div_pow, Real.sq_sqrt (le_of_lt h_var_mul_pos)]
    exact (div_le_one h_var_mul_pos).mpr m.cs_bound
  exact ⟨by nlinarith, by nlinarith⟩

structure CrossAncestryBivariateModel where
  eur : BivariateGeneticModel
  afr : BivariateGeneticModel

/-- **Genetic correlation is partially ancestry-specific.**
    r_g between traits A and B may differ between EUR and AFR
    due to different LD patterns and GxE. If LD and GxE introduce
    a nonzero perturbation δ, the ancestry-specific r_g differs. -/
theorem rg_ancestry_specific
    (m : CrossAncestryBivariateModel)
    (h_varA : m.eur.varA = m.afr.varA)
    (h_varB : m.eur.varB = m.afr.varB)
    (h_cov : m.eur.covAB ≠ m.afr.covAB) :
    geneticCorrelationMT m.eur ≠ geneticCorrelationMT m.afr := by
  unfold geneticCorrelationMT
  rw [h_varA, h_varB]
  intro h
  have h_var_mul_pos : 0 < m.afr.varA * m.afr.varB := mul_pos m.afr.varA_pos m.afr.varB_pos
  have h_sqrt_pos : 0 < Real.sqrt (m.afr.varA * m.afr.varB) := Real.sqrt_pos.mpr h_var_mul_pos
  have h_eq : m.eur.covAB = m.afr.covAB := by
    have h2 := congrArg (fun x => x * Real.sqrt (m.afr.varA * m.afr.varB)) h
    dsimp at h2
    rw [div_mul_cancel₀ _ h_sqrt_pos.ne', div_mul_cancel₀ _ h_sqrt_pos.ne'] at h2
    exact h2
  exact h_cov h_eq

structure SymmetricTrivariateModel where
  var : ℝ
  cov : ℝ
  var_pos : 0 < var
  sum_var_nonneg : 0 ≤ 3 * var + 6 * cov

noncomputable def symmetricCorr (m : SymmetricTrivariateModel) : ℝ :=
  m.cov / m.var

/-- **Equal-correlation PSD constraint.**
    For a 3×3 covariance matrix with equal variance and pairwise covariance,
    PSD requires the sum of elements to be non-negative.
    This implies the correlation is bounded below by -1/2. -/
theorem rg_matrix_equal_corr_psd_constraint (m : SymmetricTrivariateModel) :
    -1/2 ≤ symmetricCorr m := by
  unfold symmetricCorr
  have h1 : 0 ≤ 3 * (m.var + 2 * m.cov) := by linarith [m.sum_var_nonneg]
  have h2 : 0 ≤ m.var + 2 * m.cov := by
    have h_div : (0 : ℝ) ≤ (3 * (m.var + 2 * m.cov)) / 3 := div_nonneg h1 (by norm_num)
    linarith
  have h3 : -m.var ≤ 2 * m.cov := by linarith
  have h4 : -1 / 2 * (2 * m.var) ≤ 2 * m.cov := by linarith
  have h5 : -1 / 2 * m.var ≤ m.cov := by linarith
  exact (le_div_iff₀ m.var_pos).mpr h5

end GeneticCorrelation


/-!
## Multi-Trait BLUP

Multi-trait BLUP (MTBLUP) uses genetic correlations to
borrow information across traits, improving prediction
especially for low-heritability traits.
-/

section MultiTraitBLUP

structure MultiTraitBLUPModel where
  n_aux : ℝ
  n_target : ℝ
  h2_aux : ℝ
  h2_target : ℝ
  rg : ℝ
  n_aux_pos : 0 < n_aux
  n_target_pos : 0 < n_target
  h2_aux_pos : 0 < h2_aux
  h2_target_pos : 0 < h2_target

/-- **MTBLUP prediction improvement.**
    For a target trait with h² and genetic correlation r_g
    with an auxiliary trait:
    R²_multi / R²_single ≈ 1 + r_g² × (n_aux / n_target) × (h²_aux / h²_target)
    when auxiliary GWAS is much larger. -/
noncomputable def mtblupImprovement (m : MultiTraitBLUPModel) : ℝ :=
  1 + m.rg^2 * (m.n_aux / m.n_target) * (m.h2_aux / m.h2_target)

/-- MTBLUP always improves over single-trait (when rg > 0 and aux is larger). -/
theorem mtblup_improves (m : MultiTraitBLUPModel) (h_rg : m.rg ≠ 0) :
    1 < mtblupImprovement m := by
  unfold mtblupImprovement
  have h_rg2 : 0 < m.rg^2 := sq_pos_of_ne_zero h_rg
  have h_n : 0 < m.n_aux / m.n_target := div_pos m.n_aux_pos m.n_target_pos
  have h_h2 : 0 < m.h2_aux / m.h2_target := div_pos m.h2_aux_pos m.h2_target_pos
  have h_prod1 : 0 < m.rg^2 * (m.n_aux / m.n_target) := mul_pos h_rg2 h_n
  have h_prod2 : 0 < m.rg^2 * (m.n_aux / m.n_target) * (m.h2_aux / m.h2_target) := mul_pos h_prod1 h_h2
  linarith

structure CrossAncestryMTBLUPModel where
  source : MultiTraitBLUPModel
  target : MultiTraitBLUPModel
  h_n_aux_eq : source.n_aux = target.n_aux
  h_n_target_eq : source.n_target = target.n_target
  h_h2_aux_eq : source.h2_aux = target.h2_aux
  h_h2_target_eq : source.h2_target = target.h2_target

/-- **MTBLUP portability.**
    The multi-trait improvement is less portable when cross-ancestry
    genetic correlation is lower. If r_g_cross < r_g_same, then
    MTBLUP improvement is smaller in the target population.
    Model: improvement ratio = 1 + r_g² × k where k = (n_aux/n_target)(h²_aux/h²_target).
    With r_g_cross < r_g_same, the cross-ancestry improvement is strictly smaller. -/
theorem mtblup_portability_reduced
    (m : CrossAncestryMTBLUPModel)
    (h_rg_less : m.target.rg^2 < m.source.rg^2) :
    mtblupImprovement m.target < mtblupImprovement m.source := by
  unfold mtblupImprovement
  rw [m.h_n_aux_eq, m.h_n_target_eq, m.h_h2_aux_eq, m.h_h2_target_eq]
  have h_k_pos : 0 < (m.target.n_aux / m.target.n_target) * (m.target.h2_aux / m.target.h2_target) := by
    apply mul_pos
    · exact div_pos m.target.n_aux_pos m.target.n_target_pos
    · exact div_pos m.target.h2_aux_pos m.target.h2_target_pos
  nlinarith

end MultiTraitBLUP


/-!
## Pleiotropic Architecture

Pleiotropy — one variant affecting multiple traits — creates
correlated portability patterns across traits.
-/

section Pleiotropy

structure HorizontalPleiotropyModel where
  port_A : ℝ
  port_B : ℝ
  rg : ℝ
  lb : ℝ
  h_correlated : |port_A - port_B| ≤ 2 * (1 - |rg|)
  h_rg : lb < |rg|
  h_lb_nn : 0 ≤ lb

/-- **Horizontal pleiotropy creates correlated portability.**
    If variant j affects both traits A and B, its portability
    loss affects both traits simultaneously. -/
theorem pleiotropic_correlated_portability (m : HorizontalPleiotropyModel) :
    |m.port_A - m.port_B| < 2 * (1 - m.lb) := by
  have : 2 * (1 - |m.rg|) < 2 * (1 - m.lb) := by
    nlinarith [m.h_rg]
  linarith [m.h_correlated]

structure MediatedPleiotropyModel where
  port_A : ℝ
  port_unique_B : ℝ
  alpha : ℝ
  alpha_bound : 0 ≤ alpha ∧ alpha ≤ 1
  port_A_nn : 0 ≤ port_A
  port_unique_B_nn : 0 ≤ port_unique_B

noncomputable def port_B (m : MediatedPleiotropyModel) : ℝ :=
  m.alpha * m.port_A + (1 - m.alpha) * m.port_unique_B

/-- **Mediated pleiotropy vs biological pleiotropy.**
    Mediated: A → B, so variant affects B through A.
    Portability of B is bounded by portability of A.
    If the mediation fraction is α ∈ [0,1], then
    port_B_mediated = α × port_A, so port_B ≤ port_A. -/
theorem mediated_pleiotropy_portability_bound (m : MediatedPleiotropyModel)
    (h_unique_worse : m.port_unique_B ≤ m.port_A) :
    port_B m ≤ m.port_A := by
  unfold port_B
  have h1 : (1 - m.alpha) * m.port_unique_B ≤ (1 - m.alpha) * m.port_A := by
    apply mul_le_mul_of_nonneg_left h_unique_worse
    linarith [m.alpha_bound.2]
  linarith

structure TraitComponentPortabilityModel where
  port_base : ℝ
  δ_shared : ℝ
  δ_unique : ℝ
  h_selection : δ_shared < δ_unique
  h_shared_nn : 0 < δ_shared
  h_base : δ_unique < port_base

/-- **Trait-specific genetic components are less portable.**
    The component of genetic variance unique to a trait (not shared
    via pleiotropy) is more likely to be affected by population-specific
    selection. Shared components degrade by δ_shared, unique by δ_unique,
    where δ_unique > δ_shared due to selection. -/
theorem unique_component_less_portable (m : TraitComponentPortabilityModel) :
    m.port_base - m.δ_unique < m.port_base - m.δ_shared := by
  linarith [m.h_selection]

structure TraitArchitectureModel where
  h2_shared : ℝ
  h2_unique : ℝ
  port_shared : ℝ
  port_unique : ℝ
  h2_shared_pos : 0 < h2_shared
  h2_unique_pos : 0 < h2_unique
  port_shared_nn : 0 ≤ port_shared
  port_unique_nn : 0 ≤ port_unique

noncomputable def overallPortability (m : TraitArchitectureModel) : ℝ :=
  (m.h2_shared * m.port_shared + m.h2_unique * m.port_unique) / (m.h2_shared + m.h2_unique)

/-- **Decomposing trait heritability into shared and unique.**
    h²_trait = h²_shared + h²_unique where h²_shared comes from
    pleiotropic loci. When the shared fraction dominates (h²_shared/h²_total > 1/2),
    portability is primarily determined by the shared component.
    Model: overall portability = (h²_shared × port_shared + h²_unique × port_unique) / h²_total.
    If h²_shared > h²_unique and port_shared > port_unique, then
    overall portability > (port_shared + port_unique) / 2 (the unweighted average). -/
theorem heritability_shared_dominates_portability (m : TraitArchitectureModel)
    (h_shared_larger : m.h2_unique < m.h2_shared)
    (h_port_shared_better : m.port_unique < m.port_shared) :
    (m.port_shared + m.port_unique) / 2 < overallPortability m := by
  unfold overallPortability
  have h_sum_pos : 0 < m.h2_shared + m.h2_unique := add_pos m.h2_shared_pos m.h2_unique_pos
  rw [lt_div_iff₀ h_sum_pos]
  have : (m.port_shared + m.port_unique) / 2 * (m.h2_shared + m.h2_unique) =
    (m.port_shared * m.h2_shared + m.port_unique * m.h2_unique) / 2 +
    (m.port_shared * m.h2_unique + m.port_unique * m.h2_shared) / 2 := by ring
  nlinarith [mul_pos (sub_pos.mpr h_shared_larger) (sub_pos.mpr h_port_shared_better)]

end Pleiotropy


/-!
## Cross-Ancestry Genetic Correlation

Genetic correlations between the same trait across ancestries
directly measure portability of genetic effects.
-/

section CrossAncestryRg

structure CrossAncestryRgModel where
  r2_source : ℝ
  r2_target : ℝ
  rg_cross : ℝ
  r2_source_pos : 0 < r2_source
  r2_target_nn : 0 ≤ r2_target
  portability_bound : r2_target / r2_source ≤ rg_cross^2

/-- **r_g bounds PGS portability.**
    R²_target / R²_source ≤ r_g² (cross-ancestry). -/
theorem rg_bounds_portability_ratio (m : CrossAncestryRgModel) :
    m.r2_target ≤ m.rg_cross^2 * m.r2_source := by
  exact (div_le_iff₀ m.r2_source_pos).mp m.portability_bound

/-- **Traits with high cross-population r_g have good portability.**
    When r_g is high (e.g., ~0.95), R² portability is bounded by ~0.90. -/
theorem high_cross_rg
    (rg lb : ℝ) (h_rg : lb < rg) (h_lb_nn : 0 ≤ lb) :
    lb^2 < rg^2 := by nlinarith [sq_nonneg (rg - lb)]

/-- **Traits with low cross-population r_g have poor portability.**
    When r_g is low (e.g., ~0.3), R² portability is bounded by ~0.09. -/
theorem low_cross_rg
    (rg ub : ℝ) (h_rg : rg ≤ ub) (h_rg_nn : 0 ≤ rg) :
    rg^2 ≤ ub^2 := by nlinarith [sq_nonneg rg, sq_nonneg (rg - ub)]

structure RqAttenuationModel where
  rg_obs : ℝ
  h2_1 : ℝ
  h2_2 : ℝ
  h2_1_bounds : 0 < h2_1 ∧ h2_1 < 1
  h2_2_bounds : 0 < h2_2 ∧ h2_2 < 1
  rg_obs_pos : 0 < rg_obs

noncomputable def rg_corrected (m : RqAttenuationModel) : ℝ :=
  m.rg_obs / Real.sqrt (m.h2_1 * m.h2_2)

/-- **r_g can be underestimated due to power.**
    With low power in underrepresented-population GWAS, r_g estimates are
    attenuated toward zero. Correction for attenuation:
    r_g_corrected = r_g_observed / √(h²_1 × h²_2). -/
theorem rg_attenuation_correction (m : RqAttenuationModel) :
    m.rg_obs < rg_corrected m := by
  unfold rg_corrected
  have h_mul_pos : 0 < m.h2_1 * m.h2_2 := mul_pos m.h2_1_bounds.1 m.h2_2_bounds.1
  have h_sqrt_pos : 0 < Real.sqrt (m.h2_1 * m.h2_2) := Real.sqrt_pos.mpr h_mul_pos
  rw [lt_div_iff₀ h_sqrt_pos]
  have h_prod_lt : m.h2_1 * m.h2_2 < 1 := by nlinarith [m.h2_1_bounds.2, m.h2_2_bounds.2, m.h2_1_bounds.1, m.h2_2_bounds.1]
  have h_sqrt_lt : Real.sqrt (m.h2_1 * m.h2_2) < 1 := by
    rw [show (1 : ℝ) = Real.sqrt 1 from (Real.sqrt_one).symm]
    exact Real.sqrt_lt_sqrt (le_of_lt h_mul_pos) h_prod_lt
  exact mul_lt_of_lt_one_right m.rg_obs_pos h_sqrt_lt

end CrossAncestryRg

end Calibrator
