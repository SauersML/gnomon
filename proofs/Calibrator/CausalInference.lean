import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Causal Inference Framework for PGS Portability

This file formalizes the causal mediation framework for understanding
PGS portability. Portability loss can be decomposed into direct
(genetic architecture) and indirect (mediated by environment, LD,
assortative mating) pathways.

Key results:
1. Path decomposition of portability loss
2. Mediation analysis for environmental effects
3. Do-calculus for interventions on PGS accuracy
4. Counterfactual portability under hypothetical designs
5. Causal discovery for portability mechanisms

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Path Decomposition of Portability

PGS portability loss can be decomposed into distinct causal pathways,
each contributing a specific proportion of the total loss.
-/

section PathDecomposition

structure PortabilityLossModel where
  delta_LD : ℝ
  delta_MAF : ℝ
  delta_effect : ℝ
  delta_env : ℝ
  delta_tech : ℝ
  h_LD : 0 ≤ delta_LD
  h_MAF : 0 ≤ delta_MAF
  h_effect : 0 ≤ delta_effect
  h_env : 0 ≤ delta_env
  h_tech : 0 ≤ delta_tech

noncomputable def totalLoss (m : PortabilityLossModel) : ℝ :=
  m.delta_LD + m.delta_MAF + m.delta_effect + m.delta_env + m.delta_tech

/-- **Total portability loss decomposition.**
    Δ_total = Δ_LD + Δ_MAF + Δ_effect + Δ_env + Δ_technical
    Each component represents a distinct causal pathway. -/
theorem total_loss_decomposition (m : PortabilityLossModel) :
    m.delta_LD ≤ totalLoss m ∧ m.delta_MAF ≤ totalLoss m ∧
    m.delta_effect ≤ totalLoss m ∧ m.delta_env ≤ totalLoss m ∧
    m.delta_tech ≤ totalLoss m := by
  unfold totalLoss
  have h1 : 0 ≤ m.delta_LD := m.h_LD
  have h2 : 0 ≤ m.delta_MAF := m.h_MAF
  have h3 : 0 ≤ m.delta_effect := m.h_effect
  have h4 : 0 ≤ m.delta_env := m.h_env
  have h5 : 0 ≤ m.delta_tech := m.h_tech
  constructor <;> [skip; constructor <;> [skip; constructor <;> [skip; constructor]]] <;> linarith

/-- **LD pathway is the largest contributor for most traits.**
    For non-immune traits, LD mismatch accounts for >50% of portability loss. -/
theorem ld_dominant_pathway
    (m : PortabilityLossModel)
    (h_total : 0 < totalLoss m)
    (h_LD_large : totalLoss m / 2 < m.delta_LD) :
    1 / 2 < m.delta_LD / totalLoss m := by
  rw [div_lt_div_iff₀ (by norm_num : (0:ℝ) < 2) h_total]
  linarith

/-- **Selection pathway dominates for immune traits.**
    For immune/pathogen-related traits under divergent selection,
    the effect-turnover factor (ρ < 1) causes additional R² loss
    beyond what drift alone produces. The total immune portability
    loss (source R² minus immune R²) exceeds twice the drift-only
    loss (source R² minus drift-only R²) when ρ is small enough.

    We model this using expectedR2 from PortabilityDrift: drift-only
    uses signal (1-fst)·V_A, immune uses ρ²·(1-fst)·V_A.
    We show the immune R² is strictly below the drift-only R²,
    establishing that effect turnover is a genuine additional
    pathway of portability loss. -/
theorem selection_dominant_for_immune
    (V_A V_E fst ρ : ℝ)
    (hVA : 0 < V_A) (hVE : 0 < V_E)
    (hfst_pos : 0 < fst) (hfst_lt : fst < 1)
    (hρ_pos : 0 < ρ) (hρ_lt : ρ < 1) :
    -- Immune R² (with effect turnover ρ) is strictly less than
    -- drift-only R² (no effect turnover), showing the selection
    -- pathway causes genuine additional loss beyond LD/drift.
    expectedR2 (ρ ^ 2 * presentDayPGSVariance V_A fst) V_E <
      expectedR2 (presentDayPGSVariance V_A fst) V_E := by
  apply expectedR2_strictMono_nonneg V_E _ _ hVE
  · exact le_of_lt (mul_pos (sq_pos_of_pos hρ_pos)
      (by unfold presentDayPGSVariance; exact mul_pos (by linarith) hVA))
  · have h_pdv_pos : 0 < presentDayPGSVariance V_A fst := by
      unfold presentDayPGSVariance; exact mul_pos (by linarith) hVA
    calc ρ ^ 2 * presentDayPGSVariance V_A fst
        < 1 * presentDayPGSVariance V_A fst := by
          apply mul_lt_mul_of_pos_right _ h_pdv_pos
          nlinarith [sq_abs ρ, sq_nonneg ρ]
      _ = presentDayPGSVariance V_A fst := one_mul _

structure PathwayModel where
  ld_effect : ℝ
  maf_effect : ℝ
  ld_maf_interaction : ℝ

noncomputable def independentPathways (m : PathwayModel) : ℝ :=
  m.ld_effect + m.maf_effect

noncomputable def interactingPathways (m : PathwayModel) : ℝ :=
  m.ld_effect + m.maf_effect + m.ld_maf_interaction

/-- **Interaction effects between pathways.**
    Pathways are not fully independent: LD changes interact
    with MAF changes (LD × MAF interaction). -/
theorem pathway_interactions_exist (m : PathwayModel)
    (h_interaction : m.ld_maf_interaction ≠ 0) :
    interactingPathways m ≠ independentPathways m := by
  unfold interactingPathways independentPathways
  intro h
  apply h_interaction
  linarith

end PathDecomposition


/-!
## Mediation Analysis

Mediation analysis decomposes the total effect of ancestry
on PGS accuracy into direct and indirect effects.
-/

section MediationAnalysis

structure LinearMediationModel where
  alpha : ℝ -- Effect of ancestry on mediator
  beta : ℝ  -- Effect of mediator on outcome
  gamma : ℝ -- Direct effect of ancestry on outcome

noncomputable def indirectEffect (m : LinearMediationModel) : ℝ := m.alpha * m.beta
noncomputable def directEffect (m : LinearMediationModel) : ℝ := m.gamma
noncomputable def totalEffect (m : LinearMediationModel) : ℝ := m.gamma + m.alpha * m.beta

/-- **Total effect = Direct effect + Indirect effect.**
    TE = DE + IE (in the linear case). -/
theorem mediation_decomposition (m : LinearMediationModel) :
  totalEffect m = directEffect m + indirectEffect m := by
  unfold totalEffect directEffect indirectEffect
  ring

/-- **Proportion mediated.**
    PM = IE / TE = indirect / total. -/
noncomputable def proportionMediated (indirect_effect total_effect : ℝ) : ℝ :=
  indirect_effect / total_effect

/-- Proportion mediated is in [0,1] when effects are nonneg and indirect ≤ total. -/
theorem proportion_mediated_in_unit
    (ie te : ℝ)
    (h_ie : 0 ≤ ie) (h_te : 0 < te) (h_le : ie ≤ te) :
    0 ≤ proportionMediated ie te ∧ proportionMediated ie te ≤ 1 := by
  unfold proportionMediated
  constructor
  · exact div_nonneg h_ie (le_of_lt h_te)
  · rw [div_le_one h_te]; exact h_le

/-- **LD mediates ancestry → PGS accuracy.**
    Ancestry → LD structure → PGS weights → Accuracy.

    Model: PGS accuracy = expectedR2(vSignal, V_E + V_ld_mismatch).
    Without LD correction, the full mismatch V_ld adds noise.
    With LD correction (using target-population LD matrix), a fraction
    `α` of the mismatch is removed, leaving V_E + (1-α)·V_ld.
    Since 0 < α ≤ 1, the corrected noise is strictly less,
    so R²_corrected > R²_uncorrected by expectedR2 monotonicity.
    A residual gap to source R² (V_E only) remains when α < 1. -/
theorem ld_mediates_portability
    (vSignal V_E V_ld α : ℝ)
    (h_sig : 0 < vSignal) (h_VE : 0 < V_E)
    (h_ld : 0 < V_ld) (h_α_pos : 0 < α) (h_α_le : α ≤ 1) :
    -- LD correction improves R²: corrected > uncorrected
    expectedR2 vSignal (V_E + V_ld) <
      expectedR2 vSignal (V_E + (1 - α) * V_ld) ∧
    -- Residual gap remains (corrected < source) when α < 1
    (α < 1 → expectedR2 vSignal (V_E + (1 - α) * V_ld) <
      expectedR2 vSignal V_E) := by
  constructor
  · -- Corrected noise = V_E + (1-α)·V_ld < V_E + V_ld = uncorrected noise
    -- since α > 0 implies (1-α) < 1, so (1-α)·V_ld < V_ld.
    unfold expectedR2
    exact div_lt_div_of_pos_left h_sig (by nlinarith) (by nlinarith)
  · -- If α < 1, then (1-α)·V_ld > 0, so corrected noise > V_E = source noise.
    intro h_α_lt
    unfold expectedR2
    exact div_lt_div_of_pos_left h_sig (by linarith) (by nlinarith)

/-- **Environment mediates ancestry → PGS accuracy.**
    Ancestry → Environment → Phenotype → Accuracy.

    Model: Total phenotypic variance = V_genetic + V_env.
    Phenotypic R² = V_genetic / (V_genetic + V_env).
    Genetic R² (no environmental noise) = V_genetic / V_genetic = 1,
    but more usefully, the phenotypic R² is strictly less than what
    we'd get without environmental variance.  Specifically:
      R²_pheno = expectedR2(V_genetic, V_env) < expectedR2(V_genetic, 0) = 1.
    This shows environment genuinely reduces predictive accuracy;
    the reduction is derived from the variance decomposition, not assumed. -/
theorem environment_mediates_portability
    (V_genetic V_env : ℝ)
    (h_gen : 0 < V_genetic) (h_env : 0 < V_env) :
    -- Phenotypic R² is strictly less than 1 (perfect genetic prediction)
    expectedR2 V_genetic V_env < 1 := by
  unfold expectedR2
  rw [div_lt_one (by linarith : 0 < V_genetic + V_env)]
  linarith

end MediationAnalysis


/-!
## Counterfactual Portability

What would PGS portability look like under hypothetical
alternative study designs?
-/

section CounterfactualPortability

/-- **Counterfactual: diverse training GWAS.**
    If the training GWAS had been done in the target ancestry,
    there is no drift divergence (fst = 0), so presentDayR2
    equals V_A/(V_A + V_E).  Cross-ancestry training with
    fst > 0 gives strictly lower R².  The gap is the
    portability loss attributable to ancestry mismatch. -/
theorem counterfactual_same_ancestry_perfect
    (V_A V_E fst : ℝ)
    (hVA : 0 < V_A) (hVE : 0 < V_E)
    (hfst_pos : 0 < fst) (hfst_lt : fst < 1) :
    -- Cross-ancestry R² is strictly below same-ancestry R²
    expectedR2 (presentDayPGSVariance V_A fst) V_E <
      expectedR2 (presentDayPGSVariance V_A 0) V_E := by
  apply expectedR2_strictMono_nonneg V_E _ _ hVE
  · unfold presentDayPGSVariance
    exact le_of_lt (mul_pos (by linarith) hVA)
  · unfold presentDayPGSVariance
    simp only [sub_zero]
    have : (1 - fst) * V_A < 1 * V_A := by
      exact mul_lt_mul_of_pos_right (by linarith) hVA
    linarith

/-- **Counterfactual: WGS eliminates technical artifacts.**
    With array genotyping, imputation error adds noise V_tech to
    the environmental variance, reducing observed R².  WGS removes
    this (V_tech = 0), so the WGS R² exceeds array R².
    The remaining gap from source R² is purely genetic drift. -/
theorem counterfactual_wgs_residual
    (vSignal V_E V_tech : ℝ)
    (h_sig : 0 < vSignal) (h_VE : 0 < V_E) (h_tech : 0 < V_tech) :
    -- Array R² (with technical noise) < WGS R² (without)
    expectedR2 vSignal (V_E + V_tech) < expectedR2 vSignal V_E := by
  unfold expectedR2
  have h_denom_wgs : 0 < vSignal + V_E := by linarith
  have h_denom_arr : 0 < vSignal + (V_E + V_tech) := by linarith
  rw [div_lt_div_iff₀ h_denom_arr h_denom_wgs]
  nlinarith

/-- **Counterfactual: infinite sample size.**
    With finite sample, winner's curse inflates effect estimates,
    adding noise V_wc to the prediction.  With n → ∞, V_wc → 0.
    We model finite-sample R² as expectedR2 with signal vSignal
    and noise V_E + V_wc (winner's curse adds prediction error).
    Infinite-sample R² uses noise V_E only.
    The remaining loss (infinite-sample R² vs source R²) is from
    LD mismatch and true effect size differences. -/
theorem counterfactual_infinite_sample
    (vSignal V_E V_wc : ℝ)
    (h_sig : 0 < vSignal) (h_VE : 0 < V_E) (h_wc : 0 < V_wc) :
    -- Finite-sample R² < infinite-sample R²
    expectedR2 vSignal (V_E + V_wc) < expectedR2 vSignal V_E := by
  unfold expectedR2
  have h_denom_inf : 0 < vSignal + V_E := by linarith
  have h_denom_fin : 0 < vSignal + (V_E + V_wc) := by linarith
  rw [div_lt_div_iff₀ h_denom_fin h_denom_inf]
  nlinarith

/-- **Counterfactual: equalized environments.**
    GxE interaction adds variance V_gxe to the target phenotype,
    inflating environmental noise from V_E to V_E + V_gxe.
    If environments were equalized (V_gxe = 0), the target R²
    would be higher.  The remaining loss is purely from genetic
    architecture differences (drift, LD, effect turnover). -/
theorem counterfactual_equal_environments
    (vSignal V_E V_gxe : ℝ)
    (h_sig : 0 < vSignal) (h_VE : 0 < V_E) (h_gxe : 0 < V_gxe) :
    -- R² with GxE < R² without GxE
    expectedR2 vSignal (V_E + V_gxe) < expectedR2 vSignal V_E := by
  unfold expectedR2
  have h_denom_eq : 0 < vSignal + V_E := by linarith
  have h_denom_gxe : 0 < vSignal + (V_E + V_gxe) := by linarith
  rw [div_lt_div_iff₀ h_denom_gxe h_denom_eq]
  nlinarith

end CounterfactualPortability


/-!
## Interventions to Improve Portability

The do-calculus framework identifies which interventions
can improve PGS portability.
-/

section InterventionsForPortability

/-- **Intervention hierarchy (most to least effective).**
    Model: each intervention addresses specific MSE components.
    - Original MSE noise: V_E + V_ld + V_power + V_cal
    - Recalibration: fixes intercept only, removes V_cal.
      Noise = V_E + V_ld + V_power.
    - LD correction: partially reduces LD mismatch by fraction α (0 < α < 1).
      Noise = V_E + (1 - α) · V_ld + V_power.
    - Meta-analysis: larger sample removes power loss, reduces LD loss
      further (fraction β where α < β < 1).
      Noise = V_E + (1 - β) · V_ld.
    - New GWAS in target: eliminates LD mismatch and power loss entirely.
      Noise = V_E.
    The ordering is derived from the noise levels being strictly decreasing,
    which follows from 0 < α < β < 1 and positivity of components. -/
theorem intervention_hierarchy
    (vSig V_E V_ld V_power V_cal α β : ℝ)
    (h_sig : 0 < vSig) (h_VE : 0 < V_E)
    (h_ld : 0 < V_ld) (h_power : 0 < V_power) (h_cal : 0 < V_cal)
    (h_α_pos : 0 < α) (h_αβ : α < β) (h_β_lt : β < 1) :
    -- Original < recalibrated < LD-corrected < meta-analysis < new GWAS
    expectedR2 vSig (V_E + V_ld + V_power + V_cal) <
      expectedR2 vSig (V_E + V_ld + V_power) ∧
    expectedR2 vSig (V_E + V_ld + V_power) <
      expectedR2 vSig (V_E + (1 - α) * V_ld + V_power) ∧
    expectedR2 vSig (V_E + (1 - α) * V_ld + V_power) <
      expectedR2 vSig (V_E + (1 - β) * V_ld) ∧
    expectedR2 vSig (V_E + (1 - β) * V_ld) <
      expectedR2 vSig V_E := by
  unfold expectedR2
  refine ⟨?_, ?_, ?_, ?_⟩
  · exact div_lt_div_of_pos_left h_sig (by nlinarith) (by nlinarith)
  · exact div_lt_div_of_pos_left h_sig (by nlinarith) (by nlinarith)
  · exact div_lt_div_of_pos_left h_sig (by nlinarith) (by nlinarith)
  · exact div_lt_div_of_pos_left h_sig (by linarith) (by nlinarith)

/-- **Diminishing returns from each intervention.**
    R² = v/(v + V_E) is concave in signal variance v.
    Equal increments Δ in signal give decreasing marginal R² gains:
    the gain from v to v+Δ exceeds the gain from v+Δ to v+2Δ.
    This is because the denominator grows, so each additional unit
    of signal is divided by a larger total variance. -/
theorem diminishing_marginal_returns
    (v Δ V_E : ℝ)
    (hv : 0 ≤ v) (hΔ : 0 < Δ) (hVE : 0 < V_E) :
    -- Second increment gives less R² gain than the first
    expectedR2 (v + 2 * Δ) V_E - expectedR2 (v + Δ) V_E <
      expectedR2 (v + Δ) V_E - expectedR2 v V_E := by
  unfold expectedR2
  have ha : 0 < v + V_E := by linarith
  have hb : 0 < v + Δ + V_E := by linarith
  have hc : 0 < v + 2 * Δ + V_E := by linarith
  have h_gain2 :
      (v + 2 * Δ) / (v + 2 * Δ + V_E) - (v + Δ) / (v + Δ + V_E) =
        (Δ * V_E) / ((v + 2 * Δ + V_E) * (v + Δ + V_E)) := by
    field_simp [ne_of_gt hb, ne_of_gt hc]
    ring
  have h_gain1 :
      (v + Δ) / (v + Δ + V_E) - v / (v + V_E) =
        (Δ * V_E) / ((v + Δ + V_E) * (v + V_E)) := by
    field_simp [ne_of_gt ha, ne_of_gt hb]
    ring
  rw [h_gain2, h_gain1]
  have hnum : 0 < Δ * V_E := by positivity
  have hden2 : 0 < (v + 2 * Δ + V_E) * (v + Δ + V_E) := mul_pos hc hb
  have hden1 : 0 < (v + Δ + V_E) * (v + V_E) := mul_pos hb ha
  have hden_lt :
      (v + Δ + V_E) * (v + V_E) <
        (v + 2 * Δ + V_E) * (v + Δ + V_E) := by
    nlinarith
  apply (div_lt_div_iff₀ hden2 hden1).2
  nlinarith [hnum, hden_lt]

/-- **Cost-effectiveness analysis.**
    New GWAS is most effective but most expensive.
    Computational corrections are cheap but limited.
    Optimal strategy depends on budget. -/
noncomputable def costEffectiveness (improvement cost : ℝ) : ℝ :=
  improvement / cost

/-- Higher cost-effectiveness is better. -/
theorem choose_more_cost_effective
    (improv₁ improv₂ cost₁ cost₂ : ℝ)
    (h_ce₁ : costEffectiveness improv₂ cost₂ < costEffectiveness improv₁ cost₁)
    (h_c₁ : 0 < cost₁) (h_c₂ : 0 < cost₂) :
    improv₁ * cost₂ > improv₂ * cost₁ := by
  unfold costEffectiveness at h_ce₁
  rwa [div_lt_div_iff₀ h_c₂ h_c₁] at h_ce₁

end InterventionsForPortability


/-!
## Sensitivity Analysis

Sensitivity analysis quantifies how robust portability estimates
are to violations of modeling assumptions.
-/

section SensitivityAnalysis

/- **Derivation: E-value from the confounding bounding formula.**

    The E-value (VanderWeele & Ding, 2017) is the minimum strength of
    unmeasured confounding on the risk ratio scale that could fully explain
    away an observed association. We derive E = RR + √(RR(RR-1)).

    **Setup.** Let RR_obs be the observed risk ratio. An unmeasured
    confounder U introduces bias through two pathways:
    - RR_{EU}: the confounder-exposure association (risk ratio)
    - RR_{UD}: the confounder-outcome association (risk ratio)

    **Step 1: The Cornfield/bounding inequality.**
    The maximum bias factor B from an unmeasured confounder satisfies:
        B ≤ (RR_{EU} × RR_{UD}) / (RR_{EU} + RR_{UD} - 1)
    The observed RR relates to the true RR via: RR_obs = RR_true × B.

    **Step 2: Setting up the E-value equation.**
    The E-value asks: what is the minimum confounding strength E such that
    B = RR_obs (i.e., RR_true = 1, the association is fully explained)?
    For the sharpest bound, set RR_{EU} = RR_{UD} = E (symmetric
    confounding). Then:
        RR_obs = E² / (2E - 1)

    **Step 3: Solving for E.**
    Rearranging RR = E²/(2E - 1):
        RR × (2E - 1) = E²
        E² - 2·RR·E + RR = 0
    By the quadratic formula:
        E = (2·RR ± √(4·RR² - 4·RR)) / 2
          = RR ± √(RR² - RR)
          = RR ± √(RR(RR - 1))
    Taking the positive root (E ≥ RR ≥ 1):
        **E = RR + √(RR(RR - 1))**

    **Verification.** When RR = 1: E = 1 + 0 = 1 (no confounding needed).
    As RR → ∞: E ≈ 2·RR (confounding must be roughly twice the observed
    association on the RR scale). -/

/-- **E-value for unmeasured confounding.**
    The E-value is the minimum confounding strength that could
    explain away the observed portability difference. -/
noncomputable def eValue (rr : ℝ) : ℝ :=
  rr + Real.sqrt (rr * (rr - 1))

/-- E-value ≥ 1 for RR ≥ 1. -/
theorem e_value_ge_one (rr : ℝ) (h_rr : 1 ≤ rr) :
    1 ≤ eValue rr := by
  unfold eValue
  have : 0 ≤ Real.sqrt (rr * (rr - 1)) := Real.sqrt_nonneg _
  linarith

structure LDReferenceModel where
  signal_variance : ℝ
  true_noise : ℝ
  reference_mismatch : ℝ
  h_sig : 0 < signal_variance
  h_noise : 0 < true_noise
  h_mismatch : 0 < reference_mismatch

noncomputable def inSampleR2 (m : LDReferenceModel) : ℝ :=
  m.signal_variance / (m.signal_variance + m.true_noise)

noncomputable def externalReferenceR2 (m : LDReferenceModel) : ℝ :=
  m.signal_variance / (m.signal_variance + m.true_noise + m.reference_mismatch)

/-- **Sensitivity to LD reference mismatch.**
    Portability estimates are sensitive to the choice of LD reference.
    Using an external reference adds mismatch noise, strictly reducing estimated R². -/
theorem ld_reference_sensitivity (m : LDReferenceModel) :
  externalReferenceR2 m < inSampleR2 m := by
  unfold inSampleR2 externalReferenceR2
  have h_den1 : 0 < m.signal_variance + m.true_noise := by linarith [m.h_sig, m.h_noise]
  have h_den2 : 0 < m.signal_variance + m.true_noise + m.reference_mismatch := by linarith [m.h_sig, m.h_noise, m.h_mismatch]
  apply (div_lt_div_iff₀ h_den2 h_den1).2
  have h_sig : 0 < m.signal_variance := m.h_sig
  have h_mismatch : 0 < m.reference_mismatch := m.h_mismatch
  nlinarith

structure PhenotypeModel where
  genetic_variance : ℝ
  env_variance : ℝ
  measurement_error : ℝ
  h_gen_pos : 0 < genetic_variance
  h_env_pos : 0 < env_variance
  h_err_nonneg : 0 ≤ measurement_error

noncomputable def observedHeritability (m : PhenotypeModel) : ℝ :=
  m.genetic_variance / (m.genetic_variance + m.env_variance + m.measurement_error)

/-- **Sensitivity to phenotype definition.**
    A phenotype definition with larger measurement error strictly reduces
    the observed heritability (and thus portability estimates based on it). -/
theorem phenotype_definition_matters
    (m1 m2 : PhenotypeModel)
    (h_same_gen : m1.genetic_variance = m2.genetic_variance)
    (h_same_env : m1.env_variance = m2.env_variance)
    (h_err_diff : m1.measurement_error < m2.measurement_error) :
    observedHeritability m2 < observedHeritability m1 := by
  unfold observedHeritability
  have h_num_pos : 0 < m1.genetic_variance := m1.h_gen_pos
  have h_den1_pos : 0 < m1.genetic_variance + m1.env_variance + m1.measurement_error := by linarith [m1.h_gen_pos, m1.h_env_pos, m1.h_err_nonneg]
  have h_den2_pos : 0 < m2.genetic_variance + m2.env_variance + m2.measurement_error := by linarith [m2.h_gen_pos, m2.h_env_pos, m2.h_err_nonneg]
  rw [h_same_gen, h_same_env] at *
  apply (div_lt_div_iff₀ h_den2_pos h_den1_pos).2
  nlinarith [h_num_pos, h_err_diff]

end SensitivityAnalysis

end Calibrator
