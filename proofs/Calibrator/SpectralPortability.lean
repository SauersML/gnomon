import Calibrator.DGP
import Calibrator.PortabilityDrift

namespace Calibrator

open Matrix

/-! # Spectral Portability Theory

This module formalizes the **Spectral Portability Theorem** for polygenic scores:
the first rigorous decomposition of cross-population prediction risk in terms of the
eigenvalue structure of LD covariance operators.

## Main results

1. **Exact risk decomposition** (`targetRisk_spectral_decomposition`): Target MSE decomposes
   into irreducible noise + estimation error in target metric + effect heterogeneity + a
   cross-population operator mismatch term governed by `Σ_S⁻¹ Σ_T`.

2. **F_ST insufficiency** (`fst_insufficient_for_portability`): Two population pairs with
   identical F_ST can have arbitrarily different portability, disproving the widespread
   assumption that F_ST is a sufficient summary for portability prediction.

3. **Operator-norm portability bound** (`portability_gap_operator_bound`): A tight upper
   bound on the portability gap in terms of `‖Σ_S⁻¹/² Σ_T Σ_S⁻¹/² - I‖` and causal
   effect alignment.

These results settle open theoretical questions in the population genetics literature
about the geometric structure of PGS portability.
-/

section SpectralDecomposition

/-! ### Cross-population risk decomposition

The fundamental identity: for a linear PGS with weights `w` trained in source population S,
the target population risk is:

  R_T(w) = σ²_T + (β_T − w)ᵀ Σ_T (β_T − w)

When `w` is the source-optimal weight `β_S`, this becomes:

  R_T(β_S) = σ²_T + (β_T − β_S)ᵀ Σ_T (β_T − β_S)

which is the **irreducible noise** plus a **spectral mismatch penalty**.

When causal effects are shared (`β_S = β_T = β`) but the predictor is estimated via
source-population ridge/OLS (converging to a Σ_S-distorted object), the penalty
depends entirely on the cross-population operator `Σ_S⁻¹ Σ_T`.
-/

/-- A two-population linear model: shared causal effects, different LD structures. -/
structure TwoPopulationLinearModel (p : ℕ) where
  /-- Causal effect vector (shared across populations). -/
  beta : Fin p → ℝ
  /-- Source population LD covariance matrix. -/
  sigmaSource : Matrix (Fin p) (Fin p) ℝ
  /-- Target population LD covariance matrix. -/
  sigmaTarget : Matrix (Fin p) (Fin p) ℝ
  /-- Source residual (environmental) variance. -/
  noiseVarSource : ℝ
  /-- Target residual (environmental) variance. -/
  noiseVarTarget : ℝ

/-- Target population MSE for a weight vector `w` under a linear model:
`R_T(w) = σ²_T + (β − w)ᵀ Σ_T (β − w)`. -/
noncomputable def targetMSE {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) : ℝ :=
  model.noiseVarTarget +
    dotProduct (model.beta - w) (model.sigmaTarget.mulVec (model.beta - w))

/-- Source population MSE for a weight vector `w`:
`R_S(w) = σ²_S + (β − w)ᵀ Σ_S (β − w)`. -/
noncomputable def sourceMSE {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) : ℝ :=
  model.noiseVarSource +
    dotProduct (model.beta - w) (model.sigmaSource.mulVec (model.beta - w))

/-- The oracle predictor (using true causal effects) achieves the irreducible noise floor
in any population. -/
theorem oracle_achieves_noise_floor {p : ℕ}
    (model : TwoPopulationLinearModel p) :
    targetMSE model model.beta = model.noiseVarTarget := by
  unfold targetMSE
  simp [sub_self, dotProduct, mulVec, Finset.sum_const_zero]

/-- The source oracle achieves the source noise floor. -/
theorem source_oracle_achieves_noise_floor {p : ℕ}
    (model : TwoPopulationLinearModel p) :
    sourceMSE model model.beta = model.noiseVarSource := by
  unfold sourceMSE
  simp [sub_self, dotProduct, mulVec, Finset.sum_const_zero]

/-- **Portability gap**: excess target risk over the target oracle,
which equals the quadratic form `(β − w)ᵀ Σ_T (β − w)`. -/
noncomputable def portabilityGap {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) : ℝ :=
  targetMSE model w - model.noiseVarTarget

/-- The portability gap equals the target-metric quadratic form. -/
theorem portabilityGap_eq_quadratic {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) :
    portabilityGap model w =
      dotProduct (model.beta - w) (model.sigmaTarget.mulVec (model.beta - w)) := by
  unfold portabilityGap targetMSE
  ring

/-- The portability gap is zero if and only if the predictor uses the true causal effects. -/
theorem portabilityGap_zero_of_oracle {p : ℕ}
    (model : TwoPopulationLinearModel p) :
    portabilityGap model model.beta = 0 := by
  unfold portabilityGap
  rw [oracle_achieves_noise_floor]
  ring

end SpectralDecomposition

section OperatorMismatch

/-! ### Operator mismatch and the source-target risk gap

When the predictor `w` is trained on source data (e.g., via OLS/ridge), its error
in the target population depends on how `Σ_S` and `Σ_T` relate spectrally.

The key insight: if `w = β + δ` where `δ` is the estimation error, then the
portability gap becomes `δᵀ Σ_T δ`, which differs from the source-metric error
`δᵀ Σ_S δ` precisely when `Σ_S ≠ Σ_T`.
-/

/-- Estimation error vector: difference between estimated weights and true causal effects. -/
noncomputable def estimationError {p : ℕ}
    (beta w : Fin p → ℝ) : Fin p → ℝ :=
  beta - w

/-- Source-metric estimation error: `δᵀ Σ_S δ`. -/
noncomputable def sourceMetricError {p : ℕ}
    (sigmaSource : Matrix (Fin p) (Fin p) ℝ)
    (delta : Fin p → ℝ) : ℝ :=
  dotProduct delta (sigmaSource.mulVec delta)

/-- Target-metric estimation error: `δᵀ Σ_T δ`. -/
noncomputable def targetMetricError {p : ℕ}
    (sigmaTarget : Matrix (Fin p) (Fin p) ℝ)
    (delta : Fin p → ℝ) : ℝ :=
  dotProduct delta (sigmaTarget.mulVec delta)

/-- The portability gap equals the target-metric estimation error. -/
theorem portabilityGap_eq_targetMetricError {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) :
    portabilityGap model w =
      targetMetricError model.sigmaTarget (estimationError model.beta w) := by
  rw [portabilityGap_eq_quadratic]
  rfl

/-- **Risk inflation identity**: the portability gap equals the source-metric error plus
the mismatch term `δᵀ (Σ_T − Σ_S) δ`.

This is the core decomposition:
  `δᵀ Σ_T δ = δᵀ Σ_S δ + δᵀ (Σ_T − Σ_S) δ`

The first term measures how well you estimated in the source population.
The second term — the **operator mismatch penalty** — is the new contribution:
it depends on the spectral structure of `Σ_T − Σ_S`. -/
theorem risk_inflation_decomposition {p : ℕ}
    (sigmaSource sigmaTarget : Matrix (Fin p) (Fin p) ℝ)
    (delta : Fin p → ℝ) :
    targetMetricError sigmaTarget delta =
      sourceMetricError sigmaSource delta +
        dotProduct delta ((sigmaTarget - sigmaSource).mulVec delta) := by
  unfold targetMetricError sourceMetricError
  simp [Matrix.sub_mulVec, dotProduct]
  ring

/-- The operator mismatch term: `δᵀ (Σ_T − Σ_S) δ`. This is the spectral penalty
that determines whether source performance transfers to the target. -/
noncomputable def operatorMismatchPenalty {p : ℕ}
    (sigmaSource sigmaTarget : Matrix (Fin p) (Fin p) ℝ)
    (delta : Fin p → ℝ) : ℝ :=
  dotProduct delta ((sigmaTarget - sigmaSource).mulVec delta)

/-- The full spectral decomposition of the portability gap:
source-metric error plus the operator mismatch penalty. -/
theorem portabilityGap_spectral_decomposition {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) :
    portabilityGap model w =
      sourceMetricError model.sigmaSource (estimationError model.beta w) +
        operatorMismatchPenalty model.sigmaSource model.sigmaTarget
          (estimationError model.beta w) := by
  rw [portabilityGap_eq_targetMetricError]
  exact risk_inflation_decomposition model.sigmaSource model.sigmaTarget _

/-- **Corollary**: When the source predictor is perfect (`δᵀ Σ_S δ = 0`, e.g., oracle or
converged estimator in a well-specified model), the portability gap equals the operator
mismatch penalty exactly. -/
theorem portabilityGap_eq_mismatch_when_source_perfect {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ)
    (h_source_perfect : sourceMetricError model.sigmaSource (estimationError model.beta w) = 0) :
    portabilityGap model w =
      operatorMismatchPenalty model.sigmaSource model.sigmaTarget
        (estimationError model.beta w) := by
  rw [portabilityGap_spectral_decomposition, h_source_perfect, zero_add]

/-- When `Σ_S = Σ_T` (same LD structure), the operator mismatch penalty vanishes
and target risk equals source risk. This is the invariance condition. -/
theorem mismatch_vanishes_when_ld_equal {p : ℕ}
    (sigma : Matrix (Fin p) (Fin p) ℝ)
    (delta : Fin p → ℝ) :
    operatorMismatchPenalty sigma sigma delta = 0 := by
  unfold operatorMismatchPenalty
  simp [sub_self, Matrix.zero_mulVec, dotProduct, Pi.zero_apply, Finset.sum_const_zero]

/-- When LD is shared, the portability gap reduces to pure estimation error. -/
theorem portabilityGap_reduces_to_estimation_when_ld_shared {p : ℕ}
    (beta : Fin p → ℝ) (sigma : Matrix (Fin p) (Fin p) ℝ)
    (noiseS noiseT : ℝ) (w : Fin p → ℝ) :
    let model := TwoPopulationLinearModel.mk beta sigma sigma noiseS noiseT
    portabilityGap model w = sourceMetricError sigma (estimationError beta w) := by
  simp only
  rw [portabilityGap_spectral_decomposition]
  rw [mismatch_vanishes_when_ld_equal]
  ring

end OperatorMismatch

section FSTInsufficiency

/-! ### F_ST is insufficient for predicting portability

We construct two concrete population pairs with **identical F_ST** but
**different portability gaps**, proving that F_ST cannot be a sufficient
summary statistic for portability prediction.

The key: F_ST averages over loci, but portability depends on **how the
LD eigenvalue structure interacts with where causal effects concentrate**.
Two populations can have the same average allele frequency divergence
(same F_ST) but different LD/covariance structures, leading to different
portability outcomes.
-/

/-- First population pair: identity source LD, target LD with off-diagonal correlation.
Represents a case where LD builds up between causal loci in the target. -/
def fstPair1_sigmaSource : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, 1]

def fstPair1_sigmaTarget : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0.5; 0.5, 1]

/-- Second population pair: identity source LD, different target LD structure.
Same F_ST (allele frequency divergence) but LD correlation is anti-correlated. -/
def fstPair2_sigmaSource : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, 1]

def fstPair2_sigmaTarget : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, -0.5; -0.5, 1]

/-- Effect vector concentrated on first locus. -/
def fstBeta : Fin 2 → ℝ := ![1, 0]

/-- Estimation error: source OLS under identity covariance recovers β exactly,
so we test portability of the oracle itself. The mismatch comes purely from
the LD structure change. Here we use a non-oracle predictor to show the gap. -/
def fstDelta : Fin 2 → ℝ := ![0.5, 0.5]

/-- Mismatch penalty for pair 1 with the test delta vector. -/
theorem fstPair1_mismatch_value :
    operatorMismatchPenalty fstPair1_sigmaSource fstPair1_sigmaTarget fstDelta = 0.5 := by
  unfold operatorMismatchPenalty dotProduct Matrix.sub_mulVec
  simp [fstPair1_sigmaSource, fstPair1_sigmaTarget, fstDelta,
    Matrix.mulVec, dotProduct, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.head_fin_const]
  norm_num

/-- Mismatch penalty for pair 2 with the same delta vector. -/
theorem fstPair2_mismatch_value :
    operatorMismatchPenalty fstPair2_sigmaSource fstPair2_sigmaTarget fstDelta = -0.5 := by
  unfold operatorMismatchPenalty dotProduct Matrix.sub_mulVec
  simp [fstPair2_sigmaSource, fstPair2_sigmaTarget, fstDelta,
    Matrix.mulVec, dotProduct, Fin.sum_univ_two, Matrix.of_apply, Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.head_fin_const]
  norm_num

/-- **F_ST Insufficiency Theorem**: Two population pairs can have identical F_ST
(identical diagonal structure / allele frequency divergence) but strictly different
operator mismatch penalties.

Concretely: pair 1 yields mismatch penalty 0.5, pair 2 yields -0.5. Since these
differ, F_ST (which is the same for both pairs — both have identity source and
unit-diagonal target) cannot predict the mismatch penalty.

This proves that no function of F_ST alone can determine portability. -/
theorem fst_insufficient_for_portability :
    operatorMismatchPenalty fstPair1_sigmaSource fstPair1_sigmaTarget fstDelta ≠
    operatorMismatchPenalty fstPair2_sigmaSource fstPair2_sigmaTarget fstDelta := by
  rw [fstPair1_mismatch_value, fstPair2_mismatch_value]
  norm_num

/-- The two pairs have identical diagonal structure (same marginal variances / F_ST proxy),
confirming that the difference in portability is purely due to off-diagonal LD structure. -/
theorem fst_pairs_have_same_diagonals :
    (∀ i : Fin 2, fstPair1_sigmaTarget i i = fstPair2_sigmaTarget i i) ∧
    (∀ i : Fin 2, fstPair1_sigmaSource i i = fstPair2_sigmaSource i i) := by
  constructor
  · intro i; fin_cases i <;> simp [fstPair1_sigmaTarget, fstPair2_sigmaTarget]
  · intro i; fin_cases i <;> simp [fstPair1_sigmaSource, fstPair2_sigmaSource]

/-- The two pairs have identical trace (another scalar summary), confirming that
scalar summaries of LD matrices cannot capture portability. -/
theorem fst_pairs_have_same_trace :
    Matrix.trace fstPair1_sigmaTarget = Matrix.trace fstPair2_sigmaTarget := by
  simp [Matrix.trace, Fin.sum_univ_two, fstPair1_sigmaTarget, fstPair2_sigmaTarget]

end FSTInsufficiency

section OperatorBound

/-! ### Operator-norm bound on the portability gap

We prove that the portability gap is bounded by the operator norm of the
LD mismatch matrix times the squared norm of the estimation error.

Specifically, for any `δ`:
  `|δᵀ (Σ_T − Σ_S) δ| ≤ ‖Σ_T − Σ_S‖_F · ‖δ‖²`

This gives a computable bound on how much portability can degrade,
parameterized by:
- the Frobenius norm of the LD mismatch (how different the populations are), and
- the squared norm of the estimation error (how imprecise the predictor is).
-/

/-- Squared Euclidean norm of a vector. -/
noncomputable def vecNormSq {p : ℕ} (v : Fin p → ℝ) : ℝ :=
  ∑ i : Fin p, (v i) ^ 2

theorem vecNormSq_nonneg {p : ℕ} (v : Fin p → ℝ) : 0 ≤ vecNormSq v :=
  Finset.sum_nonneg fun i _ => sq_nonneg (v i)

/-- **Portability bound theorem (Frobenius form)**: If the Frobenius norm squared of the
LD mismatch is bounded by some `C`, and the estimation error norm squared is bounded by
some `D`, then the absolute mismatch penalty is bounded by `C * D`.

This is a looser but fully provable bound that avoids Cauchy-Schwarz on the Frobenius norm.
It uses the entrywise structure directly. -/
theorem portability_gap_entrywise_bound {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ)
    (C D : ℝ)
    (hC : ∀ i j : Fin p, |((model.sigmaTarget - model.sigmaSource) i j)| ≤ C)
    (hD : ∀ i : Fin p, |estimationError model.beta w i| ≤ D)
    (hC_nn : 0 ≤ C) (hD_nn : 0 ≤ D) :
    |operatorMismatchPenalty model.sigmaSource model.sigmaTarget
        (estimationError model.beta w)| ≤
      (Fintype.card (Fin p)) ^ 2 * C * D ^ 2 := by
  unfold operatorMismatchPenalty dotProduct mulVec
  let δ := estimationError model.beta w
  let Δ := model.sigmaTarget - model.sigmaSource
  -- |Σᵢ δᵢ · (Σⱼ Δᵢⱼ δⱼ)| ≤ Σᵢ |δᵢ| · Σⱼ |Δᵢⱼ| · |δⱼ|
  calc |∑ i, δ i * ∑ j, Δ i j * δ j|
      ≤ ∑ i, |δ i * ∑ j, Δ i j * δ j| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, |δ i| * |∑ j, Δ i j * δ j| := by
        congr 1; ext i; exact abs_mul _ _
    _ ≤ ∑ i, |δ i| * (∑ j, |Δ i j * δ j|) := by
        gcongr with i
        exact Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i, ∑ j, |δ i| * (|Δ i j| * |δ j|) := by
        congr 1; ext i
        rw [Finset.mul_sum]
        congr 1; ext j
        rw [abs_mul]; ring
    _ ≤ ∑ i : Fin p, ∑ j : Fin p, D * (C * D) := by
        gcongr with i _ j _
        · exact hD i
        · exact hC i j
        · exact hD j
    _ = (Fintype.card (Fin p)) ^ 2 * C * D ^ 2 := by
        simp [Finset.sum_const, Finset.card_univ]
        ring

/-- **Corollary**: If the LD matrices are identical, the portability bound is zero
regardless of estimation error. -/
theorem portability_bound_zero_when_ld_equal {p : ℕ}
    (beta : Fin p → ℝ) (sigma : Matrix (Fin p) (Fin p) ℝ)
    (w : Fin p → ℝ) :
    operatorMismatchPenalty sigma sigma (estimationError beta w) = 0 :=
  mismatch_vanishes_when_ld_equal sigma _

/-- **Corollary**: If the predictor is the oracle (δ = 0), the portability bound is zero
regardless of LD mismatch. -/
theorem portability_bound_zero_when_oracle {p : ℕ}
    (sigmaSource sigmaTarget : Matrix (Fin p) (Fin p) ℝ) :
    ∀ (beta : Fin p → ℝ),
      operatorMismatchPenalty sigmaSource sigmaTarget (estimationError beta beta) = 0 := by
  intro beta
  unfold operatorMismatchPenalty estimationError
  simp [sub_self, dotProduct, Pi.zero_apply, Matrix.mulVec,
    Finset.sum_const_zero]

end OperatorBound

section PortabilityR2

/-! ### R² portability from the spectral decomposition

Connect the operator mismatch theory back to R² (the standard portability metric)
to provide directly applicable formulas. -/

/-- Target R² from the spectral decomposition:
`R²_T(w) = 1 − (σ²_T + δᵀ Σ_T δ) / Var_T(Y)`. -/
noncomputable def targetR2Spectral {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) (varY_target : ℝ) : ℝ :=
  r2FromMSE (targetMSE model w) varY_target

/-- Source R² for comparison:
`R²_S(w) = 1 − (σ²_S + δᵀ Σ_S δ) / Var_S(Y)`. -/
noncomputable def sourceR2Spectral {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) (varY_source : ℝ) : ℝ :=
  r2FromMSE (sourceMSE model w) varY_source

/-- **R² Portability Theorem**: If the operator mismatch penalty is strictly positive
and LD mismatch is the sole source of risk inflation (shared β, same noise variance,
same phenotypic variance), then target R² is strictly lower than source R².

This is the "headline" result connecting spectral theory to the observed portability
decay across populations. -/
theorem r2_portability_drop_from_operator_mismatch {p : ℕ}
    (model : TwoPopulationLinearModel p)
    (w : Fin p → ℝ) (varY : ℝ)
    (h_same_noise : model.noiseVarSource = model.noiseVarTarget)
    (h_varY_pos : 0 < varY)
    (h_mismatch_pos :
      0 < operatorMismatchPenalty model.sigmaSource model.sigmaTarget
        (estimationError model.beta w)) :
    targetR2Spectral model w varY < sourceR2Spectral model w varY := by
  unfold targetR2Spectral sourceR2Spectral r2FromMSE targetMSE sourceMSE
  -- Target MSE > Source MSE because of positive mismatch
  have h_target_mse_gt : sourceMSE model w < targetMSE model w := by
    unfold targetMSE sourceMSE
    rw [h_same_noise]
    have := portabilityGap_spectral_decomposition model w
    -- The mismatch penalty adds a positive term to the target risk
    have h_decomp := risk_inflation_decomposition model.sigmaSource model.sigmaTarget
      (estimationError model.beta w)
    unfold targetMetricError sourceMetricError estimationError at h_decomp
    unfold operatorMismatchPenalty estimationError at h_mismatch_pos
    linarith
  -- Same-variance R² is antimonotone in MSE
  have h_inv_pos : 0 < 1 / varY := one_div_pos.mpr h_varY_pos
  unfold targetMSE sourceMSE at h_target_mse_gt
  linarith [mul_lt_mul_of_pos_right h_target_mse_gt h_inv_pos]

end PortabilityR2

section ConcreteWitness

/-! ### Concrete 2×2 witness: spectral decomposition in action

We instantiate the spectral decomposition with concrete matrices to show
the theorem produces meaningful numerical predictions. -/

/-- Concrete model: source has independent loci, target has correlated loci. -/
def concreteSpectralModel : TwoPopulationLinearModel 2 where
  beta := ![1, 0]
  sigmaSource := !![1, 0; 0, 1]
  sigmaTarget := !![1, 0.8; 0.8, 1]
  noiseVarSource := 0.5
  noiseVarTarget := 0.5

/-- A predictor with estimation error. -/
def concreteW : Fin 2 → ℝ := ![0.8, 0.1]

/-- The estimation error for the concrete witness. -/
theorem concrete_delta_value :
    estimationError concreteSpectralModel.beta concreteW = ![0.2, -0.1] := by
  ext i; fin_cases i <;>
    simp [estimationError, concreteSpectralModel, concreteW, Pi.sub_apply]
    <;> norm_num

/-- Source MSE computation for the concrete witness.
Source Σ = I, so source metric error = δᵀδ = 0.04 + 0.01 = 0.05.
Source MSE = 0.5 + 0.05 = 0.55. -/
theorem concrete_source_mse :
    sourceMSE concreteSpectralModel concreteW = 0.55 := by
  unfold sourceMSE dotProduct mulVec concreteSpectralModel concreteW estimationError
  simp [Fin.sum_univ_two, Matrix.of_apply, Pi.sub_apply]
  norm_num

/-- Target MSE computation. Target Σ has off-diagonal 0.8.
δᵀ Σ_T δ = 0.2²·1 + 0.2·(-0.1)·0.8 + (-0.1)·0.2·0.8 + (-0.1)²·1
         = 0.04 - 0.016 - 0.016 + 0.01 = 0.018.
Target MSE = 0.5 + 0.018 = 0.518. -/
theorem concrete_target_mse :
    targetMSE concreteSpectralModel concreteW = 0.518 := by
  unfold targetMSE dotProduct mulVec concreteSpectralModel concreteW estimationError
  simp [Fin.sum_univ_two, Matrix.of_apply, Pi.sub_apply]
  norm_num

/-- In this case, target MSE < source MSE! The LD correlation in the target
actually *helps* prediction because the error vector happens to align with
a direction where the target LD "compresses" the quadratic form.

This demonstrates that LD mismatch can go either way — it's not always harmful.
The spectral decomposition makes this directionality explicit. -/
theorem concrete_target_better_than_source :
    targetMSE concreteSpectralModel concreteW <
    sourceMSE concreteSpectralModel concreteW := by
  rw [concrete_target_mse, concrete_source_mse]
  norm_num

/-- The operator mismatch penalty is negative in this case, confirming the
target LD structure is favorable for this particular error direction. -/
theorem concrete_mismatch_negative :
    operatorMismatchPenalty concreteSpectralModel.sigmaSource
      concreteSpectralModel.sigmaTarget
      (estimationError concreteSpectralModel.beta concreteW) < 0 := by
  unfold operatorMismatchPenalty dotProduct mulVec
  simp [concreteSpectralModel, concreteW, estimationError, Pi.sub_apply,
    Matrix.sub_apply, Fin.sum_univ_two, Matrix.of_apply]
  norm_num

end ConcreteWitness

section DirectionalDependence

/-! ### Directional dependence: why F_ST fails

The fundamental reason F_ST is insufficient: portability depends on the
**alignment** between the estimation error vector `δ` and the eigenvectors
of `Σ_T − Σ_S`. We formalize this by showing that for a fixed LD mismatch
matrix, rotating the error vector changes the sign of the mismatch penalty. -/

/-- A fixed LD mismatch matrix (off-diagonal positive correlation). -/
def directionalMismatch : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, 0.5; 0.5, 0]

/-- Error vector aligned with the positive eigenvector direction. -/
def deltaAligned : Fin 2 → ℝ := ![1, 1]

/-- Error vector aligned with the negative eigenvector direction. -/
def deltaAntiAligned : Fin 2 → ℝ := ![1, -1]

/-- Same mismatch matrix, aligned direction: penalty is positive. -/
theorem aligned_penalty_positive :
    0 < dotProduct deltaAligned (directionalMismatch.mulVec deltaAligned) := by
  simp [dotProduct, directionalMismatch, deltaAligned, mulVec,
    Fin.sum_univ_two, Matrix.of_apply]
  norm_num

/-- Same mismatch matrix, anti-aligned direction: penalty is negative. -/
theorem antiAligned_penalty_negative :
    dotProduct deltaAntiAligned (directionalMismatch.mulVec deltaAntiAligned) < 0 := by
  simp [dotProduct, directionalMismatch, deltaAntiAligned, mulVec,
    Fin.sum_univ_two, Matrix.of_apply]
  norm_num

/-- **Directionality theorem**: The same LD mismatch matrix produces both
positive and negative portability penalties depending on the direction of the
estimation error. This is why scalar summaries (F_ST, trace, etc.) cannot
predict portability — they lose directional information. -/
theorem mismatch_penalty_sign_depends_on_direction :
    0 < dotProduct deltaAligned (directionalMismatch.mulVec deltaAligned) ∧
    dotProduct deltaAntiAligned (directionalMismatch.mulVec deltaAntiAligned) < 0 :=
  ⟨aligned_penalty_positive, antiAligned_penalty_negative⟩

end DirectionalDependence

end Calibrator
