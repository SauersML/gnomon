-- proofs/Calibrator.lean
-- This file will contain all definitions, theorems, and proofs for the project.

import Mathlib.Probability.ConditionalProbability
import Mathlib.Probability.ConditionalExpectation
import Mathlib.Probability.Notation
import Mathlib.Probability.Integration
import Mathlib.Probability.Moments.Variance
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.LinearAlgebra.GeneralLinearGroup
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Data.Fin.Basic

-- The project's root namespace, as defined in `lakefile.lean`.
namespace Calibrator

/-!
=================================================================
## Part 1: Definitions
=================================================================

This section contains the formal mathematical definitions for:
- Core types (Phenotype Y, PGS P, Principal Components PC).
- The Phenotype-Informed GLM structure.
- Baseline models (Raw Score, Normalized Score).
- The four Data-Generating Process (DGP) scenarios.
-/

/-!
### Core Types

These match the paper's notation exactly:
- Y_j: The observed phenotype (could be binary for disease or continuous for biomarker)
- P_j: The raw polygenic score computed from SNPs
- PC_j: The principal components representing ancestry
-/

/-- The observed phenotype Y_j. Using `abbrev` means Phenotype and ℝ are interchangeable. -/
abbrev Phenotype := ℝ

/-- The raw, unadjusted Polygenic Score P_j -/
abbrev PGS := ℝ

/-- Principal Components PC_j = (PC_{j1}, ..., PC_{jk}).
    This is a function type: given index l ∈ {1,...,k}, returns PC_{jl}.
    Note: Fin k represents {0,...,k-1}, so we'll need to handle the index shift. -/
def PC (k : ℕ) := Fin k → ℝ

/-!
### Basis Functions

From the paper: B_m(P) for m = 0, ..., p where B_0(P) ≡ 1
-/

/-- Basis functions B_m(P) for transforming the PGS.
    `structure` creates a record type with named fields and proofs. -/
structure BasisFunctions (p : ℕ) where
  /-- B takes an index m ∈ {0,...,p} and returns a function PGS → ℝ -/
  B : Fin (p + 1) → (PGS → ℝ)
  /-- Proof that B_0 is the constant function 1 -/
  B_zero_is_one : B 0 = fun _ => 1

/-- Example: Linear basis with B_0(P) = 1, B_1(P) = P -/
def linearBasis : BasisFunctions 1 where
  B := fun m =>
    if m = 0 then fun _ => 1  -- B_0(P) = 1
    else fun p => p           -- B_1(P) = P
  B_zero_is_one := by
    -- This proves B 0 = fun _ => 1
    simp [funext_iff]

/-- Example: Quadratic basis with B_0(P) = 1, B_1(P) = P, B_2(P) = P² -/
def quadraticBasis : BasisFunctions 2 where
  B := fun m =>
    if m.val = 0 then fun _ => 1      -- B_0(P) = 1
    else if m.val = 1 then fun p => p  -- B_1(P) = P
    else fun p => p^2                  -- B_2(P) = P²
  B_zero_is_one := by simp [funext_iff, Fin.val_zero]

/-!
### Model Parameters

The γ parameters from Equation (1) in the paper
-/

/-- The model parameters γ = {γ_ml | m=0..p, l=0..k}.
    This is a function of two arguments:
    - First argument m ∈ {0,...,p} indexes the basis function
    - Second argument l ∈ {0,...,k} where l=0 is baseline, l>0 indexes PCs -/
def GammaParams (p k : ℕ) := Fin (p + 1) → Fin (k + 1) → ℝ

/-!
### Key Model Equations

These implement Equations (1) and (3) from the paper
-/

/-- Ancestry-dependent coefficient from Equation (1):
    α_m(PC_j) = γ_{m0} + Σ_{l=1}^k γ_{ml} PC_{jl}

    Note the index handling:
    - γ m 0 gives γ_{m0} (the baseline)
    - For the sum, l : Fin k represents l ∈ {1,...,k} in the paper
    - We use l.succ to shift l to {1,...,k} for indexing γ -/
def alpha (γ : GammaParams p k) (m : Fin (p + 1)) (pc : PC k) : ℝ :=
  γ m 0 + ∑ l : Fin k, γ m l.succ * pc l

/-- Linear predictor from Equation (3):
    η_j = Σ_{m=0}^p α_m(PC_j) B_m(P_j)

    This computes the weighted sum of basis functions. -/
def linearPredictor (γ : GammaParams p k) (B : BasisFunctions p) (pgs : PGS) (pc : PC k) : ℝ :=
  ∑ m : Fin (p + 1), alpha γ m pc * B.B m pgs

/-!
### GLM Components

Link functions and distributions for different phenotype types
-/

/-- GLM link functions. `inductive` creates a type with distinct constructors. -/
inductive LinkFunction
  | logit     -- For binary outcomes: link(π) = log(π/(1-π))
  | identity  -- For continuous outcomes: link(μ) = μ

/-- Apply the link function to transform from mean to linear predictor space -/
noncomputable def applyLink : LinkFunction → ℝ → ℝ
  | LinkFunction.logit, π => Real.log (π / (1 - π))
  | LinkFunction.identity, μ => μ

/-- Apply inverse link to get predictions.
    `noncomputable` because Real.exp cannot be computed exactly. -/
noncomputable def applyInverseLink : LinkFunction → ℝ → ℝ
  | LinkFunction.logit, η => 1 / (1 + Real.exp (-η))  -- sigmoid function
  | LinkFunction.identity, η => η

/-- Distribution families for GLM -/
inductive DistributionFamily
  | Bernoulli  -- Y_j ~ Bernoulli(π_j) for binary outcomes
  | Gaussian   -- Y_j ~ Normal(μ_j, σ²) for continuous outcomes

/-!
### The Phenotype-Informed GLM Model
-/

/-- Complete specification of a phenotype-informed GLM model -/
structure PhenotypeInformedGLM (p k : ℕ) where
  /-- Choice of basis functions B_m -/
  basis : BasisFunctions p
  /-- Fitted parameters γ_ml -/
  gamma : GammaParams p k
  /-- Link function (logit or identity) -/
  link : LinkFunction
  /-- Distribution family -/
  dist : DistributionFamily

/-- Make a prediction for a new individual with PGS=pgs and ancestry=pc.
    This implements the full prediction pipeline:
    1. Compute η = linearPredictor
    2. Apply inverse link to get Ŷ -/
noncomputable def predict (model : PhenotypeInformedGLM p k) (pgs : PGS) (pc : PC k) : ℝ :=
  let eta := linearPredictor model.gamma model.basis pgs pc
  applyInverseLink model.link eta

/-!
### Baseline Models for Comparison

These are the simpler models we compare against
-/

/-- Raw score model: Ŷ = g⁻¹(β₀ + β_P × P) with no ancestry adjustment -/
structure RawScoreModel where
  beta_P : ℝ        -- Coefficient for PGS
  intercept : ℝ     -- β₀
  link : LinkFunction

/-- Prediction using raw score -/
noncomputable def predictRaw (model : RawScoreModel) (pgs : PGS) : ℝ :=
  applyInverseLink model.link (model.intercept + model.beta_P * pgs)

/-- Normalized score model: First adjust P based on PC, then use adjusted score -/
structure NormalizedScoreModel (k : ℕ) where
  /-- Function that computes how much to subtract from P based on PC -/
  adjustmentCoeffs : PC k → ℝ
  /-- Coefficient for the adjusted PGS -/
  beta_P_adj : ℝ
  /-- Intercept -/
  intercept : ℝ
  /-- Link function -/
  link : LinkFunction

/-- Compute adjusted PGS: P' = P - adjustment(PC) -/
def adjustedPGS (model : NormalizedScoreModel k) (pgs : PGS) (pc : PC k) : ℝ :=
  pgs - model.adjustmentCoeffs pc

/-- Prediction using normalized score -/
noncomputable def predictNormalized (model : NormalizedScoreModel k) (pgs : PGS) (pc : PC k) : ℝ :=
  let p_adj := adjustedPGS model pgs pc
  applyInverseLink model.link (model.intercept + model.beta_P_adj * p_adj)

/-!
### The Four Scenarios

From the paper, representing different relationships between P, PC, and Y
-/

/-- The four scenarios describing different data-generating processes -/
inductive Scenario
  | RealGeneticDifferences      -- Scenario 1: True genetic effects vary by ancestry
  | DifferentialAccuracy        -- Scenario 2: PGS accuracy varies due to LD patterns
  | EnvironmentalCorrelation    -- Scenario 3: Environmental factors correlate with PCs
  | NeutralDifferences          -- Scenario 4: Neutral drift causes PGS differences

/-!
### Data Structures

For representing individuals and datasets
-/

/-- Data for a single individual j -/
structure Individual (k : ℕ) where
  /-- Observed phenotype Y_j -/
  phenotype : Phenotype
  /-- Raw polygenic score P_j -/
  pgs : PGS
  /-- Principal components PC_j -/
  pc : PC k

/-- Training dataset of n individuals.
    This is a function from indices {0,...,n-1} to Individual records. -/
def TrainingData (n k : ℕ) := Fin n → Individual k

/-- Data generating process specification for theoretical analysis -/
structure DataGeneratingProcess (k : ℕ) where
  /-- Which of the four scenarios this represents -/
  scenario : Scenario
  /-- True conditional expectation E[Y|P,PC] -/
  trueExpectation : PGS → PC k → ℝ
  /-- How P is distributed given PC (for studying confounding) -/
  pgsGivenPC : PC k → PGS → ℝ  -- Density function
  /-- Environmental effect correlated with PC (relevant for Scenario 3) -/
  environmentalEffect : PC k → ℝ

/-- Result of fitting a model to data -/
structure FittedModel (p k : ℕ) where
  /-- The fitted phenotype-informed GLM -/
  model : PhenotypeInformedGLM p k
  /-- Log-likelihood value on training data -/
  logLikelihood : ℝ
  /-- Standard errors for parameters (if computed) -/
  standardErrors : Option (GammaParams p k)


/-!
=================================================================
## Part 2: Theorems and Proofs
=================================================================

This section will state and prove the core claims of the framework.
-/

/-! ### Claim 1: Formalization of Scenarios

**The four scenarios can be mathematically formalized with precise data-generating processes.** Each scenario corresponds to a specific relationship between the true E[Y|P,PC], the distribution of P given PC, and the causal structure—allowing us to define exactly when we're in each scenario rather than relying on verbal descriptions.
-/

/-! ### Claim 1: Formalization of Scenarios

**The four scenarios can be mathematically formalized with precise data-generating processes.** Each scenario corresponds to a specific relationship between the true E[Y|P,PC], the distribution of P given PC, and the causal structure—allowing us to define exactly when we're in each scenario rather than relying on verbal descriptions.
-/

/-! ### Claim 1: Formalization of Scenarios

**The four scenarios can be mathematically formalized with precise data-generating processes.** Each scenario corresponds to a specific relationship between the true E[Y|P,PC], the distribution of P given PC, and the causal structure—allowing us to define exactly when we're in each scenario rather than relying on verbal descriptions.
-/

/-- Scenario 1: Real genetic differences in causal SNPs correlating with ancestry.
    The true genetic effect varies with ancestry. -/
noncomputable def Scenario1DGP (k : ℕ) : DataGeneratingProcess k where
  scenario := Scenario.RealGeneticDifferences
  -- E[Y|P,PC] has a genuine P×PC interaction due to ancestry-specific genetic effects
  trueExpectation := fun p pc =>
    let baseEffect := p  -- Base genetic effect
    let ancestryModulation := (∑ l : Fin k, 0.1 * pc l) * p  -- PC modulates genetic effect
    baseEffect + ancestryModulation
  -- P is shifted by ancestry due to different allele frequencies
  pgsGivenPC := fun pc p =>
    let mean := if h : k > 0 then 0.5 * pc ⟨0, h⟩ else 0  -- Mean PGS shifts with first PC
    Real.exp (-(p - mean)^2 / 2)  -- Gaussian density (unnormalized)
  -- No environmental confounding in pure Scenario 1
  environmentalEffect := fun _ => 0

/-- Scenario 2: Differential accuracy due to LD patterns.
    PGS accuracy (not effect) varies by ancestry. -/
noncomputable def Scenario2DGP (k : ℕ) : DataGeneratingProcess k where
  scenario := Scenario.DifferentialAccuracy
  -- The relationship weakens with distance from training ancestry, creating a P×PC interaction
  trueExpectation := fun p pc =>
    let accuracy := Real.exp (-0.5 * (∑ l : Fin k, (pc l)^2))  -- Accuracy decays with PC distance
    accuracy * p  -- P predicts Y but with ancestry-dependent accuracy
  -- P distribution may be independent of PC in a pure accuracy scenario
  pgsGivenPC := fun _ p =>
    Real.exp (-p^2 / 2)  -- Standard normal (unnormalized)
  -- No environmental effect
  environmentalEffect := fun _ => 0

/-- Scenario 3: Environmental factors correlating with ancestry.
    PC correlates with environmental factors affecting Y. -/
noncomputable def Scenario3DGP (k : ℕ) : DataGeneratingProcess k where
  scenario := Scenario.EnvironmentalCorrelation
  -- True genetic effect is constant, but environment adds PC-correlated effect (additive model)
  trueExpectation := fun p pc =>
    p + if h : k > 0 then pc ⟨0, h⟩ else 0  -- Genetic effect + environmental effect
  -- P may correlate with PC due to population structure
  pgsGivenPC := fun pc p =>
    let mean := if h : k > 0 then 0.3 * pc ⟨0, h⟩ else 0
    Real.exp (-(p - mean)^2 / 2)
  -- Environmental effect correlates with PC
  environmentalEffect := fun pc => if h : k > 0 then pc ⟨0, h⟩ else 0

/-- Scenario 4: Neutral differences due to population history.
    P varies with PC but this variation is non-causal. -/
noncomputable def Scenario4DGP (k : ℕ) : DataGeneratingProcess k where
  scenario := Scenario.NeutralDifferences
  -- Y depends on a "true" PGS, which is the observed P minus the neutral PC shift
  trueExpectation := fun p pc =>
    let truePGS := p - if h : k > 0 then 0.8 * pc ⟨0, h⟩ else 0  -- Remove neutral shift
    truePGS  -- Only true genetic component affects Y
  -- P is shifted by PC due to neutral drift
  pgsGivenPC := fun pc p =>
    let mean := if h : k > 0 then 0.8 * pc ⟨0, h⟩ else 0  -- Neutral shift in mean
    Real.exp (-(p - mean)^2 / 2)
  -- No environmental effect
  environmentalEffect := fun _ => 0

/-!
Key distinctions between scenarios:

1. **Scenario 1**: Both P distribution AND P's effect on Y vary with PC
2. **Scenario 2**: P's predictive accuracy varies with PC, also creating an interaction
3. **Scenario 3**: PC affects Y through environment, creating confounding (no P×PC interaction)
4. **Scenario 4**: P distribution varies with PC but this is non-causal drift (no P×PC interaction)

These formalizations enable precise mathematical analysis rather than verbal descriptions.
-/

/-- Helper: Check if a DGP exhibits P×PC interaction in the true expectation -/
def hasRealInteraction (dgp : DataGeneratingProcess k) : Prop :=
  ∃ (p₁ p₂ : PGS) (pc₁ pc₂ : PC k),
    (dgp.trueExpectation p₂ pc₁ - dgp.trueExpectation p₁ pc₁) ≠
    (dgp.trueExpectation p₂ pc₂ - dgp.trueExpectation p₁ pc₂)

/-- Helper: Check if P distribution depends on PC -/
def hasPGSDependenceOnPC (dgp : DataGeneratingProcess k) : Prop :=
  ∃ (p : PGS) (pc₁ pc₂ : PC k),
    pc₁ ≠ pc₂ ∧ dgp.pgsGivenPC pc₁ p ≠ dgp.pgsGivenPC pc₂ p

/-- Helper: Check for environmental confounding -/
def hasEnvironmentalConfounding (dgp : DataGeneratingProcess k) : Prop :=
  dgp.environmentalEffect ≠ (fun _ => 0)

/-- Claim 1 Formalization: Each scenario has distinct mathematical properties -/
theorem scenarios_are_distinct (k : ℕ) (h_k_pos : k > 0) :
  let s1 := Scenario1DGP k
  let s2 := Scenario2DGP k
  let s3 := Scenario3DGP k
  let s4 := Scenario4DGP k
  -- Scenario 1 has a P×PC interaction (true genetic effect modulation)
  (hasRealInteraction s1) ∧
  -- Scenario 2 also has a P×PC interaction (differential accuracy)
  (hasRealInteraction s2) ∧
  -- Scenario 3 has environmental confounding but no P×PC interaction
  (hasEnvironmentalConfounding s3 ∧ ¬hasRealInteraction s3) ∧
  -- Scenario 4 has P-PC dependence but no interaction and no environmental confounding
  (hasPGSDependenceOnPC s4 ∧ ¬hasRealInteraction s4 ∧ ¬hasEnvironmentalConfounding s4) := by
  sorry  -- Proof would verify these properties hold

/-! ### Claim 2: Necessity of Phenotype Data

**Scenarios 1 and 4 can produce identical PGS distributions but require opposite signs for the optimal interaction coefficient.** Formally: ∃ two data-generating processes where P|PC has identical distributions, but optimal prediction requires γ₁ₗ > 0 in one case and γ₁ₗ < 0 in the other along the same PC axis—proving phenotype data is mathematically necessary.
-/

/-! ### Claim 3: Independence Condition

**When PGS and PC are statistically independent, all interaction terms vanish in the optimal predictor.** Specifically: if Corr(P, PC_l) = 0 for all l in the population, then the Bayes-optimal prediction function has γ_ml = 0 for all m ≥ 1, l ≥ 1, reducing to a standard additive GLM.
-/

/-! ### Claim 4: Prediction-Causality Trade-off

**No method can simultaneously minimize prediction error and provide unbiased causal effect estimates in scenarios with PC-correlated environmental confounding.** This is a fundamental impossibility result: optimizing for prediction accuracy by leveraging environmental correlations is mathematically incompatible with estimating pure genetic effects.
-/

/-! ### Claim 5: Bayes-Optimality within Model Class

**The phenotype-informed method achieves Bayes-optimal prediction within its model class.** Among all functions in the linear-in-parameters GLM family with the specified basis functions, the phenotype-informed method minimizes E[(Y - f(P,PC))²], while raw and normalized methods are constrained to more restrictive subclasses.
-/

/-! ### Claim 6: Parameter Identifiability

**The model parameters are identifiable under standard GLM conditions.** Given linearly independent PCs and basis functions (non-singular design matrix), the parameters γ = {γ_ml} are uniquely determined by the population distribution of (Y,P,PC).
-/

/-! ### Claim 7: Quantitative Error of Normalization

**Normalization reduces accuracy in Scenario 1 by exactly the explained variance.** When true genetic effects correlate with PCs, normalization's prediction error increases by R²(genetic_effect, PC) × Var(genetic_effect) compared to the oracle predictor.
-/

/-! ### Claim 8: Quantitative Bias of Raw Scores

**Raw scores produce maximally biased predictions among unbiased linear predictors in pure Scenario 4.** When PC-correlated PGS differences are entirely due to neutral drift with no outcome association, raw score predictions exhibit bias equal to β_P × Cov(P,PC) × PC for each individual.
-/

/-! ### Claim 9: Correction of Multiplicative Bias

**Under pure multiplicative bias, the Bayes-optimal parameters exactly correct the distortion.** If E[Y|P,PC] = f(k(PC)×P) where k(PC) represents ancestry-specific PGS scaling, then the optimal parameters satisfy γ₁ₗ ∝ ∂log(k)/∂PC_l, exactly undoing the bias.
-/

/-! ### Claim 10: Shrinkage Effect

**Heteroscedastic PGS noise induces shrinkage-like calibration coefficients.** When Var(P - P_true | PC) varies with ancestry, the method automatically learns coefficients that shrink predictions toward ancestry-specific means in proportion to the local noise level.
-/

/-! ### Claim 11: Invariance to PC Transformation

**The method is invariant to affine transformations of PC space.** For any invertible linear transformation A and translation b, fitting the model with PC' = APC + b yields identical predictions Ŷ, ensuring results don't depend on arbitrary choices in PC computation.
-/

/-! ### Claim 12: Limitation - Extrapolation Risk

**Model predictions are unreliable for individuals outside the training PC support.** The calibration function learned from data in a bounded PC region cannot be guaranteed to extrapolate correctly to individuals with PC values far outside this region, constituting a fundamental limitation for application to unrepresented ancestries.
-/

/-! ### Claim 13: Limitation - Context Specificity

**Models are context-specific and may not transfer between environments.** A model trained in one environment is optimally calibrated only if PC-outcome associations (including environmental confounding) remain constant; different PC-environment correlations in a new setting will reduce predictive accuracy.
-/

end Calibrator
