# Calibration adapter crate

`gnomon/calibrate` is the domain adapter layer for Gnomon's calibration and
survival workflows. It owns schema/data policy, PGS/PC/sex feature semantics,
artifact mapping, and stable `gnomon::calibrate::*` entrypoints.

Core numerical engine modules (basis construction, PIRLS/REML, HMC, ALO,
reparameterization, diagnostics, and shared math types) live in the separate
solver engine repository and are imported by this crate.

## Statistical model

The crate dispatches three model families from `estimate.rs`, all sharing the
same marginal term-collection skeleton:

```
a(x) = β₀ + f_pgs(PGS) + γ_sex·sex + Σ_j f_j(PC_j)
```

Each `f_*` is a 1-D **Duchon RBF smooth** (farthest-point centers, linear
nullspace, length scale 1.0, power 1) and `γ_sex` is a doubly-penalized
unconstrained linear term. Tensor PGS×PC interactions and a sex×PGS varying
coefficient are intentionally **not** built in v1 — the marginal-slope warps
described below absorb PGS-by-covariate departures from linearity. This is
encoded in [`construction.rs`](construction.rs).

### Likelihoods

- **Probit (binary)** for `phenotype ∈ {0,1}`. Even if `LinkFunction::Logit` is
  configured, the base link used inside the family is hard-coded to probit
  ([`estimate.rs`](estimate.rs)).
- **Identity (Gaussian)** for continuous `phenotype`.

### Binary path — Bernoulli marginal-slope

A two-step fit that produces a calibrated probit index whose location and
slope are both covariate-dependent, plus two cubic warps:

1. **CTN prefit.** The PGS column is treated as a continuous response and fit
   with a `TransformationNormal` (GAMLSS-style: smooth `T` and smooth
   `log σ`, both conditional on `sex` linear + each `PC_j` Duchon smooth) so
   that the fitted η acts as a covariate-adjusted latent normal score `z` per
   row. This replaces the discrete phenotype in the score-warp step (the CTN
   warp itself needs a continuous response).
2. **De-nested cubic transport kernel** (`gam/src/families/cubic_cell_kernel.rs`):
   ```
   η(z, x) = a(x) + b(x)·z + b(x)·δ_h(z) + δ_w(a(x) + b(x)·z)
   P(Y=1 | x) = Φ( η(z, x) )                 (· / √(1+σ²) if Gaussian-shift frailty)
   ```
   where
   - `a(x)` is the marginal location above,
   - `log b(x) = f_logslope(PGS)` (an 8-center Duchon smooth on PGS only, from
     [`build_logslope_termspec`](construction.rs)),
   - `δ_h(z)` is the **score-warp** cubic spline deviation block in `z`
     (`DeviationBlockConfig::triple_penalty_default`),
   - `δ_w(·)` is the **link-wiggle** cubic spline deviation block applied to
     the affine core `a + b·z` (same triple-penalty default).

   The "de-nested" name is literal: the kernel is the additive correction
   `b·δ_h(z) + δ_w(a+b·z)` around the affine core, not the nested composition
   `L(a + b·H(z))`.

### Identity path — Gaussian location-scale (GAMLSS)

A `GaussianLocationScaleFitRequest` from `gam::families::gamlss`: a
distributional regression in which **both** the conditional mean and the
conditional log scale are smoothed jointly. The fitted model is

```
y | x  ~  N( μ(x), σ(x)² )
μ(x)      = β₀^μ + f_pgs^μ(PGS) + γ_sex^μ·sex + Σ_j f_j^μ(PC_j)
log σ(x)  = β₀^σ + f_pgs^σ(PGS) + γ_sex^σ·sex + Σ_j f_j^σ(PC_j)
```

Each `f_*` is again a Duchon RBF smooth (same `build_marginal_termspec`
machinery as the binary path) and each `γ_sex^*` is a doubly-penalized
linear term. The two channels are fit jointly under their own GAMLSS
likelihood — there is no CTN prefit (a continuous response can drive the
scale channel directly) and no marginal-slope transport kernel.

A cubic triple-penalty **link wiggle** (`WigglePenaltyConfig::cubic_triple_operator_default`)
is enabled by default and seeded from a no-wiggle pilot fit; its knots,
degree, and coefficients are persisted so prediction reproduces the warp.
PIRLS handles the inner Newton step; outer BFGS on `log λ` maximizes REML
across the joint `(μ, log σ, wiggle)` parameter blocks.

### Survival path — Survival marginal-slope

Same marginal + log-slope + score-warp + link-wiggle as the binary path, plus
a `time_block` (`build_time_block_input` in [`survival.rs`](survival.rs)),
optional `timewiggle_block`, and a derivative guard. Outcome is the
`(age_entry, age_exit, event_target)` triple; base link is probit; PGS warp is
again seeded from a CTN prefit on PGS conditional on sex + PCs.

## Penalties and smoothing selection

Each Duchon smooth carries a default operator penalty (`DuchonOperatorPenaltySpec::default()`)
respecting the linear nullspace, so intercepts and linear-in-PGS / linear-in-PC
components remain unpenalized by construction. The sex linear term is fitted
with `double_penalty = true`. The score-warp and link-wiggle blocks each use
`WigglePenaltyConfig::cubic_triple_operator_default` — a triple-penalty cubic
spline (multiple operator orders, double-penalty on, monotonicity epsilon).

Smoothing parameters (`λ`) are learned rather than fixed. The solver engine
implements a nested optimization à la Wood (2011): inner PIRLS for fixed `λ`,
outer BFGS on marginal likelihood — **REML** for Gaussian fits, **LAML** for
non-Gaussian (binary, survival). Both objectives include stabilization priors
and null-space accounting. This is **empirical Bayes**: hyperparameters are
estimated from the data via marginal likelihood, then coefficients are
inferred conditional on those point estimates.

At biobank scale the binary/survival families optionally use a stratified
Horvitz–Thompson subsample of outer rows for the first phase of BFGS
iterations (`auto_outer_subsample`), reverting to full data for the polish
phase so `outer_tol` is reached on exact gradients.

## Optimization strategy

The P-IRLS solver iteratively forms working responses and weights, solves the
penalized normal equations with the `faer` linear algebra backend, and checks
for hazards such as separation or ill-conditioning. Transformed Hessians, trace
corrections, and effective degrees of freedom are cached so the outer REML/LAML
optimizer can evaluate gradients efficiently. The final Hessian, effective
penalty factors, and fitted scale (when applicable) are preserved inside the
[`TrainedModel`](model.rs) artifact for downstream uncertainty estimates.

## Uncertainty estimation

The stored penalized Hessian enables standard error estimation at prediction time
via the delta method: `Var(η) = x' H⁻¹ x`. However, these intervals have important
limitations:

**Smoothing bias**: Penalized splines systematically flatten peaks, fill valleys,
and round corners. The Hessian-based SE captures _parameter uncertainty_ but not
_smoothing-induced bias_. At extremes of the predictor space (high/low PGS, rare
ancestries), the confidence interval may be centered incorrectly and under-cover.

**Conditional vs. unconditional**: The current implementation computes the
_conditional_ variance treating spline coefficients as fixed parameters. The
_unconditional_ approach (averaging over the prior on coefficients) would give
wider intervals but is "too large where bias is small and too small where bias
is large" (Nychka 1988). Neither approach is perfect.

**Practical guidance**: Treat SEs as approximate. They are most reliable in
smooth regions of the predictor space with dense training data. For clinical
use, a measure of proximity to training support (e.g. the peeled-hull distance
implemented in `gam::terms::hull` — not yet wired through this adapter) may be
more informative than the SE magnitude.

**Point estimate choice (mode vs. mean)**: The current implementation returns the
posterior mode (MAP estimate from PIRLS). For risk predictions ("you have 13%
chance of X"), the posterior mean is theoretically preferable because it minimizes
Brier score / squared prediction error. If MCMC sampling were added post-BFGS,
the posterior mean of the risk (averaging f(patient, β) over β samples) would
give more accurate calibrated probabilities than the mode. The mode answers "what's
the single most probable β?" while the mean answers "what risk should I report to
minimize prediction error on average?" For patient-facing risk estimates, the mean
is the Bayes-optimal choice.

See Ruppert, Wand, Carroll "Semiparametric Regression" Ch. 6.6-6.9 for theoretical
background on confidence intervals for penalized splines.

## What lives where

Adapter/domain files in `gnomon/calibrate`:
- [`data.rs`](data.rs) and [`survival_data.rs`](survival_data.rs): file/schema
  policy, ingestion, domain validation, and training bundles.
- [`construction.rs`](construction.rs): `TermCollectionSpec` builders for the
  marginal collection (`build_marginal_termspec`) and the PGS log-slope
  channel (`build_logslope_termspec`), plus the `duchon_smooth` helper.
- [`estimate.rs`](estimate.rs): `train_model` / `train_survival_model` thin
  adapters over `gam::fit_model`. Picks the family (Standard / Bernoulli
  marginal-slope / Survival marginal-slope), runs the CTN prefit when a
  latent `z` is needed, and wraps the result in a `FittedModelPayload`.
- [`survival.rs`](survival.rs): time-block and time-wiggle-block builders for
  the survival family.
- [`model.rs`](model.rs): `ModelConfig`, `TrainedModel`, and serde composition
  for gnomon artifacts.

Engine-owned modules in the separate `gam` crate:
- `families/bernoulli_marginal_slope.rs`, `families/survival_marginal_slope.rs`,
  `families/transformation_normal.rs`, `families/cubic_cell_kernel.rs`,
  `families/marginal_slope_shared.rs`, `families/row_kernel.rs`.
- `smooth/*` (Duchon basis, term-collection design freeze, anisotropic length-scale opt),
  `pirls`, `estimate::reml::unified`, `custom_family`, `probability`.

## Training flow at a glance

1. **Load and validate data** – `data::load_training_data` reads the TSV with
   Polars, verifies column types (including the binary `sex` column), enforces
   the minimum-row requirement, and returns `TrainingData` (phenotype, score,
   sex, PCs, weights — defaulting to ones if the column is absent).
2. **Build the column matrix and term specs** – `estimate.rs::build_training_matrix`
   lays out columns as `phenotype | pgs | sex | pc1..pck | weights`. Then
   `construction::build_marginal_termspec` assembles the marginal Duchon-smooth
   collection and, for the binary/survival families,
   `construction::build_logslope_termspec` adds the PGS log-slope smooth.
3. **CTN prefit (binary and survival only)** – `ctn_prefit_latent_z` fits a
   `TransformationNormal` of PGS conditional on sex + PC smooths and returns
   the per-row latent normal score `z` used as the calibrated input to the
   marginal-slope kernel.
4. **Fit the family** – `gam::fit_model` is called with one of
   `FitRequest::GaussianLocationScale` (Identity link),
   `FitRequest::BernoulliMarginalSlope` (Probit/Logit), or
   `FitRequest::SurvivalMarginalSlope` (survival). The engine alternates PIRLS
   (inner) and BFGS over `log λ` (outer), maximizing REML (Gaussian) or LAML
   (binary/survival). Score-warp, link-wiggle, and link-wiggle deviation
   blocks are seeded from a no-wiggle pilot fit before the joint refit.
5. **Freeze and persist** – `freeze_term_collection_from_design` snapshots the
   resolved term-spec + design (knots, transforms, lambdas) into a
   `FittedModelPayload`; this is wrapped in `TrainedModel` along with the
   `ModelConfig` so prediction can reconstruct the exact bases.

## Prediction path

Prediction reconstructs the resolved term-collection bases from the stored
knot vectors and transformation matrices, then:

- **Identity** — evaluates `μ(x)` and `log σ(x)` from the mean and noise
  term collections (the engine's `GaussianLocationScalePredictor`), applies
  the stored cubic link wiggle, and reports a Gaussian density.
- **Binary/Survival** — evaluates `a(x)` and `b(x)`, applies the cubic
  transport kernel `η = a + b·z + b·δ_h(z) + δ_w(a+b·z)` using the stored
  `DeviationRuntime` blocks, and maps to probabilities via `Φ(·)` (with the
  `1/√(1+σ²)` rescale if Gaussian-shift frailty was fitted).

### Posterior-predictive uncertainty (sketch)

The stored penalized Hessian already encodes local curvature around the fitted
coefficients. Treating the coefficients as approximately
`β ~ Normal(β̂, H⁻¹)` yields a lightweight posterior predictive routine:

1. Compute a Cholesky factor of `H⁻¹` after training (or factor `H` and solve
   for draws on demand).
2. At inference, draw `β⁽¹⁾…β⁽M⁾` from that multivariate normal.
3. For a new design vector `x`, evaluate `η⁽ᵐ⁾ = x'β⁽ᵐ⁾` and transform with
   the link (e.g., `p⁽ᵐ⁾ = sigmoid(η⁽ᵐ⁾)` for logistic fits).
4. Use the empirical quantiles of `{p⁽ᵐ⁾}` as credible intervals; the samples
   themselves represent the full distribution of the individual's risk.

This adds on the order of 50–100 lines of inference code (sampling, linkage,
quantiles) and requires no access to the training data—only the fitted
coefficients and Hessian. It inherits the standard large-sample assumptions of a
Gaussian posterior around the optimum and ignores higher-order asymmetry.

## Expected data format

Training and inference both operate on tab-separated value (TSV) files with a
header row and strictly named columns. The loader surfaces actionable errors if
any of the schema requirements below are violated.

### Training inputs

`data::load_training_data` expects the following columns:

- `phenotype` – numeric response (0/1 for probit fits, real-valued for
  Gaussian fits). Missing values are not permitted.
- `score` – the standardized polygenic score used as the primary smooth.
- `sex` – binary indicator encoded as 0/1. Other encodings are rejected during
  type coercion.
- `PC1`, `PC2`, …, `PCk` – one column per requested principal component. The
  number of PCs must match the `num_pcs` configuration supplied to the CLI or
  library entry point. Columns are required even if they are all zeros.
- `weights` (optional) – positive prior weights. When omitted, the loader
  supplies a length-`n` vector of ones so unweighted fits do not require a
  synthetic column.
- `sample_id` (optional) – string identifiers passed through to the serialized
  model for auditing. Absent IDs are replaced with deterministic `1`, `2`, …
  labels.

All required columns must be finite, and at least 20 rows are recommended for a
stable fit. The loader prints the resolved schema so callers can confirm the
exact set of covariates that entered the design.

### Inference inputs

`data::load_prediction_data` enforces the same structure for prediction-time
files, minus the response column:

- `score`, `sex`, and the `PC*` columns must be present and numeric.
- Optional `sample_id` values are used to label rows in the prediction outputs
  (filling with `1`, `2`, … when absent).

Weights and phenotypes are ignored during inference. Prediction data is
validated with the same finite-value checks as training data, ensuring that
the deployed spline bases receive well-formed covariates.

## Repo split note

Path ownership is intentionally split:
- Adapter/domain layer: `gnomon/calibrate`
- Math/solver engine: separate solver repository

Contract summary:
- `gnomon/calibrate` performs domain layout assembly and passes full-size `P x P`
  penalties and numeric arrays into the solver engine.
- The solver engine performs PIRLS/REML/HMC/ALO and returns fit outputs.
- Public call flow remains adapter-stable via `gnomon::calibrate::*` entrypoints.
