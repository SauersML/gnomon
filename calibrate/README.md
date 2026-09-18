# Calibration adapter crate

`gnomon/calibrate` is the domain adapter layer for Gnomon's calibration and
survival workflows. It owns schema/data policy, PGS/PC/sex feature semantics,
artifact mapping, and stable `gnomon::calibrate::*` entrypoints.

Training writes `model.json`, an atomic, versioned bundle containing the model
configuration and gam's saved model: the fitted predictor, the feature schema
and ranges, and, for a marginal-slope model, the latent law the fit anchored
on. Phenotype and sample weights are never prediction features. Bundles written
by earlier versions require refitting.

Core numerical engine modules (basis construction, PIRLS/REML, the
marginal-slope kernels, persistence, prediction, and shared math types) live in
the separate solver engine repository and are imported by this crate. The
marginal-slope models train through gam's formula route (`fit_formula_to_payload`),
the path gam's CLI and gamfit use, so the saved model is the one those tools save
for the same fit. The Gaussian location-scale model is requested directly,
because the formula route refuses a link wiggle on a non-binomial family; its
saved model is gam's own location-scale assembly
(`assemble_location_scale_payload`).

## Statistical model

The crate dispatches three model families from `estimate.rs`. They share one
context skeleton over sex and the principal components:

```
c(x) = β₀ + γ_sex·sex + f(PC₁, …, PC_k)
```

`f` is one joint Duchon smooth over the leading `k` principal components
(`s(PC1, …, PCk, type=duchon, centers=m)`), never a smooth per component, and
`γ_sex` is a penalized linear term. The kernel is gam's scale-free structural
default: no length scale, an affine null space (`k + 1` columns) and spectral
power `s = (k − 1)/2`, the kernel `r³` in every dimension, which satisfies
Duchon's existence condition `2(p + s) > k` (`p = 2`) at every `k`. The formula
text, and the equivalent term specifications of the Gaussian model, are
assembled in [`construction.rs`](construction.rs).

`PcSmoothConfig::for_pcs(k)` derives the joint smooth's center counts from
`k`: `⌈3k/2⌉` in the context and `⌈5k/4⌉` in the slope (9 and 8 at `k = 6`,
24 and 20 at `k = 16`), never fewer than `k + 2`, since gam needs more centers
than the null space has columns. The smooth works for any `k`; the examples use
6 PCs, which fit much faster than 16. A configuration with too few centers, or with an explicit
power at which the kernel does not exist, is refused before any fit.
`--pgs-centers` (at least four) sizes the score smooth of the Gaussian model;
the marginal-slope models do not smooth the score as a covariate, because the
score is their latent coordinate. A saved bundle from before the joint smooth
(format version 2, one smooth per PC) is refused by name at load; retrain it.

### Binary path — Bernoulli marginal-slope

For `phenotype ∈ {0,1}` (every value exactly 0 or 1), recorded as
`LinkFunction::Probit`: the base link is probit, and a configuration naming another
binary link is refused. With `z` the score, the model is

```
P(Y=1 | x, z) = Φ(η(x, z)),   η(x, z) = α(x) + b(x)·z + b(x)·δ_h(z) + δ_w(α(x) + b(x)·z)
```

where `q(x)` is the marginal index over the context formula, `b(x)` the slope
over `1 + f(PC₁, …, PC_k)` (its own joint smooth), `δ_h` the score-warp and `δ_w` the link-deviation
cubic blocks (`linkwiggle()` on the slope and marginal formulas), and `α(x)` is
defined by the anchoring equation on the declared law `{(u_k, w_k)}` of the
score:

```
Σ_k w_k Φ(η(x, u_k)) = Φ(q(x))
```

The score enters as given; no transform of it is fitted. `ModelConfig::latent_law`
(CLI `--latent-law`) declares its law:

- `empirical` (the default): the weighted empirical law of the training rows'
  scores, the same rows, weights and eligibility the outcome model is fitted
  on. gam compresses it to at most 65 equal-mass nodes and standardizes it to
  mean 0 and sd 1. It is pooled over the context; whether it is adequate within
  a context stratum is a diagnostic, not an assumption.
- `standard-normal`: an explicit declaration that the score is already standard
  normal given the context, the case in which the anchor has the closed form
  `α = q·√(1 + b²)`. It is a declared special case, never the target of a
  transform.

The saved model persists the declared law and prediction replays it.

### Identity path — Gaussian location-scale (GAMLSS)

For a continuous `phenotype`, a distributional regression in which the
conditional mean and the conditional log scale are smoothed jointly:

```
y | x  ~  N( μ(x), σ(x)² )
μ(x)      = f_score^μ(score) + c^μ(x)
log σ(x)  = f_score^σ(score) + c^σ(x)
```

`f_score` is a Duchon smooth of the score (gam's default kernel, `--pgs-centers`
centers) and `c` the context skeleton above;
both channels share the same terms, and gam's cubic triple-penalty link wiggle
lets the mean flex away from a strict additive form. gam's formula route
refuses `linkwiggle()` for a non-binomial family, so the fit is a direct
`GaussianLocationScaleFitRequest` over the term specifications from
`construction.rs`. gam standardizes the response for the fit; the saved model
records that scale and the link wiggle, so prediction reproduces both.

### Survival path — Survival marginal-slope

The outcome is `Surv(age_entry, age_exit, event_target) ~ sex + f(PC₁, …, PC_k)`,
the slope formula is `1 + f(PC₁, …, PC_k)`, and the score is the latent coordinate,
declared exactly as on the binary path. The base link is probit.

gam's I-spline time basis carries the baseline at gam's default degree and knot
count unless `SurvivalModelConfig.baseline_knots` / `baseline_degree` (CLI
`--survival-baseline-knots` / `--survival-baseline-degree`) name them; the
`--survival-time-wiggle-*` options likewise fall back to gam's `timewiggle()`
defaults. The baseline
starts from unit-shape Weibull offsets at the mean exit age. gam chooses the time
anchor: marginal-slope centres the time basis at the median exit age, because an
earliest-entry anchor on delayed-entry ages inflates the unpenalized time column
until every smoothing seed is refused (gam #751). gam persists the anchor, knots and
offsets. Under the empirical law gam anchors the index on a rigid baseline:
there are no score-warp or link-deviation blocks, and a baseline time wiggle is
refused before fitting. `--survival-time-wiggle*` therefore requires
`--latent-law standard-normal`.

Survival calibration accepts score, sex, and the configured PCs. Extra static
covariates are rejected because the prediction API has no corresponding inputs.
Competing events are censoring events for this net, cause-specific model, so a
survival prediction's `net_risk_entry` and `net_risk_exit` are `1 − exp(−H)` of
the target event's cause-specific hazard: net risk, the risk under independent
censoring by the competing event, not the cumulative incidence that the competing
event lowers (#2384).

## Penalties and smoothing selection

Each Duchon smooth carries gam's default operator penalty, respecting its
linear nullspace, so intercepts and linear components remain unpenalized by
construction. The score-warp and link-deviation blocks use gam's cubic
triple-operator default (multiple operator orders, double penalty, monotonicity
epsilon).

Smoothing parameters (`λ`) are learned rather than fixed. The solver engine
implements a nested optimization à la Wood (2011): inner PIRLS for fixed `λ`,
outer optimization on marginal likelihood — **REML** for Gaussian fits, **LAML**
for non-Gaussian (binary, survival). Both objectives include stabilization
priors and null-space accounting. This is **empirical Bayes**: hyperparameters
are estimated from the data via marginal likelihood, then coefficients are
inferred conditional on those point estimates. gnomon passes no outer bounds of
its own: the outer loop runs under gam's defaults (60 iterations at 1e-5 for the
Gaussian location-scale fit, gam's own certificates on the marginal-slope formula
route), and the inner solver runs to gam's own certificates. No smooth carries a
length scale, so no spatial length-scale search runs, and calibrate sets no
iteration cap or tolerance anywhere.

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
- [`construction.rs`](construction.rs): column names and the formula text for
  the context and slope formulas.
- [`estimate.rs`](estimate.rs): `train_model` / `train_survival_model`. Builds
  the named training table and gam's `FitConfig`, picks the family (Gaussian
  location-scale / Bernoulli marginal-slope / survival marginal-slope) and the
  declared latent law, fits through `fit_formula_to_payload`, and records the
  predictor-only schema.
- [`survival.rs`](survival.rs): survival data types and input validation.
- [`model.rs`](model.rs): `ModelConfig`, `LatentLaw`, `TrainedModel`, prediction,
  and serde composition for gnomon artifacts.

Engine-owned modules live in the separate `gam` workspace (`gam-models`:
`bms`, `survival::marginal_slope`, `gamlss`, `fit_orchestration`, `inference`;
`gam-terms`: Duchon bases and formula parsing; `gam-predict`: predictors).

## Training flow at a glance

1. **Load and validate data** – `data::load_training_data` reads the TSV with
   Polars, verifies column types (including the binary `sex` column), enforces
   the minimum-row requirement, and returns `TrainingData` (phenotype, score,
   sex, PCs, weights — defaulting to ones if the column is absent).
2. **Build the table and formulas** – `estimate.rs` lays out the predictor
   columns as `score | sex | PC1..PCk`, followed by the outcome, time and weight
   columns the formulas name, and `construction.rs` writes the context and
   slope formulas.
3. **Fit** – `fit_formula_to_payload` resolves the formula and `FitConfig`
   (family, slope formula, `z_column = score`, the declared `latent_measure`,
   the outer-loop bounds) and fits the family. The engine alternates PIRLS
   (inner) and optimization over `log λ` (outer), maximizing REML (Gaussian) or
   LAML (binary/survival).
4. **Persist** – gam assembles the `FittedModelPayload` (frozen bases, lambdas,
   coefficients, declared latent law, survival time metadata); gnomon records
   the predictor-only schema and wraps it in `TrainedModel` with the
   `ModelConfig`.

## Prediction path

Prediction builds the predictor matrix in the saved header order and asks gam's
predictor for the saved model:

- **Identity** — evaluates `μ(x)` and `log σ(x)` from the mean and noise
  term collections, applies the stored link wiggle, and reports the mean.
- **Binary** — evaluates `q(x)` and `b(x)`, solves the anchor on the saved
  latent law, applies the stored deviation blocks, and maps to probabilities
  via `Φ(·)`.
- **Survival** — evaluates the plug-in cumulative hazard at each row's entry
  and exit ages on one coefficient vector, and reports the conditional risk
  `1 − exp(−(H(exit) − H(entry)))`.

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
- `score` – the polygenic score: the latent coordinate of a marginal-slope fit
  and a smooth covariate of a Gaussian fit.
- `sex` – binary indicator encoded as 0/1. Any other value (a 1/2 PLINK
  coding, or 0 for unknown beside 1/2) is rejected when the table is loaded,
  for training and for prediction alike.
- `PC1`, `PC2`, …, `PCk` – one column per requested principal component. The
  number of PCs must match the `num_pcs` configuration supplied to the CLI or
  library entry point. Columns are required even if they are all zeros.
- `weights` (optional) – positive prior weights. When omitted, the loader
  supplies a length-`n` vector of ones so unweighted fits do not require a
  synthetic column.
- `sample_id` (optional) – string identifiers. Training ignores the column; the
  model stores no per-row identifiers. Prediction copies it to the output table,
  and rows without one are labelled with deterministic `1`, `2`, … labels.

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
- `gnomon/calibrate` names the columns, writes the formulas and declares the
  latent law; it passes a named table and a `FitConfig` to the solver engine.
- The solver engine performs basis construction, PIRLS/REML and persistence,
  and returns the saved model.
- Public call flow remains adapter-stable via `gnomon::calibrate::*` entrypoints.
