# Calibration crate

The `calibrate/` crate hosts the end-to-end pipeline that fits Gnomon's penalized
additive model and its optional post-hoc calibrator. The code in this directory
is responsible for ingesting the tabular training data, constructing spline
bases and penalties, solving the penalized iteratively reweighted least squares
(P-IRLS) updates with REML smoothing selection, and (when enabled) training a
secondary calibrator that sharpens probability calibration or identity-scale
accuracy.

## What lives where

- [`data.rs`](data.rs) reads TSV inputs with Polars, enforces the strict schema
  (`phenotype`, `score`, `PC*`, `weights`), and produces the `TrainingData`
  bundle consumed by the optimizer. The same module also validates prediction
  inputs and generates stable sample identifiers.
- [`basis.rs`](basis.rs) builds B-spline bases, applies sum-to-zero constraints,
  and constructs difference penalties that ultimately drive smoothness control
  for both the primary model and the calibrator.
- [`construction.rs`](construction.rs) converts configuration into concrete
  design matrices and penalty layouts. It tracks how every block of columns maps
  to penalties and interaction terms so the downstream optimizer can assemble
  the correct objective.
- [`pirls.rs`](pirls.rs) implements the weighted least squares inner loop shared
  by the base fit and the calibrator. It accepts arbitrary designs along with
  penalty metadata produced by `ModelLayout`.
- [`estimate.rs`](estimate.rs) is the orchestrator. `train_model` triggers design
  construction, runs REML/BFGS over smoothing parameters, guards against
  degeneracy, and optionally launches the calibrator fitting routine.
- [`hull.rs`](hull.rs) derives a peeled convex hull around the training domain
  and provides signed-distance queries that the calibrator uses to detect and
  damp extrapolation.
- [`calibrator.rs`](calibrator.rs) converts base-model diagnostics into
  calibrator features, builds its own constrained design, fits smoothing
  parameters, and exposes inference helpers for applying the learned correction.
- [`model.rs`](model.rs) defines the serialized `TrainedModel`, handles
  persistence, and applies both the baseline GAM and the optional calibrator at
  prediction time.
- [`faer_ndarray.rs`](faer_ndarray.rs) bridges `ndarray` structures with the
  `faer` linear algebra backend for eigen decompositions and solves, which keeps
  large penalized systems numerically stable.

## Training flow at a glance

1. **Load and validate data** – Callers invoke `data::load_training_data`, which
   reads the TSV file lazily with Polars, verifies column types, enforces the
   minimum-row requirement, and returns `TrainingData` containing the phenotype,
   scores, PCs, and sample weights.
2. **Construct the base GAM** – `estimate::train_model` delegates to
   `construction::build_design_and_penalty_matrices`. This step selects knot
   vectors via `basis`, assembles the block-structured design matrix `X`, and
   pairs each block with its penalty matrices. The resulting `ModelLayout`
   tracks intercept, PC main effects, tensor-product interactions, and
   null-space bases.
3. **Optimize smoothing parameters** – With the design in place, the REML
   optimizer in `estimate.rs` alternates between P-IRLS solves (via `pirls.rs`)
   and BFGS updates over log-smoothing parameters until convergence or until it
   detects degeneracy (separation, rank deficiency, etc.).
4. **Capture geometric guards** – During training the optimizer builds a peeled
   hull from the polygenic score and principal components. The hull and its
   signed-distance function inform both the calibrator and future prediction
   time clamping.
5. **Optional calibrator fitting** – If `model::calibrator_enabled()` returns
   true, `estimate::train_model` computes approximate leave-one-out (ALO)
   diagnostics from the converged fit using `calibrator::compute_alo_features`.
   Those diagnostics (baseline predictor, its standard error, and hull distance)
   feed into `calibrator::build_calibrator_design`, which mirrors the spline
   machinery to create a smaller additive model. REML smoothing selection is
   reused through `calibrator::fit_calibrator`, yielding coefficients, block
   lambdas, and (for identity models) a residual scale estimate.
6. **Persist the result** – The fitted coefficients, smoothing parameters, hull,
   and calibrator (if present) are serialized through `model::TrainedModel` so
   that downstream tools can reload them exactly.

## Prediction path

`ModelConfig::predict` reconstructs the required spline bases using the stored
knot vectors and transformation matrices, evaluates the base GAM, and, when a
calibrator is attached, calls `calibrator::predict_calibrator` to adjust the
linear predictor. Logistic models return calibrated probabilities; identity
models add the calibrator's correction in linear space. Distance-to-hull checks
run in the same order as training, ensuring extrapolation handling remains
consistent.
