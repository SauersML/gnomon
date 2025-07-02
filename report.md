# GetDoc Report - Tue, 1 Jul 2025 22:28:57 -0500

## Compiler Output (Errors and Warnings)

### Diagnostics for: default features

```text
WARNING: dead_code: warning: variant `EigendecompositionFailed` is never constructed
  --> src/../calibrate/estimate.rs:51:5
   |
43 | pub enum EstimationError {
   |          --------------- variant in this enum
...
51 |     EigendecompositionFailed(ndarray_linalg::error::LinalgError),
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: `EstimationError` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis
   = note: `#[warn(dead_code)]` on by default
    (Diagnostic primary location: src/../calibrate/estimate.rs:51)
WARNING: dead_code: warning: field `final_weights` is never read
   --> src/../calibrate/estimate.rs:182:20
    |
178 |     pub(super) struct PirlsResult {
    |                       ----------- field in this struct
...
182 |         pub(super) final_weights: Array1<f64>,
    |                    ^^^^^^^^^^^^^
    |
    = note: `PirlsResult` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis
    (Diagnostic primary location: src/../calibrate/estimate.rs:182)
WARNING: dead_code: warning: function `main` is never used
  --> src/../score/main.rs:57:4
   |
57 | fn main() {
   |    ^^^^
    (Diagnostic primary location: src/../score/main.rs:57)
WARNING: dead_code: warning: function `run_gnomon` is never used
  --> src/../score/main.rs:84:4
   |
84 | fn run_gnomon() -> Result<(), Box<dyn Error + Send + Sync>> {
   |    ^^^^^^^^^^
    (Diagnostic primary location: src/../score/main.rs:84)
```

### Diagnostics for: --no-default-features

```text
WARNING: dead_code: warning: variant `EigendecompositionFailed` is never constructed
  --> src/../calibrate/estimate.rs:51:5
   |
43 | pub enum EstimationError {
   |          --------------- variant in this enum
...
51 |     EigendecompositionFailed(ndarray_linalg::error::LinalgError),
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: `EstimationError` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis
   = note: `#[warn(dead_code)]` on by default
    (Diagnostic primary location: src/../calibrate/estimate.rs:51)
WARNING: dead_code: warning: field `final_weights` is never read
   --> src/../calibrate/estimate.rs:182:20
    |
178 |     pub(super) struct PirlsResult {
    |                       ----------- field in this struct
...
182 |         pub(super) final_weights: Array1<f64>,
    |                    ^^^^^^^^^^^^^
    |
    = note: `PirlsResult` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis
    (Diagnostic primary location: src/../calibrate/estimate.rs:182)
WARNING: dead_code: warning: function `main` is never used
  --> src/../score/main.rs:57:4
   |
57 | fn main() {
   |    ^^^^
    (Diagnostic primary location: src/../score/main.rs:57)
WARNING: dead_code: warning: function `run_gnomon` is never used
  --> src/../score/main.rs:84:4
   |
84 | fn run_gnomon() -> Result<(), Box<dyn Error + Send + Sync>> {
   |    ^^^^^^^^^^
    (Diagnostic primary location: src/../score/main.rs:84)
```

### Diagnostics for: --no-default-features --features no-inline-profiling

```text
WARNING: dead_code: warning: variant `EigendecompositionFailed` is never constructed
  --> src/../calibrate/estimate.rs:51:5
   |
43 | pub enum EstimationError {
   |          --------------- variant in this enum
...
51 |     EigendecompositionFailed(ndarray_linalg::error::LinalgError),
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: `EstimationError` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis
   = note: `#[warn(dead_code)]` on by default
    (Diagnostic primary location: src/../calibrate/estimate.rs:51)
WARNING: dead_code: warning: field `final_weights` is never read
   --> src/../calibrate/estimate.rs:182:20
    |
178 |     pub(super) struct PirlsResult {
    |                       ----------- field in this struct
...
182 |         pub(super) final_weights: Array1<f64>,
    |                    ^^^^^^^^^^^^^
    |
    = note: `PirlsResult` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis
    (Diagnostic primary location: src/../calibrate/estimate.rs:182)
WARNING: dead_code: warning: function `main` is never used
  --> src/../score/main.rs:57:4
   |
57 | fn main() {
   |    ^^^^
    (Diagnostic primary location: src/../score/main.rs:57)
WARNING: dead_code: warning: function `run_gnomon` is never used
  --> src/../score/main.rs:84:4
   |
84 | fn run_gnomon() -> Result<(), Box<dyn Error + Send + Sync>> {
   |    ^^^^^^^^^^
    (Diagnostic primary location: src/../score/main.rs:84)
```

### Diagnostics for: --all-features

```text
WARNING: dead_code: warning: variant `EigendecompositionFailed` is never constructed
  --> src/../calibrate/estimate.rs:51:5
   |
43 | pub enum EstimationError {
   |          --------------- variant in this enum
...
51 |     EigendecompositionFailed(ndarray_linalg::error::LinalgError),
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
   |
   = note: `EstimationError` has a derived impl for the trait `Debug`, but this is intentionally ignored during dead code analysis
   = note: `#[warn(dead_code)]` on by default
    (Diagnostic primary location: src/../calibrate/estimate.rs:51)
WARNING: dead_code: warning: field `final_weights` is never read
   --> src/../calibrate/estimate.rs:182:20
    |
178 |     pub(super) struct PirlsResult {
    |                       ----------- field in this struct
...
182 |         pub(super) final_weights: Array1<f64>,
    |                    ^^^^^^^^^^^^^
    |
    = note: `PirlsResult` has a derived impl for the trait `Clone`, but this is intentionally ignored during dead code analysis
    (Diagnostic primary location: src/../calibrate/estimate.rs:182)
WARNING: dead_code: warning: function `main` is never used
  --> src/../score/main.rs:57:4
   |
57 | fn main() {
   |    ^^^^
    (Diagnostic primary location: src/../score/main.rs:57)
WARNING: dead_code: warning: function `run_gnomon` is never used
  --> src/../score/main.rs:84:4
   |
84 | fn run_gnomon() -> Result<(), Box<dyn Error + Send + Sync>> {
   |    ^^^^^^^^^^
    (Diagnostic primary location: src/../score/main.rs:84)
```

No third-party crate information extracted (or no third-party files were implicated).
