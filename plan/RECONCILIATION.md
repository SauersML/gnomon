# Survival Model Implementation: Reconciliation of Conflicting Plans

## Executive Summary

Four survival integration plans exist with **mutually incompatible designs**. This document identifies contradictions and provides a unified recommendation.

## Critical Contradictions

### 1. Data Schema (BLOCKING)

| Plan | Age Fields | Event Representation | Weights |
|------|-----------|---------------------|---------|
| 48271 | `age_entry`, `age_exit` | Two f64 columns: `event_target`, `event_competing` | `censoring_weights`, `prior_weights` |
| 59213 | Undefined | Undefined | Undefined |
| 72953 | `age_entry`, `age_exit` | Single u8 column: `event_type` (0/1/2) | `weights` |
| 8362 | `age_start`, `age_stop` | Two f64 columns: `event`, `competing` | `weights` |

**Impact**: Cannot implement without choosing one schema.

### 2. Risk-Set Handling (BLOCKING)

**48271**: "no separate risk-set pruning is required" - pure per-subject full likelihood

**59213**: Requires `calibrate::survival::fine_gray` module with:
- Kaplan-Meier censoring survivals `G(t)`
- Event-time slices
- At-risk indices per event time
- Normalized weights `w_i G_i(t_k)`
- Risk-set traversals

**Impact**: Fundamentally different computational approaches.

### 3. Numerical Integration (BLOCKING)

**48271 & 59213 (section 7.2)**: "no numerical quadrature is required" - cumulative hazard is `exp(η_exit) - exp(η_entry)`

**72953**: Requires Gauss-Kronrod quadrature:
```
Evaluate η(t) and ∂η/∂t for a Gauss–Kronrod grid spanning [current_age, horizon_age]
Integrate the subdistribution hazard h(t) = H(t) ⋅ ∂η/∂t over the grid
```

**59213 (section 3.3)**: "Store optional Gauss–Kronrod weights if higher-order integration becomes necessary"

**Impact**: Contradictory requirements for prediction.

### 4. API Architecture (BLOCKING)

| Plan | Approach |
|------|----------|
| 48271 | `enum ModelFamily { Gam(LinkFunction), Survival(SurvivalSpec) }` |
| 59213 | `enum LinkFunction { Logit, Identity, RoystonParmarSurvival(...) }` |
| 72953 | `enum LinkFunction { Logit, Identity, FineGrayRp }` + `enum ModelFamily::SurvivalFineGray` |
| 8362 | `enum ModelFamily { Standard(LinkFunction), Survival(SurvivalSpec) }` |

**Impact**: Four different top-level API designs.

### 5. PIRLS Integration (BLOCKING)

**48271**: Replace existing PIRLS interface with trait returning dense gradients/Hessians

**59213**: Boxed `LikelihoodFamily` trait tied to `LinkFunction`

**8362**: `PirlsFamily` trait with explicit initialization hooks

**Impact**: Three different trait designs for the same purpose.

## Theoretical Foundation (web_search_results.md)

All plans claim to implement "Royston-Parmar Fine-Gray with full likelihood" based on:

1. **Parametric baseline**: `H_0^*(a) = exp(f_0(log(a)))` with B-spline `f_0`
2. **Full likelihood**: Per-subject contributions, not partial likelihood
3. **Fine-Gray subdistribution**: Models target event hazard, competing events contribute survival term only

**Key insight from literature**: With parametric baseline, you CAN write full likelihood without risk sets.

## Recommended Resolution

### Decision: Follow 48271 Architecture with Clarifications

**Rationale**:
1. ✅ Pure full likelihood (no risk-set machinery) - simplest, most efficient
2. ✅ `ModelFamily` separation - cleanest API boundary
3. ✅ Explicit about "no quadrature needed" for cumulative hazard
4. ✅ Already partially merged (section 4.2 cleanup completed)

### Unified Specification

#### Data Schema
```rust
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,      // Left truncation age
    pub age_exit: Array1<f64>,       // Event/censoring age
    pub event_target: Array1<f64>,   // 0/1: target event indicator
    pub event_competing: Array1<f64>, // 0/1: competing event indicator
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    pub weights: Array1<f64>,        // Sample weights (default 1.0)
}
```

**Validation**: `event_target` and `event_competing` mutually exclusive (both can be 0 for censoring).

#### Likelihood (Full, No Risk Sets)

Per-subject contribution:
```
ℓ_i = w_i [δ_target log λ_i^*(a_exit) - (H_i^*(a_exit) - H_i^*(a_entry))]
```

Where:
- `λ_i^*(a) = H_i^*(a) · (∂η_i/∂a)(a)` (subdistribution hazard)
- `H_i^*(a) = exp(η_i(a))` (cumulative subdistribution hazard)
- `η_i(a) = f_0(log(a)) + x_i^T β + g_{pgs}(PGS_i, log(a))`

**Censored observations** (`δ_target = 0`, `δ_competing = 0`):
- Contribute only survival term: `-w_i (H_i^*(a_exit) - H_i^*(a_entry))`

**Competing events** (`δ_competing = 1`):
- Same as censored: contribute survival term only
- This is Fine-Gray semantics: competing events treated as censored for target hazard

#### Cumulative Hazard Computation (No Quadrature)

**Training**: Cache at entry/exit ages
```
H_i^*(a_entry) = exp(η_i(a_entry))
H_i^*(a_exit) = exp(η_i(a_exit))
ΔH_i = H_i^*(a_exit) - H_i^*(a_entry)
```

**Prediction**: For horizon `h` from current age `a_0`
```
H_i^*(a_0) = exp(η_i(a_0))
H_i^*(a_0 + h) = exp(η_i(a_0 + h))
CIF(a_0, h) = 1 - exp(-(H_i^*(a_0 + h) - H_i^*(a_0)))
```

**No numerical integration needed** because:
1. Parametric baseline allows direct evaluation at any age
2. Cumulative hazard is exponential of linear predictor
3. Difference of exponentials is exact

#### API Architecture
```rust
pub enum ModelFamily {
    Gam(LinkFunction),
    Survival(SurvivalSpec),
}

pub struct SurvivalSpec {
    pub age_basis: BasisConfig,
    pub num_age_knots: usize,
    pub time_varying_basis: Option<BasisConfig>,
    pub competing_events_present: bool,
}
```

#### PIRLS Integration

Extend with trait:
```rust
pub trait ModelLikelihood {
    fn compute_working_quantities(&self, eta: &Array1<f64>) 
        -> (Array1<f64>, Array2<f64>, f64); // (score, hessian, deviance)
}
```

Implementations:
- `LogisticLikelihood` (existing, diagonal Hessian)
- `GaussianLikelihood` (existing, diagonal Hessian)
- `SurvivalLikelihood` (new, dense Hessian from per-subject contributions)

## What to Discard

### From 59213
- ❌ Risk-set construction module
- ❌ Kaplan-Meier weight computation
- ❌ Event-time slices
- ❌ `LinkFunction::RoystonParmarSurvival` variant

**Why**: Full likelihood doesn't need risk sets. This is leftover partial likelihood machinery.

### From 72953
- ❌ Gauss-Kronrod quadrature for predictions
- ❌ Single `event_type` column
- ❌ Both `LinkFunction::FineGrayRp` AND `ModelFamily::SurvivalFineGray`

**Why**: Quadrature unnecessary with parametric baseline. Dual API confusing.

### From 8362
- ❌ `age_start`/`age_stop` naming
- ❌ `AnalysisMode` enum
- ❌ `PirlsFamily` trait with initialization hooks

**Why**: Inconsistent naming. Overly complex trait design.

## Implementation Priority

1. **Data schema** (48271): Implement `SurvivalTrainingData` with dual event indicators
2. **Likelihood** (48271): Per-subject full likelihood, no risk sets
3. **Basis caching** (48271): Entry/exit design matrices with derivatives
4. **PIRLS trait** (48271): `ModelLikelihood` with dense Hessian support
5. **Prediction** (48271): Direct evaluation, no quadrature
6. **Calibration** (48271): Integrate with existing calibrator

## Open Questions

1. **Censoring weights**: 48271 includes optional `censoring_weights` field. Purpose unclear if not using IPCW. Recommend: Remove or clarify as "sample weights only".

2. **Competing risk modeling**: Current approach treats competing events as censored for target hazard. This is Fine-Gray semantics but doesn't model competing hazard. Document limitation.

3. **Extrapolation**: All plans silent on predicting beyond `max(age_exit)`. Recommend: Add age range guards and warning.

## Testing Strategy

1. **Unit tests**: Verify likelihood matches manual calculation on toy data
2. **Integration tests**: Compare to R `rstpm2` on published datasets
3. **Regression tests**: Ensure logistic/Gaussian paths unchanged
4. **Edge cases**: All censored, all competing, heavy left truncation

## References

- web_search_results.md: Theoretical justification for full likelihood approach
- Royston & Parmar (2002): Flexible parametric survival models
- Fine & Gray (1999): Subdistribution hazards for competing risks
