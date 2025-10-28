# Poisson Quadrature Cross-Check (Support Document)

This document records how we will use Poisson pseudo-observations to validate the Fine–Gray RP implementation. It is *not* an independent production path; see [`survival_72953.md`](./survival_72953.md) for the canonical model.

## 1. Objective
Provide a numerical approximation to the full Fine–Gray likelihood by discretising age into small intervals and fitting a Poisson regression with offsets that encode cumulative exposure. Matching coefficients and smoothing behaviour between this approximation and the analytic implementation offers an end-to-end regression test (following the strategy of Bender et al., 2005, and Crowther & Lambert, 2014, Section 3.5).

## 2. Construction
1. Choose quadrature nodes `a_i = entry`, `b_i = exit`, and intermediate points via Gauss–Legendre or equally spaced bins (5–7 per subject is typically sufficient for validation datasets).
2. For each interval `[t_{q-1}, t_q)` create a pseudo-row with:
   - Exposure `E_{iq} = H_i(t_q) - H_i(t_{q-1})` evaluated using the current RP coefficients.
   - Outcome `y_{iq}` equal to 1 only if the target event occurs in that interval (Fine–Gray keeps competing events in the risk set, so they contribute zero counts with positive exposure).
   - Offset `log(E_{iq})` and covariates evaluated at the interval midpoint.
3. Fit a Poisson log-link GAM using the existing infrastructure. Because exposure is already encoded in the offset, the Poisson score matches the survival score up to discretisation error.

## 3. Relationship to the canonical IRLS
- As the quadrature grid is refined, the Poisson working weights converge to `w_i H_i(t)` from the analytic formulas in `survival_48271.md`.
- Left truncation is handled by omitting intervals before `entry` and using the same exposure subtraction as the continuous model.
- Time-varying effects (PGS×age) are evaluated at quadrature nodes, ensuring the approximation respects their smooth structure.

## 4. Usage plan
- Implement an optional debug mode (`--survival-validate-poisson`) that, after fitting the analytic model, generates pseudo-rows and refits the Poisson approximation with the *same* smoothing parameters. We compare coefficients, fitted CIFs, and score vectors.
- Differences beyond tolerance trigger diagnostics but do not block production runs; the Poisson route exists solely to catch coding errors in the analytic derivatives.

## 5. Computational considerations
- Even at 7 nodes per subject, a million-subject cohort yields ≈7 million pseudo-rows—manageable for offline validation but not for routine training. Hence the feature is opt-in.
- Memory reuse: allocate buffers for one batch of subjects at a time to avoid quadratic growth.

This support path guarantees that the analytic Fine–Gray implementation is auditable and reproducible without reintroducing the flawed partial-likelihood formulation.
