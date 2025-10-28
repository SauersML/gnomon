# Poisson Quadrature Cross-Check (Support Document)

This note explains how we use Poisson pseudo-observations to **validate** the analytic Fine–Gray RP implementation described in `survival_72953.md`. It is not a production alternative.

## 1. Construction of pseudo-observations
1. Partition each subject’s age interval `[a_i, b_i]` into quadrature nodes `a_i = t_{i0} < t_{i1} < … < t_{iQ} = b_i`. Gauss–Legendre nodes offer higher accuracy, but equally spaced grids (5–7 points) suffice for regression tests.
2. For each interval `(t_{iq-1}, t_{iq}]`, build a pseudo-row with
   - Exposure `E_{iq} = H_i(t_{iq}) - H_i(t_{iq-1})` using the current RP coefficients.
   - Linear predictor `η_{iq}` evaluated at the midpoint age, including time-varying effects.
   - Response `y_{iq} = 1` only if the target event occurs in the interval; otherwise `y_{iq} = 0`.
   - Offset `log(E_{iq})` so that the Poisson mean satisfies `μ_{iq} = E_{iq} exp(η_{iq})`.
3. Fit a Poisson GAM with the same smoothing penalties as the analytic model. The quadrature grid handles left truncation automatically by omitting intervals below the entry age.

## 2. Relationship to the analytic likelihood
- Summing the Poisson log-likelihood across pseudo-rows approximates the continuous-time likelihood `Σ_i w_i [ d_i log h_i - H_i(b_i) + H_i(a_i) ]` with a Riemann sum. As the grid refines, the Poisson score converges to the analytic score in `survival_48271.md`.
- The hazard derivative term appears implicitly: differentiating the offset-adjusted Poisson log-likelihood produces the same `D_exit / (∂η/∂a)` correction found analytically.
- Because the quadrature path multiplies the row count by `Q`, it is computationally heavier than the analytic fit and is therefore restricted to validation workflows.

## 3. Validation workflow
1. Fit the analytic Fine–Gray RP model.
2. Generate pseudo-observations with the fitted coefficients and re-fit the Poisson approximation while **freezing** smoothing parameters.
3. Compare:
   - Score vectors from both fits (should agree within numerical tolerance).
   - Predicted CIFs on a validation grid of ages and covariate patterns.
   - Penalised log-likelihood values.
4. Raise diagnostics if discrepancies exceed tolerance; otherwise record the match to document derivative correctness.

## 4. Computational guidance
- Materialise pseudo-rows in batches to limit memory usage; discard each batch after evaluating the Poisson score.
- Parallelise across subjects because pseudo-row generation is embarrassingly parallel.
- Use the same monotonicity guard for `∂η/∂a` as in the analytic fit to avoid numerical differences caused by clipping.

This cross-check ties the quadrature-based intuition to the analytic derivatives without reviving the discredited partial-likelihood draft.
