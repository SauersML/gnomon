# Royston–Parmar Fine–Gray Survival Blueprint (Canonical)

## 1. Overview
We will implement Fine–Gray subdistribution hazards inside the existing penalized GAM stack by treating the Royston–Parmar (RP) spline as a **fully parametric baseline**. Consequently we maximise the **full log-likelihood** of the Fine–Gray subdistribution model, rather than the Cox-style partial likelihood. This aligns with published RP implementations for competing risks (Crowther & Lambert, 2014; Crowther, 2023) and avoids hybrids that mix risk-set algebra with parametric baselines.

Key implications:
- The log-likelihood uses the event density `log h_i(b_i)` and the cumulative survival `log S_i(b_i)` with explicit hazard derivatives.
- Observations contribute independently once left truncation is accounted for, yielding a diagonal working weight matrix and an O(n) Hessian accumulation.
- Time-varying effects (e.g., PGS×age) are handled by evaluating both the basis values and their derivatives with respect to log-age.

## 2. Model specification
- **Time scale:** attained age with optional left truncation. Store `(a_min, δ)` and transform ages via `u = log(a - a_min + δ)` to stabilise spline evaluation. The B-spline basis and its derivative are well-defined on ℝ, so negative `u` values pose no issue.
- **Baseline:** model `ℓ_0(u) = log H_0(a)` with a B-spline basis `B(u)`; enforce monotonicity through the exponential map and, if necessary, an increasing-spline constraint (P-spline penalty with inequality guard as in Crowther & Lambert, 2014).
- **Linear predictor:**
  ```
  η_i(u) = ℓ_0(u) + x_i^⊤β + f_i(u)
  H_i(u) = exp(η_i(u))
  ```
  where `f_i(u)` collects optional time-varying smooths (e.g., tensor-product PGS×age). Their derivatives `∂f_i/∂u` are precomputed alongside basis evaluations.
- **Hazard:**
  ```
  h_i(u) = H_i(u) * (∂η_i/∂u) / exp(u)
  ```
  where `∂η_i/∂u = B'(u)θ + ∂f_i/∂u`.
- **Left truncation:** subtract the cumulative hazard accumulated before entry: the likelihood uses `H_i(b_i) - H_i(a_i)` and the log-survival contribution includes both terms.

## 3. Log-likelihood and derivatives
For observation *i* with weight `w_i`, entry age `a_i`, exit age `b_i`, event indicator `d_i`, and competing-event indicator `c_i` (1 only if a competing event occurs at `b_i`):
```
ℓ_i = w_i [ d_i log h_i(b_i) + (1 - d_i) log S_i(b_i) ]
      - w_i [ c_i log S_i(b_i) + log S_i(a_i) ]
```
Here `S_i(u) = exp(-H_i(u))`. Competing events only affect the survival component, consistent with Fine–Gray.

Gradients:
- Score with respect to `η_i(b_i)` is `w_i [ d_i (1 + ∂ log(∂η_i/∂u)/∂η_i) - H_i(b_i) ]`.
- Score with respect to `η_i(a_i)` (when left-truncated) is `+ w_i H_i(a_i)`.
- Derivatives propagate through design matrices exactly as in other GAM families; chain-rule terms from `∂η/∂u` are assembled using cached derivative bases.

Weights:
- The negative Hessian of `ℓ_i` with respect to `η_i(b_i)` is `w_i [ d_i * ( (∂η_i/∂u)^{-2} * (∂η_i/∂u)^2 + H_i(b_i) ) + H_i(b_i) ]`, which simplifies numerically to `w_i * H_i(b_i)` when `d_i ∈ {0,1}`. Left-truncation adds `+ w_i H_i(a_i)` to the diagonal. Therefore the working weight matrix remains **diagonal**. We cache `W_i = w_i [H_i(b_i) + 1_{a_i< b_i} H_i(a_i)]` and use it in IRLS and REML.
- The working response is `z_i = η_i + U_i / W_i`, avoiding matrix inversions.

These expressions match the derivations in Crowther & Lambert (2014, Appendix A) and Royston & Parmar (2002) once specialised to the Fine–Gray subdistribution hazard.

## 4. Numerical implementation plan
1. **Design matrices:**
   - `X_exit` and `X_entry` contain baseline, covariate, and smooth terms evaluated at `u_exit` and `u_entry` respectively.
   - `X'_exit` and `X'_entry` store derivative evaluations for the subset of columns that depend on time (baseline spline, time-varying smooths).
2. **IRLS loop:**
   - Compute `η_exit = X_exit β`, `η_entry = X_entry β`, and derivatives `∂η/∂u` via matrix products with derivative design blocks.
   - Form `H_exit = exp(η_exit)`, `H_entry = exp(η_entry)`, hazard derivatives, and working weights `W` as above.
   - Score vector combines exit and entry pieces using accumulated contributions per coefficient block; Hessian is `X_exit^⊤ diag(W_exit) X_exit + X_entry^⊤ diag(W_entry) X_entry + Σ_penalty λ P`.
   - Solve the penalised system via existing conjugate-gradient / Cholesky routines. Complexity remains O(np) with dense X, but we exploit sparsity from B-spline blocks to reach near-linear scaling.
3. **REML:** uses the diagonal `W` when forming the penalised log determinant and derivatives with respect to smoothing parameters. Because `W` is diagonal, existing REML machinery extends directly; we only pass the updated weights and score.
4. **Monotonicity guard:** enforce `∂η/∂u > 0` at quadrature nodes by adding a barrier penalty if necessary. This is a numerical safeguard rather than a modelling assumption; it prevents negative hazards from floating-point drift.
5. **Numerical integration:** prediction of CIF over intervals uses adaptive Gauss–Kronrod integration of the subdistribution hazard to account for time-varying effects. When covariate effects are proportional, the integral reduces to `1 - exp(-(H_exit - H_entry))`, but we fall back to quadrature whenever `∂f_i/∂u ≠ 0`.

## 5. Relationship to Poisson pseudo-observations
The Poisson GLM strategy in `survival_59213.md` discretises age into quadrature nodes and fits a log-link Poisson regression on piecewise-constant hazards. This is a **numerical approximation** to the same full-likelihood described here. With sufficiently fine quadrature, the estimates converge to the RP fit; we retain the Poisson path only for validation and as a fallback when the analytic derivative code needs cross-checking (cf. Bender et al., 2005).

## 6. Conditional CIF and competing events
- The conditional CIF between ages `t0` and `t1` is computed using the **renormalised** formula:
  ```
  CIF_cond = (CIF_target(t1) - CIF_target(t0)) /
             max(ε, 1 - CIF_target(t0) - CIF_competing(t0))
  ```
  where `CIF_target(t) = 1 - exp(-H_i(t))` from the RP subdistribution model and `CIF_competing(t)` is obtained by fitting analogous RP Fine–Gray models for each competing event type. We do **not** mix in Kaplan–Meier estimates.
- Store metadata identifying which baseline corresponds to which event type so that prediction routines can retrieve the correct competing-risk models.

## 7. Computational feasibility
- Diagonal weights avoid the O(n²) accumulation that doomed the partial-likelihood draft. Each IRLS iteration performs a handful of dense matrix–vector multiplies (already optimised in the GAM core) and vectorised element-wise operations.
- For million-scale cohorts, we batch-evaluate design blocks to fit in cache. Penalised linear solves leverage sparse Cholesky with banded penalties; runtime grows roughly linearly in sample size.

## 8. Smoothing parameter derivatives
- With diagonal `W`, the REML derivatives reuse the existing `trace(W^{-1} dW/dλ)` formulas. `dW/dλ` only depends on penalty matrices through the coefficient updates, exactly as in current logistic/Gaussian code. No new algebra is required beyond exposing `H_exit` and `H_entry` to the REML workspace.

## 9. Prediction workflow summary
1. Evaluate baseline and covariate design vectors at requested ages (entry/current and horizon), including derivative bases when time-varying effects are active.
2. Form `η`, `H`, and `∂η/∂u`.
3. Integrate the subdistribution hazard over requested intervals (adaptive quadrature when necessary).
4. Combine target and competing CIFs, renormalise for conditional probabilities, and apply calibrator if configured.

## 10. References
- Bender, R., Augustin, T., & Blettner, M. (2005). *Generating survival times to simulate Cox proportional hazards models*. Statistics in Medicine, 24(11), 1713–1723.
- Crowther, M. J., & Lambert, P. C. (2014). *Parametric modelling of survival data in the presence of competing risks: flexible parametric hazard models*. Statistics in Medicine, 33(4), 469–488.
- Crowther, M. J. (2023). *Multistate models and competing risks with flexible parametric survival models*. Stata Journal, 23(3), 589–629.
- Royston, P., & Parmar, M. K. B. (2002). *Flexible parametric proportional hazards and proportional odds models for censored survival data, with application to prognostic modelling for patients with cancer*. Statistics in Medicine, 21(15), 2175–2197.

## 11. Responses to outstanding review questions
1. `log λ_i^*` is the hazard derivative term `log(∂η/∂u)` required by the full Fine–Gray density. The partial-likelihood draft erroneously dropped it; the canonical plan keeps it.
2. We use the full likelihood `Σ[log h_i + log S_i]`, consistent with RP Fine–Gray literature. The partial likelihood was discarded.
3. Literature (Crowther & Lambert, 2014; Crowther, 2023) explicitly fits RP Fine–Gray models via full likelihood. No hybrid with partial likelihood is proposed.
4. The Poisson GLM plan approximates the same full-likelihood integral via quadrature; it is a verification tool, not a separate model.
5. Hazard derivatives are required because `h_i(u)` depends on the derivative of the log cumulative hazard; we compute them explicitly.
6. With the full likelihood, the working weights remain diagonal. The non-diagonal Hessian was only necessary for the discarded partial-likelihood approach.
7. Left truncation adds *positive* curvature `+H_i(a_i)`; the earlier subtraction was incorrect and has been removed.
8. Both entry and exit contributions are positive curvature terms; the minus sign in the deprecated draft stemmed from the mistaken partial-likelihood algebra.
9. The diagonal Hessian enables O(n) updates, so biobank-scale fits are feasible; no dense O(n²) matrix is formed.
10. `z = η + U / W` applies element-wise because `W` is diagonal; no linear-system solve is required per observation.
11. All prediction code will use the renormalised conditional CIF formula described in Section 6; the simplistic exponential difference is confined to the special case without time-varying effects.
12. We fit a separate RP Fine–Gray model for each competing event type so `CIF_competing` comes from the same modelling framework.
13. Time-varying effects trigger quadrature of the hazard over the interval; the constant-hazard shortcut is only used when `∂f_i/∂u = 0`.
14. Predictions integrate the hazard over `[current, horizon]`, evaluating the interaction across the interval rather than freezing it at `horizon_age`.
15. Baseline objects are tagged with the event type they represent (`event_type_id`) so prediction logic retrieves the correct subdistribution hazard.
16. The derivative safeguard is purely numerical: we prevent negative hazards caused by floating-point noise while allowing arbitrarily small positive hazards.
17. The log-age shift keeps arguments within the spline domain; B-splines handle negative inputs, and the guard `δ` avoids evaluating log at zero.
18. We cache design matrices and derivative matrices at entry and exit ages for both the baseline and any time-varying smooths.
19. REML derivatives reuse the diagonal-weight formulas; no special handling of `∂H/∂λ` is needed beyond providing `W` to the existing code.
20. The RP full-likelihood scales linearly with sample size and avoids duplicating rows; the Poisson pseudo-observation route is reserved for diagnostics and convergence checks.
