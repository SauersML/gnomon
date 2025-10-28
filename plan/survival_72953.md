# Royston–Parmar Fine–Gray Survival Blueprint (Canonical)

## 1. Why the full Fine–Gray likelihood
Fine–Gray subdistribution hazards describe the density of experiencing the target event in the presence of competing events. For a subject who enters the study at age `a_i` and exits at `b_i`, the continuous-time likelihood contribution is

```
ℓ_i = log f_i(b_i) - log S_i(a_i)
```

where `f_i` is the Fine–Gray event density for the target cause and `S_i` is the corresponding subdistribution survival function. Because Royston–Parmar (RP) models parameterise the **log cumulative hazard** `η_i(a) = log H_i(a)` with splines, the event density expands to

```
log f_i(b_i) = log h_i(b_i) + log S_i(b_i) = η_i(b_i) + log(∂H_i/∂a|_{b_i}) - log J(b_i) - H_i(b_i)
```

with `J(b_i)` representing the Jacobian of the time transformation detailed in Section 2. The term `log(∂H_i/∂a)` is the hazard derivative that earlier drafts labelled `log λ_i^*`. Removing it would drop the normalising constant of the density and violate the Fine–Gray likelihood. Consequently, we abandon the Cox-style partial likelihood and maximise the **full log-likelihood**

```
ℓ = Σ_i w_i [ d_i ( η_i(b_i) + log(∂η_i/∂a|_{b_i}) - log J(b_i) ) - H_i(b_i) + H_i(a_i) ]
```

where `d_i` is one for target events, zero otherwise, and `w_i` encodes sampling weights. This expression matches the derivation in Crowther & Lambert (2014, Appendix A) once translated to attained-age time and left truncation. It also aligns with the Stata implementation distributed with Crowther (2023). Any hybrid that mixes risk sets with a parametric baseline would double-count the Jacobian and break the curvature structure, which is why PR #540 was rejected.

## 2. Parameterisation and bases
- **Time scale.** We store a minimum attainable age `a_min` and a positive shift `δ`. Ages are mapped to `u = log(a - a_min + δ)` so that the spline basis receives well-scaled inputs even when `a ≈ a_min`. The Jacobian in Section 1 is `J(a) = a - a_min + δ`.
- **Log cumulative hazard.** The baseline is expressed as `ℓ_0(u) = B(u)θ`, where `B(u)` is a B-spline basis. Penalised differences of `θ` enforce smoothness, while monotonicity is obtained by exponentiating `η = ℓ_0 + covariate contributions` and adding a barrier if numerical derivatives threaten negativity.
- **Covariates.** Static effects enter linearly through `x_i^⊤β`. Time-varying effects (e.g., PGS×age) use tensor-product smooths `f_i(u)` with precomputed derivative bases `B_f'(u)`.
- **Hazard.** Differentiating the linear predictor yields
  ```
  ∂η_i/∂a = (∂η_i/∂u) * (∂u/∂a) = (B'(u)θ + ∂f_i/∂u) / (a - a_min + δ)
  h_i(a) = H_i(a) * ∂η_i/∂a
  ```
  Both `η` and its derivative are linear in the spline coefficients, so we can accumulate them efficiently.

## 3. Gradients and curvature
### 3.1 Score contributions
Let `X_exit[i, :]` be the design row for `η_i(b_i)` and `D_exit[i, :]` the design row for `∂η_i/∂a` (including the Jacobian). The score for coefficient vector `β̃` (collecting baseline and covariate parameters) is

```
U_i^{exit} = w_i [ d_i ( X_exit[i,:] + D_exit[i,:] / (∂η_i/∂a|_{b_i}) ) - H_i(b_i) X_exit[i,:] ]
U_i^{entry} = + w_i H_i(a_i) X_entry[i,:]
```

where `X_entry` mirrors `X_exit` at the entry age. Competing events have `d_i = 0` and therefore contribute only the survival term `-H_i(b_i) X_exit[i,:]`; this mirrors right-censoring in ordinary survival analysis and requires no special term.

### 3.2 Hessian structure
The observed negative Hessian decomposes into three parts:

1. **Exit survival term.** `H_exit = Σ_i w_i H_i(b_i) X_exit[i,:]^⊤ X_exit[i,:]` (positive semi-definite).
2. **Entry survival term.** `H_entry = Σ_i w_i H_i(a_i) X_entry[i,:]^⊤ X_entry[i,:]` (positive semi-definite, subtracted because `ℓ` increases with `H_i(a_i)`).
3. **Hazard derivative correction.** For each event (`d_i = 1`),
   ```
   H_i^{deriv} = w_i * (D_exit[i,:]^⊤ D_exit[i,:]) / (∂η_i/∂a|_{b_i})^2
   ```
   which is a rank-one, positive semi-definite matrix.

The total **negative** Hessian is therefore

```
-∂²ℓ = H_exit - H_entry + Σ_{i:d_i=1} H_i^{deriv} + Σ_k λ_k P_k
```

The first two terms arise from the exponential map and mirror standard Poisson/PH curvature. The derivative correction is new but sparse: it only touches event rows and has rank one per event. We accumulate it by storing `D_exit` for the event set and applying batched outer products. When left truncation is rare, `H_entry` is small; when it is common we include it explicitly and rely on the penalised trust-region solver (Section 4) to handle the indefinite directions that left truncation can introduce.

### 3.3 Working response
Because the Hessian is no longer diagonal, the classic scalar IRLS update `z = η + U/W` does not apply. Instead we solve the penalised Newton step

```
( -∂²ℓ ) δ = U
β̃_{new} = β̃_{old} + δ
```

using a damped Cholesky or preconditioned conjugate gradient routine. This matches the strategy already used in our GAM stack for non-canonical links. The low-rank derivative corrections are applied via Sherman–Morrison updates so that each Newton step remains `O(n p)` despite the non-diagonal structure.

## 4. Numerical solution strategy
1. **Caching.** Store `η_exit`, `η_entry`, `H_exit`, `H_entry`, `∂η/∂a` and design rows (`X`, `D`) in contiguous memory. Event row indices are tracked separately for the derivative corrections.
2. **Gradient/Hessian assembly.** Compute `U` and the three curvature components as described in Section 3.2. The derivative term uses fused multiply–adds to accumulate the outer products efficiently.
3. **Solver.** Start from penalised iteratively reweighted least squares with a line search. When the assembled Hessian is not positive definite (which can occur under heavy left truncation), apply Levenberg–Marquardt damping by adding `γ diag(P)` until positive definiteness is restored.
4. **Smoothing parameter updates.** REML/LAML optimisation uses the same assembled Hessian; the derivative of the penalised deviance with respect to smoothing parameters requires traces of products involving `-∂²ℓ`. We reuse the existing mgcv-style spd solve with Hutchinson trace estimation, now fed by the accurate Hessian rather than a diagonal proxy.
5. **Convergence checks.** Monitor the maximum absolute score and the relative change in log-likelihood; fall back to line search if the Newton step increases the penalised deviance.

## 5. Prediction and CIF computation
To predict cumulative incidence between ages `t0` and `t1`:

1. Evaluate `η(t)` and `∂η/∂a` on an adaptive Gauss–Kronrod grid; this respects continuously varying PGS×age effects.
2. Integrate the subdistribution hazard `h(t) = H(t) ∂η/∂a` across the grid to obtain `H(t1) - H(t0)`; exponentiate to recover `CIF_target(t) = 1 - exp(-H(t))`.
3. Combine with competing subdistribution hazards fitted analogously and renormalise via
   ```
   CIF_cond(t1 | t0) = (CIF_target(t1) - CIF_target(t0)) /
                       max(ε, 1 - CIF_target(t0) - CIF_competing(t0))
   ```
4. When no time-varying effects are present and the hazard ratio is constant, the integral collapses to the familiar difference `1 - exp(-(H(t1) - H(t0)))`; otherwise the quadrature path is mandatory.

## 6. Relationship to Poisson pseudo-observations
`survival_59213.md` details a Poisson regression built from quadrature nodes `(t_{q-1}, t_q]`. The pseudo-row log-likelihood is `y_{iq} log μ_{iq} - μ_{iq}` with `μ_{iq} = E_{iq} exp(x_{iq}^⊤ β̃)` and exposure `E_{iq} = H(t_q) - H(t_{q-1})`. As the grid refines, the Poisson score converges to the analytic score derived above. This path therefore serves as a regression test for the analytic derivatives and confirms that we have not invented a hybrid likelihood.

## 7. Data structures and metadata
- Store baseline coefficients alongside `(event_type_id, a_min, δ, spline_basis_descriptor)` so that prediction code can locate the appropriate subdistribution hazard.
- Cache `X_exit`, `X_entry`, and derivative blocks `D_exit`, `D_entry` to avoid recomputation when evaluating time-varying effects.
- Guard `∂η/∂a` against underflow by clipping at `1e-12` before taking logarithms; this is purely a numerical stabiliser and does not impose a modelling floor on the hazard.

## 8. Computational feasibility
- Exit and entry survival terms are diagonal in observation space and cost `O(n p)` per iteration.
- Derivative corrections are rank-one per event; for biobank cohorts with millions of individuals but far fewer events per iteration, the accumulated cost remains manageable. We exploit sparsity by batching events and using BLAS-3 updates.
- Memory use is governed by storing two dense design matrices (exit and entry) plus their derivative counterparts; we stream them in blocks when `n` is very large.

## 9. Literature support
- Crowther, M. J., & Lambert, P. C. (2014). *Parametric modelling of survival data in the presence of competing risks: flexible parametric hazard models*. Statistics in Medicine, 33(4), 469–488.
- Crowther, M. J. (2023). *Multistate models and competing risks with flexible parametric survival models*. Stata Journal, 23(3), 589–629.
- Royston, P., & Parmar, M. K. B. (2002). *Flexible parametric proportional hazards and proportional odds models for censored survival data*. Statistics in Medicine, 21(15), 2175–2197.
- Beyersmann, J., Latouche, A., Buchholz, A., & Schumacher, M. (2009). *Cause-specific versus subdistribution hazard models in competing risks: what do we need to know?* Statistics in Medicine, 28(7), 1089–1107. (Confirms that IPCW weighting belongs to Cox-style partial likelihood, not to fully parametric RP models.)

## 10. Detailed responses to the 20 review questions
1. **Hazard derivative vs. `log λ_i^*`.** In the full likelihood the event density is `h_i(b_i) S_i(b_i)`. Because `h_i = H_i * ∂η_i/∂a`, the logarithm naturally contains `log(∂η_i/∂a)`—identical to the earlier `log λ_i^*`. Omitting it corresponds to integrating the density with respect to the wrong measure.
2. **Partial vs. full likelihood.** RP models are fully parametric and therefore must use the complete density. Crowther & Lambert (2014) explicitly derive the log-likelihood we adopt; there is no published evidence supporting a Cox-style partial likelihood with an RP baseline.
3. **Literature support.** The references in Section 9 implement Fine–Gray with full likelihoods. Beyersmann et al. (2009) show that IPCW weighting is a device for Cox partial likelihoods; since we abandon the partial likelihood, IPCW weighting is not used here.
4. **Poisson pseudo-observations.** The Poisson cross-check approximates the same integrals via quadrature. With a sufficiently fine grid the coefficients converge to those from the analytic fit, so the two plans are numerically equivalent.
5. **Hazard derivatives.** Computing `h_i` requires `∂η_i/∂a`; prediction and fitting therefore evaluate derivatives explicitly. Earlier claims that we could avoid derivatives were incorrect and have been removed.
6. **Hessian structure.** The survival terms are diagonal in observation space, but the hazard-derivative term introduces rank-one updates per event. We retain these low-rank corrections; they are the price of using the correct likelihood.
7. **Left truncation weights.** `H_i(a_i)` enters with a positive sign in the log-likelihood, so its curvature contribution subtracts from the negative Hessian. We model it explicitly instead of writing exit-minus-entry heuristics.
8. **Entry curvature sign.** Both entry and exit hazards are positive; the minus sign in the obsolete draft arose from mixing partial-likelihood algebra with left truncation. The canonical plan keeps their true signs inside the Newton system.
9. **Scalability of the full Hessian.** Event counts are much smaller than cohort size, so the low-rank updates add only `O(p^2 * n_events)` work. We batch them and apply Sherman–Morrison updates to stay within linear-time scaling in practice.
10. **Working response.** Because the Hessian is not diagonal, we no longer use scalar `U/W` updates. Instead we solve the Newton system directly, just as we do for other non-canonical GAM families.
11. **Conditional CIF formula.** The renormalised expression in Section 5 is the default for all plans. The exponential difference is recognised as the proportional-hazard special case and implemented only when time-varying effects are absent.
12. **Competing events.** Each competing cause receives its own RP Fine–Gray fit, so `CIF_competing` comes from the same modelling framework rather than from Kaplan–Meier estimators.
13. **Time-varying effects in prediction.** Adaptive quadrature over `[t0, t1]` integrates `h(t)` with `∂η/∂a` evaluated at every node, so interactions that vary with age are respected exactly.
14. **Evaluating interactions.** Predictions do not freeze the effect at `t1`; the quadrature grid samples the entire interval so the interaction is effectively integrated.
15. **Baseline storage.** Baseline splines are bundled with metadata specifying the event type, the `(a_min, δ)` transform, and the derivative basis so that prediction code cannot mix up cause-specific baselines.
16. **Derivative safeguard.** The `∂η/∂a > 0` guard prevents numerical sign flips caused by floating-point error. It does not impose a lower bound on the true hazard, which may be arbitrarily small.
17. **Log-age shift.** B-splines are defined on ℝ, so negative `u` values are acceptable. The `δ` shift prevents evaluating `log 0` at the lower age boundary, satisfying the regularity conditions in Royston & Parmar (2002).
18. **Design matrices to cache.** We store both the value and derivative design matrices at entry and exit ages for every smooth that depends on time. This ensures we can form scores and Hessians without repeated spline evaluation.
19. **REML derivatives.** Because we assemble the full Hessian, REML/LAML updates naturally differentiate through it. The low-rank derivative corrections are included when computing the trace terms required by Wood-style smoothing parameter updates.
20. **Computational comparison with Poisson GLM.** The analytic full likelihood evaluates each subject once per iteration. The Poisson approximation multiplies the dataset by the number of quadrature nodes, so it is orders of magnitude slower at biobank scale and therefore relegated to validation.
