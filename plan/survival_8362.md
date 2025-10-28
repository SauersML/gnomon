# Royston–Parmar Fine–Gray Plan (Archived Draft)

This file records that the earlier partial-likelihood draft has been superseded. All implementation work must follow the canonical blueprint in [`survival_72953.md`](./survival_72953.md).

Key differences between the archived draft and the canonical plan:

1. **Likelihood.** We now maximise the full Fine–Gray likelihood, including the hazard-derivative term `log(∂η/∂a)` and the Jacobian of the log-age transform. The partial-likelihood algebra previously sketched here is incorrect for a parametric RP baseline.
2. **Curvature.** The canonical plan retains the rank-one Hessian corrections that arise from the hazard derivative and handles them with damped Newton solvers. Treating the weights as diagonal, as in the draft, would misrepresent the curvature.
3. **Prediction.** Conditional CIFs use quadrature over the fitted hazard so that time-varying effects are integrated rather than frozen at the horizon age.
4. **Competing risks.** Each cause is modelled with its own RP Fine–Gray fit; no Kaplan–Meier mixing is permitted.
5. **Validation.** Poisson pseudo-observations serve only as a regression test and are derived from the same full likelihood.

The draft content is intentionally absent to prevent the outdated approach from resurfacing.
