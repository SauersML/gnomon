# Identity-link Gradient Check Failure: Root Cause Analysis

## Failure symptoms
* The unit test `calibrate::estimate::optimizer_progress_tests::test_optimizer_makes_progress_from_initial_guess_identity` aborts during the mandatory gradient check.  The finite-difference guard reports cosine similarity 0.998735 and relative L2 error 7.19×10⁻², while all masked components pass individually, so the test panics even though the component-wise comparison looks clean.【00cd65†L1-L25】【9a19c3†L1-L24】
* The single-direction sanity probe logged in the same gradient check shows the analytic directional derivative disagreeing with the numerical secant by roughly 4.2×10⁻², confirming the global discrepancy is not an artifact of masking.【d32d95†L1-L18】

## What the cost and gradient compute
* For the identity link, the REML cost used by the optimizer is
  \[
  V(\rho) = \frac{D_p(\rho)}{2\phi(\rho)} + \tfrac{1}{2}\log|H(\rho)| - \tfrac{1}{2}\log|S_\lambda(\rho)|_+ + \frac{n-M_p}{2}\log\big(2\pi\,\phi(\rho)\big),
  \]
  where the profile scale \(\phi\) is **explicitly defined as** \(\phi(\rho)=D_p(\rho)/(n-\mathrm{edf}(\rho))\).  Both the deviance term and the log term therefore inherit the \(\rho\)-dependence of \(\phi\).【5ac4a7†L83-L136】
* The analytic gradient routine, however, intentionally applies an envelope-theorem shortcut: it differentiates the deviance, log|H|, and log|S| pieces while **holding the profiled \(\phi\) fixed**, explicitly stating that “no dφ/dρ term is propagated.”  The final assembled gradient is just
  \[
  \frac{\lambda_k}{2\phi}\beta^\top S_k\beta + \frac{\lambda_k}{2}\operatorname{tr}\big(H^{-1}S_k\big) - \frac{1}{2}\operatorname{det1}_k,
  \]
  with no contribution from the \(\log(2\pi\phi)\) factor or from the \(\phi\)-dependence of the deviance term.【9481d1†L1-L88】

## Why the mismatch happens
* The REML score we actually evaluate already substitutes \(\phi(\rho)=D_p/(n-\mathrm{edf})\); that substitute is **not** the stationary point of the un-profiled objective when \(\mathrm{edf}\) depends on \(\rho\).  As a consequence, \(\frac{d\phi}{d\rho}\neq 0\), and both
  \(\frac{D_p}{2\phi}\) and \(\frac{n-M_p}{2}\log(2\pi\phi)\) contribute extra terms to the true derivative of the profiled cost.
* The finite-difference gradient that powers the runtime check differentiates the *actual* profiled cost, so it includes those \(\phi\)-derivative contributions.  The analytic routine omits them by design, which leaves a global bias (visible in the cosine/L2 metrics and the one-dimensional secant probe) even though individual components look consistent once very small entries are masked.【00cd65†L1-L25】【d32d95†L1-L18】【9481d1†L1-L88】

## Root cause
The “always-on” gradient check fails because the analytic REML gradient for the identity link purposefully ignores the \(\rho\)-dependence of the profiled scale \(\phi = D_p/(n-\mathrm{edf})\), while the cost function used in the optimizer differentiates through that dependence.  This mismatch injects the missing \(d\phi/d\rho\) terms into the finite-difference reference, producing the observed 7% relative error and sub-threshold cosine similarity despite a perfect per-component pass rate.【5ac4a7†L83-L136】【9481d1†L1-L88】【00cd65†L1-L25】

