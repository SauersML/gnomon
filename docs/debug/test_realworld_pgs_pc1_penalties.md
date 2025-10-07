# Why `test_realworld_pgs_pc1_penalties_not_both_hugging_positive_bound` fails

## What the test is expecting
The regression test samples a single training fold from a synthetic "real world" fixture and verifies that the two anisotropic PGS×PC1 smoothing penalties do **not** both sit on the +ρ wall. It collects the log smoothing parameters returned by `train_model` and asserts that at least one of `f(PGS,PC1)[1]` or `f(PGS,PC1)[2]` lies inside the box defined by `RHO_BOUND`.【F:calibrate/estimate.rs†L3367-L3397】【F:calibrate/estimate.rs†L69-L101】

## How the fixture drives the inner solver
The fixture deliberately injects very sharp logistic signals. Every sample draws its own scale and coefficient multipliers before the probability is evaluated:

- the "true" logit multiplies the interaction contribution by coefficients as high as 1.8 and then rescales the whole expression by `response_scales` up to 1.35,【F:calibrate/estimate.rs†L3171-L3192】
- the interaction term itself is a `tanh` of `PGS·PC1` plus additional noise, which frequently pushes the logit far away from zero.【F:calibrate/estimate.rs†L3181-L3187】

Because the fold contains 1,375 training samples (5/6 of 1,650), many rows land in the regime where the working weights `μ(1-μ)` are almost zero.【F:calibrate/estimate.rs†L3275-L3336】 In that regime `XᵀWX` loses rank along the same columns that the anisotropic interaction penalties try to control, so the raw penalized Hessian produced by PIRLS develops small **negative** eigenvalues.

## What happens in the REML gradient
`compute_gradient_with_bundle` explicitly checks the eigenvalues coming back from PIRLS. Whenever it sees a negative eigenvalue smaller than `-1e-4`, it aborts the analytic gradient and returns a "retreat" gradient whose components are `-(|ρ|+1)` for every penalty.【F:calibrate/estimate.rs†L2709-L2792】 That vector is always negative, so the BFGS outer loop interprets it as strong evidence that the cost decreases as ρ increases.

The problem is that the saturated rows from the fixture repeatedly trigger this safeguard for **both** anisotropic interaction penalties. As soon as one gradient evaluation sees the bad eigenvalue, the optimizer no longer receives curvature information for those parameters—only the retreat direction.

## Why both penalties hug the +ρ bound
Because the retreat gradient is negative in every component, each BFGS step pushes ρ upward. The optimizer works in the unconstrained `z` space, but `ρ = RHO_BOUND · tanh(z)` still clips the result to ±30.【F:calibrate/estimate.rs†L69-L101】 With no interior gradient ever fed back for the interaction block, the two `f(PGS,PC1)` penalties climb until they hit the hard ceiling and stay there. The test then observes both components within 1.0 of `RHO_BOUND` and fails.

## Root cause
The failure is **not** that the optimizer ignores a strong interaction—it is that the synthetic fixture routinely drives the logistic PIRLS fit into an indefinite region. Once `penalized_hessian_transformed` exhibits eigenvalues below `-1e-4`, the REML gradient code discards the real derivative and replaces it with the retreat vector. That defensive fallback always points toward larger ρ, so both anisotropic penalties end up glued to the +ρ bound and violate the test's expectation.
