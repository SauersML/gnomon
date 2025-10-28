# Fine–Gray Gradient/Hessian Notes (Canonical Full-Likelihood)

## 1. Notation
- `w_i`: observation weight.
- `d_i`: indicator that the target event occurs at exit age `b_i`.
- `η_i(a)`: log cumulative subdistribution hazard.
- `H_i(a) = exp(η_i(a))` and `S_i(a) = exp(-H_i(a))`.
- `X_exit[i,:]`: design row for `η_i(b_i)`; `X_entry[i,:]`: design row for `η_i(a_i)`.
- `D_exit[i,:]`: design row for the derivative `∂η_i/∂a|_{b_i}`; `D_entry[i,:]` for `∂η_i/∂a|_{a_i}` when required.
- `J(a) = a - a_min + δ`: Jacobian of the log-age transform.

## 2. Log-likelihood recap
Per subject,

```
ℓ_i = w_i [ d_i ( η_i(b_i) + log(∂η_i/∂a|_{b_i}) - log J(b_i) ) - H_i(b_i) + H_i(a_i) ]
```

Competing events correspond to `d_i = 0`; they therefore omit the hazard derivative term and contribute only via the survival component `-H_i(b_i)` and the left-truncation adjustment `+H_i(a_i)`.

## 3. Score vectors
### 3.1 Exit contribution
```
U_i^{exit} = w_i [ d_i ( X_exit[i,:] + D_exit[i,:] / (∂η_i/∂a|_{b_i}) ) - H_i(b_i) X_exit[i,:] ]
```
The first piece comes from differentiating the event density, and the second arises from the survival term `-H_i(b_i)`.

### 3.2 Entry contribution
```
U_i^{entry} = + w_i H_i(a_i) X_entry[i,:]
```
Left truncation increases the log-likelihood by the cumulative hazard accumulated before entry.

### 3.3 Total score
We assemble the score for the full coefficient vector `β̃` by summing exit and entry pieces and adding penalty gradients `-Σ_k λ_k P_k β̃`.

## 4. Hessian components
### 4.1 Exit survival term
```
H_exit = Σ_i w_i H_i(b_i) X_exit[i,:]^⊤ X_exit[i,:]
```
### 4.2 Entry survival term
```
H_entry = Σ_i w_i H_i(a_i) X_entry[i,:]^⊤ X_entry[i,:]
```
This term subtracts from the negative Hessian because `ℓ` increases with `H_i(a_i)`.

### 4.3 Hazard derivative correction
For `d_i = 1`,
```
H_i^{deriv} = w_i (D_exit[i,:]^⊤ D_exit[i,:]) / (∂η_i/∂a|_{b_i})^2
```
No correction is required at entry: the Jacobian only appears in the density evaluated at the event time.

### 4.4 Penalised negative Hessian
```
-∂²ℓ = H_exit - H_entry + Σ_{i:d_i=1} H_i^{deriv} + Σ_k λ_k P_k
```
We explicitly assemble the low-rank derivative updates and add them to the survival curvature before applying penalties.

## 5. Solving the Newton system
- The Newton step solves `( -∂²ℓ ) δ = U`. We use damped Cholesky when the matrix is positive definite and switch to preconditioned conjugate gradients with Levenberg–Marquardt damping otherwise.
- The hazard-derivative outer products are applied via Sherman–Morrison updates to avoid forming dense matrices when the design is sparse.
- REML/LAML derivatives require the same assembled Hessian; we feed it directly into the trace computations used for smoothing parameter updates.

## 6. Numerical safeguards
- Clip `∂η_i/∂a` at `1e-12` before taking logarithms to prevent catastrophic cancellation.
- When the Jacobian `J(a)` is very small (subjects entering near `a_min`), evaluate `log J(a)` with compensated arithmetic.
- Validate that `H_i(a)` remains monotone increasing in `a`; violations trigger the barrier penalty described in `survival_72953.md`.

These notes replace the outdated diagonal-weight heuristics and line up exactly with the canonical blueprint.
