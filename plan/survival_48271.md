# Fine–Gray Gradient/Hessian Notes (Aligned with Canonical Plan)

This note consolidates the gradient and curvature expressions that will actually ship with the canonical full-likelihood blueprint (`survival_72953.md`). The previous version mixed risk-set algebra with sign mistakes; the formulas below replace it entirely.

## 1. Notation recap
- `w_i`: observation weight.
- `d_i`: indicator that the target event occurs at `b_i`.
- `c_i`: indicator that a competing event occurs at `b_i`.
- `η_i(u)`: log cumulative subdistribution hazard at age `u`.
- `H_i(u) = exp(η_i(u))` and `S_i(u) = exp(-H_i(u))`.
- `∂η_i/∂u`: derivative of the linear predictor with respect to log-age.

## 2. Score contributions
At exit age `b_i`:
```
U_i^{exit} = w_i [ d_i * (1 + ∂ log(∂η_i/∂u)/∂η_i) - H_i(b_i) ]
```
At entry age `a_i` (only when left-truncated):
```
U_i^{entry} = + w_i H_i(a_i)
```
The total score for coefficient block `β` is assembled by multiplying these scalars with the corresponding design rows (`X_exit[i, :]` and `X_entry[i, :]`) and adding penalty derivatives.

## 3. Hessian / working weights
The negative Hessian per observation is diagonal:
```
W_i^{exit} = w_i H_i(b_i)
W_i^{entry} = w_i H_i(a_i)
```
The penalised system therefore uses
```
H = X_exit^⊤ diag(W^{exit}) X_exit + X_entry^⊤ diag(W^{entry}) X_entry + Σ_k λ_k P_k
```
All curvature terms are positive. The erroneous minus sign from the old draft has been removed; it was an artefact of differentiating the partial likelihood.

## 4. Working response
The element-wise IRLS update remains
```
z_i = η_i + (U_i^{exit} + U_i^{entry}) / (W_i^{exit} + W_i^{entry})
```
which is equivalent to solving `W δ = U` because `W` is diagonal.

## 5. Implementation pointers
- Cache `H_i` and `∂η_i/∂u` once per iteration.
- Guard `∂η_i/∂u` against numerical underflow; clip at `1e-9` before taking logs.
- When `d_i = 0`, drop the derivative term and the exit score simplifies to `-w_i H_i(b_i)`.
- Competing events contribute through `c_i` in the likelihood but do not change the gradient formulas because the survival term already accounts for them.

These expressions are consistent with Appendix A of Crowther & Lambert (2014) and feed directly into the canonical plan.
