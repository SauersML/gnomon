# Royston–Parmar Fine–Gray Plan (Archived Draft)

This document recorded an earlier attempt that combined a Royston–Parmar baseline with a Fine–Gray **partial** likelihood. After reviewing the issues highlighted in PR #540 and the follow-up questions, we have retired that approach in favour of the canonical full-likelihood blueprint in [`survival_72953.md`](./survival_72953.md).

Key corrections relative to the archived draft:

1. **Likelihood choice.** We now maximise the full Fine–Gray likelihood with hazard derivatives (`log h_i + log S_i`). The partial-likelihood risk-set algebra conflicted with a parametric baseline and produced inconsistent curvature terms.
2. **Hessian structure.** The canonical plan uses diagonal working weights (`W_i = w_i[H_i(b_i) + 1_{a_i<b_i} H_i(a_i)]`), eliminating the O(n²) accumulation and aligning with the Royston–Parmar literature.
3. **Prediction formulas.** Conditional CIFs now use the renormalised expression from Section 6 of the canonical plan, ensuring compatibility with time-varying effects.
4. **Time-varying effects.** Instead of freezing PGS×age terms at the horizon, the canonical plan integrates the hazard across the interval using cached derivative bases.
5. **Competing risks.** Competing CIFs will be estimated via separate RP Fine–Gray fits, removing the ad hoc Kaplan–Meier mixture.

The archived content is left intentionally blank to prevent resurrecting outdated algebra. All implementation work must follow `survival_72953.md`.
