/*
----------------------------------------------------------------------------------------------------
// 1. EXECUTIVE SUMMARY
----------------------------------------------------------------------------------------------------
//
//  PROBLEM:
//      We have a linear prediction model built from thousands of binary variables. Some of these
//      variables are imputed, and we have a confidence score (a probability `p`) for each
//      imputation. We must decide how to incorporate these uncertain variables into the final
//      patient score.
//
//  TWO COMPETING STRATEGIES:
//      A) The Filtering Method (`S_F`): Set a confidence threshold. Discard all variables below
//         the threshold, treating their contribution as zero.
//      B) The Expectation Method (`S_E`): Use all variables, but weight the effect size of each
//         imputed variable by its probability of being in the "active" state (1).
//
//      We must implement the Expectation Method (`S_E`). It is mathematically guaranteed to
//      produce predictions with a lower or equal Mean Squared Error (MSE) than any filtering
//      method. It is the most accurate predictor that can be constructed from our model and data.
//
----------------------------------------------------------------------------------------------------
// 2. FORMAL PROBLEM DEFINITION
----------------------------------------------------------------------------------------------------
//
//  OBJECTIVE:
//      To construct a predictor, `S`, that is the most accurate possible estimate of a patient's
//      true, underlying value, `Y_true`. Accuracy is formally defined by minimizing the
//      Mean Squared Error (MSE).
//
//      Minimize: E[(S - Y_true)²]
//
//  MODEL DEFINITION:
//      The true value `Y_true` is determined by a linear combination of binary variables.
//
//      Y_true = β₀ + Σᵢ βᵢxᵢ
//
//      Where:
//          * `Y_true`: The true, unobservable value for the patient (e.g., true genetic risk).
//          * `β₀`: The model intercept or baseline value.
//          * `βᵢ`: The known, pre-trained weight (effect size) for variable `i`.
//          * `xᵢ`: The true, unknown binary state (0 or 1) of variable `i`.
//
//  AVAILABLE DATA:
//      For each variable `i`, we do not know `xᵢ` with certainty. Instead, we have a
//      probability derived from our imputation model:
//
//      pᵢ = P(xᵢ = 1 | Data)
//
//      This `pᵢ` is the posterior probability that the true state is 1, given all available
//      data for the patient. For non-imputed variables, `pᵢ` is either 0 or 1.
//
----------------------------------------------------------------------------------------------------
// 3. MATHEMATICAL DEFINITION OF THE COMPETING METHODS
----------------------------------------------------------------------------------------------------
//
//  METHOD E: THE EXPECTATION SCORE (S_E)
//      This score is the mathematical expectation of `Y_true`, conditioned on the available data.
//      S_E = E[Y_true | Data] = E[β₀ + Σᵢ βᵢxᵢ | Data] = β₀ + Σᵢ βᵢ * E[xᵢ | Data]
//      Since E[xᵢ | Data] = pᵢ, the formula is:
//
//      S_E = β₀ + Σᵢ βᵢpᵢ
//
//  METHOD F: THE FILTERING SCORE (S_F)
//      This score is based on a confidence threshold `τ`. We partition all variables `I` into
//      a high-confidence set `I_H` and a low-confidence set `I_L`, which is discarded.
//
//      I_L = {i ∈ I | 1-τ < pᵢ < τ}  // The set of discarded variables.
//
//      The score is constructed by assuming the contribution of all variables in `I_L` is zero.
//      (For simplicity, we assume the hard-imputed value `x̂ᵢ` is used for the kept set `I_H`).
//
//      S_F = β₀ + Σ_{i ∈ I_H} βᵢx̂ᵢ
//
----------------------------------------------------------------------------------------------------
// 4. MINIMIZATION OF MEAN SQUARED ERROR
----------------------------------------------------------------------------------------------------
//
//  We will now prove that MSE(S_E) ≤ MSE(S_F) for any choice of threshold `τ`.
//
//  A. MSE Analysis of the Expectation Score (S_E)
//  ==============================================
//
//  The error of S_E is:
//      Error(S_E) = S_E - Y_true = (β₀ + Σᵢ βᵢpᵢ) - (β₀ + Σᵢ βᵢxᵢ) = Σᵢ βᵢ(pᵢ - xᵢ)
//
//  The MSE is the expected squared error. Assuming uncorrelated errors between variables:
//      MSE(S_E) = E[ (Σᵢ βᵢ(pᵢ - xᵢ))² ] = Σᵢ βᵢ² * E[(pᵢ - xᵢ)²]
//
//  By definition, `pᵢ` is the conditional mean of `xᵢ`, so the error term `(pᵢ - xᵢ)` has a
//  mean of zero. Thus, `E[(pᵢ - xᵢ)²]` is the conditional variance of `xᵢ`.
//  The variance of a Bernoulli variable `xᵢ` with P(xᵢ=1)=pᵢ is `pᵢ(1 - pᵢ)`.
//
//      MSE(S_E) = Σᵢ βᵢ² * pᵢ(1 - pᵢ)
//
//
//  B. MSE Analysis of the Filtering Score (S_F)
//  ============================================
//
//  The error of S_F is the sum of contributions from all discarded variables:
//      Error(S_F) = S_F - Y_true = (β₀ + Σ_{i ∈ I_H} βᵢxᵢ) - (β₀ + Σ_{i ∈ I_H} βᵢxᵢ + Σ_{i ∈ I_L} βᵢxᵢ)
//      Error(S_F) = - Σ_{i ∈ I_L} βᵢxᵢ
//
//  The MSE of any estimator can be decomposed into `Bias² + Variance`.
//
//      1. Bias of S_F:
//         Bias(S_F) = E[S_F - Y_true] = E[-Σ_{i ∈ I_L} βᵢxᵢ] = -Σ_{i ∈ I_L} βᵢpᵢ
//         S_F is a BIASED estimator. It systematically underestimates or overestimates the true score.
//
//      2. Variance of the Error of S_F:
//         Var(Error(S_F)) = Var(-Σ_{i ∈ I_L} βᵢxᵢ) = Σ_{i ∈ I_L} βᵢ² * Var(xᵢ) = Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//
//  Combining these:
//      MSE(S_F) = (Bias(S_F))² + Var(Error(S_F))
//      MSE(S_F) = (-Σ_{i ∈ I_L} βᵢpᵢ)² + Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//      MSE(S_F) = (Σ_{i ∈ I_L} βᵢpᵢ)² + Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//
//
//  C. Direct Comparison
//  ====================
//
//  Let's compare the MSE of the two methods. Note that for the high-confidence variables
//  kept by S_F, their `pᵢ` is effectively 0 or 1, making their `pᵢ(1-pᵢ)` term zero.
//  Therefore, the error for S_E comes only from the low-confidence set `I_L`.
//
//      MSE(S_E) = Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ)
//
//      MSE(S_F) = Σ_{i ∈ I_L} βᵢ² * pᵢ(1 - pᵢ) + (Σ_{i ∈ I_L} βᵢpᵢ)²
//
//  The difference is immediately obvious:
//
//      MSE(S_F) - MSE(S_E) = (Σ_{i ∈ I_L} βᵢpᵢ)²
//
//  This term is a square and is therefore ALWAYS greater than or equal to zero.
//
//      => MSE(S_F) ≥ MSE(S_E)
//
//  Q.E.D.
//
----------------------------------------------------------------------------------------------------
// 5. REFUTATION OF COMMON COUNTER-ARGUMENTS
----------------------------------------------------------------------------------------------------
//
//  FALLACY 1: "The Expectation Score is too 'timid' and has low variance. A score needs
//              high variance to discriminate between high and low-risk patients."
//
//  CORRECTION: This confuses score variance with predictive power. The formal measure of a
//  predictor's power is its R-squared (`R²`), where `R² = 1 - (MSE / Var(Y_true))`. To maximize
//  `R²`, one must MINIMIZE MSE. Our proof shows that S_E minimizes MSE, therefore S_E maximizes
//  the predictive power (`R²`). The "boldness" of S_F is not signal; it is amplified noise,
//  which increases error and lowers predictive power.
//
//  FALLACY 2: "The proof only shows S_E is the best estimator of the genetic score `G`, not
//              the best predictor of the final clinical phenotype `Y_phenotype`."
//
//  CORRECTION: This is false. The optimal predictor for `Y_phenotype` is, by definition, its
//  conditional expectation `E[Y_phenotype | Data]`.
//
//      S_optimal = E[Y_phenotype | Data] = E[G + ε | Data]
//                = E[G | Data] + E[ε | Data]
//
//  Since the non-genetic component `ε` is independent of the genetic `Data`, `E[ε | Data] = E[ε] = 0`.
//  This leaves `S_optimal = E[G | Data]`, which is precisely the formula for `S_E`. The Expectation
//  Score is the optimal predictor for both the genetic score and the final clinical outcome.
//
----------------------------------------------------------------------------------------------------
// 6. FINAL IMPLEMENTATION DIRECTIVE
----------------------------------------------------------------------------------------------------
//
//  The correct, optimal, and required implementation is the Expectation-Based Method.
//
//  For each patient, the score MUST be calculated as follows:
//
//      let score = intercept + effects_and_probabilities.iter().map(|(beta, p)| beta * p).sum();
//
//  Where:
//      * `intercept` corresponds to `β₀`.
//      * `beta` corresponds to the effect size `βᵢ`.
//      * `p` corresponds to the probability `pᵢ` that the variable's true state is 1.
//
//  This approach is not a heuristic. It is the direct implementation of the provably optimal
//  predictor that minimizes prediction error and maximizes predictive power.
//
====================================================================================================
*/
