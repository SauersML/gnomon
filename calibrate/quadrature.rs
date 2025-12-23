//! Gauss-Hermite Quadrature for Posterior Mean Predictions
//!
//! This module provides functions to compute the posterior mean of predictions
//! by integrating over the uncertainty in the linear predictor using
//! Gauss-Hermite quadrature.
//!
//! # Background
//!
//! Standard predictions return `g⁻¹(η̂)` where `η̂` is the point estimate (mode).
//! For curved link functions like logit or survival transforms, this differs from
//! the posterior mean `E[g⁻¹(η)]` where `η ~ N(η̂, σ²)`.
//!
//! The posterior mean:
//! - Minimizes squared prediction error (Brier score)
//! - Is more conservative at extreme predictions
//! - Accounts for parameter uncertainty in the final probability
//!
//! # Implementation
//!
//! We use 7-point Gauss-Hermite quadrature, which is exact for polynomials up
//! to degree 13. For the smooth link functions used in practice, this provides
//! excellent accuracy with minimal computational cost.
//!
//! # Key Assumptions and Limitations
//!
//! Gaussian linear predictor: GHQ assumes the linear predictor η follows a
//! Gaussian distribution. Under a multivariate normal posterior for β (from the
//! Hessian), any linear combination η = Xβ is exactly Gaussian. This assumption
//! is consistent with LAML (Laplace Approximate Marginal Likelihood) used for
//! smoothing parameter selection.
//!
//! Non-Gaussian risk output: GHQ does NOT assume the risk is Gaussian. It
//! correctly integrates through nonlinear link functions (sigmoid, survival
//! transforms) to capture skewed risk distributions.
//!
//! Survival sensitivity: For survival models with double-exponential transforms
//! (e.g., 1 - exp(-exp(η))), small differences in η are amplified in the tails.
//! At extreme horizons, this tail sensitivity means GHQ-based intervals may be
//! slightly underconfident. HMC would provide marginally more accurate tail
//! quantiles at significant computational cost.
//!
//! B-spline local support: At any evaluation point, only ~k+1 spline basis
//! functions are nonzero (typically 4 for cubic splines). However, the linear
//! predictor also includes PGS, sex effects, interactions, and PC effects, so
//! the total remains a sum of many terms.
//!
//! # Alternative: HMC
//!
//! For cases where the Gaussian assumption on η is questionable (very rare
//! diseases with <500 cases, extreme non-Gaussianity in the coefficient
//! posterior), Hamiltonian Monte Carlo could sample β directly and compute
//! risk for each sample. This is 100-1000x more expensive but makes no
//! distributional assumptions.

/// Gauss-Hermite quadrature nodes (abscissas) for 7-point rule.
/// These are the roots of the Hermite polynomial H₇(x).
/// Symmetric around zero: ±2.6519613568..., ±1.6735516287..., ±0.8162878828..., 0
const GH_NODES_7: [f64; 7] = [
    -2.651961356835233,
    -1.673551628767471,
    -0.816287882858965,
    0.0,
    0.816287882858965,
    1.673551628767471,
    2.651961356835233,
];

/// Gauss-Hermite quadrature weights for 7-point rule.
/// These are pre-normalized so they sum to 1 (for integration against standard normal).
/// Raw Hermite weights divided by sqrt(pi) to convert from exp(-x²) to normal density.
const GH_WEIGHTS_7: [f64; 7] = [
    0.0009717812450995,
    0.0545155828191270,
    0.4256072526101277,
    0.8102646175568073,
    0.4256072526101277,
    0.0545155828191270,
    0.0009717812450995,
];

/// Normalization factor: sum of weights should be sqrt(pi) for Hermite,
/// but we've pre-normalized to sum to ~1.77 for direct use.
const GH_WEIGHT_SUM: f64 = 1.7724538509055159; // sqrt(pi)

/// Computes the posterior mean probability for a logistic model using
/// Gauss-Hermite quadrature.
///
/// Given:
/// - `eta`: point estimate of linear predictor (log-odds)
/// - `se_eta`: standard error of eta (from Hessian)
///
/// Returns: E[sigmoid(η)] where η ~ N(eta, se_eta²)
///
/// When `se_eta` is zero or very small, this reduces to `sigmoid(eta)`.
#[inline]
pub fn logit_posterior_mean(eta: f64, se_eta: f64) -> f64 {
    // If SE is negligible, return the mode (standard sigmoid)
    if se_eta < 1e-10 {
        return sigmoid(eta);
    }

    // Gauss-Hermite integration: E[f(η)] = ∫ f(η) φ(η) dη
    // Transform: η = eta + sqrt(2) * se_eta * x, where x ~ standard Hermite measure
    // This gives: E[f(η)] ≈ (1/sqrt(π)) Σᵢ wᵢ f(eta + sqrt(2) * se_eta * xᵢ)
    let scale = std::f64::consts::SQRT_2 * se_eta;
    let mut sum = 0.0;

    for i in 0..7 {
        let eta_i = eta + scale * GH_NODES_7[i];
        let prob_i = sigmoid(eta_i);
        sum += GH_WEIGHTS_7[i] * prob_i;
    }

    // Normalize by sqrt(pi) since Hermite weights are for exp(-x²) measure
    let mean_prob = sum / GH_WEIGHT_SUM;

    // Clamp to valid probability range
    mean_prob.clamp(1e-10, 1.0 - 1e-10)
}

/// Computes posterior mean probabilities for a batch of predictions.
///
/// This is the vectorized version of `logit_posterior_mean`.
pub fn logit_posterior_mean_batch(
    eta: &ndarray::Array1<f64>,
    se_eta: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    ndarray::Zip::from(eta)
        .and(se_eta)
        .map_collect(|&e, &se| logit_posterior_mean(e, se))
}

/// Standard sigmoid function with numerical stability.
#[inline]
fn sigmoid(x: f64) -> f64 {
    let x_clamped = x.clamp(-700.0, 700.0);
    1.0 / (1.0 + f64::exp(-x_clamped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_zero_se_returns_mode() {
        // When SE is zero, posterior mean should equal mode
        let eta = 1.5;
        let se = 0.0;
        let mean = logit_posterior_mean(eta, se);
        let mode = sigmoid(eta);
        assert_relative_eq!(mean, mode, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_at_zero() {
        // At eta=0 (50% probability), mean should still be ~50%
        let eta = 0.0;
        let se = 1.0;
        let mean = logit_posterior_mean(eta, se);
        // Due to symmetry of sigmoid around 0, mean ≈ mode
        assert_relative_eq!(mean, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_shrinkage_at_extremes() {
        // At extreme eta, mean should be pulled toward 0.5
        let eta = 3.0; // mode = sigmoid(3) ≈ 0.953
        let se = 1.0;
        let mean = logit_posterior_mean(eta, se);
        let mode = sigmoid(eta);

        // Mean should be less than mode (shrunk toward 0.5)
        assert!(mean < mode, "Expected mean {} < mode {}", mean, mode);
        // But still reasonably high
        assert!(mean > 0.8, "Mean {} should still be high", mean);
    }

    #[test]
    fn test_matches_monte_carlo() {
        // Compare quadrature to Monte Carlo with many samples
        let eta = 2.0;
        let se = 0.8;

        let quad_mean = logit_posterior_mean(eta, se);

        // Monte Carlo with 100,000 samples
        let n_samples = 100_000;
        let mut mc_sum = 0.0;
        let mut rng_state = 12345u64; // Simple LCG for reproducibility
        for _ in 0..n_samples {
            // Box-Muller for normal samples
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng_state as f64) / (u64::MAX as f64)).max(1e-10); // Prevent ln(0)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (rng_state as f64) / (u64::MAX as f64);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let eta_sample = eta + se * z;
            mc_sum += sigmoid(eta_sample);
        }
        let mc_mean = mc_sum / (n_samples as f64);

        // Should match within Monte Carlo sampling error (~0.01)
        assert_relative_eq!(quad_mean, mc_mean, epsilon = 0.01);
    }

    #[test]
    fn test_weights_sum() {
        // Verify weights sum to sqrt(pi)
        let sum: f64 = GH_WEIGHTS_7.iter().sum();
        assert_relative_eq!(sum, GH_WEIGHT_SUM, epsilon = 1e-10);
    }
}
