use crate::calibrate::basis::{apply_sum_to_zero_constraint, create_difference_penalty_matrix, null_range_whiten};
use crate::calibrate::model::BasisConfig;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::hull::PeeledHull;
use crate::calibrate::model::LinkFunction;
use crate::calibrate::pirls; // for PirlsResult
// no penalty root helpers needed directly here

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
// no direct ndarray-linalg imports needed here
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{Llt as FaerLlt, Solve as FaerSolve};
use serde::{Serialize, Deserialize};
// Use the shared optimizer facade from estimate.rs
use crate::calibrate::estimate::{optimize_external_design, ExternalOptimOptions};

/// Features used to train the calibrator GAM
pub struct CalibratorFeatures {
    pub pred: Array1<f64>, // η̃ (logit) or μ̃ (identity)
    pub se: Array1<f64>,   // SẼ on the same scale
    pub dist: Array1<f64>, // signed distance to peeled hull (negative inside)
}

/// Configuration of the calibrator smooths and penalties
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibratorSpec {
    pub link: LinkFunction,
    pub pred_basis: BasisConfig,
    pub se_basis: BasisConfig,
    pub penalty_order_pred: usize,
    pub penalty_order_se: usize,
    pub double_penalty_ridge: f64,
    pub distance_hinge: bool,
}

/// Schema and parameters needed to rebuild the calibrator design at inference
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibratorModel {
    pub spec: CalibratorSpec,
    // Knot vectors and STZ transforms used for each smooth
    pub knots_pred: Array1<f64>,
    pub knots_se: Array1<f64>,
    pub stz_pred: Array2<f64>,
    pub stz_se: Array2<f64>,

    // Standardization parameters for inputs
    pub standardize_pred: (f64, f64), // mean, std
    pub standardize_se: (f64, f64),
    pub standardize_dist: (f64, f64),

    // Fitted lambdas
    pub lambda_pred: f64,
    pub lambda_se: f64,

    // Flattened coefficients and column schema
    pub coefficients: Array1<f64>,
    pub column_spans: (usize, usize, bool), // (#B_pred, #B_se, has_dist)

    // Optional Gaussian scale
    pub scale: Option<f64>,
}

/// Internal schema returned when building the calibrator design
pub struct InternalSchema {
    pub knots_pred: Array1<f64>,
    pub knots_se: Array1<f64>,
    pub stz_pred: Array2<f64>,
    pub stz_se: Array2<f64>,
    pub standardize_pred: (f64, f64),
    pub standardize_se: (f64, f64),
    pub standardize_dist: (f64, f64),
    pub column_spans: (usize, usize, bool),
}

/// Compute ALO features (η̃/μ̃, SẼ, signed distance) from a single base fit
pub fn compute_alo_features(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    prior_weights: ArrayView1<f64>,
    raw_train: ArrayView2<f64>,
    hull_opt: Option<&PeeledHull>,
    link: LinkFunction,
) -> Result<CalibratorFeatures, EstimationError> {
    let n = base.x_transformed.nrows();
    let p = base.x_transformed.ncols();

    // Prepare U = sqrt(W) X and z
    let w = &base.final_weights;
    let sqrt_w = w.mapv(f64::sqrt);
    let mut u = base.x_transformed.clone();
    let sqrt_w_col = sqrt_w.view().insert_axis(Axis(1));
    u *= &sqrt_w_col;

    // Factor K = X' W X + S_λ (use stabilized penalized Hessian from PIRLS). Prefer faer LLᵀ and multi-RHS solves.
    let mut k = base.stabilized_hessian_transformed.clone();
    // tiny ridge for safety
    for i in 0..p { k[[i,i]] += 1e-12; }
    let k_f = FaerMat::<f64>::from_fn(p, p, |i, j| k[[i, j]]);
    let llt = FaerLlt::new(k_f.as_ref(), Side::Lower)
        .map_err(|_| EstimationError::ModelIsIllConditioned { condition_number: f64::INFINITY })?;

    // Compute a_ii via block solves: solve K S = U^T, then a_i = u_i · s_i
    let block = 8192usize.min(n.max(1));
    let mut aii = Array1::<f64>::zeros(n);
    let ut = u.t(); // p x n
    let mut col_start = 0usize;
    while col_start < n {
        let col_end = (col_start + block).min(n);
        let cols = col_end - col_start;
        // RHS block: p x cols (slice of U^T)
        let rhs_block = ut.slice(s![.., col_start..col_end]).to_owned();
        // Solve K S = RHS for S (multi-RHS) using faer LLᵀ
        let rhs_f = FaerMat::<f64>::from_fn(p, cols, |i, j| rhs_block[[i, j]]);
        let s_block = llt.solve(rhs_f.as_ref()); // p x cols
        // Accumulate a_ii = u_i · s_i
        for j in 0..cols {
            let row_idx = col_start + j;
            let mut dot = 0.0;
            for i in 0..p { dot += u[[row_idx, i]] * s_block[(i, j)]; }
            aii[row_idx] = dot;
        }
        col_start = col_end;
    }

    // Clip leverage for stability
    aii.mapv_inplace(|v| v.clamp(0.0, 0.995));

    // eta_hat from base
    let eta_hat = base.x_transformed.dot(&base.beta_transformed);
    let z = base.solve_working_response.clone();

    // LOO predictor
    let mut eta_tilde = Array1::<f64>::zeros(n);
    for i in 0..n {
        let denom = (1.0 - aii[i]).max(1e-6);
        eta_tilde[i] = (eta_hat[i] - aii[i] * z[i]) / denom;
    }

    // LOO variance and SE
    let mut se_tilde = Array1::<f64>::zeros(n);
    let phi = match link {
        LinkFunction::Logit => 1.0,
        LinkFunction::Identity => {
            // Compute Gaussian dispersion from prior weights: φ = sum(w_prior * r^2) / (sum(w_prior) - edf)
            let mut rss = 0.0;
            let mut wsum = 0.0;
            for i in 0..n { let r = y[i] - base.final_mu[i]; rss += prior_weights[i] * r * r; wsum += prior_weights[i]; }
            let denom = (wsum - base.edf).max(1.0);
            (rss / denom).max(1e-12)
        }
    };
    for i in 0..n {
        let denom = (1.0 - aii[i]).max(1e-6);
        let wi = w[i];
        let ci = if wi > 1e-12 {
            aii[i] / wi
        } else {
            // Rare fallback: compute c_i = x_i K^{-1} x_i via one RHS solve using faer
            let xrow = base.x_transformed.row(i).to_owned();
            let rhs_f = FaerMat::<f64>::from_fn(p, 1, |r, _| xrow[r]);
            let s = llt.solve(rhs_f.as_ref()); // p x 1
            let mut dot = 0.0;
            for r in 0..p { dot += xrow[r] * s[(r, 0)]; }
            dot
        };
        // LOO variance inflation: Var_LOO = Var_full / (1 - aii)^2
        let var_tilde = phi * (ci / (denom * denom).max(1e-12));
        se_tilde[i] = var_tilde.max(0.0).sqrt();
    }

    // Diagnostics: leverage and dispersion summary
    let mut a = aii.to_vec();
    a.sort_by(|x,y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let p95_idx = if n>1 { ((0.95_f64 * (n as f64 - 1.0)).round() as usize).min(n-1) } else {0};
    let a_mean: f64 = aii.iter().sum::<f64>() / (n as f64).max(1.0);
    let a_p95 = a[p95_idx];
    let a_hi = aii.iter().filter(|v| **v > 0.9).count();
    eprintln!(
        "[CAL] ALO: n={}, mean(a_ii)={:.3e}, p95(a_ii)={:.3e}, frac(a_ii>0.9)={:.2}% , phi={:.3e}",
        n, a_mean, a_p95, 100.0 * (a_hi as f64) / (n as f64).max(1.0), phi
    );

    // Signed distance to peeled hull for raw predictors (zeros if no hull)
    let dist = if let Some(hull) = hull_opt { hull.signed_distance_many(raw_train) } else { Array1::zeros(raw_train.nrows()) };

    // For Identity link, pred is mean (same as eta). For Logit, pred is eta.
    let pred = match link { LinkFunction::Logit => eta_tilde, LinkFunction::Identity => eta_tilde };

    Ok(CalibratorFeatures { pred, se: se_tilde, dist })
}

/// Build calibrator design matrix, penalties and schema from features and spec
pub fn build_calibrator_design(
    features: &CalibratorFeatures,
    spec: &CalibratorSpec,
) -> Result<(Array2<f64>, Vec<Array2<f64>>, InternalSchema), EstimationError> {
    let n = features.pred.len();

    // Standardize inputs and record parameters
    fn mean_and_std_raw(v: &Array1<f64>) -> (f64, f64) {
        let mean = v.sum() / (v.len() as f64);
        let mut var = 0.0;
        for &x in v.iter() { let d = x - mean; var += d * d; }
        var /= v.len().max(1) as f64;
        (mean, var.sqrt())
    }
    fn standardize_with(mean: f64, std: f64, v: &Array1<f64>) -> (Array1<f64>, (f64, f64)) {
        let s_use = std.max(1e-8);
        (v.mapv(|x| (x - mean) / s_use), (mean, s_use))
    }

    let (pred_mean, pred_std_raw) = mean_and_std_raw(&features.pred);
    let (se_mean, se_std_raw) = mean_and_std_raw(&features.se);
    let (dist_mean, dist_std_raw) = mean_and_std_raw(&features.dist);

    let (pred_std, pred_ms) = standardize_with(pred_mean, pred_std_raw, &features.pred);
    let (se_std, se_ms) = standardize_with(se_mean, se_std_raw, &features.se);
    let (dist_std, dist_ms) = standardize_with(dist_mean, dist_std_raw, &features.dist);

    // Build B-spline bases
    // Note: ranges not needed with explicit knots; keep se range for degeneracy check below
    let se_min = se_std.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let se_max = se_std.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Build anti-resonant bases using mid-quantile knots with ghost boundaries
    fn make_midquantile_knots(vals_std: &Array1<f64>, degree: usize, target_internal: usize, min_internal: usize, max_internal: usize) -> Array1<f64> {
        use ndarray::Array1 as A1;
        let n = vals_std.len();
        let m = target_internal.clamp(min_internal, max_internal);
        if n == 0 || m == 0 { return A1::from(vec![0.0; degree + 1 + degree + 1]); }
        // copy and sort for empirical quantiles
        let mut v = vals_std.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let quant_at = |u: f64| -> f64 {
            let t = u.clamp(0.0, 1.0) * ((n - 1) as f64);
            let i = t.floor() as usize;
            let frac = t - (i as f64);
            if i + 1 < n { v[i] * (1.0 - frac) + v[i + 1] * frac } else { v[n - 1] }
        };
        let mut internal = Vec::with_capacity(m);
        for j in 0..m { let u = (j as f64 + 0.5) / ((m + 1) as f64); internal.push(quant_at(u)); }
        // ghost half-step via median spacing
        let h = if internal.len() >= 2 {
            let mut diffs: Vec<f64> = internal.windows(2).map(|w| w[1] - w[0]).collect();
            diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            diffs[diffs.len() / 2].abs().max(1e-8)
        } else { 1.0 };
        let left = internal.first().copied().unwrap_or(0.0) - 0.5 * h;
        let right = internal.last().copied().unwrap_or(0.0) + 0.5 * h;
        // open uniform-style: repeat boundaries degree+1 times
        let mut knots = Vec::with_capacity(m + 2 * (degree + 1));
        for _ in 0..(degree + 1) { knots.push(left); }
        knots.extend(internal.into_iter());
        for _ in 0..(degree + 1) { knots.push(right); }
        A1::from(knots)
    }

    // Anti-resonant internal counts with parity/coprime nudges
    fn choose_internal_counts(pred_target: usize, se_target: usize) -> (usize, usize) {
        fn sanitize(x: usize, lo: usize, hi: usize) -> usize { x.max(lo).min(hi) }
        fn gcd(mut a: usize, mut b: usize) -> usize { while b != 0 { let t = a % b; a = b; b = t; } a }
        fn coprime(a: usize, b: usize) -> bool { gcd(a.max(1), b.max(1)) == 1 }
        let mp = sanitize(pred_target, 8, 64);
        let mut ms = sanitize(se_target,   4, 32);
        if mp == ms { ms = sanitize(ms + 1, 4, 32); }
        if (mp % 2) == (ms % 2) { if ms + 1 <= 32 { ms += 1; } else { ms = sanitize(ms - 1, 4, 32); } }
        if !coprime(mp, ms) {
            for delta in [1isize, -1, 2, -2] {
                let cand = (ms as isize + delta) as usize;
                if cand >= 4 && cand <= 32 && (cand % 2) != (mp % 2) && coprime(mp, cand) { ms = cand; break; }
            }
        }
        (mp, ms)
    }
    let (mp_int, ms_int) = choose_internal_counts(spec.pred_basis.num_knots, spec.se_basis.num_knots);
    let knots_pred = make_midquantile_knots(&pred_std, spec.pred_basis.degree, mp_int, 8, 64);
    let knots_se = make_midquantile_knots(&se_std, spec.se_basis.degree, ms_int, 4, 32);
    let (b_pred_raw, _) = crate::calibrate::basis::create_bspline_basis_with_knots(
        pred_std.view(), knots_pred.view(), spec.pred_basis.degree
    )?;
    let (b_se_raw, _) = crate::calibrate::basis::create_bspline_basis_with_knots(
        se_std.view(), knots_se.view(), spec.se_basis.degree
    )?;

    // Apply sum-to-zero constraints (unweighted)
    let (b_pred_c, stz_pred) = apply_sum_to_zero_constraint(b_pred_raw.view(), None)?;
    let (mut b_se_c, stz_se) = apply_sum_to_zero_constraint(b_se_raw.view(), None)?;

    // Penalties for constrained bases
    let s_pred_raw = create_difference_penalty_matrix(b_pred_c.ncols(), spec.penalty_order_pred)?;
    let s_se_raw = create_difference_penalty_matrix(b_se_c.ncols(), spec.penalty_order_se)?;

    // Nullspace-only ridge (double-penalty): add tiny ridge on the nullspace block only
    fn add_nullspace_ridge(s_raw: &Array2<f64>, ridge: f64) -> Result<Array2<f64>, EstimationError> {
        // Preserve original curvature penalty; only add a tiny ridge on the nullspace.
        if ridge <= 0.0 { return Ok(s_raw.clone()); }
        let (z_null, _z_range_w) = null_range_whiten(s_raw)
            .map_err(|e| EstimationError::BasisError(e))?;
        let mut s = s_raw.clone();
        if z_null.ncols() > 0 {
            let p_null = z_null.dot(&z_null.t());
            let p_null_ridge = p_null.mapv(|v| v * ridge);
            s = s + &p_null_ridge;
        }
        Ok(s)
    }

    let ridge = spec.double_penalty_ridge.max(0.0);
    let s_pred = add_nullspace_ridge(&s_pred_raw, ridge)?;
    let mut s_se = add_nullspace_ridge(&s_se_raw, ridge)?;

    // Degenerate se fallback: if se_std has near-zero range, replace s2 with a single centered linear term and zero penalty
    let se_range = se_max - se_min;
    if se_range.abs() < 1e-8 {
        // single centered column
        b_se_c = se_std.view().insert_axis(Axis(1)).to_owned();
        s_se = Array2::<f64>::zeros((1, 1));
    }

    // Decide whether to include distance column based on raw std (and by implication, hull presence)
    let use_dist = dist_std_raw > 1e-8;
    // Assemble X = [1 | B_pred | B_se | dist?]
    let p_cols = 1 + b_pred_c.ncols() + b_se_c.ncols() + if use_dist { 1 } else { 0 };
    let mut x = Array2::<f64>::zeros((n, p_cols));
    // intercept
    for i in 0..n { x[[i, 0]] = 1.0; }
    // B_pred
    x.slice_mut(s![.., 1..1 + b_pred_c.ncols()]).assign(&b_pred_c);
    // B_se
    let se_off = 1 + b_pred_c.ncols();
    x.slice_mut(s![.., se_off..se_off + b_se_c.ncols()]).assign(&b_se_c);
    // dist (optionally hinge)
    if use_dist {
        let dist_off = se_off + b_se_c.ncols();
        if spec.distance_hinge {
            for i in 0..n { x[[i, dist_off]] = dist_std[i].max(0.0); }
        } else {
            for i in 0..n { x[[i, dist_off]] = dist_std[i]; }
        }
    }

    // Full penalty matrices aligned to X columns (zeros for unpenalized cols)
    let p = x.ncols();
    let mut s_pred_p = Array2::<f64>::zeros((p, p));
    let mut s_se_p = Array2::<f64>::zeros((p, p));
    // Place into the appropriate diagonal blocks
    for i in 0..b_pred_c.ncols() { for j in 0..b_pred_c.ncols() { s_pred_p[[1+i, 1+j]] = s_pred[[i,j]]; } }
    for i in 0..b_se_c.ncols() { for j in 0..b_se_c.ncols() { s_se_p[[se_off+i, se_off+j]] = s_se[[i,j]]; } }

    let penalties = vec![s_pred_p, s_se_p];
    // Diagnostics: design summary
    let m_pred_int = (knots_pred.len() as isize - 2 * (spec.pred_basis.degree as isize + 1)).max(0) as usize;
    let m_se_int = (knots_se.len() as isize - 2 * (spec.se_basis.degree as isize + 1)).max(0) as usize;
    eprintln!(
        "[CAL] design: n={}, p={}, pred_cols={}, se_cols={}, has_dist={}, deg_pred={}, deg_se={}, m_pred_int={}, m_se_int={}, se_degenerate={}",
        n, x.ncols(), b_pred_c.ncols(), b_se_c.ncols(), use_dist,
        spec.pred_basis.degree, spec.se_basis.degree, m_pred_int, m_se_int,
        (se_max - se_min).abs() < 1e-8
    );
    let schema = InternalSchema {
        knots_pred,
        knots_se,
        stz_pred,
        stz_se,
        standardize_pred: pred_ms,
        standardize_se: se_ms,
        standardize_dist: dist_ms,
        column_spans: (b_pred_c.ncols(), b_se_c.ncols(), use_dist),
    };

    Ok((x, penalties, schema))
}

/// Predict with a fitted calibrator model given raw features
pub fn predict_calibrator(
    model: &CalibratorModel,
    pred: ArrayView1<f64>,
    se: ArrayView1<f64>,
    dist: ArrayView1<f64>,
) -> Array1<f64> {
    // Standardize inputs using stored params
    let (mp, sp) = model.standardize_pred;
    let (ms, ss) = model.standardize_se;
    let (md, sd) = model.standardize_dist;
    let pred_std = pred.mapv(|x| (x - mp) / sp.max(1e-8));
    let se_std = se.mapv(|x| (x - ms) / ss.max(1e-8));
    let dist_std = dist.mapv(|x| (x - md) / sd.max(1e-8));

    // Build bases using stored knots, then apply stored STZ transforms
    let (b_pred_raw, _) = crate::calibrate::basis::create_bspline_basis_with_knots(
        pred_std.view(),
        model.knots_pred.view(),
        model.spec.pred_basis.degree,
    ).expect("pred basis with knots");
    let (b_se_raw, _) = crate::calibrate::basis::create_bspline_basis_with_knots(
        se_std.view(),
        model.knots_se.view(),
        model.spec.se_basis.degree,
    ).expect("se basis with knots");

    let b_pred = b_pred_raw.dot(&model.stz_pred);
    let b_se = b_se_raw.dot(&model.stz_se);

    // Assemble X = [1 | B_pred | B_se | dist]
    let n = pred.len();
    let (n_pred_cols, n_se_cols, has_dist) = model.column_spans;
    let p_cols = 1 + n_pred_cols + n_se_cols + if has_dist { 1 } else { 0 };
    let mut x = Array2::<f64>::zeros((n, p_cols));
    for i in 0..n { x[[i, 0]] = 1.0; }
    if n_pred_cols > 0 {
        x.slice_mut(s![.., 1..1 + n_pred_cols]).assign(&b_pred.slice(s![.., ..n_pred_cols]));
    }
    if n_se_cols > 0 {
        let off = 1 + n_pred_cols;
        x.slice_mut(s![.., off..off + n_se_cols]).assign(&b_se.slice(s![.., ..n_se_cols]));
    }
    if has_dist {
        let off = 1 + n_pred_cols + n_se_cols;
        if model.spec.distance_hinge {
            for i in 0..n { x[[i, off]] = dist_std[i].max(0.0); }
        } else {
            for i in 0..n { x[[i, off]] = dist_std[i]; }
        }
    }

    // Linear predictor and mean (no offset)
    let eta = x.dot(&model.coefficients);
    match model.spec.link {
        LinkFunction::Logit => {
            let eta_c = eta.mapv(|e| e.clamp(-40.0, 40.0));
            let mut probs = eta_c.mapv(|e| 1.0 / (1.0 + (-e).exp()));
            probs.mapv_inplace(|p| p.clamp(1e-8, 1.0 - 1e-8));
            probs
        }
        LinkFunction::Identity => eta,
    }
}

/// Fit the calibrator by optimizing two smoothing parameters with a tiny grid search
pub fn fit_calibrator(
    y: ArrayView1<f64>,
    prior_weights: ArrayView1<f64>,
    x: ArrayView2<f64>,
    penalties: &[Array2<f64>],
    link: LinkFunction,
    _spec: &CalibratorSpec,
) -> Result<(Array1<f64>, [f64;2], f64, (f64, f64), (usize, f64)), EstimationError> {
    let opts = ExternalOptimOptions { link, max_iter: 50, tol: 1e-3 };
    eprintln!(
        "[CAL] fit: starting external REML/BFGS on X=[{}×{}], penalties={} (link={:?})",
        x.nrows(), x.ncols(), penalties.len(), link
    );
    let res = optimize_external_design(y, prior_weights, x, penalties, &opts)?;
    let lambdas = [res.lambdas[0], res.lambdas[1]];
    let edf_pred = *res.edf_by_block.get(0).unwrap_or(&0.0);
    let edf_se = *res.edf_by_block.get(1).unwrap_or(&0.0);
    eprintln!(
        "[CAL] fit: done. lambda_pred={:.3e}, lambda_se={:.3e}, edf_pred={:.2}, edf_se={:.2}, edf_total={:.2}, scale={:.3e}",
        lambdas[0], lambdas[1], edf_pred, edf_se, res.edf_total, res.scale
    );
    Ok((res.beta, lambdas, res.scale, (edf_pred, edf_se), (res.iterations, res.final_grad_norm)))
}

// (removed local optimizer; using shared optimize_external_design)
