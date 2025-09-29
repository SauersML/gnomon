Short answer: you’re mixing scales/weights.

What your test does:
	•	Builds U using only the prior weights w (u = x_transformed * sqrt(w)), then
	•	Uses that same u both to (a) get the leverage a_{ii} = u_i^\top K^{-1} u_i and (b) to project the covariance for the SE, i.e. var_full = (K^{-1} u_i)^T (U^T U) (K^{-1} u_i).

Why this breaks:
	1.	Missing the GLM working weights.
For logistic IRLS, the working weight for obs i is
W_i \;=\; w_i\;\mu_i(1-\mu_i)
(not just w_i). The hat/leverage must be built with U = \sqrt{W}\,X. You only use \sqrt{w}, so your a_{ii} and X^\top W X are inconsistent with the fit.
	2.	Wrong vector for the SE projection.
The standard error you want to compare against compute_alo_features is (on the η-scale, i.e. linear predictor):
\mathrm{Var}(\hat\eta_i)\;=\; x_i^\top K^{-1}\,(X^\top W X)\,K^{-1} x_i,
i.e. you project with the unweighted row x_i (in the same transformed parameterization used to build K), not with the weighted row u_i=\sqrt{W_i}\,x_i. Using u_i on both sides multiplies the variance by W_i and puts you on the wrong scale.
	3.	ALO ratio identity depends on (2) and on penalty.
The often-quoted inflation
\frac{\text{SE}{\text{LOO}}}{\text{SE}{\text{full}}} \;\approx\; \frac{1}{\sqrt{1-a_{ii}}}
holds exactly for unpenalized least squares (and still works well for GLMs when you keep U=\sqrt{W}X and project the SE with x_i). If your fitter sneaks in a tiny ridge/jitter (very common, e.g. \lambda I for stability), the equality isn’t exact; with random data, a 1e-9 tolerance is unrealistically tight.

⸻

Minimal fixes (no hacks — just matching the math)
	•	Build U with the working weights actually used by the fit:

// from the fit: fitted probs mu_i
let p_hat = fit_res.mu.clone(); // or whatever holds the fitted μ
let w_work = &w * &(&p_hat * &(1.0 - &p_hat)); // elementwise
let sqrt_w_work = w_work.mapv(f64::sqrt);

// U = sqrt(W_work) * X_transformed
let mut U = fit_res.x_transformed.clone();
U *= &sqrt_w_work.view().insert_axis(Axis(1));

let XtWX = U.t().dot(&U);
let K = fit_res.penalized_hessian_transformed.clone(); // matches the fitter
let K_f = FaerMat::<f64>::from_fn(p, p, |i, j| K[[i, j]]);
let factor = FaerLlt::new(K_f.as_ref(), Side::Lower).unwrap();


	•	Use ui for leverage, xi for SE:

for i in 0..n {
    let xi = fit_res.x_transformed.row(i).to_owned(); // unweighted row (η-scale)
    let ui = U.row(i).to_owned();                      // weighted row (for leverage)

    // leverage a_ii = u_i^T K^{-1} u_i
    let rhs_u = FaerMat::<f64>::from_fn(p, 1, |r, _| ui[r]);
    let s_u = factor.solve(rhs_u.as_ref());
    let aii = ui.iter().zip(0..p).map(|(u_r, r)| u_r * s_u[(r, 0)]).sum::<f64>();

    // var_full(η_i) = x_i^T K^{-1} (X^T W X) K^{-1} x_i
    let rhs_x = FaerMat::<f64>::from_fn(p, 1, |r, _| xi[r]);
    let s_x = factor.solve(rhs_x.as_ref());
    let s_x_arr = Array1::from_shape_fn(p, |j| s_x[(j, 0)]);
    let ti = XtWX.dot(&s_x_arr);
    let var_full = s_x_arr.dot(&ti);

    let denom = (1.0 - aii).max(1e-12);
    let se_full = var_full.sqrt();
    let se_loo_manual = (var_full / denom).sqrt();

    // compare to ALO (η-scale)
    let alo_se_eta = alo_features.se[i]; // assumes η-scale; if response-scale, divide by μ_i(1-μ_i)
    assert!((alo_se_eta - se_loo_manual).abs() <= 1e-7 * (1.0 + se_loo_manual.abs()));
    assert!(((se_loo_manual / se_full) - denom.sqrt().recip()).abs() <= 1e-7);
}


	•	If compute_alo_features.se is on the response scale (probability), convert to η-scale before comparing:

let gprime = (p_hat[i] * (1.0 - p_hat[i])).max(1e-12);
let alo_se_eta = alo_features.se[i] / gprime;


	•	If your fitter adds a small ridge by default, either set it to zero in the fit used by the test, or keep it and relax the tolerance to ~1e-6–1e-7 to account for the (real) deviation from the unpenalized identity.

⸻

Why your current assertions fail
	•	a_{ii} is computed with the wrong U (missing \mu(1-\mu)), so your 1/sqrt(1-a_ii) is the wrong inflation.
	•	Your “manual” SE uses u_i instead of x_i, putting it on a weighted scale, while ALO’s se is (almost certainly) on the η-scale.
	•	Any tiny ridge in K makes the identity exactness go away; with a 1e-9 gate, even perfectly consistent code will trip due to numerical and modeling differences.

Fix the weight convention + projection vector as above, and the test will line up without hacks.
