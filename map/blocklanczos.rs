//! Adaptive randomized block-Lanczos eigensolver for the implicit sample
//! covariance operator.
//!
//! # Why this exists
//!
//! The fit's expensive resource is **one traversal of the genotype data**, not
//! arithmetic. `C = XXᵀ/(n−1)` is never formed; the covariance operator instead
//! evaluates `C·Q` for a whole block of `b` sample-space vectors in a single
//! streamed pass over the variants (`Xᵀ Q` per variant tile, then `X · proj`).
//!
//! A vector-at-a-time Krylov solver throws that away: faer's Arnoldi expands its
//! subspace one column per operator application (`iterate_arnoldi`), so a
//! 64-dimensional Krylov space costs 64 whole-genome passes before restarts. The
//! arithmetic is trivial; the passes are not.
//!
//! This solver advances every requested component per pass. Each iteration is
//! exactly one `C·Q` — one genome pass — and grows the Krylov space by `b`
//! dimensions rather than by 1. Getting 20 PCs typically costs a handful of
//! passes rather than dozens.
//!
//! # The method
//!
//! Block Arnoldi restricted to a self-adjoint operator — i.e. block Lanczos with
//! full reorthogonalization — with Rayleigh–Ritz extraction and residual-driven
//! stopping: the randomized block-Krylov family (Rokhlin–Szlam–Tygert; Musco &
//! Musco), specialized to a genotype operator whose block application is one
//! sequential pass.
//!
//! Starting from an orthonormal random block `Q₀`, it builds
//! `K = [Q₀, Q₁, …, Q_j]` by orthogonalizing each operator image against
//! *everything already retained* and **keeping the coefficients**:
//!
//! ```text
//!   Z          = C·Q_j                       (one genome pass)
//!   for i ≤ j:   H_ij ← Q_iᵀ Z ; Z ← Z − Q_i H_ij     (twice, accumulating H)
//!   Q_{j+1} B_j = qr(Z)
//! ```
//!
//! so that, for the arithmetic that *actually ran*,
//!
//! ```text
//!   C K = K H + Q_{j+1} B_j E_jᵀ
//! ```
//!
//! holds to the orthonormality of `[K | Q_{j+1}]` and nothing else.
//!
//! The textbook form of this recurrence is three-term: in exact arithmetic
//! `H_ij = 0` for `i < j−1`, so `H` is block tridiagonal and can be assembled
//! from `A_j = H_jj` and `B_j` alone. That structure is what buys a *short*
//! recurrence — the ability to forget the old blocks. This solver cannot forget
//! them: it reorthogonalizes against the whole basis every step precisely
//! because the floating-point `H_ij` are *not* zero, and it keeps the whole
//! basis anyway because the Ritz vectors are expressed in it. Storing the
//! coefficients it already computes therefore costs `O((jb)²)` numbers — a few
//! hundred square, free against a genome pass — and buys a projected operator
//! `T` that is the real `KᵀCK` rather than an idealization of it.
//!
//! `T` is the symmetric part of `H`; `C` is self-adjoint, so the asymmetry is
//! pure floating-point error, and forcing the symmetry is what lets the dense
//! solve be a symmetric eigendecomposition. The discarded skew part is not
//! thrown away — it reappears below, in the residual.
//!
//! # Stopping without extra passes
//!
//! With `u = K s` and `T s = θ s`, the recurrence above gives
//!
//! ```text
//!   C u − θ u = K (H s − θ s) + Q_{j+1} B_j s_tail
//! ```
//!
//! and the two terms are orthogonal, so
//!
//! ```text
//!   ‖C u − θ u‖² = ‖H s − θ s‖² + ‖B_j s_tail‖².
//! ```
//!
//! Both terms live in the projected space. So the residual of every Ritz pair
//! is available exactly, for the algorithm that ran, at no genotype I/O — and
//! `H s − θ s = (H − T)s` is precisely the term a block-tridiagonal assembly
//! drops. Convergence is *measured*, never assumed, which is what makes an
//! adaptive pass count possible: depth grows only while the residual says it
//! must, instead of running a fixed iteration count chosen for someone else's
//! dataset.
//!
//! A second, complementary criterion tracks the *subspace* rather than
//! individual vectors: the mean explained variance
//! `MEV = ‖U_prevᵀ U‖_F² / k` between successive Ritz bases. Nearly degenerate
//! PCs rotate freely within their eigenspace, so a per-vector test can report
//! non-convergence forever on a subspace that is in fact settled; MEV is
//! invariant to that rotation. Both must pass.
//!
//! While the basis is growing, `K_new`'s leading columns *are* `K_old`, so the
//! overlap `U_prevᵀ U` equals `S_prevᵀ S` with `S_prev` zero-padded — a few
//! hundred square rather than `n × k`. The `n`-sized lift is deferred to the
//! moments that genuinely need vectors: returning, and restarting.
//!
//! # Clustered spectra
//!
//! Oversampling means `θ_{k+1}` is always computed, so the solver can see the
//! `k/k+1` gap it is being asked to split. When that gap is tight the requested
//! top-k is not a well-conditioned object on its own, and more depth does not
//! fix it: there is no unique k-dimensional invariant subspace to converge to.
//! Two things follow. The solver widens its guard band once — extra columns cost
//! arithmetic *inside* a pass, while extra depth costs another pass — and it
//! certifies the span of the whole straddling cluster rather than of exactly `k`
//! columns, reporting through [`BlockKrylovOutcome::truncation_splits_cluster`]
//! that the requested truncation cuts an eigenspace in half.

use faer::linalg::matmul::matmul;
use faer::prelude::{IntoConst, Reborrow, ReborrowMut};
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};
use std::fmt;

/// Largest dense Rayleigh–Ritz problem to assemble, in dimensions. The
/// projected problem is `b(j+1)` square; past this the basis is restarted
/// instead of grown, which bounds both the dense solve and the basis memory.
const MAX_PROJECTED_DIM: usize = 1024;

/// Floor for relative-residual denominators, so a zero Ritz value cannot turn a
/// finite residual into an infinite relative one. Mirrors the variance epsilon
/// the HWE scaler uses; kept local so this module stays self-contained and
/// testable against a dense operator with no genotype machinery in scope.
const RITZ_SCALE_FLOOR: f64 = 1.0e-12;

/// Columns orthogonalized as one panel in [`orthonormalize`].
///
/// Everything *across* panels is `matmul`; only the `p × p` interior of a panel
/// runs scalar. Widening the panel moves work out of GEMM and back into scalar
/// loops, narrowing it multiplies the number of GEMM calls over the finished
/// columns. Eight is near the minimum of that trade for the block widths this
/// solver uses (16–64) and sample counts in the hundreds of thousands.
const ORTHO_PANEL: usize = 8;

/// A covariance operator whose block application costs one data pass.
pub trait BlockOperator {
    type Error;

    /// Dimension of the (square, self-adjoint, positive semi-definite) operator.
    fn dim(&self) -> usize;

    /// `out ← C · q`, for all columns of `q` in a single pass over the data.
    ///
    /// The column count is free in *passes*, not in arithmetic: an
    /// implementation must serve any `b` in one traversal, but the traversal
    /// itself is `O(n·p·b)` for `p` variants. Which of the two dominates decides
    /// how much oversampling is worth buying. When the pass is I/O- or
    /// decode-bound — a compressed genotype file read from disk — widening the
    /// block is very nearly free, and oversampling aggressively is the right
    /// trade against another pass. When it is GEMM-bound — genotypes already
    /// resident and cheap to decode — doubling `b` roughly doubles the dominant
    /// arithmetic, and a wider block has to earn its keep against the pass it
    /// saves.
    fn apply_block(&self, out: MatMut<'_, f64>, q: MatRef<'_, f64>) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub enum BlockKrylovError<E> {
    Operator(E),
    Eigen(String),
    Invalid(&'static str),
}

impl<E: fmt::Display> fmt::Display for BlockKrylovError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator(err) => write!(f, "covariance operator failed: {err}"),
            Self::Eigen(msg) => write!(f, "projected eigendecomposition failed: {msg}"),
            Self::Invalid(msg) => write!(f, "{msg}"),
        }
    }
}

/// Tuning for [`block_krylov_eigen`].
#[derive(Clone, Copy, Debug)]
pub struct BlockKrylovParams {
    /// Columns per block. Oversampling beyond `k` buys convergence within a
    /// pass rather than across passes.
    pub block_width: usize,
    /// Minimum operator applications before convergence is tested. Two gives
    /// the Rayleigh–Ritz problem a Krylov space of `2b` to work with.
    pub min_passes: usize,
    /// Hard ceiling on operator applications (genome passes).
    pub max_passes: usize,
    /// Relative Ritz-residual tolerance, `‖Cu − θu‖ / θ`.
    pub residual_tol: f64,
    /// Tolerance on `1 − MEV` between successive certified Ritz subspaces.
    pub mev_tol: f64,
    /// Byte budget for the retained Krylov basis; hitting it forces a restart.
    pub basis_budget_bytes: usize,
    /// Gap ratio `(θ_k − θ_{k+1}) / θ_k` below which the requested boundary is
    /// treated as spectrally clustered, triggering one guard-band widening
    /// rather than further depth, and certification of the cluster's span
    /// rather than of exactly `k` columns.
    pub cluster_gap: f64,
    /// Fixed PRNG seed. Fits must be reproducible run to run.
    pub seed: u64,
}

impl BlockKrylovParams {
    /// Default ceiling on genome passes; see [`Self::auto`].
    pub const DEFAULT_MAX_PASSES: usize = 32;

    /// Defaults derived from the request and the machine, not from constants
    /// baked in for someone else's cohort.
    ///
    /// Oversampling is `k + min(32, max(8, k/2))`: a flat `k + 10` is thin for
    /// large `k`, and unbounded `1.5k` wastes arithmetic for very large `k`.
    ///
    /// `max_passes` is a **pass economy, not a fallback**. Each pass is a whole
    /// traversal of a logical matrix that can run to terabytes, so a ceiling
    /// generous enough to "eventually get there" buys a multi-hour run instead
    /// of an answer — which is why the ceiling is reported pass by pass, and
    /// why a fit that reaches it is refused rather than shipped. But the
    /// ceiling has to cover the requests that *are* well posed. Eight passes
    /// covers continental structure, whose leading eigenvalues stand clear of
    /// the bulk; a within-population fit asking for sixteen components sits on
    /// a spectrum that flattens into noise almost immediately, and its
    /// residual shrinks by roughly a factor of two a pass. Observed on a
    /// 222k-sample European panel: `1.6e-2` after eight passes, still falling.
    /// Thirty-two reaches `1e-6` on that shape with margin; a request still
    /// short at thirty-two is a diagnosis, not a case for more scanning. The
    /// three causes, in order of likelihood: the requested `k` lands inside a
    /// near-degenerate cluster (see
    /// [`BlockKrylovOutcome::truncation_splits_cluster`] and `boundary_gap` —
    /// ask for a different `k`, not for more passes); the block is too narrow
    /// for the spectrum's decay, so each pass buys too little (widen
    /// `block_width`); or `residual_tol` is tighter than the operator's own
    /// accuracy can support.
    pub fn auto(k: usize, dim: usize, basis_budget_bytes: usize) -> Self {
        let oversample = 32.min(8.max(k.div_ceil(2)));
        let block_width = (k + oversample).clamp(1, dim.max(1));
        Self {
            block_width,
            min_passes: 2,
            max_passes: Self::DEFAULT_MAX_PASSES,
            residual_tol: 1e-6,
            mev_tol: 1e-6,
            basis_budget_bytes,
            cluster_gap: 1e-3,
            seed: 0x243F_6A88_85A3_08D3,
        }
    }
}

/// Converged eigenpairs plus everything needed to explain the run.
#[derive(Debug)]
pub struct BlockKrylovOutcome {
    /// Ritz values, descending.
    pub values: Vec<f64>,
    /// Ritz vectors, one column each, matching `values`.
    pub vectors: Mat<f64>,
    /// Operator applications performed, i.e. genome passes.
    pub passes: usize,
    /// Whether both stopping criteria were met.
    pub converged: bool,
    /// Worst relative Ritz residual over the *certified* pairs — the returned
    /// ones, plus any straddling cluster they were certified with.
    pub max_relative_residual: f64,
    /// `1 − MEV` between the final two certified subspaces, once measurable.
    pub subspace_delta: f64,
    /// `(θ_k − θ_{k+1}) / θ_k` at the requested boundary, if it was observed.
    pub boundary_gap: Option<f64>,
    /// Times the basis was restarted, under the memory budget or to widen.
    pub restarts: usize,
    /// Leading Ritz pairs the stopping criteria were applied to. Equal to
    /// `values.len()` unless a cluster straddles the requested boundary, in
    /// which case the whole cluster was certified and only `k` returned.
    pub certified_components: usize,
    /// Set when `θ_k` and `θ_{k+1}` are closer than `cluster_gap`: the
    /// requested truncation cuts through a near-degenerate eigenspace, so which
    /// directions land inside the returned `k` is arbitrary within that
    /// eigenspace. The span is converged; the `k`-th column individually is not
    /// a meaningful object, and no amount of further work makes it one.
    pub truncation_splits_cluster: bool,
}

/// One Rayleigh–Ritz step's answer, still in coefficient space.
///
/// The `n`-sized lift is the expensive part of an iteration once `n` is half a
/// million, and nothing inside the loop needs it: residuals come from the
/// projection, and successive subspaces can be compared through their
/// coefficients while the basis is only growing. So the loop carries this, and
/// [`finish`] pays for vectors at the exits.
struct Stage {
    /// Ritz values for the returned pairs, descending.
    values: Vec<f64>,
    /// Ritz coefficients for the retained guard band, `dim × k_retained`.
    coefficients: Mat<f64>,
    /// Columns returned to the caller (`k`, or fewer if the projected problem
    /// is smaller).
    output: usize,
    /// Columns the stopping criteria were applied to; `>= output`.
    certified: usize,
    passes: usize,
    converged: bool,
    max_relative_residual: f64,
    subspace_delta: f64,
    boundary_gap: Option<f64>,
    splits_cluster: bool,
}

/// The previous iteration's certified subspace, in whichever representation is
/// still valid.
enum PreviousTop {
    None,
    /// Coefficients in the *current* Krylov basis. While that basis only grows,
    /// `K_new`'s leading `dim` columns are exactly `K_old`, so zero-padding
    /// these expresses the same vectors in the new basis and the overlap is a
    /// projected-size GEMM.
    Coefficients { s: Mat<f64>, dim: usize },
    /// Sample-space copy. A restart rebuilds the basis from scratch, so the
    /// prefix relation is gone and the comparison has to happen in `n`. That
    /// costs one lift per restart, not one per pass.
    Lifted(Mat<f64>),
}

/// Top-`k` eigenpairs of `op` by adaptive randomized block Lanczos.
pub fn block_krylov_eigen<Op: BlockOperator>(
    op: &Op,
    k: usize,
    params: BlockKrylovParams,
    par: Par,
) -> Result<BlockKrylovOutcome, BlockKrylovError<Op::Error>> {
    let n = op.dim();
    if n == 0 || k == 0 {
        return Ok(BlockKrylovOutcome {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
            passes: 0,
            converged: true,
            max_relative_residual: 0.0,
            subspace_delta: 0.0,
            boundary_gap: None,
            restarts: 0,
            certified_components: 0,
            truncation_splits_cluster: false,
        });
    }
    if k > n {
        return Err(BlockKrylovError::Invalid(
            "requested more components than the operator's dimension",
        ));
    }

    // A block wider than half the dense ceiling leaves room for a single block,
    // which is subspace iteration rather than a Krylov space. Cap the width so
    // two blocks always fit under `MAX_PROJECTED_DIM`; oversampling is the part
    // that gives way, since it is the part that was discretionary.
    let width_cap = n.min((MAX_PROJECTED_DIM / 2).max(1));
    let mut width = params.block_width.clamp(1, width_cap);
    let mut widened = false;

    let mut passes = 0usize;
    let mut restarts = 0usize;
    let mut previous = PreviousTop::None;
    let mut best: Option<BlockKrylovOutcome> = None;

    let mut start = random_start_block(n, width, params.seed);
    orthonormalize(start.as_mut(), None, par);

    'restart: loop {
        // Depth limits, in bytes, recomputed here rather than once up front:
        // `width` grows when the guard band widens, and a budget derived from
        // the original width stops describing the allocation being made.
        //
        // This bounds the *retained basis* only. The true peak also carries the
        // trailing block `Q_{j+1}`, the start block's clone, the operator image
        // `Z`, the previous certified Ritz matrix and whatever scratch
        // `apply_block` needs — several more `n × width` objects. Treat this as
        // the term that grows with depth, not as the process's high-water mark.
        let block_bytes = n
            .saturating_mul(width)
            .saturating_mul(std::mem::size_of::<f64>());
        let budget_depth = if block_bytes == 0 {
            usize::MAX
        } else {
            // Two blocks is the smallest thing that is a Krylov space at all,
            // so the byte budget may not force fewer. The dense ceiling below
            // still may: it is a hard limit on a cubic-time solve, where this
            // is a soft limit on memory that is already an underestimate.
            (params.basis_budget_bytes / block_bytes).max(2)
        };
        let projected_depth = (MAX_PROJECTED_DIM / width).max(1);
        let max_depth = budget_depth.min(projected_depth);
        let max_dim = max_depth * width;

        let mut blocks: Vec<Mat<f64>> = vec![start.clone()];
        // The projection, in full storage: `h[(i·b + r, l·b + c)]` is entry
        // `(r, c)` of `H_il`. Column block `l` is written once, at step `l`, and
        // never revised — which is what makes the residual identity below a
        // statement about the arithmetic that ran.
        let mut h = Mat::<f64>::zeros(max_dim, max_dim);
        let mut stage: Option<Stage> = None;

        for depth in 0..max_depth {
            if passes >= params.max_passes {
                break;
            }

            // --- the one expensive step: a single pass over the genotypes ---
            let mut z = Mat::<f64>::zeros(n, width);
            op.apply_block(z.as_mut(), blocks[depth].as_ref())
                .map_err(BlockKrylovError::Operator)?;
            passes += 1;

            // Scale reference for the exhaustion test, taken before anything is
            // subtracted: a Krylov space is exhausted when `C Q_j` lands
            // entirely inside the space already retained, which is a statement
            // about how much of `Z` *survives* the projection relative to how
            // much arrived — not about `Z`'s absolute size, which carries the
            // operator's units.
            let image_norm = frobenius_norm(z.as_ref());

            // Two rounds of block Gram–Schmidt against the whole retained
            // basis, keeping every coefficient. One round loses orthogonality
            // on a clustered spectrum, and the basis is what the Ritz vectors
            // are expressed in. This subsumes the three-term recurrence: the
            // rounds against `Q_j` and `Q_{j−1}` compute `A_j` and `B_{j−1}ᵀ`
            // from the vectors instead of assuming them, which is both cheaper
            // (two fewer products) and what makes `H` describe the executed
            // arithmetic.
            let col_base = depth * width;
            for _ in 0..2 {
                for (i, retained) in blocks.iter().enumerate() {
                    let mut coeff = Mat::<f64>::zeros(width, width);
                    matmul(
                        coeff.as_mut(),
                        Accum::Replace,
                        retained.as_ref().transpose(),
                        z.as_ref(),
                        1.0,
                        par,
                    );
                    matmul(
                        z.as_mut(),
                        Accum::Add,
                        retained.as_ref(),
                        coeff.as_ref(),
                        -1.0,
                        par,
                    );
                    let row_base = i * width;
                    for c in 0..width {
                        for r in 0..width {
                            h[(row_base + r, col_base + c)] += coeff[(r, c)];
                        }
                    }
                }
            }
            let residual_norm = frobenius_norm(z.as_ref());

            // Q_{j+1} B_j = qr(Z). The new block extends the Krylov basis that
            // the Ritz vectors are expressed in, so it must be retained.
            let mut beta = Mat::<f64>::zeros(width, width);
            let rank = orthonormalize(z.as_mut(), Some(beta.as_mut()), par);
            // `B_j` couples block `j` to block `j+1`. It is outside the window
            // the *current* Rayleigh–Ritz sees — that is exactly why it carries
            // the residual — and becomes an interior entry at the next step, so
            // it is written now and only while there is a block row to hold it.
            let beta_row = (depth + 1) * width;
            if beta_row + width <= max_dim {
                for c in 0..width {
                    for r in 0..width {
                        h[(beta_row + r, col_base + c)] = beta[(r, c)];
                    }
                }
            }
            blocks.push(z);

            // --- Rayleigh–Ritz on the true projected operator ---
            let dim = (depth + 1) * width;
            let window = h.as_ref().submatrix(0, 0, dim, dim);
            let t = symmetric_part(window);
            let (theta, s) = dense_eigen_desc(t.as_ref())?;

            let available = theta.len();
            let output = k.min(available);
            // Guard band: what is carried across a restart. The block width is
            // the natural size — it is exactly what a restart block holds, so
            // carrying this much means a restart needs no random padding at all
            // and loses nothing it had converged.
            let guard = width.max(output).min(available);
            let (certified, splits_cluster) =
                certification_span(&theta, output, guard, params.cluster_gap);

            let boundary_gap = if available > output
                && output > 0
                && theta[output - 1].abs() > RITZ_SCALE_FLOOR
            {
                Some((theta[output - 1] - theta[output]) / theta[output - 1].abs())
            } else {
                None
            };

            // Residuals from the projection alone — no extra genome pass.
            let residuals = ritz_residuals(
                window,
                &theta,
                beta.as_ref(),
                s.as_ref(),
                certified,
                width,
                par,
            );
            let mut max_relative = 0.0f64;
            for (idx, residual) in residuals.iter().enumerate() {
                let scale = theta[idx].abs().max(RITZ_SCALE_FLOOR);
                max_relative = max_relative.max(residual / scale);
            }

            let mut coefficients = Mat::<f64>::zeros(dim, guard);
            copy_into(&mut coefficients, s.as_ref(), dim, guard);

            // Compared on the leading columns the two iterations have in
            // common, not on an exact column-count match. `certified` is
            // cluster-aware, so it moves as Ritz values settle around the
            // requested boundary — and requiring the counts to be equal made
            // any movement fall through to infinity, which left `subspace_ok`
            // permanently false and meant convergence could never be declared
            // at all. Measured on a 250k-sample fit: `subspace_delta` came out
            // infinite after eight passes, so the run reported non-convergence
            // no matter how small its residuals were.
            //
            // The leading Ritz vectors are the stable part of both subspaces,
            // so comparing `min(prior, current)` of them is the meaningful
            // question — is the subspace we keep re-finding the same one — and
            // it stays well defined while the guard band breathes.
            let subspace_delta = match &previous {
                PreviousTop::Coefficients {
                    s: prior,
                    dim: prior_dim,
                } if *prior_dim <= dim && prior.ncols() > 0 => {
                    let shared_cols = prior.ncols().min(certified);
                    // `K_new`'s leading `prior_dim` columns are `K_old`, so
                    // truncating the new coefficients to those rows expresses
                    // both subspaces in the same basis.
                    let shared = coefficients
                        .as_ref()
                        .subcols(0, shared_cols)
                        .subrows(0, *prior_dim);
                    1.0 - mev(prior.as_ref().subcols(0, shared_cols), shared, par)
                }
                PreviousTop::Lifted(prior) if prior.ncols() > 0 => {
                    let shared_cols = prior.ncols().min(certified);
                    let lifted = lift_ritz_vectors(
                        &blocks,
                        coefficients.as_ref().subcols(0, shared_cols),
                        shared_cols,
                        width,
                        n,
                        par,
                    );
                    1.0 - mev(prior.as_ref().subcols(0, shared_cols), lifted.as_ref(), par)
                }
                // Only the first iteration has nothing to compare against.
                _ => f64::INFINITY,
            };
            let mut carried = Mat::<f64>::zeros(dim, certified);
            copy_into(&mut carried, coefficients.as_ref(), dim, certified);
            previous = PreviousTop::Coefficients { s: carried, dim };

            let residual_ok = max_relative <= params.residual_tol;
            let subspace_ok = subspace_delta <= params.mev_tol;
            let enough_passes = passes >= params.min_passes;
            let converged = residual_ok && subspace_ok && enough_passes;

            // A fully exhausted Krylov space is a *complete* answer, not a
            // failure: the basis already spans an invariant subspace, so no
            // further pass can add information. Partial rank deficiency is not
            // exhaustion — the surviving columns still carry the recurrence,
            // and the collapsed ones only make the residual estimate
            // conservative (‖Q v‖ ≤ ‖v‖ when Q has zero columns), so the
            // residual test remains the authority on convergence.
            let exhausted = rank == 0
                || residual_norm <= f64::EPSILON * 64.0 * (dim as f64).sqrt() * image_norm;

            let current = Stage {
                values: theta[..output].to_vec(),
                coefficients,
                output,
                certified,
                passes,
                converged: converged || exhausted,
                max_relative_residual: max_relative,
                subspace_delta,
                boundary_gap,
                splits_cluster,
            };

            if converged || exhausted {
                let (outcome, _) = finish(&blocks, &current, width, n, restarts, par);
                return Ok(outcome);
            }

            // A clustered k/k+1 boundary is not fixed by more depth: PC k and
            // PC k+1 are rotating inside one near-degenerate eigenspace. Widen
            // the guard band once instead — extra columns are paid for inside a
            // pass, extra depth costs a whole new one.
            if !widened
                && enough_passes
                && residual_ok
                && boundary_gap.is_some_and(|gap| gap < params.cluster_gap)
                && width < width_cap
            {
                let (outcome, retained) = finish(&blocks, &current, width, n, restarts, par);
                previous = PreviousTop::Lifted(leading_columns(&retained, current.certified));
                best = Some(outcome);
                widened = true;
                width = (width + width.div_ceil(2)).min(width_cap);
                start = restart_block(&retained, n, width, params.seed, restarts, par);
                restarts += 1;
                continue 'restart;
            }

            stage = Some(current);

            if depth + 1 >= max_depth && passes < params.max_passes {
                // Memory or dense-solve ceiling reached with work left to do:
                // restart from the current Ritz block. The guard band goes with
                // it, so what the basis had converged is kept rather than
                // regenerated from the same deterministic padding.
                let current = stage.take().expect("stage was set immediately above");
                let (outcome, retained) = finish(&blocks, &current, width, n, restarts, par);
                previous = PreviousTop::Lifted(leading_columns(&retained, current.certified));
                best = Some(outcome);
                start = restart_block(&retained, n, width, params.seed, restarts, par);
                restarts += 1;
                continue 'restart;
            }
        }

        // The pass ceiling stopped the depth loop. The basis is still alive
        // here and about to be dropped, so this is the last chance to lift.
        if let Some(current) = stage.as_ref() {
            let (outcome, _) = finish(&blocks, current, width, n, restarts, par);
            best = Some(outcome);
        }
        break;
    }

    best.map(|outcome| BlockKrylovOutcome { restarts, ..outcome })
        .ok_or(BlockKrylovError::Invalid(
            "block Krylov solver produced no Ritz pairs",
        ))
}

/// Lift the projected answer into sample space.
///
/// This is the `n × k` work the iteration defers: it happens when the solver
/// returns or restarts, not once per pass. Returns the outcome and the full
/// retained guard band, of which the outcome's vectors are the leading columns.
fn finish(
    blocks: &[Mat<f64>],
    stage: &Stage,
    width: usize,
    n: usize,
    restarts: usize,
    par: Par,
) -> (BlockKrylovOutcome, Mat<f64>) {
    let guard = stage.coefficients.ncols();
    let retained = lift_ritz_vectors(blocks, stage.coefficients.as_ref(), guard, width, n, par);
    let vectors = leading_columns(&retained, stage.output);
    let outcome = BlockKrylovOutcome {
        values: stage.values.clone(),
        vectors,
        passes: stage.passes,
        converged: stage.converged,
        max_relative_residual: stage.max_relative_residual,
        subspace_delta: stage.subspace_delta,
        boundary_gap: stage.boundary_gap,
        restarts,
        certified_components: stage.certified,
        truncation_splits_cluster: stage.splits_cluster,
    };
    (outcome, retained)
}

/// How many leading Ritz pairs have to converge *together* for the requested
/// top-`k` to mean anything.
///
/// If `λ_k ≈ λ_{k+1}` there is no unique k-dimensional invariant subspace: the
/// two directions rotate freely inside one eigenspace, so a subspace test on
/// exactly `k` columns can oscillate forever while the eigenproblem is in fact
/// solved. What *is* well defined is the span of the whole cluster, so that is
/// what gets certified — and the caller is told separately that the truncation
/// it asked for cuts an eigenspace, because that is a property of the question
/// rather than a failure of the answer.
///
/// Returns `(columns to certify, whether the boundary splits a cluster)`.
fn certification_span(theta: &[f64], k: usize, cap: usize, gap_tol: f64) -> (usize, bool) {
    let available = theta.len();
    if k == 0 || k >= available {
        return (k.min(available), false);
    }
    if theta[0].abs() <= RITZ_SCALE_FLOOR {
        // No usable spectral scale: every gap is "tiny" and the statement is
        // empty. A null operator is not a cluster.
        return (k, false);
    }

    let relative_gap = |i: usize| {
        let scale = theta[i].abs().max(RITZ_SCALE_FLOOR);
        (theta[i] - theta[i + 1]) / scale
    };
    if relative_gap(k - 1) >= gap_tol {
        return (k, false);
    }

    let mut end = k;
    while end < available {
        if relative_gap(end - 1) >= gap_tol {
            break;
        }
        end += 1;
        if end > cap {
            // The cluster runs past the guard band this run is carrying, so
            // there is no cluster-level certificate on offer. Certify what was
            // asked for, and let the flag say why it may never settle.
            return (k, true);
        }
    }
    (end, true)
}

/// Deterministic pseudo-random start block with the constant direction removed.
///
/// The standardized genotype columns are centered, so `Xᵀ1 = 0` and `C·1 = 0`:
/// the all-ones vector spans the operator's null direction. Any component along
/// it is wasted — worse, seeding *with* it makes the whole Krylov sequence zero
/// in exact arithmetic. Projecting it out costs one pass over the block.
///
/// SplitMix64 rather than `rand`, so a fit reproduces bit-for-bit across runs,
/// machines and dependency bumps.
fn random_start_block(n: usize, width: usize, seed: u64) -> Mat<f64> {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 / (1u64 << 53) as f64).mul_add(2.0, -1.0)
    };

    let mut block = Mat::<f64>::zeros(n, width);
    for col in 0..width {
        for row in 0..n {
            block[(row, col)] = next();
        }
    }
    remove_constant_direction(block.as_mut());
    block
}

/// Rebuild a start block from the retained Ritz directions, padding with fresh
/// pseudo-random ones only where the block is wider than what was retained.
///
/// The guard band is the point. The widening path fires exactly when
/// `θ_k ≈ θ_{k+1}` — it has just concluded that PC k and PC k+1 are one cluster
/// — so carrying only the first `k` would discard the approximation to the very
/// direction that triggered it. The memory-driven restart gains the same way:
/// it now carries whatever the basis had converged, instead of regenerating the
/// same deterministic padding on every restart.
///
/// Only the padding has its constant direction removed. The retained columns
/// are approximate eigenvectors of the operator being solved; projecting them
/// perturbs the answer rather than cleaning it up.
fn restart_block(
    retained: &Mat<f64>,
    n: usize,
    width: usize,
    seed: u64,
    restart_index: usize,
    par: Par,
) -> Mat<f64> {
    let carried = retained.ncols().min(width);
    let mut block = Mat::<f64>::zeros(n, width);
    for col in 0..carried {
        for row in 0..n {
            block[(row, col)] = retained[(row, col)];
        }
    }
    if carried < width {
        // The padding is stirred by the restart index so that two restarts do
        // not explore the same fresh directions, while a rerun of the same fit
        // still reproduces bit for bit.
        let padding_seed = seed
            .wrapping_add(0x51_7C_C1_B7_27_22_0A_95)
            .wrapping_add((restart_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let padding = random_start_block(n, width - carried, padding_seed);
        for col in carried..width {
            for row in 0..n {
                block[(row, col)] = padding[(row, col - carried)];
            }
        }
    }
    orthonormalize(block.as_mut(), None, par);
    block
}

fn remove_constant_direction(mut block: MatMut<'_, f64>) {
    let n = block.nrows();
    if n == 0 {
        return;
    }
    for col in 0..block.ncols() {
        let mut sum = 0.0;
        for row in 0..n {
            sum += block[(row, col)];
        }
        let mean = sum / n as f64;
        for row in 0..n {
            block[(row, col)] -= mean;
        }
    }
}

/// Block Gram–Schmidt QR: writes the upper-triangular factor into `r` when
/// requested and returns the numerical rank.
///
/// Why blocked rather than the textbook column loop: `n` is the sample count,
/// 100k–500k in a real fit, and a scalar modified Gram–Schmidt sweeps the whole
/// `n × b` block once per (column, earlier-column) pair. That is `O(n b²)` in
/// serial scalar loads with no parallelism — at `n = 500k` and `b = 60`, tens of
/// gigabytes of memory traffic, which can rival the genome pass this function
/// exists to protect. So columns are processed in narrow panels: everything
/// *across* panels is two `matmul` calls per round, and only the `p × p`
/// interior of a panel stays scalar.
///
/// Across panels the scheme is BCGS2 — block classical Gram–Schmidt applied
/// twice — which is the form with an `O(ε)` orthogonality bound; a single sweep
/// does not have one, and the regime a converging Krylov space enters is
/// exactly the near-rank-deficient one where that matters.
///
/// Three properties callers depend on, none of which the blocking may lose:
///
/// * `Z = Q R` for the returned `R`. `R` *is* the `B_j` of the recurrence, and
///   the residual certificate is stated in terms of it.
/// * Reorthogonalization, above.
/// * Rank deficiency reported rather than normalized away: a collapsed column
///   is how an exhausted Krylov space announces itself, and normalizing noise
///   would hand the basis a direction that means nothing.
///
/// The collapse threshold is relative to the block's own largest column. `Z` is
/// `C·Q` and carries the operator's units, so a fixed absolute threshold
/// quietly changes meaning when the covariance is rescaled.
fn orthonormalize(mut block: MatMut<'_, f64>, mut r: Option<MatMut<'_, f64>>, par: Par) -> usize {
    let n = block.nrows();
    let width = block.ncols();
    if let Some(r) = r.as_mut() {
        r.fill(0.0);
    }
    if n == 0 || width == 0 {
        return 0;
    }

    let mut scale = 0.0f64;
    for col in 0..width {
        let mut square = 0.0;
        for row in 0..n {
            let value = block[(row, col)];
            square += value * value;
        }
        scale = scale.max(square.sqrt());
    }
    let drop_tol = f64::EPSILON * 64.0 * (n as f64).sqrt() * scale;

    let mut rank = 0usize;
    let mut start = 0usize;
    while start < width {
        let panel = ORTHO_PANEL.min(width - start);

        if start > 0 {
            let mut coeff = Mat::<f64>::zeros(start, panel);
            for _ in 0..2 {
                let (done, current) = block.rb_mut().split_at_col_mut(start);
                let done = done.into_const();
                let mut current = current.subcols_mut(0, panel);
                matmul(
                    coeff.as_mut(),
                    Accum::Replace,
                    done.transpose(),
                    current.rb(),
                    1.0,
                    par,
                );
                matmul(current.rb_mut(), Accum::Add, done, coeff.as_ref(), -1.0, par);
                if let Some(r) = r.as_mut() {
                    for c in 0..panel {
                        for p in 0..start {
                            r[(p, start + c)] += coeff[(p, c)];
                        }
                    }
                }
            }
        }

        // Interior of the panel: the finished columns to its left are already
        // orthonormal and are not revisited, so this is plain MGS over at most
        // `ORTHO_PANEL` columns, twice for the same reason as above.
        for col in start..(start + panel) {
            for _ in 0..2 {
                for prev in start..col {
                    let mut dot = 0.0;
                    for row in 0..n {
                        dot += block[(row, prev)] * block[(row, col)];
                    }
                    if dot != 0.0 {
                        for row in 0..n {
                            let shift = dot * block[(row, prev)];
                            block[(row, col)] -= shift;
                        }
                        if let Some(r) = r.as_mut() {
                            r[(prev, col)] += dot;
                        }
                    }
                }
            }

            let mut norm_square = 0.0;
            for row in 0..n {
                norm_square += block[(row, col)] * block[(row, col)];
            }
            let norm = norm_square.sqrt();
            if let Some(r) = r.as_mut() {
                r[(col, col)] = norm;
            }

            // A NaN column collapses rather than being normalized into one.
            if norm.is_nan() || norm <= drop_tol {
                for row in 0..n {
                    block[(row, col)] = 0.0;
                }
                continue;
            }

            let inv = norm.recip();
            for row in 0..n {
                block[(row, col)] *= inv;
            }
            rank += 1;
        }

        start += panel;
    }

    rank
}

fn frobenius_norm(mat: MatRef<'_, f64>) -> f64 {
    let mut total = 0.0;
    for col in 0..mat.ncols() {
        for row in 0..mat.nrows() {
            total += mat[(row, col)] * mat[(row, col)];
        }
    }
    total.sqrt()
}

fn copy_into(dst: &mut Mat<f64>, src: MatRef<'_, f64>, rows: usize, cols: usize) {
    for col in 0..cols {
        for row in 0..rows {
            dst[(row, col)] = src[(row, col)];
        }
    }
}

fn leading_columns(mat: &Mat<f64>, cols: usize) -> Mat<f64> {
    let cols = cols.min(mat.ncols());
    let mut out = Mat::<f64>::zeros(mat.nrows(), cols);
    copy_into(&mut out, mat.as_ref(), mat.nrows(), cols);
    out
}

/// `T = (H + Hᵀ)/2`.
///
/// `C` is self-adjoint, so `H = KᵀCK` is symmetric in exact arithmetic and the
/// asymmetry is pure floating-point error. Forcing the symmetry is what lets
/// Rayleigh–Ritz be a *symmetric* eigendecomposition — real eigenvalues,
/// orthonormal `S`, both of which the rest of this module assumes. The part
/// removed here is not discarded: `H s − θ s = (H − T)s` is one of the two
/// terms of the residual, so it is measured rather than assumed away.
fn symmetric_part(h: MatRef<'_, f64>) -> Mat<f64> {
    let dim = h.nrows();
    let mut t = Mat::<f64>::zeros(dim, dim);
    for col in 0..dim {
        for row in 0..dim {
            t[(row, col)] = 0.5 * (h[(row, col)] + h[(col, row)]);
        }
    }
    t
}

fn dense_eigen_desc<E>(mat: MatRef<'_, f64>) -> Result<(Vec<f64>, Mat<f64>), BlockKrylovError<E>> {
    let eig = mat
        .self_adjoint_eigen(Side::Lower)
        .map_err(|err| BlockKrylovError::Eigen(format!("{err:?}")))?;
    let values = eig.S();
    let vectors = eig.U();
    let dim = mat.nrows();

    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sorted_values = Vec::with_capacity(dim);
    let mut sorted_vectors = Mat::<f64>::zeros(dim, dim);
    for (out_col, &src_col) in order.iter().enumerate() {
        sorted_values.push(values[src_col]);
        for row in 0..dim {
            sorted_vectors[(row, out_col)] = vectors[(row, src_col)];
        }
    }

    Ok((sorted_values, sorted_vectors))
}

/// Exact residual of every certified Ritz pair, from the projection alone.
///
/// Each step recorded `C Q_l = Σ_{i≤l} Q_i H_il + Q_{l+1} B_l` *as executed*,
/// coefficients and all, so over the retained basis `K`
///
/// ```text
///   C K = K H + Q_{j+1} B_j E_jᵀ
/// ```
///
/// describes the arithmetic that ran rather than its exact-arithmetic
/// idealization. With `u = K s` and `T s = θ s`,
///
/// ```text
///   C u − θ u = K (H s − θ s) + Q_{j+1} B_j s_tail
/// ```
///
/// and `[K | Q_{j+1}]` is orthonormal, so the two terms are orthogonal and
///
/// ```text
///   ‖C u − θ u‖² = ‖H s − θ s‖² + ‖B_j s_tail‖².
/// ```
///
/// The first term is exactly what a block-tridiagonal assembly drops: it is
/// `(H − T)s`, the asymmetry of the executed projection, zero only in exact
/// arithmetic. Keeping it is the difference between a certificate about this
/// run and a certificate about an idealized one.
///
/// One assumption survives — the orthonormality of `[K | Q_{j+1}]` — and that is
/// what the two rounds of reorthogonalization exist to hold at `O(ε)`. Columns
/// dropped by a rank-deficient QR only make the result conservative: `‖Q v‖ ≤
/// ‖v‖` when `Q` has zero columns.
///
/// Everything here is projected-size, a few hundred square, so the certificate
/// costs no genotype I/O.
fn ritz_residuals(
    h: MatRef<'_, f64>,
    theta: &[f64],
    beta: MatRef<'_, f64>,
    s: MatRef<'_, f64>,
    keep: usize,
    width: usize,
    par: Par,
) -> Vec<f64> {
    let dim = s.nrows();
    if keep == 0 {
        return Vec::new();
    }
    if dim < width {
        return vec![f64::INFINITY; keep];
    }
    let s_keep = s.subcols(0, keep);

    // Interior: ‖H s − θ s‖, i.e. ‖(H − T) s‖.
    let mut interior = Mat::<f64>::zeros(dim, keep);
    matmul(interior.as_mut(), Accum::Replace, h, s_keep, 1.0, par);
    for col in 0..keep {
        let value = theta[col];
        for row in 0..dim {
            interior[(row, col)] -= value * s_keep[(row, col)];
        }
    }

    // Trailing: ‖B_j s_tail‖, the part that leaves the retained basis.
    let mut trailing = Mat::<f64>::zeros(width, keep);
    matmul(
        trailing.as_mut(),
        Accum::Replace,
        beta,
        s_keep.subrows(dim - width, width),
        1.0,
        par,
    );

    (0..keep)
        .map(|col| {
            let mut square = 0.0;
            for row in 0..dim {
                square += interior[(row, col)] * interior[(row, col)];
            }
            for row in 0..width {
                square += trailing[(row, col)] * trailing[(row, col)];
            }
            square.sqrt()
        })
        .collect()
}

/// `U = K · S[:, ..keep]`, accumulated block by block so the full basis is never
/// copied into one matrix.
fn lift_ritz_vectors(
    blocks: &[Mat<f64>],
    s: MatRef<'_, f64>,
    keep: usize,
    width: usize,
    n: usize,
    par: Par,
) -> Mat<f64> {
    let mut out = Mat::<f64>::zeros(n, keep);
    for (j, block) in blocks.iter().enumerate() {
        let base = j * width;
        if base >= s.nrows() {
            break;
        }
        let rows = width.min(s.nrows() - base);
        let mut coeff = Mat::<f64>::zeros(rows, keep);
        for col in 0..keep {
            for row in 0..rows {
                coeff[(row, col)] = s[(base + row, col)];
            }
        }
        let block_view = block.as_ref().subcols(0, rows);
        matmul(out.as_mut(), Accum::Add, block_view, coeff.as_ref(), 1.0, par);
    }
    out
}

/// Mean explained variance between two orthonormal bases: `‖AᵀB‖_F² / k`.
///
/// Invariant to rotations *within* a nearly degenerate eigenspace, which is
/// what makes it the right convergence test for PCA — individual eigenvectors
/// of a clustered spectrum never settle, but the subspace they span does.
///
/// Both arguments may be coefficient matrices rather than sample-space ones:
/// for `U = K S` with `K` orthonormal and shared, `U_prevᵀ U = S_prevᵀ S`, so
/// the same number comes out of a projected-size product.
fn mev(previous: MatRef<'_, f64>, current: MatRef<'_, f64>, par: Par) -> f64 {
    let k = current.ncols();
    if k == 0 {
        return 1.0;
    }
    let mut overlap = Mat::<f64>::zeros(previous.ncols(), k);
    matmul(
        overlap.as_mut(),
        Accum::Replace,
        previous.transpose(),
        current,
        1.0,
        par,
    );
    let mut total = 0.0;
    for col in 0..overlap.ncols() {
        for row in 0..overlap.nrows() {
            total += overlap[(row, col)] * overlap[(row, col)];
        }
    }
    total / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Operator with an exactly known spectrum whose application costs `O(n·b)`.
    ///
    /// The suite has to run at a sample count where the block is genuinely much
    /// narrower than the space: that is the regime this solver is written for,
    /// and the only one in which "Krylov convergence" is being tested at all. A
    /// `V diag(λ) Vᵀ` built from a dense random `V` cannot go there — the setup
    /// alone is `O(n³)`.
    ///
    /// So `V = I − 2vvᵀ`, a Householder reflector: orthonormal and self-inverse,
    /// which makes the eigenvalues exactly `λ` and the eigenvectors exactly the
    /// columns of `V`, while an application is two rank-1 updates. A Krylov
    /// method is orthogonally equivariant and the start block is random, so a
    /// reflector is as general a basis as a Haar-random one for this purpose.
    ///
    /// `spectrum` must be non-increasing, so that eigenvalue `i` and
    /// `eigenvector(i)` are the `i`-th largest pair.
    struct ReflectedSpectrum {
        spectrum: Vec<f64>,
        reflector: Vec<f64>,
    }

    fn reflected(spectrum: Vec<f64>, seed: u64) -> ReflectedSpectrum {
        let n = spectrum.len();
        let raw = random_start_block(n, 1, seed);
        let mut square = 0.0;
        for row in 0..n {
            square += raw[(row, 0)] * raw[(row, 0)];
        }
        let inv = square.sqrt().recip();
        let reflector: Vec<f64> = (0..n).map(|row| raw[(row, 0)] * inv).collect();
        ReflectedSpectrum {
            spectrum,
            reflector,
        }
    }

    impl ReflectedSpectrum {
        /// The exact `i`-th eigenvector, `V e_i = e_i − 2 v_i v`.
        fn eigenvector(&self, i: usize) -> Vec<f64> {
            let n = self.spectrum.len();
            let scale = 2.0 * self.reflector[i];
            let mut out: Vec<f64> = (0..n).map(|row| -scale * self.reflector[row]).collect();
            out[i] += 1.0;
            out
        }

        fn exact_basis(&self, k: usize) -> Mat<f64> {
            let n = self.spectrum.len();
            let mut out = Mat::<f64>::zeros(n, k);
            for col in 0..k {
                let vector = self.eigenvector(col);
                for row in 0..n {
                    out[(row, col)] = vector[row];
                }
            }
            out
        }
    }

    impl BlockOperator for ReflectedSpectrum {
        type Error = std::convert::Infallible;

        fn dim(&self) -> usize {
            self.spectrum.len()
        }

        fn apply_block(
            &self,
            mut out: MatMut<'_, f64>,
            q: MatRef<'_, f64>,
        ) -> Result<(), Self::Error> {
            let n = self.spectrum.len();
            for col in 0..q.ncols() {
                let mut projection = 0.0;
                for row in 0..n {
                    projection += self.reflector[row] * q[(row, col)];
                }
                let mut reprojection = 0.0;
                for row in 0..n {
                    let reflected = q[(row, col)] - 2.0 * projection * self.reflector[row];
                    let scaled = reflected * self.spectrum[row];
                    out[(row, col)] = scaled;
                    reprojection += self.reflector[row] * scaled;
                }
                for row in 0..n {
                    out[(row, col)] -= 2.0 * reprojection * self.reflector[row];
                }
            }
            Ok(())
        }
    }

    /// `100·ratio^i + floor`, padded to `n` — a clean geometric head over a flat
    /// tail, non-increasing by construction.
    fn geometric(n: usize, ratio: f64, floor: f64) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 * ratio.powi(i as i32) + floor)
            .collect()
    }

    fn assert_values_match(values: &[f64], expected: &[f64], tolerance: f64) {
        for (idx, value) in values.iter().enumerate() {
            let target = expected[idx];
            assert!(
                (value - target).abs() <= tolerance * target.abs().max(1.0),
                "eigenvalue {idx}: got {value}, expected {target}"
            );
        }
    }

    /// True `max_i ‖C u_i − θ_i u_i‖ / |θ_i|`, measured with the operator.
    fn measured_relative_residual(op: &ReflectedSpectrum, outcome: &BlockKrylovOutcome) -> f64 {
        let u = &outcome.vectors;
        let mut image = Mat::<f64>::zeros(u.nrows(), u.ncols());
        op.apply_block(image.as_mut(), u.as_ref()).expect("apply");
        let mut worst = 0.0f64;
        for col in 0..u.ncols() {
            let value = outcome.values[col];
            let residual: f64 = (0..u.nrows())
                .map(|row| {
                    let r = image[(row, col)] - value * u[(row, col)];
                    r * r
                })
                .sum::<f64>()
                .sqrt();
            worst = worst.max(residual / value.abs().max(RITZ_SCALE_FLOOR));
        }
        worst
    }

    #[test]
    fn the_reported_residual_is_the_residual() {
        // The load-bearing test for the projected operator. `max_relative_residual`
        // is not advertised as a bound but as *the* residual of the pairs being
        // returned, so it is checked against `‖C u − θ u‖` measured directly —
        // and checked at several points along the trajectory, while the residual
        // is still large enough that the comparison discriminates. A projection
        // that describes an idealized recurrence rather than the executed one
        // shows up here, and nowhere else in the suite.
        let n = 1500;
        let k = 5;
        // Slow decay on purpose: the residual has to stay well above roundoff
        // for several passes or the comparison is vacuous.
        let op = reflected(geometric(n, 0.95, 1.0), 41);

        for budget in [2usize, 3, 4, 6] {
            let mut params = BlockKrylovParams::auto(k, n, 1 << 30);
            params.max_passes = budget;
            params.min_passes = budget;
            // Unsatisfiable, so the run always spends its whole budget and is
            // sampled mid-flight rather than at its own stopping point.
            params.residual_tol = 0.0;
            // Pin the certified set to exactly the returned columns, so the two
            // numbers being compared are maxima over the same pairs. Ritz values
            // this early in a run are not accurate enough to make the default
            // cluster test reproducible, and that is not what is under test here.
            params.cluster_gap = 0.0;
            let outcome = block_krylov_eigen(&op, k, params, Par::Seq).expect("solver runs");

            assert_eq!(outcome.passes, budget);
            assert_eq!(outcome.certified_components, k, "no cluster at this boundary");

            let measured = measured_relative_residual(&op, &outcome);
            assert!(
                measured > 1e-9,
                "budget {budget}: residual {measured} is already at roundoff, so this \
                 comparison would prove nothing"
            );
            assert!(
                (outcome.max_relative_residual - measured).abs() <= 1e-6 * measured,
                "budget {budget}: reported {} but the true residual is {measured}",
                outcome.max_relative_residual
            );
        }
    }

    #[test]
    fn recovers_a_well_separated_spectrum() {
        let n = 1500;
        let spectrum = geometric(n, 0.75, 1.0);
        let op = reflected(spectrum.clone(), 11);

        let params = BlockKrylovParams::auto(8, n, 1 << 30);
        let outcome = block_krylov_eigen(&op, 8, params, Par::Seq).expect("solver runs");

        assert!(outcome.converged, "solver should converge on a clean gap");
        assert_eq!(outcome.vectors.ncols(), 8);
        assert!(!outcome.truncation_splits_cluster);
        assert_values_match(&outcome.values, &spectrum, 1e-6);
    }

    #[test]
    fn costs_far_fewer_passes_than_the_krylov_dimension() {
        let n = 2000;
        let spectrum: Vec<f64> = (0..n)
            .map(|i| 50.0 * 0.5f64.powi(i.min(60) as i32))
            .collect();
        let op = reflected(spectrum.clone(), 12);

        let params = BlockKrylovParams::auto(10, n, 1 << 30);
        let outcome = block_krylov_eigen(&op, 10, params, Par::Seq).expect("solver runs");

        assert!(outcome.converged);
        // The whole point: the block is 18 wide, so six passes stand in for a
        // 108-dimensional Krylov space that a scalar Arnoldi would pay 108
        // operator applications to build.
        assert!(
            outcome.passes <= 6,
            "expected a handful of passes, took {}",
            outcome.passes
        );
        assert_values_match(&outcome.values, &spectrum, 1e-6);
    }

    #[test]
    fn an_exact_degeneracy_across_the_k_boundary_is_certified_as_a_cluster() {
        // λ_5 = λ_6 exactly, with the truncation asked for at k = 5. There is no
        // five-dimensional invariant subspace here at all: the fifth direction
        // is whatever rotation of the degenerate plane the arithmetic happened
        // to land on. The span of the cluster is the object that converges, and
        // the caller has to be told that its k cuts an eigenspace.
        let n = 1500;
        let mut spectrum = vec![100.0, 80.0, 60.0, 40.0, 25.0, 25.0, 12.0, 9.0, 7.0, 5.0];
        while spectrum.len() < n {
            let last = *spectrum.last().expect("non-empty");
            spectrum.push(last * 0.9);
        }
        let op = reflected(spectrum.clone(), 13);

        let mut params = BlockKrylovParams::auto(5, n, 1 << 30);
        // Loosened from the 1e-3 default only to decouple this test from *how
        // tightly* the two copies of 25 have merged at the iteration the solver
        // happens to stop on. What is under test is that a boundary inside one
        // eigenspace is certified across the eigenspace, not the calibration of
        // the threshold that recognizes one.
        params.cluster_gap = 1e-2;
        let outcome = block_krylov_eigen(&op, 5, params, Par::Seq).expect("solver runs");

        assert!(
            outcome.truncation_splits_cluster,
            "a k that lands inside an exact degeneracy must be reported as such"
        );
        assert!(
            outcome.certified_components >= 6,
            "the cluster, not the first five columns, is what was certified: got {}",
            outcome.certified_components
        );
        assert_eq!(outcome.vectors.ncols(), 5, "only k is returned");
        let gap = outcome.boundary_gap.expect("the boundary gap was observable");
        assert!(gap.abs() < params.cluster_gap, "boundary gap {gap}");
        assert!(outcome.converged, "the cluster's span still converges");
        assert_values_match(&outcome.values, &spectrum, 1e-6);
    }

    #[test]
    fn a_tiny_boundary_gap_extends_the_certificate_past_k() {
        // Not degenerate, just closer than `cluster_gap`: the same reasoning
        // applies, and the certificate has to reach the far side of the pair.
        let n = 1200;
        let mut spectrum = vec![
            120.0,
            90.0,
            70.0,
            55.0,
            42.0,
            30.0,
            30.0 * (1.0 - 1e-5),
            12.0,
            9.0,
            6.0,
        ];
        while spectrum.len() < n {
            let last = *spectrum.last().expect("non-empty");
            spectrum.push(last * 0.9);
        }
        let op = reflected(spectrum.clone(), 19);

        let mut params = BlockKrylovParams::auto(6, n, 1 << 30);
        // Comfortably above the 1e-5 gap being planted and comfortably below the
        // 0.6 gap on the far side of the pair, so which columns land in the
        // cluster is decided by the spectrum rather than by how far the Ritz
        // values have converged.
        params.cluster_gap = 1e-2;
        let outcome = block_krylov_eigen(&op, 6, params, Par::Seq).expect("solver runs");

        assert!(outcome.truncation_splits_cluster);
        assert!(
            outcome.certified_components >= 7,
            "the seventh pair is inside the boundary cluster: certified {}",
            outcome.certified_components
        );
        assert!(outcome.converged);
        assert_values_match(&outcome.values, &spectrum, 1e-6);
    }

    #[test]
    fn a_flat_spectrum_is_reported_honestly() {
        // The hard case: a slowly decaying tail with no usable gap anywhere,
        // i.e. fine within-population structure. Every boundary is clustered and
        // the cluster runs past anything the run is carrying, so there is no
        // cluster-level certificate to fall back on either. Either it converges,
        // or it says it did not — what it must never do is quietly return a
        // wrong subspace while claiming convergence.
        let n = 1000;
        let spectrum: Vec<f64> = (0..n).map(|i| 10.0 - 0.002 * i as f64).collect();
        let op = reflected(spectrum.clone(), 16);

        let params = BlockKrylovParams::auto(8, n, 1 << 30);
        let outcome = block_krylov_eigen(&op, 8, params, Par::Seq).expect("solver runs");

        if outcome.converged {
            assert_values_match(&outcome.values, &spectrum, 1e-4);
        } else {
            assert!(
                outcome.max_relative_residual > params.residual_tol
                    || outcome.subspace_delta > params.mev_tol,
                "non-convergence must be backed by a measurement that failed"
            );
        }
        assert!(outcome.passes <= params.max_passes);
    }

    #[test]
    fn a_basis_budget_that_forces_restarts_still_converges() {
        // A budget too small to hold more than the two blocks that make a Krylov
        // space at all, so the solver restarts every other pass. What each
        // restart carries is the whole guard band, not just the k columns being
        // returned; a restart that kept only k and re-padded with the same
        // deterministic random directions would be throwing away most of a pass
        // every time.
        let n = 1200;
        let spectrum = geometric(n, 0.8, 1.0);
        let op = reflected(spectrum.clone(), 23);

        let mut params = BlockKrylovParams::auto(6, n, 1);
        // Convergence cannot be declared before the depth ceiling forces the
        // first restart, so this run exercises the restart path deterministically.
        params.min_passes = 4;
        params.max_passes = 16;
        let outcome = block_krylov_eigen(&op, 6, params, Par::Seq).expect("solver runs");

        assert!(outcome.restarts >= 1, "the budget should have forced a restart");
        assert!(
            outcome.converged,
            "restarting is a memory strategy, not a reason to give up: residual {}, delta {}",
            outcome.max_relative_residual, outcome.subspace_delta
        );
        assert_values_match(&outcome.values, &spectrum, 1e-6);
    }

    #[test]
    fn an_exhausted_krylov_space_is_a_complete_answer() {
        // Numerical rank loss: the operator's range is four-dimensional, so the
        // block runs out of directions long before the pass ceiling. That is a
        // finished eigenproblem, not a failure — but only if exhaustion is
        // measured against the size of the operator image rather than against a
        // fixed absolute threshold, which carries no meaning once `C` is scaled.
        let n = 800;
        let mut spectrum = vec![0.0; n];
        spectrum[0] = 50.0;
        spectrum[1] = 30.0;
        spectrum[2] = 18.0;
        spectrum[3] = 9.0;
        let op = reflected(spectrum.clone(), 27);

        let params = BlockKrylovParams::auto(3, n, 1 << 30);
        let outcome = block_krylov_eigen(&op, 3, params, Par::Seq).expect("solver runs");

        assert!(outcome.converged, "an invariant subspace is a complete answer");
        assert!(
            outcome.passes <= 3,
            "exhaustion should be seen almost immediately, took {}",
            outcome.passes
        );
        assert_values_match(&outcome.values, &spectrum, 1e-8);
    }

    #[test]
    fn ritz_vectors_are_orthonormal_and_span_the_exact_subspace() {
        let n = 1500;
        let k = 6;
        let op = reflected(geometric(n, 0.85, 1.0), 14);

        let params = BlockKrylovParams::auto(k, n, 1 << 30);
        let outcome = block_krylov_eigen(&op, k, params, Par::Seq).expect("solver runs");
        assert!(outcome.converged);
        let u = &outcome.vectors;

        for a in 0..u.ncols() {
            for b in 0..u.ncols() {
                let dot: f64 = (0..u.nrows()).map(|row| u[(row, a)] * u[(row, b)]).sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "U column {a}·{b} = {dot}, expected {expected}"
                );
            }
        }

        // Eigenvalues can be right while the subspace is wrong, so check the
        // subspace directly against the operator's exact eigenvectors.
        let overlap = mev(op.exact_basis(k).as_ref(), u.as_ref(), Par::Seq);
        assert!(
            1.0 - overlap < 1e-8,
            "recovered subspace differs from the exact one: 1-MEV = {}",
            1.0 - overlap
        );
    }

    #[test]
    fn orthonormalize_returns_the_factor_it_promises() {
        // The residual certificate is stated in terms of the `R` this returns,
        // and the exhaustion signal in terms of its rank, so both are pinned
        // here rather than inferred from the solver converging.
        let n = 400;
        let width = 20;
        let mut block = random_start_block(n, width, 99);
        // Two exact duplicates: one inside a panel, one across a panel boundary,
        // since those take different paths through the blocking.
        for row in 0..n {
            let inside = block[(row, 2)];
            let across = block[(row, 3)];
            block[(row, 5)] = inside;
            block[(row, 13)] = across;
        }
        let original = block.clone();

        let mut r = Mat::<f64>::zeros(width, width);
        let rank = orthonormalize(block.as_mut(), Some(r.as_mut()), Par::Seq);
        assert_eq!(rank, width - 2, "both duplicates must be reported as collapsed");

        for col in [5usize, 13] {
            for row in 0..n {
                assert_eq!(block[(row, col)], 0.0, "collapsed column {col} must be zeroed");
            }
        }

        for a in 0..width {
            if a == 5 || a == 13 {
                continue;
            }
            for b in 0..width {
                if b == 5 || b == 13 {
                    continue;
                }
                let dot: f64 = (0..n).map(|row| block[(row, a)] * block[(row, b)]).sum();
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q column {a}·{b} = {dot}, expected {expected}"
                );
            }
        }

        let mut product = Mat::<f64>::zeros(n, width);
        matmul(
            product.as_mut(),
            Accum::Replace,
            block.as_ref(),
            r.as_ref(),
            1.0,
            Par::Seq,
        );
        for col in 0..width {
            for row in 0..n {
                assert!(
                    (product[(row, col)] - original[(row, col)]).abs() < 1e-9,
                    "QR != Z at ({row},{col}): {} vs {}",
                    product[(row, col)],
                    original[(row, col)]
                );
            }
        }

        // R is upper triangular.
        for col in 0..width {
            for row in (col + 1)..width {
                assert_eq!(r[(row, col)], 0.0, "R is not upper triangular at ({row},{col})");
            }
        }
    }

    #[test]
    fn certification_reaches_the_far_side_of_a_boundary_cluster() {
        // The end-to-end degeneracy tests depend on Ritz values being accurate
        // enough to expose a cluster; this pins the rule itself, with the
        // spectrum handed in directly.
        let tol = 1e-3;

        let clean = [100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0];
        assert_eq!(certification_span(&clean, 4, 7, tol), (4, false));

        let straddling = [100.0, 80.0, 60.0, 40.0, 40.0, 10.0, 5.0];
        assert_eq!(certification_span(&straddling, 4, 7, tol), (5, true));

        // A run of three: the whole cluster, not merely the next column.
        let run = [100.0, 80.0, 60.0, 40.0, 40.0, 40.0, 5.0];
        assert_eq!(certification_span(&run, 4, 7, tol), (6, true));

        // The cluster runs past what this run is carrying, so there is no
        // cluster-level certificate on offer — but the caller still learns that
        // its k lands inside one.
        assert_eq!(certification_span(&run, 4, 5, tol), (4, true));

        // Nothing beyond k was computed, so there is no boundary to judge.
        assert_eq!(certification_span(&clean, 7, 7, tol), (7, false));

        // A spectrum with no scale is not a cluster.
        assert_eq!(certification_span(&[0.0, 0.0, 0.0, 0.0], 2, 4, tol), (2, false));
    }

    #[test]
    fn the_all_ones_direction_is_never_the_starting_vector() {
        // The regression this whole module exists downstream of: a centered
        // covariance annihilates 1, so a start block with any constant
        // component wastes it.
        let block = random_start_block(500, 8, 7);
        for col in 0..block.ncols() {
            let sum: f64 = (0..block.nrows()).map(|row| block[(row, col)]).sum();
            assert!(
                sum.abs() < 1e-9,
                "start column {col} carries a constant component: sum={sum}"
            );
        }
    }

    #[test]
    fn the_subspace_criterion_can_actually_be_satisfied() {
        // Regression: the subspace test used to require the certified column
        // count to be identical between iterations. That count is
        // cluster-aware, so it moves while the Ritz values settle, and any
        // movement left the delta at infinity — which made `subspace_ok`
        // permanently false and convergence unreachable regardless of how
        // small the residuals were. A 250k-sample fit reported
        // `subspace_delta = inf` after eight passes because of this.
        //
        // A clean, well-separated spectrum must therefore both converge and
        // report a finite subspace delta.
        let op = reflected(geometric(600, 0.86, 1.0), 41);

        let params = BlockKrylovParams::auto(8, op.dim(), 1 << 30);
        let outcome = block_krylov_eigen(&op, 8, params, Par::Seq).expect("solver runs");

        assert!(
            outcome.subspace_delta.is_finite(),
            "subspace delta must be measurable after more than one iteration, got {}",
            outcome.subspace_delta
        );
        assert!(
            outcome.converged,
            "a well-separated spectrum must be able to converge: \
             residual {}, subspace delta {}",
            outcome.max_relative_residual, outcome.subspace_delta
        );
    }

    #[test]
    fn a_fit_is_reproducible_run_to_run() {
        // Scientific output: the same input must give bit-identical components,
        // which is why the start block comes from a fixed stream rather than
        // from entropy — restart padding included.
        let n = 1000;
        let op = reflected(geometric(n, 0.9, 1.0), 15);
        let mut params = BlockKrylovParams::auto(4, n, 1);
        params.min_passes = 4;
        params.max_passes = 10;

        let first = block_krylov_eigen(&op, 4, params, Par::Seq).expect("first run");
        let second = block_krylov_eigen(&op, 4, params, Par::Seq).expect("second run");

        assert!(first.restarts >= 1, "this run should exercise the restart path");
        assert_eq!(first.passes, second.passes);
        assert_eq!(first.restarts, second.restarts);
        assert_eq!(first.values, second.values, "eigenvalues must be identical");
        for col in 0..first.vectors.ncols() {
            for row in 0..first.vectors.nrows() {
                assert_eq!(
                    first.vectors[(row, col)],
                    second.vectors[(row, col)],
                    "vector entry ({row},{col}) differs between runs"
                );
            }
        }
    }

    #[test]
    fn a_null_operator_reports_no_positive_structure() {
        // Every direction is annihilated; the solver must terminate rather than
        // spin, and must not invent eigenvalues or read a cluster into a
        // spectrum that has no scale at all.
        let n = 600;
        let op = reflected(vec![0.0; n], 31);
        let params = BlockKrylovParams::auto(4, n, 1 << 30);
        let outcome = block_krylov_eigen(&op, 4, params, Par::Seq).expect("solver runs");

        assert!(outcome.converged);
        assert!(!outcome.truncation_splits_cluster);
        assert!(outcome.passes <= params.max_passes);
        for value in &outcome.values {
            assert!(value.abs() < 1e-10, "expected zero spectrum, got {value}");
        }
    }
}
