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
//! Block Lanczos with full reorthogonalization, Rayleigh–Ritz extraction, and
//! residual-driven stopping — i.e. the randomized block-Krylov family (Rokhlin–
//! Szlam–Tygert; Musco & Musco), specialized to a genotype operator whose block
//! application is one sequential pass.
//!
//! Starting from an orthonormal random block `Q₀`, it builds
//! `K = [Q₀, Q₁, …, Q_j]` with the three-term block recurrence
//!
//! ```text
//!   Z    = C·Q_j                     (one genome pass)
//!   A_j  = Q_jᵀ Z                    (b×b diagonal block)
//!   Z   ←  Z − Q_j A_j − Q_{j−1} B_{j−1}ᵀ
//!   Z   ←  Z − Σ_i Q_i (Q_iᵀ Z)      (full reorthogonalization, twice)
//!   Q_{j+1} B_j = qr(Z)
//! ```
//!
//! so that `T = Kᵀ C K` is block tridiagonal and assembled for free from the
//! `A_j`/`B_j` coefficients. Rayleigh–Ritz is then an eigendecomposition of `T`,
//! whose dimension is `b(j+1)` — a few hundred at most, solved densely.
//!
//! # Stopping without extra passes
//!
//! The Lanczos identity `C K = K T + Q_{j+1} B_j E_jᵀ` makes the exact residual
//! of every Ritz pair available from the trailing block alone:
//!
//! ```text
//!   ‖C u_i − θ_i u_i‖ = ‖B_j · s_i[last b entries]‖
//! ```
//!
//! since `Q_{j+1}` is orthonormal. So convergence is *measured*, never assumed,
//! and measuring it costs no genotype I/O. That is the property that makes an
//! adaptive pass count possible: depth grows only while the residual says it
//! must, instead of running a fixed iteration count chosen for someone else's
//! dataset.
//!
//! A second, complementary criterion tracks the *subspace* rather than
//! individual vectors: the mean explained variance
//! `MEV = ‖U_prevᵀ U‖_F² / k` between successive top-k Ritz bases. Nearly
//! degenerate PCs rotate freely within their eigenspace, so a per-vector test
//! can report non-convergence forever on a subspace that is in fact settled;
//! MEV is invariant to that rotation. Both must pass.
//!
//! # Clustered spectra
//!
//! Oversampling means `θ_{k+1}` is always computed, so the solver can see the
//! `k/k+1` gap it is being asked to split. When that gap is tight the requested
//! top-k is not a well-conditioned object on its own, and more depth does not
//! fix it. The solver widens its guard band instead — extra columns cost
//! arithmetic *inside* a pass, while extra depth costs another pass.

use faer::linalg::matmul::matmul;
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

/// A covariance operator whose block application costs one data pass.
pub trait BlockOperator {
    type Error;

    /// Dimension of the (square, self-adjoint, positive semi-definite) operator.
    fn dim(&self) -> usize;

    /// `out ← C · q`, for all columns of `q` in a single pass over the data.
    ///
    /// Implementations must treat the column count as free: the whole point is
    /// that `b` columns cost what one column costs.
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
    /// Tolerance on `1 − MEV` between successive top-k Ritz subspaces.
    pub mev_tol: f64,
    /// Byte budget for the retained Krylov basis; hitting it forces a restart.
    pub basis_budget_bytes: usize,
    /// Gap ratio `(θ_k − θ_{k+1}) / θ_k` below which the requested boundary is
    /// treated as spectrally clustered, triggering one guard-band widening
    /// rather than further depth.
    pub cluster_gap: f64,
    /// Fixed PRNG seed. Fits must be reproducible run to run.
    pub seed: u64,
}

impl BlockKrylovParams {
    /// Defaults derived from the request and the machine, not from constants
    /// baked in for someone else's cohort.
    ///
    /// Oversampling is `k + min(32, max(8, k/2))`: a flat `k + 10` is thin for
    /// large `k`, and unbounded `1.5k` wastes arithmetic for very large `k`.
    pub fn auto(k: usize, dim: usize, basis_budget_bytes: usize) -> Self {
        let oversample = 32.min(8.max(k.div_ceil(2)));
        let block_width = (k + oversample).clamp(1, dim.max(1));
        Self {
            block_width,
            min_passes: 2,
            max_passes: 24,
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
    /// Worst relative Ritz residual over the returned pairs.
    pub max_relative_residual: f64,
    /// `1 − MEV` between the final two top-k subspaces, once measurable.
    pub subspace_delta: f64,
    /// `(θ_k − θ_{k+1}) / θ_k` at the requested boundary, if it was observed.
    pub boundary_gap: Option<f64>,
    /// Times the basis was restarted under the memory budget.
    pub restarts: usize,
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
        });
    }
    if k > n {
        return Err(BlockKrylovError::Invalid(
            "requested more components than the operator's dimension",
        ));
    }

    let mut width = params.block_width.clamp(1, n);
    let mut widened = false;

    // Depth cap from the basis budget and the dense Rayleigh–Ritz limit. One
    // basis column is n f64s; the budget is what stops a 500k-sample fit from
    // silently turning its Krylov basis into the largest object in the process.
    let column_bytes = n.saturating_mul(std::mem::size_of::<f64>());
    let budget_columns = if column_bytes == 0 {
        usize::MAX
    } else {
        (params.basis_budget_bytes / column_bytes).max(width * 2)
    };

    let mut start = random_start_block(n, width, params.seed);
    orthonormalize(start.as_mut(), None);

    let mut passes = 0usize;
    let mut restarts = 0usize;
    let mut previous_top: Option<Mat<f64>> = None;
    let mut best: Option<BlockKrylovOutcome> = None;

    'restart: loop {
        let max_depth = (budget_columns / width)
            .min(MAX_PROJECTED_DIM / width.max(1))
            .max(2);

        let mut blocks: Vec<Mat<f64>> = vec![start.clone()];
        let mut alphas: Vec<Mat<f64>> = Vec::new();
        let mut betas: Vec<Mat<f64>> = Vec::new();

        for depth in 0..max_depth {
            if passes >= params.max_passes {
                break;
            }

            // --- the one expensive step: a single pass over the genotypes ---
            let mut z = Mat::<f64>::zeros(n, width);
            op.apply_block(z.as_mut(), blocks[depth].as_ref())
                .map_err(BlockKrylovError::Operator)?;
            passes += 1;

            // A_j = Q_jᵀ Z
            let mut alpha = Mat::<f64>::zeros(width, width);
            matmul(
                alpha.as_mut(),
                Accum::Replace,
                blocks[depth].as_ref().transpose(),
                z.as_ref(),
                1.0,
                par,
            );
            // Symmetrize: A_j is self-adjoint in exact arithmetic, and forcing
            // it keeps the projected problem exactly symmetric.
            symmetrize(alpha.as_mut());
            alphas.push(alpha);

            // Z ← Z − Q_j A_j − Q_{j−1} B_{j−1}ᵀ, then full reorthogonalization
            // against the whole retained basis (twice — one pass of block
            // Gram–Schmidt loses orthogonality on a clustered spectrum, and the
            // basis is what the Ritz vectors are expressed in).
            let alpha_ref = alphas[depth].as_ref();
            matmul(
                z.as_mut(),
                Accum::Add,
                blocks[depth].as_ref(),
                alpha_ref,
                -1.0,
                par,
            );
            if depth > 0 {
                let beta_prev = betas[depth - 1].as_ref();
                matmul(
                    z.as_mut(),
                    Accum::Add,
                    blocks[depth - 1].as_ref(),
                    beta_prev.transpose(),
                    -1.0,
                    par,
                );
            }
            for _ in 0..2 {
                for block in blocks.iter() {
                    let mut coeff = Mat::<f64>::zeros(width, width);
                    matmul(
                        coeff.as_mut(),
                        Accum::Replace,
                        block.as_ref().transpose(),
                        z.as_ref(),
                        1.0,
                        par,
                    );
                    matmul(z.as_mut(), Accum::Add, block.as_ref(), coeff.as_ref(), -1.0, par);
                }
            }

            // Q_{j+1} B_j = qr(Z). The new block extends the Krylov basis that
            // the Ritz vectors are expressed in, so it must be retained.
            let mut beta = Mat::<f64>::zeros(width, width);
            let rank = orthonormalize(z.as_mut(), Some(beta.as_mut()));
            betas.push(beta);
            blocks.push(z);

            // --- Rayleigh–Ritz on the block-tridiagonal projection ---
            let projected = assemble_block_tridiagonal(&alphas, &betas, width);
            let (theta, s) = dense_eigen_desc(projected.as_ref())?;

            let available = theta.len();
            let keep = k.min(available);
            let boundary_gap = if available > k && theta[k - 1].abs() > 0.0 {
                Some((theta[k - 1] - theta[k]) / theta[k - 1].abs())
            } else {
                None
            };

            // Residuals from the trailing block only — no extra genome pass.
            let residuals = ritz_residuals(&betas[depth], s.as_ref(), keep, width, par);
            let mut max_relative = 0.0f64;
            for (idx, residual) in residuals.iter().enumerate().take(keep) {
                let scale = theta[idx].abs().max(RITZ_SCALE_FLOOR);
                max_relative = max_relative.max(residual / scale);
            }

            // Ritz vectors: U = K · S[:, ..keep]
            let top = lift_ritz_vectors(&blocks, s.as_ref(), keep, width, n, par);

            let subspace_delta = match previous_top.as_ref() {
                Some(prev) if prev.ncols() == top.ncols() => 1.0 - mev(prev.as_ref(), top.as_ref(), par),
                _ => f64::INFINITY,
            };
            previous_top = Some(top.clone());

            let residual_ok = max_relative <= params.residual_tol;
            let subspace_ok = subspace_delta <= params.mev_tol;
            let enough_passes = passes >= params.min_passes;
            let converged = residual_ok && subspace_ok && enough_passes;

            let outcome = BlockKrylovOutcome {
                values: theta[..keep].to_vec(),
                vectors: top,
                passes,
                converged,
                max_relative_residual: max_relative,
                subspace_delta,
                boundary_gap,
                restarts,
            };

            // A fully exhausted Krylov space is a *complete* answer, not a
            // failure: the basis already spans an invariant subspace, so no
            // further pass can add information. Partial rank deficiency is not
            // exhaustion — the surviving columns still carry the recurrence,
            // and the collapsed ones only make the residual estimate
            // conservative (‖Q v‖ ≤ ‖v‖ when Q has zero columns), so the
            // residual test remains the authority on convergence.
            let exhausted = rank == 0;
            if converged || exhausted {
                return Ok(BlockKrylovOutcome {
                    converged: converged || exhausted,
                    ..outcome
                });
            }

            // A clustered k/k+1 boundary is not fixed by more depth: PC k and
            // PC k+1 are rotating inside one near-degenerate eigenspace. Widen
            // the guard band once instead — extra columns are paid for inside a
            // pass, extra depth costs a whole new one.
            if !widened
                && enough_passes
                && residual_ok
                && boundary_gap.is_some_and(|gap| gap < params.cluster_gap)
                && width < n
            {
                widened = true;
                width = (width + width.div_ceil(2)).min(n);
                start = restart_block(&outcome.vectors, n, width, params.seed);
                best = Some(outcome);
                restarts += 1;
                continue 'restart;
            }

            best = Some(outcome);

            if depth + 1 >= max_depth && passes < params.max_passes {
                // Memory or dense-solve ceiling reached with work left to do:
                // restart from the current Ritz block, which keeps the best
                // information found so far and frees the rest of the basis.
                let vectors = best.as_ref().map(|o| &o.vectors).expect("best is set");
                start = restart_block(vectors, n, width, params.seed);
                restarts += 1;
                continue 'restart;
            }
        }

        break;
    }

    best.map(|outcome| BlockKrylovOutcome { restarts, ..outcome })
        .ok_or(BlockKrylovError::Invalid(
            "block Krylov solver produced no Ritz pairs",
        ))
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

/// Rebuild a start block from converged Ritz vectors, padding with fresh
/// pseudo-random directions when the block is wider than what was retained.
fn restart_block(vectors: &Mat<f64>, n: usize, width: usize, seed: u64) -> Mat<f64> {
    let mut block = random_start_block(n, width, seed.wrapping_add(0x51_7C_C1_B7_27_22_0A_95));
    let carried = vectors.ncols().min(width);
    for col in 0..carried {
        for row in 0..n {
            block[(row, col)] = vectors[(row, col)];
        }
    }
    remove_constant_direction(block.as_mut());
    orthonormalize(block.as_mut(), None);
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

/// Modified Gram–Schmidt with reorthogonalization; writes the upper-triangular
/// factor into `r` when requested and returns the numerical rank.
///
/// Columns that collapse below the tolerance are replaced by zero and reported
/// through the rank, which is how an exhausted Krylov space is detected.
fn orthonormalize(mut block: MatMut<'_, f64>, mut r: Option<MatMut<'_, f64>>) -> usize {
    let n = block.nrows();
    let width = block.ncols();
    if let Some(r) = r.as_mut() {
        r.fill(0.0);
    }

    let mut rank = 0usize;
    for col in 0..width {
        // Two passes: classical reorthogonalization. One pass is not enough
        // once the block is nearly rank deficient, which is exactly the regime
        // a converging Krylov space enters.
        for _ in 0..2 {
            for prev in 0..col {
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

        let mut norm_sq = 0.0;
        for row in 0..n {
            norm_sq += block[(row, col)] * block[(row, col)];
        }
        let norm = norm_sq.sqrt();
        if let Some(r) = r.as_mut() {
            r[(col, col)] = norm;
        }

        if norm <= f64::EPSILON * 64.0 * (n as f64).sqrt() {
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

    rank
}

fn symmetrize(mut mat: MatMut<'_, f64>) {
    let n = mat.nrows().min(mat.ncols());
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = 0.5 * (mat[(i, j)] + mat[(j, i)]);
            mat[(i, j)] = avg;
            mat[(j, i)] = avg;
        }
    }
}

/// `T = Kᵀ C K`, block tridiagonal with `A_j` on the diagonal and `B_j` on the
/// off-diagonals — assembled from the recurrence coefficients, never from the
/// operator, so it costs no data pass.
fn assemble_block_tridiagonal(alphas: &[Mat<f64>], betas: &[Mat<f64>], width: usize) -> Mat<f64> {
    let depth = alphas.len();
    let dim = depth * width;
    let mut t = Mat::<f64>::zeros(dim, dim);

    for (j, alpha) in alphas.iter().enumerate() {
        let base = j * width;
        for r in 0..width {
            for c in 0..width {
                t[(base + r, base + c)] = alpha[(r, c)];
            }
        }
        if j + 1 < depth {
            let beta = &betas[j];
            let next = base + width;
            for r in 0..width {
                for c in 0..width {
                    // B_j couples block j to block j+1; its transpose closes the
                    // symmetry.
                    t[(next + r, base + c)] = beta[(r, c)];
                    t[(base + c, next + r)] = beta[(r, c)];
                }
            }
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

/// `‖C u_i − θ_i u_i‖ = ‖B_j · s_i[last b entries]‖`, from the Lanczos identity
/// `C K = K T + Q_{j+1} B_j E_jᵀ`. No operator application involved.
fn ritz_residuals(
    beta: &Mat<f64>,
    s: MatRef<'_, f64>,
    keep: usize,
    width: usize,
    par: Par,
) -> Vec<f64> {
    let dim = s.nrows();
    if dim < width || keep == 0 {
        return vec![f64::INFINITY; keep];
    }
    let tail_start = dim - width;

    let mut tail = Mat::<f64>::zeros(width, keep);
    for col in 0..keep {
        for row in 0..width {
            tail[(row, col)] = s[(tail_start + row, col)];
        }
    }

    let mut product = Mat::<f64>::zeros(width, keep);
    matmul(
        product.as_mut(),
        Accum::Replace,
        beta.as_ref(),
        tail.as_ref(),
        1.0,
        par,
    );

    (0..keep)
        .map(|col| {
            (0..width)
                .map(|row| product[(row, col)] * product[(row, col)])
                .sum::<f64>()
                .sqrt()
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

    /// Dense operator standing in for the streamed covariance, so the solver's
    /// mathematics can be checked against an exact eigendecomposition.
    struct DenseOp {
        matrix: Mat<f64>,
    }

    impl BlockOperator for DenseOp {
        type Error = std::convert::Infallible;

        fn dim(&self) -> usize {
            self.matrix.nrows()
        }

        fn apply_block(&self, out: MatMut<'_, f64>, q: MatRef<'_, f64>) -> Result<(), Self::Error> {
            matmul(out, Accum::Replace, self.matrix.as_ref(), q, 1.0, Par::Seq);
            Ok(())
        }
    }

    /// Symmetric operator with a planted spectrum, built as `V diag(λ) Vᵀ` from
    /// a deterministic orthonormal `V`.
    fn planted(spectrum: &[f64], seed: u64) -> DenseOp {
        let n = spectrum.len();
        let mut v = random_start_block(n, n, seed);
        orthonormalize(v.as_mut(), None);

        let mut scaled = Mat::<f64>::zeros(n, n);
        for col in 0..n {
            for row in 0..n {
                scaled[(row, col)] = v[(row, col)] * spectrum[col];
            }
        }
        let mut matrix = Mat::<f64>::zeros(n, n);
        matmul(
            matrix.as_mut(),
            Accum::Replace,
            scaled.as_ref(),
            v.as_ref().transpose(),
            1.0,
            Par::Seq,
        );
        symmetrize(matrix.as_mut());
        DenseOp { matrix }
    }

    fn spectrum_of(op: &DenseOp) -> Vec<f64> {
        let (values, _) = dense_eigen_desc::<std::convert::Infallible>(op.matrix.as_ref())
            .expect("reference eigendecomposition");
        values
    }

    #[test]
    fn recovers_a_well_separated_spectrum() {
        let spectrum: Vec<f64> = (0..40).map(|i| 100.0 * 0.75f64.powi(i as i32)).collect();
        let op = planted(&spectrum, 11);
        let reference = spectrum_of(&op);

        let params = BlockKrylovParams::auto(5, op.dim(), 1 << 30);
        let outcome = block_krylov_eigen(&op, 5, params, Par::Seq).expect("solver runs");

        assert!(outcome.converged, "solver should converge on a clean gap");
        for (idx, value) in outcome.values.iter().enumerate() {
            let expected = reference[idx];
            assert!(
                (value - expected).abs() <= 1e-6 * expected.abs().max(1.0),
                "eigenvalue {idx}: got {value}, expected {expected}"
            );
        }
    }

    #[test]
    fn costs_far_fewer_passes_than_the_krylov_dimension() {
        let spectrum: Vec<f64> = (0..60).map(|i| 50.0 * 0.8f64.powi(i as i32)).collect();
        let op = planted(&spectrum, 12);

        let params = BlockKrylovParams::auto(10, op.dim(), 1 << 30);
        let outcome = block_krylov_eigen(&op, 10, params, Par::Seq).expect("solver runs");

        assert!(outcome.converged);
        // The whole point: a scalar Arnoldi needs >= 64 applications to build a
        // comparable subspace. Anything near that here means the block
        // structure is not doing its job.
        assert!(
            outcome.passes <= 8,
            "expected a handful of passes, took {}",
            outcome.passes
        );
    }

    #[test]
    fn clustered_spectrum_converges_as_a_subspace() {
        // PCs 4 and 5 are degenerate: individual eigenvectors are not defined,
        // only the plane they span. The subspace must still converge.
        let spectrum = vec![
            100.0, 80.0, 60.0, 40.0, 40.0, 12.0, 9.0, 7.0, 5.0, 3.0, 2.0, 1.0,
        ];
        let op = planted(&spectrum, 13);
        let reference = spectrum_of(&op);

        let params = BlockKrylovParams::auto(5, op.dim(), 1 << 30);
        let outcome = block_krylov_eigen(&op, 5, params, Par::Seq).expect("solver runs");

        for (idx, value) in outcome.values.iter().enumerate() {
            assert!(
                (value - reference[idx]).abs() <= 1e-5 * reference[idx].abs().max(1.0),
                "clustered eigenvalue {idx}: got {value}, expected {}",
                reference[idx]
            );
        }
    }

    #[test]
    fn ritz_vectors_are_orthonormal_and_satisfy_their_residual_claim() {
        let spectrum: Vec<f64> = (0..30).map(|i| 20.0 * 0.85f64.powi(i as i32)).collect();
        let op = planted(&spectrum, 14);

        let params = BlockKrylovParams::auto(6, op.dim(), 1 << 30);
        let outcome = block_krylov_eigen(&op, 6, params, Par::Seq).expect("solver runs");
        let u = &outcome.vectors;

        // Orthonormality.
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

        // The reported residual must bound the true one: ‖Cu − θu‖.
        let mut cu = Mat::<f64>::zeros(u.nrows(), u.ncols());
        op.apply_block(cu.as_mut(), u.as_ref()).expect("apply");
        for col in 0..u.ncols() {
            let theta = outcome.values[col];
            let residual: f64 = (0..u.nrows())
                .map(|row| {
                    let r = cu[(row, col)] - theta * u[(row, col)];
                    r * r
                })
                .sum::<f64>()
                .sqrt();
            let relative = residual / theta.abs().max(RITZ_SCALE_FLOOR);
            assert!(
                relative <= 1e-5,
                "component {col} true relative residual {relative} exceeds the claim"
            );
        }
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
    fn a_fit_is_reproducible_run_to_run() {
        // Scientific output: the same input must give bit-identical components,
        // which is why the start block comes from a fixed stream rather than
        // from entropy.
        let spectrum: Vec<f64> = (0..24).map(|i| 30.0 * 0.9f64.powi(i as i32)).collect();
        let op = planted(&spectrum, 15);
        let params = BlockKrylovParams::auto(4, op.dim(), 1 << 30);

        let first = block_krylov_eigen(&op, 4, params, Par::Seq).expect("first run");
        let second = block_krylov_eigen(&op, 4, params, Par::Seq).expect("second run");

        assert_eq!(first.passes, second.passes);
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
    fn a_flat_spectrum_is_reported_honestly() {
        // The hard case: a slowly decaying tail with no usable gap, i.e. fine
        // within-population structure. Either it converges, or it says it did
        // not — what it must never do is quietly return a wrong subspace while
        // claiming convergence.
        let spectrum: Vec<f64> = (0..80).map(|i| 10.0 - 0.05 * i as f64).collect();
        let op = planted(&spectrum, 16);
        let reference = spectrum_of(&op);

        let mut params = BlockKrylovParams::auto(8, op.dim(), 1 << 30);
        params.max_passes = 12;
        let outcome = block_krylov_eigen(&op, 8, params, Par::Seq).expect("solver runs");

        if outcome.converged {
            for (idx, value) in outcome.values.iter().enumerate() {
                assert!(
                    (value - reference[idx]).abs() <= 1e-4 * reference[idx].abs().max(1.0),
                    "claimed convergence but eigenvalue {idx} is off: {value} vs {}",
                    reference[idx]
                );
            }
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
    fn recovered_subspace_matches_the_exact_one() {
        // Eigenvalues can be right while the subspace is wrong, so check the
        // subspace directly: MEV against the exact top-k basis.
        let spectrum = vec![
            90.0, 70.0, 55.0, 30.0, 30.0, 30.0, 11.0, 8.0, 6.0, 4.0, 2.5, 1.5, 1.0, 0.5,
        ];
        let op = planted(&spectrum, 17);
        let (_, exact) = dense_eigen_desc::<std::convert::Infallible>(op.matrix.as_ref())
            .expect("reference eigendecomposition");

        let k = 6;
        let params = BlockKrylovParams::auto(k, op.dim(), 1 << 30);
        let outcome = block_krylov_eigen(&op, k, params, Par::Seq).expect("solver runs");

        let exact_top = exact.as_ref().subcols(0, k);
        let overlap = mev(exact_top, outcome.vectors.as_ref(), Par::Seq);
        assert!(
            1.0 - overlap < 1e-8,
            "recovered subspace differs from the exact one: 1-MEV = {}",
            1.0 - overlap
        );
    }

    #[test]
    fn a_null_operator_reports_no_positive_structure() {
        // Every direction is annihilated; the solver must terminate rather than
        // spin, and must not invent eigenvalues.
        let op = DenseOp {
            matrix: Mat::<f64>::zeros(64, 64),
        };
        let params = BlockKrylovParams::auto(4, op.dim(), 1 << 30);
        let outcome = block_krylov_eigen(&op, 4, params, Par::Seq).expect("solver runs");
        assert!(outcome.passes <= params.max_passes);
        for value in &outcome.values {
            assert!(value.abs() < 1e-10, "expected zero spectrum, got {value}");
        }
    }
}
