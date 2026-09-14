//! Dense products whose arithmetic does not depend on the thread count.
//!
//! faer parallelizes a product by cutting it into as many pieces as it has
//! threads, and the pieces decide how every sum is grouped. The answer then
//! depends on the thread count at the level of roundoff, which near-degenerate
//! principal axes amplify into visible rotations. faer's parallel kernels also
//! run on spindle, whose barrier spin-waits: at 32 threads that waiting was 60%
//! of a PCA's cycles.
//!
//! The kernels here cut a product into row leaves whose size is a function of
//! the product's shape alone, run every leaf through faer's sequential kernel,
//! and combine the leaves in a fixed order. Rayon decides only which thread
//! computes which leaf, so the result is the same bits on one thread or on a
//! hundred, and no thread spins waiting for another.

use dyn_stack::{MemBuffer, MemStack};
use faer::diag::Diag;
use faer::linalg::evd::{self, EvdError};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side, unzip, zip};
use rayon::prelude::*;

/// Floating-point operations a leaf is given at least, so that its arithmetic
/// dwarfs the cost of scheduling it.
const LEAF_FLOPS: usize = 1 << 21;

/// Most leaves one product is cut into: enough to keep a few hundred threads
/// busy, few enough that a Gram reduction allocates a bounded number of small
/// matrices.
const MAX_LEAVES: usize = 1 << 10;

/// Rows per leaf for a product over `rows` rows costing `flops_per_row` each.
///
/// A pure function of the shape. It must never consult the thread count, the
/// cache sizes or anything else about the machine: the leaves group every sum,
/// so the leaves are part of the answer.
pub(crate) fn leaf_rows(rows: usize, flops_per_row: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let for_work = LEAF_FLOPS.div_ceil(flops_per_row.max(1));
    let for_count = rows.div_ceil(MAX_LEAVES);
    for_work.max(for_count).min(rows)
}

/// `out ← out + alpha·a·b` for [`Accum::Add`], `out ← alpha·a·b` for
/// [`Accum::Replace`], one row leaf at a time.
///
/// Row `i` of the product depends only on row `i` of `a`, so the leaves write
/// disjoint rows and there is nothing to combine.
pub(crate) fn mul_rows(
    out: MatMut<'_, f64>,
    accum: Accum,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    alpha: f64,
) {
    let leaf = leaf_rows(out.nrows(), a.ncols().saturating_mul(b.ncols()));
    mul_rows_in_leaves(out, accum, a, b, alpha, leaf);
}

fn mul_rows_in_leaves(
    out: MatMut<'_, f64>,
    accum: Accum,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    alpha: f64,
    leaf: usize,
) {
    debug_assert_eq!(out.nrows(), a.nrows());
    debug_assert_eq!(out.ncols(), b.ncols());
    debug_assert_eq!(a.ncols(), b.nrows());
    if out.nrows() == 0 || out.ncols() == 0 {
        return;
    }
    out.par_row_chunks_mut(leaf)
        .enumerate()
        .for_each(|(index, chunk)| {
            let rows = chunk.nrows();
            matmul(
                chunk,
                accum,
                a.subrows(index * leaf, rows),
                b,
                alpha,
                Par::Seq,
            );
        });
}

/// `out ← aᵀ·b`, summed over row leaves along a fixed binary tree.
///
/// Every entry of the product sums over every row, so leaves cannot write
/// disjoint entries. Each leaf forms `aᵀ·b` over its own rows, and sibling
/// subtrees are added, left into right's complement, at splits fixed by the
/// leaf count.
pub(crate) fn gram_rows(out: MatMut<'_, f64>, a: MatRef<'_, f64>, b: MatRef<'_, f64>) {
    let leaf = leaf_rows(a.nrows(), a.ncols().saturating_mul(b.ncols()));
    gram_rows_in_leaves(out, a, b, leaf);
}

fn gram_rows_in_leaves(
    mut out: MatMut<'_, f64>,
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    leaf: usize,
) {
    debug_assert_eq!(a.nrows(), b.nrows());
    debug_assert_eq!(out.nrows(), a.ncols());
    debug_assert_eq!(out.ncols(), b.ncols());
    if out.nrows() == 0 || out.ncols() == 0 {
        return;
    }
    let leaves = a.nrows().div_ceil(leaf);
    if leaves <= 1 {
        matmul(out, Accum::Replace, a.transpose(), b, 1.0, Par::Seq);
        return;
    }
    let total = gram_subtree(a, b, leaf, 0, leaves);
    out.copy_from(total.as_ref());
}

fn gram_subtree(
    a: MatRef<'_, f64>,
    b: MatRef<'_, f64>,
    leaf: usize,
    first: usize,
    count: usize,
) -> Mat<f64> {
    if count == 1 {
        let start = first * leaf;
        let rows = leaf.min(a.nrows() - start);
        let mut product = Mat::zeros(a.ncols(), b.ncols());
        matmul(
            product.as_mut(),
            Accum::Replace,
            a.subrows(start, rows).transpose(),
            b.subrows(start, rows),
            1.0,
            Par::Seq,
        );
        return product;
    }
    let left_count = count / 2;
    let (mut left, right) = rayon::join(
        || gram_subtree(a, b, leaf, first, left_count),
        || gram_subtree(a, b, leaf, first + left_count, count - left_count),
    );
    zip!(left.as_mut(), right.as_ref()).for_each(|unzip!(into, from)| *into += *from);
    left
}

/// `A = U·diag(s)·Uᵀ` for a symmetric `A` given by its `side` triangle,
/// computed sequentially.
///
/// faer's `self_adjoint_eigen` runs at the process-wide parallelism, and a
/// parallel tridiagonalization groups its sums by the thread count. This is the
/// same decomposition with the same parameters at `Par::Seq`, and it consults
/// no global state.
pub(crate) fn self_adjoint_eigen_seq(
    matrix: MatRef<'_, f64>,
    side: Side,
) -> Result<(Diag<f64>, Mat<f64>), EvdError> {
    let dim = matrix.nrows();
    assert_eq!(dim, matrix.ncols(), "a symmetric matrix is square");
    let lower = match side {
        Side::Lower => matrix,
        Side::Upper => matrix.transpose(),
    };
    let mut values = Diag::zeros(dim);
    let mut vectors = Mat::zeros(dim, dim);
    let mut memory = MemBuffer::new(evd::self_adjoint_evd_scratch::<f64>(
        dim,
        evd::ComputeEigenvectors::Yes,
        Par::Seq,
        Default::default(),
    ));
    evd::self_adjoint_evd(
        lower,
        values.as_mut(),
        Some(vectors.as_mut()),
        Par::Seq,
        MemStack::new(&mut memory),
        Default::default(),
    )?;
    Ok((values, vectors))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(rows: usize, cols: usize, seed: u64) -> Mat<f64> {
        let mut state = seed;
        Mat::from_fn(rows, cols, |_, _| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
    }

    fn with_threads<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(work)
    }

    fn assert_same_bits(lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>, what: &str) {
        assert_eq!((lhs.nrows(), lhs.ncols()), (rhs.nrows(), rhs.ncols()));
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                assert_eq!(
                    lhs[(row, col)].to_bits(),
                    rhs[(row, col)].to_bits(),
                    "{what}: entry ({row}, {col}) {:e} vs {:e}",
                    lhs[(row, col)],
                    rhs[(row, col)]
                );
            }
        }
    }

    fn max_abs_diff(lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>) -> f64 {
        let mut worst = 0.0f64;
        for col in 0..lhs.ncols() {
            for row in 0..lhs.nrows() {
                worst = worst.max((lhs[(row, col)] - rhs[(row, col)]).abs());
            }
        }
        worst
    }

    #[test]
    fn leaves_depend_on_the_shape_and_stay_bounded() {
        assert_eq!(leaf_rows(0, 10), 1);
        assert_eq!(leaf_rows(5, usize::MAX), 1);
        assert_eq!(leaf_rows(5, 1), 5);
        for rows in [1usize, 7, 1_000, 100_000, 2_000_000] {
            for flops in [1usize, 30, 15_360, 1 << 24] {
                let leaf = leaf_rows(rows, flops);
                assert!(
                    (1..=rows).contains(&leaf),
                    "rows {rows} flops {flops} leaf {leaf}"
                );
                assert!(rows.div_ceil(leaf) <= MAX_LEAVES);
            }
        }
    }

    #[test]
    fn row_products_do_not_depend_on_the_thread_count() {
        let a = pseudo_random(1_003, 37, 11);
        let b = pseudo_random(37, 9, 12);
        let start = pseudo_random(1_003, 9, 13);
        let product = |threads: usize, accum: Accum| {
            with_threads(threads, || {
                let mut out = start.clone();
                mul_rows_in_leaves(out.as_mut(), accum, a.as_ref(), b.as_ref(), -0.75, 17);
                out
            })
        };

        for accum in [Accum::Add, Accum::Replace] {
            let serial = product(1, accum);
            let mut reference = match accum {
                Accum::Add => start.clone(),
                Accum::Replace => Mat::zeros(1_003, 9),
            };
            matmul(
                reference.as_mut(),
                Accum::Add,
                a.as_ref(),
                b.as_ref(),
                -0.75,
                Par::Seq,
            );
            assert!(max_abs_diff(serial.as_ref(), reference.as_ref()) < 1e-12);
            for threads in [2, 3, 8] {
                assert_same_bits(
                    serial.as_ref(),
                    product(threads, accum).as_ref(),
                    &format!("mul_rows at {threads} threads"),
                );
            }
        }
    }

    #[test]
    fn gram_products_do_not_depend_on_the_thread_count() {
        let a = pseudo_random(2_049, 23, 21);
        let b = pseudo_random(2_049, 6, 22);
        let product = |threads: usize, leaf: usize| {
            with_threads(threads, || {
                let mut out = Mat::from_fn(23, 6, |_, _| f64::NAN);
                gram_rows_in_leaves(out.as_mut(), a.as_ref(), b.as_ref(), leaf);
                out
            })
        };

        let mut reference = Mat::zeros(23, 6);
        matmul(
            reference.as_mut(),
            Accum::Replace,
            a.as_ref().transpose(),
            b.as_ref(),
            1.0,
            Par::Seq,
        );
        for leaf in [1usize, 31, 2_048, 4_096] {
            let serial = product(1, leaf);
            assert!(max_abs_diff(serial.as_ref(), reference.as_ref()) < 1e-10);
            for threads in [2, 3, 8] {
                assert_same_bits(
                    serial.as_ref(),
                    product(threads, leaf).as_ref(),
                    &format!("gram_rows with {leaf}-row leaves at {threads} threads"),
                );
            }
        }
    }

    #[test]
    fn sequential_eigendecomposition_reconstructs_from_either_triangle() {
        let dim = 37;
        let base = pseudo_random(dim, dim, 31);
        let mut symmetric = Mat::zeros(dim, dim);
        matmul(
            symmetric.as_mut(),
            Accum::Replace,
            base.as_ref(),
            base.as_ref().transpose(),
            1.0,
            Par::Seq,
        );
        let (lower_values, lower_vectors) =
            self_adjoint_eigen_seq(symmetric.as_ref(), Side::Lower).expect("lower triangle");
        let (upper_values, _) =
            self_adjoint_eigen_seq(symmetric.as_ref(), Side::Upper).expect("upper triangle");
        let lower_values = lower_values.as_ref();
        let upper_values = upper_values.as_ref();
        let scale = lower_values[dim - 1].abs().max(1.0);

        let mut image = Mat::zeros(dim, dim);
        matmul(
            image.as_mut(),
            Accum::Replace,
            symmetric.as_ref(),
            lower_vectors.as_ref(),
            1.0,
            Par::Seq,
        );
        for col in 0..dim {
            assert!((lower_values[col] - upper_values[col]).abs() <= 1e-10 * scale);
            for row in 0..dim {
                let expected = lower_vectors[(row, col)] * lower_values[col];
                assert!(
                    (image[(row, col)] - expected).abs() <= 1e-10 * scale,
                    "A·U != U·S at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn empty_products_touch_nothing_they_should_not() {
        let a = Mat::<f64>::zeros(0, 4);
        let b = Mat::<f64>::zeros(0, 3);
        let mut gram = Mat::from_fn(4, 3, |_, _| 5.0);
        gram_rows(gram.as_mut(), a.as_ref(), b.as_ref());
        assert!(
            gram.col_iter()
                .all(|col| col.iter().all(|&value| value == 0.0))
        );

        let tall = pseudo_random(10, 0, 3);
        let inner = Mat::<f64>::zeros(0, 2);
        let mut out = Mat::from_fn(10, 2, |_, _| 2.0);
        mul_rows(out.as_mut(), Accum::Add, tall.as_ref(), inner.as_ref(), 1.0);
        assert!(
            out.col_iter()
                .all(|col| col.iter().all(|&value| value == 2.0))
        );
        mul_rows(
            out.as_mut(),
            Accum::Replace,
            tall.as_ref(),
            inner.as_ref(),
            1.0,
        );
        assert!(
            out.col_iter()
                .all(|col| col.iter().all(|&value| value == 0.0))
        );
    }
}
