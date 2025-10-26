use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagRef};
use faer::linalg::solvers::{self, Solve};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::matrix_free::eigen::{self as mf_eigen, PartialEigenInfo, PartialEigenParams};
use faer::{Mat, MatMut, MatRef, Par, Side, get_global_parallelism};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix1, Ix2};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FaerLinalgError {
    #[error("SVD failed to converge")]
    SvdNoConvergence,
    #[error("Self-adjoint eigendecomposition failed: {0:?}")]
    SelfAdjointEigen(solvers::EvdError),
    #[error("Cholesky factorization failed: {0:?}")]
    Cholesky(solvers::LltError),
}

fn mat_to_array(mat: MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(i, j)| mat[(i, j)])
}

fn diag_to_array(diag: DiagRef<'_, f64>) -> Array1<f64> {
    let mat = diag.column_vector().as_mat();
    Array1::from_shape_fn(mat.nrows(), |i| mat[(i, 0)])
}

enum FaerStorage<'a> {
    Borrowed(MatRef<'a, f64>),
    Owned(Mat<f64>),
}

impl<'a> FaerStorage<'a> {
    #[inline]
    fn as_ref(&self) -> MatRef<'_, f64> {
        match self {
            FaerStorage::Borrowed(view) => *view,
            FaerStorage::Owned(mat) => mat.as_ref(),
        }
    }
}

pub struct FaerArrayView<'a> {
    storage: FaerStorage<'a>,
}

impl<'a> FaerArrayView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix2>) -> Self {
        let storage = if let Some(slice) = array.as_slice_memory_order() {
            if array.is_standard_layout() {
                FaerStorage::Borrowed(MatRef::from_row_major_slice(
                    slice,
                    array.nrows(),
                    array.ncols(),
                ))
            } else if array.t().is_standard_layout() {
                FaerStorage::Borrowed(MatRef::from_column_major_slice(
                    slice,
                    array.nrows(),
                    array.ncols(),
                ))
            } else {
                let (rows, cols) = array.dim();
                let owned = Mat::from_fn(rows, cols, |i, j| array[(i, j)]);
                FaerStorage::Owned(owned)
            }
        } else {
            let (rows, cols) = array.dim();
            let owned = Mat::from_fn(rows, cols, |i, j| array[(i, j)]);
            FaerStorage::Owned(owned)
        };
        Self { storage }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        self.storage.as_ref()
    }
}

pub struct FaerColView<'a> {
    storage: FaerStorage<'a>,
}

impl<'a> FaerColView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix1>) -> Self {
        let len = array.len();
        let storage = if let Some(slice) = array.as_slice() {
            FaerStorage::Borrowed(MatRef::from_row_major_slice(slice, len, 1))
        } else {
            let owned = Mat::from_fn(len, 1, |i, _| array[i]);
            FaerStorage::Owned(owned)
        };
        Self { storage }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        self.storage.as_ref()
    }
}

pub trait FaerSvd {
    fn svd(
        &self,
        compute_u: bool,
        compute_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerSvd for ArrayBase<S, Ix2> {
    fn svd(
        &self,
        compute_u: bool,
        compute_vt: bool,
    ) -> Result<(Option<Array2<f64>>, Array1<f64>, Option<Array2<f64>>), FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let faer_mat = faer_view.as_ref();
        if !compute_u && !compute_vt {
            let (rows, cols) = faer_mat.shape();
            let mut singular = Diag::<f64>::zeros(rows.min(cols));
            let par = get_global_parallelism();
            let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
                rows,
                cols,
                ComputeSvdVectors::No,
                ComputeSvdVectors::No,
                par,
                Default::default(),
            ));
            let mut stack = MemStack::new(&mut mem);
            svd::svd(
                faer_mat,
                singular.as_mut(),
                None,
                None,
                par,
                &mut stack,
                Default::default(),
            )
            .map_err(|_| FaerLinalgError::SvdNoConvergence)?;
            let singular_values = diag_to_array(singular.as_ref());
            return Ok((None, singular_values, None));
        }

        let (rows, cols) = faer_mat.shape();
        let compute_u_flag = if compute_u {
            ComputeSvdVectors::Full
        } else {
            ComputeSvdVectors::No
        };
        let compute_v_flag = if compute_vt {
            ComputeSvdVectors::Full
        } else {
            ComputeSvdVectors::No
        };

        let mut singular = Diag::<f64>::zeros(rows.min(cols));
        let mut u_storage = compute_u.then(|| Mat::<f64>::zeros(rows, rows));
        let mut v_storage = compute_vt.then(|| Mat::<f64>::zeros(cols, cols));

        let par = get_global_parallelism();
        let mut mem = MemBuffer::new(svd::svd_scratch::<f64>(
            rows,
            cols,
            compute_u_flag,
            compute_v_flag,
            par,
            Default::default(),
        ));
        let mut stack = MemStack::new(&mut mem);

        svd::svd(
            faer_mat.as_ref(),
            singular.as_mut(),
            u_storage.as_mut().map(|mat| mat.as_mut()),
            v_storage.as_mut().map(|mat| mat.as_mut()),
            par,
            &mut stack,
            Default::default(),
        )
        .map_err(|_| FaerLinalgError::SvdNoConvergence)?;

        let singular_values = diag_to_array(singular.as_ref());
        let u_opt = u_storage.map(|mat| mat_to_array(mat.as_ref()));
        let vt_opt = v_storage.map(|mat| {
            let mat_ref = mat.as_ref();
            Array2::from_shape_fn((mat_ref.ncols(), mat_ref.nrows()), |(i, j)| mat_ref[(j, i)])
        });

        Ok((u_opt, singular_values, vt_opt))
    }
}

pub trait FaerEigh {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerEigh for ArrayBase<S, Ix2> {
    fn eigh(&self, side: Side) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let eigen = faer_view
            .as_ref()
            .self_adjoint_eigen(side)
            .map_err(FaerLinalgError::SelfAdjointEigen)?;
        let values = diag_to_array(eigen.S());
        let vectors = mat_to_array(eigen.U());
        Ok((values, vectors))
    }
}

fn sequential_par() -> Par {
    get_global_parallelism()
}

fn normalized_start_vector(n: usize) -> Mat<f64> {
    let mut v0 = Mat::<f64>::zeros(n, 1);
    if n > 0 {
        v0[(0, 0)] = 1.0;
    }
    v0
}

fn compute_partial_eigen(
    mat: MatRef<'_, f64>,
    eigvecs: MatMut<'_, f64>,
    eigvals: &mut [f64],
    n_eigval: usize,
    params: PartialEigenParams,
) -> Result<PartialEigenInfo, FaerLinalgError> {
    let mut v0 = normalized_start_vector(mat.nrows());
    let par = sequential_par();
    let mut mem = MemBuffer::new(mf_eigen::partial_eigen_scratch(&mat, n_eigval, par, params));
    let mut stack = MemStack::new(&mut mem);
    let info = mf_eigen::partial_self_adjoint_eigen(
        eigvecs,
        eigvals,
        &mat,
        v0.as_ref().col(0),
        f64::EPSILON * 128.0,
        par,
        &mut stack,
        params,
    );
    Ok(info)
}

fn partial_eigen_params(n: usize, n_eigval: usize) -> PartialEigenParams {
    let min_dim = n.min(n_eigval.max(PARTIAL_MIN_DIM));
    let max_dim = n.min((min_dim * 2).max(min_dim));
    PartialEigenParams {
        min_dim,
        max_dim,
        max_restarts: 1000,
        ..Default::default()
    }
}

fn full_positive_eigh(
    matrix: &Array2<f64>,
    tolerance: f64,
) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
    let (values, vectors) = matrix.eigh(Side::Lower)?;
    if values.is_empty() {
        return Ok((values, Array2::zeros((matrix.nrows(), 0))));
    }
    let mut idxs: Vec<usize> = (0..values.len()).collect();
    idxs.sort_by(|&i, &j| {
        values[j]
            .partial_cmp(&values[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut filtered_vals = Vec::new();
    let mut filtered_vecs = Vec::new();
    for idx in idxs {
        let val = values[idx];
        if val > tolerance {
            filtered_vals.push(val);
            filtered_vecs.push(vectors.column(idx).to_owned());
        }
    }
    if filtered_vals.is_empty() {
        return Ok((Array1::zeros(0), Array2::zeros((matrix.nrows(), 0))));
    }
    let eigenvalues = Array1::from(filtered_vals);
    let mut eigenvectors = Array2::zeros((matrix.nrows(), eigenvalues.len()));
    for (col, vec) in filtered_vecs.into_iter().enumerate() {
        eigenvectors.column_mut(col).assign(&vec);
    }
    Ok((eigenvalues, eigenvectors))
}

pub fn partial_positive_eigh(
    matrix: &Array2<f64>,
    tolerance: f64,
) -> Result<(Array1<f64>, Array2<f64>), FaerLinalgError> {
    let n = matrix.nrows();
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }

    if n <= 32 {
        return full_positive_eigh(matrix, tolerance);
    }

    let faer_view = FaerArrayView::new(matrix);
    let mat = faer_view.as_ref();

    let mut largest_vec = Mat::<f64>::zeros(n, 1);
    let mut largest_val = vec![0.0_f64; 1];
    let params = partial_eigen_params(n, 1);
    let info = compute_partial_eigen(mat, largest_vec.as_mut(), &mut largest_val, 1, params)?;
    if info.n_converged_eigen < 1 {
        return full_positive_eigh(matrix, tolerance);
    }
    let max_eig = largest_val[0].abs();
    let tol = if max_eig > 0.0 {
        (max_eig * tolerance).max(tolerance)
    } else {
        tolerance
    };

    let mut guess = 8.min(n);
    let mut best_vals = Vec::<f64>::new();
    let mut best_vecs = Mat::<f64>::zeros(n, 0);

    loop {
        if guess == 0 {
            guess = 1;
        }
        let mut eigvecs = Mat::<f64>::zeros(n, guess);
        let mut eigvals = vec![0.0_f64; guess];
        let params = partial_eigen_params(n, guess);
        let info = compute_partial_eigen(mat, eigvecs.as_mut(), &mut eigvals, guess, params)?;
        if info.n_converged_eigen < guess {
            return full_positive_eigh(matrix, tolerance);
        }
        let mut indices: Vec<usize> = (0..guess).collect();
        indices.sort_by(|&i, &j| {
            eigvals[j]
                .partial_cmp(&eigvals[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut count = 0;
        for &idx in &indices {
            if eigvals[idx] > tol {
                count += 1;
            }
        }
        if count == 0 {
            return Ok((Array1::zeros(0), Array2::zeros((n, 0))));
        }
        if count < guess || guess == n {
            let mut final_vecs = Mat::<f64>::zeros(n, count);
            let mut final_vals = Vec::with_capacity(count);
            let mut filled = 0;
            for &idx in &indices {
                if eigvals[idx] > tol {
                    final_vals.push(eigvals[idx]);
                    final_vecs.col_mut(filled).copy_from(&eigvecs.col(idx));
                    filled += 1;
                }
                if filled == count {
                    break;
                }
            }
            best_vals = final_vals;
            best_vecs = final_vecs;
            break;
        }
        guess = Ord::min(n, guess * 2);
    }

    let eigenvalues = Array1::from(best_vals);
    let eigenvectors = mat_to_array(best_vecs.as_ref());
    Ok((eigenvalues, eigenvectors))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn random_psd(n: usize, rank: usize, rng: &mut StdRng) -> Array2<f64> {
        let mut mat = Array2::zeros((rank, n));
        for i in 0..rank {
            for j in 0..n {
                mat[(i, j)] = rng.gen_range(-1.0..1.0);
            }
        }
        let psd = mat.t().dot(&mat);
        psd
    }

    #[test]
    fn partial_positive_matches_full_eigenpairs() {
        let mut rng = StdRng::seed_from_u64(42);
        let matrix = random_psd(48, 17, &mut rng);

        let (partial_vals, partial_vecs) = partial_positive_eigh(&matrix, 1e-12).unwrap();
        let (full_vals, full_vecs) = matrix.eigh(Side::Lower).unwrap();

        let max_full = full_vals.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = if max_full > 0.0 {
            max_full * 1e-12
        } else {
            1e-12
        };
        let mut ordered: Vec<(usize, f64)> = full_vals
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > tol)
            .map(|(i, &val)| (i, val))
            .collect();
        ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        assert_eq!(partial_vals.len(), ordered.len());
        for (idx, (&val_partial, &(full_idx, full_val))) in
            partial_vals.iter().zip(ordered.iter()).enumerate()
        {
            assert!(
                (val_partial - full_val).abs() <= 1e-8 * full_vals[0].abs().max(1.0),
                "eigenvalue mismatch at {}: {} vs {}",
                idx,
                val_partial,
                full_val
            );

            let vec_partial = partial_vecs.column(idx);
            let vec_full = full_vecs.column(full_idx);
            let alignment = vec_partial.dot(&vec_full);
            assert!(alignment.abs() > 1.0 - 1e-6);
        }
    }
}

pub struct FaerCholeskyFactor {
    factor: solvers::Llt<f64>,
}

impl FaerCholeskyFactor {
    pub fn solve_vec(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let rhs_view = FaerColView::new(rhs);
        let sol = self.factor.solve(rhs_view.as_ref());
        Array1::from_shape_fn(rhs.len(), |i| sol[(i, 0)])
    }

    pub fn solve_mat(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let rhs_view = FaerArrayView::new(rhs);
        let sol = self.factor.solve(rhs_view.as_ref());
        mat_to_array(sol.as_ref())
    }

    pub fn diag(&self) -> Array1<f64> {
        diag_to_array(self.factor.L().diagonal())
    }
}

pub trait FaerCholesky {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerCholesky for ArrayBase<S, Ix2> {
    fn cholesky(&self, side: Side) -> Result<FaerCholeskyFactor, FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let factor = faer_view
            .as_ref()
            .llt(side)
            .map_err(FaerLinalgError::Cholesky)?;
        Ok(FaerCholeskyFactor { factor })
    }
}

pub trait FaerQr {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerQr for ArrayBase<S, Ix2> {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError> {
        let faer_view = FaerArrayView::new(self);
        let qr = faer_view.as_ref().qr();
        let q = qr.compute_Q();
        let r = qr.R();
        Ok((mat_to_array(q.as_ref()), mat_to_array(r)))
    }
}
const PARTIAL_MIN_DIM: usize = 32;
