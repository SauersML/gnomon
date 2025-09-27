use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagRef};
use faer::linalg::solvers::{self, Solve};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::{Mat, MatRef, Side, get_global_parallelism};
use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
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

fn array_to_faer<S: Data<Elem = f64>>(array: &ArrayBase<S, Ix2>) -> Mat<f64> {
    let (rows, cols) = array.dim();
    Mat::from_fn(rows, cols, |i, j| array[(i, j)])
}

fn mat_to_array(mat: MatRef<'_, f64>) -> Array2<f64> {
    Array2::from_shape_fn((mat.nrows(), mat.ncols()), |(i, j)| mat[(i, j)])
}

fn diag_to_array(diag: DiagRef<'_, f64>) -> Array1<f64> {
    let mat = diag.column_vector().as_mat();
    Array1::from_shape_fn(mat.nrows(), |i| mat[(i, 0)])
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
        let faer_mat = array_to_faer(self);
        if !compute_u && !compute_vt {
            let singular_values = faer_mat
                .singular_values()
                .map_err(|_| FaerLinalgError::SvdNoConvergence)?;
            return Ok((None, Array1::from_vec(singular_values), None));
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
        let faer_mat = array_to_faer(self);
        let eigen = faer_mat
            .self_adjoint_eigen(side)
            .map_err(FaerLinalgError::SelfAdjointEigen)?;
        let values = diag_to_array(eigen.S());
        let vectors = mat_to_array(eigen.U());
        Ok((values, vectors))
    }
}

pub struct FaerCholeskyFactor {
    factor: solvers::Llt<f64>,
}

impl FaerCholeskyFactor {
    pub fn solve_vec(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let rhs_mat = Mat::from_fn(rhs.len(), 1, |i, _| rhs[i]);
        let sol = self.factor.solve(rhs_mat.as_ref());
        Array1::from_shape_fn(rhs.len(), |i| sol[(i, 0)])
    }

    pub fn solve_mat(&self, rhs: &Array2<f64>) -> Array2<f64> {
        let (rows, cols) = rhs.dim();
        let rhs_mat = Mat::from_fn(rows, cols, |i, j| rhs[(i, j)]);
        let sol = self.factor.solve(rhs_mat.as_ref());
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
        let faer_mat = array_to_faer(self);
        let factor = faer_mat.llt(side).map_err(FaerLinalgError::Cholesky)?;
        Ok(FaerCholeskyFactor { factor })
    }
}

pub trait FaerQr {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError>;
}

impl<S: Data<Elem = f64>> FaerQr for ArrayBase<S, Ix2> {
    fn qr(&self) -> Result<(Array2<f64>, Array2<f64>), FaerLinalgError> {
        let faer_mat = array_to_faer(self);
        let qr = faer_mat.qr();
        let q = qr.compute_Q();
        let r = qr.R();
        Ok((mat_to_array(q.as_ref()), mat_to_array(r)))
    }
}
