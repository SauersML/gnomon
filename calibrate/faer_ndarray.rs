use dyn_stack::{MemBuffer, MemStack};
use faer::diag::{Diag, DiagRef};
use faer::linalg::solvers::{self, Solve};
use faer::linalg::svd::{self, ComputeSvdVectors};
use faer::{Mat, MatMut, MatRef, Side, get_global_parallelism};
use ndarray::{Array1, Array2, ArrayBase, CowArray, Data, Ix1, Ix2};
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

pub struct FaerArrayView<'a> {
    rows: usize,
    cols: usize,
    data: CowArray<'a, f64, Ix2>,
}

pub fn array2_to_matmut(array: &mut Array2<f64>) -> MatMut<'_, f64> {
    let (rows, cols) = array.dim();
    let slice = array
        .as_slice_mut()
        .expect("standard layout guarantees contiguous slice");
    MatMut::from_row_major_slice_mut(slice, rows, cols)
}

pub fn array1_to_col_matmut(array: &mut Array1<f64>) -> MatMut<'_, f64> {
    let len = array.len();
    let slice = array
        .as_slice_mut()
        .expect("standard layout guarantees contiguous slice");
    MatMut::from_row_major_slice_mut(slice, len, 1)
}

impl<'a> FaerArrayView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix2>) -> Self {
        let (rows, cols) = array.dim();
        let data = array.as_standard_layout();
        Self { rows, cols, data }
    }

    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, f64> {
        let slice = self
            .data
            .as_slice()
            .expect("standard layout guarantees contiguous slice");
        MatRef::from_row_major_slice(slice, self.rows, self.cols)
    }

    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

pub struct FaerVectorView<'a> {
    len: usize,
    data: CowArray<'a, f64, Ix1>,
}

impl<'a> FaerVectorView<'a> {
    pub fn new<S: Data<Elem = f64>>(array: &'a ArrayBase<S, Ix1>) -> Self {
        let len = array.len();
        let data = array.as_standard_layout();
        Self { len, data }
    }

    #[inline]
    pub fn as_col_ref(&self) -> MatRef<'_, f64> {
        let slice = self
            .data
            .as_slice()
            .expect("standard layout guarantees contiguous slice");
        MatRef::from_column_major_slice(slice, self.len, 1)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
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
        let faer_view = FaerArrayView::new(self);
        let (rows, cols) = faer_view.dims();
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
            faer_view.as_ref(),
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
        let eigen = faer::linalg::solvers::SelfAdjointEigen::new(faer_view.as_ref(), side)
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
        let rhs_view = FaerVectorView::new(rhs);
        let sol = self.factor.solve(rhs_view.as_col_ref());
        Array1::from_shape_fn(rhs_view.len(), |i| sol[(i, 0)])
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
        let factor = faer::linalg::solvers::Llt::new(faer_view.as_ref(), side)
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
        let qr = faer::linalg::solvers::Qr::new(faer_view.as_ref());
        let q = qr.compute_Q();
        let r = qr.R();
        Ok((mat_to_array(q.as_ref()), mat_to_array(r)))
    }
}
