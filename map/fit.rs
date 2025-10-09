use super::progress::{FitProgressObserver, FitProgressStage, NoopFitProgress};
use core::cmp::{Ordering, min};
use core::fmt;
use core::marker::PhantomData;
use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::col::Col;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::linalg::{temp_mat_scratch, temp_mat_uninit};
use faer::mat::AsMatMut;
use faer::matrix_free::LinOp;
use faer::matrix_free::eigen::{
    PartialEigenInfo, PartialEigenParams, partial_eigen_scratch, partial_self_adjoint_eigen,
};
use faer::prelude::ReborrowMut;
use faer::{
    Accum, ColMut, Mat, MatMut, MatRef, Par, Side, get_global_parallelism, set_global_parallelism,
    unzip, zip,
};
use rayon::prelude::*;
use serde::de::Error as DeError;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::UnsafeCell;
use std::convert::Infallible;
use std::error::Error;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
pub const DEFAULT_BLOCK_WIDTH: usize = 2_048;
const DENSE_EIGEN_FALLBACK_THRESHOLD: usize = 64;
const INITIAL_PARTIAL_COMPONENTS: usize = 32;
const MAX_PARTIAL_COMPONENTS: usize = 512;

struct ParallelismGuard {
    previous: Par,
}

impl ParallelismGuard {
    fn new() -> Self {
        let previous = get_global_parallelism();
        let desired = Par::rayon(rayon::current_num_threads());
        set_global_parallelism(desired);
        Self { previous }
    }

    fn active_parallelism(&self) -> Par {
        get_global_parallelism()
    }
}

impl Drop for ParallelismGuard {
    fn drop(&mut self) {
        set_global_parallelism(self.previous);
    }
}

#[derive(Debug)]
struct OperatorError;

pub trait VariantBlockSource {
    type Error;

    fn n_samples(&self) -> usize;
    fn n_variants(&self) -> usize;
    fn reset(&mut self) -> Result<(), Self::Error>;
    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error>;
}

pub struct DenseBlockSource<'a> {
    data: &'a [f64],
    dims: (usize, usize),
    cursor: usize,
}

impl<'a> DenseBlockSource<'a> {
    pub fn new(data: &'a [f64], n_samples: usize, n_variants: usize) -> Result<Self, HwePcaError> {
        if n_samples == 0 {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: n_samples must be positive",
            ));
        }
        if n_variants == 0 {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: n_variants must be positive",
            ));
        }
        let expected = n_samples
            .checked_mul(n_variants)
            .ok_or_else(|| HwePcaError::InvalidInput("DenseBlockSource: dimension overflow"))?;
        if data.len() != expected {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: data length does not match dimensions",
            ));
        }
        Ok(Self {
            data,
            dims: (n_samples, n_variants),
            cursor: 0,
        })
    }
}

impl<'a> VariantBlockSource for DenseBlockSource<'a> {
    type Error = Infallible;

    fn n_samples(&self) -> usize {
        self.dims.0
    }

    fn n_variants(&self) -> usize {
        self.dims.1
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.cursor = 0;
        Ok(())
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        let remaining = self.n_variants().saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok(0);
        }
        let ncols = min(max_variants, remaining);
        let nrows = self.n_samples();
        let len = nrows * ncols;
        let start = self.cursor * nrows;
        let end = start + len;
        storage[..len].copy_from_slice(&self.data[start..end]);
        self.cursor += ncols;
        Ok(ncols)
    }
}

#[derive(Debug)]
pub enum HwePcaError {
    InvalidInput(&'static str),
    Source(Box<dyn Error + Send + Sync + 'static>),
    Eigen(String),
}

impl fmt::Display for HwePcaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HwePcaError::InvalidInput(msg) => f.write_str(msg),
            HwePcaError::Source(err) => write!(f, "source error: {err}"),
            HwePcaError::Eigen(msg) => f.write_str(msg),
        }
    }
}

impl Error for HwePcaError {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HweScaler {
    frequencies: Vec<f64>,
    scales: Vec<f64>,
}

impl HweScaler {
    fn new(frequencies: Vec<f64>, scales: Vec<f64>) -> Self {
        Self {
            frequencies,
            scales,
        }
    }

    pub fn allele_frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    pub fn variant_scales(&self) -> &[f64] {
        &self.scales
    }

    pub(crate) fn standardize_block(&self, block: MatMut<'_, f64>, variant_range: Range<usize>) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let scales = &self.scales[start..end];
        let filled = freqs.len();

        debug_assert_eq!(filled, block.ncols());
        let block = block.subcols_mut(0, filled);

        let apply_standardization = |mut column: ColMut<'_, f64>, mean: f64, inv: f64| {
            if inv == 0.0 {
                column.fill(0.0);
                return;
            }

            let contiguous = column
                .try_as_col_major_mut()
                .expect("projection block column must be contiguous");
            let values = contiguous.as_slice_mut();
            standardize_column_simd(values, mean, inv);
        };

        let use_parallel = filled >= 32
            && rayon::current_num_threads() > 1
            && matches!(get_global_parallelism(), Par::Seq);

        if use_parallel {
            block
                .par_col_iter_mut()
                .enumerate()
                .for_each(|(idx, column)| {
                    let freq = freqs[idx];
                    let scale = scales[idx];
                    let mean = 2.0 * freq;
                    let denom = scale.max(HWE_SCALE_FLOOR);
                    let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                    apply_standardization(column, mean, inv);
                });
        } else {
            for (idx, column) in block.col_iter_mut().enumerate() {
                let freq = freqs[idx];
                let scale = scales[idx];
                let mean = 2.0 * freq;
                let denom = scale.max(HWE_SCALE_FLOOR);
                let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                apply_standardization(column, mean, inv);
            }
        }
    }

    pub(crate) fn standardize_block_with_mask(
        &self,
        block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        presence_out: MatMut<'_, f64>,
    ) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let filled = freqs.len();

        debug_assert_eq!(filled, block.ncols());
        debug_assert_eq!(filled, presence_out.ncols());
        debug_assert_eq!(block.nrows(), presence_out.nrows());

        let block_ref = block.as_ref();

        for (presence_col, column) in presence_out
            .subcols_mut(0, filled)
            .col_iter_mut()
            .zip(block_ref.subcols(0, filled).col_iter())
        {
            let mut presence_slice = presence_col
                .try_as_col_major_mut()
                .expect("presence column must be contiguous");
            let column_slice = column
                .try_as_col_major()
                .expect("projection column must be contiguous");
            for (presence, &raw) in presence_slice
                .as_mut()
                .iter_mut()
                .zip(column_slice.as_slice())
            {
                *presence = if raw.is_finite() { 1.0 } else { 0.0 };
            }
        }

        self.standardize_block(block, variant_range);
    }
}

#[cfg(target_feature = "avx512f")]
const STANDARDIZATION_SIMD_LANES: usize = 8;

#[cfg(all(
    not(target_feature = "avx512f"),
    any(
        target_feature = "avx",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
const STANDARDIZATION_SIMD_LANES: usize = 4;

#[cfg(all(
    not(target_feature = "avx512f"),
    not(any(
        target_feature = "avx",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))
))]
const STANDARDIZATION_SIMD_LANES: usize = 2;

#[inline(always)]
fn standardize_column_simd(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_impl::<STANDARDIZATION_SIMD_LANES>(values, mean, inv);
}

#[inline(always)]
fn standardize_column_simd_impl<const LANES: usize>(values: &mut [f64], mean: f64, inv: f64)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mean_simd = Simd::<f64, LANES>::splat(mean);
    let inv_simd = Simd::<f64, LANES>::splat(inv);
    let zero = Simd::<f64, LANES>::splat(0.0);

    let (chunks, remainder) = values.as_chunks_mut::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_slice(chunk);
        let mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let result = mask.select(standardized, zero);
        *chunk = result.to_array();
    }

    for value in remainder {
        let raw = *value;
        *value = if raw.is_finite() {
            (raw - mean) * inv
        } else {
            0.0
        };
    }
}

#[inline(always)]
fn sum_and_count_finite(values: &[f64]) -> (f64, usize) {
    sum_and_count_finite_impl::<STANDARDIZATION_SIMD_LANES>(values)
}

#[inline(always)]
fn sum_and_count_finite_impl<const LANES: usize>(values: &[f64]) -> (f64, usize)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    let zero = Simd::<f64, LANES>::splat(0.0);

    let (chunks, remainder) = values.as_chunks::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_slice(chunk);
        let mask = lane.is_finite();
        let finite = mask.select(lane, zero);
        sum += finite.reduce_sum();
        count += mask
            .to_array()
            .into_iter()
            .map(|flag| flag as usize)
            .sum::<usize>();
    }

    for &value in remainder {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }

    (sum, count)
}

#[derive(Clone, Debug)]
pub struct HwePcaModel {
    n_samples: usize,
    n_variants: usize,
    scaler: HweScaler,
    eigenvalues: Vec<f64>,
    singular_values: Vec<f64>,
    sample_basis: Mat<f64>,
    sample_scores: Mat<f64>,
    loadings: Mat<f64>,
}

impl HwePcaModel {
    pub fn fit<S>(source: &mut S) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
    {
        let mut progress = NoopFitProgress::default();
        Self::fit_with_progress(source, &mut progress)
    }

    pub fn fit_with_progress<S, P>(source: &mut S, progress: &mut P) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
        P: FitProgressObserver,
    {
        let n_samples = source.n_samples();
        let n_variants = source.n_variants();

        if n_samples < 2 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least two samples",
            ));
        }
        if n_variants == 0 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least one variant",
            ));
        }

        let parallelism_guard = ParallelismGuard::new();
        if matches!(parallelism_guard.active_parallelism(), Par::Seq) {}
        let block_capacity = min(DEFAULT_BLOCK_WIDTH.max(1), n_variants);
        let mut block_storage = vec![0.0f64; n_samples * block_capacity];

        let scaler =
            stream_allele_statistics(source, block_capacity, &mut block_storage, progress)?;

        let operator =
            StandardizedCovarianceOp::new(source, &scaler, block_capacity, block_storage);

        progress.on_stage_start(FitProgressStage::GramMatrix, n_variants);
        let decomposition_result = compute_covariance_eigenpairs(&operator);
        let (source, mut block_storage) = operator.into_parts();
        let decomposition = decomposition_result?;
        progress.on_stage_finish(FitProgressStage::GramMatrix);

        if decomposition.values.is_empty() {
            return Err(HwePcaError::Eigen(
                "All eigenvalues are numerically zero; increase cohort size or review input data"
                    .into(),
            ));
        }

        let (singular_values, sample_scores) = build_sample_scores(n_samples, &decomposition);

        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        let loadings = compute_variant_loadings(
            source,
            &scaler,
            block_capacity,
            &mut block_storage,
            decomposition.vectors.as_ref(),
            &singular_values,
            progress,
        )?;

        Ok(Self {
            n_samples,
            n_variants,
            scaler,
            eigenvalues: decomposition.values,
            singular_values,
            sample_basis: decomposition.vectors,
            sample_scores,
            loadings,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn n_variants(&self) -> usize {
        self.n_variants
    }

    pub fn scaler(&self) -> &HweScaler {
        &self.scaler
    }

    pub fn components(&self) -> usize {
        self.eigenvalues.len()
    }

    pub fn explained_variance(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total: f64 = self.eigenvalues.iter().copied().sum();
        if total > 0.0 {
            self.eigenvalues
                .iter()
                .map(|&lambda| lambda / total)
                .collect()
        } else {
            vec![0.0; self.eigenvalues.len()]
        }
    }

    pub fn sample_basis(&self) -> MatRef<'_, f64> {
        self.sample_basis.as_ref()
    }

    pub fn sample_scores(&self) -> MatRef<'_, f64> {
        self.sample_scores.as_ref()
    }

    pub fn variant_loadings(&self) -> MatRef<'_, f64> {
        self.loadings.as_ref()
    }
}

struct Eigenpairs {
    values: Vec<f64>,
    vectors: Mat<f64>,
}

struct StandardizedCovarianceOp<'a, S>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    source: *mut S,
    scaler: &'a HweScaler,
    block_storage: UnsafeCell<Option<Vec<f64>>>,
    n_samples: usize,
    n_variants: usize,
    block_capacity: usize,
    scale: f64,
    error: UnsafeCell<Option<HwePcaError>>,
    marker: PhantomData<&'a mut S>,
}

impl<'a, S> StandardizedCovarianceOp<'a, S>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    fn new(
        source: &'a mut S,
        scaler: &'a HweScaler,
        block_capacity: usize,
        block_storage: Vec<f64>,
    ) -> Self {
        let n_samples = source.n_samples();
        let n_variants = source.n_variants();
        assert_eq!(
            block_storage.len(),
            n_samples * block_capacity,
            "block storage size must match block capacity",
        );
        let scale = 1.0 / ((n_samples - 1) as f64);
        Self {
            source: source as *mut S,
            scaler,
            block_storage: UnsafeCell::new(Some(block_storage)),
            n_samples,
            n_variants,
            block_capacity,
            scale,
            error: UnsafeCell::new(None),
            marker: PhantomData,
        }
    }

    fn into_parts(self) -> (&'a mut S, Vec<f64>) {
        let source = self.source;
        let storage = unsafe {
            (*self.block_storage.get())
                .take()
                .expect("block storage already taken")
        };
        (unsafe { &mut *source }, storage)
    }

    fn take_error(&self) -> Option<HwePcaError> {
        unsafe { (*self.error.get()).take() }
    }

    fn fail_source(&self, err: S::Error) -> ! {
        self.record_error(HwePcaError::Source(Box::new(err)))
    }

    fn fail_invalid(&self, msg: &'static str) -> ! {
        self.record_error(HwePcaError::InvalidInput(msg))
    }

    fn record_error(&self, err: HwePcaError) -> ! {
        let slot = unsafe { &mut *self.error.get() };
        if slot.is_none() {
            *slot = Some(err);
        }
        std::panic::panic_any(OperatorError);
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }
}

impl<'a, S> fmt::Debug for StandardizedCovarianceOp<'a, S>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StandardizedCovarianceOp")
            .field("n_samples", &self.n_samples)
            .field("n_variants", &self.n_variants)
            .field("block_capacity", &self.block_capacity)
            .finish()
    }
}

unsafe impl<'a, S> Sync for StandardizedCovarianceOp<'a, S>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
}

impl<'a, S> LinOp<f64> for StandardizedCovarianceOp<'a, S>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        match par {
            Par::Seq => {}
            _ => {}
        }
        temp_mat_scratch::<f64>(self.block_capacity, rhs_ncols)
    }

    fn nrows(&self) -> usize {
        self.n_samples
    }

    fn ncols(&self) -> usize {
        self.n_samples
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        match par {
            Par::Seq => {}
            _ => {}
        }
        debug_assert_eq!(out.nrows(), self.n_samples);
        debug_assert_eq!(rhs.nrows(), self.n_samples);

        out.fill(0.0);

        if rhs.ncols() == 0 {
            return;
        }

        let block_storage_opt = unsafe { &mut *self.block_storage.get() };
        let block_storage = block_storage_opt
            .as_mut()
            .expect("block storage missing during operator application");

        let source = unsafe { &mut *self.source };

        if let Err(err) = source.reset() {
            self.fail_source(err);
        }

        let (mut proj_uninit, _) =
            unsafe { temp_mat_uninit::<f64, _, _>(self.block_capacity, rhs.ncols(), stack) };
        let mut proj_storage = proj_uninit.as_mat_mut();

        let mut processed = 0usize;

        loop {
            let filled = match source.next_block_into(self.block_capacity, &mut block_storage[..]) {
                Ok(filled) => filled,
                Err(err) => self.fail_source(err),
            };

            if filled == 0 {
                break;
            }

            if processed + filled > self.n_variants {
                self.fail_invalid("VariantBlockSource returned more variants than reported");
            }

            let block_slice = &mut block_storage[..self.n_samples * filled];
            let mut block =
                MatMut::from_column_major_slice_mut(block_slice, self.n_samples, filled);
            self.scaler
                .standardize_block(block.as_mut(), processed..processed + filled);

            let mut proj_block = proj_storage.rb_mut().subrows_mut(0, filled);

            matmul(
                proj_block.as_mut(),
                Accum::Replace,
                block.as_ref().transpose(),
                rhs,
                1.0,
                get_global_parallelism(),
            );

            matmul(
                out.rb_mut(),
                Accum::Add,
                block.as_ref(),
                proj_block.as_ref(),
                self.scale,
                get_global_parallelism(),
            );

            processed += filled;
        }

        if processed != self.n_variants {
            self.fail_invalid("VariantBlockSource terminated early during covariance accumulation");
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

fn compute_covariance_eigenpairs<S>(
    operator: &StandardizedCovarianceOp<'_, S>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    let n = operator.n_samples();
    if n == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(0, 0),
        });
    }

    if n <= DENSE_EIGEN_FALLBACK_THRESHOLD {
        return compute_covariance_eigenpairs_dense(operator);
    }

    let max_rank = n.saturating_sub(1);
    if max_rank == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    let mut max_partial = ((n - 1) / 2).min(MAX_PARTIAL_COMPONENTS);
    if max_partial == 0 {
        return compute_covariance_eigenpairs_dense(operator);
    }
    max_partial = max_partial.min(max_rank);

    let mut target = INITIAL_PARTIAL_COMPONENTS.min(max_partial);
    if target == 0 {
        target = max_partial;
    }

    let par = Par::Seq;
    let normalization = (n as f64).sqrt();
    let v0 = Col::from_fn(n, |_| 1.0 / normalization);

    loop {
        let params = partial_solver_params(n, target);
        let (info, eigvals, eigvecs) = run_partial_eigensolver(operator, target, par, &v0, params)?;

        let n_converged = info.n_converged_eigen.min(target);
        let mut ordering = Vec::with_capacity(n_converged);
        for idx in 0..n_converged {
            ordering.push((idx, eigvals[idx]));
        }

        ordering.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

        let mut positive = 0usize;
        for &(_, value) in &ordering {
            if value <= EIGENVALUE_EPSILON {
                break;
            }
            positive += 1;
        }

        if positive == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        if positive == target && target < max_partial {
            let next_target = (target * 2).min(max_partial);
            if next_target > target {
                target = next_target;
                continue;
            }
        }

        let mut values = Vec::with_capacity(positive);
        let mut vectors = Mat::zeros(n, positive);
        for (out_idx, (src_idx, value)) in ordering.into_iter().take(positive).enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = eigvecs[(row, src_idx)];
            }
        }

        return Ok(Eigenpairs { values, vectors });
    }
}

fn compute_covariance_eigenpairs_dense<S>(
    operator: &StandardizedCovarianceOp<'_, S>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    let mut covariance = accumulate_covariance_matrix(operator)?;
    let n = covariance.nrows();

    for col in 0..n {
        for row in 0..col {
            let avg = 0.5 * (covariance[(row, col)] + covariance[(col, row)]);
            covariance[(row, col)] = avg;
            covariance[(col, row)] = avg;
        }
    }

    let eig = SelfAdjointEigen::new(covariance.as_ref(), Side::Lower)
        .map_err(|err| HwePcaError::Eigen(format!("dense eigendecomposition failed: {err:?}")))?;

    let diag = eig.S();
    let basis = eig.U();

    let mut ordering = Vec::with_capacity(n);
    for idx in 0..n {
        ordering.push((idx, diag[idx]));
    }

    ordering.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

    let mut positive = 0usize;
    for &(_, value) in &ordering {
        if value <= EIGENVALUE_EPSILON {
            break;
        }
        positive += 1;
    }

    if positive == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    let mut values = Vec::with_capacity(positive);
    let mut vectors = Mat::zeros(n, positive);
    for (out_idx, (src_idx, value)) in ordering.into_iter().take(positive).enumerate() {
        values.push(value);
        for row in 0..n {
            vectors[(row, out_idx)] = basis[(row, src_idx)];
        }
    }

    Ok(Eigenpairs { values, vectors })
}

fn accumulate_covariance_matrix<S>(
    operator: &StandardizedCovarianceOp<'_, S>,
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    let n_samples = operator.n_samples;
    let n_variants = operator.n_variants;
    let mut covariance = Mat::zeros(n_samples, n_samples);

    if n_samples == 0 || n_variants == 0 {
        return Ok(covariance);
    }

    let block_capacity = operator.block_capacity;

    let block_storage_opt = unsafe { &mut *operator.block_storage.get() };
    let block_storage = block_storage_opt
        .as_mut()
        .expect("block storage missing during covariance accumulation");

    let source = unsafe { &mut *operator.source };
    source
        .reset()
        .map_err(|err| HwePcaError::Source(Box::new(err)))?;

    let mut processed = 0usize;

    loop {
        let filled = source
            .next_block_into(block_capacity, &mut block_storage[..])
            .map_err(|err| HwePcaError::Source(Box::new(err)))?;

        if filled == 0 {
            break;
        }

        if processed + filled > n_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource returned more variants than reported",
            ));
        }

        let mut block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        operator
            .scaler
            .standardize_block(block.as_mut(), processed..processed + filled);

        matmul(
            covariance.as_mut(),
            Accum::Add,
            block.as_ref(),
            block.as_ref().transpose(),
            operator.scale,
            get_global_parallelism(),
        );

        processed += filled;
    }

    if processed != n_variants {
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource terminated early during covariance accumulation",
        ));
    }

    Ok(covariance)
}

fn partial_solver_params(n: usize, target: usize) -> PartialEigenParams {
    let mut params = PartialEigenParams::default();
    let max_available = n.saturating_sub(1);
    let min_dim = ((2 * target).max(64)).min(max_available);
    let mut max_dim = ((4 * target).max(128)).min(max_available);
    if max_dim <= min_dim {
        max_dim = min_dim.min(max_available);
    }
    params.min_dim = min_dim.max(target);
    params.max_dim = max_dim.max(params.min_dim);
    params.max_restarts = 2048;
    params
}

fn run_partial_eigensolver<S>(
    operator: &StandardizedCovarianceOp<'_, S>,
    target: usize,
    par: Par,
    v0: &Col<f64>,
    params: PartialEigenParams,
) -> Result<(PartialEigenInfo, Vec<f64>, Mat<f64>), HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
{
    let n = operator.n_samples();
    let mut eigvecs = Mat::zeros(n, target);
    let mut eigvals = vec![0.0f64; target];

    let scratch = partial_eigen_scratch(operator, target, par, params);
    let mut mem = MemBuffer::new(scratch);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut stack = MemStack::new(&mut mem);
        partial_self_adjoint_eigen(
            eigvecs.as_mut(),
            &mut eigvals,
            operator,
            v0.as_ref(),
            f64::EPSILON * 128.0,
            par,
            &mut stack,
            params,
        )
    }));

    let info = match result {
        Ok(info) => info,
        Err(payload) => {
            if payload.downcast_ref::<OperatorError>().is_some() {
                let err = operator.take_error().unwrap_or_else(|| {
                    HwePcaError::Eigen(
                        "matrix-free eigensolver aborted with an internal error".into(),
                    )
                });
                return Err(err);
            }
            std::panic::resume_unwind(payload);
        }
    };

    if let Some(err) = operator.take_error() {
        return Err(err);
    }

    Ok((info, eigvals, eigvecs))
}

fn stream_allele_statistics<S, P>(
    source: &mut S,
    block_capacity: usize,
    block_storage: &mut [f64],
    progress: &mut P,
) -> Result<HweScaler, HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver,
{
    let n_samples = source.n_samples();
    let n_variants = source.n_variants();
    let mut allele_sums = vec![0.0f64; n_variants];
    let mut allele_counts = vec![0usize; n_variants];
    let mut processed = 0usize;

    progress.on_stage_start(FitProgressStage::AlleleStatistics, n_variants);

    source
        .reset()
        .map_err(|e| HwePcaError::Source(Box::new(e)))?;

    loop {
        let filled = source
            .next_block_into(block_capacity, block_storage)
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        if filled == 0 {
            break;
        }
        if processed + filled > n_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource returned more variants than reported",
            ));
        }

        let block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        for (local_col, column) in block.as_ref().col_iter().enumerate() {
            let variant_index = processed + local_col;
            let contiguous = column
                .try_as_col_major()
                .expect("variant block column must be contiguous");
            let (sum, calls) = sum_and_count_finite(contiguous.as_slice());
            allele_sums[variant_index] += sum;
            allele_counts[variant_index] += calls;
        }

        processed += filled;
        progress.on_stage_advance(FitProgressStage::AlleleStatistics, processed);
    }

    if processed != n_variants {
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource terminated early during allele counting",
        ));
    }

    progress.on_stage_finish(FitProgressStage::AlleleStatistics);

    let mut frequencies = vec![0.0f64; n_variants];
    let mut scales = vec![1.0f64; n_variants];

    for (idx, (&sum, &calls)) in allele_sums.iter().zip(&allele_counts).enumerate() {
        if calls == 0 {
            frequencies[idx] = 0.0;
            scales[idx] = HWE_SCALE_FLOOR;
            continue;
        }
        let mean_genotype = sum / (calls as f64);
        let freq = (mean_genotype / 2.0).clamp(0.0, 1.0);
        let variance = (2.0 * freq * (1.0 - freq)).max(HWE_VARIANCE_EPSILON);
        frequencies[idx] = freq;
        let scale = variance.sqrt();
        scales[idx] = if scale < HWE_SCALE_FLOOR {
            HWE_SCALE_FLOOR
        } else {
            scale
        };
    }

    Ok(HweScaler::new(frequencies, scales))
}

fn build_sample_scores(n_samples: usize, decomposition: &Eigenpairs) -> (Vec<f64>, Mat<f64>) {
    let mut singular_values = Vec::with_capacity(decomposition.values.len());
    let mut sample_scores = decomposition.vectors.clone();

    for (&lambda, mut column) in decomposition
        .values
        .iter()
        .zip(sample_scores.col_iter_mut())
    {
        let sigma = ((n_samples - 1) as f64 * lambda).sqrt();
        singular_values.push(sigma);
        zip!(&mut column).for_each(|unzip!(value)| {
            *value *= sigma;
        });
    }

    (singular_values, sample_scores)
}

fn compute_variant_loadings<S, P>(
    source: &mut S,
    scaler: &HweScaler,
    block_capacity: usize,
    block_storage: &mut [f64],
    sample_basis: MatRef<'_, f64>,
    singular_values: &[f64],
    progress: &mut P,
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver,
{
    let n_samples = source.n_samples();
    let n_variants = source.n_variants();
    let n_components = singular_values.len();
    let mut loadings = Mat::zeros(n_variants, n_components);
    let mut processed = 0usize;
    let mut chunk_storage = vec![0.0f64; block_capacity * n_components];
    let inverse_singular: Vec<f64> = singular_values
        .iter()
        .map(|&sigma| if sigma > 0.0 { 1.0 / sigma } else { 0.0 })
        .collect();

    progress.on_stage_start(FitProgressStage::Loadings, n_variants);

    loop {
        let filled = source
            .next_block_into(block_capacity, block_storage)
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        if filled == 0 {
            break;
        }
        if processed + filled > n_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource returned more variants than reported",
            ));
        }

        let mut block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );
        scaler.standardize_block(block.as_mut(), processed..processed + filled);

        let block_ref = block.as_ref();

        let mut chunk = MatMut::from_column_major_slice_mut(
            &mut chunk_storage[..filled * n_components],
            filled,
            n_components,
        );

        matmul(
            chunk.as_mut(),
            Accum::Replace,
            block_ref.transpose(),
            sample_basis,
            1.0,
            faer::get_global_parallelism(),
        );

        for local_col in 0..filled {
            let global_variant = processed + local_col;
            for component in 0..n_components {
                loadings[(global_variant, component)] =
                    chunk[(local_col, component)] * inverse_singular[component];
            }
        }

        processed += filled;
        progress.on_stage_advance(FitProgressStage::Loadings, processed);
    }

    if processed != n_variants {
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource terminated early while computing loadings",
        ));
    }

    progress.on_stage_finish(FitProgressStage::Loadings);

    Ok(loadings)
}

#[derive(Serialize, Deserialize)]
struct MatrixData {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl MatrixData {
    fn from_mat(mat: MatRef<'_, f64>) -> Self {
        let mut data = Vec::with_capacity(mat.nrows() * mat.ncols());
        for col in 0..mat.ncols() {
            for row in 0..mat.nrows() {
                data.push(mat[(row, col)]);
            }
        }
        Self {
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            data,
        }
    }

    fn into_mat(self) -> Result<Mat<f64>, String> {
        let MatrixData { nrows, ncols, data } = self;
        if data.len() != nrows * ncols {
            return Err("matrix data length does not match dimensions".into());
        }
        let mut mat = Mat::zeros(nrows, ncols);
        for col in 0..ncols {
            for row in 0..nrows {
                mat[(row, col)] = data[col * nrows + row];
            }
        }
        Ok(mat)
    }
}

impl Serialize for HwePcaModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("HwePcaModel", 8)?;
        state.serialize_field("n_samples", &self.n_samples)?;
        state.serialize_field("n_variants", &self.n_variants)?;
        state.serialize_field("scaler", &self.scaler)?;
        state.serialize_field("eigenvalues", &self.eigenvalues)?;
        state.serialize_field("singular_values", &self.singular_values)?;
        state.serialize_field(
            "sample_basis",
            &MatrixData::from_mat(self.sample_basis.as_ref()),
        )?;
        state.serialize_field(
            "sample_scores",
            &MatrixData::from_mat(self.sample_scores.as_ref()),
        )?;
        state.serialize_field("loadings", &MatrixData::from_mat(self.loadings.as_ref()))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HwePcaModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ModelData {
            n_samples: usize,
            n_variants: usize,
            scaler: HweScaler,
            eigenvalues: Vec<f64>,
            singular_values: Vec<f64>,
            sample_basis: MatrixData,
            sample_scores: MatrixData,
            loadings: MatrixData,
        }

        let raw = ModelData::deserialize(deserializer)?;
        Ok(HwePcaModel {
            n_samples: raw.n_samples,
            n_variants: raw.n_variants,
            scaler: raw.scaler,
            eigenvalues: raw.eigenvalues,
            singular_values: raw.singular_values,
            sample_basis: raw.sample_basis.into_mat().map_err(DeError::custom)?,
            sample_scores: raw.sample_scores.into_mat().map_err(DeError::custom)?,
            loadings: raw.loadings.into_mat().map_err(DeError::custom)?,
        })
    }
}
