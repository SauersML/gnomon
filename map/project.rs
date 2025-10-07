use core::cmp::min;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut};
use std::error::Error;

use super::fit::{
    DEFAULT_BLOCK_WIDTH, DenseBlockSource, HwePcaError, HwePcaModel, HweScaler, VariantBlockSource,
};

pub struct HwePcaProjector<'model> {
    model: &'model HwePcaModel,
}

impl HwePcaModel {
    pub fn projector(&self) -> HwePcaProjector<'_> {
        HwePcaProjector { model: self }
    }

    pub fn project_dense(
        &self,
        data: &[f64],
        n_samples: usize,
        n_variants: usize,
    ) -> Result<Mat<f64>, HwePcaError> {
        let mut source = DenseBlockSource::new(data, n_samples, n_variants)?;
        self.projector().project(&mut source)
    }
}

impl<'model> HwePcaProjector<'model> {
    pub fn project<S>(&self, source: &mut S) -> Result<Mat<f64>, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let n_samples = source.n_samples();
        let mut scores = Mat::zeros(n_samples, self.model.components());
        self.project_into(source, scores.as_mut())?;
        Ok(scores)
    }

    pub fn project_into<S>(
        &self,
        source: &mut S,
        mut scores: MatMut<'_, f64>,
    ) -> Result<(), HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let n_samples = source.n_samples();
        let n_variants = source.n_variants();

        if n_samples == 0 {
            return Err(HwePcaError::InvalidInput(
                "Projection requires at least one sample",
            ));
        }
        if n_variants != self.model.n_variants() {
            return Err(HwePcaError::InvalidInput(
                "Projection variant dimension must match fitted model",
            ));
        }
        if scores.nrows() != n_samples {
            return Err(HwePcaError::InvalidInput(
                "Projection output row count mismatch",
            ));
        }
        if scores.ncols() != self.model.components() {
            return Err(HwePcaError::InvalidInput(
                "Projection output column count must equal number of components",
            ));
        }

        scores.fill(0.0);

        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;

        let block_capacity =
            projection_block_capacity(self.model.n_samples(), n_samples, n_variants);
        let elements = n_samples
            .checked_mul(block_capacity)
            .ok_or_else(|| HwePcaError::InvalidInput("Projection workspace size overflow"))?;
        let mut block_storage = vec![0.0f64; elements];
        let scaler = self.model.scaler();
        let loadings = self.model.variant_loadings();
        let mut processed = 0usize;

        loop {
            let filled = source
                .next_block_into(block_capacity, &mut block_storage)
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
            standardize_projection_block(scaler, block.as_mut(), processed, filled);

            let standardized = block.into_const();
            let loadings_block = loadings.submatrix(processed, 0, filled, self.model.components());

            matmul(
                scores.as_mut(),
                Accum::Add,
                standardized,
                loadings_block,
                1.0,
                faer::get_global_parallelism(),
            );

            processed += filled;
        }

        if processed != n_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early during projection",
            ));
        }

        Ok(())
    }

    pub fn model(&self) -> &'model HwePcaModel {
        self.model
    }
}

fn standardize_projection_block(
    scaler: &HweScaler,
    block: MatMut<'_, f64>,
    offset: usize,
    filled: usize,
) {
    scaler.standardize_block(block, offset..offset + filled);
}

fn projection_block_capacity(
    fitted_samples: usize,
    projected_samples: usize,
    n_variants: usize,
) -> usize {
    if n_variants == 0 {
        return 1;
    }
    let default = DEFAULT_BLOCK_WIDTH.max(1);
    let safe_reference = fitted_samples.checked_mul(default).unwrap_or(usize::MAX);
    let mut capacity = if projected_samples == 0 {
        default
    } else {
        safe_reference / projected_samples
    };
    if capacity == 0 {
        capacity = 1;
    }
    capacity = min(capacity, default);
    capacity = min(capacity, n_variants);
    capacity
}
