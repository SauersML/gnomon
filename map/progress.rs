use std::fmt;

/// Stages reported during model fitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FitProgressStage {
    AlleleStatistics,
    GramMatrix,
    Loadings,
}

impl FitProgressStage {
    pub fn describe(self) -> &'static str {
        match self {
            Self::AlleleStatistics => "allele statistics",
            Self::GramMatrix => "Gram matrix accumulation",
            Self::Loadings => "variant loading computation",
        }
    }
}

impl fmt::Display for FitProgressStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.describe())
    }
}

/// Observer for reporting incremental progress while fitting a model.
pub trait FitProgressObserver {
    fn on_stage_start(&mut self, stage: FitProgressStage, total_variants: usize) {
        let _ = (stage, total_variants);
    }
    fn on_stage_advance(&mut self, stage: FitProgressStage, processed_variants: usize) {
        let _ = (stage, processed_variants);
    }
    fn on_stage_finish(&mut self, stage: FitProgressStage) {
        let _ = stage;
    }
}

#[derive(Default)]
pub struct NoopFitProgress;

impl FitProgressObserver for NoopFitProgress {}

/// Stages reported during projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectionProgressStage {
    Projection,
}

impl ProjectionProgressStage {
    pub fn describe(self) -> &'static str {
        match self {
            Self::Projection => "sample projection",
        }
    }
}

impl fmt::Display for ProjectionProgressStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.describe())
    }
}

/// Observer for reporting incremental progress during projection.
pub trait ProjectionProgressObserver {
    fn on_stage_start(&mut self, stage: ProjectionProgressStage, total_variants: usize) {
        let _ = (stage, total_variants);
    }
    fn on_stage_advance(&mut self, stage: ProjectionProgressStage, processed_variants: usize) {
        let _ = (stage, processed_variants);
    }
    fn on_stage_finish(&mut self, stage: ProjectionProgressStage) {
        let _ = stage;
    }
}

#[derive(Default)]
pub struct NoopProjectionProgress;

impl ProjectionProgressObserver for NoopProjectionProgress {}
