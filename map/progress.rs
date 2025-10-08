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
    fn on_stage_start(&mut self, _stage: FitProgressStage, _total_variants: usize) {}
    fn on_stage_advance(&mut self, _stage: FitProgressStage, _processed_variants: usize) {}
    fn on_stage_finish(&mut self, _stage: FitProgressStage) {}
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
    fn on_stage_start(&mut self, _stage: ProjectionProgressStage, _total_variants: usize) {}
    fn on_stage_advance(&mut self, _stage: ProjectionProgressStage, _processed_variants: usize) {}
    fn on_stage_finish(&mut self, _stage: ProjectionProgressStage) {}
}

#[derive(Default)]
pub struct NoopProjectionProgress;

impl ProjectionProgressObserver for NoopProjectionProgress {}
