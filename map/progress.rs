use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::io::IsTerminal;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const PROGRESS_TICK_INTERVAL: Duration = Duration::from_millis(100);

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
pub trait FitProgressObserver: Send + Sync {
    fn on_stage_start(&self, stage: FitProgressStage, total_variants: usize) {
        let _ = (stage, total_variants);
    }
    fn on_stage_estimate(&self, stage: FitProgressStage, estimated_total: usize) {
        let _ = (stage, estimated_total);
    }
    fn on_stage_advance(&self, stage: FitProgressStage, processed_variants: usize) {
        let _ = (stage, processed_variants);
    }
    fn on_stage_total(&self, stage: FitProgressStage, total_variants: usize) {
        let _ = (stage, total_variants);
    }
    fn on_stage_finish(&self, stage: FitProgressStage) {
        let _ = stage;
    }
}

#[derive(Default)]
pub struct NoopFitProgress;

impl FitProgressObserver for NoopFitProgress {}

#[derive(Clone)]
pub struct StageProgressHandle<P>
where
    P: FitProgressObserver,
{
    observer: Arc<P>,
    stage: FitProgressStage,
}

impl<P> StageProgressHandle<P>
where
    P: FitProgressObserver,
{
    pub fn new(observer: Arc<P>, stage: FitProgressStage) -> Self {
        Self { observer, stage }
    }

    pub fn observer(&self) -> &Arc<P> {
        &self.observer
    }

    pub fn advance(&self, processed_variants: usize) {
        self.observer
            .on_stage_advance(self.stage, processed_variants);
    }

    pub fn estimate(&self, estimated_total: usize) {
        self.observer.on_stage_estimate(self.stage, estimated_total);
    }

    pub fn set_total(&self, total_variants: usize) {
        self.observer.on_stage_total(self.stage, total_variants);
    }

    pub fn finish(self) {
        self.observer.on_stage_finish(self.stage);
    }
}

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
pub trait ProjectionProgressObserver: Send + Sync {
    fn on_stage_start(&self, stage: ProjectionProgressStage, total_variants: usize) {
        let _ = (stage, total_variants);
    }
    fn on_stage_advance(&self, stage: ProjectionProgressStage, processed_variants: usize) {
        let _ = (stage, processed_variants);
    }
    fn on_stage_finish(&self, stage: ProjectionProgressStage) {
        let _ = stage;
    }
}

#[derive(Default)]
pub struct NoopProjectionProgress;

impl ProjectionProgressObserver for NoopProjectionProgress {}

fn progress_draw_target() -> ProgressDrawTarget {
    if std::io::stdout().is_terminal() {
        ProgressDrawTarget::stdout()
    } else {
        ProgressDrawTarget::hidden()
    }
}

fn determinate_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} {msg:<40} {percent:>3}% |{bar:40.cyan/blue}| {pos}/{len} [{elapsed_precise}<{eta_precise}]",
    )
    .expect("valid progress template")
    .progress_chars("=>-")
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.green} {msg:<60} {pos:>8} [{elapsed_precise}]")
        .expect("valid spinner template")
}

enum StageBarMode {
    Determinate { total: u64 },
    Spinner { approximate: Option<u64> },
}

struct ManagedStageBar {
    stage: FitProgressStage,
    bar: ProgressBar,
    mode: StageBarMode,
}

impl ManagedStageBar {
    fn new(stage: FitProgressStage, total: usize) -> Self {
        if total > 0 {
            let bar = ProgressBar::new(total as u64);
            bar.set_draw_target(progress_draw_target());
            bar.set_style(determinate_style());
            bar.set_message(ConsoleFitProgress::stage_message(stage));
            bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
            Self {
                stage,
                bar,
                mode: StageBarMode::Determinate {
                    total: total as u64,
                },
            }
        } else {
            let bar = ProgressBar::new_spinner();
            bar.set_draw_target(progress_draw_target());
            bar.set_style(spinner_style());
            bar.set_message(ConsoleFitProgress::stage_message(stage));
            bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
            Self {
                stage,
                bar,
                mode: StageBarMode::Spinner { approximate: None },
            }
        }
    }

    fn update(&self, processed_variants: usize) {
        match &self.mode {
            StageBarMode::Determinate { total } => {
                let capped = processed_variants.min(*total as usize) as u64;
                self.bar.set_position(capped);
            }
            StageBarMode::Spinner { .. } => {
                self.bar.set_position(processed_variants as u64);
            }
        }
    }

    fn set_estimate(&mut self, estimated_total: usize) {
        if let StageBarMode::Spinner { approximate } = &mut self.mode {
            *approximate = Some(estimated_total as u64);
            self.refresh_spinner_message(*approximate);
        }
    }

    fn refresh_spinner_message(&self, approximate: Option<u64>) {
        let base_message = ConsoleFitProgress::stage_message(self.stage);
        if let Some(approx) = approximate {
            self.bar
                .set_message(format!("{base_message} (≈{} variants)", approx));
        } else {
            self.bar.set_message(base_message);
        }
    }

    fn set_total(&mut self, new_total: usize) {
        if new_total == 0 {
            return;
        }
        let total = new_total as u64;
        let current = self.bar.position().min(total);
        self.bar.set_style(determinate_style());
        self.bar.set_length(total);
        self.bar
            .set_message(ConsoleFitProgress::stage_message(self.stage));
        self.bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
        self.bar.set_position(current);
        self.mode = StageBarMode::Determinate { total };
    }

    fn finish(self, message: &'static str) {
        self.bar.finish_with_message(message);
    }

    fn abandon(self, message: String) {
        self.bar.abandon_with_message(message);
    }
}

pub struct ConsoleFitProgress {
    inner: Mutex<HashMap<FitProgressStage, ManagedStageBar>>,
}

impl ConsoleFitProgress {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    pub fn stage_message(stage: FitProgressStage) -> &'static str {
        match stage {
            FitProgressStage::AlleleStatistics => "Estimating allele statistics",
            FitProgressStage::GramMatrix => "Accumulating Gram matrix",
            FitProgressStage::Loadings => "Computing variant loadings",
        }
    }

    pub fn stage_complete(stage: FitProgressStage) -> &'static str {
        match stage {
            FitProgressStage::AlleleStatistics => "Allele statistics complete",
            FitProgressStage::GramMatrix => "Gram matrix finalized",
            FitProgressStage::Loadings => "Variant loadings complete",
        }
    }
}

impl Default for ConsoleFitProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl FitProgressObserver for ConsoleFitProgress {
    fn on_stage_start(&self, stage: FitProgressStage, total_variants: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(existing) = inner.remove(&stage) {
            log::warn!(
                "restarting progress tracking for stage '{}'; previous progress abandoned",
                stage
            );
            existing.abandon(format!("{} (restarted)", Self::stage_message(stage)));
        }

        let bar = ManagedStageBar::new(stage, total_variants);
        inner.insert(stage, bar);
    }

    fn on_stage_estimate(&self, stage: FitProgressStage, estimated_total: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(existing) = inner.get_mut(&stage) {
            existing.set_estimate(estimated_total);
        }
    }

    fn on_stage_total(&self, stage: FitProgressStage, total_variants: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(existing) = inner.get_mut(&stage) {
            existing.set_total(total_variants);
        }
    }

    fn on_stage_advance(&self, stage: FitProgressStage, processed_variants: usize) {
        let inner = self.inner.lock().unwrap();
        if let Some(bar) = inner.get(&stage) {
            bar.update(processed_variants);
        } else {
            log::warn!(
                "received progress update for stage '{}' with no active progress bar",
                stage
            );
        }
    }

    fn on_stage_finish(&self, stage: FitProgressStage) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(bar) = inner.remove(&stage) {
            bar.finish(Self::stage_complete(stage));
        } else {
            log::warn!(
                "received completion for stage '{}' with no active progress bar",
                stage
            );
        }
    }
}

impl Drop for ConsoleFitProgress {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        for (stage, bar) in inner.drain() {
            bar.abandon(format!("{} (aborted)", Self::stage_message(stage)));
        }
    }
}

enum ProjectionStageBar {
    Determinate {
        total: u64,
        bar: ProgressBar,
    },
    Spinner {
        bar: ProgressBar,
    },
}

impl ProjectionStageBar {
    fn new(stage: ProjectionProgressStage, total: usize) -> Self {
        if total > 0 {
            let bar = ProgressBar::new(total as u64);
            bar.set_draw_target(progress_draw_target());
            bar.set_style(determinate_style());
            bar.set_message(ConsoleProjectionProgress::stage_message(stage));
            bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
            Self::Determinate {
                total: total as u64,
                bar,
            }
        } else {
            let bar = ProgressBar::new_spinner();
            bar.set_draw_target(progress_draw_target());
            bar.set_style(spinner_style());
            bar.set_message(ConsoleProjectionProgress::stage_message(stage));
            bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
            Self::Spinner { bar }
        }
    }

    fn update(&self, processed_variants: usize) {
        match self {
            ProjectionStageBar::Determinate { total, bar } => {
                let capped = processed_variants.min(*total as usize) as u64;
                bar.set_position(capped);
            }
            ProjectionStageBar::Spinner { bar, .. } => {
                bar.set_position(processed_variants as u64);
            }
        }
    }

    fn finish(self, message: &'static str) {
        match self {
            ProjectionStageBar::Determinate { bar, .. }
            | ProjectionStageBar::Spinner { bar, .. } => bar.finish_with_message(message),
        }
    }

    fn abandon(self, message: String) {
        match self {
            ProjectionStageBar::Determinate { bar, .. }
            | ProjectionStageBar::Spinner { bar, .. } => bar.abandon_with_message(message),
        }
    }
}

pub struct ConsoleProjectionProgress {
    inner: Mutex<Option<(ProjectionProgressStage, ProjectionStageBar)>>,
}

impl ConsoleProjectionProgress {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }

    fn stage_message(stage: ProjectionProgressStage) -> &'static str {
        match stage {
            ProjectionProgressStage::Projection => "Projecting samples",
        }
    }

    fn stage_complete(stage: ProjectionProgressStage) -> &'static str {
        match stage {
            ProjectionProgressStage::Projection => "Projection complete",
        }
    }
}

impl Default for ConsoleProjectionProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl ProjectionProgressObserver for ConsoleProjectionProgress {
    fn on_stage_start(&self, stage: ProjectionProgressStage, total_variants: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some((current_stage, bar)) = inner.take() {
            log::warn!(
                "starting new projection stage '{}' before finishing '{}'",
                stage,
                current_stage
            );
            bar.abandon(format!(
                "{} (interrupted)",
                Self::stage_message(current_stage)
            ));
        }

        let bar = ProjectionStageBar::new(stage, total_variants);
        inner.replace((stage, bar));
    }

    fn on_stage_advance(&self, stage: ProjectionProgressStage, processed_variants: usize) {
        let inner = self.inner.lock().unwrap();
        if let Some((current, bar)) = inner.as_ref() {
            if *current == stage {
                bar.update(processed_variants);
            } else {
                log::warn!(
                    "received progress for projection stage '{}' while '{}' is active",
                    stage,
                    current
                );
            }
        } else {
            log::warn!(
                "received projection progress for stage '{}' with no active progress bar",
                stage
            );
        }
    }

    fn on_stage_finish(&self, stage: ProjectionProgressStage) {
        let mut inner = self.inner.lock().unwrap();
        if let Some((current, bar)) = inner.take() {
            if current == stage {
                bar.finish(Self::stage_complete(stage));
            } else {
                log::warn!(
                    "received completion for projection stage '{}' while '{}' is active",
                    stage,
                    current
                );
                bar.abandon(format!(
                    "{} (completed out of order)",
                    Self::stage_message(current)
                ));
            }
        } else {
            log::warn!(
                "received completion for projection stage '{}' with no active progress bar",
                stage
            );
        }
    }
}

impl Drop for ConsoleProjectionProgress {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        if let Some((stage, bar)) = inner.take() {
            bar.abandon(format!("{} (aborted)", Self::stage_message(stage)));
        }
    }
}

pub fn fit_progress() -> Arc<ConsoleFitProgress> {
    Arc::new(ConsoleFitProgress::new())
}

pub fn projection_progress() -> Arc<ConsoleProjectionProgress> {
    Arc::new(ConsoleProjectionProgress::new())
}
