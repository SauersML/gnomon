use indicatif::{HumanBytes, ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::io::IsTerminal;
use std::mem;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const PROGRESS_TICK_INTERVAL: Duration = Duration::from_millis(100);

/// Minimum interval between CI log lines for the same stage.
const CI_LOG_INTERVAL: Duration = Duration::from_secs(10);
/// Minimum percentage increase before emitting a CI log line.
const CI_LOG_PERCENT_THRESHOLD: f64 = 5.0;

/// Output mode determines how progress is reported based on the runtime environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputMode {
    /// Interactive terminal with ANSI support - use animated progress bars.
    Terminal,
    /// CI environment (GitHub Actions, etc.) - periodic log lines without ANSI.
    Ci,
    /// Non-interactive output (piped, redirected) - minimal periodic logs.
    Quiet,
}

impl OutputMode {
    /// Detect the appropriate output mode based on environment.
    pub fn detect() -> Self {
        // Check for terminal first
        if std::io::stdout().is_terminal() {
            return Self::Terminal;
        }

        // Check for CI environments
        if env::var("CI").is_ok()
            || env::var("GITHUB_ACTIONS").is_ok()
            || env::var("GITLAB_CI").is_ok()
            || env::var("JENKINS_URL").is_ok()
            || env::var("BUILDKITE").is_ok()
        {
            return Self::Ci;
        }

        // Check for Jupyter/notebook environments - treat as CI for now
        // (periodic log lines work well in notebook output cells)
        if env::var("JPY_PARENT_PID").is_ok() || env::var("JUPYTER_RUNTIME_DIR").is_ok() {
            return Self::Ci;
        }

        // Default to quiet mode for piped/redirected output
        Self::Quiet
    }
}

/// Stages reported during model fitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FitProgressStage {
    AlleleStatistics,
    LdWeights,
    GramMatrix,
    Loadings,
}

impl FitProgressStage {
    pub fn describe(self) -> &'static str {
        match self {
            Self::AlleleStatistics => "allele statistics",
            Self::LdWeights => "LD weight computation",
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
    fn on_stage_finish(&self, _: FitProgressStage) {}
    fn on_stage_bytes(
        &self,
        stage: FitProgressStage,
        processed_bytes: u64,
        total_bytes: Option<u64>,
    ) {
        let _ = (stage, processed_bytes, total_bytes);
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

    pub fn advance_bytes(&self, processed_bytes: u64, total_bytes: Option<u64>) {
        self.observer
            .on_stage_bytes(self.stage, processed_bytes, total_bytes);
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
    fn on_stage_total(&self, stage: ProjectionProgressStage, total_variants: usize) {
        let _ = (stage, total_variants);
    }
    fn on_stage_finish(&self, _: ProjectionProgressStage) {}
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StageUnits {
    Variants,
    Bytes,
}

struct ManagedStageBar {
    stage: FitProgressStage,
    bar: ProgressBar,
    mode: StageBarMode,
    units: StageUnits,
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
                units: StageUnits::Variants,
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
                units: StageUnits::Variants,
            }
        }
    }

    fn update(&self, processed_variants: usize) {
        assert!(self.units == StageUnits::Variants);
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
        if self.units != StageUnits::Variants {
            return;
        }
        let new_value = match &mut self.mode {
            StageBarMode::Spinner { approximate } => {
                *approximate = Some(estimated_total as u64);
                *approximate
            }
            StageBarMode::Determinate { .. } => return,
        };
        self.refresh_spinner_message(new_value);
    }

    fn refresh_spinner_message(&self, approximate: Option<u64>) {
        if self.units != StageUnits::Variants {
            return;
        }
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
        self.units = StageUnits::Variants;
    }

    fn update_bytes(&mut self, processed_bytes: u64, total_bytes: Option<u64>) {
        self.units = StageUnits::Bytes;

        match (total_bytes, &mut self.mode) {
            (Some(total), StageBarMode::Determinate { total: existing }) => {
                if *existing != total {
                    self.bar.set_length(total);
                    *existing = total;
                }
                self.bar.set_style(determinate_style());
            }
            (Some(total), _) => {
                self.bar.set_style(determinate_style());
                self.bar.set_length(total);
                self.bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
                self.mode = StageBarMode::Determinate { total };
            }
            (None, StageBarMode::Spinner { approximate }) => {
                if approximate.is_some() {
                    *approximate = None;
                }
                self.bar.set_style(spinner_style());
            }
            (None, _) => {
                self.bar.set_style(spinner_style());
                self.bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
                self.mode = StageBarMode::Spinner { approximate: None };
            }
        }

        let position = match (&self.mode, total_bytes) {
            (StageBarMode::Determinate { total }, _) => processed_bytes.min(*total),
            (StageBarMode::Spinner { .. }, _) => processed_bytes,
        };
        self.bar.set_position(position);

        let base_message = ConsoleFitProgress::stage_message(self.stage);
        let message = match total_bytes {
            Some(total) => format!(
                "{base_message} ({} / {} read)",
                HumanBytes(processed_bytes),
                HumanBytes(total)
            ),
            None => format!("{base_message} ({} read)", HumanBytes(processed_bytes)),
        };
        self.bar.set_message(message);
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
            FitProgressStage::LdWeights => "Computing LD weights",
            FitProgressStage::GramMatrix => "Accumulating Gram matrix",
            FitProgressStage::Loadings => "Computing variant loadings",
        }
    }

    pub fn stage_complete(stage: FitProgressStage) -> &'static str {
        match stage {
            FitProgressStage::AlleleStatistics => "Allele statistics complete",
            FitProgressStage::LdWeights => "LD weights computed",
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

    fn on_stage_bytes(
        &self,
        stage: FitProgressStage,
        processed_bytes: u64,
        total_bytes: Option<u64>,
    ) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(bar) = inner.get_mut(&stage) {
            bar.update_bytes(processed_bytes, total_bytes);
        } else {
            log::warn!(
                "received byte progress update for stage '{}' with no active progress bar",
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

/// State for tracking when to emit the next log line for a stage.
struct CiStageState {
    total: Option<usize>,
    processed: usize,
    last_log_time: Instant,
    last_log_percent: f64,
    started: Instant,
}

impl CiStageState {
    fn new(total: usize) -> Self {
        Self {
            total: if total > 0 { Some(total) } else { None },
            processed: 0,
            last_log_time: Instant::now(),
            last_log_percent: 0.0,
            started: Instant::now(),
        }
    }

    fn should_log(&self) -> bool {
        let elapsed = self.last_log_time.elapsed();
        if elapsed >= CI_LOG_INTERVAL {
            return true;
        }

        if let Some(total) = self.total
            && total > 0 {
                let current_percent = (self.processed as f64 / total as f64) * 100.0;
                if current_percent - self.last_log_percent >= CI_LOG_PERCENT_THRESHOLD {
                    return true;
                }
            }

        false
    }

    fn mark_logged(&mut self) {
        self.last_log_time = Instant::now();
        if let Some(total) = self.total
            && total > 0 {
                self.last_log_percent = (self.processed as f64 / total as f64) * 100.0;
            }
    }

    fn format_progress(&self, stage: FitProgressStage) -> String {
        let stage_name = match stage {
            FitProgressStage::AlleleStatistics => "Estimating allele statistics",
            FitProgressStage::LdWeights => "Computing LD weights",
            FitProgressStage::GramMatrix => "Accumulating Gram matrix",
            FitProgressStage::Loadings => "Computing variant loadings",
        };

        if let Some(total) = self.total {
            let percent = if total > 0 {
                (self.processed as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            format!(
                "[PCA] {}... {:.0}% ({} / {} variants)",
                stage_name, percent, self.processed, total
            )
        } else {
            format!("[PCA] {}... {} variants", stage_name, self.processed)
        }
    }

    fn format_complete(&self, stage: FitProgressStage) -> String {
        let stage_name = match stage {
            FitProgressStage::AlleleStatistics => "Allele statistics complete",
            FitProgressStage::LdWeights => "LD weights computed",
            FitProgressStage::GramMatrix => "Gram matrix finalized",
            FitProgressStage::Loadings => "Variant loadings complete",
        };

        let elapsed = self.started.elapsed();
        let secs = elapsed.as_secs_f64();
        if let Some(total) = self.total {
            format!("[PCA] {} ({} variants, {:.1}s)", stage_name, total, secs)
        } else {
            format!(
                "[PCA] {} ({} variants, {:.1}s)",
                stage_name, self.processed, secs
            )
        }
    }
}

/// Progress observer that emits periodic log lines for CI environments.
///
/// Throttles output to avoid spam: only emits a log line when at least 5%
/// more progress has been made OR at least 10 seconds have elapsed since
/// the last log line.
pub struct CiFitProgress {
    inner: Mutex<HashMap<FitProgressStage, CiStageState>>,
}

impl CiFitProgress {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for CiFitProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl FitProgressObserver for CiFitProgress {
    fn on_stage_start(&self, stage: FitProgressStage, total_variants: usize) {
        let mut inner = self.inner.lock().unwrap();
        inner.insert(stage, CiStageState::new(total_variants));

        // Emit initial log line
        let stage_name = match stage {
            FitProgressStage::AlleleStatistics => "Estimating allele statistics",
            FitProgressStage::LdWeights => "Computing LD weights",
            FitProgressStage::GramMatrix => "Accumulating Gram matrix",
            FitProgressStage::Loadings => "Computing variant loadings",
        };
        if total_variants > 0 {
            println!("[PCA] {}... (0 / {} variants)", stage_name, total_variants);
        } else {
            println!("[PCA] {}...", stage_name);
        }
    }

    fn on_stage_estimate(&self, stage: FitProgressStage, estimated_total: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.get_mut(&stage)
            && state.total.is_none() && estimated_total > 0 {
                state.total = Some(estimated_total);
            }
    }

    fn on_stage_total(&self, stage: FitProgressStage, total_variants: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.get_mut(&stage) {
            state.total = Some(total_variants);
        }
    }

    fn on_stage_advance(&self, stage: FitProgressStage, processed_variants: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.get_mut(&stage) {
            state.processed = processed_variants;
            if state.should_log() {
                println!("{}", state.format_progress(stage));
                state.mark_logged();
            }
        }
    }

    fn on_stage_bytes(
        &self,
        stage: FitProgressStage,
        processed_bytes: u64,
        total_bytes: Option<u64>,
    ) {
        // For CI mode, we don't track byte-level progress separately
        // Just update based on bytes if we have a total
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.get_mut(&stage)
            && let Some(total) = total_bytes
                && total > 0 {
                    // Approximate variant progress from bytes
                    let approx_variants = if let Some(variant_total) = state.total {
                        ((processed_bytes as f64 / total as f64) * variant_total as f64) as usize
                    } else {
                        processed_bytes as usize
                    };
                    state.processed = approx_variants;
                    if state.should_log() {
                        let msg = format!(
                            "[PCA] {} ({} / {} read)",
                            ConsoleFitProgress::stage_message(stage),
                            HumanBytes(processed_bytes),
                            HumanBytes(total)
                        );
                        println!("{}", msg);
                        state.mark_logged();
                    }
                }
    }

    fn on_stage_finish(&self, stage: FitProgressStage) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(state) = inner.remove(&stage) {
            println!("{}", state.format_complete(stage));
        }
    }
}

enum ProjectionStageBar {
    Determinate {
        total: u64,
        bar: ProgressBar,
        processed: u64,
    },
    Spinner {
        bar: ProgressBar,
        processed: u64,
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
                processed: 0,
            }
        } else {
            let bar = ProgressBar::new_spinner();
            bar.set_draw_target(progress_draw_target());
            bar.set_style(spinner_style());
            bar.set_message(ConsoleProjectionProgress::stage_message(stage));
            bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
            Self::Spinner { bar, processed: 0 }
        }
    }

    fn update(&mut self, processed_variants: usize) {
        match self {
            ProjectionStageBar::Determinate {
                total,
                bar,
                processed,
            } => {
                let capped = processed_variants.min(*total as usize) as u64;
                *processed = capped;
                bar.set_position(capped);
            }
            ProjectionStageBar::Spinner { bar, processed } => {
                *processed = processed_variants as u64;
                bar.set_position(processed_variants as u64);
            }
        }
    }

    fn set_total(&mut self, stage: ProjectionProgressStage, total: usize) {
        if total == 0 {
            return;
        }
        let total_u64 = total as u64;
        match self {
            ProjectionStageBar::Determinate {
                total: current_total,
                bar,
                processed,
            } => {
                if *current_total != total_u64 {
                    *current_total = total_u64;
                    bar.set_length(total_u64);
                }
                let capped = (*processed).min(total_u64);
                *processed = capped;
                bar.set_style(determinate_style());
                bar.set_message(ConsoleProjectionProgress::stage_message(stage));
                bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
                bar.set_position(capped);
            }
            ProjectionStageBar::Spinner { bar, processed } => {
                let processed_value = (*processed).min(total_u64);
                let bar = mem::replace(bar, ProgressBar::hidden());
                bar.set_style(determinate_style());
                bar.set_length(total_u64);
                bar.set_message(ConsoleProjectionProgress::stage_message(stage));
                bar.enable_steady_tick(PROGRESS_TICK_INTERVAL);
                bar.set_position(processed_value);
                *self = ProjectionStageBar::Determinate {
                    total: total_u64,
                    bar,
                    processed: processed_value,
                };
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
        let mut inner = self.inner.lock().unwrap();
        if let Some((current, bar)) = inner.as_mut() {
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

    fn on_stage_total(&self, stage: ProjectionProgressStage, total_variants: usize) {
        let mut inner = self.inner.lock().unwrap();
        if let Some((current, bar)) = inner.as_mut() {
            if *current == stage {
                bar.set_total(stage, total_variants);
            } else {
                log::warn!(
                    "received total update for projection stage '{}' while '{}' is active",
                    stage,
                    current
                );
            }
        } else {
            log::warn!(
                "received total update for projection stage '{}' with no active progress bar",
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

/// Adaptive progress observer that automatically selects the appropriate output
/// mode based on the runtime environment.
pub enum AdaptiveFitProgress {
    Terminal(ConsoleFitProgress),
    Ci(CiFitProgress),
}

impl FitProgressObserver for AdaptiveFitProgress {
    fn on_stage_start(&self, stage: FitProgressStage, total_variants: usize) {
        match self {
            Self::Terminal(p) => p.on_stage_start(stage, total_variants),
            Self::Ci(p) => p.on_stage_start(stage, total_variants),
        }
    }

    fn on_stage_estimate(&self, stage: FitProgressStage, estimated_total: usize) {
        match self {
            Self::Terminal(p) => p.on_stage_estimate(stage, estimated_total),
            Self::Ci(p) => p.on_stage_estimate(stage, estimated_total),
        }
    }

    fn on_stage_advance(&self, stage: FitProgressStage, processed_variants: usize) {
        match self {
            Self::Terminal(p) => p.on_stage_advance(stage, processed_variants),
            Self::Ci(p) => p.on_stage_advance(stage, processed_variants),
        }
    }

    fn on_stage_total(&self, stage: FitProgressStage, total_variants: usize) {
        match self {
            Self::Terminal(p) => p.on_stage_total(stage, total_variants),
            Self::Ci(p) => p.on_stage_total(stage, total_variants),
        }
    }

    fn on_stage_finish(&self, stage: FitProgressStage) {
        match self {
            Self::Terminal(p) => p.on_stage_finish(stage),
            Self::Ci(p) => p.on_stage_finish(stage),
        }
    }

    fn on_stage_bytes(
        &self,
        stage: FitProgressStage,
        processed_bytes: u64,
        total_bytes: Option<u64>,
    ) {
        match self {
            Self::Terminal(p) => p.on_stage_bytes(stage, processed_bytes, total_bytes),
            Self::Ci(p) => p.on_stage_bytes(stage, processed_bytes, total_bytes),
        }
    }
}

/// Create a fit progress observer appropriate for the current environment.
///
/// - In terminals: Returns animated progress bars via `indicatif`
/// - In CI/GHA: Returns periodic log lines (throttled to ~20 per stage)
/// - In quiet mode: Returns minimal periodic output
pub fn fit_progress() -> Arc<AdaptiveFitProgress> {
    match OutputMode::detect() {
        OutputMode::Terminal => Arc::new(AdaptiveFitProgress::Terminal(ConsoleFitProgress::new())),
        OutputMode::Ci => Arc::new(AdaptiveFitProgress::Ci(CiFitProgress::new())),
        OutputMode::Quiet => Arc::new(AdaptiveFitProgress::Ci(CiFitProgress::new())), // Same as CI for now
    }
}

pub fn projection_progress() -> Arc<ConsoleProjectionProgress> {
    Arc::new(ConsoleProjectionProgress::new())
}
