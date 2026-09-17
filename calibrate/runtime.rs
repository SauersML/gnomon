//! The worker pool every gam fit and prediction in calibrate runs on.
//!
//! gam needs Rayon workers with a wide stack and builds its global pool that way
//! in `gam::init_parallelism` (`RAYON_WORKER_STACK_SIZE`, 64 MiB). Its survival
//! marginal-slope row kernel evaluates each row's likelihood on fourth-order jet
//! towers held by value (341 f64 each) inside a parallel map, and Rayon inlines
//! that closure into every nested split frame. On Rayon's default 2 MiB worker
//! stack a 2,000-row survival fit on 12 threads reserved ~78 KB per frame and
//! aborted the process with a stack overflow 26 frames deep. The global pool can
//! be configured only once per process, and `gnomon all`, another subcommand or
//! any earlier parallel call may already have brought it up with default stacks,
//! so calibrate keeps its own pool with gam's worker stack.

use std::sync::OnceLock;

/// gam's worker stack (`RAYON_WORKER_STACK_SIZE` in gam's `src/lib.rs`).
const GAM_WORKER_STACK_BYTES: usize = 64 << 20;

static GAM_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

/// Runs `work` on calibrate's gam worker pool, sized like the global pool. The
/// first call also runs `gam::init_parallelism`, which registers gam's
/// higher-order LAML corrector and rho-posterior escalator: without them gam
/// declines both corrections, and a fit would not match gam's own CLI.
pub(crate) fn on_gam_pool<R: Send>(work: impl FnOnce() -> R + Send) -> Result<R, String> {
    let pool = GAM_POOL.get_or_init(|| {
        gam::init_parallelism();
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads())
            .stack_size(GAM_WORKER_STACK_BYTES)
            .thread_name(|index| format!("calibrate-gam-{index}"))
            .build()
            .map_err(|error| format!("could not build calibrate's gam worker pool: {error}"))
    });
    match pool {
        Ok(pool) => Ok(pool.install(work)),
        Err(error) => Err(error.clone()),
    }
}
