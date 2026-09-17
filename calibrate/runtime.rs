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

/// Declares gam's GPU policy off when cudarc's own loader cannot open libcuda.
///
/// gnomon's release profile aborts on panic. gam's GPU probe creates a CUDA
/// context as its first action and turns cudarc's missing-libcuda panic into a
/// CPU fallback by catching the unwind, so under `panic = "abort"` the probe
/// aborted `gnomon-calibrate train` (SIGABRT from cudarc's `panic_no_lib_found`)
/// in the first joint Hessian of a binary fit on a node with no libcuda. Without
/// libcuda no GPU kernel can run, and gam resolves an off policy without
/// probing. gam keeps the first policy configured in a process, so this runs
/// before gam's first fit; a host with libcuda keeps gam's automatic policy.
fn declare_cpu_kernels_without_libcuda() {
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        gam::gpu::configure_global_policy(gam::gpu::GpuPolicy::Off);
    }
}

/// Runs `work` on calibrate's gam worker pool, sized like the global pool. The
/// first call also runs `gam::init_parallelism`, which registers gam's
/// higher-order LAML corrector and rho-posterior escalator: without them gam
/// declines both corrections, and a fit would not match gam's own CLI.
pub(crate) fn on_gam_pool<R: Send>(work: impl FnOnce() -> R + Send) -> Result<R, String> {
    let pool = GAM_POOL.get_or_init(|| {
        declare_cpu_kernels_without_libcuda();
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

#[cfg(test)]
mod tests {
    use super::on_gam_pool;

    #[test]
    fn gam_policy_is_off_where_libcuda_cannot_be_loaded() {
        on_gam_pool(|| ()).expect("calibrate's gam worker pool");
        let libcuda = unsafe { cudarc::driver::sys::is_culib_present() };
        assert!(
            libcuda || gam::gpu::global_policy() == gam::gpu::GpuPolicy::Off,
            "no libcuda, yet gam's GPU policy is {:?}",
            gam::gpu::global_policy()
        );
    }
}
