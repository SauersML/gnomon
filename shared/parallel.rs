//! Centralized, idempotent initialization of Rayon's global thread pool.
//!
//! # Why this exists
//!
//! Rayon exposes a single process-wide *global* thread pool. It can be
//! configured explicitly exactly once via
//! [`rayon::ThreadPoolBuilder::build_global`]; a second call — or a call made
//! *after* the pool has already been brought up — returns
//! [`rayon::ThreadPoolBuildError`] (`GlobalPoolAlreadyInitialized`).
//!
//! Crucially, Rayon also initializes the global pool **lazily** the first time
//! any parallel primitive runs (`par_iter`, `rayon::scope`, `rayon::join`,
//! faer's rayon backend, etc.). So by the time an explicit `build_global()`
//! runs, the pool may already exist even though nobody called `build_global()`
//! before.
//!
//! `gnomon all` is the path that exposes this: it runs the `score`, `project`
//! and `terms` phases inside a *single* process. Phase 1 (VCF→PLINK conversion
//! via `convert_genome`) executes `into_par_iter()`, which lazily stands up the
//! global pool. The later `score` phase then calls `build_global()`, which now
//! fails because the pool already exists — and the original code `.expect(..)`d
//! that result, aborting the whole process with SIGABRT (-6).
//!
//! The fix is to funnel *every* global-pool initialization through a single
//! guarded helper that:
//!   * runs the explicit `build_global()` at most once (via [`Once`]), and
//!   * treats `GlobalPoolAlreadyInitialized` as success rather than a fatal
//!     error (the pool exists with sensible defaults, which is all we need).

use std::sync::Once;

static RAYON_GLOBAL_INIT: Once = Once::new();

/// Initialize Rayon's global thread pool exactly once, idempotently.
///
/// Safe to call from any phase (`score`, `project`, `terms`, …) and any number
/// of times. The first call attempts an explicit
/// [`rayon::ThreadPoolBuilder::build_global`] using all available cores. If the
/// global pool has *already* been initialized — whether by a prior call here or
/// lazily by an earlier `par_iter`/faer/etc. — that condition is swallowed:
/// the pool exists, which is exactly the desired end state.
///
/// This makes the multi-phase single-process path (`gnomon all`) safe: no
/// phase can abort the process by racing a second `build_global()`.
pub fn init_global_thread_pool() {
    RAYON_GLOBAL_INIT.call_once(|| {
        match rayon::ThreadPoolBuilder::new().build_global() {
            Ok(()) => {}
            Err(err) => {
                // The only expected error is that the global pool was already
                // brought up (e.g. lazily by a prior parallel op in another
                // phase). That is fine — the pool is live and usable. We log
                // for observability but do NOT abort: aborting here is exactly
                // the `gnomon all` SIGABRT bug this helper exists to prevent.
                eprintln!(
                    "> Rayon global thread pool already initialized; reusing existing pool ({err})."
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::init_global_thread_pool;
    use rayon::prelude::*;

    /// Reproduces the `gnomon all` multi-phase single-process abort scenario:
    /// an early phase (here: the VCF→PLINK conversion's `into_par_iter`)
    /// lazily brings up Rayon's global pool, and a LATER phase then tries to
    /// configure it explicitly. The historical code `.expect()`d
    /// `build_global()` and aborted with SIGABRT when the pool already existed.
    ///
    /// The whole point of `init_global_thread_pool` is that this must NOT
    /// panic, regardless of which phase touched the pool first.
    #[test]
    fn idempotent_init_after_lazy_pool_bringup_does_not_panic() {
        // Phase-1 stand-in: any parallel primitive lazily initializes the
        // global pool with rayon's defaults.
        let lazy_sum: u64 = (0u64..10_000).into_par_iter().sum();
        assert_eq!(lazy_sum, 49_995_000);

        // Phase-2/3/4 stand-in: explicit init AFTER the pool already exists.
        // Pre-fix this aborted the process; now it is a swallowed no-op.
        init_global_thread_pool();

        // And calling it repeatedly (every phase calls it) is still safe.
        init_global_thread_pool();
        init_global_thread_pool();

        // The pool remains fully usable afterwards.
        let post: u64 = (0u64..10_000).into_par_iter().sum();
        assert_eq!(post, 49_995_000);
    }
}
