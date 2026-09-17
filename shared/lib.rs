#![feature(portable_simd)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

// The score orchestrator (`score/main.rs`) refers to this crate by name
// (`gnomon::score::...`), as the command-line programs do. This self-alias
// makes those paths resolve to the current crate when the file is compiled
// here, so one copy serves the binaries and the Python extension alike.
extern crate self as gnomon;

/// Centralized, idempotent Rayon global thread-pool initialization. Used by
/// every Rayon-using phase (`score`, `project`, `terms`) so that the
/// multi-phase `gnomon all` driver cannot abort on a racing `build_global()`.
pub mod parallel;

pub mod pipeline_error;

/// The stderr logger the command-line programs install, so library warnings print.
pub mod logging;

pub(crate) mod genotype_table;

pub mod files;
pub mod bcf_genotypes;
pub mod variant_header;

/// Atomic publication of results and caches, so a concurrent
/// reader never observes a partially written file.
pub mod output;

pub mod memory;

/// CPUs this process can run on: affinity, cgroup quota and NUMA nodes.
pub mod cpu;

mod range_fetch;

pub(crate) mod cuda_utils;

pub mod shared {
    pub use super::files;
}

pub mod adapt_plink2;

#[path = "../score/mod.rs"]
pub mod score;

#[path = "../terms/mod.rs"]
pub mod terms;

pub mod batch {
    pub use crate::score::batch::*;
}

pub use score::{complex, decide, download, io, pipeline, prepare, reformat, types};

#[path = "../map/mod.rs"]
pub mod map;

#[path = "../calibrate/mod.rs"]
pub mod calibrate;

/// The `gnomon score` orchestrator, compiled here once: the command-line
/// programs and the in-process Python bindings both call its
/// `run_gnomon_with_args`.
#[path = "../score/main.rs"]
pub mod score_main;
