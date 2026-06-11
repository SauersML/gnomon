#![feature(portable_simd)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

// The CLI's score orchestrator (`score/main.rs`) refers to this crate by name
// (`gnomon::score::...`) because it is normally compiled into the binary, where
// `gnomon` is an external dependency. When the `python` feature pulls that same
// file into the library (so the in-process bindings can call
// `run_gnomon_with_args` directly), this self-alias makes those `gnomon::`
// paths resolve to the current crate.
#[cfg(feature = "python")]
extern crate self as gnomon;

/// Centralized, idempotent Rayon global thread-pool initialization. Used by
/// every Rayon-using phase (`score`, `project`, `terms`) so that the
/// multi-phase `gnomon all` driver cannot abort on a racing `build_global()`.
#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
pub mod parallel;

#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
pub mod files;

pub mod shared {
    #[cfg(any(feature = "score", feature = "map", feature = "terms"))]
    pub use super::files;
}

#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
pub mod adapt_plink2;

#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
#[path = "../score/mod.rs"]
pub mod score;

#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
#[path = "../terms/mod.rs"]
pub mod terms;

#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
pub mod batch {
    pub use crate::score::batch::*;
}

#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
pub use score::{complex, decide, download, io, kernel, pipeline, prepare, reformat, types};

#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
#[path = "../map/mod.rs"]
pub mod map;

#[cfg(feature = "calibrate")]
#[path = "../calibrate/mod.rs"]
pub mod calibrate;

// Pull the CLI score orchestrator into the library so the in-process Python
// bindings call the very same `run_gnomon_with_args` the `gnomon score`
// subcommand dispatches to (instead of shelling out to the binary).
#[cfg(feature = "python")]
#[path = "../score/main.rs"]
pub mod score_main;

#[cfg(feature = "python")]
pub mod python_ext;
