#![feature(portable_simd)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

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
