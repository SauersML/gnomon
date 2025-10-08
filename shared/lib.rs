#![feature(portable_simd)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

pub mod files;

pub mod shared {
    pub use super::files;
}

#[path = "../score/mod.rs"]
pub mod score;

pub use score::{batch, complex, decide, download, io, kernel, pipeline, prepare, reformat, types};

#[path = "../map/mod.rs"]
pub mod map;

#[path = "../calibrate/mod.rs"]
pub mod calibrate;
