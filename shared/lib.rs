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
mod score;

pub use score::*;

#[path = "../map/lib.rs"]
pub mod map;

#[path = "../calibrate/lib.rs"]
pub mod calibrate;
