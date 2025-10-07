#![feature(portable_simd)] // Do not remove this
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]
pub mod batch;
pub mod complex;
pub mod decide;
pub mod download;
pub mod io;
pub mod kernel;
pub mod pipeline;
pub mod prepare;
pub mod reformat;
#[path = "../shared/files.rs"]
pub mod shared_files;
pub mod types;
pub mod shared {
    pub use super::shared_files as files;
}

#[path = "../map/mod.rs"]
pub mod map;

// Add calibrate module
#[path = "../calibrate/lib.rs"]
pub mod calibrate;
