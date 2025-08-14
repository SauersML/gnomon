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
pub mod types;

// Add calibrate module
#[path = "../calibrate/lib.rs"]
pub mod calibrate;
