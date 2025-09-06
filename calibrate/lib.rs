#![deny(dead_code)]
#![deny(unused_imports)]
#![allow(non_snake_case)]

pub mod basis;
pub mod construction;
pub mod data;

pub mod estimate;
pub mod hull;
pub mod model;
pub mod pirls;
pub mod calibrator;
// No global functions here; settings are scoped to owning modules.
