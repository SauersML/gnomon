#![deny(dead_code)]
#![deny(unused_imports)]
#![allow(non_snake_case)]

pub mod basis;
pub mod construction;
pub mod data;

#[cfg(feature = "survival-data")]
pub mod survival_data;

pub mod calibrator;
pub mod estimate;
pub mod faer_ndarray;
pub mod hull;
pub mod joint;
pub mod model;
pub mod matrix;
pub mod pirls;
pub mod survival;
pub mod quadrature;
pub mod hmc;
