#![deny(dead_code)]
#![deny(unused_imports)]
#![allow(non_snake_case)]

pub mod basis;
pub mod construction;
pub mod data;
pub mod survival_data;

pub mod calibrator;
pub mod diagnostics;
pub mod estimate;
pub mod faer_ndarray;
pub mod hmc;
pub mod hull;
pub mod joint;
pub mod matrix;
pub mod model;
pub mod pirls;
pub mod quadrature;
pub mod seeding;
pub mod survival;
pub mod types;
pub mod visualizer;

#[cfg(test)]
pub mod test_fixtures;
