#![deny(dead_code)]
#![deny(unused_imports)]
#![allow(non_snake_case)]
// Numerical computing often has complex types and many parameters - these are intentional
#![allow(clippy::type_complexity)]
#![allow(clippy::result_large_err)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::manual_clamp)]

pub mod basis;
pub mod construction;
pub mod data;

#[cfg(feature = "survival-data")]
pub mod survival_data;

pub mod calibrator;
pub mod estimate;
pub mod faer_ndarray;
pub mod hull;
pub mod model;
pub mod matrix;
pub mod pirls;
pub mod survival;
pub mod quadrature;
pub mod hmc;
