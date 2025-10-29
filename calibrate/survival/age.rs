use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Parameters controlling the guarded log-age transformation used throughout the
/// survival pipeline.  The transform maps chronological age to a log scale while
/// ensuring the argument to `log` never becomes zero or negative.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub a_min: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn fit(age_entry: &ArrayView1<'_, f64>, delta: f64) -> Self {
        let a_min = age_entry.iter().copied().fold(f64::INFINITY, f64::min);
        Self { a_min, delta }
    }

    pub fn transform_value(&self, age: f64) -> f64 {
        (age - self.a_min + self.delta).ln()
    }

    pub fn transform(&self, age: &Array1<f64>) -> Array1<f64> {
        age.map(|&v| self.transform_value(v))
    }

    pub fn derivative_factor(&self, age: f64) -> f64 {
        1.0 / (age - self.a_min + self.delta)
    }

    pub fn derivative_factors(&self, ages: &Array1<f64>) -> Array1<f64> {
        ages.map(|&v| self.derivative_factor(v))
    }
}

/// Helper encapsulating the guarded log-age transform and its derivative factors.
#[derive(Debug, Clone)]
pub struct GuardedLogAge {
    pub transform: AgeTransform,
    pub entry_log_age: Array1<f64>,
    pub exit_log_age: Array1<f64>,
    pub exit_derivative_scale: Array1<f64>,
}

impl GuardedLogAge {
    pub fn new(transform: AgeTransform, age_entry: &Array1<f64>, age_exit: &Array1<f64>) -> Self {
        let entry_log_age = transform.transform(age_entry);
        let exit_log_age = transform.transform(age_exit);
        let exit_derivative_scale = transform.derivative_factors(age_exit);
        Self {
            transform,
            entry_log_age,
            exit_log_age,
            exit_derivative_scale,
        }
    }
}
