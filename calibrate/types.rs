use ndarray::{Array1, ArrayView1};
use std::ops::{Deref, DerefMut};

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct Coefficients(pub Array1<f64>);

impl Coefficients {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }

    pub fn zeros(len: usize) -> Self {
        Self(Array1::zeros(len))
    }

    pub fn into_inner(self) -> Array1<f64> {
        self.0
    }

    pub fn as_view(&self) -> ArrayView1<'_, f64> {
        self.0.view()
    }
}

impl Deref for Coefficients {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Coefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Array1<f64>> for Coefficients {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<Coefficients> for Array1<f64> {
    fn from(values: Coefficients) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearPredictor(pub Array1<f64>);

impl LinearPredictor {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }

    pub fn zeros(len: usize) -> Self {
        Self(Array1::zeros(len))
    }

    pub fn into_inner(self) -> Array1<f64> {
        self.0
    }

    pub fn as_view(&self) -> ArrayView1<'_, f64> {
        self.0.view()
    }
}

impl Deref for LinearPredictor {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LinearPredictor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Array1<f64>> for LinearPredictor {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<LinearPredictor> for Array1<f64> {
    fn from(values: LinearPredictor) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct LogSmoothingParams(pub Array1<f64>);

impl LogSmoothingParams {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }

    pub fn exp(&self) -> Array1<f64> {
        self.0.mapv(f64::exp)
    }

    pub fn as_view(&self) -> ArrayView1<'_, f64> {
        self.0.view()
    }
}

impl Deref for LogSmoothingParams {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LogSmoothingParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Array1<f64>> for LogSmoothingParams {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<LogSmoothingParams> for Array1<f64> {
    fn from(values: LogSmoothingParams) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct LogSmoothingParamsView<'a>(pub ArrayView1<'a, f64>);

impl<'a> LogSmoothingParamsView<'a> {
    pub fn new(values: ArrayView1<'a, f64>) -> Self {
        Self(values)
    }

    pub fn exp(&self) -> Array1<f64> {
        self.0.mapv(f64::exp)
    }

    pub fn as_view(&self) -> ArrayView1<'a, f64> {
        self.0
    }
}

impl<'a> Deref for LogSmoothingParamsView<'a> {
    type Target = ArrayView1<'a, f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
