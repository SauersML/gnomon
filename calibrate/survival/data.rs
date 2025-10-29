use ndarray::{Array1, Array2, ArrayView1};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

use super::age::AgeTransform;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateViews<'a> {
    pub pgs: ArrayView1<'a, f64>,
    pub sex: ArrayView1<'a, f64>,
    pub pcs: ArrayView1<'a, f64>,
}

#[derive(Debug, Clone)]
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    pub age_transform: AgeTransform,
}

#[derive(Debug, Clone)]
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: CovariateViews<'a>,
}

#[derive(Error, Debug)]
pub enum SurvivalDataError {
    #[error("polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("column {0} missing")]
    MissingColumn(String),
    #[error("column {0} contains nulls")]
    Nulls(String),
    #[error("column {0} contains non-finite values")]
    NonFinite(String),
    #[error("age_entry must be strictly less than age_exit")]
    InvalidAgeOrder,
    #[error("event indicators must be mutually exclusive")]
    InvalidEvents,
}

fn load_columns(path: &str) -> Result<DataFrame, SurvivalDataError> {
    let df = CsvReader::new(File::open(Path::new(path))?)
        .with_options(
            CsvReadOptions::default()
                .with_has_header(true)
                .with_parse_options(CsvParseOptions::default().with_separator(b'\t')),
        )
        .finish()?;
    Ok(df)
}

fn check_column<T>(df: &DataFrame, name: &str) -> Result<Series, SurvivalDataError> {
    if !df.get_column_names().iter().any(|c| c == name) {
        return Err(SurvivalDataError::MissingColumn(name.to_string()));
    }
    let series = df.column(name)?.clone();
    if series.null_count() > 0 {
        return Err(SurvivalDataError::Nulls(name.to_string()));
    }
    Ok(series)
}

fn series_to_array(series: Series) -> Result<Array1<f64>, SurvivalDataError> {
    let chunked = series
        .f64()
        .ok_or_else(|| SurvivalDataError::NonFinite(series.name().to_string()))?;
    let vals: Vec<f64> = chunked
        .into_no_null_iter()
        .map(|v| {
            if !v.is_finite() {
                Err(SurvivalDataError::NonFinite(series.name().to_string()))
            } else {
                Ok(v)
            }
        })
        .collect::<Result<_, _>>()?;
    Ok(Array1::from_vec(vals))
}

fn series_to_u8(series: Series) -> Result<Array1<u8>, SurvivalDataError> {
    let chunked = series
        .u8()
        .ok_or_else(|| SurvivalDataError::NonFinite(series.name().to_string()))?;
    let vals: Vec<u8> = chunked.into_no_null_iter().collect();
    Ok(Array1::from_vec(vals))
}

pub fn load_survival_training_data(
    path: &str,
    num_pcs: usize,
) -> Result<SurvivalTrainingData, SurvivalDataError> {
    let df = load_columns(path)?;
    let required: HashSet<&str> = [
        "age_entry",
        "age_exit",
        "event_target",
        "event_competing",
        "score",
        "sex",
    ]
    .into_iter()
    .collect();

    for col in &required {
        if !df.get_column_names().iter().any(|c| c == *col) {
            return Err(SurvivalDataError::MissingColumn(col.to_string()));
        }
    }

    let age_entry = series_to_array(check_column(&df, "age_entry")?)?;
    let age_exit = series_to_array(check_column(&df, "age_exit")?)?;
    if !age_entry
        .iter()
        .zip(age_exit.iter())
        .all(|(&a0, &a1)| a0.is_finite() && a1.is_finite() && a0 < a1)
    {
        return Err(SurvivalDataError::InvalidAgeOrder);
    }

    let event_target = series_to_u8(check_column(&df, "event_target")?)?;
    let event_competing = series_to_u8(check_column(&df, "event_competing")?)?;
    if !event_target
        .iter()
        .zip(event_competing.iter())
        .all(|(&e_t, &e_c)| (e_t == 0 || e_t == 1) && (e_c == 0 || e_c == 1) && (e_t + e_c <= 1))
    {
        return Err(SurvivalDataError::InvalidEvents);
    }

    let sample_weight = if df.get_column_names().iter().any(|c| c == "sample_weight") {
        series_to_array(check_column(&df, "sample_weight")?)?
    } else {
        Array1::ones(age_entry.len())
    };

    let pgs = series_to_array(check_column(&df, "score")?)?;
    let sex = series_to_array(check_column(&df, "sex")?)?;

    let mut pcs_matrix = Array2::zeros((age_entry.len(), num_pcs));
    for i in 0..num_pcs {
        let name = format!("pc{}", i + 1);
        let column = series_to_array(check_column(&df, &name)?)?;
        pcs_matrix.column_mut(i).assign(&column);
    }

    let age_transform = AgeTransform::fit(&age_entry.view(), 0.1);

    Ok(SurvivalTrainingData {
        age_entry,
        age_exit,
        event_target,
        event_competing,
        sample_weight,
        pgs,
        sex,
        pcs: pcs_matrix,
        age_transform,
    })
}
