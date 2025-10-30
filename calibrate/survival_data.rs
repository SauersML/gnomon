use crate::calibrate::survival::{
    AgeTransform, SurvivalError, SurvivalPredictionInputs, SurvivalTrainingData,
    validate_survival_inputs,
};
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// Errors surfaced while reading or validating survival datasets.
#[derive(Debug, Error)]
pub enum SurvivalDataError {
    #[error("Error from the underlying Polars library: {0}")]
    Polars(#[from] PolarsError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("The required column '{0}' was not found in the input file.")]
    ColumnNotFound(String),
    #[error(
        "Column '{column_name}' could not be converted to the expected type '{expected_type}'. (Found type: {found_type})"
    )]
    ColumnWrongType {
        column_name: String,
        expected_type: &'static str,
        found_type: String,
    },
    #[error("Missing or null values were found in the column '{0}'.")]
    MissingValues(String),
    #[error(
        "Column '{column_name}' has {found} rows but {expected} were expected based on age_entry."
    )]
    LengthMismatch {
        column_name: String,
        expected: usize,
        found: usize,
    },
    #[error("Validation error: {0}")]
    Validation(#[from] SurvivalError),
}

/// Bundle containing frequency-weighted survival training data and the cached age transform.
#[derive(Debug)]
pub struct SurvivalTrainingBundle {
    pub data: SurvivalTrainingData,
    pub age_transform: AgeTransform,
}

/// Owned arrays backing `SurvivalPredictionInputs` alongside the raw covariates.
#[derive(Debug)]
pub struct SurvivalPredictionData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    pub covariates: Array2<f64>,
}

impl SurvivalPredictionData {
    /// Borrow the owned arrays as `SurvivalPredictionInputs` suitable for scoring.
    pub fn as_inputs(&self) -> SurvivalPredictionInputs<'_> {
        SurvivalPredictionInputs {
            age_entry: self.age_entry.view(),
            age_exit: self.age_exit.view(),
            event_target: self.event_target.view(),
            event_competing: self.event_competing.view(),
            sample_weight: self.sample_weight.view(),
            covariates: self.covariates.view(),
        }
    }
}

/// Load survival training data from a TSV or Parquet file, validating and caching the age transform.
pub fn load_survival_training_data(
    path: &str,
    num_pcs: usize,
    guard_delta: f64,
) -> Result<SurvivalTrainingBundle, SurvivalDataError> {
    let arrays = read_survival_arrays(path, num_pcs)?;

    validate_survival_inputs(
        arrays.age_entry.view(),
        arrays.age_exit.view(),
        arrays.event_target.view(),
        arrays.event_competing.view(),
        arrays.sample_weight.view(),
        arrays.pgs.view(),
        arrays.sex.view(),
        arrays.pcs.view(),
    )?;

    let age_transform = AgeTransform::from_training(&arrays.age_entry, guard_delta)?;

    Ok(SurvivalTrainingBundle {
        data: SurvivalTrainingData {
            age_entry: arrays.age_entry,
            age_exit: arrays.age_exit,
            event_target: arrays.event_target,
            event_competing: arrays.event_competing,
            sample_weight: arrays.sample_weight,
            pgs: arrays.pgs,
            sex: arrays.sex,
            pcs: arrays.pcs,
        },
        age_transform,
    })
}

/// Load survival prediction data from a TSV or Parquet file, validating the arrays and
/// preparing owned covariate storage.
pub fn load_survival_prediction_data(
    path: &str,
    num_pcs: usize,
) -> Result<SurvivalPredictionData, SurvivalDataError> {
    let arrays = read_survival_arrays(path, num_pcs)?;

    validate_survival_inputs(
        arrays.age_entry.view(),
        arrays.age_exit.view(),
        arrays.event_target.view(),
        arrays.event_competing.view(),
        arrays.sample_weight.view(),
        arrays.pgs.view(),
        arrays.sex.view(),
        arrays.pcs.view(),
    )?;

    let covariates = assemble_covariate_matrix(&arrays.pgs, &arrays.sex, &arrays.pcs);

    Ok(SurvivalPredictionData {
        age_entry: arrays.age_entry,
        age_exit: arrays.age_exit,
        event_target: arrays.event_target,
        event_competing: arrays.event_competing,
        sample_weight: arrays.sample_weight,
        pgs: arrays.pgs,
        sex: arrays.sex,
        pcs: arrays.pcs,
        covariates,
    })
}

#[derive(Debug)]
struct SurvivalArrays {
    age_entry: Array1<f64>,
    age_exit: Array1<f64>,
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    sample_weight: Array1<f64>,
    pgs: Array1<f64>,
    sex: Array1<f64>,
    pcs: Array2<f64>,
}

fn read_survival_arrays(path: &str, num_pcs: usize) -> Result<SurvivalArrays, SurvivalDataError> {
    let df = read_tabular(path)?;
    let name_map = build_case_insensitive_map(df.get_column_names());

    let age_entry = extract_f64_column(&df, &name_map, "age_entry")?;
    let age_exit = extract_f64_column(&df, &name_map, "age_exit")?;
    let event_target = extract_u8_column(&df, &name_map, "event_target")?;
    let event_competing = extract_u8_column(&df, &name_map, "event_competing")?;
    let pgs = extract_f64_column(&df, &name_map, "pgs")?;
    let sex = extract_f64_column(&df, &name_map, "sex")?;

    let n = age_entry.len();
    let sample_weight = if let Some(name) = name_map.get("sample_weight") {
        let weights = extract_f64_column(&df, &name_map, "sample_weight")?;
        if weights.len() != n {
            return Err(SurvivalDataError::LengthMismatch {
                column_name: name.clone(),
                expected: n,
                found: weights.len(),
            });
        }
        weights
    } else {
        Array1::from_elem(n, 1.0)
    };

    let mut pc_columns = Vec::with_capacity(num_pcs);
    for idx in 0..num_pcs {
        let key = format!("pc{}", idx + 1);
        let values = extract_f64_column(&df, &name_map, &key)?;
        if values.len() != n {
            let actual = name_map
                .get(&key.to_lowercase())
                .cloned()
                .unwrap_or(key.clone());
            return Err(SurvivalDataError::LengthMismatch {
                column_name: actual,
                expected: n,
                found: values.len(),
            });
        }
        pc_columns.push(values);
    }

    let mut pcs = Array2::<f64>::zeros((n, num_pcs));
    for (j, column) in pc_columns.into_iter().enumerate() {
        pcs.column_mut(j).assign(&column);
    }

    Ok(SurvivalArrays {
        age_entry,
        age_exit,
        event_target,
        event_competing,
        sample_weight,
        pgs,
        sex,
        pcs,
    })
}

fn read_tabular(path: &str) -> Result<DataFrame, SurvivalDataError> {
    let path = Path::new(path);
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .as_deref()
    {
        Some("parquet") | Some("pq") => {
            let file = File::open(path)?;
            ParquetReader::new(file)
                .finish()
                .map_err(SurvivalDataError::from)
        }
        _ => {
            let file = File::open(path)?;
            CsvReader::new(file)
                .with_has_header(true)
                .with_separator(b'\t')
                .finish()
                .map_err(SurvivalDataError::from)
        }
    }
}

fn build_case_insensitive_map(names: Vec<String>) -> HashMap<String, String> {
    let mut map = HashMap::with_capacity(names.len());
    for name in names {
        map.insert(name.to_ascii_lowercase(), name);
    }
    map
}

fn extract_f64_column(
    df: &DataFrame,
    map: &HashMap<String, String>,
    key: &str,
) -> Result<Array1<f64>, SurvivalDataError> {
    let actual = map
        .get(&key.to_ascii_lowercase())
        .ok_or_else(|| SurvivalDataError::ColumnNotFound(key.to_string()))?;
    let series = df
        .column(actual)
        .map_err(|_| SurvivalDataError::ColumnNotFound(actual.clone()))?;
    let dtype = series.dtype().clone();
    let series = if dtype != DataType::Float64 {
        series
            .cast(&DataType::Float64)
            .map_err(|_| SurvivalDataError::ColumnWrongType {
                column_name: actual.clone(),
                expected_type: "float",
                found_type: dtype.to_string(),
            })?
    } else {
        series.clone()
    };
    let values = series.f64().expect("casted to f64");
    if values.null_count() > 0 {
        return Err(SurvivalDataError::MissingValues(actual.clone()));
    }
    Ok(Array1::from_iter(values.into_no_null_iter()))
}

fn extract_u8_column(
    df: &DataFrame,
    map: &HashMap<String, String>,
    key: &str,
) -> Result<Array1<u8>, SurvivalDataError> {
    let actual = map
        .get(&key.to_ascii_lowercase())
        .ok_or_else(|| SurvivalDataError::ColumnNotFound(key.to_string()))?;
    let series = df
        .column(actual)
        .map_err(|_| SurvivalDataError::ColumnNotFound(actual.clone()))?;
    let dtype = series.dtype().clone();
    let series = if dtype != DataType::UInt8 {
        series
            .cast(&DataType::UInt8)
            .map_err(|_| SurvivalDataError::ColumnWrongType {
                column_name: actual.clone(),
                expected_type: "integer",
                found_type: dtype.to_string(),
            })?
    } else {
        series.clone()
    };
    let values = series.u8().expect("casted to u8");
    if values.null_count() > 0 {
        return Err(SurvivalDataError::MissingValues(actual.clone()));
    }
    Ok(Array1::from_iter(values.into_no_null_iter()))
}

fn assemble_covariate_matrix(
    pgs: &Array1<f64>,
    sex: &Array1<f64>,
    pcs: &Array2<f64>,
) -> Array2<f64> {
    let n = pgs.len();
    let mut matrix = Array2::<f64>::zeros((n, 2 + pcs.ncols()));
    matrix.column_mut(0).assign(pgs);
    matrix.column_mut(1).assign(sex);
    for j in 0..pcs.ncols() {
        let column = pcs.column(j);
        matrix.column_mut(2 + j).assign(&column);
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{CsvWriter, DataFrame, ParquetWriter, SerWriter, Series, df};
    use tempfile::NamedTempFile;

    fn sample_dataframe() -> DataFrame {
        df! {
            "age_entry" => &[50.0, 60.0, 70.0],
            "age_exit" => &[55.0, 65.0, 75.0],
            "event_target" => &[1u8, 0, 0],
            "event_competing" => &[0u8, 1, 0],
            "sample_weight" => &[1.0, 2.0, 3.0],
            "pgs" => &[0.1, 0.2, 0.3],
            "sex" => &[0.0, 1.0, 0.0],
            "pc1" => &[0.5, 0.6, 0.7],
            "pc2" => &[1.5, 1.6, 1.7]
        }
        .expect("construct sample dataframe")
    }

    fn write_tsv(df: &DataFrame) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("tempfile");
        let mut writer = CsvWriter::new(file.as_file_mut());
        writer
            .with_delimiter(b'\t')
            .finish(df.clone())
            .expect("write tsv");
        file
    }

    fn write_parquet(df: &DataFrame) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("tempfile");
        let mut writer = ParquetWriter::new(file.as_file_mut());
        writer.finish(df.clone()).expect("write parquet");
        file
    }

    #[test]
    fn training_loader_reads_tsv() {
        let df = sample_dataframe();
        let file = write_tsv(&df);
        let bundle = load_survival_training_data(file.path().to_str().unwrap(), 2, 0.1)
            .expect("load training data");
        assert_eq!(bundle.data.age_entry.len(), 3);
        assert_eq!(bundle.age_transform.minimum_age, 50.0);
        assert_eq!(bundle.data.sample_weight[1], 2.0);
    }

    #[test]
    fn training_loader_defaults_weights() {
        let mut df = sample_dataframe();
        df.drop_in_place("sample_weight").unwrap();
        let file = write_tsv(&df);
        let bundle = load_survival_training_data(file.path().to_str().unwrap(), 2, 0.1)
            .expect("load training data");
        assert!(
            bundle
                .data
                .sample_weight
                .iter()
                .all(|&w| (w - 1.0).abs() < 1e-12)
        );
    }

    #[test]
    fn prediction_loader_reads_parquet() {
        let df = sample_dataframe();
        let file = write_parquet(&df);
        let prediction = load_survival_prediction_data(file.path().to_str().unwrap(), 2)
            .expect("load prediction");
        let inputs = prediction.as_inputs();
        assert_eq!(inputs.covariates.ncols(), 4);
        assert_eq!(inputs.age_exit.len(), 3);
    }

    #[test]
    fn loader_rejects_conflicting_events() {
        let mut df = sample_dataframe();
        df = df
            .with_column(Series::new("event_target", &[1u8, 1, 0]))
            .unwrap();
        let file = write_tsv(&df);
        let err = load_survival_training_data(file.path().to_str().unwrap(), 2, 0.1)
            .expect_err("conflicting events");
        match err {
            SurvivalDataError::Validation(SurvivalError::ConflictingEvents) => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
