//! # Data Loading and Validation Module
//!
//! This module serves as the exclusive entry point for user-provided data.
//! Its primary responsibility is to read tabular data files (CSV/TSV),
//! validate them against a strict, predefined schema, and transform them
//! into the clean `ndarray` structures required by the application's
//! statistical core.
//!
//! ## Design Philosophy
//! - **Strict Schema:** Column names are not configurable. The module enforces
//!   the use of `phenotype`, `score`, `PC1`, `PC2`, etc. This simplifies the
//!   user interface and eliminates a class of configuration errors.
//! - **User-Centric Errors:** Failures are assumed to be user-input errors.
//!   The `DataError` enum is designed to provide clear, actionable feedback.
//! - **Performance:** It leverages the `polars` Lazy API to minimize memory
//!   usage and I/O by only loading required columns from disk.

use ndarray::{Array1, Array2};
use polars::prelude::*;

use std::collections::HashSet;
use thiserror::Error;

// --- Public Data Structures ---

/// A container for validated data ready for model training.
#[derive(Debug)]
pub struct TrainingData {
    /// The phenotype vector (`Y`), from the 'phenotype' column.
    pub y: Array1<f64>,
    /// The Polygenic Score vector (`P`), from the 'score' column.
    pub p: Array1<f64>,
    /// The Principal Components matrix (`PC`), from 'PC1', 'PC2', ... columns.
    /// Shape: [n_samples, num_pcs].
    pub pcs: Array2<f64>,
}

/// A container for validated data ready for prediction.
#[derive(Debug)]
pub struct PredictionData {
    /// The Polygenic Score vector (`P`), from the 'score' column.
    pub p: Array1<f64>,
    /// The Principal Components matrix (`PC`), from 'PC1', 'PC2', ... columns.
    /// Shape: [n_samples, num_pcs].
    pub pcs: Array2<f64>,
}

/// A comprehensive error type for all data loading and validation failures.
#[derive(Error, Debug)]
pub enum DataError {
    #[error("Error from the underlying Polars DataFrame library: {0}")]
    PolarsError(#[from] PolarsError),
    #[error("The required column '{0}' was not found in the input file. Please check spelling and case.")]
    ColumnNotFound(String),
    #[error("The required column '{column_name}' could not be converted to the expected type '{expected_type}'. It contains non-numeric data. (Found type: {found_type})")]
    ColumnWrongType {
        column_name: String,
        expected_type: &'static str,
        found_type: String,
    },
    #[error("Missing or null values were found in the required column '{0}'. This tool requires complete data with no missing values.")]
    MissingValuesFound(String),
    #[error("Input file contains only {found} data rows, but at least {required} are recommended for a stable model.")]
    InsufficientRows { found: usize, required: usize },
}

/// Loads and validates data specifically for model training.
pub fn load_training_data(path: &str, num_pcs: usize) -> Result<TrainingData, DataError> {
    let (p, pcs, y_opt) = internal::load_data(path, num_pcs, true)?;
    // This unwrap is safe because we passed `include_phenotype: true`.
    let y = y_opt.unwrap();
    Ok(TrainingData { y, p, pcs })
}

/// Loads and validates data specifically for prediction.
pub fn load_prediction_data(path: &str, num_pcs: usize) -> Result<PredictionData, DataError> {
    let (p, pcs, _) = internal::load_data(path, num_pcs, false)?;
    Ok(PredictionData { p, pcs })
}

/// Internal module for shared data loading logic.
mod internal {
    use super::*;

    const MINIMUM_ROWS: usize = 20;

    /// The single, unified data loading function. It reads a file, validates it against
    /// a dynamically generated schema, and returns the core `ndarray` objects.
    pub(super) fn load_data(
        path: &str,
        num_pcs: usize,
        include_phenotype: bool,
    ) -> Result<(Array1<f64>, Array2<f64>, Option<Array1<f64>>), DataError> {
        // --- 1. Generate the exact list of required column names ---
        let pc_names: Vec<String> = (1..=num_pcs).map(|i| format!("PC{}", i)).collect();
        let mut required_cols: Vec<String> = Vec::with_capacity(2 + num_pcs);
        if include_phenotype {
            required_cols.push("phenotype".to_string());
        }
        required_cols.push("score".to_string());
        required_cols.extend_from_slice(&pc_names);

        // --- 2. Delegate reading and structural validation to the core helper ---
        let df = load_and_validate_dataframe(path, &required_cols)?;

        // --- 3. Extract and convert columns, with specific type-error handling ---
        let phenotype_opt = if include_phenotype {
            Some(series_to_f64_array(df.column("phenotype")?)?)
        } else {
            None
        };
        let pgs = series_to_f64_array(df.column("score")?)?;
        let pcs_polars = df.select(&pc_names)?.to_ndarray::<Float64Type>()?;
        let pcs = Array2::from_shape_vec(pcs_polars.dim(), pcs_polars.into_raw_vec()).unwrap();

        Ok((pgs, pcs, phenotype_opt))
    }

    /// Reads a file into a Polars DataFrame, validating schema and data integrity.
    fn load_and_validate_dataframe(
        path: &str,
        required_cols: &[String],
    ) -> Result<DataFrame, DataError> {
        println!("Loading data from '{}'", path);
        let mut lf = LazyCsvReader::new(path)
            .with_separator(b'\t')
            .with_infer_schema_length(Some(100))
            .finish()?;

        let available_cols: HashSet<_> = lf.collect_schema()?.iter_names().map(|s| s.to_string()).collect();
        for col_name in required_cols {
            if !available_cols.contains(col_name) {
                return Err(DataError::ColumnNotFound(col_name.clone()));
            }
        }
        println!("All required columns found: {:?}", required_cols);

        let col_exprs: Vec<Expr> = required_cols.iter().map(|name| col(name)).collect();
        let df = lf.select(col_exprs).collect()?;
        println!("Successfully loaded {} rows.", df.height());

        if df.height() < MINIMUM_ROWS {
            return Err(DataError::InsufficientRows {
                found: df.height(),
                required: MINIMUM_ROWS,
            });
        }
        for col_name in required_cols {
            if df.column(col_name)?.null_count() > 0 {
                return Err(DataError::MissingValuesFound(col_name.clone()));
            }
        }
        println!("Data validation successful: no missing values found.");
        Ok(df)
    }

    /// Helper to convert a Polars Series to an ndarray Array1<f64>, providing
    /// a more specific error message on failure.
    fn series_to_f64_array(series: &Series) -> Result<Array1<f64>, DataError> {
        let polars_array = series
            .f64()
            .map_err(|_| DataError::ColumnWrongType {
                column_name: series.name().to_string(),
                expected_type: "f64 (numeric)",
                found_type: format!("{:?}", series.dtype()),
            })?
            .to_ndarray()?;
        
        // Convert from polars ndarray (0.15) to our ndarray (0.16)
        Array1::from_shape_vec(polars_array.dim(), polars_array.into_raw_vec())
            .map_err(|_| DataError::ColumnWrongType {
                column_name: series.name().to_string(),
                expected_type: "f64 (numeric)",
                found_type: "conversion error".to_string(),
            })
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{self, Write};
    use tempfile::NamedTempFile;

    /// A robust helper to create a temporary CSV file for testing.
    fn create_test_csv(content: &str) -> io::Result<NamedTempFile> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "{}", content)?;
        file.flush()?;
        Ok(file)
    }

    /// Generates CSV content with a specified number of data rows.
    fn generate_csv_content(header: &str, data_row: &str, num_rows: usize) -> String {
        let data_rows = std::iter::repeat(data_row).take(num_rows).collect::<Vec<_>>().join("\n");
        format!("{}\n{}", header, data_rows)
    }
    
    const TEST_HEADER: &str = "phenotype,score,PC1,PC2,extra_col";
    const TEST_DATA_ROW: &str = "1,1.5,0.1,0.2,A";

    #[test]
    fn test_load_training_data_success() {
        let content = generate_csv_content(TEST_HEADER, TEST_DATA_ROW, 30);
        let file = create_test_csv(&content).unwrap();
        let data = load_training_data(file.path().to_str().unwrap(), 2).unwrap();

        assert_eq!(data.y.len(), 30);
        assert_eq!(data.p.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 2]);
        assert_eq!(data.y[0], 1.0);
        assert_eq!(data.p[0], 1.5);
        assert_eq!(data.pcs[[0, 1]], 0.2);
    }

    #[test]
    fn test_load_prediction_data_success() {
        let content = generate_csv_content(TEST_HEADER, TEST_DATA_ROW, 30);
        let file = create_test_csv(&content).unwrap();
        let data = load_prediction_data(file.path().to_str().unwrap(), 1).unwrap();

        assert_eq!(data.p.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 1]);
        assert_eq!(data.p[0], 1.5);
        assert_eq!(data.pcs[[0, 0]], 0.1);
    }

    #[test]
    fn test_error_column_not_found() {
        let content = generate_csv_content("phenotype,score,PC1", "1,1.5,0.1", 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 2).unwrap_err();
        match err {
            DataError::ColumnNotFound(col) => assert_eq!(col, "PC2"),
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_error_missing_values() {
        let content = generate_csv_content("phenotype,score,PC1", "1,,0.1", 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::MissingValuesFound(col) => assert_eq!(col, "score"),
            _ => panic!("Expected MissingValuesFound error"),
        }
    }

    #[test]
    fn test_error_wrong_type() {
        let content = generate_csv_content("phenotype,score,PC1", "1,not_a_number,0.1", 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::ColumnWrongType { column_name, expected_type, found_type } => {
                assert_eq!(column_name, "score");
                assert_eq!(expected_type, "f64 (numeric)");
                assert_eq!(found_type, "Utf8");
            },
            _ => panic!("Expected ColumnWrongType error"),
        }
    }

    #[test]
    fn test_error_insufficient_rows() {
        let content = generate_csv_content(TEST_HEADER, TEST_DATA_ROW, 5);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::InsufficientRows { found, required } => {
                assert_eq!(found, 5);
                assert_eq!(required, 20);
            },
            _ => panic!("Expected InsufficientRows error"),
        }
    }
}
