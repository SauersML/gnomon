//! # Data Loading and Validation Module
//!
//! This module serves as the exclusive entry point for user-provided data.
//! Its primary responsibility is to read tabular data files (TSV),
//! validate them against a strict, predefined schema, and transform them
//! into the clean `ndarray` structures required by the application's
//! statistical core.
//!
//! - Strict Schema: Column names are not configurable. The module enforces
//!   the use of `phenotype`, `score`, `PC1`, `PC2`, etc. This simplifies the
//!   user interface and eliminates a class of configuration errors.
//! - User-Centric Errors: Failures are assumed to be user-input errors.
//!   The `DataError` enum is designed to provide clear, actionable feedback.
//! - Training and prediction share schema validation and predictor extraction.
//!   Training-only requirements do not constrain prediction batches.

use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

/// A container for validated data ready for model training.
#[derive(Debug)]
pub struct TrainingData {
    /// The phenotype vector (`Y`), from the 'phenotype' column.
    pub y: Array1<f64>,
    /// The Polygenic Score vector (`P`), from the 'score' column.
    pub p: Array1<f64>,
    /// Binary sex indicator (e.g., 0 for female, 1 for male).
    pub sex: Array1<f64>,
    /// The Principal Components matrix (`PC`), from 'PC1', 'PC2', ... columns.
    /// Shape: [n_samples, num_pcs].
    pub pcs: Array2<f64>,
    /// Prior weights, or unit weights when the column is absent.
    pub weights: Array1<f64>,
}

/// A container for validated data ready for prediction.
#[derive(Debug)]
pub struct PredictionData {
    /// The Polygenic Score vector (`P`), from the 'score' column.
    pub p: Array1<f64>,
    /// Binary sex indicator (e.g., 0 for female, 1 for male).
    pub sex: Array1<f64>,
    /// The Principal Components matrix (`PC`), from 'PC1', 'PC2', ... columns.
    /// Shape: [n_samples, num_pcs].
    pub pcs: Array2<f64>,
    /// Optional sample identifiers. If an input column `sample_id` exists, it is used;
    /// otherwise sequential IDs (1-based) are generated as strings.
    pub sample_ids: Vec<String>,
}

/// A comprehensive error type for all data loading and validation failures.
#[derive(Error, Debug)]
pub enum DataError {
    #[error("Error from the underlying Polars DataFrame library: {0}")]
    PolarsError(#[from] PolarsError),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error(
        "The required column '{0}' was not found in the input file. Please check spelling and case."
    )]
    ColumnNotFound(String),
    #[error(
        "The required column '{column_name}' could not be converted to the expected type '{expected_type}'. It contains non-numeric data. (Found type: {found_type})"
    )]
    ColumnWrongType {
        column_name: String,
        expected_type: &'static str,
        found_type: String,
    },
    #[error(
        "Missing or null values were found in the required column '{0}'. This tool requires complete data with no missing values."
    )]
    MissingValuesFound(String),
    #[error(
        "Input file contains {found} data rows, but this operation requires at least {required}."
    )]
    InsufficientRows { found: usize, required: usize },

    #[error(
        "Non-finite values (NaN or Infinity) were found in the required column '{0}'. This tool requires all data to be finite."
    )]
    NonFiniteValuesFound(String),
}

/// Loads and validates data specifically for model training.
pub fn load_training_data(path: &str, num_pcs: usize) -> Result<TrainingData, DataError> {
    // Do not require an explicit 'weights' column. If missing, default to 1.0.
    let df = internal::load_data(path, num_pcs, true)?;
    let (p, sex, pcs) = internal::extract_predictors(&df, num_pcs)?;
    let y = internal::extract_f64_column(&df, "phenotype")?;
    let weights = internal::extract_weights(&df)?;
    Ok(TrainingData {
        y,
        p,
        sex,
        pcs,
        weights,
    })
}

/// Loads and validates data specifically for prediction.
pub fn load_prediction_data(path: &str, num_pcs: usize) -> Result<PredictionData, DataError> {
    let df = internal::load_data(path, num_pcs, false)?;
    let (p, sex, pcs) = internal::extract_predictors(&df, num_pcs)?;
    let sample_ids = internal::extract_sample_ids(&df)?;
    Ok(PredictionData {
        p,
        sex,
        pcs,
        sample_ids,
    })
}

/// Internal module for shared data loading logic.
mod internal {
    use super::*;

    const MINIMUM_ROWS: usize = 20;

    /// Read a TSV file into a Polars DataFrame.
    fn read_tsv(path: &str) -> Result<DataFrame, DataError> {
        let mut schema = Schema::with_capacity(1);
        schema.with_column("sample_id".into(), DataType::String);
        Ok(CsvReader::new(File::open(Path::new(path))?)
            .with_options(
                CsvReadOptions::default()
                    .with_has_header(true)
                    .with_schema_overwrite(Some(std::sync::Arc::new(schema)))
                    .with_parse_options(CsvParseOptions::default().with_separator(b'\t')),
            )
            .finish()?)
    }

    /// Extract a single column from a DataFrame as a validated, finite Array1<f64>.
    /// Checks for nulls before and after casting, and validates all values are finite.
    pub(super) fn extract_f64_column(
        df: &DataFrame,
        column_name: &str,
    ) -> Result<Array1<f64>, DataError> {
        let series = df.column(column_name)?;
        if series.null_count() > 0 {
            return Err(DataError::MissingValuesFound(column_name.to_string()));
        }
        let casted = series
            .cast(&DataType::Float64)
            .map_err(|_| DataError::ColumnWrongType {
                column_name: column_name.to_string(),
                expected_type: "f64 (numeric)",
                found_type: format!("{:?}", series.dtype()),
            })?;
        if casted.null_count() > 0 {
            return Err(DataError::ColumnWrongType {
                column_name: column_name.to_string(),
                expected_type: "f64 (numeric)",
                found_type: format!("{:?}", series.dtype()),
            });
        }
        let arr = Array1::from_vec(casted.rechunk().f64()?.into_no_null_iter().collect());
        if arr.iter().any(|&v| !v.is_finite()) {
            return Err(DataError::NonFiniteValuesFound(column_name.to_string()));
        }
        Ok(arr)
    }

    /// Preserve supplied identifiers exactly; generate row numbers only if absent.
    pub(super) fn extract_sample_ids(df: &DataFrame) -> Result<Vec<String>, DataError> {
        let has_col = df.get_column_names().iter().any(|c| c == &"sample_id");
        if !has_col {
            return Ok((1..=df.height()).map(|i| i.to_string()).collect());
        }
        let s = df.column("sample_id")?;
        if s.null_count() > 0 {
            return Err(DataError::MissingValuesFound("sample_id".to_string()));
        }
        Ok(s.str()?.into_no_null_iter().map(str::to_owned).collect())
    }

    /// Stack per-column PC arrays into an (n_rows, n_cols) matrix.
    fn stack_pc_matrix(pc_arrays: &[Array1<f64>], fallback_rows: usize) -> Array2<f64> {
        if pc_arrays.is_empty() {
            return Array2::zeros((fallback_rows, 0));
        }
        let n_rows = pc_arrays[0].len();
        let n_cols = pc_arrays.len();
        let mut flat = Vec::with_capacity(n_rows * n_cols);
        for col in pc_arrays {
            flat.extend(col.iter().cloned());
        }
        Array2::from_shape_vec((n_cols, n_rows), flat)
            .expect("PC arrays should have consistent dimensions")
            .reversed_axes()
    }

    /// Validate supplied weights independently of whether the column is optional.
    pub(super) fn extract_weights(df: &DataFrame) -> Result<Array1<f64>, DataError> {
        let has_weights = df.get_column_names().iter().any(|c| c == &"weights");
        if !has_weights {
            return Ok(Array1::from_elem(df.height(), 1.0));
        }
        let weights = extract_f64_column(df, "weights")?;
        for (i, &w) in weights.iter().enumerate() {
            if w < 0.0 {
                return Err(DataError::ColumnWrongType {
                    column_name: "weights".to_string(),
                    expected_type: "non-negative f64 values",
                    found_type: format!("negative value {} at row {}", w, i + 1),
                });
            }
        }
        if !weights.iter().any(|&w| w > 0.0) {
            return Err(DataError::ColumnWrongType {
                column_name: "weights".to_string(),
                expected_type: "at least one positive weight",
                found_type: "all weights are zero".to_string(),
            });
        }
        Ok(weights)
    }

    /// Read once and validate the schema; the row minimum applies only to training.
    pub(super) fn load_data(
        path: &str,
        num_pcs: usize,
        include_phenotype: bool,
    ) -> Result<DataFrame, DataError> {
        let pc_names: Vec<String> = (1..=num_pcs).map(|i| format!("PC{i}")).collect();
        let mut required_cols: Vec<String> = Vec::with_capacity(3 + num_pcs);
        if include_phenotype {
            required_cols.push("phenotype".to_string());
        }
        required_cols.push("score".to_string());
        required_cols.push("sex".to_string());
        required_cols.extend_from_slice(&pc_names);

        println!("Loading data from '{path}'");
        let df = read_tsv(path)?;
        println!("Successfully loaded data file.");

        let minimum_rows = if include_phenotype { MINIMUM_ROWS } else { 1 };
        if df.height() < minimum_rows {
            return Err(DataError::InsufficientRows {
                found: df.height(),
                required: minimum_rows,
            });
        }

        let columns_set: HashSet<String> = df
            .get_column_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        for col_name in &required_cols {
            if !columns_set.contains(col_name) {
                return Err(DataError::ColumnNotFound(col_name.clone()));
            }
        }
        Ok(df)
    }

    pub(super) fn extract_predictors(
        df: &DataFrame,
        num_pcs: usize,
    ) -> Result<(Array1<f64>, Array1<f64>, Array2<f64>), DataError> {
        let pgs = extract_f64_column(df, "score")?;
        let sex = extract_f64_column(df, "sex")?;
        let mut pc_arrays = Vec::with_capacity(num_pcs);
        for i in 1..=num_pcs {
            pc_arrays.push(extract_f64_column(df, &format!("PC{i}"))?);
        }
        let pcs = stack_pc_matrix(&pc_arrays, pgs.len());
        Ok((pgs, sex, pcs))
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::io::{self, Write};
    use tempfile::NamedTempFile;

    /// A robust helper to create a temporary CSV file for testing.
    fn create_test_csv(content: &str) -> io::Result<NamedTempFile> {
        let mut file = NamedTempFile::new()?;
        writeln!(file, "{}", content)?;
        file.flush()?;
        Ok(file)
    }

    #[test]
    fn test_non_finite_values_rejected_in_score() {
        let header = "phenotype\tscore\tsex\tPC1";
        let data_row = "1.0\tNaN\t0\t0.1"; // NaN should be parsed as f64::NAN
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::NonFiniteValuesFound(col) => assert_eq!(col, "score"),
            other => panic!("Expected NonFiniteValuesFound(score), got {:?}", other),
        }
    }

    #[test]
    fn test_non_finite_values_rejected_in_pc() {
        let header = "phenotype\tscore\tsex\tPC1";
        let data_row = "1.0\t2.0\t0\tNaN"; // NaN in PC1
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 1).unwrap_err();
        match err {
            DataError::NonFiniteValuesFound(col) => assert_eq!(col, "PC1"),
            other => panic!("Expected NonFiniteValuesFound(PC1), got {:?}", other),
        }
    }

    /// Generates CSV content with a specified number of data rows.
    fn generate_csv_content(header: &str, data_row: &str, num_rows: usize) -> String {
        let data_rows = std::iter::repeat(data_row)
            .take(num_rows)
            .collect::<Vec<_>>()
            .join("\n");
        format!("{}\n{}", header, data_rows)
    }

    const TEST_HEADER: &str = "phenotype\tscore\tsex\tPC1\tPC2\textra_col";
    const TEST_DATA_ROW: &str = "1.0\t1.5\t0\t0.1\t0.2\t1.0";

    #[test]
    fn test_load_training_data_success() {
        // Create test data that includes required columns including weights
        let header = "phenotype\tscore\tsex\tPC1\tPC2\tweights";

        // Generate varied test data with different values in each row
        let mut rows = Vec::with_capacity(31);
        rows.push(header.to_string());

        for i in 0..30 {
            let row = format!(
                "{:.2}\t{:.2}\t{}\t{:.3}\t{:.3}\t{:.1}",
                i as f64 / 10.0,
                (i as f64 + 5.0) / 10.0,
                i % 2,
                (i as f64 - 2.0) / 20.0,
                (i as f64 + 3.0) / 15.0,
                1.0 // Add weight of 1.0 for each row
            );
            rows.push(row);
        }

        let content = rows.join("\n");
        let file = create_test_csv(&content).unwrap();
        let data = load_training_data(file.path().to_str().unwrap(), 2).unwrap();

        // Test dimensions
        assert_eq!(data.y.len(), 30);
        assert_eq!(data.p.len(), 30);
        assert_eq!(data.sex.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 2]);

        // Test specific values in the first row
        assert_abs_diff_eq!(data.y[0], 0.00, epsilon = 1e-6);
        assert_abs_diff_eq!(data.p[0], 0.50, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 0]], -0.100, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 1]], 0.200, epsilon = 1e-6);

        // Test specific values in a middle row
        assert_abs_diff_eq!(data.y[15], 1.50, epsilon = 1e-6);
        assert_abs_diff_eq!(data.p[15], 2.00, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[15], (15 % 2) as f64, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[15, 0]], 0.650, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[15, 1]], 1.200, epsilon = 1e-6);

        // Test specific values in the last row
        assert_abs_diff_eq!(data.y[29], 2.90, epsilon = 1e-6);
        assert_abs_diff_eq!(data.p[29], 3.40, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[29], (29 % 2) as f64, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[29, 0]], 1.350, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[29, 1]], 2.133, epsilon = 1e-6);
    }

    #[test]
    fn test_pcs_matrix_structure_with_custom_data() {
        // Create test data with different values for each row including weights
        let custom_header = "phenotype\tscore\tsex\tPC1\tPC2\tweights";

        // Generate 20 rows with a clear pattern to meet the minimum row requirement
        let mut custom_rows = Vec::with_capacity(21);
        custom_rows.push(custom_header.to_string());

        for i in 0..20 {
            let row = format!(
                "{:.1}\t{:.1}\t{}\t{:.1}\t{:.1}\t{:.1}",
                i as f64 / 10.0,
                (i as f64 + 1.0) / 10.0,
                (i % 2),
                (i as f64 + 2.0) / 10.0,
                (i as f64 + 3.0) / 10.0,
                1.0 // Add weight of 1.0 for each row
            );
            custom_rows.push(row); // Push the owned String, not a reference
        }

        let custom_content = custom_rows.join("\n");
        let file = create_test_csv(&custom_content).unwrap();
        let data = load_training_data(file.path().to_str().unwrap(), 2).unwrap();

        // Verify dimensions
        assert_eq!(data.sex.len(), 20);
        assert_eq!(data.pcs.shape(), &[20, 2]);

        // Verify the matrix contents for a few key elements
        assert_abs_diff_eq!(data.sex[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 0]], 0.2, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 1]], 0.3, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[10, 0]], 1.2, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[10, 1]], 1.3, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[19, 0]], 2.1, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[19, 1]], 2.2, epsilon = 1e-6);
    }

    #[test]
    fn test_load_prediction_data_success() {
        // Create test data that includes required columns including weights
        let header = "score\tsex\tPC1\tweights";
        let data_row = "1.5\t1\t0.1\t1.0"; // Added dummy weight
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let data = load_prediction_data(file.path().to_str().unwrap(), 1).unwrap();

        assert_eq!(data.p.len(), 30);
        assert_eq!(data.sex.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 1]);
        assert_abs_diff_eq!(data.p[0], 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(data.sex[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(data.pcs[[0, 0]], 0.1, epsilon = 1e-6);
    }

    #[test]
    fn prediction_accepts_one_sample_and_preserves_identifiers() {
        for sample_id in ["000123", "sample-A", "9007199254740993"] {
            let file = create_test_csv(&format!(
                "sample_id\tscore\tsex\tPC1\n{sample_id}\t1.5\t1\t0.1"
            ))
            .unwrap();
            let data = load_prediction_data(file.path().to_str().unwrap(), 1).unwrap();
            assert_eq!(data.sample_ids, vec![sample_id]);
            assert_eq!(data.pcs.shape(), &[1, 1]);
            assert_eq!(data.p[0], 1.5);
        }
    }

    #[test]
    fn prediction_rejects_missing_supplied_identifiers() {
        let file = create_test_csv("sample_id\tscore\tsex\nkept\t1.5\t1\n\t2.0\t0").unwrap();
        assert!(matches!(
            load_prediction_data(file.path().to_str().unwrap(), 0),
            Err(DataError::MissingValuesFound(column)) if column == "sample_id"
        ));
    }

    #[test]
    fn prediction_without_ids_uses_row_numbers_and_ignores_training_columns() {
        let file =
            create_test_csv("score\tsex\tweights\tphenotype\n1.5\t1\tunused\tunused").unwrap();
        let data = load_prediction_data(file.path().to_str().unwrap(), 0).unwrap();
        assert_eq!(data.sample_ids, vec!["1"]);
        assert_eq!(data.pcs.shape(), &[1, 0]);
    }

    #[test]
    fn training_rejects_invalid_optional_weights() {
        for weight in ["-1", "0"] {
            let content = generate_csv_content(
                "phenotype\tscore\tsex\tweights",
                &format!("1\t1.5\t0\t{weight}"),
                20,
            );
            let file = create_test_csv(&content).unwrap();
            assert!(matches!(
                load_training_data(file.path().to_str().unwrap(), 0),
                Err(DataError::ColumnWrongType { column_name, .. }) if column_name == "weights"
            ));
        }
    }

    #[test]
    fn test_error_column_not_found() {
        let content = generate_csv_content("phenotype\tscore\tsex\tPC1", "1.0\t1.5\t0\t0.1", 30);
        let file = create_test_csv(&content).unwrap();
        let err = load_training_data(file.path().to_str().unwrap(), 2).unwrap_err();
        match err {
            DataError::ColumnNotFound(col) => assert_eq!(col, "PC2"),
            _ => panic!("Expected ColumnNotFound error"),
        }
    }

    #[test]
    fn test_error_missing_values() {
        // Create test data with missing values in the score column
        let content = generate_csv_content("phenotype\tscore\tsex\tPC1", "1.0\t\t0\t0.1", 30);
        let file = create_test_csv(&content).unwrap();

        // Call the function that should detect the error
        let result = load_training_data(file.path().to_str().unwrap(), 1);

        // Verify the function returns the correct error
        match result {
            Err(DataError::MissingValuesFound(col_name)) => {
                assert_eq!(col_name, "score", "Expected error for 'score' column");
            }
            _ => panic!(
                "Expected MissingValuesFound error for 'score' column, but got {:?}",
                result
            ),
        }
    }

    #[test]
    fn test_error_wrong_type() {
        // Create test data with a non-numeric value in the score column
        let content = generate_csv_content(
            "phenotype\tscore\tsex\tPC1",
            "1.0\tnot_a_number\t0\t0.1",
            30,
        );
        let file = create_test_csv(&content).unwrap();

        // Call the function that should detect the error
        let result = load_training_data(file.path().to_str().unwrap(), 1);

        // Verify the function returns the correct error
        match result {
            Err(DataError::ColumnWrongType {
                column_name,
                expected_type,
                found_type,
            }) => {
                assert_eq!(column_name, "score", "Expected error for 'score' column");
                assert_eq!(expected_type, "f64 (numeric)", "Expected f64 type");
                assert!(
                    found_type.contains("String") || found_type.contains("text"),
                    "Expected found_type to indicate String or text, got {}",
                    found_type
                );
            }
            _ => panic!(
                "Expected ColumnWrongType error for 'score' column, but got {:?}",
                result
            ),
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
            }
            _ => panic!("Expected InsufficientRows error"),
        }
    }
}
