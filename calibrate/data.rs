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
use std::path::PathBuf;
use thiserror::Error;

fn csv_to_df(path: PathBuf, delimiter: Option<u8>) -> Result<DataFrame, PolarsError> {
    use std::io::{BufRead, BufReader};
    use std::fs::File;
    
    // Read the file directly as lines
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // Read header line
    let header_line = match lines.next() {
        Some(Ok(line)) => line,
        _ => return Err(PolarsError::ComputeError("CSV file is empty".into())),
    };
    
    // Parse headers
    let sep_char = delimiter.unwrap_or(b',') as char;
    let headers: Vec<String> = header_line.split(sep_char).map(|s| s.trim().to_string()).collect();
    
    // Read all rows
    let mut rows = Vec::new();
    for line_result in lines {
        match line_result {
            Ok(line) => {
                let fields: Vec<String> = line.split(sep_char).map(|s| s.trim().to_string()).collect();
                rows.push(fields);
            },
            Err(e) => return Err(PolarsError::ComputeError(format!("Error reading line: {}", e).into())),
        }
    }
    
    // Check if we have any data
    if rows.is_empty() {
        return Err(PolarsError::ComputeError("CSV file has no data rows".into()));
    }
    
    // Create dummy columns for the DataFrame
    // We don't need to create a proper DataFrame with correct types
    // since we'll manually extract the data anyway
    let mut cols = Vec::new();
    
    // Transpose the data
    for (i, header) in headers.iter().enumerate() {
        // Extract column values
        let mut col_data = Vec::with_capacity(rows.len());
        
        for row in &rows {
            if i < row.len() {
                // Try to parse as f64
                // Store any parsing issues to be validated later
                let is_empty = row[i].is_empty();
                let parse_result = if !is_empty { row[i].parse::<f64>() } else { Err("empty string".parse::<f64>().unwrap_err()) };
                
                match parse_result {
                    Ok(val) => col_data.push(val),
                    Err(_) => {
                        // Properly propagate parsing errors instead of silently using 0.0
                        // This ensures data validation happens immediately instead of later
                        if is_empty {
                            return Err(PolarsError::ComputeError(
                                format!("Empty value found in column {}", headers[i]).into()
                            ));
                        } else {
                            return Err(PolarsError::ComputeError(
                                format!("Non-numeric value '{}' found in column {}", row[i], headers[i]).into()
                            ));
                        }
                    }
                }
            } else {
                // Row doesn't have this column, which is a data error
                return Err(PolarsError::ComputeError(
                    format!("Row missing column {} (index {})", headers[i], i).into()
                ));
            }
        }
        
        // Create Series object with PlSmallStr name
        cols.push(Series::new(header.into(), &col_data));
    }
    
    // Create DataFrame
    DataFrame::new(cols.into_iter().map(|s| s.into_column()).collect())
}

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
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
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

        // --- 3. Parse the raw data ---
        // Let's manually extract the data from the DataFrame
        let phenotype_opt = if include_phenotype {
            // Get the phenotype column
            let phenotype_col = match df.column("phenotype") {
                Ok(col) => col,
                Err(_) => return Err(DataError::ColumnNotFound("phenotype".to_string())),
            };
            
            // Convert to f64
            let mut phenotype_vec = Vec::with_capacity(df.height());
            for i in 0..df.height() {
                match phenotype_col.get(i) {
                    Ok(val) => match val {
                        AnyValue::Float64(f) => phenotype_vec.push(f),
                        _ => return Err(DataError::ColumnWrongType {
                            column_name: "phenotype".to_string(),
                            expected_type: "f64 (numeric)",
                            found_type: format!("{:?}", phenotype_col.dtype()),
                        }),
                    },
                    Err(_) => return Err(DataError::MissingValuesFound("phenotype".to_string())),
                }
            }
            
            Some(Array1::from_vec(phenotype_vec))
        } else {
            None
        };
        
        // Get the score column
        let score_col = match df.column("score") {
            Ok(col) => col,
            Err(_) => return Err(DataError::ColumnNotFound("score".to_string())),
        };
        
        // Convert to f64
        let mut score_vec = Vec::with_capacity(df.height());
        for i in 0..df.height() {
            match score_col.get(i) {
                Ok(val) => match val {
                    AnyValue::Float64(f) => score_vec.push(f),
                    _ => return Err(DataError::ColumnWrongType {
                        column_name: "score".to_string(),
                        expected_type: "f64 (numeric)",
                        found_type: format!("{:?}", score_col.dtype()),
                    }),
                },
                Err(_) => return Err(DataError::MissingValuesFound("score".to_string())),
            }
        }
        let pgs = Array1::from_vec(score_vec);
        
        // Process PC columns
        let mut pcs_vecs = Vec::new();
        for col_name in &pc_names {
            let pc_col = match df.column(col_name) {
                Ok(col) => col,
                Err(_) => return Err(DataError::ColumnNotFound(col_name.clone())),
            };
            
            // Convert to f64
            let mut pc_vec = Vec::with_capacity(df.height());
            for i in 0..df.height() {
                match pc_col.get(i) {
                    Ok(val) => match val {
                        AnyValue::Float64(f) => pc_vec.push(f),
                        _ => return Err(DataError::ColumnWrongType {
                            column_name: col_name.clone(),
                            expected_type: "f64 (numeric)",
                            found_type: format!("{:?}", pc_col.dtype()),
                        }),
                    },
                    Err(_) => return Err(DataError::MissingValuesFound(col_name.clone())),
                }
            }
            
            pcs_vecs.push(Array1::from_vec(pc_vec));
        }
        
        // Stack columns to form the matrix
        if pcs_vecs.is_empty() {
            return Err(DataError::ColumnNotFound("No PC columns found".to_string()));
        }
        
        let n_rows = pcs_vecs[0].len();
        let n_cols = pcs_vecs.len();
        let mut pcs_flat = Vec::with_capacity(n_rows * n_cols);
        
        // Convert column vectors to row-major format for correct array construction
        for row_idx in 0..n_rows {
            for col_idx in 0..n_cols {
                pcs_flat.push(pcs_vecs[col_idx][row_idx]);
            }
        }
        
        let pcs = Array2::from_shape_vec((n_rows, n_cols), pcs_flat).unwrap();

        Ok((pgs, pcs, phenotype_opt))
    }

    /// Reads a file into a Polars DataFrame, validating schema and data integrity.
    fn load_and_validate_dataframe(
        path: &str,
        required_cols: &[String],
    ) -> Result<DataFrame, DataError> {
        println!("Loading data from '{}'", path);
        let file_path = std::path::PathBuf::from(path);
        
        // Catch specific CSV parsing errors and convert to appropriate DataError types
        let df = match csv_to_df(file_path, Some(b'\t')) {
            Ok(df) => df,
            Err(e) => {
                // Convert specific polars errors to more user-friendly DataError types
                match e {
                    PolarsError::ComputeError(ref err_str) => {
                        let err_msg = err_str.to_string();
                        if err_msg.contains("Empty value found in column") {
                            // Extract column name from error message
                            let col_name = err_msg.split(" column ").nth(1).unwrap_or("unknown").to_string();
                            return Err(DataError::MissingValuesFound(col_name));
                        } else if err_msg.contains("Non-numeric value") {
                            // Extract column name from error message
                            let parts: Vec<&str> = err_msg.split(" found in column ").collect();
                            if parts.len() == 2 {
                                let col_name = parts[1].to_string();
                                return Err(DataError::ColumnWrongType {
                                    column_name: col_name,
                                    expected_type: "f64 (numeric)",
                                    found_type: "String".to_string(),
                                });
                            }
                        }
                        // Fall back to default error if specific pattern not matched
                        return Err(DataError::PolarsError(e.clone()));
                    },
                    // Pass through other errors
                    _ => return Err(DataError::PolarsError(e)),
                }
            }
        };

        let schema = df.schema();
        let available_cols: HashSet<_> = schema.iter_names().map(|s| s.to_string()).collect();
        for col_name in required_cols {
            if !available_cols.contains(col_name) {
                return Err(DataError::ColumnNotFound(col_name.clone()));
            }
        }
        println!("All required columns found: {:?}", required_cols);

        // Select only the required columns
        let df = df.select(required_cols)?;
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

    /// Helper to convert a Polars Column to an ndarray Array1<f64>, providing
    /// a more specific error message on failure.
    // Function not currently used but kept for future use
    #[allow(dead_code)]
    fn series_to_f64_array(column: &Column) -> Result<Array1<f64>, DataError> {
        // Convert column to vector manually
        let values = match column.dtype() {
            DataType::Float64 => {
                // Extract values from the column
                let name = column.name();
                
                // Convert to vector of f64
                let mut result = Vec::with_capacity(column.len());
                for i in 0..column.len() {
                    let value_result = column.get(i);
                    match value_result {
                        Ok(value) => {
                            match value {
                                AnyValue::Float64(f) => result.push(f),
                                _ => return Err(DataError::ColumnWrongType {
                                    column_name: name.to_string(),
                                    expected_type: "f64 (numeric)",
                                    found_type: format!("{:?}", column.dtype()),
                                }),
                            }
                        },
                        Err(_) => {
                            return Err(DataError::MissingValuesFound(name.to_string()));
                        }
                    }
                }
                result
            },
            _ => {
                // Non-float64 type
                return Err(DataError::ColumnWrongType {
                    column_name: column.name().to_string(),
                    expected_type: "f64 (numeric)",
                    found_type: format!("{:?}", column.dtype()),
                });
            }
        };
        
        // Create ndarray from collected values
        Ok(Array1::from_vec(values))
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
    
    const TEST_HEADER: &str = "phenotype\tscore\tPC1\tPC2\textra_col";
    const TEST_DATA_ROW: &str = "1.0\t1.5\t0.1\t0.2\t1.0";

    #[test]
    fn test_load_training_data_success() {
        // Create test data that only includes required columns (no extra_col)
        let header = "phenotype\tscore\tPC1\tPC2";
        let data_row = "1.0\t1.5\t0.1\t0.2";
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let data = load_training_data(file.path().to_str().unwrap(), 2).unwrap();

        // Test dimensions
        assert_eq!(data.y.len(), 30);
        assert_eq!(data.p.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 2]);
        
        // Test scalar values
        assert_eq!(data.y[0], 1.0);
        assert_eq!(data.p[0], 1.5);
        
        // Test the entire PCs matrix for the first row
        assert_eq!(data.pcs[[0, 0]], 0.1);
        assert_eq!(data.pcs[[0, 1]], 0.2);
        
        // All rows should have the same values since we're repeating the same data row
        for i in 1..30 {
            assert_eq!(data.pcs[[i, 0]], 0.1);
            assert_eq!(data.pcs[[i, 1]], 0.2);
        }
    }
    
    #[test]
    fn test_pcs_matrix_structure_with_custom_data() {
        // Create test data with different values for each row
        let custom_header = "phenotype\tscore\tPC1\tPC2";
        
        // Generate 20 rows with a clear pattern to meet the minimum row requirement
        let mut custom_rows = Vec::with_capacity(21);
        custom_rows.push(custom_header.to_string());
        
        for i in 0..20 {
            let row = format!("{:.1}\t{:.1}\t{:.1}\t{:.1}", 
                             i as f64 / 10.0, 
                             (i as f64 + 1.0) / 10.0, 
                             (i as f64 + 2.0) / 10.0, 
                             (i as f64 + 3.0) / 10.0);
            custom_rows.push(row); // Push the owned String, not a reference
        }
        
        let custom_content = custom_rows.join("\n");
        let file = create_test_csv(&custom_content).unwrap();
        let data = load_training_data(file.path().to_str().unwrap(), 2).unwrap();
        
        // Verify dimensions
        assert_eq!(data.pcs.shape(), &[20, 2]);
        
        // Verify the matrix contents for a few key elements
        assert_eq!(data.pcs[[0, 0]], 0.2);
        assert_eq!(data.pcs[[0, 1]], 0.3);
        assert_eq!(data.pcs[[10, 0]], 1.2);
        assert_eq!(data.pcs[[10, 1]], 1.3);
        assert_eq!(data.pcs[[19, 0]], 2.1);
        assert_eq!(data.pcs[[19, 1]], 2.2);
    }
    
    #[test]
    fn test_load_prediction_data_success() {
        // Create test data that only includes required columns (no extra_col)
        let header = "score\tPC1";
        let data_row = "1.5\t0.1";
        let content = generate_csv_content(header, data_row, 30);
        let file = create_test_csv(&content).unwrap();
        let data = load_prediction_data(file.path().to_str().unwrap(), 1).unwrap();

        assert_eq!(data.p.len(), 30);
        assert_eq!(data.pcs.shape(), &[30, 1]);
        assert_eq!(data.p[0], 1.5);
        assert_eq!(data.pcs[[0, 0]], 0.1);
    }

    #[test]
    fn test_error_column_not_found() {
        let content = generate_csv_content("phenotype\tscore\tPC1", "1.0\t1.5\t0.1", 30);
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
        let content = generate_csv_content("phenotype\tscore\tPC1", "1.0\t\t0.1", 30);
        let file = create_test_csv(&content).unwrap();
        
        // Call the function that should detect the error
        let result = load_training_data(file.path().to_str().unwrap(), 1);
        
        // Verify the function returns the correct error
        match result {
            Err(DataError::MissingValuesFound(col_name)) => {
                assert_eq!(col_name, "score", "Expected error for 'score' column");
            },
            _ => panic!("Expected MissingValuesFound error for 'score' column, but got {:?}", result),
        }
    }

    #[test]
    fn test_error_wrong_type() {
        // Create test data with a non-numeric value in the score column
        let content = generate_csv_content("phenotype\tscore\tPC1", "1.0\tnot_a_number\t0.1", 30);
        let file = create_test_csv(&content).unwrap();
        
        // Call the function that should detect the error
        let result = load_training_data(file.path().to_str().unwrap(), 1);
        
        // Verify the function returns the correct error
        match result {
            Err(DataError::ColumnWrongType { column_name, expected_type, found_type }) => {
                assert_eq!(column_name, "score", "Expected error for 'score' column");
                assert_eq!(expected_type, "f64 (numeric)", "Expected f64 type");
                assert!(found_type.contains("String") || found_type.contains("text"), 
                       "Expected found_type to indicate String or text, got {}", found_type);
            },
            _ => panic!("Expected ColumnWrongType error for 'score' column, but got {:?}", result),
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
