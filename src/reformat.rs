// ========================================================================================
//
//               PGS CATALOG SCORE FILE DIAGNOSTICS & REFORMATTING
//
// ========================================================================================
//
// This module acts as a specialist "consultant" for handling score files. Its sole
// responsibility is to diagnose files that fail initial parsing and, if they match
// the known PGS Catalog format, convert them into the gnomon-native format.
//
// This entire module is invoked as a fallback, so that the core `prepare` engine
// remains decoupled from the complexities of foreign file formats.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// A comprehensive error type for the reformatting process.
#[derive(Debug)]
pub enum ReformatError {
    /// An error occurred during file I/O.
    Io(io::Error),
    /// The file was inspected but does not match the expected PGS Catalog format.
    NotPgsFormat,
    /// The file appears to be a PGS Catalog file but is missing essential columns.
    MissingRequiredColumns(String),
}

/// Attempts to diagnose and convert a file from PGS Catalog format to gnomon-native format.
///
/// If the input file is not a valid PGS Catalog file, it will return an error.
/// On success, it writes a new, permanent, descriptively-named file to the same
/// directory as the input and returns the path to this new file.
pub fn reformat_pgs_file(input_path: &Path) -> Result<PathBuf, ReformatError> {
    // --- Phase 1: Diagnosis, Metadata Extraction, and Header Parsing ---
    let input_file = File::open(input_path)?;
    let mut reader = BufReader::new(input_file);

    let (score_name, column_map) = parse_metadata_and_header(&mut reader)?;

    // --- Phase 2: Data Transformation and Writing ---
    let output_path = generate_output_path(input_path);
    let output_file = File::create(&output_path)?;
    let mut writer = BufWriter::new(output_file);

    // Write the gnomon-native header.
    writeln!(writer, "snp_id\teffect_allele\t{}", score_name)?;

    // Get references to required column indices for performance in the loop.
    let effect_allele_idx = *column_map.get("effect_allele").unwrap();
    let effect_weight_idx = *column_map.get("effect_weight").unwrap();

    // Process the rest of the file line by line.
    for line_result in reader.lines() {
        let line = line_result?;
        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() < column_map.len() {
            continue; // Skip malformed or short lines.
        }

        // Determine the best available SNP identifier using the priority logic.
        if let Some(snp_id) = determine_snp_id(&fields, &column_map) {
            let effect_allele = fields[effect_allele_idx];
            let effect_weight = fields[effect_weight_idx];
            writeln!(writer, "{}\t{}\t{}", snp_id, effect_allele, effect_weight)?;
        }
    }

    writer.flush()?;
    Ok(output_path)
}

// ========================================================================================
//                           PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

/// Parses the PGS Catalog header to extract metadata and column indices.
///
/// This function reads from the beginning of the file, consuming the comment lines
/// and the single data header line, leaving the reader ready to process data rows.
fn parse_metadata_and_header(
    reader: &mut BufReader<File>,
) -> Result<(String, HashMap<String, usize>), ReformatError> {
    let mut score_name = None;
    let mut header_line = None;
    let mut is_pgs_file = false;

    // A temporary buffer to avoid repeated allocations in the loop.
    let mut line_buf = String::new();

    // Scan until we find the data header line (the first non-comment line).
    while reader.read_line(&mut line_buf)? > 0 {
        if line_buf.starts_with('#') {
            // Check for the definitive PGS Catalog signature.
            if line_buf.contains("###PGS CATALOG SCORING FILE") {
                is_pgs_file = true;
            }
            // Extract the pgs_id as the preferred score name.
            if let Some(id) = line_buf.strip_prefix("#pgs_id=") {
                score_name = Some(id.trim().to_string());
            }
        } else {
            // This is the data header.
            header_line = Some(line_buf.clone());
            break;
        }
        line_buf.clear();
    }

    // --- Validation after scanning ---
    if !is_pgs_file {
        return Err(ReformatError::NotPgsFormat);
    }

    let header = header_line.ok_or(ReformatError::MissingRequiredColumns(
        "No data header found after metadata.".to_string(),
    ))?;

    let column_map: HashMap<String, usize> = header
        .trim()
        .split('\t')
        .enumerate()
        .map(|(i, name)| (name.to_string(), i))
        .collect();

    // Verify all required columns are present.
    let required_cols = [
        "effect_allele",
        "effect_weight",
        "chr_name",
        "chr_position",
    ];
    for &col in &required_cols {
        if !column_map.contains_key(col) {
            return Err(ReformatError::MissingRequiredColumns(format!(
                "Required column '{}' not found in header.",
                col
            )));
        }
    }

    // Use a default score name if pgs_id was not found.
    let final_score_name = score_name.unwrap_or_else(|| "PGS_SCORE".to_string());

    Ok((final_score_name, column_map))
}

/// Determines the best variant identifier based on a defined priority.
///
/// Priority: hm_rsID > rsID > hm_chr:hm_pos > chr_name:chr_position
fn determine_snp_id(fields: &[&str], column_map: &HashMap<String, usize>) -> Option<String> {
    // 1. Harmonized rsID (hm_rsID)
    if let Some(&idx) = column_map.get("hm_rsID") {
        if let Some(val) = fields.get(idx) {
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
    }

    // 2. Author-reported rsID
    if let Some(&idx) = column_map.get("rsID") {
        if let Some(val) = fields.get(idx) {
            if !val.is_empty() {
                return Some(val.to_string());
            }
        }
    }

    // 3. Harmonized Position (hm_chr:hm_pos)
    if let (Some(&chr_idx), Some(&pos_idx)) = (column_map.get("hm_chr"), column_map.get("hm_pos")) {
        if let (Some(chr), Some(pos)) = (fields.get(chr_idx), fields.get(pos_idx)) {
            if !chr.is_empty() && !pos.is_empty() {
                // Do not prepend "chr". The canonical ID is based on the raw chromosome and position.
                return Some(format!("{}:{}", chr, pos));
            }
        }
    }

    // 4. Author-reported Position (chr_name:chr_position)
    if let (Some(&chr_idx), Some(&pos_idx)) = (
        column_map.get("chr_name"),
        column_map.get("chr_position"),
    ) {
        if let (Some(chr), Some(pos)) = (fields.get(chr_idx), fields.get(pos_idx)) {
            if !chr.is_empty() && !pos.is_empty() {
                // Do not prepend "chr". The canonical ID is based on the raw chromosome and position.
                return Some(format!("{}:{}", chr, pos));
            }
        }
    }

    None // No valid identifier could be constructed.
}

/// Generates a descriptive output path for the reformatted file.
///
/// Example: `/path/to/PGS000123.txt` -> `/path/to/PGS000123.gnomon_format.tsv`
fn generate_output_path(input_path: &Path) -> PathBuf {
    let stem = input_path.file_stem().unwrap_or_default();
    let new_filename = format!("{}.gnomon_format.tsv", stem.to_string_lossy());
    input_path.with_file_name(new_filename)
}

// ========================================================================================
//                                    ERROR HANDLING
// ========================================================================================

impl Display for ReformatError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ReformatError::Io(e) => write!(f, "I/O Error during reformatting: {}", e),
            ReformatError::NotPgsFormat => {
                write!(f, "File does not appear to be in PGS Catalog format.")
            }
            ReformatError::MissingRequiredColumns(s) => {
                write!(f, "Invalid PGS Catalog file format: {}", s)
            }
        }
    }
}

impl Error for ReformatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReformatError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ReformatError {
    fn from(err: io::Error) -> Self {
        ReformatError::Io(err)
    }
}
