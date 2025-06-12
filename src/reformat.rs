// ========================================================================================
//
//               PGS CATALOG SCORE FILE DIAGNOSTICS & REFORMATTING
//
// ========================================================================================

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
    /// The file was inspected but does not match the definitive PGS Catalog format signature.
    NotPgsFormat,
    /// The file appears to be a PGS Catalog file but is missing essential columns
    /// required for a valid, unambiguous translation.
    MissingRequiredColumns(String),
}

/// Attempts to diagnose and convert a single file from PGS Catalog format to gnomon-native format.
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

    // Write the gnomon-native header. This 4-column format is critical as it
    // preserves the `other_allele` needed for unambiguous reconciliation.
    writeln!(
        writer,
        "snp_id\teffect_allele\tother_allele\t{}",
        score_name
    )?;

    // Pre-fetch column indices for maximum performance in the hot loop.
    // The `unwrap()` calls here are safe because `parse_metadata_and_header`
    // has already validated that these columns exist.
    let effect_allele_idx = *column_map.get("effect_allele").unwrap();
    let effect_weight_idx = *column_map.get("effect_weight").unwrap();

    // Prioritize the harmonized/inferred other_allele if present, as it's more likely
    // to be correct and complete. Fall back to the author-reported one.
    let other_allele_idx = *column_map
        .get("hm_inferOtherAllele")
        .or_else(|| column_map.get("other_allele"))
        .unwrap();

    // Process the rest of the file line by line, translating each valid row.
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() <= effect_allele_idx
            || fields.len() <= effect_weight_idx
            || fields.len() <= other_allele_idx
        {
            continue; // Skip malformed or short lines.
        }

        // Determine the best available SNP identifier using the strict priority logic.
        if let Some(snp_id) = determine_snp_id(&fields, &column_map) {
            let effect_allele = fields[effect_allele_idx];
            let other_allele = fields[other_allele_idx];
            let effect_weight = fields[effect_weight_idx];

            // Skip rows where essential information is missing.
            if snp_id.is_empty() || effect_allele.is_empty() || other_allele.is_empty() {
                continue;
            }
            
            writeln!(
                writer,
                "{}\t{}\t{}\t{}",
                snp_id, effect_allele, other_allele, effect_weight
            )?;
        }
    }

    writer.flush()?;
    Ok(output_path)
}

// ========================================================================================
//                           PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

/// Parses the PGS Catalog header to extract metadata and perform rigorous validation.
///
/// This function ensures the file is not only a PGS Catalog file but also contains
/// all the necessary columns to be translated into a valid gnomon-native file.
fn parse_metadata_and_header(
    reader: &mut BufReader<File>,
) -> Result<(String, HashMap<String, usize>), ReformatError> {
    let mut score_name = None;
    let mut header_line = None;
    let mut is_pgs_file = false;

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
            header_line = Some(line_buf.clone());
            break;
        }
        line_buf.clear();
    }

    if !is_pgs_file {
        return Err(ReformatError::NotPgsFormat);
    }

    let header = header_line.ok_or(ReformatError::MissingRequiredColumns(
        "No data header line found after metadata.".to_string(),
    ))?;

    let column_map: HashMap<String, usize> = header
        .trim()
        .split('\t')
        .enumerate()
        .map(|(i, name)| (name.to_string(), i))
        .collect();

    // --- RIGOROUS VALIDATION ---
    // A file is only translatable if it contains all necessary fields.
    let has_effect_allele = column_map.contains_key("effect_allele");
    let has_effect_weight = column_map.contains_key("effect_weight");

    if !has_effect_allele || !has_effect_weight {
        return Err(ReformatError::MissingRequiredColumns(
            "File must contain 'effect_allele' and 'effect_weight' columns.".to_string(),
        ));
    }

    // Must have at least one valid source for positional ID.
    let has_author_pos =
        column_map.contains_key("chr_name") && column_map.contains_key("chr_position");
    let has_harmonized_pos =
        column_map.contains_key("hm_chr") && column_map.contains_key("hm_pos");

    if !has_author_pos && !has_harmonized_pos {
        return Err(ReformatError::MissingRequiredColumns(
            "File must contain ('chr_name' and 'chr_position') or ('hm_chr' and 'hm_pos')."
                .to_string(),
        ));
    }

    // Must have at least one valid source for the other allele for unambiguous matching.
    let has_other_allele = column_map.contains_key("other_allele")
        || column_map.contains_key("hm_inferOtherAllele");
    if !has_other_allele {
        return Err(ReformatError::MissingRequiredColumns(
            "File must contain 'other_allele' or 'hm_inferOtherAllele' for unambiguous matching."
                .to_string(),
        ));
    }

    let final_score_name = score_name.unwrap_or_else(|| "PGS_SCORE".to_string());

    Ok((final_score_name, column_map))
}

/// Determines the canonical variant identifier based on a strict positional priority.
///
/// This function enforces the use of positional identifiers (`chromosome:position`),
/// which is the only format supported by the downstream reconciliation engine.
/// It explicitly ignores rsID and other non-positional identifiers.
///
/// Priority: `hm_chr:hm_pos` (harmonized) > `chr_name:chr_position` (author-reported).
fn determine_snp_id(fields: &[&str], column_map: &HashMap<String, usize>) -> Option<String> {
    // 1. Prioritize Harmonized Position (hm_chr:hm_pos) - Highest quality data.
    if let (Some(&chr_idx), Some(&pos_idx)) = (column_map.get("hm_chr"), column_map.get("hm_pos"))
    {
        if let (Some(chr), Some(pos)) = (fields.get(chr_idx), fields.get(pos_idx)) {
            if !chr.is_empty() && !pos.is_empty() {
                return Some(format!("{}:{}", chr, pos));
            }
        }
    }

    // 2. Fallback to Author-reported Position (chr_name:chr_position) - If harmonized fails.
    if let (Some(&chr_idx), Some(&pos_idx)) = (
        column_map.get("chr_name"),
        column_map.get("chr_position"),
    ) {
        if let (Some(chr), Some(pos)) = (fields.get(chr_idx), fields.get(pos_idx)) {
            if !chr.is_empty() && !pos.is_empty() {
                return Some(format!("{}:{}", chr, pos));
            }
        }
    }

    // If no valid positional identifier can be constructed, this variant cannot be used.
    None
}

/// Generates a descriptive output path for the reformatted file.
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
            ReformatError::NotPgsFormat => write!(
                f,
                "File does not appear to be in PGS Catalog format (missing '###PGS CATALOG SCORING FILE' signature)."
            ),
            ReformatError::MissingRequiredColumns(s) => {
                write!(f, "Invalid or incomplete PGS Catalog file: {}", s)
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
