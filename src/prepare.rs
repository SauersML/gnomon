// ========================================================================================
//
//          THE CONFIGURATION, VALIDATION, AND PREPARATION ENGINE
//
// ========================================================================================
//
// This module serves as the application's immutable configuration and validation
// layer. Its mission is to transform raw, untrusted user inputs into a single,
// cohesive, and scientifically correct "computation blueprint."
//
// This blueprint, the `PreparationResult` struct, is a proof token guaranteeing that
// all data has been loaded, validated, reconciled against scientific parameters,
// and structured for the extreme performance requirements of the Staged
// Block-Pivoting Engine.

use crate::types::{PersonSubset, PreparationResult, Reconciliation};
use ahash::{AHashMap, AHashSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};

// ========================================================================================
//                                  PUBLIC API
// ========================================================================================

/// A comprehensive, production-grade error type for the preparation phase.
#[derive(Debug)]
pub enum PrepError {
    /// An error occurred during file I/O, with the associated file path.
    Io(io::Error, PathBuf),
    /// An error occurred parsing a text file (e.g., malformed lines).
    Parse(String),
    /// The header of a file was invalid or missing required columns.
    Header(String),
    /// The number of scores for a SNP did not match the header.
    InconsistentScores(String),
    /// An error occurred parsing a floating-point number.
    ParseFloat(ParseFloatError),
    /// One or more individual IDs from the keep file were not found in the .fam file.
    InconsistentKeepId(String),
}

/// The single entry point for the preparation phase. It orchestrates file parsing,
/// individual subsetting, scientific reconciliation, sorting, and data structuring.
pub fn prepare_for_computation(
    plink_prefix: &Path,
    score_path: &Path,
    keep_file: Option<&Path>,
) -> Result<PreparationResult, PrepError> {
    // Phase 1: Establish the universe of all individuals.
    let fam_path = plink_prefix.with_extension("fam");
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fam_path)?;
    let total_people_in_fam = all_person_iids.len();

    // Phase 2: Resolve the target individual subset.
    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();

    // Phase 3: Parse score file and reconcile with genotype data.
    let bim_path = plink_prefix.with_extension("bim");
    let (score_names, flat_weights, mut score_map, score_file_skipped_snps) =
        parse_scores_and_flatten(score_path)?;
    let (mut tasks, total_snps_in_bim, bim_file_skipped_snps) =
        build_reconciliation_plan(&bim_path, &mut score_map)?;

    // Report summary of skipped SNPs.
    let total_skipped = score_file_skipped_snps + bim_file_skipped_snps;
    if total_skipped > 0 {
        eprintln!(
            "\nInfo: A total of {} SNPs were skipped due to data inconsistencies. See warnings above for details.",
            total_skipped
        );
    }

    let num_reconciled_snps = tasks.len();
    if num_reconciled_snps == 0 {
        return Err(PrepError::Parse(
            "No overlapping SNPs found between the score file and the genotype data.".to_string(),
        ));
    }

    // Phase 4: Sort by .bed file order for sequential access.
    tasks.sort_unstable_by_key(|task| task.bed_row_index);

    // Phase 5: Finalize data structures for the engine.
    let (interleaved_weights, required_snp_indices, reconciliation_instructions) =
        finalize_from_sorted_plan(tasks, &flat_weights, score_names.len());

    // Phase 6: Construct and return the validated "proof token".
    Ok(PreparationResult {
        interleaved_weights,
        required_snp_indices,
        reconciliation_instructions,
        person_subset,
        score_names,
        final_person_iids,
        total_people_in_fam,
        num_people_to_score,
        num_reconciled_snps,
        total_snps_in_bim,
    })
}

// ========================================================================================
//                             PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

/// A temporary data structure holding the plan for a single SNP.
struct ReconciliationTask {
    bed_row_index: usize,
    instruction: Reconciliation,
    weights_start_index: usize,
}

/// Determines the exact subset of individuals to score based on the optional keep file.
/// This uses an efficient O(N_keep) algorithm.
fn resolve_person_subset(
    keep_file: Option<&Path>,
    all_person_iids: &[String],
    iid_to_original_idx: &AHashMap<String, u32>,
) -> Result<(PersonSubset, Vec<String>), PrepError> {
    if let Some(path) = keep_file {
        eprintln!("> Subsetting individuals based on keep file: {}", path.display());
        let iids_to_keep = read_iids_to_set(path)?;

        let mut found_people = Vec::with_capacity(iids_to_keep.len());
        let mut missing_ids = Vec::new();

        // O(N_keep) ALGORITHM: Iterate the smaller set.
        for iid in iids_to_keep {
            if let Some(&original_idx) = iid_to_original_idx.get(&iid) {
                found_people.push((original_idx, iid));
            } else {
                missing_ids.push(iid);
            }
        }

        // Actionable Error Handling
        if !missing_ids.is_empty() {
            return Err(PrepError::InconsistentKeepId(format_missing_ids_error(
                missing_ids,
            )));
        }
        
        // Sort by original index to maintain a stable, deterministic order.
        found_people.sort_unstable_by_key(|(idx, _)| *idx);

        let final_person_iids = found_people.iter().map(|(_, iid)| iid.clone()).collect();
        let subset_indices = found_people.into_iter().map(|(idx, _)| idx).collect();

        Ok((PersonSubset::Indices(subset_indices), final_person_iids))
    } else {
        // If no keep file, we process everyone.
        Ok((PersonSubset::All, all_person_iids.to_vec()))
    }
}

/// Parses a .fam file to produce both an ordered list of all individual IDs and
/// a fast lookup map from IID to its original 0-based index.
fn parse_fam_and_build_lookup(
    fam_path: &Path,
) -> Result<(Vec<String>, AHashMap<String, u32>), PrepError> {
    let file = File::open(fam_path).map_err(|e| PrepError::Io(e, fam_path.to_path_buf()))?;
    let reader = BufReader::new(file);
    let mut person_iids = Vec::new();
    let mut iid_to_idx = AHashMap::new();

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| PrepError::Io(e, fam_path.to_path_buf()))?;
        let mut parts = line.split_whitespace();
        let _fid = parts.next(); // FID is not used.
        let iid = parts
            .next()
            .ok_or_else(|| PrepError::Parse(format!("Missing IID in .fam file on line {}", i + 1)))?
            .to_string();

        person_iids.push(iid.clone());
        iid_to_idx.insert(iid, i as u32);
    }
    Ok((person_iids, iid_to_idx))
}

/// Efficiently reads a file of line-separated IDs into a HashSet for fast lookups.
fn read_iids_to_set(path: &Path) -> Result<AHashSet<String>, PrepError> {
    let file = File::open(path).map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
    let reader = BufReader::new(file);
    let mut set = AHashSet::new();
    for line_result in reader.lines() {
        let line = line_result.map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            set.insert(trimmed.to_string());
        }
    }
    Ok(set)
}

/// Parses the score file robustly, finding columns by name rather than fixed order.
fn parse_scores_and_flatten(
    score_path: &Path,
) -> Result<(Vec<String>, Vec<f32>, AHashMap<String, (usize, u8)>, usize), PrepError> {
    let file = File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    let mut reader = BufReader::new(file);
    let mut header_line = String::new();
    reader
        .read_line(&mut header_line)
        .map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;

    // Flexible Header Parsing
    let header_fields: Vec<_> = header_line.trim().split_whitespace().collect();
    let required_cols = ["snp_id", "effect_allele"];
    let col_indices = find_column_indices(&header_fields, &required_cols)?;
    let snp_id_idx = col_indices[0];
    let effect_allele_idx = col_indices[1];

    let score_names: Vec<String> = header_fields
        .iter()
        .enumerate()
        .filter(|(i, _)| !col_indices.contains(i))
        .map(|(_, &name)| name.to_string())
        .collect();
        
    if score_names.is_empty() {
        return Err(PrepError::Header(
            "No score columns found in score file header.".to_string(),
        ));
    }
    let num_scores = score_names.len();

    let mut flat_weights = Vec::new();
    let mut score_map = AHashMap::new();
    let mut skipped_snp_count = 0;

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.is_empty() { continue; }

        let snp_id = fields.get(snp_id_idx).ok_or_else(|| PrepError::Parse("Missing snp_id column in data row.".to_string()))?;
        let effect_allele_str = fields.get(effect_allele_idx).ok_or_else(|| PrepError::Parse("Missing effect_allele column in data row.".to_string()))?;

        if effect_allele_str.len() != 1 {
            eprintln!("Warning: Skipping SNP '{}' due to invalid allele format (must be a single character).", snp_id);
            skipped_snp_count += 1;
            continue;
        }
        let effect_allele_u8 = effect_allele_str.as_bytes()[0];
        let weights_start_index = flat_weights.len();
        
        let mut weights_count = 0;
        for (i, &field) in fields.iter().enumerate() {
            if !col_indices.contains(&i) {
                weights_count += 1;
                flat_weights.push(field.parse()?);
            }
        }
        
        if weights_count != num_scores {
             return Err(PrepError::InconsistentScores(format!(
                "SNP '{}' has {} weights, but header implies {}.",
                snp_id, weights_count, num_scores
            )));
        }

        score_map.insert(
            snp_id.to_string(),
            (weights_start_index, effect_allele_u8),
        );
    }
    Ok((score_names, flat_weights, score_map, skipped_snp_count))
}

/// Reads the .bim file, compares against the score map, and builds a list of tasks.
/// This function's internal logic is preserved as it was already robust.
fn build_reconciliation_plan(
    bim_path: &Path,
    score_map: &mut AHashMap<String, (usize, u8)>,
) -> Result<(Vec<ReconciliationTask>, usize, usize), PrepError> {
    let file = File::open(bim_path).map_err(|e| PrepError::Io(e, bim_path.to_path_buf()))?;
    let reader = BufReader::new(file);

    let mut tasks = Vec::with_capacity(score_map.len());
    let mut total_bim_snps = 0;
    let mut skipped_snp_count = 0;

    for (line_number, line_result) in reader.lines().enumerate() {
        total_bim_snps += 1;
        let line = line_result.map_err(|e| PrepError::Io(e, bim_path.to_path_buf()))?;
        let mut parts = line.split_whitespace();

        let chromosome = parts.next();
        let snp_id_col2 = parts.next(); // The original ID from column 2, used for warnings.
        let _cm_pos = parts.next();
        let bp_pos = parts.next();
        let allele1_str = parts.next();
        let allele2_str = parts.next();

        if let (Some(chr), Some(id_col2), Some(pos), Some(a1_str), Some(a2_str)) =
            (chromosome, snp_id_col2, bp_pos, allele1_str, allele2_str)
        {
            // The canonical key for lookup is always chr:pos.
            let canonical_id = format!("{}:{}", chr, pos);

            // Attempt to find the variant in the score map using the canonical chr:pos ID.
            if let Some(&(weights_start_index, effect_allele_u8)) = score_map.get(&canonical_id) {
                // This variant exists in the score file. Now check its allele structure.
                if a1_str.len() == 1 && a2_str.len() == 1 {
                    // It's a valid biallelic SNP. Proceed with reconciliation.
                    let bim_a1 = a1_str.as_bytes()[0];
                    let bim_a2 = a2_str.as_bytes()[0];

                    let instruction = if effect_allele_u8 == bim_a2 {
                        Reconciliation::Identity
                    } else if effect_allele_u8 == bim_a1 {
                        Reconciliation::Flip
                    } else {
                        // Alleles in score file don't match alleles in .bim file for this chr:pos.
                        eprintln!("Warning: Skipping SNP '{}' (at {}:{}) due to allele mismatch (effect: '{}', bim: '{}/{}').", id_col2, chr, pos, effect_allele_u8 as char, bim_a1 as char, bim_a2 as char);
                        skipped_snp_count += 1;
                        score_map.remove(&canonical_id); // Consume the SNP to prevent other warnings.
                        continue;
                    };

                    tasks.push(ReconciliationTask {
                        bed_row_index: line_number,
                        instruction,
                        weights_start_index,
                    });
                    
                    // The SNP has been successfully processed, remove it from the map.
                    score_map.remove(&canonical_id);

                } else {
                    // The variant was found by chr:pos, but its alleles are not simple (e.g., an indel).
                    eprintln!("Warning: Skipping SNP '{}' in .bim file due to invalid allele format (alleles: '{}'/'{}').", id_col2, a1_str, a2_str);
                    skipped_snp_count += 1;
                    score_map.remove(&canonical_id); // Consume the SNP.
                }
            }
            // If the canonical_id is not in the score_map, we silently ignore it, as it's not relevant.
        } else {
            return Err(PrepError::Parse(format!(
                "Malformed line #{} in .bim file: {}",
                line_number + 1,
                line
            )));
        }
    }
    Ok((tasks, total_bim_snps, skipped_snp_count))
}

/// This function's internal logic is preserved as it was already correct.
fn finalize_from_sorted_plan(
    tasks: Vec<ReconciliationTask>,
    flat_weights: &[f32],
    num_scores: usize,
) -> (Vec<f32>, Vec<usize>, Vec<Reconciliation>) {
    let num_reconciled_snps = tasks.len();
    let mut interleaved_weights = vec![0.0f32; num_reconciled_snps * num_scores];
    let mut required_snp_indices = Vec::with_capacity(num_reconciled_snps);
    let mut reconciliation_instructions = Vec::with_capacity(num_reconciled_snps);

    for (snp_i, task) in tasks.into_iter().enumerate() {
        required_snp_indices.push(task.bed_row_index);
        reconciliation_instructions.push(task.instruction);
        let source_slice =
            &flat_weights[task.weights_start_index..task.weights_start_index + num_scores];
        let dest_start = snp_i * num_scores;
        let dest_slice = &mut interleaved_weights[dest_start..dest_start + num_scores];
        dest_slice.copy_from_slice(source_slice);
    }
    (
        interleaved_weights,
        required_snp_indices,
        reconciliation_instructions,
    )
}

/// Finds the 0-based indices of required column names in a header. Case-insensitive.
fn find_column_indices(header_fields: &[&str], required_cols: &[&str]) -> Result<Vec<usize>, PrepError> {
    let mut indices = Vec::with_capacity(required_cols.len());
    for &required_col in required_cols {
        let found_index = header_fields
            .iter()
            .position(|&field| field.eq_ignore_ascii_case(required_col))
            .ok_or_else(|| PrepError::Header(format!("Missing required header column: '{}'", required_col)))?;
        indices.push(found_index);
    }
    Ok(indices)
}

/// Creates a user-friendly, actionable error message from a list of missing IDs.
fn format_missing_ids_error(missing_ids: Vec<String>) -> String {
    let sample_size = 5;
    let sample: Vec<_> = missing_ids.iter().take(sample_size).collect();
    let sample_str = sample
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ");

    if missing_ids.len() > sample_size {
        format!(
            "{} individuals from the keep file were not found in the .fam file. Sample of missing IDs: [{}...]",
            missing_ids.len(),
            sample_str
        )
    } else {
        format!(
            "{} individuals from the keep file were not found in the .fam file: [{}]",
            missing_ids.len(),
            sample_str
        )
    }
}


// ========================================================================================
//                                    ERROR HANDLING
// ========================================================================================

impl Display for PrepError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            PrepError::Io(e, path) => write!(f, "I/O Error for file '{}': {}", path.display(), e),
            PrepError::Parse(s) => write!(f, "Parse Error: {}", s),
            PrepError::Header(s) => write!(f, "Invalid Header: {}", s),
            PrepError::InconsistentScores(s) => write!(f, "Inconsistent Data: {}", s),
            PrepError::ParseFloat(e) => write!(f, "Numeric Parse Error: {}", e),
            PrepError::InconsistentKeepId(s) => write!(f, "Configuration Error: {}", s),
        }
    }
}

impl Error for PrepError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PrepError::Io(e, _) => Some(e),
            PrepError::ParseFloat(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ParseFloatError> for PrepError {
    fn from(err: ParseFloatError) -> Self {
        PrepError::ParseFloat(err)
    }
}
