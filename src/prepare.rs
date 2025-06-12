// ========================================================================================
//
//                       THE PREPARATION "COMPILER"
//
// ========================================================================================
//
// This module transforms raw user inputs into a hyper-optimized "computation
// blueprint." It functions as a two-pass compiler:
//
// 1. **Skeleton Pass:** It reads all genotype and score metadata to discover the
//    universe of variants, validates them for consistency, and defines the final
//    memory layout of the compute matrices. This pass fixes data corruption bugs
//    and enables robust support for complex variants.
//
// 2. **Population Pass:** It uses the memory layout from the first pass to populate
//    the final data matrices in a highly parallel, contention-free manner.
//
// The output is a `PreparationResult` struct, which is a "proof token" that the
// downstream compute engine can execute without further validation.

use crate::types::{PersonSubset, PreparationResult};
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};

// The number of SIMD lanes in the kernel. This MUST be kept in sync with kernel.rs.
const LANE_COUNT: usize = 8;

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
    /// One or more individual IDs from the keep file were not found in the .fam file.
    InconsistentKeepId(String),
    /// No variants were found that existed in both the score file and genotype data.
    NoOverlappingVariants,
}

/// The single entry point for the preparation phase. Orchestrates the two-pass
/// compilation of user data into the final `PreparationResult`.
pub fn prepare_for_computation(
    plink_prefix: &Path,
    score_path: &Path,
    keep_file: Option<&Path>,
) -> Result<PreparationResult, PrepError> {
    // --- Phase 0: Individual Subsetting ---
    let fam_path = plink_prefix.with_extension("fam");
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fam_path)?;
    let total_people_in_fam = all_person_iids.len();
    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();

    // --- Pass 1: Skeleton Construction ---
    eprintln!("> Pass 1: Indexing variants and defining memory layout...");
    let bim_path = plink_prefix.with_extension("bim");
    let (bim_index, total_snps_in_bim) = index_bim_file(&bim_path)?;
    let (score_map, score_names) = parse_and_group_score_file(score_path)?;
    let (reconciliation_map, required_bim_indices_set) =
        reconcile_scores_and_genotypes(&score_map, &bim_index)?;

    if reconciliation_map.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }

    // Finalize the memory layout by creating the definitive sorted index lists and lookup maps.
    let mut required_bim_indices: Vec<usize> = required_bim_indices_set.into_iter().collect();
    required_bim_indices.sort_unstable(); // Ensure sorted order for sequential .bed access
    let num_reconciled_variants = required_bim_indices.len();
    let bim_row_to_matrix_row = build_bim_to_matrix_map(&required_bim_indices, total_snps_in_bim);
    let score_name_to_col_index: AHashMap<_, _> =
        score_names.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

    // --- Pass 2: Parallel Payload Population ---
    eprintln!("> Pass 2: Populating compute matrices in parallel...");
    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let mut aligned_weights_matrix = vec![0.0f32; num_reconciled_variants * stride];
    let mut correction_constants_matrix = vec![0.0f32; num_reconciled_variants * stride];

    // --- FIX: Get raw pointers BEFORE the closure ---
    let weights_ptr = aligned_weights_matrix.as_mut_ptr();
    let corrections_ptr = correction_constants_matrix.as_mut_ptr();

    // Parallel population pass: Iterate over the sorted `required_bim_indices`.
    // This gives both the dense `matrix_row` (from enumerate) and the sparse `bim_row_index`.
    required_bim_indices
        .par_iter()
        .enumerate()
        .for_each(move |(matrix_row, &bim_row_index)| { // Add `move` here
            // The task is now retrieved via a direct lookup using the bim_row_index.
            // It's expected that every bim_row_index in required_bim_indices has a task in reconciliation_map.
            let task = &reconciliation_map[&bim_row_index];

            for (score_name, weight) in &task.weights { // Added '&' before task.weights
                let col_idx = score_name_to_col_index[score_name.as_str()];
                let matrix_offset = matrix_row * stride + col_idx;

                // This is the core mathematical transformation.
                let aligned_weight = *weight * task.sign; // Dereference weight
                let correction_constant = if task.sign < 0.0 {
                    2.0 * *weight // Correction is 2*W only on a flip; Dereference weight
                } else {
                    0.0
                };

                // SAFETY: Writes are to disjoint `matrix_row` sections.
                // Each (matrix_row, col_idx) is unique per written element.
                // `aligned_weights_matrix` and `correction_constants_matrix` are sized
                // num_reconciled_variants * stride, matrix_row < num_reconciled_variants,
                // col_idx < score_names.len() <= stride. So, matrix_offset is in bounds.
                // --- FIX: Use pointer arithmetic for the write ---
                unsafe {
                    // This is now safe because each thread writes to a disjoint memory region.
                    *weights_ptr.add(matrix_offset) = aligned_weight;
                    *corrections_ptr.add(matrix_offset) = correction_constant;
                }
            }
        });

    // --- Phase 3: Construct and return the validated "proof token" ---
    let bytes_per_snp = (total_people_in_fam as u64 + 3) / 4;

    Ok(PreparationResult {
        aligned_weights_matrix,
        correction_constants_matrix,
        stride,
        bim_row_to_matrix_row,
        required_bim_indices,
        score_names,
        person_subset,
        final_person_iids,
        num_people_to_score,
        total_people_in_fam,
        total_snps_in_bim,
        num_reconciled_variants,
        bytes_per_snp,
    })
}

// ========================================================================================
//                             PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

/// An intermediate representation of a variant from the `.bim` file.
#[derive(Clone)]
struct BimRecord {
    allele1: String,
    allele2: String,
}

/// The result of reconciling a variant between the score file and the `.bim` file.
/// This contains the pre-calculated terms needed for the final computation.
struct ReconciliationTask {
    /// A value of 1.0 or -1.0, determined by whether the score's effect allele
    /// matches the BIM file's first allele.
    sign: f32,
    /// The map of score names to their raw effect weights.
    weights: AHashMap<String, f32>,
}


/// Pass 1, Step 1: Reads the `.bim` file once to create an in-memory index
/// mapping a `chr:pos` key to all variants at that locus.
fn index_bim_file(bim_path: &Path) -> Result<(AHashMap<String, Vec<(BimRecord, usize)>>, usize), PrepError> {
    let file = File::open(bim_path).map_err(|e| PrepError::Io(e, bim_path.to_path_buf()))?;
    let reader = BufReader::new(file);
    let mut bim_index = AHashMap::new();
    let mut total_lines = 0;

    for (i, line_result) in reader.lines().enumerate() {
        total_lines += 1;
        let line = line_result.map_err(|e| PrepError::Io(e, bim_path.to_path_buf()))?;
        let mut parts = line.split_whitespace();
        let chr = parts.next();
        let _id = parts.next();
        let _cm = parts.next();
        let pos = parts.next();
        let a1 = parts.next();
        let a2 = parts.next();

        if let (Some(chr), Some(pos), Some(a1), Some(a2)) = (chr, pos, a1, a2) {
            let key = format!("{}:{}", chr, pos);
            // ACTION: Ensure alleles are uppercased here
            let record = BimRecord { allele1: a1.to_uppercase(), allele2: a2.to_uppercase() };
            bim_index.entry(key).or_insert_with(Vec::new).push((record, i));
        }
    }
    Ok((bim_index, total_lines))
}


/// Pass 1, Step 2: Parses the score file, handling multi-column scores and
/// grouping all rules by variant `chr:pos`.
fn parse_and_group_score_file(score_path: &Path) -> Result<(AHashMap<String, (String, String, AHashMap<String, f32>)>, Vec<String>), PrepError> {
    let file = File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    let mut reader = BufReader::new(file);
    let mut header_line = String::new();
    reader.read_line(&mut header_line).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;

    let header_fields: Vec<_> = header_line.trim().split_whitespace().collect();
    let snp_id_idx = find_column_index(&header_fields, "snp_id")?;
    let effect_allele_idx = find_column_index(&header_fields, "effect_allele")?;
    // The "other_allele" is now a required column for robust indel/multiallelic matching.
    let other_allele_idx = find_column_index(&header_fields, "other_allele")?;

    // Store column index and score name string directly
    let score_name_cols: Vec<(usize, &str)> = header_fields
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != snp_id_idx && *i != effect_allele_idx && *i != other_allele_idx)
        .map(|(idx, col_name_ref)| (idx, *col_name_ref))
        .collect();

    let score_names: Vec<String> = score_name_cols.iter().map(|(_, name)| name.to_string()).collect();
    if score_names.is_empty() {
        return Err(PrepError::Header("No score columns found in score file header. Expected columns like 'snp_id', 'effect_allele', 'other_allele', and one or more score weights.".to_string()));
    }

    let mut score_map = AHashMap::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
        if line.trim().is_empty() { continue; }
        let fields: Vec<_> = line.split_whitespace().collect();

        let snp_id = fields.get(snp_id_idx).ok_or_else(|| PrepError::Parse("Missing snp_id column in data row.".to_string()))?.to_string();
        // ACTION: Ensure alleles are uppercased here
        let effect_allele = fields.get(effect_allele_idx).ok_or_else(|| PrepError::Parse("Missing effect_allele column in data row.".to_string()))?.to_uppercase();
        let other_allele = fields.get(other_allele_idx).ok_or_else(|| PrepError::Parse("Missing other_allele column in data row.".to_string()))?.to_uppercase();

        let mut weights = AHashMap::with_capacity(score_names.len());
        // Iterate over the collected (column_index, score_name_str) pairs
        for (col_idx, actual_score_name) in &score_name_cols {
            // Dereference col_idx to use as usize for .get()
            let val_str:&str = fields.get(*col_idx).ok_or_else(|| PrepError::Parse(format!("Missing score value for column '{}' in snp '{}'", actual_score_name, snp_id)))?;
            let weight: f32 = val_str.parse().map_err(|_| PrepError::Parse(format!("Invalid numeric value '{}' for score '{}' in snp '{}'", val_str, actual_score_name, snp_id)))?;
            weights.insert(actual_score_name.to_string(), weight);
        }

        score_map.insert(snp_id, (effect_allele, other_allele, weights));
    }

    Ok((score_map, score_names))
}


/// Pass 1, Step 3: The core reconciliation logic. It iterates through the score map,
/// finds matching variants in the BIM index, validates them, and builds the
/// final plan for populating the matrices.
fn reconcile_scores_and_genotypes(
    score_map: &AHashMap<String, (String, String, AHashMap<String, f32>)>,
    bim_index: &AHashMap<String, Vec<(BimRecord, usize)>>,
) -> Result<(AHashMap<usize, ReconciliationTask>, BTreeSet<usize>), PrepError> {
    let mut reconciliation_map = AHashMap::new();
    let mut required_bim_indices_set = BTreeSet::new();

    for (snp_id, (effect_allele, other_allele, weights)) in score_map {
        if let Some(bim_records) = bim_index.get(snp_id) {
            // Alleles from score file are already uppercased. BIM alleles are also uppercased.
            let score_alleles: AHashSet<&str> =
                [effect_allele.as_str(), other_allele.as_str()].iter().copied().collect();

            let matching_bim = bim_records.iter().find(|(bim_record, _)| {
                // Collect bim_alleles as AHashSet<&str> using .as_str()
                let bim_alleles: AHashSet<&str> =
                    [bim_record.allele1.as_str(), bim_record.allele2.as_str()].iter().copied().collect();
                score_alleles == bim_alleles // Case-sensitive match is fine now due to normalization
            });

            if let Some((bim_record, bim_row_index)) = matching_bim {
                required_bim_indices_set.insert(*bim_row_index);

                // --- FIX: Invert the sign logic to match reality ---
                // The pivot_tile function unpacks the dosage of `allele2`.
                // Therefore, the math must be aligned to that reality.
                let sign = if effect_allele == &bim_record.allele2 {
                    1.0 // Correct: Effect allele IS allele2, use dosage as-is.
                } else {
                    -1.0 // Correct: Effect allele is allele1, so we must flip the A2 dosage.
                };

                reconciliation_map.insert(*bim_row_index, ReconciliationTask {
                    sign,
                    weights: weights.clone(),
                });
            }
        }
    }
    Ok((reconciliation_map, required_bim_indices_set))
}

/// Builds the final O(1) lookup map from original BIM row index to final matrix row index.
fn build_bim_to_matrix_map(
    required_bim_indices: &[usize],
    total_bim_snps: usize,
) -> Vec<Option<usize>> {
    let mut bim_row_to_matrix_row = vec![None; total_bim_snps];
    for (matrix_row, &bim_row) in required_bim_indices.iter().enumerate() {
        if bim_row < total_bim_snps {
            bim_row_to_matrix_row[bim_row] = Some(matrix_row);
        }
    }
    bim_row_to_matrix_row
}


/// Finds the 0-based index of a required column name in a header. Case-insensitive.
fn find_column_index(header_fields: &[&str], required_col: &str) -> Result<usize, PrepError> {
    header_fields
        .iter()
        .position(|&field| field.eq_ignore_ascii_case(required_col))
        .ok_or_else(|| PrepError::Header(format!("Missing required header column: '{}'", required_col)))
}


/// Determines the exact subset of individuals to score based on the optional keep file.
fn resolve_person_subset(
    keep_file: Option<&Path>,
    all_person_iids: &[String],
    iid_to_original_idx: &AHashMap<String, u32>,
) -> Result<(PersonSubset, Vec<String>), PrepError> {
    if let Some(path) = keep_file {
        eprintln!("> Subsetting individuals based on keep file: {}", path.display());
        let file = File::open(path).map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
        let reader = BufReader::new(file);
        let iids_to_keep: AHashSet<String> = reader
            .lines()
            .filter_map(Result::ok)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut found_people = Vec::with_capacity(iids_to_keep.len());
        let mut missing_ids = Vec::new();

        for iid in iids_to_keep {
            if let Some(&original_idx) = iid_to_original_idx.get(&iid) {
                found_people.push((original_idx, iid));
            } else {
                missing_ids.push(iid);
            }
        }

        if !missing_ids.is_empty() {
            return Err(PrepError::InconsistentKeepId(format_missing_ids_error(missing_ids)));
        }

        found_people.sort_unstable_by_key(|(idx, _)| *idx);
        let final_person_iids = found_people.iter().map(|(_, iid)| iid.clone()).collect();
        let subset_indices = found_people.into_iter().map(|(idx, _)| idx).collect();
        Ok((PersonSubset::Indices(subset_indices), final_person_iids))
    } else {
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
        let iid = line.split_whitespace().nth(1).ok_or_else(|| PrepError::Parse(format!("Missing IID in .fam file on line {}", i + 1)))?.to_string();
        person_iids.push(iid.clone());
        iid_to_idx.insert(iid, i as u32);
    }
    Ok((person_iids, iid_to_idx))
}

/// Creates a user-friendly, actionable error message from a list of missing IDs.
fn format_missing_ids_error(missing_ids: Vec<String>) -> String {
    let sample_size = 5;
    let sample: Vec<_> = missing_ids.iter().take(sample_size).collect();
    let sample_str = sample.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
    if missing_ids.len() > sample_size {
        format!("{} individuals from the keep file were not found in the .fam file. Sample of missing IDs: [{}...]", missing_ids.len(), sample_str)
    } else {
        format!("{} individuals from the keep file were not found in the .fam file: [{}]", missing_ids.len(), sample_str)
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
            PrepError::InconsistentKeepId(s) => write!(f, "Configuration Error: {}", s),
            PrepError::NoOverlappingVariants => write!(f, "No overlapping variants found between the score file and the genotype data."),
        }
    }
}

impl Error for PrepError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PrepError::Io(e, _) => Some(e),
            _ => None,
        }
    }
}

impl From<ParseFloatError> for PrepError {
    fn from(err: ParseFloatError) -> Self {
        PrepError::Parse(format!("Could not parse numeric value: {}", err))
    }
}
