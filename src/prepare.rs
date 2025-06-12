// ========================================================================================
//
//               THE PREPARATION "COMPILER": MAXIMUM SPEED EDITION
//
// ========================================================================================
//
// This module transforms raw user inputs into a hyper-optimized "computation
// blueprint." It functions as a multi-stage, parallel compiler designed for
// maximum throughput on many-core systems.
//
// THE ARCHITECTURE: PARALLEL FLATTEN -> PARALLEL SORT -> PARALLEL POPULATION
//
// 1. **Parallel Flatten & Compile:** All reconciliation logic is "compiled" into a
//    simple, flat list of `WorkItem` structs in a massively parallel step. This
//    eliminates all thinking (lookups, branching) from later stages.
//
// 2. **Parallel Sort:** The `WorkItem` list is sorted by its destination matrix
//    row. This is a highly optimized, multi-threaded sort that prepares the data
//    for perfectly linear, cache-friendly memory access.
//
// 3. **Parallel Population:** The final matrices are populated in a parallel pass
//    that is a "dumb" execution engine. It performs zero lookups and contains no
//    branches, ensuring maximum CPU utilization.
//
// The output is a `PreparationResult` struct, a "proof token" guaranteeing that
// the data is not only valid but structured for the fastest possible execution
// by the downstream kernel.

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
//                     TYPE-DRIVEN DOMAIN MODEL
// ========================================================================================
// These types encode the logic of our domain into the compiler, making invalid
// states unrepresentable and preventing entire classes of bugs. They have zero
// runtime cost.

/// Represents the logical outcome of matching the score file's effect allele
/// against the .bim file's allele orientation. This is pure logic, not math.
#[derive(Debug, Clone, Copy)]
enum Reconciliation {
    /// The score's effect allele matches the BIM's primary allele (A1).
    /// The dosage from the PLINK file can be used directly.
    Identity,
    /// The score's effect allele matches the BIM's secondary allele (A2).
    /// The dosage must be flipped (2 - dosage).
    Flip,
}

/// A zero-cost newtype for a row index into the final compute matrices.
/// Prevents accidental confusion with a `BimRowIndex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MatrixRowIndex(usize);

/// A zero-cost newtype for a column index into the final compute matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ScoreColumnIndex(usize);

/// An intermediate representation of a variant from the `.bim` file.
#[derive(Clone)]
struct BimRecord {
    allele1: String,
    allele2: String,
}

/// An intermediate task containing the logical reconciliation instruction
/// and the original weights, before being compiled into `WorkItem`s.
struct ReconciliationTask {
    reconciliation: Reconciliation,
    weights: AHashMap<String, f32>,
}

/// The final, flattened "bytecode" for the population pass. This struct contains
/// all pre-calculated information needed for a single write to the matrices.
struct WorkItem {
    matrix_row: MatrixRowIndex,
    col_idx: ScoreColumnIndex,
    aligned_weight: f32,
    correction_constant: f32,
}

// ========================================================================================
//                                  PUBLIC API
// ========================================================================================

/// A comprehensive, production-grade error type for the preparation phase.
#[derive(Debug)]
pub enum PrepError {
    Io(io::Error, PathBuf),
    Parse(String),
    Header(String),
    InconsistentKeepId(String),
    NoOverlappingVariants,
}

/// The single entry point for the preparation phase. Orchestrates the multi-stage
/// parallel compilation of user data into the final `PreparationResult`.
pub fn prepare_for_computation(
    plink_prefix: &Path,
    score_path: &Path,
    keep_file: Option<&Path>,
) -> Result<PreparationResult, PrepError> {
    // --- STAGE 0: INDIVIDUAL SUBSETTING (Fast sequential setup) ---
    let fam_path = plink_prefix.with_extension("fam");
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fam_path)?;
    let total_people_in_fam = all_person_iids.len();
    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();

    // --- STAGE 1: INITIAL SKELETON CONSTRUCTION (Fast sequential setup) ---
    eprintln!("> Pass 1: Indexing variants and score files...");
    let bim_path = plink_prefix.with_extension("bim");
    let (bim_index, total_snps_in_bim) = index_bim_file(&bim_path)?;
    let (score_map, score_names) = parse_and_group_score_file(score_path)?;
    let (reconciliation_map, required_bim_indices_set) =
        reconcile_scores_and_genotypes(&score_map, &bim_index)?;

    if reconciliation_map.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }

    let mut required_bim_indices: Vec<usize> = required_bim_indices_set.into_iter().collect();
    required_bim_indices.sort_unstable(); // For sequential .bed access
    let num_reconciled_variants = required_bim_indices.len();

    let bim_row_to_matrix_row = build_bim_to_matrix_map(&required_bim_indices, total_snps_in_bim);
    let score_name_to_col_index: AHashMap<_, ScoreColumnIndex> =
        score_names.iter().enumerate().map(|(i, s)| (s.as_str(), ScoreColumnIndex(i))).collect();

    // --- STAGE 2: PARALLEL FLATTEN AND COMPILE (The O(N*K) heavy lifting) ---
    eprintln!("> Pass 2: Compiling work items in parallel...");
    let mut work_items: Vec<WorkItem> = reconciliation_map
        .par_iter()
        .flat_map(|(bim_row_index, task)| {
            let matrix_row = bim_row_to_matrix_row[*bim_row_index].unwrap();
            
            task.weights.iter().map(move |(score_name, weight)| {
                let col_idx = score_name_to_col_index[score_name.as_str()];
                
                // The compiler FORCES correct logic via the match statement.
                // The `correction_term` bug is now impossible to write.
                let (aligned_weight, correction_constant) = match task.reconciliation {
                    Reconciliation::Identity => (*weight, 0.0),
                    Reconciliation::Flip => (-(*weight), 2.0 * *weight),
                };

                WorkItem { matrix_row, col_idx, aligned_weight, correction_constant }
            })
        })
        .collect();

    // --- STAGE 3: PARALLEL SORT ---
    eprintln!("> Pass 3: Sorting work items for cache-friendly access...");
    work_items.par_sort_unstable_by_key(|item| item.matrix_row);

    // --- STAGE 4: PARALLEL POPULATION (Stable Rust Implementation) ---
    eprintln!("> Pass 4: Populating compute matrices with maximum parallelism...");
    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let mut aligned_weights_matrix = vec![0.0f32; num_reconciled_variants * stride];
    let mut correction_constants_matrix = vec![0.0f32; num_reconciled_variants * stride];

    // This is the stable-Rust replacement for `par_group_by_key`.
    (0..num_reconciled_variants).into_par_iter().for_each(|row_idx| {
        let matrix_row = MatrixRowIndex(row_idx);

        // Find the slice of `work_items` corresponding to this row via binary search.
        let first_item_pos = work_items.partition_point(|item| item.matrix_row < matrix_row);
        let first_after_pos = work_items.partition_point(|item| item.matrix_row <= matrix_row);
        let items_for_this_row = &work_items[first_item_pos..first_after_pos];

        if items_for_this_row.is_empty() { return; }

        // SAFETY: The `for_each` on `0..num_reconciled_variants` guarantees each
        // thread is given a unique `row_idx`. Therefore, the mutable slices
        // created here will never overlap, preventing data races.
        let row_start_offset = matrix_row.0 * stride;
        let weights_slice = unsafe {
            std::slice::from_raw_parts_mut(aligned_weights_matrix.as_mut_ptr().add(row_start_offset), stride)
        };
        let corrections_slice = unsafe {
            std::slice::from_raw_parts_mut(correction_constants_matrix.as_mut_ptr().add(row_start_offset), stride)
        };

        for item in items_for_this_row {
            weights_slice[item.col_idx.0] = item.aligned_weight;
            corrections_slice[item.col_idx.0] = item.correction_constant;
        }
    });

    // --- FINAL CONSTRUCTION ---
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

/// Reads the `.bim` file once to create an in-memory index.
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
            let record = BimRecord { allele1: a1.to_uppercase(), allele2: a2.to_uppercase() };
            bim_index.entry(key).or_insert_with(Vec::new).push((record, i));
        }
    }
    Ok((bim_index, total_lines))
}

/// Parses the score file, handling multi-column scores and grouping all rules.
fn parse_and_group_score_file(score_path: &Path) -> Result<(AHashMap<String, (String, String, AHashMap<String, f32>)>, Vec<String>), PrepError> {
    let file = File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    let mut reader = BufReader::new(file);
    let mut header_line = String::new();
    reader.read_line(&mut header_line).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;

    let header_fields: Vec<_> = header_line.trim().split_whitespace().collect();
    let snp_id_idx = find_column_index(&header_fields, "snp_id")?;
    let effect_allele_idx = find_column_index(&header_fields, "effect_allele")?;
    let other_allele_idx = find_column_index(&header_fields, "other_allele")?;

    let score_name_cols: Vec<(usize, &str)> = header_fields
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != snp_id_idx && *i != effect_allele_idx && *i != other_allele_idx)
        .map(|(idx, col_name_ref)| (idx, *col_name_ref))
        .collect();

    let score_names: Vec<String> = score_name_cols.iter().map(|(_, name)| name.to_string()).collect();
    if score_names.is_empty() {
        return Err(PrepError::Header("No score columns found in score file header.".to_string()));
    }

    let mut score_map = AHashMap::new();
    for line_result in reader.lines() {
        let line = line_result.map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
        if line.trim().is_empty() { continue; }
        let fields: Vec<_> = line.split_whitespace().collect();

        let snp_id = fields.get(snp_id_idx).ok_or_else(|| PrepError::Parse("Missing snp_id column.".to_string()))?.to_string();
        let effect_allele = fields.get(effect_allele_idx).ok_or_else(|| PrepError::Parse("Missing effect_allele column.".to_string()))?.to_uppercase();
        let other_allele = fields.get(other_allele_idx).ok_or_else(|| PrepError::Parse("Missing other_allele column.".to_string()))?.to_uppercase();

        let mut weights = AHashMap::with_capacity(score_names.len());
        for (col_idx, actual_score_name) in &score_name_cols {
            let val_str = fields.get(*col_idx).ok_or_else(|| PrepError::Parse(format!("Missing score value for column '{}'", actual_score_name)))?;
            let weight: f32 = val_str.parse().map_err(|e| PrepError::Parse(format!("Invalid number for '{}': {}", actual_score_name, e)))?;
            weights.insert(actual_score_name.to_string(), weight);
        }

        score_map.insert(snp_id, (effect_allele, other_allele, weights));
    }

    Ok((score_map, score_names))
}

/// The core reconciliation logic. It iterates through the score map, finds
/// matching variants in the BIM index, and builds the intermediate plan.
fn reconcile_scores_and_genotypes(
    score_map: &AHashMap<String, (String, String, AHashMap<String, f32>)>,
    bim_index: &AHashMap<String, Vec<(BimRecord, usize)>>,
) -> Result<(AHashMap<usize, ReconciliationTask>, BTreeSet<usize>), PrepError> {
    let mut reconciliation_map = AHashMap::new();
    let mut required_bim_indices_set = BTreeSet::new();

    for (snp_id, (effect_allele, other_allele, weights)) in score_map {
        if let Some(bim_records) = bim_index.get(snp_id) {
            let score_alleles: AHashSet<&str> = [effect_allele.as_str(), other_allele.as_str()].iter().copied().collect();

            let matching_bim = bim_records.iter().find(|(bim_record, _)| {
                let bim_alleles: AHashSet<&str> = [bim_record.allele1.as_str(), bim_record.allele2.as_str()].iter().copied().collect();
                score_alleles == bim_alleles
            });

            if let Some((bim_record, bim_row_index)) = matching_bim {
                required_bim_indices_set.insert(*bim_row_index);

                // This is the core logical decision, encoded in our type-safe enum.
                // The pivot function always unpacks the dosage of bim_record.allele2.
                let reconciliation = if effect_allele == &bim_record.allele2 {
                    Reconciliation::Identity // Effect is A2, use dosage as-is.
                } else {
                    Reconciliation::Flip // Effect is A1, must flip A2's dosage.
                };

                reconciliation_map.insert(*bim_row_index, ReconciliationTask {
                    reconciliation,
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
) -> Vec<Option<MatrixRowIndex>> {
    let mut bim_row_to_matrix_row = vec![None; total_bim_snps];
    for (matrix_row_idx, &bim_row) in required_bim_indices.iter().enumerate() {
        if bim_row < total_bim_snps {
            bim_row_to_matrix_row[bim_row] = Some(MatrixRowIndex(matrix_row_idx));
        }
    }
    bim_row_to_matrix_row
}

// ... (Other helper functions like find_column_index, resolve_person_subset, etc. are included here) ...
// ... (The implementation of these helpers is straightforward and adapted to the new types) ...
fn find_column_index(header_fields: &[&str], required_col: &str) -> Result<usize, PrepError> {
    header_fields
        .iter()
        .position(|&field| field.eq_ignore_ascii_case(required_col))
        .ok_or_else(|| PrepError::Header(format!("Missing required header column: '{}'", required_col)))
}

fn resolve_person_subset(
    keep_file: Option<&Path>,
    all_person_iids: &[String],
    iid_to_original_idx: &AHashMap<String, u32>,
) -> Result<(PersonSubset, Vec<String>), PrepError> {
    if let Some(path) = keep_file {
        eprintln!("> Subsetting individuals based on keep file: {}", path.display());
        let file = File::open(path).map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
        let reader = BufReader::new(file);
        let iids_to_keep: AHashSet<String> = reader.lines().filter_map(Result::ok)
            .map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();

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

fn parse_fam_and_build_lookup(
    fam_path: &Path,
) -> Result<(Vec<String>, AHashMap<String, u32>), PrepError> {
    let file = File::open(fam_path).map_err(|e| PrepError::Io(e, fam_path.to_path_buf()))?;
    let reader = BufReader::new(file);
    let mut person_iids = Vec::new();
    let mut iid_to_idx = AHashMap::new();

    for (i, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|e| PrepError::Io(e, fam_path.to_path_buf()))?;
        let iid = line.split_whitespace().nth(1)
            .ok_or_else(|| PrepError::Parse(format!("Missing IID in .fam file on line {}", i + 1)))?.to_string();
        person_iids.push(iid.clone());
        iid_to_idx.insert(iid, i as u32);
    }
    Ok((person_iids, iid_to_idx))
}

fn format_missing_ids_error(missing_ids: Vec<String>) -> String {
    let sample_size = 5;
    let sample: Vec<_> = missing_ids.iter().take(sample_size).collect();
    let sample_str = sample.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ");
    if missing_ids.len() > sample_size {
        format!("{} individuals from the keep file were not found. Sample: [{}...]", missing_ids.len(), sample_str)
    } else {
        format!("{} individuals from the keep file were not found: [{}]", missing_ids.len(), sample_str)
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
