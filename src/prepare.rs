// ========================================================================================
//
//               THE PREPARATION "COMPILER"
//
// ========================================================================================
//
// This module transforms raw user inputs into an optimized "computation
// blueprint." It functions as a multi-stage, parallel compiler designed for
// maximum throughput on many-core systems. It correctly handles and merges
// multiple score files.

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
use std::sync::atomic::{AtomicBool, Ordering};

// A flag to ensure we only print the allele-reconciliation warning once per run.
static AMBIGUITY_WARNING_PRINTED: AtomicBool = AtomicBool::new(false);

// The number of SIMD lanes in the kernel. This MUST be kept in sync with kernel.rs.
const LANE_COUNT: usize = 8;

// ========================================================================================
//                     TYPE-DRIVEN DOMAIN MODEL
// ========================================================================================

#[derive(Debug, Clone, Copy)]
enum Reconciliation {
    Flip,
    Identity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MatrixRowIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ScoreColumnIndex(usize);

#[derive(Clone)]
struct BimRecord {
    allele1: String,
    allele2: String,
}

struct WorkItem {
    matrix_row: MatrixRowIndex,
    col_idx: ScoreColumnIndex,
    aligned_weight: f32,
    correction_constant: f32,
}

// ========================================================================================
//                                  PUBLIC API
// ========================================================================================

#[derive(Debug)]
pub enum PrepError {
    Io(io::Error, PathBuf),
    Parse(String),
    Header(String),
    InconsistentKeepId(String),
    NoOverlappingVariants,
}

pub fn prepare_for_computation(
    plink_prefix: &Path,
    score_files: &[PathBuf],
    keep_file: Option<&Path>,
) -> Result<PreparationResult, PrepError> {
    // --- STAGE 0: SEQUENTIAL SETUP ---
    let fam_path = plink_prefix.with_extension("fam");
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fam_path)?;
    let total_people_in_fam = all_person_iids.len();
    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();

    // --- STAGE 1: PARALLEL PARSE & MERGE SCORE FILES ---
    eprintln!(
        "> Pass 1: Parsing and merging {} score file(s) in parallel...",
        score_files.len()
    );
    let parsed_data = score_files
        .par_iter()
        .map(|path| parse_and_group_score_file(path))
        .collect::<Result<Vec<_>, _>>()?;

    let mut final_score_map: AHashMap<String, Vec<(String, String, AHashMap<String, f32>)>> =
        AHashMap::new();
    let mut final_score_names = BTreeSet::new(); // BTreeSet gives a stable, sorted order.

    for (mut score_map_part, score_names_part) in parsed_data {
        for name in score_names_part {
            final_score_names.insert(name);
        }
        for (snp_id, entries) in score_map_part.drain() {
            final_score_map.entry(snp_id).or_default().extend(entries);
        }
    }
    let score_names: Vec<String> = final_score_names.into_iter().collect();

    // --- STAGE 2: SKELETON & MAP CONSTRUCTION ---
    eprintln!("> Pass 2: Indexing genotype data...");
    let bim_path = plink_prefix.with_extension("bim");
    let (bim_index, total_snps_in_bim) = index_bim_file(&bim_path)?;
    // Create a reverse map from the original bim file row index to its snp_id string.
    // This is used for fast, user-friendly error reporting if duplicates are found.
    let bim_row_to_snp_id: AHashMap<usize, &str> = bim_index
        .iter()
        .flat_map(|(snp_id, records)| {
            records
                .iter()
                .map(move |(_, bim_row)| (*bim_row, snp_id.as_str()))
        })
        .collect();

    // We need to know which BIM rows are used to build the dense matrix map.
    let required_bim_indices_set = discover_required_bim_indices(&final_score_map, &bim_index)?;
    if required_bim_indices_set.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }

    let mut required_bim_indices: Vec<usize> = required_bim_indices_set.into_iter().collect();
    required_bim_indices.sort_unstable();
    let num_reconciled_variants = required_bim_indices.len();

    let bim_row_to_matrix_row_typed =
        build_bim_to_matrix_map(&required_bim_indices, total_snps_in_bim);
    let score_name_to_col_index: AHashMap<&str, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), ScoreColumnIndex(i)))
        .collect();

    // --- STAGE 3: PARALLEL COMPILE (The O(N*K) heavy lifting) ---
    eprintln!("> Pass 3: Reconciling variants and compiling work items...");
    let mut work_items: Vec<WorkItem> = final_score_map
        .par_iter()
        .flat_map(|(snp_id, score_lines)| {
            let mut work_for_position = Vec::new();
            if let Some(bim_records) = bim_index.get(snp_id) {
                for (effect_allele, _other_allele, weights) in score_lines {
                    // This two-pass logic ensures that we prioritize a perfect allele match before
                    // falling back to a more permissive match on only the effect allele. This
                    // resolves ambiguity when multiple variants exist at the same position.
                    // Pass 1: Look for a perfect match on both alleles.
                    let perfect_match = bim_records.iter().find(|(bim_record, _)| {
                        (&bim_record.allele1 == effect_allele && &bim_record.allele2 == other_allele) ||
                        (&bim_record.allele1 == other_allele && &bim_record.allele2 == effect_allele)
                    });

                    // Use the perfect match if it exists. Otherwise, fall back to the old permissive logic.
                    let usable_match = perfect_match.or_else(|| {
                        // Pass 2: Permissive match (only if no perfect match was found).
                        bim_records.iter().find(|(bim_record, _)| {
                            effect_allele == &bim_record.allele1 || effect_allele == &bim_record.allele2
                        })
                    });

                    if let Some((bim_record, bim_row_index)) = usable_match {
                        // The `bim_row_to_matrix_row_typed` lookup should always succeed here,
                        // because the `discover_required_bim_indices` pass has already
                        // identified this `bim_row_index` as being required.
                        if let Some(matrix_row) = bim_row_to_matrix_row_typed[*bim_row_index] {
                            // The dosage extracted by the kernel is always for `allele2` in the BIM file.
                            // Therefore, if the score's effect allele is `allele2`, the weight is used as-is.
                            // If the score's effect allele is `allele1`, we must flip the sign of the weight
                            // and add a correction constant.
                            let reconciliation = if effect_allele == &bim_record.allele2 {
                                Reconciliation::Identity
                            } else {
                                Reconciliation::Flip
                            };

                            // Iterate over the weights for *this specific score line* and generate
                            // a WorkItem for each. No data is ever overwritten.
                            for (score_name, weight) in weights {
                                if let Some(&col_idx) =
                                    score_name_to_col_index.get(score_name.as_str())
                                {
                                    let (aligned_weight, correction_constant) =
                                        match reconciliation {
                                            Reconciliation::Identity => (*weight, 0.0),
                                            Reconciliation::Flip => (-(*weight), 2.0 * *weight),
                                        };
                                    work_for_position.push(WorkItem {
                                        matrix_row,
                                        col_idx,
                                        aligned_weight,
                                        correction_constant,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            work_for_position
        })
        .collect();

    // --- STAGE 4: COMPILE FINAL DATA STRUCTURES FROM WORK ITEMS ---
    eprintln!("> Pass 4: Compiling final data structures for kernel...");

    // To correctly count variants per score and build the missingness map, we need
    // a unique set of (variant, score) pairings.
    let mut unique_pairs: Vec<(MatrixRowIndex, ScoreColumnIndex)> = work_items
        .par_iter()
        .map(|item| (item.matrix_row, item.col_idx))
        .collect();
    unique_pairs.par_sort_unstable();
    unique_pairs.dedup();

    // --- Calculate total variants per score ---
    let mut score_variant_counts = vec![0u32; score_names.len()];
    for &(_, score_col) in &unique_pairs {
        score_variant_counts[score_col.0] += 1;
    }

    // --- Build the variant_to_scores_map (missingness blueprint) ---
    let mut variant_to_scores_map: Vec<Vec<u16>> = vec![Vec::new(); num_reconciled_variants];
    // This is a sequential grouping operation on the sorted `unique_pairs`, which is fast.
    for &(variant_row, score_col) in &unique_pairs {
        // `variant_row.0` is guaranteed to be a valid index by construction.
        variant_to_scores_map[variant_row.0].push(score_col.0 as u16);
    }

    // --- Build the sparse variant_to_corrections_map ---
    // This map is critical for efficiently correcting scores when a flipped variant is missing.
    let mut variant_to_corrections_map: Vec<Vec<(u16, f32)>> = vec![vec![]; num_reconciled_variants];
    // We iterate over the sorted `work_items` to build the map efficiently.
    work_items
        .iter()
        // We only care about items that have a non-zero correction constant.
        .filter(|item| item.correction_constant != 0.0)
        .for_each(|item| {
            // `item.matrix_row.0` is guaranteed to be a valid index.
            variant_to_corrections_map[item.matrix_row.0]
                .push((item.col_idx.0 as u16, item.correction_constant));
        });

// --- STAGE 5: PARALLEL SORT & VALIDATION ---
    eprintln!("> Pass 5: Sorting and validating work items...");
    // Sort by matrix row and then column index. This groups any potential duplicate
    // definitions for the same (variant, score) cell right next to each other,
    // which allows for a fast linear scan to detect them.
    work_items.par_sort_unstable_by_key(|item| (item.matrix_row, item.col_idx));

    // After sorting, a single pass over adjacent items can efficiently detect duplicates.
    // This check ensures that every variant is defined only once per score.
    if let Some(w) = work_items.windows(2).find(|w| {
        w[0].matrix_row == w[1].matrix_row && w[0].col_idx == w[1].col_idx
    }) {
        let bad_item = &w[0];
        let score_name = &score_names[bad_item.col_idx.0];
        // To find the human-readable snp_id, we map from our internal dense matrix row
        // index back to the original .bim file row index, and then use the reverse
        // map to find the snp_id string.
        let bim_row = required_bim_indices[bad_item.matrix_row.0];
        let snp_id = bim_row_to_snp_id
            .get(&bim_row)
            .unwrap_or(&"unknown variant");

        return Err(PrepError::Parse(format!(
            "Ambiguous input: The score '{}' is defined more than once for variant '{}'. Each variant must have exactly one definition per score.",
            score_name, snp_id
        )));
    }

    // --- STAGE 6: POPULATE COMPUTE STRUCTURES ---
    eprintln!("> Pass 6: Populating compute structures with maximum parallelism...");

    // First, calculate the dosage-independent base scores by summing all correction
    // constants from the entire set of work items. This is a fast, sequential pass.
    let mut base_scores = vec![0.0f32; score_names.len()];
    for item in &work_items {
        base_scores[item.col_idx.0] += item.correction_constant;
    }

    // Second, populate the aligned weights matrix in parallel. This matrix only
    // contains the dosage-dependent component of the score calculation.
    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let mut aligned_weights_matrix = vec![0.0f32; num_reconciled_variants * stride];

    aligned_weights_matrix
        .par_chunks_mut(stride)
        .enumerate()
        .for_each(|(row_idx, weights_slice)| {
            let matrix_row = MatrixRowIndex(row_idx);
            let first_item_pos = work_items.partition_point(|item| item.matrix_row < matrix_row);
            let first_after_pos = work_items.partition_point(|item| item.matrix_row <= matrix_row);
            let items_for_this_row = &work_items[first_item_pos..first_after_pos];

            for item in items_for_this_row {
                weights_slice[item.col_idx.0] += item.aligned_weight;
            }
        });

    // --- FINAL CONSTRUCTION ---
    let bim_row_to_matrix_row: Vec<Option<usize>> = bim_row_to_matrix_row_typed
        .into_iter()
        .map(|opt_idx| opt_idx.map(|idx| idx.0))
        .collect();
    let bytes_per_snp = (total_people_in_fam as u64 + 3) / 4;
    Ok(PreparationResult {
        aligned_weights_matrix,
        base_scores,
        stride,
        bim_row_to_matrix_row,
        required_bim_indices,
        score_names,
        score_variant_counts,
        variant_to_scores_map,
        variant_to_corrections_map,
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

fn index_bim_file(
    bim_path: &Path,
) -> Result<(AHashMap<String, Vec<(BimRecord, usize)>>, usize), PrepError> {
    let file = File::open(bim_path).map_err(|e| PrepError::Io(e, bim_path.to_path_buf()))?;
    let reader = BufReader::new(file);
    let mut bim_index = AHashMap::new();
    let mut total_lines = 0;
    for (i, line_result) in reader.lines().enumerate() {
        total_lines += 1;
        let line = line_result.map_err(|e| PrepError::Io(e, bim_path.to_path_buf()))?;
        let mut parts = line.split_whitespace();
        let chr = parts.next();
        let _id = parts.next(); // rsID is ignored
        let _cm = parts.next();
        let pos = parts.next();
        let a1 = parts.next();
        let a2 = parts.next();

        if let (Some(chr), Some(pos), Some(a1), Some(a2)) = (chr, pos, a1, a2) {
            let key = format!("{}:{}", chr, pos);
            let record = BimRecord {
                allele1: a1.to_string(),
                allele2: a2.to_string(),
            };
            bim_index
                .entry(key)
                .or_insert_with(Vec::new)
                .push((record, i));
        }
    }
    Ok((bim_index, total_lines))
}

/// Parses a single score file in the strict gnomon-native format.
fn parse_and_group_score_file(
    score_path: &Path,
) -> Result<
    (
        AHashMap<String, Vec<(String, String, AHashMap<String, f32>)>>,
        Vec<String>,
    ),
    PrepError,
> {
    let file =
        File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    let mut reader = BufReader::new(file);
    let mut header_line = String::new();

    loop {
        header_line.clear();
        if reader
            .read_line(&mut header_line)
            .map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?
            == 0
        {
            return Err(PrepError::Header(
                "Score file is empty or contains only metadata.".to_string(),
            ));
        }
        if !header_line.starts_with('#') {
            break;
        }
    }

    let header_parts: Vec<&str> = header_line.trim().split('\t').collect();
    if header_parts.len() < 4 {
        return Err(PrepError::Header(format!(
            "Invalid header in '{}': Expected at least 4 columns (snp_id, effect_allele, other_allele, score_name), found {}",
            score_path.display(),
            header_parts.len()
        )));
    }
    if header_parts[0] != "snp_id"
        || header_parts[1] != "effect_allele"
        || header_parts[2] != "other_allele"
    {
        return Err(PrepError::Header(format!("Invalid header in '{}': Must start with 'snp_id\teffect_allele\tother_allele'.", score_path.display())));
    }

    let score_names: Vec<String> = header_parts[3..].iter().map(|s| s.to_string()).collect();
    let mut score_map: AHashMap<String, Vec<(String, String, AHashMap<String, f32>)>> =
        AHashMap::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != header_parts.len() {
            continue; // Skip malformed lines
        }

        let snp_id = fields[0].to_string();
        let effect_allele = fields[1].to_string();
        let other_allele = fields[2].to_string();

        let mut weights = AHashMap::with_capacity(score_names.len());
        for (i, name) in score_names.iter().enumerate() {
            let weight_str = fields[i + 3];
            if !weight_str.is_empty() {
                let weight: f32 = weight_str.parse().map_err(|e| {
                    PrepError::Parse(format!(
                        "Invalid number for score '{}' in file '{}': {}",
                        name,
                        score_path.display(),
                        e
                    ))
                })?;
                weights.insert(name.clone(), weight);
            }
        }
        score_map
            .entry(snp_id)
            .or_default()
            .push((effect_allele, other_allele, weights));
    }
    Ok((score_map, score_names))
}

/// Scans the inputs to find which variants are needed, also performing ambiguity checks.
fn discover_required_bim_indices(
    score_map: &AHashMap<String, Vec<(String, String, AHashMap<String, f32>)>>,
    bim_index: &AHashMap<String, Vec<(BimRecord, usize)>>,
) -> Result<BTreeSet<usize>, PrepError> {
    let required_indices_results: Vec<_> = score_map
        .par_iter()
        .map(|(snp_id, score_lines)| {
            let mut set_for_snp = BTreeSet::new();
            if let Some(bim_records) = bim_index.get(snp_id) {
                for (effect_allele, other_allele, _weights) in score_lines {
                    // This intelligent two-pass search ensures we only mark a variant as "required"
                    // if it is the best possible match, resolving ambiguity correctly from the start.
                    // Pass 1: Look for a perfect match on both alleles.
                    let perfect_match = bim_records.iter().find(|(bim_record, _)| {
                        (&bim_record.allele1 == effect_allele && &bim_record.allele2 == other_allele) ||
                        (&bim_record.allele1 == other_allele && &bim_record.allele2 == effect_allele)
                    });

                    // Use the perfect match if it exists. Otherwise, fall back to permissive logic.
                    let usable_match = perfect_match.or_else(|| {
                        // Pass 2: Permissive match (only if no perfect match was found).
                        bim_records.iter().find(|(bim_record, _)| {
                            effect_allele == &bim_record.allele1 || effect_allele == &bim_record.allele2
                        })
                    });

                    if let Some((bim_record, bim_row_index)) = usable_match {
                        // The variant is usable. Add its bim row index to the set of required indices.
                        set_for_snp.insert(*bim_row_index);

                        let bim_alleles_set: AHashSet<&str> =
                            [bim_record.allele1.as_str(), bim_record.allele2.as_str()]
                                .iter()
                                .copied()
                                .collect();
                        let score_alleles_set: AHashSet<&str> =
                            [effect_allele.as_str(), other_allele.as_str()]
                                .iter()
                                .copied()
                                .collect();

                        // If the allele sets are non-identical, this is an "imperfect" match.
                        // We print a single, global warning to inform the user that this automatic
                        // reconciliation is happening.
                        if bim_alleles_set != score_alleles_set {
                            if !AMBIGUITY_WARNING_PRINTED.swap(true, Ordering::Relaxed) {
                                eprintln!(
                                    "Warning: At least one variant was automatically reconciled due to mismatched allele sets (e.g., variant '{}'). Gnomon is proceeding by matching on the effect allele.",
                                    snp_id
                                );
                            }
                        }
                    }
                }
            }
            Ok::<_, PrepError>(set_for_snp)
        })
        .collect();

    let mut final_set = BTreeSet::new();
    for result in required_indices_results {
        final_set.extend(result?);
    }
    Ok(final_set)
}

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
            return Err(PrepError::InconsistentKeepId(format_missing_ids_error(
                missing_ids,
            )));
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
        let iid = line
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| PrepError::Parse(format!("Missing IID in .fam file on line {}", i + 1)))?
            .to_string();
        person_iids.push(iid.clone());
        iid_to_idx.insert(iid, i as u32);
    }
    Ok((person_iids, iid_to_idx))
}

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
            "{} individuals from keep file not found. Sample: [{}...]",
            missing_ids.len(),
            sample_str
        )
    } else {
        format!(
            "{} individuals from keep file not found: [{}]",
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
            PrepError::InconsistentKeepId(s) => write!(f, "Configuration Error: {}", s),
            PrepError::NoOverlappingVariants => write!(
                f,
                "No overlapping variants found between score file(s) and genotype data."
            ),
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
