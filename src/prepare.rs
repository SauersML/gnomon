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

use crate::types::{
    BimRowIndex, ReconciledVariantIndex, PersonSubset, PreparationResult, ScoreColumnIndex,
};
use ahash::{AHashMap, AHashSet};
use nonmax::NonMaxU32;
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

#[derive(Clone)]
struct BimRecord {
    allele1: String,
    allele2: String,
}

/// A temporary, internal representation of a single score definition for a
/// variant, created during the reconciliation process.
struct WorkItem {
    reconciled_variant_idx: ReconciledVariantIndex,
    col_idx: ScoreColumnIndex,
    weight: f32,
    is_flipped: bool,
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
    
    // Build the mapping from original .fam index to the final, compact output index.
    // This is essential for the variant-major compute path to work with subsets.
    let mut person_fam_to_output_idx = vec![None; total_people_in_fam];
    for (output_idx, iid) in final_person_iids.iter().enumerate() {
        if let Some(&original_fam_idx) = iid_to_original_idx.get(iid) {
            person_fam_to_output_idx[original_fam_idx as usize] = Some(output_idx as u32);
        }
    }
    
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
        for (variant_id, entries) in score_map_part.drain() {
            final_score_map.entry(variant_id).or_default().extend(entries);
        }
    }
    let score_names: Vec<String> = final_score_names.into_iter().collect();

    // --- STAGE 2: SKELETON & MAP CONSTRUCTION ---
    eprintln!("> Pass 2: Indexing genotype data...");
    let bim_path = plink_prefix.with_extension("bim");
    let (bim_index, total_variants_in_bim) = index_bim_file(&bim_path)?;
    // Create a reverse map from the original bim file row index to its variant_id string.
    // This is used for fast, user-friendly error reporting if duplicates are found.
    let bim_row_to_variant_id: AHashMap<BimRowIndex, &str> = bim_index
        .iter()
        .flat_map(|(variant_id, records)| {
            records
                .iter()
                .map(move |(_, bim_row)| (*bim_row, variant_id.as_str()))
        })
        .collect();

    // We need to know which BIM rows are used to build the dense matrix map.
    let required_bim_indices_set = discover_required_bim_indices(&final_score_map, &bim_index)?;
    if required_bim_indices_set.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }

                let mut required_bim_indices: Vec<BimRowIndex> =
                    required_bim_indices_set.into_iter().collect();
    required_bim_indices.sort_unstable();
    let num_reconciled_variants = required_bim_indices.len();

                // This map is an intermediate product that uses niche-optimization for memory efficiency.
                // It maps from the original .bim row index to a dense matrix row index.
    let bim_row_to_reconciled_variant_map =
        build_bim_to_reconciled_variant_map(&required_bim_indices, total_variants_in_bim);
    let score_name_to_col_index: AHashMap<&str, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), ScoreColumnIndex(i)))
        .collect();

    // --- STAGE 3: PARALLEL COMPILE (The O(N*K) heavy lifting) ---
    eprintln!("> Pass 3: Reconciling variants and compiling work items...");
    let mut work_items: Vec<WorkItem> = final_score_map
        .par_iter()
        .flat_map(|(variant_id, score_lines)| {
            let mut work_for_position = Vec::new();
            if let Some(bim_records) = bim_index.get(variant_id) {
                for (effect_allele, other_allele, weights) in score_lines {
                    // Call the single authoritative resolver to get the definitive list of matches.
                    let winning_matches =
                        resolve_matches_for_score_line(effect_allele, other_allele, bim_records);

                    // The `process_match` helper function is perfect for turning our definitive
                    // matches into WorkItems. We just need to loop over the resolver's results.
                    for (bim_record, bim_row_index) in winning_matches {
                        process_match(
                            bim_record,
                            *bim_row_index,
                            effect_allele,
                            weights,
                            &bim_row_to_reconciled_variant_map,
                            &score_name_to_col_index,
                            &mut work_for_position,
                        );
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
    let mut unique_pairs: Vec<(ReconciledVariantIndex, ScoreColumnIndex)> = work_items
        .par_iter()
        .map(|item| (item.reconciled_variant_idx, item.col_idx))
        .collect();
    unique_pairs.par_sort_unstable();
    unique_pairs.dedup();

    // --- Calculate total variants per score ---
    let mut score_variant_counts = vec![0u32; score_names.len()];
    for &(_, score_col) in &unique_pairs {
        score_variant_counts[score_col.0] += 1;
    }

    // --- Build the variant_to_scores_map (missingness blueprint) ---
    let mut variant_to_scores_map: Vec<Vec<ScoreColumnIndex>> =
        vec![Vec::new(); num_reconciled_variants];
    // This is a sequential grouping operation on the sorted `unique_pairs`, which is fast.
    for &(variant_row, score_col) in &unique_pairs {
        // `variant_row.0` is guaranteed to be a valid index by construction.
        variant_to_scores_map[variant_row.0 as usize].push(score_col);
    }

    // --- STAGE 5: PARALLEL SORT & VALIDATION ---
    eprintln!("> Pass 5: Sorting and validating work items...");
    // Sort by matrix row and then column index. This groups any potential duplicate
    // definitions for the same (variant, score) cell right next to each other,
    // which allows for a fast linear scan to detect them.
    work_items.par_sort_unstable_by_key(|item| (item.reconciled_variant_idx, item.col_idx));

    // After sorting, a single pass over adjacent items can efficiently detect duplicates.
    // This check ensures that every variant is defined only once per score.
    if let Some(w) = work_items.windows(2).find(|w| {
        w[0].reconciled_variant_idx == w[1].reconciled_variant_idx && w[0].col_idx == w[1].col_idx
    }) {
        let bad_item = &w[0];
        let score_name = &score_names[bad_item.col_idx.0];
        // To find the human-readable variant_id, we map from our internal dense matrix row
        // index back to the original .bim file row index, and then use the reverse
        // map to find the variant_id string.
                let bim_row = required_bim_indices[bad_item.reconciled_variant_idx.0 as usize];
        let variant_id = bim_row_to_variant_id
            .get(&bim_row)
            .unwrap_or(&"unknown variant");

        return Err(PrepError::Parse(format!(
            "Ambiguous input: The score '{}' is defined more than once for variant '{}'. Each variant must have exactly one definition per score.",
            score_name, variant_id
        )));
    }

    // --- STAGE 6: POPULATE COMPUTE STRUCTURES ---
    eprintln!("> Pass 6: Populating compute structures with maximum parallelism...");

    // The `stride` is the padded row width for each variant's data, ensuring
    // that each row can be processed with full SIMD vectors without scalar fallbacks.
    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let matrix_size = num_reconciled_variants * stride;

    let mut weights_matrix = vec![0.0f32; matrix_size];
    let mut flip_mask_matrix = vec![0u8; matrix_size];

    // Use a parallel iterator over chunks to populate the matrices contention-free.
    // Accessing disjoint row chunks via `par_chunks_mut` prevents data races.
    weights_matrix
        .par_chunks_mut(stride)
        .zip(flip_mask_matrix.par_chunks_mut(stride))
        .enumerate()
        .for_each(|(row_idx, (weight_row_slice, flip_row_slice))| {
            let reconciled_variant_idx = ReconciledVariantIndex(row_idx as u32);

            // `partition_point` is used on the sorted `work_items` to efficiently find
            // the slice of items corresponding to the current variant row. This is a key
            // algorithmic optimization that avoids repeated linear scans.
            let first_item_pos =
                work_items.partition_point(|item| item.reconciled_variant_idx < reconciled_variant_idx);
            let first_after_pos =
                work_items.partition_point(|item| item.reconciled_variant_idx <= reconciled_variant_idx);
            let items_for_this_row = &work_items[first_item_pos..first_after_pos];

            for item in items_for_this_row {
                let col = item.col_idx.0;
                weight_row_slice[col] = item.weight;
                if item.is_flipped {
                    flip_row_slice[col] = 1;
                }
            }
        });

    // --- FINAL CONSTRUCTION ---
    let bytes_per_variant = (total_people_in_fam as u64 + 3) / 4;

    // Create the reverse mapping from a compact output index back to the original .fam index.
    // This is an essential optimization for the high-performance variant-major path.
    let mut output_idx_to_fam_idx = Vec::with_capacity(num_people_to_score);
    for iid in &final_person_iids {
        // This lookup is guaranteed to succeed because final_person_iids was derived from all_person_iids.
        let original_fam_idx = *iid_to_original_idx.get(iid).unwrap();
        output_idx_to_fam_idx.push(original_fam_idx);
    }

    Ok(PreparationResult::new(
        weights_matrix,
        flip_mask_matrix,
        stride,
        required_bim_indices,
        score_names,
        score_variant_counts,
        variant_to_scores_map,
        person_subset,
        final_person_iids,
        num_people_to_score,
        total_people_in_fam,
        total_variants_in_bim,
        num_reconciled_variants,
        bytes_per_variant,
        person_fam_to_output_idx,
        output_idx_to_fam_idx,
    ))
}
// ========================================================================================
//                             PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

/// Processes a matched variant and generates all corresponding `WorkItem`s.
/// This helper function centralizes the work generation logic to avoid duplication.
#[inline(always)]
fn process_match(
    bim_record: &BimRecord,
    bim_row_index: BimRowIndex,
    effect_allele: &str,
    weights: &AHashMap<String, f32>,
    bim_row_to_reconciled_variant_map: &[Option<NonMaxU32>],
    score_name_to_col_index: &AHashMap<&str, ScoreColumnIndex>,
    work_for_position: &mut Vec<WorkItem>,
) {
    // This map uses Option<NonMaxU32> for memory efficiency (4 bytes vs 16).
    // If a mapping exists, we get the inner `u32` and construct a type-safe `ReconciledVariantIndex`.
    if let Some(non_max_index) = bim_row_to_reconciled_variant_map[bim_row_index.0 as usize] {
        let reconciled_variant_idx = ReconciledVariantIndex(non_max_index.get());
        let reconciliation = if effect_allele == &bim_record.allele2 {
            Reconciliation::Identity
        } else {
            Reconciliation::Flip
        };

        for (score_name, weight) in weights {
            if let Some(&col_idx) = score_name_to_col_index.get(score_name.as_str()) {
                let is_flipped = matches!(reconciliation, Reconciliation::Flip);
                work_for_position.push(WorkItem {
                    reconciled_variant_idx,
                    col_idx,
                    weight: *weight,
                    is_flipped,
                });
            }
        }
    }
}

fn index_bim_file(
    bim_path: &Path,
) -> Result<(AHashMap<String, Vec<(BimRecord, BimRowIndex)>>, usize), PrepError> {
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
                        .push((record, BimRowIndex(i as u32)));
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
            "Invalid header in '{}': Expected at least 4 columns (variant_id, effect_allele, other_allele, score_name), found {}",
            score_path.display(),
            header_parts.len()
        )));
    }
    if header_parts[0] != "variant_id"
        || header_parts[1] != "effect_allele"
        || header_parts[2] != "other_allele"
    {
        return Err(PrepError::Header(format!("Invalid header in '{}': Must start with 'variant_id\teffect_allele\tother_allele'.", score_path.display())));
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

        let variant_id = fields[0].to_string();
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
            .entry(variant_id)
            .or_default()
            .push((effect_allele, other_allele, weights));
    }
    Ok((score_map, score_names))
}

/// Scans the inputs to find which variants are needed, also performing ambiguity checks.
fn discover_required_bim_indices(
    score_map: &AHashMap<String, Vec<(String, String, AHashMap<String, f32>)>>,
    bim_index: &AHashMap<String, Vec<(BimRecord, BimRowIndex)>>,
) -> Result<BTreeSet<BimRowIndex>, PrepError> {
    let required_indices_results: Vec<_> = score_map
        .par_iter()
        .map(|(variant_id, score_lines)| {
            let mut set_for_variant = BTreeSet::new();
            if let Some(bim_records) = bim_index.get(variant_id) {
                for (effect_allele, other_allele, _weights) in score_lines {
                    // Call the single authoritative resolver to get the definitive list of matches.
                    let winning_matches =
                        resolve_matches_for_score_line(effect_allele, other_allele, bim_records);

                    // For each valid match returned by the resolver...
                    for (bim_record, bim_row_index) in winning_matches {
                        // Add its bim row index to the set of required indices.
                        set_for_variant.insert(*bim_row_index);

                        // The existing ambiguity warning logic is still useful to inform the user
                        // about permissive matching, but now it acts on a definitively chosen match.
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

                        if bim_alleles_set != score_alleles_set {
                            if !AMBIGUITY_WARNING_PRINTED.swap(true, Ordering::Relaxed) {
                                eprintln!(
                                    "Warning: At least one variant was automatically reconciled due to mismatched allele sets (e.g., variant '{}'). Gnomon is proceeding by matching on the effect allele.",
                                    variant_id
                                );
                            }
                        }
                    }
                }
            }
            Ok::<_, PrepError>(set_for_variant)
        })
        .collect();

    let mut final_set = BTreeSet::new();
    for result in required_indices_results {
        final_set.extend(result?);
    }
    Ok(final_set)
}

fn build_bim_to_reconciled_variant_map(
    required_bim_indices: &[BimRowIndex],
    total_variants_in_bim: usize,
) -> Vec<Option<NonMaxU32>> {
    let mut bim_row_to_reconciled_variant_map = vec![None; total_variants_in_bim];
    for (reconciled_variant_idx, &bim_row) in required_bim_indices.iter().enumerate() {
        if (bim_row.0 as usize) < total_variants_in_bim {
            // Create a NonMaxU32, which is guaranteed not to be u32::MAX. This is safe as
            // the number of variants will not approach this limit.
            let non_max_index = NonMaxU32::new(reconciled_variant_idx as u32).unwrap();
            bim_row_to_reconciled_variant_map[bim_row.0 as usize] = Some(non_max_index);
        }
    }
    bim_row_to_reconciled_variant_map
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

/// The single, authoritative resolver for matching a score line to BIM records.
///
/// This function implements the "smart" reconciliation logic:
/// 1. Prioritizes perfect matches (both alleles match). It is non-greedy and returns all perfect matches.
/// 2. If no perfect matches are found, it looks for a single, unambiguous permissive match (effect allele matches).
/// 3. If multiple permissive matches are found (and no perfect matches exist), it's considered ambiguous and returns nothing.
///
/// # Arguments
/// * `effect_allele` - The effect allele from the score file.
/// * `other_allele` - The other allele from the score file.
/// * `bim_records_for_position` - A slice of all BIM records found at this chromosomal position.
///
/// # Returns
/// A `Vec` containing references to the winning `(BimRecord, BimRowIndex)` tuples. An empty
/// vector signifies that the score line should be discarded for this position (either no match or ambiguous).
fn resolve_matches_for_score_line<'a>(
    effect_allele: &str,
    other_allele: &str,
    bim_records_for_position: &'a [(BimRecord, BimRowIndex)],
) -> Vec<&'a (BimRecord, BimRowIndex)> {
    // --- Pass 1: Greedily find all perfect matches. ---
    let perfect_matches: Vec<_> = bim_records_for_position
        .iter()
        .filter(|(bim_record, _)| {
            (&bim_record.allele1 == effect_allele && &bim_record.allele2 == other_allele) ||
            (&bim_record.allele1 == other_allele && &bim_record.allele2 == effect_allele)
        })
        .collect();

    // If we have any perfect matches, they are the winners. Return them immediately.
    if !perfect_matches.is_empty() {
        return perfect_matches;
    }

    // --- Pass 2: No perfect matches found. Evaluate permissive matches for ambiguity. ---
    let mut permissive_candidate: Option<&(BimRecord, BimRowIndex)> = None;
    let mut is_ambiguous = false;

    for bim_tuple in bim_records_for_position {
        let (bim_record, _) = bim_tuple;
        // A permissive match is when the effect allele is one of the two alleles in the BIM file.
        if bim_record.allele1 == effect_allele || bim_record.allele2 == effect_allele {
            if permissive_candidate.is_some() {
                // We have already found one permissive candidate. Finding another means it's ambiguous.
                is_ambiguous = true;
                break; // No need to check further; the ambiguity is confirmed.
            }
            permissive_candidate = Some(bim_tuple);
        }
    }

    // If it's not ambiguous and we found exactly one candidate, that's our winner.
    if !is_ambiguous {
        if let Some(candidate) = permissive_candidate {
            return vec![candidate];
        }
    }

    // Otherwise (ambiguous or no matches found), return an empty Vec to signify "discard".
    Vec::new()
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
