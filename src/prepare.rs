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
use std::time::Instant;

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
    AmbiguousReconciliation(String),
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
    let parse_start = Instant::now();
    let parsed_data = score_files
        .par_iter()
        .map(|path| {
            let file_start = Instant::now();
            let result = parse_and_group_score_file(path);
            let file_time = file_start.elapsed();
            eprintln!("  > TIMING: Parsing '{}' took {:.2?}", path.display(), file_time);
            result
        })
        .collect::<Result<Vec<_>, _>>()?;
    eprintln!("> TIMING: Parallel parsing of all {} files took {:.2?}", score_files.len(), parse_start.elapsed());

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
    let bim_start = Instant::now();
    let (bim_index, total_variants_in_bim) = index_bim_file(&bim_path)?;
    eprintln!("> TIMING: BIM file indexing took {:.2?} for {} variants", bim_start.elapsed(), total_variants_in_bim);
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
    let reconcile_start = Instant::now();
    let work_items_results: Vec<Result<Vec<WorkItem>, PrepError>> = final_score_map
        .par_iter()
        .map(|(variant_id, score_lines)| {
            let mut work_for_position = Vec::new();
            if let Some(bim_records) = bim_index.get(variant_id) {
                for (effect_allele, other_allele, weights) in score_lines {
                    // Call the single authoritative resolver to get the definitive list of matches.
                    // This can now return a fatal error if ambiguity is detected.
                    let winning_matches =
                        resolve_matches_for_score_line(variant_id, effect_allele, other_allele, bim_records)?;

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
            Ok(work_for_position)
        })
        .collect();

    let mut work_items: Vec<WorkItem> = Vec::new();
    for result in work_items_results {
        work_items.extend(result?);
    }

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
    eprintln!("> TIMING: Pass 3 reconciliation generated {} work items in {:.2?}", work_items.len(), reconcile_start.elapsed());
    eprintln!("  > About to sort {} work items", work_items.len());
    let sort_start = Instant::now();
    // Sort by matrix row and then column index. This groups any potential duplicate
    // definitions for the same (variant, score) cell right next to each other,
    // which allows for a fast linear scan to detect them.
    // The (u32, usize) key is packed into a single u64 for faster sort comparisons.
    // The row index gets the high bits, ensuring it's the primary sort key.
    work_items.par_sort_unstable_by_key(|item| {
        ((item.reconciled_variant_idx.0 as u64) << 32) | (item.col_idx.0 as u64)
    });
    eprintln!("  > TIMING: Sort took {:.2?}", sort_start.elapsed());

    // After sorting, a single pass over adjacent items can efficiently detect duplicates.
    // This check ensures that every variant is defined only once per score.
    if let Some(w) = work_items.windows(2).find(|w| {
        let key1 =
            ((w[0].reconciled_variant_idx.0 as u64) << 32) | (w[0].col_idx.0 as u64);
        let key2 =
            ((w[1].reconciled_variant_idx.0 as u64) << 32) | (w[1].col_idx.0 as u64);
        key1 == key2
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
    let matrix_pop_start = Instant::now();
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
    eprintln!("> TIMING: Pass 6 matrix population took {:.2?}", matrix_pop_start.elapsed());

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
    let io_start = Instant::now();
    let file =
        File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    eprintln!("    > File '{}' size: {} MB", score_path.display(), file_size / 1_000_000);
    
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
    let n_score_columns = score_names.len();
    eprintln!("    > Found {} score columns in file", n_score_columns);
    
    let mut score_map: AHashMap<String, Vec<(String, String, AHashMap<String, f32>)>> =
        AHashMap::new();

    let parse_start = Instant::now();
    let mut line_count = 0;
    let mut total_floats_parsed = 0;
    
    for line_result in reader.lines() {
        line_count += 1;
        if line_count % 50000 == 0 {
            eprintln!("      ... parsed {} lines", line_count);
        }
        
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
                total_floats_parsed += 1;
            }
        }
        score_map
            .entry(variant_id)
            .or_default()
            .push((effect_allele, other_allele, weights));
    }
    
    let io_time = io_start.elapsed();
    let parse_time = parse_start.elapsed();
    eprintln!("    > TIMING: File I/O setup took {:.2?}", io_time - parse_time);
    eprintln!("    > TIMING: Parsing {} lines with {} floats took {:.2?}", 
              line_count, total_floats_parsed, parse_time);
    eprintln!("    > TIMING: Total file processing took {:.2?}", io_time);
    
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
                        resolve_matches_for_score_line(variant_id, effect_allele, other_allele, bim_records)?;

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

/// Parses only the headers of multiple score files to quickly build a complete,
/// sorted, and unique list of all score columns across all files.
///
/// # Arguments
///
/// * `score_files`: A slice of paths to the gnomon-native score files.
///
/// # Returns
///
/// A `Result` containing a `Vec<String>` of all unique score names found,
/// sorted alphabetically. If any file cannot be read or has an invalid
/// header, a `PrepError` is returned.
pub fn parse_score_file_headers_only(
    score_files: &[PathBuf],
) -> Result<Vec<String>, PrepError> {
    // Use a parallel iterator over the input files.
    // `try_flat_map` will process each file and flatten the resulting lists of
    // score names into a single parallel iterator.
    // `collect` into a Result<BTreeSet, ...> handles both error propagation
    // and the merging of results into a single, unique, sorted collection.
    let all_score_names: BTreeSet<String> = score_files
        .par_iter()
        .try_flat_map(|path| -> Result<Vec<String>, PrepError> {
            // This closure is executed in parallel for each score file.
            let file = File::open(path).map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
            let mut reader = BufReader::new(file);
            let mut header_line = String::new();

            // Scan past any metadata/comment lines at the beginning of the file.
            loop {
                header_line.clear();
                // Check for I/O errors or an empty file.
                if reader
                    .read_line(&mut header_line)
                    .map_err(|e| PrepError::Io(e, path.to_path_buf()))?
                    == 0
                {
                    // If we reach EOF without finding a non-comment line, the file
                    // is considered invalid as it lacks a header.
                    return Err(PrepError::Header(format!(
                        "Score file '{}' is empty or contains only metadata lines.",
                        path.display()
                    )));
                }

                // If the line is not a comment, we've found the header.
                if !header_line.starts_with('#') {
                    break;
                }
            }

            // --- Validate the header structure ---
            let header_parts: Vec<&str> = header_line.trim().split('\t').collect();
            let expected_prefix = &["variant_id", "effect_allele", "other_allele"];

            if header_parts.len() < 3 || &header_parts[0..3] != expected_prefix {
                return Err(PrepError::Header(format!(
                    "Invalid header in '{}': Must start with 'variant_id\\teffect_allele\\tother_allele'.",
                    path.display()
                )));
            }

            // --- Extract score names and return them ---
            // The score names are all columns after the first three.
            // Convert from &str to String for the final collection.
            let score_names = header_parts[3..]
                .iter()
                .map(|s| s.to_string())
                .collect();

            Ok(score_names)
        })
        .collect::<Result<BTreeSet<String>, _>>()?;

    // Convert the final, sorted BTreeSet into a Vec.
    Ok(all_score_names.into_iter().collect())
}

fn compile_score_file_to_map(
    score_path: &Path,
    bim_index: &AHashMap<String, Vec<(BimRecord, BimRowIndex)>>,
    bim_row_to_reconciled_variant_map: &[Option<NonMaxU32>],
    score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
) -> Result<AHashMap<u64, (f32, bool)>, PrepError> {
    // --- Phase 1: Memory Map and Header Parsing ---
    let file = File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    // SAFETY: The file is read-only, which is safe for memory mapping.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    mmap.advise(memmap2::Advice::Sequential)
        .map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;

    let mut line_iterator = mmap.split(|&b| b == b'\n');

    // Find the header line and the start of the data section.
    let mut data_start_offset = 0;
    let header_line = loop {
        match line_iterator.next() {
            Some(line) => {
                data_start_offset += line.len() + 1; // +1 for the newline
                if !line.starts_with(b"#") && !line.is_empty() {
                    break line;
                }
            }
            None => return Ok(AHashMap::new()), // File is empty or only has comments
        }
    };

    let header_str = std::str::from_utf8(header_line)
        .map_err(|_| PrepError::Header("Header contains invalid UTF-8".to_string()))?;
    
    // Create a map from this file's column index (e.g., column 4) to the global score index.
    let file_specific_column_map: Vec<ScoreColumnIndex> = header_str
        .trim()
        .split('\t')
        .skip(3) // Skip variant_id, effect_allele, other_allele
        .map(|name| {
            score_name_to_col_index.get(name)
                .copied() // Use copied() to get ScoreColumnIndex, not &ScoreColumnIndex
                .ok_or_else(|| PrepError::Header(format!("Score '{}' from file '{}' was not found in the global score list.", name, score_path.display())))
        })
        .collect::<Result<_, _>>()?;


    // --- Phase 2: Parallel Chunk Processing ---
    
    /// Helper to divide a byte slice into chunks ending at newlines.
    fn find_chunk_boundaries(data: &[u8], target_chunk_size: usize) -> Vec<&[u8]> {
        if data.is_empty() {
            return Vec::new();
        }
        let mut boundaries = Vec::new();
        let mut current_pos = 0;
        while current_pos < data.len() {
            let end = (current_pos + target_chunk_size).min(data.len());
            // If we're at the end of the file, the chunk is just the rest.
            if end == data.len() {
                boundaries.push(&data[current_pos..end]);
                break;
            }
            // Find the next newline *after* our target end point.
            let next_newline = memchr::memchr(b'\n', &data[end..]).map_or(data.len(), |i| end + i);
            boundaries.push(&data[current_pos..next_newline]);
            current_pos = next_newline + 1;
        }
        boundaries
    }

    let data_slice = &mmap[data_start_offset..];
    let chunks = find_chunk_boundaries(data_slice, 16 * 1024 * 1024); // 16 MB chunks

    chunks
        .into_par_iter()
        .map(|chunk| -> Result<AHashMap<u64, (f32, bool)>, PrepError> {
            let mut local_map = AHashMap::new();

            for line_bytes in chunk.split(|&b| b == b'\n') {
                if line_bytes.is_empty() { continue; }

                let mut fields = line_bytes.split(|&b| b == b'\t');
                let variant_id_bytes = fields.next().unwrap_or_default();
                let effect_allele_bytes = fields.next().unwrap_or_default();
                let other_allele_bytes = fields.next().unwrap_or_default();
                
                // Fields are only converted to &str when needed for a lookup.
                let variant_id = std::str::from_utf8(variant_id_bytes).map_err(|_| PrepError::Parse("Invalid UTF-8 in variant_id".to_string()))?;
                if let Some(bim_records) = bim_index.get(variant_id) {
                    let effect_allele = std::str::from_utf8(effect_allele_bytes).map_err(|_| PrepError::Parse("Invalid UTF-8 in effect_allele".to_string()))?;
                    let other_allele = std::str::from_utf8(other_allele_bytes).map_err(|_| PrepError::Parse("Invalid UTF-8 in other_allele".to_string()))?;
                    
                    let winning_matches = resolve_matches_for_score_line(effect_allele, other_allele, bim_records);

                    for (bim_record, bim_row_index) in winning_matches {
                        if let Some(non_max_idx) = bim_row_to_reconciled_variant_map[bim_row_index.0 as usize] {
                            let reconciled_variant_idx = non_max_idx.get();
                            let is_flipped = effect_allele != &bim_record.allele2;

                            for (i, weight_bytes) in fields.enumerate() {
                                if !weight_bytes.is_empty() {
                                    if let Ok(weight) = lexical_core::parse::<f32>(weight_bytes) {
                                        let score_col_idx = file_specific_column_map[i];
                                        let key = ((reconciled_variant_idx as u64) << 32) | (score_col_idx.0 as u64);

                                        if local_map.insert(key, (weight, is_flipped)).is_some() {
                                            return Err(PrepError::Parse(format!(
                                                "Ambiguous input: The score '{}' is defined more than once for variant '{}' within the same file '{}'.",
                                                header_str.split('\t').nth(3 + i).unwrap_or("?"), variant_id, score_path.display()
                                            )));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(local_map)
        })
        .try_reduce(AHashMap::new, |mut acc_map, local_map| {
            for (key, value) in local_map {
                if acc_map.insert(key, value).is_some() {
                    // This error is technically for duplicates across chunks, but from the user's
                    // perspective, it's still a duplicate within the same file.
                    return Err(PrepError::Parse(format!(
                        "Ambiguous input: A variant is defined for the same score multiple times in file '{}'.",
                        score_path.display()
                    )));
                }
            }
            Ok(acc_map)
        })
}

/// The single, authoritative resolver for matching a score line to BIM records.
///
/// This function implements the "smart" reconciliation logic with a "fail-fast" philosophy:
/// 1. Prioritizes perfect matches (both alleles match). It returns all perfect matches found.
/// 2. If no perfect matches are found, it looks for a single, unambiguous permissive match (where only the effect allele matches).
/// 3. If multiple permissive matches are found, the situation is considered a critical data ambiguity, and the program will
///    terminate with a detailed error. This is a deliberate design choice to prevent silent data-dropping.
///
/// # Rationale for Failing Fast
/// The original behavior was to silently discard a variant if its effect allele matched multiple different genotype records
/// (e.g., effect allele 'A' matching both 'A/T' and 'A/C' records at the same position). This is dangerous because a sample
/// that truly has the 'A' allele would be ignored, leading to an incomplete and potentially incorrect final score.
///
/// # Arguments
/// * `variant_id` - The `chr:pos` identifier for error reporting.
/// * `effect_allele` - The effect allele from the score file.
/// * `other_allele` - The other allele from the score file.
/// * `bim_records_for_position` - A slice of all BIM records found at this chromosomal position.
///
/// # Returns
/// A `Result` containing:
/// - `Ok(Vec<...>)`: A vector of winning `(BimRecord, BimRowIndex)` tuples. An empty vector means no match was found.
/// - `Err(PrepError::AmbiguousReconciliation)`: A fatal error if an unresolvable ambiguity is detected.
fn resolve_matches_for_score_line<'a>(
    variant_id: &str,
    effect_allele: &str,
    other_allele: &str,
    bim_records_for_position: &'a [(BimRecord, BimRowIndex)],
) -> Result<Vec<&'a (BimRecord, BimRowIndex)>, PrepError> {
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
        return Ok(perfect_matches);
    }

    // --- Pass 2: No perfect matches found. Evaluate permissive matches for ambiguity. ---
    let mut permissive_candidate: Option<&(BimRecord, BimRowIndex)> = None;

    for current_match_tuple in bim_records_for_position {
        let (current_bim_record, _) = current_match_tuple;
        // A permissive match is when the effect allele is one of the two alleles in the BIM file.
        if current_bim_record.allele1 == effect_allele || current_bim_record.allele2 == effect_allele {
            if let Some(first_match_tuple) = permissive_candidate {
                // FATAL AMBIGUITY DETECTED.
                // We have already found one permissive candidate, and we just found another.
                // This is an unresolvable situation. We must fail loudly.
                let (first_bim_record, _) = first_match_tuple;
                let error_message = format!(
                    "Fatal Data Ambiguity: Could not reconcile variant '{}'.\n\
                    > The score file defines an effect allele '{}' (with other_allele '{}').\n\
                    > This effect allele was found in multiple, different genotype records, making it impossible to choose one:\n\
                    - Match 1: Alleles are '{}' and '{}'.\n\
                    - Match 2: Alleles are '{}' and '{}'.",
                    variant_id,
                    effect_allele,
                    other_allele,
                    first_bim_record.allele1, first_bim_record.allele2,
                    current_bim_record.allele1, current_bim_record.allele2
                );
                return Err(PrepError::AmbiguousReconciliation(error_message));
            }
            // This is the first permissive match we've seen. Store it and continue.
            permissive_candidate = Some(current_match_tuple);
        }
    }

    // If we get here, there was either one permissive match or zero.
    if let Some(candidate) = permissive_candidate {
        Ok(vec![candidate])
    } else {
        // No perfect matches, no permissive matches. The variant is simply not found.
        Ok(Vec::new())
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
            PrepError::AmbiguousReconciliation(s) => write!(f, "{}", s),
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
