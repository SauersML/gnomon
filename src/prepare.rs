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
use memchr::memchr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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

struct MatrixWriter<T> {
    ptr: *mut T,
}

unsafe impl<T: Send> Send for MatrixWriter<T> {}
unsafe impl<T: Send> Sync for MatrixWriter<T> {}

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
    // --- STAGE 1: INITIAL SETUP ---
    // This logic is fast and foundational: determine which people to score and index
    // the genotype variants for fast lookups.
    eprintln!("> Pass 1: Indexing genotype and subject data...");
    let fam_path = plink_prefix.with_extension("fam");
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fam_path)?;
    let total_people_in_fam = all_person_iids.len();

    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();

    let bim_path = plink_prefix.with_extension("bim");
    let (bim_index, total_variants_in_bim) = index_bim_file(&bim_path)?;

    // --- STAGE 2: GLOBAL METADATA DISCOVERY ---
    // A fast parallel pass over just the headers of the score files to discover the
    // complete set of score names. This determines the width of our final matrices.
    eprintln!("> Pass 2: Discovering all score columns...");
    let score_names = parse_score_file_headers_only(score_files)?;
    let score_name_to_col_index: AHashMap<String, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), ScoreColumnIndex(i)))
        .collect();

    // --- STAGE 3: PARALLEL COMPILATION & REDUCTION ---
    // This is the heart of the new architecture. We process each score file in parallel,
    // creating a map of its computations. Then, we merge (reduce) all of these maps
    // into a single, unified map that represents the entire workload.
    eprintln!(
        "> Pass 3: Compiling and merging {} score file(s) in parallel...",
        score_files.len()
    );
    let overall_start_time = Instant::now();
    let unified_map: AHashMap<BimRowIndex, AHashMap<ScoreColumnIndex, (f32, bool)>> =
        score_files
            .par_iter()
            .map(|path| {
                compile_score_file_to_map(
                    path,
                    &bim_index,
                    &score_name_to_col_index,
                )
            })
            .try_reduce(AHashMap::new, |mut acc_map, file_map| {
                // This reduce closure merges the map from one file into the accumulator map.
                // It also serves as our cross-file duplicate detection system.
                for (bim_row, local_inner_map) in file_map {
                    let acc_inner_map = acc_map.entry(bim_row).or_default();
                    for (score_col, value) in local_inner_map {
                        // If the score column already exists for this variant, it's an
                        // ambiguous definition across files.
                        if acc_inner_map.insert(score_col, value).is_some() {
                            return Err(PrepError::Parse(format!(
                                "Ambiguous input: A variant is defined for the same score multiple times across different score files."
                            )));
                        }
                    }
                }
                Ok(acc_map)
            })? // This single '?' propagates any error from the parallel map-reduce operation.

    if unified_map.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }
    eprintln!("> TIMING: Pass 3 (Compilation & Merge) took {:.2?}", overall_start_time.elapsed());
    
    // --- STAGE 4: FINAL INDEXING ---
    // With the unified map, we now have the definitive set of variants needed.
    // We can now create the final, dense mapping from the original BIM index to
    // our new compact matrix index.
    eprintln!("> Pass 4: Building final variant index...");
    let mut required_bim_indices: Vec<BimRowIndex> = unified_map.keys().copied().collect();
    required_bim_indices.sort_unstable();
    let num_reconciled_variants = required_bim_indices.len();
    let bim_row_to_reconciled_variant_map =
        build_bim_to_reconciled_variant_map(&required_bim_indices, total_variants_in_bim);

    // --- STAGE 5: FINAL MATRIX POPULATION (NEW DIRECT-WRITE) ---
    // Allocate matrices to their exact final size and populate them in a single,
    // highly parallel pass with no searching or indirection.
    eprintln!("> Pass 5: Populating compute matrices...");
    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let matrix_size = num_reconciled_variants * stride;
    let mut weights_matrix = vec![0.0f32; matrix_size];
    let mut flip_mask_matrix = vec![0u8; matrix_size];

    let weights_writer = MatrixWriter { ptr: weights_matrix.as_mut_ptr() };
    let flip_writer = MatrixWriter { ptr: flip_mask_matrix.as_mut_ptr() };

    unified_map.par_iter().for_each(|(bim_row, inner_score_map)| {
        // This lookup is guaranteed to succeed because the unified_map's keys are a
        // superset of the keys used to build the reconciled map.
        let reconciled_variant_idx = bim_row_to_reconciled_variant_map[bim_row.0 as usize].unwrap().get();
        let row_offset = reconciled_variant_idx as usize * stride;

        for (&score_col, &(weight, is_flipped)) in inner_score_map {
            let final_offset = row_offset + score_col.0;
            // SAFETY: This is a parallel random-access write. It is safe because the
            // `unified_map` guarantees that each `(bim_row, score_col)` pair is unique.
            // Therefore, no two threads can ever write to the same memory location.
            unsafe {
                *weights_writer.ptr.add(final_offset) = weight;
                *flip_writer.ptr.add(final_offset) = if is_flipped { 1 } else { 0 };
            }
        }
    });

    // --- STAGE 6: FINAL METADATA ASSEMBLY ---
    // Build the remaining metadata required by the compute pipeline.
    eprintln!("> Pass 6: Assembling final metadata...");
    let mut score_variant_counts = vec![0u32; score_names.len()];
    let mut variant_to_scores_map: Vec<Vec<ScoreColumnIndex>> = vec![Vec::new(); num_reconciled_variants];

    for (bim_row, inner_score_map) in &unified_map {
        let reconciled_variant_idx = bim_row_to_reconciled_variant_map[bim_row.0 as usize].unwrap().get();
        let variant_map_entry = &mut variant_to_scores_map[reconciled_variant_idx as usize];
        for &score_col in inner_score_map.keys() {
            score_variant_counts[score_col.0] += 1;
            variant_map_entry.push(score_col);
        }
    }
    // Sort the inner vectors for deterministic results in the missingness path.
    variant_to_scores_map.par_iter_mut().for_each(|v| v.sort_unstable());
    
    // --- FINAL CONSTRUCTION ---
    let bytes_per_variant = (total_people_in_fam as u64 + 3) / 4;
    let mut output_idx_to_fam_idx = Vec::with_capacity(num_people_to_score);
    let mut person_fam_to_output_idx = vec![None; total_people_in_fam];

    for (output_idx, iid) in final_person_iids.iter().enumerate() {
        let original_fam_idx = *iid_to_original_idx.get(iid).unwrap();
        output_idx_to_fam_idx.push(original_fam_idx);
        person_fam_to_output_idx[original_fam_idx as usize] = Some(output_idx as u32);
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

/// Parses a single score file, reconciles its variants against the BIM index, and
/// compiles the results into a compact hash map of unique computations.
///
/// This function is the core of the parallel processing pipeline. It uses memory-mapped
/// I/O and parallel chunking to achieve high throughput with minimal memory allocations.
/// The output is a map from the original `BimRowIndex` to an inner map containing all
/// score data for that variant, which resolves the "chicken and egg" problem of needing
/// a dense index before all variants are known.
///
/// # Arguments
///
/// * `score_path` - The path to the gnomon-native score file to process.
/// * `bim_index` - A read-only reference to the pre-computed BIM file index.
/// * `score_name_to_col_index` - A map from score names to their global column index.
///
/// # Returns
///
/// A `Result` containing:
/// - `Ok(AHashMap<BimRowIndex, AHashMap<ScoreColumnIndex, (f32, bool)>>)`: A nested map.
///   The outer key is the original `BimRowIndex`. The inner map's key is the `ScoreColumnIndex`
///   and its value is the `(weight, is_flipped)` tuple for that specific computation.
/// - `Err(PrepError)`: If any parsing, reconciliation, or ambiguity error occurs.
fn compile_score_file_to_map(
    score_path: &Path,
    bim_index: &AHashMap<String, Vec<(BimRecord, BimRowIndex)>>,
    score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
) -> Result<AHashMap<BimRowIndex, AHashMap<ScoreColumnIndex, (f32, bool)>>, PrepError> {
    // --- Phase 1: Memory Map and Header Parsing ---
    let file = File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    // SAFETY: The file is read-only, which is safe for memory mapping.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    mmap.advise(memmap2::Advice::Sequential)
        .map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;

    // Efficiently find the first non-comment line to identify the header and the data start offset.
    let mut data_start_offset = 0;
    let header_line = {
        let mut first_line: Option<&[u8]> = None;
        for line in mmap.split(|&b| b == b'\n') {
            data_start_offset += line.len() + 1; // +1 for the newline character.
            if !line.is_empty() && !line.starts_with(b"#") {
                first_line = Some(line);
                break;
            }
        }
        first_line.ok_or_else(|| {
            PrepError::Header(format!(
                "Score file '{}' is empty or contains only metadata.",
                score_path.display()
            ))
        })?
    };

    let header_str = std::str::from_utf8(header_line)
        .map_err(|_| PrepError::Header("Header contains invalid UTF-8".to_string()))?;
    
    // Create a map from this file's specific column index (0, 1, 2...) to the global `ScoreColumnIndex`.
    // This avoids repeated string-based lookups in the hot loop.
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
    
    /// Helper to divide a byte slice into chunks ending at newlines. This ensures that
    /// each parallel task processes a set of complete lines.
    fn find_chunk_boundaries(data: &[u8], target_chunk_size: usize) -> Vec<&[u8]> {
        if data.is_empty() { return Vec::new(); }
        let mut boundaries = Vec::new();
        let mut current_pos = 0;
        while current_pos < data.len() {
            let chunk_end = (current_pos + target_chunk_size).min(data.len());
            if chunk_end == data.len() {
                boundaries.push(&data[current_pos..]);
                break;
            }
            // Find the next newline *at or after* our target end point.
            // This guarantees the chunk ends cleanly.
            let next_newline = match memchr(b'\n', &data[chunk_end..]) {
                Some(pos) => chunk_end + pos,
                None => data.len(), // If no newline, this is the last chunk.
            };
            boundaries.push(&data[current_pos..next_newline]);
            current_pos = next_newline + 1; // Start the next chunk after the newline.
        }
        boundaries
    }

    let data_slice = &mmap[data_start_offset..];
    let chunks = find_chunk_boundaries(data_slice, 16 * 1024 * 1024); // 16 MB chunks

    chunks
        .into_par_iter()
        .map(|chunk| -> Result<AHashMap<BimRowIndex, AHashMap<ScoreColumnIndex, (f32, bool)>>, PrepError> {
            let mut local_map = AHashMap::new();

            for line_bytes in chunk.split(|&b| b == b'\n') {
                if line_bytes.is_empty() { continue; }

                // This is a zero-allocation way to get the first three fields.
                let mut fields = line_bytes.splitn(4, |&b| b == b'\t');
                let (variant_id_bytes, effect_allele_bytes, other_allele_bytes, weights_bytes) = 
                    match (fields.next(), fields.next(), fields.next(), fields.next()) {
                        (Some(v), Some(e), Some(o), Some(w)) => (v, e, o, w),
                        _ => continue, // Skip malformed lines with fewer than 4 columns.
                    };
                
                let variant_id = std::str::from_utf8(variant_id_bytes).map_err(|_| PrepError::Parse("Invalid UTF-8 in variant_id".to_string()))?;
                
                if let Some(bim_records) = bim_index.get(variant_id) {
                    let effect_allele = std::str::from_utf8(effect_allele_bytes).map_err(|_| PrepError::Parse("Invalid UTF-8 in effect_allele".to_string()))?;
                    let other_allele = std::str::from_utf8(other_allele_bytes).map_err(|_| PrepError::Parse("Invalid UTF-8 in other_allele".to_string()))?;
                    
                    let winning_matches = resolve_matches_for_score_line(variant_id, effect_allele, other_allele, bim_records)?;

                    for (bim_record, bim_row_index) in winning_matches {
                        let is_flipped = effect_allele != &bim_record.allele2;
                        let inner_map = local_map.entry(*bim_row_index).or_default();

                        for (i, weight_bytes) in weights_bytes.split(|&b| b == b'\t').enumerate() {
                            if !weight_bytes.is_empty() {
                                if let Ok(weight) = lexical_core::parse::<f32>(weight_bytes) {
                                    let score_col_idx = file_specific_column_map[i];
                                    if inner_map.insert(score_col_idx, (weight, is_flipped)).is_some() {
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
            Ok(local_map)
        })
        .try_reduce(AHashMap::new, |mut acc_map, local_map| {
            for (bim_row, local_inner_map) in local_map {
                let acc_inner_map = acc_map.entry(bim_row).or_default();
                for (score_col, value) in local_inner_map {
                    if acc_inner_map.insert(score_col, value).is_some() {
                        return Err(PrepError::Parse(format!(
                            "Ambiguous input: A variant is defined for the same score multiple times in file '{}'.",
                            score_path.display()
                        )));
                    }
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
