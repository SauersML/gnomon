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
    BimRowIndex, GroupedComplexRule, PersonSubset, PreparationResult, ScoreColumnIndex, ScoreInfo,
};
use ahash::{AHashMap, AHashSet};
use nonmax::NonMaxU32;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::str::Utf8Error;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Instant;
use crate::types::ReconciledVariantIndex;

// The number of SIMD lanes in the kernel. This MUST be kept in sync with kernel.rs.
const LANE_COUNT: usize = 8;

// ========================================================================================
//                     TYPE-DRIVEN DOMAIN MODEL
// ========================================================================================

#[derive(Clone)]
struct BimRecord {
    allele1: String,
    allele2: String,
}

#[derive(Clone, Copy)]
struct MatrixWriter<T> {
    ptr: *mut T,
}

/// Enum to represent the outcome of reconciling one score file line.
/// This is the core of the triage system that separates the fast and slow paths.
enum ReconciliationOutcome<'a> {
    /// For the fast path. Contains a list of one or more unambiguous matches.
    /// This can include perfect matches or a single, unambiguous permissive match.
    Simple(Vec<&'a (BimRecord, BimRowIndex)>),
    /// For the slow path. Contains a list of multiple, different, plausible
    /// permissive matches, indicating a resolvable multiallelic site.
    Complex(Vec<&'a (BimRecord, BimRowIndex)>),
    /// The variant was not found in the genotype data.
    NotFound,
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
    eprintln!("> Pass 2: Discovering all score columns...");
    let score_names = parse_score_file_headers_only(score_files)?;
    let score_name_to_col_index: AHashMap<String, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), ScoreColumnIndex(i)))
        .collect();

    // --- STAGE 3: PARALLEL COMPILATION, SORTING, & MERGING ---
    eprintln!(
        "> Pass 3: Compiling and sorting data from {} score file(s) in parallel...",
        score_files.len()
    );
    let overall_start_time = Instant::now();

// Step 3a: Run compilation for each file in parallel.
let list_of_tuples: Vec<(Vec<SimpleMapping>, Vec<IntermediateComplexRule>)> = score_files
    .par_iter()
    .map(|path| compile_score_file_to_map(path, &bim_index, &score_name_to_col_index))
    .collect::<Result<_, _>>()?;
    
    // Step 3b: Unzip the list of tuples into two separate collections.
    let (simple_mappings, intermediate_complex_rules): (Vec<_>, Vec<_>) =
        list_of_tuples.into_iter().unzip();
    
    // Flatten each collection into a single master list for each path.
    let mut flat_score_data: Vec<SimpleMapping> = simple_mappings.into_iter().flatten().collect();
    let flat_complex_rules: Vec<IntermediateComplexRule> = intermediate_complex_rules.into_iter().flatten().collect();
    
    // Step 3c: Group the flat list of complex rules into the final, efficient structure.
    // We use a BTreeMap to group by the list of possible contexts.
    let mut grouped_complex_map: BTreeMap<Vec<(BimRowIndex, String, String)>, Vec<ScoreInfo>> =
        BTreeMap::new();
    for intermediate_rule in flat_complex_rules {
        let score_info = ScoreInfo {
            effect_allele: intermediate_rule.effect_allele,
            weight: intermediate_rule.weight,
            score_column_index: intermediate_rule.score_column_index,
        };
        grouped_complex_map
            .entry(intermediate_rule.possible_contexts)
            .or_default()
            .push(score_info);
    }
    let final_complex_rules: Vec<GroupedComplexRule> = grouped_complex_map
        .into_iter()
        .map(|(contexts, scores)| GroupedComplexRule {
            possible_contexts: contexts,
            score_applications: scores,
        })
        .collect();
    
    // Step 3d: Check for overlapping variants. Only fail if BOTH paths are empty.
    if flat_score_data.is_empty() && final_complex_rules.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }

    // Step 3d: Perform a highly optimized parallel sort on the fast-path data. This
    // groups all identical `(BimRowIndex, ScoreColumnIndex)` keys adjacently.
    flat_score_data.par_sort_unstable_by_key(|(key, _value)| *key);

    // Step 3e: Perform a single, fast linear scan to detect duplicates in the fast path.
    if let Some(windows) = flat_score_data.windows(2).find(|w| w[0].0 == w[1].0) {
        let duplicate_key = windows[0].0;
        return Err(PrepError::Parse(format!(
            "Ambiguous input: The variant-score pair (BIM row {}, Score column {}) was defined more than once in the simple path.",
            duplicate_key.0.0, duplicate_key.1.0
        )));
    }
    eprintln!("> TIMING: Pass 3 (Compilation & Segregation) took {:.2?}", overall_start_time.elapsed());

    // --- STAGE 4: FINAL INDEXING & DATA PREPARATION ---
    eprintln!("> Pass 4: Building final variant index and regrouping work...");

    // Create the list of unique BIM indices required for the computation. This single pass
    // over the sorted data is highly memory-efficient as it avoids a large intermediate allocation.
    let mut required_bim_indices = Vec::new();
    if let Some(first) = flat_score_data.first() {
        required_bim_indices.push(first.0.0);
        for item in flat_score_data.iter() {
            if item.0.0 != *required_bim_indices.last().unwrap() {
                required_bim_indices.push(item.0.0);
            }
        }
    }

    let num_reconciled_variants = required_bim_indices.len();
    let bim_row_to_reconciled_variant_map =
        build_bim_to_reconciled_variant_map(&required_bim_indices, total_variants_in_bim);

    // Regroup the flat list into a structure optimized for the final parallel pass.
    let mut work_by_reconciled_idx: Vec<Vec<(ScoreColumnIndex, f32, bool)>> = vec![Vec::new(); num_reconciled_variants];
    for ((bim_row, score_col), (weight, is_flipped)) in flat_score_data {
        // This lookup is guaranteed to succeed.
        let reconciled_idx = bim_row_to_reconciled_variant_map[bim_row.0 as usize].unwrap().get() as usize;
        work_by_reconciled_idx[reconciled_idx].push((score_col, weight, is_flipped));
    }

    // --- STAGE 5: Build Correction Blueprint for Multiallelic Fast-Path Variants ---
    let mut multiallelic_fast_path_corrections = AHashMap::new();
    // To efficiently find siblings, we first create a reverse map from a BIM row index
    // back to its `chr:pos` key. This avoids repeated, slow searches.
    let mut bim_row_to_chr_pos_key: Vec<Option<&str>> = vec![None; total_variants_in_bim];
    for (key, records) in &bim_index {
        for (_, bim_row_idx) in records {
            bim_row_to_chr_pos_key[bim_row_idx.0 as usize] = Some(key);
        }
    }

    for &bim_row_idx in &required_bim_indices {
        if let Some(chr_pos_key) = bim_row_to_chr_pos_key[bim_row_idx.0 as usize] {
            // This lookup is guaranteed to succeed.
            let all_records_at_locus = &bim_index[chr_pos_key];
            if all_records_at_locus.len() > 1 {
                // This is a multiallelic site handled by the fast path. It needs a correction rule.
                let reconciled_idx = bim_row_to_reconciled_variant_map[bim_row_idx.0 as usize].unwrap().get();
                let siblings: Vec<BimRowIndex> = all_records_at_locus
                    .iter()
                    .map(|(_, sibling_idx)| *sibling_idx)
                    .filter(|&sibling_idx| sibling_idx != bim_row_idx) // Exclude the variant itself
                    .collect();

                if !siblings.is_empty() {
                    multiallelic_fast_path_corrections.insert(
                        ReconciledVariantIndex(reconciled_idx),
                        siblings,
                    );
                }
            }
        }
    }

    // --- STAGE 6: Correct Locus Counting & Parallel Matrix Population ---
    eprintln!("> Pass 6: Populating all compute structures in parallel...");

    // First, correctly count the number of unique loci per score for the fast path.
    // This is done by de-duplicating (locus, score) pairs in a HashSet. This new
    // sequential step is extremely fast and ensures the primary denominator component
    // is correct before the main parallel work begins.
    let mut fast_path_locus_score_pairs: AHashSet<(&str, ScoreColumnIndex)> = AHashSet::new();
    for (reconciled_idx, scores_for_variant) in work_by_reconciled_idx.iter().enumerate() {
        let bim_row_idx = required_bim_indices[reconciled_idx];
        // This lookup is guaranteed to succeed because required_bim_indices are from the bim_index.
        let chr_pos_key = bim_row_to_chr_pos_key[bim_row_idx.0 as usize].unwrap();
        for &(score_col, _, _) in scores_for_variant {
            fast_path_locus_score_pairs.insert((chr_pos_key, score_col));
        }
    }

    // Now aggregate the final, correct counts for the number of variants per score.
    let mut score_variant_counts = vec![0u32; score_names.len()];
    // Count the unique fast-path loci for each score from the de-duplicated set.
    for &(_, score_col) in &fast_path_locus_score_pairs {
        score_variant_counts[score_col.0] += 1;
    }

    // Then, add the counts from the slow path, which are already correctly locus-based.
    for group_rule in &final_complex_rules {
        for score_info in &group_rule.score_applications {
            score_variant_counts[score_info.score_column_index.0] += 1;
        }
    }
    
    // Allocate all final data structures for parallel population.
    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let mut weights_matrix = vec![0.0f32; num_reconciled_variants * stride];
    let mut flip_mask_matrix = vec![0u8; num_reconciled_variants * stride];
    let mut variant_to_scores_map: Vec<Vec<ScoreColumnIndex>> = vec![Vec::new(); num_reconciled_variants];
    
    // Create unsafe writers to allow parallel, lock-free writes to disjoint memory regions.
    let writer_tuple = (
        MatrixWriter { ptr: weights_matrix.as_mut_ptr() },
        MatrixWriter { ptr: flip_mask_matrix.as_mut_ptr() },
        MatrixWriter { ptr: variant_to_scores_map.as_mut_ptr() },
    );

    // This parallel loop now only populates the matrices and metadata maps. The counting is done.
    work_by_reconciled_idx
        .par_iter_mut()
        .enumerate()
        .for_each_with(writer_tuple, |writers, (reconciled_idx, scores_for_variant)| {
            let (weights_writer, flip_writer, vtsm_writer) = writers;

            if scores_for_variant.is_empty() { return; }
            let row_offset = reconciled_idx * stride;

            scores_for_variant.sort_unstable_by_key(|(sc, _, _)| *sc);

            let mut vtsm_entry = Vec::with_capacity(scores_for_variant.len());
            for &(score_col, weight, is_flipped) in scores_for_variant.iter() {
                let final_offset = row_offset + score_col.0;
                unsafe {
                    *weights_writer.ptr.add(final_offset) = weight;
                    *flip_writer.ptr.add(final_offset) = if is_flipped { 1 } else { 0 };
                }
                vtsm_entry.push(score_col);
            }
            
            unsafe {
                *vtsm_writer.ptr.add(reconciled_idx) = vtsm_entry;
            }
        });
    
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
        final_complex_rules,
        multiallelic_fast_path_corrections,
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
        .map(|path| -> Result<Vec<String>, PrepError> {
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
        .collect::<Result<Vec<Vec<String>>, PrepError>>()?
        .into_iter()
        .flatten()
        .collect();

    // Convert the final, sorted BTreeSet into a Vec.
    Ok(all_score_names.into_iter().collect())
}

// This type alias makes the function signature much cleaner.
type SimpleMapping = ((BimRowIndex, ScoreColumnIndex), (f32, bool));

/// An intermediate, flat representation of a complex rule before it is grouped.
/// It contains the context information duplicated for each score.
struct IntermediateComplexRule {
    effect_allele: String,
    weight: f32,
    score_column_index: ScoreColumnIndex,
    possible_contexts: Vec<(BimRowIndex, String, String)>,
}

/// Parses a single score file, reconciles its variants against the BIM index, and
/// segregates the results into two distinct collections: one for the simple "fast
/// path" and one for the complex "slow path".
fn compile_score_file_to_map(
    score_path: &Path,
    bim_index: &AHashMap<String, Vec<(BimRecord, BimRowIndex)>>,
    score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
) -> Result<(Vec<SimpleMapping>, Vec<IntermediateComplexRule>), PrepError> {
    // --- Phase 1: Memory Map and Header Parsing ---
    let file = File::open(score_path).map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;
    mmap.advise(memmap2::Advice::Sequential)
        .map_err(|e| PrepError::Io(e, score_path.to_path_buf()))?;

    let mut data_start_offset = 0;
    let header_line = {
        let mut first_line: Option<&[u8]> = None;
        for line in mmap.split(|&b| b == b'\n') {
            data_start_offset += line.len() + 1;
            if !line.is_empty() && !line.starts_with(b"#") {
                first_line = Some(line);
                break;
            }
        }
        first_line.ok_or_else(|| PrepError::Header(format!("File is empty: '{}'", score_path.display())))?
    };

    let header_str = std::str::from_utf8(header_line)
        .map_err(|_| PrepError::Header("Header contains invalid UTF-8".to_string()))?;

    let file_specific_column_map: Vec<ScoreColumnIndex> = header_str
        .trim()
        .split('\t')
        .skip(3)
        .map(|name| {
            score_name_to_col_index.get(name)
                .copied()
                .ok_or_else(|| PrepError::Header(format!("Score '{}' from file '{}' not found in global score list.", name, score_path.display())))
        })
        .collect::<Result<_, _>>()?;

    // --- Phase 2: Segregation Loop ---
    let data_slice = &mmap[data_start_offset..];
    let mut simple_mappings = Vec::new();
    let mut complex_rules = Vec::new();

    for line_bytes in data_slice.split(|&b| b == b'\n') {
        if line_bytes.is_empty() { continue; }

        let mut fields = line_bytes.splitn(4, |&b| b == b'\t');
        let (variant_id_bytes, effect_allele_bytes, other_allele_bytes, weights_bytes) =
            match (fields.next(), fields.next(), fields.next(), fields.next()) {
                (Some(v), Some(e), Some(o), Some(w)) => (v, e, o, w),
                _ => continue, // Skip malformed lines.
            };

        let variant_id = std::str::from_utf8(variant_id_bytes)?;
        if let Some(bim_records) = bim_index.get(variant_id) {
            let effect_allele = std::str::from_utf8(effect_allele_bytes)?;
            let other_allele = std::str::from_utf8(other_allele_bytes)?;

            // Call the new triage engine.
            let match_result =
                resolve_matches_for_score_line(variant_id, effect_allele, other_allele, bim_records)?;

            match match_result {
                ReconciliationOutcome::NotFound => continue,

                // --- FAST PATH ---
                ReconciliationOutcome::Simple(matches) => {
                    for (bim_record, bim_row_index) in matches {
                        let is_flipped = effect_allele != &bim_record.allele2;
                        for (i, weight_bytes) in weights_bytes.split(|&b| b == b'\t').enumerate() {
                            if !weight_bytes.is_empty() {
                                if let Ok(weight) = lexical_core::parse::<f32>(weight_bytes) {
                                    if let Some(&score_col_idx) = file_specific_column_map.get(i) {
                                        simple_mappings.push(((*bim_row_index, score_col_idx), (weight, is_flipped)));
                                    }
                                }
                            }
                        }
                    }
                }

                // --- SLOW PATH ---
                ReconciliationOutcome::Complex(matches) => {
                    // OPTIMIZATION: Construct the context list ONCE per variant line, not once per score.
                    // This requires cloning the allele strings, which is necessary to own them.
                    let possible_contexts: Vec<_> = matches.iter().map(|(rec, idx)| {
                        (*idx, rec.allele1.clone(), rec.allele2.clone())
                    }).collect();

                    for (i, weight_bytes) in weights_bytes.split(|&b| b == b'\t').enumerate() {
                        if !weight_bytes.is_empty() {
                            if let Ok(weight) = lexical_core::parse::<f32>(weight_bytes) {
                                if let Some(&score_col_idx) = file_specific_column_map.get(i) {
                                    complex_rules.push(IntermediateComplexRule {
                                        effect_allele: effect_allele.to_string(),
                                        weight,
                                        score_column_index: score_col_idx,
                                        // This is now a cheap Vec clone, not a deep rebuild.
                                        possible_contexts: possible_contexts.clone(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Return both segregated lists.
    Ok((simple_mappings, complex_rules))
}

/// The single, authoritative resolver for matching a score line to BIM records.
/// This function acts as a TRIAGE system, classifying each match as either
/// simple (for the fast path) or complex (for the deferred slow path).
fn resolve_matches_for_score_line<'a>(
    _variant_id: &str, // DELETE or use.
    effect_allele: &str,
    _other_allele: &str, // DELETE. keeping this is a mistake.
    bim_records_for_position: &'a [(BimRecord, BimRowIndex)],
) -> Result<ReconciliationOutcome<'a>, PrepError> {
    // A BTreeMap provides automatic de-duplication (by BIM row index) and a deterministic ordering
    let mut all_possible_contexts: BTreeMap<BimRowIndex, &'a (BimRecord, BimRowIndex)> =
        BTreeMap::new();

    // A single pass is sufficient to find all plausible interpretations. A variant
    // is a candidate if its effect allele is present in the BIM context. This
    // single rule correctly covers both "perfect" and "permissive" matches,
    // as a perfect match is just a specific type of permissive match.
    for record_tuple in bim_records_for_position {
        let (bim_record, bim_row_index) = record_tuple;

        if &bim_record.allele1 == effect_allele || &bim_record.allele2 == effect_allele {
            all_possible_contexts.insert(*bim_row_index, record_tuple);
        }
    }

    // Now, make the final triage decision based on how many unique, plausible
    // interpretations were found.
    match all_possible_contexts.len() {
        0 => {
            // No matches of any kind were found for this score file variant.
            Ok(ReconciliationOutcome::NotFound)
        }
        1 => {
            // Exactly ONE unique way to interpret this variant was found. This is
            // TRULY SIMPLE. There is no ambiguity. Send to the fast path.
            let simple_matches: Vec<_> = all_possible_contexts.values().copied().collect();
            Ok(ReconciliationOutcome::Simple(simple_matches))
        }
        _ => {
            // MORE THAN ONE unique way to interpret this variant exists. This variant
            // is AMBIGUOUS and requires careful, person-by-person resolution.
            // Send to the slow path.
            let complex_matches: Vec<_> = all_possible_contexts.values().copied().collect();
            Ok(ReconciliationOutcome::Complex(complex_matches))
        }
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

impl From<Utf8Error> for PrepError {
    fn from(err: Utf8Error) -> Self {
        PrepError::Parse(format!("Invalid UTF-8 sequence in score file: {}", err))
    }
}
