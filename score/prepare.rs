// ========================================================================================
//
//               The preparation "compiler"
//
// ========================================================================================
//
// This module transforms raw user inputs into an optimized "computation
// blueprint." It now uses a low-memory, high-throughput streaming merge-join
// algorithm to handle genome-scale data.

use crate::types::{
    BimRowIndex, FilesetBoundary, GroupedComplexRule, PersonSubset, PipelineKind,
    PreparationResult, ScoreColumnIndex, ScoreInfo,
};
use ahash::{AHashMap, AHashSet};
use bumpalo::Bump;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::str::Utf8Error;
use std::time::Instant;

// The number of SIMD lanes in the kernel. This MUST be kept in sync with kernel.rs.
const LANE_COUNT: usize = 8;

// ========================================================================================
//              Type-driven domain model for streaming
// ========================================================================================

/// The primitive, sortable key used for all merge-join operations.
type VariantKey = (u8, u32);

// --- Zero-Copy Internal Data Structures ---
// These temporary structs hold string data borrowed from an arena, avoiding
// allocations in the hot path. Their lifetime `'arena` is tied to the `Bump`
// arena created at the start of `prepare_for_computation`.

/// A parsed record from a `.bim` file, holding borrowed string slices.
#[derive(Debug, Copy, Clone)]
struct KeyedBimRecord<'arena> {
    key: VariantKey,
    bim_row_index: BimRowIndex,
    allele1: &'arena str,
    allele2: &'arena str,
}

/// A parsed record from a score file, holding a borrowed string slice.
#[derive(Debug, Copy, Clone)]
struct KeyedScoreRecord<'arena> {
    key: VariantKey,
    effect_allele: &'arena str,
    other_allele: &'arena str,
    score_column_index: ScoreColumnIndex,
    weight: f32,
}

// Manual implementation to handle f32 comparison correctly.
impl<'arena> PartialEq for KeyedScoreRecord<'arena> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.weight.to_bits() == other.weight.to_bits()
    }
}
impl<'arena> Eq for KeyedScoreRecord<'arena> {}

/// A self-contained item for the merge heap.
#[derive(Debug, Copy, Clone)]
struct HeapItem<'arena> {
    record: KeyedScoreRecord<'arena>,
    file_idx: usize,
}

impl<'arena> PartialEq for HeapItem<'arena> {
    fn eq(&self, other: &Self) -> bool {
        self.record == other.record
    }
}

impl<'arena> Eq for HeapItem<'arena> {}

impl<'arena> PartialOrd for HeapItem<'arena> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'arena> Ord for HeapItem<'arena> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want a min-heap, so we reverse the comparison on the key.
        // Tie-break by file_idx to ensure deterministic order.
        other
            .record
            .key
            .cmp(&self.record.key)
            .then_with(|| other.file_idx.cmp(&self.file_idx))
    }
}

/// Manages the state of a single file reader in the KWayMergeIterator.
struct FileStream<'arena> {
    reader: BufReader<File>,
    /// A buffer for weights and column indices from the current line being processed.
    line_buffer: std::collections::VecDeque<(f32, ScoreColumnIndex)>,
    /// The key and alleles (borrowed from the arena) for the current buffered line.
    current_line_info: Option<(VariantKey, &'arena str, &'arena str)>,
    // This is a temporary buffer for reading lines from the file before they
    // are allocated into the long-lived arena.
    line_string_buffer: String,
    /// Counter for malformed lines in this specific file stream.
    malformed_lines_count: usize,
}

/// An iterator that merges multiple, sorted score files on the fly.
struct KWayMergeIterator<'arena> {
    streams: Vec<FileStream<'arena>>,
    heap: BinaryHeap<HeapItem<'arena>>,
    file_column_maps: Vec<Vec<ScoreColumnIndex>>,
    // Holds a terminal error. If Some, iteration will stop after yielding the error.
    next_error: Option<PrepError>,
    // A reference to the memory arena for zero-copy string allocations.
    bump: &'arena Bump,
}

/// An iterator that streams over one or more `.bim` files.
struct BimIterator<'a, 'arena> {
    fileset_prefixes: std::slice::Iter<'a, PathBuf>,
    current_reader: Option<Lines<BufReader<File>>>,
    global_offset: u64,
    local_line_num: u64,
    current_path: PathBuf,
    // A list of the file boundaries, collected on-the-fly during the single iteration pass.
    boundaries: Vec<FilesetBoundary>,
    // A reference to the memory arena.
    bump: &'arena Bump,
    // A temporary buffer for reading lines.
    line_string_buffer: String,
}

/// Enum to represent the outcome of reconciling one score file line.
enum ReconciliationOutcome<'a, 'arena> {
    Simple(Vec<&'a KeyedBimRecord<'arena>>),
    Complex(Vec<&'a KeyedBimRecord<'arena>>),
    NotFound,
}

// ========================================================================================
//                                  Public API
// ========================================================================================

/// A struct to hold all necessary information to debug a merge-join failure.
/// It is virtually zero-cost in the success case.
#[derive(Debug, Default)]
pub struct MergeDiagnosticInfo {
    total_bim_variants_processed: u64,
    total_score_records_processed: u64,
    // Store the last few raw keys we saw from each stream. Formatting is deferred.
    last_bim_keys_seen: std::collections::VecDeque<VariantKey>,
    last_score_keys_seen: std::collections::VecDeque<VariantKey>,
}

/// The number of recent keys to store for diagnostic reporting.
const DIAGNOSTIC_BUFFER_SIZE: usize = 10;

impl MergeDiagnosticInfo {
    /// Adds a key to the circular buffer. This is a very cheap operation.
    fn add_bim_key(&mut self, key: VariantKey) {
        if self.last_bim_keys_seen.len() == DIAGNOSTIC_BUFFER_SIZE {
            self.last_bim_keys_seen.pop_front();
        }
        self.last_bim_keys_seen.push_back(key);
    }

    /// Adds a key to the circular buffer for score file keys.
    fn add_score_key(&mut self, key: VariantKey) {
        if self.last_score_keys_seen.len() == DIAGNOSTIC_BUFFER_SIZE {
            self.last_score_keys_seen.pop_front();
        }
        self.last_score_keys_seen.push_back(key);
    }
}

#[derive(Debug)]
pub enum PrepError {
    Io(io::Error, PathBuf),
    Parse(String),
    Header(String),
    InconsistentKeepId(String),
    /// An error indicating that no variants from the score files could be matched
    /// to variants in the genotype data.
    NoOverlappingVariants(MergeDiagnosticInfo),
    AmbiguousReconciliation(String),
}

pub fn prepare_for_computation(
    fileset_prefixes: &[PathBuf],
    sorted_score_files: &[PathBuf],
    keep_file: Option<&Path>,
) -> Result<PreparationResult, PrepError> {
    // --- Stage 1: Initial setup ---
    eprintln!("> Stage 1: Indexing subject data...");
    let fam_path = fileset_prefixes[0].with_extension("fam");
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fam_path)?;
    let total_people_in_fam = all_person_iids.len();

    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();
    let total_variants_in_bim = count_total_variants(fileset_prefixes)?;

    // --- Stage 2: Global metadata discovery ---
    eprintln!("> Stage 2: Discovering all score columns...");
    let score_names = parse_score_file_headers_only(sorted_score_files)?;
    let score_name_to_col_index: AHashMap<String, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), ScoreColumnIndex(i)))
        .collect();

    // --- Stage 3: Single-pass data collection ---
    eprintln!("> Stage 3: Streaming and collecting data from all input files...");
    let overall_start_time = Instant::now();

    // The `bump` arena is used for all temporary string allocations during the
    // merge-join. This is the core of the zero-copy optimization.
    let bump = Bump::new();

    let mut diagnostics = MergeDiagnosticInfo::default();
    let mut seen_invalid_bim_chrs: AHashSet<String> = AHashSet::new();
    let mut seen_invalid_score_chrs: AHashSet<String> = AHashSet::new();

    // Create the iterators, giving them a reference to the arena.
    let mut bim_iterator = BimIterator::new(fileset_prefixes, &bump)?;
    let mut score_iterator =
        KWayMergeIterator::new(sorted_score_files, &score_name_to_col_index, &bump)?;

    let mut bim_iter = bim_iterator.by_ref().peekable();
    let mut score_iter = score_iterator.by_ref().peekable();

    // These data structures will store the results of the merge-join.
    // Allele strings borrowed from the arena are stored here temporarily.
    let mut simple_path_data: BTreeMap<BimRowIndex, BTreeMap<ScoreColumnIndex, (f32, bool)>> =
        BTreeMap::new();
    // For complex rules, the key is the set of BIM contexts. The value is a tuple
    // containing the canonical chr:pos key for the locus and all score applications.
    let mut intermediate_complex_rules: BTreeMap<
        Vec<(BimRowIndex, &str, &str)>,
        (VariantKey, Vec<(ScoreColumnIndex, f32, &str, &str)>),
    > = BTreeMap::new();

    while bim_iter.peek().is_some() && score_iter.peek().is_some() {
        let bim_key = match bim_iter.peek().unwrap() {
            Ok(rec) => rec.key,
            Err(_) => match bim_iter.next().unwrap().unwrap_err() {
                PrepError::Parse(msg) => {
                    if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                        && seen_invalid_bim_chrs.insert(chr_name.to_string())
                    {
                        eprintln!(
                            "Warning: Skipping variant(s) in BIM file due to unparsable chromosome name: '{chr_name}'."
                        );
                    }
                    continue;
                }
                e => return Err(e),
            },
        };

        let score_key = match score_iter.peek().unwrap() {
            Ok(rec) => rec.key,
            Err(_) => match score_iter.next().unwrap().unwrap_err() {
                PrepError::Parse(msg) => {
                    if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                        && seen_invalid_score_chrs.insert(chr_name.to_string())
                    {
                        eprintln!(
                            "Warning: Skipping variant(s) in score file due to unparsable chromosome name: '{chr_name}'."
                        );
                    }
                    continue;
                }
                e => return Err(e),
            },
        };

        match bim_key.cmp(&score_key) {
            Ordering::Less => {
                diagnostics.add_bim_key(bim_key);
                diagnostics.total_bim_variants_processed += 1;
                bim_iter.next();
            }
            Ordering::Greater => {
                diagnostics.add_score_key(score_key);
                diagnostics.total_score_records_processed += 1;
                score_iter.next();
            }
            Ordering::Equal => {
                let key = bim_key;
                diagnostics.add_bim_key(key);
                diagnostics.add_score_key(key);

                let mut bim_group = Vec::new();
                while let Some(Ok(peek_item)) = bim_iter.peek() {
                    if peek_item.key != key {
                        break;
                    }
                    match bim_iter.next().unwrap() {
                        Ok(item) => bim_group.push(item),
                        Err(PrepError::Parse(msg)) => {
                            if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                                && seen_invalid_bim_chrs.insert(chr_name.to_string())
                            {
                                eprintln!(
                                    "Warning: Skipping variant(s) in BIM file due to unparsable chromosome name: '{chr_name}'."
                                );
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                diagnostics.total_bim_variants_processed += bim_group.len() as u64;

                let mut score_group = Vec::new();
                while let Some(Ok(peek_item)) = score_iter.peek() {
                    if peek_item.key != key {
                        break;
                    }
                    match score_iter.next().unwrap() {
                        Ok(item) => score_group.push(item),
                        Err(PrepError::Parse(msg)) => {
                            if let Some(chr_name) = extract_chr_from_parse_error(&msg)
                                && seen_invalid_score_chrs.insert(chr_name.to_string())
                            {
                                eprintln!(
                                    "Warning: Skipping variant(s) in score file due to unparsable chromosome name: '{chr_name}'."
                                );
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                diagnostics.total_score_records_processed += score_group.len() as u64;

                for score_record in score_group {
                    let outcome = resolve_matches_for_score_line(score_record, &bim_group)?;

                    match outcome {
                        ReconciliationOutcome::Simple(matches) => {
                            for bim_rec in matches {
                                let is_flipped = score_record.effect_allele == bim_rec.allele1;
                                let score_map =
                                    simple_path_data.entry(bim_rec.bim_row_index).or_default();

                                if score_map
                                    .insert(
                                        score_record.score_column_index,
                                        (score_record.weight, is_flipped),
                                    )
                                    .is_some()
                                {
                                    return Err(PrepError::AmbiguousReconciliation(format!(
                                        "Ambiguous input: The same variant-score pair is defined with different weights across input files. Variant BIM index: {}, Score: {}",
                                        bim_rec.bim_row_index.0,
                                        score_names[score_record.score_column_index.0]
                                    )));
                                }
                            }
                        }
                        ReconciliationOutcome::Complex(matches) => {
                            let possible_contexts: Vec<_> = matches
                                .iter()
                                .map(|rec| (rec.bim_row_index, rec.allele1, rec.allele2))
                                .collect();
                            let score_info = (
                                score_record.score_column_index,
                                score_record.weight,
                                score_record.effect_allele,
                                score_record.other_allele,
                            );

                            // The key for the map is the set of BIM contexts. On first
                            // encounter, we store the canonical chr:pos key. Then we
                            // append the score application.
                            let entry = intermediate_complex_rules
                                .entry(possible_contexts)
                                .or_insert_with(|| (key, Vec::new()));

                            // The value is a tuple: (key, scores_vec), so we push to entry.1
                            entry.1.push(score_info);
                        }
                        ReconciliationOutcome::NotFound => {}
                    }
                }
            }
        }
    }
    eprintln!(
        "> TIMING: Stage 3 (Data Collection) took {:.2?}",
        overall_start_time.elapsed()
    );

    // After iterating, sum malformed line counts from all streams
    let total_malformed_lines: usize = score_iterator
        .streams
        .iter()
        .map(|s| s.malformed_lines_count)
        .sum();

    if total_malformed_lines > 0 {
        eprintln!(
            "> Warning: Skipped {total_malformed_lines} lines from score files due to missing columns (variant_id, effect_allele, other_allele)."
        );
    }

    // --- Stage 4: In-memory processing and matrix construction ---
    eprintln!("> Stage 4: Verifying data and building final matrices...");

    // The "lifetime escape hatch": convert the temporary, borrowed complex rule data
    // into final, owned `GroupedComplexRule` structs. This is the only place where
    // we perform bulk copies of the allele strings we collected.
    let final_complex_rules: Vec<GroupedComplexRule> = intermediate_complex_rules
        .into_iter()
        .map(|(contexts, (variant_key, scores))| {
            // Convert the numeric chromosome key back into a string representation.
            let chr_str = match variant_key.0 {
                23 => "X".to_string(),
                24 => "Y".to_string(),
                25 => "MT".to_string(),
                n => n.to_string(),
            };

            GroupedComplexRule {
                locus_chr_pos: (chr_str, variant_key.1),
                possible_contexts: contexts
                    .into_iter()
                    .map(|(idx, a1, a2)| (idx, a1.to_string(), a2.to_string()))
                    .collect(),
                score_applications: scores
                    .into_iter()
                    .map(|(sc_idx, weight, ea, oa)| ScoreInfo {
                        effect_allele: ea.to_string(),
                        other_allele: oa.to_string(),
                        weight,
                        score_column_index: sc_idx,
                    })
                    .collect(),
            }
        })
        .collect();

    let mut all_required_indices: BTreeSet<BimRowIndex> = BTreeSet::new();
    all_required_indices.extend(simple_path_data.keys());
    for rule in &final_complex_rules {
        for (bim_idx, _, _) in &rule.possible_contexts {
            all_required_indices.insert(*bim_idx);
        }
    }

    if all_required_indices.is_empty() {
        return Err(PrepError::NoOverlappingVariants(diagnostics));
    }

    let required_bim_indices: Vec<BimRowIndex> = all_required_indices.into_iter().collect();
    let num_reconciled_variants = required_bim_indices.len();

    let stride = score_names.len().div_ceil(LANE_COUNT) * LANE_COUNT;
    let mut weights_matrix = vec![0.0f32; num_reconciled_variants * stride];
    let mut flip_mask_matrix = vec![0u8; num_reconciled_variants * stride];
    let mut variant_to_scores_map: Vec<Vec<ScoreColumnIndex>> =
        vec![Vec::new(); num_reconciled_variants];

    weights_matrix
        .par_chunks_mut(stride)
        .zip(flip_mask_matrix.par_chunks_mut(stride))
        .zip(variant_to_scores_map.par_iter_mut())
        .enumerate()
        .for_each(
            |(reconciled_idx, ((weights_chunk, flip_chunk), vtsm_entry))| {
                let bim_row_index = required_bim_indices[reconciled_idx];
                if let Some(score_data_map) = simple_path_data.get(&bim_row_index) {
                    for (&score_col_idx, &(weight, is_flipped)) in score_data_map.iter() {
                        weights_chunk[score_col_idx.0] = weight;
                        flip_chunk[score_col_idx.0] = if is_flipped { 1 } else { 0 };
                    }
                    *vtsm_entry = score_data_map.keys().copied().collect();
                }
            },
        );

    let mut score_variant_counts = vec![0u32; score_names.len()];
    for scores in &variant_to_scores_map {
        for score_col in scores {
            score_variant_counts[score_col.0] += 1;
        }
    }
    for rule in &final_complex_rules {
        let mut counted_cols: AHashSet<ScoreColumnIndex> = AHashSet::new();
        for score_info in &rule.score_applications {
            counted_cols.insert(score_info.score_column_index);
        }
        for score_col in counted_cols {
            score_variant_counts[score_col.0] += 1;
        }
    }

    // --- Stage 5: Final assembly ---
    let bytes_per_variant = (total_people_in_fam as u64).div_ceil(4);
    let mut output_idx_to_fam_idx = Vec::with_capacity(num_people_to_score);
    let mut person_fam_to_output_idx = vec![None; total_people_in_fam];

    for (output_idx, iid) in final_person_iids.iter().enumerate() {
        let original_fam_idx = *iid_to_original_idx.get(iid).unwrap();
        output_idx_to_fam_idx.push(original_fam_idx);
        person_fam_to_output_idx[original_fam_idx as usize] = Some(output_idx as u32);
    }

    let pipeline_kind = if fileset_prefixes.len() <= 1 {
        PipelineKind::SingleFile(fileset_prefixes[0].with_extension("bed"))
    } else {
        PipelineKind::MultiFile(bim_iterator.boundaries)
    };

    Ok(PreparationResult::new(
        weights_matrix,
        flip_mask_matrix,
        stride,
        required_bim_indices,
        final_complex_rules,
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
        pipeline_kind,
    ))
}

// ========================================================================================
//                             Private implementation helpers
// ========================================================================================

/// Extracts the malformed chromosome name from a `PrepError::Parse` message.
fn extract_chr_from_parse_error(msg: &str) -> Option<&str> {
    if let Some(rest) = msg.strip_prefix("Invalid chromosome format '")
        && let Some(end_pos) = rest.find('\'')
    {
        return Some(&rest[..end_pos]);
    }
    None
}

fn count_lines(path: &Path) -> io::Result<u64> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines().count() as u64)
}

fn count_total_variants(fileset_prefixes: &[PathBuf]) -> Result<u64, PrepError> {
    fileset_prefixes
        .par_iter()
        .map(|prefix| {
            let bim_path = prefix.with_extension("bim");
            count_lines(&bim_path).map_err(|e| PrepError::Io(e, bim_path.clone()))
        })
        .try_reduce(|| 0, |a, b| Ok(a + b))
}

fn parse_key(chr_str: &str, pos_str: &str) -> Result<(u8, u32), PrepError> {
    // First, check for special, non-numeric chromosome names
    if chr_str.eq_ignore_ascii_case("X") {
        let pos_num: u32 = pos_str
            .parse()
            .map_err(|e| PrepError::Parse(format!("Invalid position '{pos_str}': {e}")))?;
        return Ok((23, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("Y") {
        let pos_num: u32 = pos_str
            .parse()
            .map_err(|e| PrepError::Parse(format!("Invalid position '{pos_str}': {e}")))?;
        return Ok((24, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("MT") {
        let pos_num: u32 = pos_str
            .parse()
            .map_err(|e| PrepError::Parse(format!("Invalid position '{pos_str}': {e}")))?;
        return Ok((25, pos_num));
    }

    // Next, handle numeric chromosomes, stripping a potential "chr" prefix case-insensitively.
    let number_part = if chr_str.len() >= 3 && chr_str[..3].eq_ignore_ascii_case("chr") {
        &chr_str[3..]
    } else {
        chr_str
    };

    // Now, parse the remaining part.
    let chr_num: u8 = number_part.parse().map_err(|_| {
        PrepError::Parse(format!(
            "Invalid chromosome format '{chr_str}'. Expected a number, 'X', 'Y', 'MT', or 'chr' prefix."
        ))
    })?;

    let pos_num: u32 = pos_str
        .parse()
        .map_err(|e| PrepError::Parse(format!("Invalid position '{pos_str}': {e}")))?;

    Ok((chr_num, pos_num))
}

impl<'a, 'arena> BimIterator<'a, 'arena> {
    fn new(fileset_prefixes: &'a [PathBuf], bump: &'arena Bump) -> Result<Self, PrepError> {
        let mut iter = Self {
            fileset_prefixes: fileset_prefixes.iter(),
            current_reader: None,
            global_offset: 0,
            local_line_num: 0,
            current_path: PathBuf::new(),
            boundaries: Vec::with_capacity(fileset_prefixes.len()),
            bump,
            line_string_buffer: String::new(),
        };
        iter.next_file()?;
        Ok(iter)
    }

    fn next_file(&mut self) -> Result<bool, PrepError> {
        self.global_offset += self.local_line_num;
        self.local_line_num = 0;

        if let Some(prefix) = self.fileset_prefixes.next() {
            self.boundaries.push(FilesetBoundary {
                bed_path: prefix.with_extension("bed"),
                starting_global_index: self.global_offset,
            });

            let bim_path = prefix.with_extension("bim");
            self.current_path = bim_path.clone();
            let file = File::open(&bim_path).map_err(|e| PrepError::Io(e, bim_path))?;
            self.current_reader = Some(BufReader::new(file).lines());
            Ok(true)
        } else {
            self.current_reader = None;
            Ok(false)
        }
    }
}

impl<'a, 'arena> Iterator for BimIterator<'a, 'arena> {
    type Item = Result<KeyedBimRecord<'arena>, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.current_reader.as_mut() {
                Some(reader) => match reader.next() {
                    Some(Ok(line)) => {
                        self.local_line_num += 1;
                        self.line_string_buffer.clear();
                        self.line_string_buffer.push_str(&line);
                        let line_in_arena = self.bump.alloc_str(&self.line_string_buffer);

                        let mut parts = line_in_arena.split_whitespace();
                        let chr = parts.next();
                        parts.next(); // Skip variant ID from BIM file
                        parts.next(); // Skip genetic distance in centiMorgans
                        let pos = parts.next();
                        let a1 = parts.next();
                        let a2 = parts.next();

                        if let (Some(chr_str), Some(pos_str), Some(a1), Some(a2)) =
                            (chr, pos, a1, a2)
                        {
                            match parse_key(chr_str, pos_str) {
                                Ok(key) => {
                                    return Some(Ok(KeyedBimRecord {
                                        key,
                                        bim_row_index: BimRowIndex(
                                            self.global_offset + self.local_line_num - 1,
                                        ),
                                        allele1: a1,
                                        allele2: a2,
                                    }));
                                }
                                Err(e) => return Some(Err(e)),
                            }
                        }
                    }
                    Some(Err(e)) => return Some(Err(PrepError::Io(e, self.current_path.clone()))),
                    None => {
                        if let Ok(false) = self.next_file() {
                            return None;
                        }
                    }
                },
                None => return None,
            }
        }
    }
}

impl<'arena> KWayMergeIterator<'arena> {
    fn new(
        file_paths: &[PathBuf],
        score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
        bump: &'arena Bump,
    ) -> Result<Self, PrepError> {
        let mut streams = Vec::with_capacity(file_paths.len());
        let mut file_column_maps = Vec::with_capacity(file_paths.len());

        for path in file_paths {
            let file = File::open(path).map_err(|e| PrepError::Io(e, path.clone()))?;
            let mut reader = BufReader::new(file);
            let mut header_line = String::new();

            loop {
                header_line.clear();
                if reader
                    .read_line(&mut header_line)
                    .map_err(|e| PrepError::Io(e, path.clone()))?
                    == 0
                {
                    break;
                }
                if !header_line.starts_with("##") {
                    break;
                }
            }

            let column_map: Vec<ScoreColumnIndex> = header_line
                .trim()
                .split('\t')
                .skip(3)
                .map(|name| {
                    score_name_to_col_index.get(name).copied().ok_or_else(|| {
                        PrepError::Header(format!(
                            "Score '{name}' from file '{path}' not found in global score list.",
                            name = name,
                            path = path.display()
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;

            file_column_maps.push(column_map);

            streams.push(FileStream {
                reader,
                line_buffer: std::collections::VecDeque::new(),
                current_line_info: None,
                line_string_buffer: String::new(),
                malformed_lines_count: 0,
            });
        }

        let mut iter = Self {
            streams,
            heap: BinaryHeap::new(),
            file_column_maps,
            next_error: None,
            bump,
        };

        for i in 0..iter.streams.len() {
            iter.replenish_from_stream(i)?
        }

        Ok(iter)
    }

    fn replenish_from_stream(&mut self, file_idx: usize) -> Result<(), PrepError> {
        // We scope the mutable borrow of the stream to check its buffer.
        if !self.streams[file_idx].line_buffer.is_empty() {
            let stream = &mut self.streams[file_idx];
            Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
            return Ok(());
        }

        // The line buffer is empty, so we must read a new line from the file.
        // Get the data we need from `self` before we mutably borrow a part of it.
        let column_map = &self.file_column_maps[file_idx];
        let bump = self.bump; // `&'arena Bump` is `Copy`, so this is a cheap reference copy.

        // Now, get the mutable borrow of the stream.
        let stream = &mut self.streams[file_idx];

        // Call the static helper function using the correct syntax.
        // This avoids the borrow checker conflict.
        if Self::read_line_into_buffer(stream, column_map, bump)? {
            Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
        }

        Ok(())
    }

    fn read_line_into_buffer(
        stream: &mut FileStream<'arena>,
        column_map: &[ScoreColumnIndex],
        bump: &'arena Bump,
    ) -> Result<bool, PrepError> {
        stream.line_buffer.clear();
        stream.current_line_info = None;

        loop {
            stream.line_string_buffer.clear();
            let bytes_read = stream
                .reader
                .read_line(&mut stream.line_string_buffer)
                .map_err(|e| PrepError::Io(e, PathBuf::new()))?;

            if bytes_read == 0 {
                return Ok(false);
            }

            if !stream.line_string_buffer.trim().is_empty()
                && !stream.line_string_buffer.starts_with('#')
            {
                break;
            }
        }
        // Use the `bump` argument, not `self.bump`.
        let line_in_arena = bump.alloc_str(&stream.line_string_buffer);

        let mut parts = line_in_arena.split('\t');
        let (variant_id, effect_allele, other_allele) =
            match (parts.next(), parts.next(), parts.next()) {
                (Some(v), Some(e), Some(o)) if !v.is_empty() && !e.is_empty() && !o.is_empty() => {
                    (v, e, o)
                } // Ensure other_allele is also not empty
                _ => {
                    stream.malformed_lines_count += 1;
                    return Ok(false); // Line doesn't have the required three non-empty columns
                }
            };

        let mut key_parts = variant_id.splitn(2, ':');
        let chr_str = key_parts.next().unwrap_or("");
        let pos_str = key_parts.next().unwrap_or("");
        let key = parse_key(chr_str, pos_str)?;
        stream.current_line_info = Some((key, effect_allele, other_allele));

        for (i, weight_str) in parts.enumerate() {
            if let Ok(weight) = weight_str.trim().parse::<f32>()
                && let Some(&score_column_index) = column_map.get(i)
            {
                stream.line_buffer.push_back((weight, score_column_index));
            }
        }
        Ok(!stream.line_buffer.is_empty())
    }

    fn push_next_from_buffer_to_heap(
        stream: &mut FileStream<'arena>,
        file_idx: usize,
        heap: &mut BinaryHeap<HeapItem<'arena>>,
    ) {
        if let Some((weight, score_column_index)) = stream.line_buffer.pop_front() {
            let (key, effect_allele, other_allele) = stream.current_line_info.unwrap();
            let record = KeyedScoreRecord {
                key,
                effect_allele,
                other_allele,
                score_column_index,
                weight,
            };
            heap.push(HeapItem { record, file_idx });
        }
    }
}

impl<'arena> Iterator for KWayMergeIterator<'arena> {
    type Item = Result<KeyedScoreRecord<'arena>, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(e) = self.next_error.take() {
            return Some(Err(e));
        }

        let top_item = self.heap.pop()?;
        let record_to_return = top_item.record;
        let file_idx = top_item.file_idx;

        if let Err(e) = self.replenish_from_stream(file_idx) {
            self.next_error = Some(e);
        }

        Some(Ok(record_to_return))
    }
}

fn resolve_matches_for_score_line<'a, 'arena>(
    score_record: KeyedScoreRecord,
    bim_records_for_position: &'a [KeyedBimRecord<'arena>],
) -> Result<ReconciliationOutcome<'a, 'arena>, PrepError> {
    let is_multiallelic_site = bim_records_for_position.len() > 1;

    if is_multiallelic_site {
        let all_contexts_for_locus = bim_records_for_position.iter().collect();
        return Ok(ReconciliationOutcome::Complex(all_contexts_for_locus));
    }

    let mut simple_matches = BTreeMap::new();
    for record_tuple in bim_records_for_position {
        if record_tuple.allele1 == score_record.effect_allele
            || record_tuple.allele2 == score_record.effect_allele
        {
            simple_matches.insert(record_tuple.bim_row_index, record_tuple);
        }
    }

    if simple_matches.is_empty() {
        Ok(ReconciliationOutcome::NotFound)
    } else {
        let matches_as_vec = simple_matches.values().copied().collect();
        Ok(ReconciliationOutcome::Simple(matches_as_vec))
    }
}

fn resolve_person_subset(
    keep_file: Option<&Path>,
    all_person_iids: &[String],
    iid_to_original_idx: &AHashMap<String, u32>,
) -> Result<(PersonSubset, Vec<String>), PrepError> {
    if let Some(path) = keep_file {
        eprintln!(
            "> Subsetting individuals based on keep file: {}",
            path.display()
        );
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
pub fn parse_score_file_headers_only(score_files: &[PathBuf]) -> Result<Vec<String>, PrepError> {
    let all_score_names: BTreeSet<String> = score_files
        .par_iter()
        .map(|path| -> Result<Vec<String>, PrepError> {
            let file = File::open(path).map_err(|e| PrepError::Io(e, path.to_path_buf()))?;
            let mut reader = BufReader::new(file);
            let mut header_line = String::new();

            loop {
                header_line.clear();
                if reader
                    .read_line(&mut header_line)
                    .map_err(|e| PrepError::Io(e, path.to_path_buf()))?
                    == 0
                {
                    return Err(PrepError::Header(format!(
                        "Score file \"{}\" is empty or contains only metadata lines.",
                        path.display()
                    )));
                }
                if !header_line.starts_with('#') {
                    break;
                }
            }

            let header_parts: Vec<&str> = header_line.trim().split('\t').collect();
            let expected_prefix = &["variant_id", "effect_allele", "other_allele"];

            if header_parts.len() < 3 || &header_parts[0..3] != expected_prefix {
                return Err(PrepError::Header(format!(
                    "Invalid header in \"{}\": Must start with 'variant_id\teffect_allele\tother_allele'.",
                    path.display()
                )));
            }

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

    Ok(all_score_names.into_iter().collect())
}

// ========================================================================================
//                                    Error handling
// ========================================================================================

impl Display for PrepError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            PrepError::Io(e, path) => write!(f, "I/O Error for file {}: {}", path.display(), e),
            PrepError::Parse(s) => write!(f, "Parse Error: {s}"),
            PrepError::Header(s) => write!(f, "Invalid Header: {s}"),
            PrepError::InconsistentKeepId(s) => write!(f, "Configuration Error: {s}"),
            PrepError::NoOverlappingVariants(diag) => {
                writeln!(
                    f,
                    "No overlapping variants found between genotype data and score files."
                )?;
                writeln!(
                    f,
                    "This likely means no variant keys (chr:pos) were identical in both sets of files."
                )?;
                writeln!(f, "\n--- DIAGNOSTIC INFORMATION ---")?;
                writeln!(
                    f,
                    "Total variants processed from BIM files: {}",
                    diag.total_bim_variants_processed
                )?;
                writeln!(
                    f,
                    "Total score records processed from Score files: {}",
                    diag.total_score_records_processed
                )?;

                if !diag.last_bim_keys_seen.is_empty() {
                    writeln!(
                        f,
                        "\nLast {} keys seen from BIM files:",
                        diag.last_bim_keys_seen.len()
                    )?;
                    for key in &diag.last_bim_keys_seen {
                        writeln!(f, "  - {}:{}", key.0, key.1)?;
                    }
                }

                if !diag.last_score_keys_seen.is_empty() {
                    writeln!(
                        f,
                        "\nLast {} keys seen from Score files (does the format match the BIM files?):",
                        diag.last_score_keys_seen.len()
                    )?;
                    for key in &diag.last_score_keys_seen {
                        writeln!(f, "  - {}:{}", key.0, key.1)?;
                    }
                }
                writeln!(
                    f,
                    "\nTIP: Please check for inconsistencies between your files."
                )
            }
            PrepError::AmbiguousReconciliation(s) => write!(f, "{s}"),
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
        PrepError::Parse(format!("Could not parse numeric value: {err}"))
    }
}

impl From<Utf8Error> for PrepError {
    fn from(err: Utf8Error) -> Self {
        PrepError::Parse(format!("Invalid UTF-8 sequence in score file: {err}"))
    }
}
