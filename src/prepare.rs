// ========================================================================================
//
//               THE PREPARATION "COMPILER"
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
use nonmax::NonMaxU32;
use rayon::prelude::*;
use std::cmp::Ordering;

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Lines};
use std::iter::Peekable;

use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::str::Utf8Error;
use std::time::Instant;

// The number of SIMD lanes in the kernel. This MUST be kept in sync with kernel.rs.
const LANE_COUNT: usize = 8;

// ========================================================================================
//              TYPE-DRIVEN DOMAIN MODEL FOR STREAMING
// ========================================================================================

/// The primitive, sortable key used for all merge-join operations.
type VariantKey = (u8, u32);

/// A parsed record from a `.bim` file, tagged with its sort key and global index.
#[derive(Debug)]
struct KeyedBimRecord {
    key: VariantKey,
    bim_row_index: BimRowIndex,
    allele1: String,
    allele2: String,
}

/// A parsed record from a score file, representing one variant-score pair.
#[derive(Debug, Clone)]
struct KeyedScoreRecord {
    key: VariantKey,
    effect_allele: String,
    score_column_index: ScoreColumnIndex,
    weight: f32,
}

// Manual implementation to handle f32 comparison correctly.
impl PartialEq for KeyedScoreRecord {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.weight.to_bits() == other.weight.to_bits()
    }
}
impl Eq for KeyedScoreRecord {}


/// A self-contained item for the merge heap, holding a single score record.
#[derive(Debug)]
struct HeapItem {
    record: KeyedScoreRecord,
    file_idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.record.key == other.record.key
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We want a min-heap, so we reverse the comparison on the key.
        // Tie-break by file_idx to ensure deterministic order.
        other.record.key.cmp(&self.record.key).then_with(|| other.file_idx.cmp(&self.file_idx))
    }
}

/// Manages the state of a single file reader in the KWayMergeIterator.
struct FileStream {
    reader: BufReader<File>,
    /// A buffer for weights and column indices from the current line being processed.
    line_buffer: std::collections::VecDeque<(f32, ScoreColumnIndex)>,
    /// The key and effect allele for the currently buffered line.
    current_line_info: Option<(VariantKey, String)>,
}

/// An iterator that merges multiple, sorted score files on the fly. It correctly
/// handles multi-score lines, yielding one `KeyedScoreRecord` per score.
struct KWayMergeIterator {
    streams: Vec<FileStream>,
    heap: BinaryHeap<HeapItem>,
    file_column_maps: Vec<Vec<ScoreColumnIndex>>,
    // Holds a terminal error. If Some, iteration will stop after yielding the error.
    next_error: Option<PrepError>,
}

/// An iterator that streams over one or more `.bim` files, creating a single
/// virtually merged and sorted stream of records.
struct BimIterator<'a> {
    fileset_prefixes: std::slice::Iter<'a, PathBuf>,
    current_reader: Option<Lines<BufReader<File>>>,
    global_offset: u64,
    local_line_num: u64,
    current_path: PathBuf,
    // A list of the file boundaries, collected on-the-fly during the single iteration pass.
    boundaries: Vec<FilesetBoundary>,
}

#[derive(Clone, Copy)]
struct MatrixWriter<T> {
    ptr: *mut T,
}

/// Enum to represent the outcome of reconciling one score file line.
/// The lifetime parameter `'a` ensures that we are borrowing from the group
/// collected in the merge-join, avoiding allocations.
enum ReconciliationOutcome<'a> {
    Simple(Vec<&'a KeyedBimRecord>),
    Complex(Vec<&'a KeyedBimRecord>),
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
    fileset_prefixes: &[PathBuf],
    sorted_score_files: &[PathBuf],
    keep_file: Option<&Path>,
) -> Result<PreparationResult, PrepError> {
    // --- STAGE 1: INITIAL SETUP ---
    // This pre-computation is fast and necessary before streaming begins.
    eprintln!("> Stage 1: Indexing subject data...");
    let fam_path = fileset_prefixes[0].with_extension("fam");
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fam_path)?;
    let total_people_in_fam = all_person_iids.len();

    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;
    let num_people_to_score = final_person_iids.len();
    let total_variants_in_bim = count_total_variants(fileset_prefixes)?;
    let total_variants_in_bim_usize = usize::try_from(total_variants_in_bim)
        .map_err(|_| PrepError::Parse("Total variants exceeds addressable memory.".to_string()))?;

    // --- STAGE 2: GLOBAL METADATA DISCOVERY ---
    // We must know all possible score columns ahead of time to build a global mapping.
    eprintln!("> Stage 2: Discovering all score columns...");
    let score_names = parse_score_file_headers_only(sorted_score_files)?;
    let score_name_to_col_index: AHashMap<String, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), ScoreColumnIndex(i)))
        .collect();

    // --- STAGE 3: SINGLE-PASS DATA COLLECTION ---
    // This is the core of the new architecture. We perform a single, group-aware
    // merge-join over all input files. Data is processed and stored in an efficient,
    // structured map, and ambiguity is checked on-the-fly, eliminating the need
    // for a large intermediate vector and a costly sort.
    eprintln!("> Stage 3: Streaming and collecting data from all input files...");
    let overall_start_time = Instant::now();

    let mut bim_iterator = BimIterator::new(fileset_prefixes)?;
    let mut bim_iter = bim_iterator.by_ref().peekable();
    let mut score_iterator =
        KWayMergeIterator::new(sorted_score_files, &score_name_to_col_index)?;
    let mut score_iter = score_iterator.by_ref().peekable();

    // This nested map structure is the core of the optimization.
    // It stores all simple-path data, grouped by variant, and allows for an
    // efficient, inline ambiguity check.
    // Key: BimRowIndex (a unique variant)
    // Value: Map<ScoreColumnIndex, (weight: f32, is_flipped: bool)>
    let mut simple_path_data: BTreeMap<BimRowIndex, BTreeMap<ScoreColumnIndex, (f32, bool)>> =
        BTreeMap::new();
    let mut intermediate_complex_rules: BTreeMap<Vec<(BimRowIndex, String, String)>, Vec<ScoreInfo>> =
        BTreeMap::new();

    while bim_iter.peek().is_some() && score_iter.peek().is_some() {
        // This is safe because we just checked that the iterators are not empty.
        let bim_key = bim_iter.peek().unwrap().as_ref().unwrap().key;
        let score_key = score_iter.peek().unwrap().as_ref().unwrap().key;

        match bim_key.cmp(&score_key) {
            Ordering::Less => {
                bim_iter.next();
            }
            Ordering::Greater => {
                score_iter.next();
            }
            Ordering::Equal => {
                let key = bim_key;
                // A variant can be defined multiple times at the same position (multiallelic).
                // Collect all BIM records for this specific locus.
                let mut bim_group = Vec::new();
                while let Some(Ok(peek_item)) = bim_iter.peek() {
                    if peek_item.key == key {
                        bim_group.push(bim_iter.next().unwrap()?);
                    } else {
                        break;
                    }
                }

                // A variant can appear in multiple score files.
                // Collect all score records for this specific locus.
                let mut score_group = Vec::new();
                while let Some(Ok(peek_item)) = score_iter.peek() {
                    if peek_item.key == key {
                        score_group.push(score_iter.next().unwrap()?);
                    } else {
                        break;
                    }
                }

                // Process the complete group for this locus.
                for score_record in score_group {
                    let outcome =
                        resolve_matches_for_score_line(&score_record.effect_allele, &bim_group)?;

                    match outcome {
                        ReconciliationOutcome::Simple(matches) => {
                            for bim_rec in matches {
                                // The compute kernel receives the dosage of `allele2` (the alternate allele)
                                // from the BIM file.
                                // - If a variant is NOT flipped, its contribution is `Weight * DosageOfAllele2`.
                                // - If a variant IS flipped, its contribution is `Weight * DosageOfAllele1`.
                                // Therefore, we must set `is_flipped` to `true` if the score's effect
                                // allele matches `allele1`.
                                let is_flipped = score_record.effect_allele == bim_rec.allele1;
                                let score_map =
                                    simple_path_data.entry(bim_rec.bim_row_index).or_default();

                                // Attempt to insert the score. If the key (ScoreColumnIndex)
                                // already exists, `insert` returns Some(old_value),
                                // indicating a duplicate definition.
                                if score_map
                                    .insert(
                                        score_record.score_column_index,
                                        (score_record.weight, is_flipped),
                                    )
                                    .is_some()
                                {
                                    return Err(PrepError::AmbiguousReconciliation(format!(
                                        "Ambiguous input: The same variant-score pair is defined with different weights across input files. Variant BIM index: {}, Score: {}",
                                        bim_rec.bim_row_index.0, score_names[score_record.score_column_index.0]
                                    )));
                                }
                            }
                        }
                        ReconciliationOutcome::Complex(matches) => {
                            let possible_contexts: Vec<_> = matches
                                .iter()
                                .map(|rec| {
                                    (rec.bim_row_index, rec.allele1.clone(), rec.allele2.clone())
                                })
                                .collect();
                            let score_info = ScoreInfo {
                                effect_allele: score_record.effect_allele,
                                weight: score_record.weight,
                                score_column_index: score_record.score_column_index,
                            };
                            intermediate_complex_rules
                                .entry(possible_contexts)
                                .or_default()
                                .push(score_info);
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

    // --- STAGE 4: IN-MEMORY PROCESSING AND MATRIX CONSTRUCTION ---
    // All file I/O is complete. The following stages are CPU-bound and operate
    // on the data collected in Stage 3.
    eprintln!("> Stage 4: Verifying data and building final matrices...");

    // The ambiguity check was already performed during the streaming stage.
    // We can now proceed directly to matrix construction.

    let final_complex_rules: Vec<GroupedComplexRule> = intermediate_complex_rules
        .into_iter()
        .map(|(contexts, scores)| GroupedComplexRule {
            possible_contexts: contexts,
            score_applications: scores,
        })
        .collect();

    // Build the definitive set of all variants that will be included in the computation.
    let mut all_required_indices: BTreeSet<BimRowIndex> = BTreeSet::new();
    for bim_row_index in simple_path_data.keys() {
        all_required_indices.insert(*bim_row_index);
    }
    for rule in &final_complex_rules {
        for (bim_idx, _, _) in &rule.possible_contexts {
            all_required_indices.insert(*bim_idx);
        }
    }

    if all_required_indices.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }

    let required_bim_indices: Vec<BimRowIndex> = all_required_indices.into_iter().collect();
    let num_reconciled_variants = required_bim_indices.len();

    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let mut weights_matrix = vec![0.0f32; num_reconciled_variants * stride];
    let mut flip_mask_matrix = vec![0u8; num_reconciled_variants * stride];
    let mut variant_to_scores_map: Vec<Vec<ScoreColumnIndex>> =
        vec![Vec::new(); num_reconciled_variants];

    // This parallel loop populates the final matrices. Each thread works on a disjoint
    // set of rows, looking up its work directly in the structured `simple_path_data` map.
    weights_matrix
        .par_chunks_mut(stride)
        .zip(flip_mask_matrix.par_chunks_mut(stride))
        .zip(variant_to_scores_map.par_iter_mut())
        .enumerate()
        .for_each(|(reconciled_idx, ((weights_chunk, flip_chunk), vtsm_entry))| {
            // Find which original BIM row corresponds to this matrix row.
            let bim_row_index = required_bim_indices[reconciled_idx];

            // Look up the prepared data for that BIM row.
            if let Some(score_data_map) = simple_path_data.get(&bim_row_index) {
                for (&score_col_idx, &(weight, is_flipped)) in score_data_map.iter() {
                    weights_chunk[score_col_idx.0] = weight;
                    flip_chunk[score_col_idx.0] = if is_flipped { 1 } else { 0 };
                }

                // Collect the score indices for this variant row. The keys of a BTreeMap are
                // already sorted, so we can collect them directly.
                *vtsm_entry = score_data_map.keys().copied().collect();
            }
        });

    // Calculate the final count of variants contributing to each score.
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

    // --- STAGE 5: FINAL ASSEMBLY ---
    // All data is prepared; construct the final PreparationResult object.
    let bytes_per_variant = (total_people_in_fam as u64 + 3) / 4;
    let mut output_idx_to_fam_idx = Vec::with_capacity(num_people_to_score);
    let mut person_fam_to_output_idx = vec![None; total_people_in_fam];

    for (output_idx, iid) in final_person_iids.iter().enumerate() {
        let original_fam_idx = *iid_to_original_idx.get(iid).unwrap();
        output_idx_to_fam_idx.push(original_fam_idx);
        person_fam_to_output_idx[original_fam_idx as usize] = Some(output_idx as u32);
    }

    // The `bim_iterator` variable is used to retrieve the boundaries that were
    // collected efficiently during the single-pass merge-join. This eliminates
    // the redundant I/O of re-reading every .bim file.
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
//                             PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

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

fn parse_key(variant_id: &str) -> Result<(u8, u32), PrepError> {
    let mut parts = variant_id.splitn(2, ':');
    let chr_str = parts.next().unwrap_or("");
    let pos_str = parts.next().unwrap_or("0");

    // First, check for special, non-numeric chromosome names in a case-insensitive way.
    if chr_str.eq_ignore_ascii_case("X") {
        let pos_num: u32 = pos_str.parse().map_err(|e| PrepError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
        return Ok((23, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("Y") {
        let pos_num: u32 = pos_str.parse().map_err(|e| PrepError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
        return Ok((24, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("MT") {
        let pos_num: u32 = pos_str.parse().map_err(|e| PrepError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
        return Ok((25, pos_num));
    }

    // Next, handle numeric chromosomes, stripping a potential "chr" prefix case-insensitively.
    // This is an allocation-free way to get the part of the string to parse.
    let number_part = if chr_str.len() >= 3 && chr_str[..3].eq_ignore_ascii_case("chr") {
        &chr_str[3..]
    } else {
        chr_str
    };

    // Now, parse the remaining part.
    let chr_num: u8 = number_part.parse().map_err(|_| {
        PrepError::Parse(format!(
            "Invalid chromosome format '{}'. Expected a number, 'X', 'Y', 'MT', or 'chr' prefix.",
            chr_str
        ))
    })?;

    let pos_num: u32 = pos_str.parse().map_err(|e| PrepError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
    
    Ok((chr_num, pos_num))
}

impl<'a> BimIterator<'a> {
    fn new(fileset_prefixes: &'a [PathBuf]) -> Result<Self, PrepError> {
        let mut iter = Self {
            fileset_prefixes: fileset_prefixes.iter(),
            current_reader: None,
            global_offset: 0,
            local_line_num: 0,
            current_path: PathBuf::new(),
            boundaries: Vec::with_capacity(fileset_prefixes.len()),
        };
        iter.next_file()?;
        Ok(iter)
    }

    fn next_file(&mut self) -> Result<bool, PrepError> {
        // Before moving to the next file, add the line count of the *previous* file
        // (stored in local_line_num) to the global offset.
        self.global_offset += self.local_line_num;
        self.local_line_num = 0;

        if let Some(prefix) = self.fileset_prefixes.next() {
            // Record the boundary for the new file. Its starting index is the current
            // global offset, which we just calculated.
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

impl<'a> Iterator for BimIterator<'a> {
    type Item = Result<KeyedBimRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.current_reader.as_mut() {
                Some(reader) => {
                    match reader.next() {
                        Some(Ok(line)) => {
                            self.local_line_num += 1;
                            let mut parts = line.split_whitespace();
                            let chr = parts.next();
                            let _id = parts.next();
                            let _cm = parts.next();
                            let pos = parts.next();
                            let a1 = parts.next();
                            let a2 = parts.next();

                            if let (Some(chr_str), Some(pos_str), Some(a1), Some(a2)) = (chr, pos, a1, a2) {
                                match parse_key(&format!("{}:{}", chr_str, pos_str)) {
                                    Ok(key) => {
                                        return Some(Ok(KeyedBimRecord {
                                            key,
                                            bim_row_index: BimRowIndex(self.global_offset + self.local_line_num - 1),
                                            allele1: a1.to_string(),
                                            allele2: a2.to_string(),
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
                    }
                }
                None => return None,
            }
        }
    }
}

impl KWayMergeIterator {
    /// Creates a new K-way merge iterator. This constructor is responsible for
    /// opening all score files, parsing their headers to build column mappings,
    /// and positioning the readers to be ready for data streaming. This combined
    /// approach ensures headers are read exactly once, preventing them from being
    /// misinterpreted as data.
    fn new(
        file_paths: &[PathBuf],
        score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
    ) -> Result<Self, PrepError> {
        let mut streams = Vec::with_capacity(file_paths.len());
        let mut file_column_maps = Vec::with_capacity(file_paths.len());

        for path in file_paths {
            let file = File::open(path).map_err(|e| PrepError::Io(e, path.clone()))?;
            let mut reader = BufReader::new(file);
            let mut header_line = String::new();

            // Read past any metadata lines (starting with ##) to find the header.
            // The underlying `parse_score_file_headers_only` has already validated
            // that a valid, non-metadata header line exists.
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
                    break; // Found the header line.
                }
            }

            // The reader's position is now correctly after the header.

            let column_map: Vec<ScoreColumnIndex> = header_line
                .trim()
                .split('\t')
                .skip(3) // Skip variant_id, effect_allele, other_allele
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

            // Store the stream with the reader positioned ready for data.
            streams.push(FileStream {
                reader,
                line_buffer: std::collections::VecDeque::new(),
                current_line_info: None,
            });
        }

        let mut iter = Self {
            streams,
            heap: BinaryHeap::new(),
            file_column_maps,
            next_error: None,
        };

        // Prime the iterator by loading the first data line from each file.
        for i in 0..iter.streams.len() {
            if let Err(e) = iter.replenish_from_stream(i) {
                return Err(e);
            }
        }

        Ok(iter)
    }

    /// Attempts to replenish the heap with the next available score record from a given file stream.
    ///
    /// This function orchestrates the reading process for a single stream.
    /// If the stream's line buffer is not empty, it pushes the next score from it.
    /// If the buffer is empty, it triggers a read of the next line from the file.
    /// If a new line is successfully read and parsed, it pushes the *first* new score
    /// onto the heap. This ensures that each active stream is represented by exactly
    /// one item in the heap.
    ///
    /// # Returns
    /// - `Ok(())` on success (or if the stream is simply exhausted).
    /// - `Err(PrepError)` if a file I/O or parsing error occurs.
    fn replenish_from_stream(&mut self, file_idx: usize) -> Result<(), PrepError> {
        let stream = &mut self.streams[file_idx];

        // If the buffer for the current line is not empty, just push the next item.
        if !stream.line_buffer.is_empty() {
            Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
            return Ok(());
        }

        // If the buffer is empty, we must read a new line from the file.
        // The `?` operator will propagate any I/O or parsing errors.
        let column_map = &self.file_column_maps[file_idx];
        if Self::read_line_into_buffer(stream, column_map)? {
            // A new line was read successfully and contained scores.
            // Push the first score record onto the heap to represent this stream.
            Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap);
        }

        // If read_line_into_buffer returned `Ok(false)`, the stream is exhausted,
        // so we do nothing, and it will no longer be represented on the heap.
        Ok(())
    }

    /// Reads the next valid data line from a file stream and populates its internal buffer.
    /// This is a low-level helper called by `replenish_from_stream`.
    fn read_line_into_buffer(
        stream: &mut FileStream,
        column_map: &[ScoreColumnIndex],
    ) -> Result<bool, PrepError> {
        stream.line_buffer.clear();
        stream.current_line_info = None;

        let mut line = String::new();

        // This loop robustly handles files with empty lines or comments interspersed
        // with data, without using recursion.
        loop {
            line.clear();
            let bytes_read = stream
                .reader
                .read_line(&mut line)
                // The file path is not available here, so we pass an empty one. The
                // error will be contextualized by a higher-level caller.
                .map_err(|e| PrepError::Io(e, PathBuf::new()))?;

            if bytes_read == 0 {
                return Ok(false); // Clean end-of-file.
            }

            // Skip lines that are empty or are comments.
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // If we reach here, we have a line that is presumed to be data.
            break;
        }

        let mut parts = line.split('\t');
        let (variant_id, effect_allele) = match (parts.next(), parts.next()) {
            (Some(v), Some(e)) if !v.is_empty() && !e.is_empty() => (v, e),
            _ => return Ok(false), // Skip malformed line that lacks required columns.
        };
        let _other_allele = parts.next(); // Consume but do not use.

        let key = parse_key(variant_id)?;
        stream.current_line_info = Some((key, effect_allele.to_string()));

        // Populate the buffer with all scores from this line.
        for (i, weight_str) in parts.enumerate() {
            if let Ok(weight) = weight_str.parse::<f32>() {
                // Use the pre-parsed column map to get the global index.
                if let Some(&score_column_index) = column_map.get(i) {
                    stream.line_buffer.push_back((weight, score_column_index));
                }
            }
        }
        // Return true only if we actually buffered one or more valid scores.
        Ok(!stream.line_buffer.is_empty())
    }

    /// Creates a `KeyedScoreRecord` from the next item in a stream's buffer and
    /// pushes a corresponding `HeapItem` onto the heap.
    fn push_next_from_buffer_to_heap(
        stream: &mut FileStream,
        file_idx: usize,
        heap: &mut BinaryHeap<HeapItem>,
    ) {
        if let Some((weight, score_column_index)) = stream.line_buffer.pop_front() {
            // This unwrap is safe because line_buffer is only populated if current_line_info is Some.
            let (key, effect_allele) = stream.current_line_info.as_ref().unwrap().clone();
            let record = KeyedScoreRecord {
                key,
                effect_allele,
                score_column_index,
                weight,
            };
            heap.push(HeapItem { record, file_idx });
        }
    }
}

impl Iterator for KWayMergeIterator {
    type Item = Result<KeyedScoreRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        // If a terminal error occurred in the previous call, return it now and stop.
        if let Some(e) = self.next_error.take() {
            return Some(Err(e));
        }

        // Pop the smallest element. If heap is empty, we're done.
        let top_item = match self.heap.pop() {
            Some(item) => item,
            None => return None,
        };

        let record_to_return = top_item.record;
        let file_idx = top_item.file_idx;

        // Try to replenish the heap from the stream we just took from.
        if let Err(e) = self.replenish_from_stream(file_idx) {
            // An error occurred. Store it to be returned on the *next* call to `next()`.
            // This ensures we don't lose the record we just popped.
            self.next_error = Some(e);
        }

        // Return the valid record we popped.
        Some(Ok(record_to_return))
    }
}


fn build_bim_to_reconciled_variant_map(
    required_bim_indices: &[BimRowIndex],
    total_variants_in_bim: usize,
) -> Vec<Option<NonMaxU32>> {
    let mut bim_row_to_reconciled_variant_map = vec![None; total_variants_in_bim];
    for (reconciled_variant_idx, &bim_row) in required_bim_indices.iter().enumerate() {
        if (bim_row.0 as usize) < total_variants_in_bim {
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
pub fn parse_score_file_headers_only(
    score_files: &[PathBuf],
) -> Result<Vec<String>, PrepError> {
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

/// The single, authoritative resolver for matching a score line to BIM records.
fn resolve_matches_for_score_line<'a>(
    effect_allele: &str,
    bim_records_for_position: &'a [KeyedBimRecord],
) -> Result<ReconciliationOutcome<'a>, PrepError> {
    let is_multiallelic_site = bim_records_for_position.len() > 1;

    if is_multiallelic_site {
        let all_contexts_for_locus = bim_records_for_position.iter().collect();
        return Ok(ReconciliationOutcome::Complex(all_contexts_for_locus));
    }

    let mut simple_matches: BTreeMap<BimRowIndex, &'a KeyedBimRecord> =
        BTreeMap::new();
    for record_tuple in bim_records_for_position {
        if &record_tuple.allele1 == effect_allele || &record_tuple.allele2 == effect_allele {
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

// ========================================================================================
//                                    ERROR HANDLING
// ========================================================================================

impl Display for PrepError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            PrepError::Io(e, path) => write!(f, "I/O Error for file {}: {}", path.display(), e),
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
