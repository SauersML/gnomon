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


/// A self-contained item for the merge heap.
#[derive(Debug)]
struct HeapItem {
    key: VariantKey,
    record: KeyedScoreRecord,
    file_idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
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
        other.key.cmp(&self.key).then_with(|| other.file_idx.cmp(&self.file_idx))
    }
}

/// Manages the state of a single file reader in the KWayMergeIterator.
struct FileStream {
    reader: Lines<BufReader<File>>,
    /// A buffer for weights and column indices from the current line being processed.
    line_buffer: std::collections::VecDeque<(f32, ScoreColumnIndex)>,
    /// The key and effect allele for the currently buffered line.
    current_line_info: Option<(VariantKey, String)>,
}

/// An iterator that merges multiple, sorted score files on the fly. It correctly
/// handles multi-score lines, yielding one `KeyedScoreRecord` per score.
struct KWayMergeIterator<'a> {
    streams: Vec<FileStream>,
    heap: BinaryHeap<HeapItem>,
    file_column_maps: &'a [Vec<ScoreColumnIndex>],
}

/// An iterator that streams over one or more `.bim` files, creating a single
/// virtually merged and sorted stream of records.
struct BimIterator<'a> {
    fileset_prefixes: std::slice::Iter<'a, PathBuf>,
    current_reader: Option<Lines<BufReader<File>>>,
    global_offset: u64,
    local_line_num: u64,
    current_path: PathBuf,
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
    eprintln!("> Stage 2: Discovering all score columns...");
    let score_names = parse_score_file_headers_only(sorted_score_files)?;
    let score_name_to_col_index: AHashMap<String, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), ScoreColumnIndex(i)))
        .collect();
    let file_column_maps =
        KWayMergeIterator::parse_file_headers(sorted_score_files, &score_name_to_col_index)?;

    // --- STAGE 3: SINGLE-PASS STREAMING MERGE-JOIN ---
    eprintln!("> Stage 3: Streaming and merging data from genotype and score files...");
    let overall_start_time = Instant::now();

    let mut bim_iter = BimIterator::new(fileset_prefixes)?.peekable();
    let mut score_iter =
        KWayMergeIterator::new(sorted_score_files, &file_column_maps)?.peekable();

    type SimpleMapping = ((BimRowIndex, ScoreColumnIndex), (f32, bool));
    let mut simple_mappings: Vec<SimpleMapping> = Vec::new();
    let mut intermediate_complex_rules: AHashMap<Vec<(BimRowIndex, String, String)>, Vec<ScoreInfo>> = AHashMap::new();

    while bim_iter.peek().is_some() && score_iter.peek().is_some() {
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
                let mut bim_group = Vec::new();
                while let Some(Ok(peek_item)) = bim_iter.peek() {
                    if peek_item.key == key {
                        bim_group.push(bim_iter.next().unwrap()?);
                    } else {
                        break;
                    }
                }

                let mut score_group = Vec::new();
                while let Some(Ok(peek_item)) = score_iter.peek() {
                    if peek_item.key == key {
                        score_group.push(score_iter.next().unwrap()?);
                    } else {
                        break;
                    }
                }

                for score_record in score_group {
                    let outcome = resolve_matches_for_score_line(
                        &score_record.effect_allele,
                        &bim_group,
                    )?;
                    match outcome {
                        ReconciliationOutcome::Simple(matches) => {
                            for bim_rec in matches {
                                let is_flipped = score_record.effect_allele != bim_rec.allele1;
                                simple_mappings.push((
                                    (bim_rec.bim_row_index, score_record.score_column_index),
                                    (score_record.weight, is_flipped),
                                ));
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
                                effect_allele: score_record.effect_allele.clone(),
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
        "> TIMING: Stage 3 (Streaming Merge) took {:.2?}",
        overall_start_time.elapsed()
    );

    // --- STAGE 4: FINAL INDEXING & DATA PREPARATION ---
    eprintln!("> Stage 4: Building final variant index and populating matrices...");

    // Sort mappings by BIM index to group them for the uniqueness check.
    simple_mappings.par_sort_unstable_by_key(|((bim_row, _), _)| *bim_row);

    // Now, find the unique, sorted BIM indices required for the calculation.
    let mut required_bim_indices = Vec::new();
    if let Some(((first_bim_row, _), _)) = simple_mappings.first() {
        required_bim_indices.push(*first_bim_row);
    }
    for ((bim_row, _), _) in &simple_mappings {
        if bim_row.0 != required_bim_indices.last().unwrap().0 {
            required_bim_indices.push(*bim_row);
        }
    }

    let num_reconciled_variants = required_bim_indices.len();
    let bim_row_to_reconciled_variant_map =
        build_bim_to_reconciled_variant_map(&required_bim_indices, total_variants_in_bim_usize);

    let stride = (score_names.len() + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
    let mut weights_matrix = vec![0.0f32; num_reconciled_variants * stride];
    let mut flip_mask_matrix = vec![0u8; num_reconciled_variants * stride];
    let mut variant_to_scores_map: Vec<Vec<ScoreColumnIndex>> =
        vec![Vec::new(); num_reconciled_variants];
    let mut score_variant_counts = vec![0u32; score_names.len()];

    for ((bim_row, score_col), (weight, is_flipped)) in simple_mappings {
        if let Some(Some(reconciled_idx_nonmax)) =
            bim_row_to_reconciled_variant_map.get(bim_row.0 as usize)
        {
            let reconciled_idx = reconciled_idx_nonmax.get() as usize;
            let final_offset = reconciled_idx * stride + score_col.0;
            weights_matrix[final_offset] = weight;
            flip_mask_matrix[final_offset] = if is_flipped { 1 } else { 0 };
            variant_to_scores_map[reconciled_idx].push(score_col);
        }
    }

    // Process complex rules
    let final_complex_rules: Vec<GroupedComplexRule> = intermediate_complex_rules
        .into_iter()
        .map(|(contexts, scores)| GroupedComplexRule {
            possible_contexts: contexts,
            score_applications: scores,
        })
        .collect();

    if required_bim_indices.is_empty() && final_complex_rules.is_empty() {
        return Err(PrepError::NoOverlappingVariants);
    }

    // Deduplicate and sort the score maps in parallel
    variant_to_scores_map.par_iter_mut().for_each(|v| {
        v.sort_unstable();
        v.dedup();
    });

    // Count variants per score from the final populated structures
    for (_reconciled_idx, scores) in variant_to_scores_map.iter().enumerate() {
        for score_col in scores {
            score_variant_counts[score_col.0] += 1;
        }
    }

    // --- FINAL CONSTRUCTION ---
    let bytes_per_variant = (total_people_in_fam as u64 + 3) / 4;
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
        let boundaries = fileset_prefixes
            .iter()
            .scan(0, |offset, prefix| {
                let path = prefix.with_extension("bim");
                let count = count_lines(&path).unwrap_or(0);
                let boundary = FilesetBoundary {
                    bed_path: prefix.with_extension("bed"),
                    starting_global_index: *offset,
                };
                *offset += count;
                Some(boundary)
            })
            .collect();
        PipelineKind::MultiFile(boundaries)
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

    let chr_num = match chr_str {
        "X" | "x" => 23,
        "Y" | "y" => 24,
        "MT" | "mt" => 25,
        _ => chr_str.parse().map_err(|e| PrepError::Parse(format!("Invalid chromosome '{}': {}", chr_str, e)))?,
    };
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
        };
        iter.next_file()?;
        Ok(iter)
    }

    fn next_file(&mut self) -> Result<bool, PrepError> {
        if let Some(prefix) = self.fileset_prefixes.next() {
            self.global_offset += self.local_line_num;
            self.local_line_num = 0;
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

impl<'a> KWayMergeIterator<'a> {
    fn new(
        file_paths: &'a [PathBuf],
        file_column_maps: &'a [Vec<ScoreColumnIndex>],
    ) -> Result<Self, PrepError> {
        let mut streams = Vec::with_capacity(file_paths.len());
        for path in file_paths {
            let file = File::open(path).map_err(|e| PrepError::Io(e, path.clone()))?;
            streams.push(FileStream {
                reader: BufReader::new(file).lines(),
                line_buffer: std::collections::VecDeque::new(),
                current_line_info: None,
            });
        }

        let mut heap = BinaryHeap::new();
        for (i, stream) in streams.iter_mut().enumerate() {
            // Prime the iterator by loading the first item from each file.
            if let Err(e) = Self::advance_stream(stream, i, &mut heap, file_column_maps[i].as_slice()) {
                return Err(e);
            }
        }

        Ok(Self { streams, heap, file_column_maps })
    }

    /// Parses the header of each score file to map local column order to the global order.
    fn parse_file_headers(
        score_files: &[PathBuf],
        score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
    ) -> Result<Vec<Vec<ScoreColumnIndex>>, PrepError> {
        score_files
            .iter()
            .map(|path| {
                let file = File::open(path).map_err(|e| PrepError::Io(e, path.clone()))?;
                let mut reader = BufReader::new(file);
                let mut header_line = String::new();
                // Skip metadata lines, which start with ##
                loop {
                    header_line.clear();
                    if reader.read_line(&mut header_line).map_err(|e| PrepError::Io(e, path.clone()))? == 0 {
                        break; // End of file
                    }
                    if !header_line.starts_with("##") {
                        break; // Found the header
                    }
                }

                header_line
                    .trim()
                    .split('\t')
                    .skip(3) // Skip variant_id, effect_allele, other_allele
                    .map(|name| {
                        score_name_to_col_index.get(name).copied().ok_or_else(|| {
                            PrepError::Header(format!(
                                "Score '{}' from file '{}' not found in global score list.",
                                name,
                                path.display()
                            ))
                        })
                    })
                    .collect()
            })
            .collect()
    }

    /// Advances a single file stream. If the line buffer is empty, it reads the
    /// next line from the file, parses it, and populates the buffer. Then, it
    /// takes the next available item from the buffer and pushes it onto the heap.
    fn advance_stream(
        stream: &mut FileStream,
        file_idx: usize,
        heap: &mut BinaryHeap<HeapItem>,
        column_map: &[ScoreColumnIndex],
    ) -> Result<(), PrepError> {
        // If the buffer is empty, read a new line from the underlying file.
        if stream.line_buffer.is_empty() {
            stream.current_line_info = None; // Clear previous line info
            if let Some(Ok(line)) = stream.reader.next() {
                let mut parts = line.splitn(4, ' ');
                let variant_id = parts.next().unwrap_or("");
                let effect_allele = parts.next().unwrap_or("");
                let _other_allele = parts.next(); // Consumed but not used here
                let weights_str = parts.next().unwrap_or("");

                if variant_id.is_empty() || effect_allele.is_empty() {
                    return Ok(()); // Skip malformed or empty lines
                }

                let key = parse_key(variant_id)?;
                stream.current_line_info = Some((key, effect_allele.to_string()));

                // Populate the buffer with all scores from this line.
                for (i, weight_str) in weights_str.split(' ').enumerate() {
                    if let Ok(weight) = weight_str.parse::<f32>() {
                        if let Some(&score_column_index) = column_map.get(i) {
                            stream.line_buffer.push_back((weight, score_column_index));
                        }
                    }
                }
            }
        }

        // If, after attempting to read a new line, the buffer has an item,
        // create a HeapItem and push it.
        if let Some((weight, score_column_index)) = stream.line_buffer.pop_front() {
            if let Some((key, effect_allele)) = &stream.current_line_info {
                let record = KeyedScoreRecord {
                    key: *key,
                    effect_allele: effect_allele.clone(),
                    score_column_index,
                    weight,
                };
                heap.push(HeapItem { key: *key, record, file_idx });
            }
        }

        Ok(())
    }
}

impl<'a> Iterator for KWayMergeIterator<'a> {
    type Item = Result<KeyedScoreRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        // Pop the smallest item from the heap.
        let heap_item = self.heap.pop()?;
        let file_idx = heap_item.file_idx;

        // The stream that this item came from needs to be advanced.
        let stream = &mut self.streams[file_idx];
        let column_map = self.file_column_maps[file_idx].as_slice();

        // Advance the stream, which will push its next item to the heap.
        if let Err(e) = Self::advance_stream(stream, file_idx, &mut self.heap, column_map) {
            return Some(Err(e));
        }

        // Return the record from the item we popped.
        Some(Ok(heap_item.record))
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
                        "Score file '{}' is empty or contains only metadata lines.",
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
                    "Invalid header in '{}': Must start with 'variant_id\teffect_allele\tother_allele'.",
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