// ========================================================================================
//
//               The preparation "compiler"
//
// ========================================================================================
//
// This module transforms raw user inputs into an optimized "computation
// blueprint." It now uses a low-memory, high-throughput streaming merge-join
// algorithm to handle genome-scale data.

use crate::pipeline_error::PipelineError;
use crate::score::io::{TextSource, open_plink_text_source, open_text_source};
use crate::score::types::{
    BimRowIndex, FilesetBoundary, GenomicRegion, GroupedComplexRule, PersonSubset, PipelineKind,
    PreparationResult, ScoreColumnIndex, ScoreInfo, parse_chromosome_label,
};
use crate::score::types::{OriginalPersonIndex, OutputPersonIndex};
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::str::Utf8Error;
use std::sync::Arc;
use std::time::Instant;

// The number of SIMD lanes in the kernel. This MUST be kept in sync with kernel.rs.
const LANE_COUNT: usize = 8;

#[path = "prepare_cache.rs"]
mod cache;

#[path = "prepare_parse.rs"]
mod parse;

#[path = "prepare_scores.rs"]
mod scores;

// ========================================================================================
//              Type-driven domain model for streaming
// ========================================================================================

/// The primitive, sortable key used for all merge-join operations.
type VariantKey = (u8, u32);

// --- Internal Data Structures ---
// These temporary structs hold just enough owned string state for deterministic,
// streaming reconciliation without retaining genome-scale arena allocations.

#[derive(Clone)]
struct FilesetPaths {
    bed: PathBuf,
    bim: PathBuf,
    fam: PathBuf,
}

/// A parsed record from a `.bim` file.
#[derive(Debug, Clone)]
struct KeyedBimRecord {
    key: VariantKey,
    bim_row_index: BimRowIndex,
    allele1: Allele,
    allele2: Allele,
}

/// A parsed record from a score file.
#[derive(Debug, Clone)]
struct KeyedScoreRecord {
    key: VariantKey,
    effect_allele: Allele,
    other_allele: Allele,
    score_column_index: ScoreColumnIndex,
    weight: f32,
}

/// Most genome rows contain one of a handful of literal alleles. Borrow those
/// literals; share longer alleles across the score columns on the same row.
/// No normalization is performed here: matching retains the input's exact text.
#[derive(Debug, Clone)]
enum Allele {
    Literal(&'static str),
    Shared(Arc<str>),
}

impl Allele {
    fn new(value: &str) -> Self {
        let literal = match value {
            "A" => "A",
            "C" => "C",
            "G" => "G",
            "T" => "T",
            "N" => "N",
            "a" => "a",
            "c" => "c",
            "g" => "g",
            "t" => "t",
            "n" => "n",
            "0" => "0",
            "I" => "I",
            "D" => "D",
            "-" => "-",
            "." => ".",
            _ => return Self::Shared(Arc::from(value)),
        };
        Self::Literal(literal)
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Literal(value) => value,
            Self::Shared(value) => value,
        }
    }
}

impl AsRef<str> for Allele {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Display for Allele {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

// Manual implementation to handle f32 comparison correctly.
impl PartialEq for KeyedScoreRecord {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.weight.to_bits() == other.weight.to_bits()
    }
}
impl Eq for KeyedScoreRecord {}

/// A self-contained item for the merge heap.
#[derive(Debug, Clone)]
struct HeapItem {
    record: KeyedScoreRecord,
    file_idx: usize,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.record == other.record
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
        other
            .record
            .key
            .cmp(&self.record.key)
            .then_with(|| other.file_idx.cmp(&self.file_idx))
    }
}

/// Manages the state of a single file reader in the KWayMergeIterator.
struct FileStream {
    reader: BufReader<File>,
    /// A buffer for weights and column indices from the current line being processed.
    line_buffer: std::collections::VecDeque<(f32, ScoreColumnIndex)>,
    /// The key and alleles for the current buffered line.
    current_line_info: Option<(VariantKey, Allele, Allele)>,
    // Temporary buffer reused for reading raw line data from the file.
    line_string_buffer: String,
    /// 1-based line number in the currently read file.
    file_line_number: u64,
    /// Counter for malformed lines in this specific file stream.
    malformed_lines_count: usize,
}

#[derive(Debug, Copy, Clone)]
struct SimpleScoreAssignment {
    // Weight applied to effect-allele dosage in canonical (BIM allele2) space.
    dosage_weight: f32,
    // Correction to subtract for missing calls at this variant/score cell.
    missing_correction: f32,
}

/// Lock-step CSR builder that guarantees aligned sparse vectors and valid row offsets.
struct CsrBuilder {
    sparse_weights: Vec<f32>,
    sparse_missing_corrections: Vec<f32>,
    sparse_score_columns: Vec<u32>,
    sparse_row_offsets: Vec<u64>,
}

impl CsrBuilder {
    fn new() -> Result<Self, PrepError> {
        let mut sparse_row_offsets = Vec::<u64>::new();
        sparse_row_offsets.try_reserve_exact(1).map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate CSR row offsets: {e}"))
        })?;
        sparse_row_offsets.push(0);
        Ok(Self {
            sparse_weights: Vec::new(),
            sparse_missing_corrections: Vec::new(),
            sparse_score_columns: Vec::new(),
            sparse_row_offsets,
        })
    }

    fn push_contribution(
        &mut self,
        score_col_idx: ScoreColumnIndex,
        assignment: SimpleScoreAssignment,
    ) -> Result<(), PrepError> {
        let col_u32 = u32::try_from(score_col_idx.0).map_err(|_| {
            PrepError::Invariant(format!(
                "Score column index {} exceeds u32::MAX while building CSR.",
                score_col_idx.0
            ))
        })?;
        // Reserve every parallel array before changing any length. Vec::push
        // alone would abort on allocation failure while growing a large panel.
        self.sparse_score_columns.try_reserve(1).map_err(|e| {
            PrepError::Invariant(format!("Cannot grow CSR score columns: {e}"))
        })?;
        self.sparse_weights.try_reserve(1).map_err(|e| {
            PrepError::Invariant(format!("Cannot grow CSR weights: {e}"))
        })?;
        self.sparse_missing_corrections.try_reserve(1).map_err(|e| {
            PrepError::Invariant(format!("Cannot grow CSR missing corrections: {e}"))
        })?;
        self.sparse_score_columns.push(col_u32);
        self.sparse_weights.push(assignment.dosage_weight);
        self.sparse_missing_corrections
            .push(assignment.missing_correction);
        Ok(())
    }

    fn finish_variant(&mut self) -> Result<(), PrepError> {
        let offset_u64 = u64::try_from(self.sparse_score_columns.len()).map_err(|_| {
            PrepError::Invariant(format!(
                "CSR non-zero count {} exceeds u64::MAX while building row offsets.",
                self.sparse_score_columns.len()
            ))
        })?;
        self.sparse_row_offsets.try_reserve(1).map_err(|e| {
            PrepError::Invariant(format!("Cannot grow CSR row offsets: {e}"))
        })?;
        self.sparse_row_offsets.push(offset_u64);
        Ok(())
    }

    fn into_parts(self) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u64>) {
        (
            self.sparse_weights,
            self.sparse_missing_corrections,
            self.sparse_score_columns,
            self.sparse_row_offsets,
        )
    }

    /// Reorders rows so their `.bim` indices ascend, carrying each row's flag
    /// and entries with it. A join over rows sorted by key emits rows in key
    /// order, but readers visit rows in file order.
    fn sort_rows_by_bim_index(
        &mut self,
        required: &mut Vec<BimRowIndex>,
        flags: &mut Vec<u8>,
    ) -> Result<(), PrepError> {
        if self.sparse_row_offsets.len() != required.len() + 1 || flags.len() != required.len() {
            return Err(PrepError::Invariant(format!(
                "CSR rows ({}) and matched variants ({}) disagree before reordering.",
                self.sparse_row_offsets.len() - 1,
                required.len()
            )));
        }
        if required.windows(2).all(|pair| pair[0] < pair[1]) {
            return Ok(());
        }
        let mut order = Vec::new();
        order.try_reserve_exact(required.len()).map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate CSR row permutation: {e}"))
        })?;
        order.extend(0..required.len());
        order.sort_unstable_by_key(|&row| required[row]);
        // Each `.bim` row has one key, so no row can have been emitted twice.
        if order
            .windows(2)
            .any(|pair| required[pair[0]] == required[pair[1]])
        {
            return Err(PrepError::Invariant(
                "A .bim row was matched under two different keys.".to_string(),
            ));
        }
        // Finish and release each old column before allocating the next one.
        // Temporary entry storage is one array, not a second complete CSR.
        fn reorder<T: Copy>(
            values: &[T],
            offsets: &[u64],
            order: &[usize],
        ) -> Result<Vec<T>, PrepError> {
            let mut sorted = Vec::new();
            sorted.try_reserve_exact(values.len()).map_err(|e| {
                PrepError::Invariant(format!("Cannot allocate CSR reordered values: {e}"))
            })?;
            for &row in order {
                sorted.extend_from_slice(&values[offsets[row] as usize..offsets[row + 1] as usize]);
            }
            Ok(sorted)
        }
        self.sparse_weights = reorder(&self.sparse_weights, &self.sparse_row_offsets, &order)?;
        self.sparse_missing_corrections = reorder(
            &self.sparse_missing_corrections,
            &self.sparse_row_offsets,
            &order,
        )?;
        self.sparse_score_columns =
            reorder(&self.sparse_score_columns, &self.sparse_row_offsets, &order)?;
        let mut offsets = Vec::new();
        let mut indices = Vec::new();
        let mut row_flags = Vec::new();
        offsets.try_reserve_exact(order.len() + 1).map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate reordered CSR offsets: {e}"))
        })?;
        indices.try_reserve_exact(order.len()).map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate reordered BIM indices: {e}"))
        })?;
        row_flags.try_reserve_exact(order.len()).map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate reordered complex flags: {e}"))
        })?;
        offsets.push(0);
        for &row in &order {
            offsets.push(
                offsets.last().unwrap() + self.sparse_row_offsets[row + 1]
                    - self.sparse_row_offsets[row],
            );
            indices.push(required[row]);
            row_flags.push(flags[row]);
        }
        self.sparse_row_offsets = offsets;
        *required = indices;
        *flags = row_flags;
        Ok(())
    }
}

#[inline(always)]
fn apply_simple_score_assignment(entry: &mut SimpleScoreAssignment, weight: f32, is_flipped: bool) {
    // Canonicalize every match into allele2-dosage space.
    // If the score effect allele matches BIM allele1, the row contributes:
    //   weight * (2 - dosage_allele2)
    // = (2*weight) + (-weight * dosage_allele2).
    if is_flipped {
        entry.dosage_weight -= weight;
        entry.missing_correction += 2.0 * weight;
    } else {
        entry.dosage_weight += weight;
    }
}

#[inline(always)]
fn allele_pair_matches(
    effect_allele: &str,
    other_allele: &str,
    bim_a1: &str,
    bim_a2: &str,
) -> bool {
    (effect_allele == bim_a1 && other_allele == bim_a2)
        || (effect_allele == bim_a2 && other_allele == bim_a1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineReadOutcome {
    Pushed,
    Skipped,
    Eof,
}

/// An iterator that merges multiple, sorted score files on the fly.
struct KWayMergeIterator {
    streams: Vec<FileStream>,
    heap: BinaryHeap<HeapItem>,
    file_column_maps: Vec<Vec<ScoreColumnIndex>>,
    // Holds a terminal error. If Some, iteration will stop after yielding the error.
    next_error: Option<PrepError>,
    region_filters: Option<Vec<Option<GenomicRegion>>>,
    region_filter_hits: Option<Vec<bool>>,
}

/// An iterator that streams over one or more `.bim` files.
struct BimIterator<'a> {
    filesets: std::slice::Iter<'a, FilesetPaths>,
    current_reader: Option<Box<dyn TextSource + 'a>>,
    global_offset: u64,
    local_line_num: u64,
    current_path: PathBuf,
    // A list of the file boundaries, collected on-the-fly during the single iteration pass.
    boundaries: Vec<FilesetBoundary>,
    total_variants: u64,
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
    /// Track any region filters supplied by the user so we can surface them in diagnostics.
    active_region_filters: Vec<(String, GenomicRegion)>,
    region_filters_without_hits: Vec<String>,
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

    fn record_region_filters(&mut self, filters: &HashMap<String, GenomicRegion>) {
        if filters.is_empty() {
            return;
        }

        self.active_region_filters = filters
            .iter()
            .map(|(name, region)| (name.clone(), *region))
            .collect();
        self.active_region_filters
            .sort_unstable_by(|a, b| a.0.cmp(&b.0));
    }

    fn record_region_without_hits(&mut self, score_name: &str) {
        self.region_filters_without_hits
            .push(score_name.to_string());
    }
}

#[derive(Debug)]
pub enum PrepError {
    Io(io::Error, PathBuf),
    Parse(String),
    Header(String),
    InconsistentKeepId(String),
    PipelineIo {
        path: PathBuf,
        message: String,
    },
    /// An error indicating that no variants from the score files could be matched
    /// to variants in the genotype data.
    NoOverlappingVariants(MergeDiagnosticInfo),
    UnsortedInput {
        source: &'static str,
        path: PathBuf,
        line_number: u64,
        previous_key: VariantKey,
        current_key: VariantKey,
    },
    GenomeBuildMismatch,
    DisjointChromosomes,
    AmbiguousReconciliation(String),
    Invariant(String),
}

enum PostMortemAction {
    None,
    Fatal(PrepError),
}

pub fn prepare_for_computation(
    fileset_prefixes: &[PathBuf],
    sorted_score_files: &[PathBuf],
    keep_file: Option<&Path>,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
) -> Result<PreparationResult, PrepError> {
    let filesets = build_fileset_paths(fileset_prefixes)?;
    // The plan cache only saves time. A plan that cannot be hashed, read, trusted or
    // held within this machine's memory budget is compiled again instead.
    let cache = match cache::PlanCache::discover(&filesets, sorted_score_files, score_regions) {
        Ok(cache) => cache,
        Err(error) => {
            eprintln!("> Compiled variant plans are unavailable for these inputs: {error}.");
            None
        }
    };
    if let Some(cache) = &cache {
        match cache.load() {
            Ok(Some(plan)) => {
                eprintln!(
                    "> Reusing content-verified compiled variant plan ({} matched rows).",
                    plan.required.len()
                );
                let (all_iids, lookup) = parse_fam_and_build_lookup(&filesets)?;
                let (subset, iids) = resolve_person_subset(keep_file, &all_iids, &lookup)?;
                return assemble_preparation(
                    plan,
                    &filesets,
                    subset,
                    iids,
                    all_iids.len(),
                    &lookup,
                );
            }
            Ok(None) => {}
            Err(error) => {
                eprintln!(
                    "> Compiling the variant plan again; the saved one was not used: {error}."
                );
            }
        }
    }
    let (prep, clean) = prepare_for_computation_with_retry(
        fileset_prefixes,
        sorted_score_files,
        keep_file,
        score_regions,
        BimRowOrder::Streamed,
    )?;
    if clean && let Some(cache) = &cache {
        // Rehash after compilation so a changed source cannot be published
        // under the digest taken before the compiler opened its readers. A
        // rehash that cannot run now, such as when memory has become short,
        // publishes nothing; only a different digest means the inputs changed.
        match cache::PlanCache::discover(&filesets, sorted_score_files, score_regions) {
            Ok(Some(current)) if !cache.same_inputs(&current) => {
                return Err(PrepError::Invariant(
                    "Variant inputs changed during compilation; retry with stable inputs.".into(),
                ));
            }
            Ok(Some(_)) => {
                if let Err(error) = cache.save(&prep) {
                    eprintln!("> Compiled variant plan was not saved: {error}.");
                }
            }
            Ok(None) => {}
            Err(error) => eprintln!("> Compiled variant plan was not saved: {error}."),
        }
    }
    Ok(prep)
}

fn prepare_for_computation_with_retry(
    fileset_prefixes: &[PathBuf],
    sorted_score_files: &[PathBuf],
    keep_file: Option<&Path>,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    bim_row_order: BimRowOrder,
) -> Result<(PreparationResult, bool), PrepError> {
    // --- Stage 1: Initial setup ---
    eprintln!("> Stage 1: Indexing subject data...");
    let fileset_paths = build_fileset_paths(fileset_prefixes)?;
    let (all_person_iids, iid_to_original_idx) = parse_fam_and_build_lookup(&fileset_paths)?;
    let total_people_in_fam = all_person_iids.len();

    let (person_subset, final_person_iids) =
        resolve_person_subset(keep_file, &all_person_iids, &iid_to_original_idx)?;

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

    let mut diagnostics = MergeDiagnosticInfo::default();
    if let Some(regions) = score_regions {
        diagnostics.record_region_filters(regions);
    }
    let mut seen_invalid_bim_chrs: AHashSet<String> = AHashSet::new();
    let mut seen_invalid_score_chrs: AHashSet<String> = AHashSet::new();

    // Local `.bim` files within the memory budget are read whole and parsed on
    // the pool; anything else streams line by line.
    let mut bim_iterator = None;
    let mut parsed_layout = None;
    let mut bim_rows = match parse::parse_local_bims(&fileset_paths) {
        Some(parse::ParsedBim {
            records,
            errors,
            boundaries,
            total_variants,
        }) => {
            let rows = BimRows::parsed(records, errors, &boundaries, &mut seen_invalid_bim_chrs);
            parsed_layout = Some((total_variants, boundaries));
            rows
        }
        None => {
            let iterator = bim_iterator.insert(BimIterator::new(&fileset_paths)?);
            match bim_row_order {
                BimRowOrder::Streamed => BimRows::streamed(iterator),
                BimRowOrder::Sorted => BimRows::sorted(iterator, &mut seen_invalid_bim_chrs)?,
            }
        }
    };
    let region_filters = score_regions.and_then(|regions| {
        let mut has_any = false;
        let mut filters = Vec::with_capacity(score_names.len());
        for name in &score_names {
            let region = regions.get(name).copied();
            if region.is_some() {
                has_any = true;
            }
            filters.push(region);
        }
        has_any.then_some(filters)
    });

    // Score files within the budget are parsed whole on the pool as well. Region
    // filters, and files the streaming merge would refuse, stream as before.
    let parsed_scores = match region_filters {
        None => scores::parse_score_files(sorted_score_files, &score_name_to_col_index),
        Some(_) => None,
    };
    let mut score_iterator = match parsed_scores {
        Some(parsed) => ScoreRows::Parsed(parsed),
        None => ScoreRows::Streamed(KWayMergeIterator::new(
            sorted_score_files,
            &score_name_to_col_index,
            region_filters.clone(),
        )?),
    };

    let rows_sorted_by_key = matches!(bim_rows, BimRows::Sorted(_));
    let mut bim_iter = bim_rows.by_ref().peekable();
    let mut score_iter = score_iterator.by_ref().peekable();

    let score_lane_groups = score_names.len().div_ceil(LANE_COUNT);
    let stride = score_lane_groups.checked_mul(LANE_COUNT).ok_or_else(|| {
        PrepError::Invariant(format!(
            "Stride overflow while padding scores: num_scores={}, lane_count={LANE_COUNT}",
            score_names.len()
        ))
    })?;
    if stride % LANE_COUNT != 0 {
        return Err(PrepError::Invariant(format!(
            "Invalid padded stride {stride}: must be divisible by lane count {LANE_COUNT}."
        )));
    }

    // Build final artifacts incrementally during Stage 3 to avoid materializing
    // genome-scale intermediate maps that duplicate the final CSR/rule structures.
    let mut required_bim_indices: Vec<BimRowIndex> = Vec::new();
    let mut required_is_complex: Vec<u8> = Vec::new();
    let mut csr_builder = CsrBuilder::new()?;
    let mut baseline_missing_sum_by_score = vec![0.0f64; score_names.len()];
    let mut score_variant_counts = vec![0u32; score_names.len()];
    let mut final_complex_rules: Vec<GroupedComplexRule> = Vec::new();
    let mut bim_group = Vec::new();
    let mut score_group = Vec::new();
    // Reuse score slots across singleton loci. Only touched columns are visited
    // or cleared, so sparse panels do not incur a full score-panel scan per locus.
    let mut simple_assignments = Vec::new();
    simple_assignments
        .try_reserve_exact(score_names.len())
        .map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate score reconciliation slots: {e}"))
        })?;
    simple_assignments.resize(score_names.len(), None::<SimpleScoreAssignment>);
    let mut touched_columns = Vec::new();
    touched_columns
        .try_reserve_exact(score_names.len())
        .map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate score reconciliation columns: {e}"))
        })?;

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

                bim_group.clear();
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

                score_group.clear();
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

                // A single marker and a single weight need no temporary trees,
                // sets, context vectors, or match lists. Emit their CSR row in
                // exactly the same arithmetic and record order as grouped loci.
                if let ([bim], [score]) = (bim_group.as_slice(), score_group.as_slice()) {
                    if allele_pair_matches(
                        score.effect_allele.as_str(),
                        score.other_allele.as_str(),
                        bim.allele1.as_str(),
                        bim.allele2.as_str(),
                    ) {
                        let mut assignment = SimpleScoreAssignment {
                            dosage_weight: 0.0,
                            missing_correction: 0.0,
                        };
                        apply_simple_score_assignment(
                            &mut assignment,
                            score.weight,
                            score.effect_allele.as_str() == bim.allele1.as_str(),
                        );
                        required_bim_indices.push(bim.bim_row_index);
                        required_is_complex.push(0);
                        csr_builder.push_contribution(score.score_column_index, assignment)?;
                        csr_builder.finish_variant()?;
                        baseline_missing_sum_by_score[score.score_column_index.0] +=
                            assignment.missing_correction as f64;
                        score_variant_counts[score.score_column_index.0] += 1;
                    }
                    continue;
                }

                if let [bim] = bim_group.as_slice() {
                    for score in &score_group {
                        if !allele_pair_matches(
                            score.effect_allele.as_str(),
                            score.other_allele.as_str(),
                            bim.allele1.as_str(),
                            bim.allele2.as_str(),
                        ) {
                            continue;
                        }
                        let slot = &mut simple_assignments[score.score_column_index.0];
                        let assignment = slot.get_or_insert_with(|| {
                            touched_columns.push(score.score_column_index);
                            SimpleScoreAssignment {
                                dosage_weight: 0.0,
                                missing_correction: 0.0,
                            }
                        });
                        // Input order matters for duplicate f32 additions.
                        apply_simple_score_assignment(
                            assignment,
                            score.weight,
                            score.effect_allele.as_str() == bim.allele1.as_str(),
                        );
                    }
                    if !touched_columns.is_empty() {
                        touched_columns.sort_unstable();
                        required_bim_indices.push(bim.bim_row_index);
                        required_is_complex.push(0);
                        for column in touched_columns.drain(..) {
                            let assignment = simple_assignments[column.0].take().unwrap();
                            csr_builder.push_contribution(column, assignment)?;
                            baseline_missing_sum_by_score[column.0] +=
                                assignment.missing_correction as f64;
                            score_variant_counts[column.0] += 1;
                        }
                        csr_builder.finish_variant()?;
                    }
                    continue;
                }

                let mut complex_for_key: BTreeMap<
                    Vec<(BimRowIndex, String, String)>,
                    Vec<(ScoreColumnIndex, f32, String, String)>,
                > = BTreeMap::new();

                for score_record in score_group.drain(..) {
                    let possible_contexts: Vec<_> = bim_group
                        .iter()
                        .filter(|rec| {
                            allele_pair_matches(
                                score_record.effect_allele.as_str(),
                                score_record.other_allele.as_str(),
                                rec.allele1.as_str(),
                                rec.allele2.as_str(),
                            )
                        })
                        .map(|rec| {
                            (
                                rec.bim_row_index,
                                rec.allele1.to_string(),
                                rec.allele2.to_string(),
                            )
                        })
                        .collect();
                    if possible_contexts.is_empty() {
                        continue;
                    }
                    let score_info = (
                        score_record.score_column_index,
                        score_record.weight,
                        score_record.effect_allele.to_string(),
                        score_record.other_allele.to_string(),
                    );
                    complex_for_key
                        .entry(possible_contexts)
                        .or_default()
                        .push(score_info);
                }

                // Finalize complex rules for this key immediately.
                let mut key_complex_indices: BTreeSet<BimRowIndex> = BTreeSet::new();
                for (contexts, scores) in complex_for_key {
                    for (bim_idx, _, _) in &contexts {
                        key_complex_indices.insert(*bim_idx);
                    }

                    for (score_col_idx, _, _, _) in &scores {
                        score_variant_counts[score_col_idx.0] += 1;
                    }

                    let chr_str = match key.0 {
                        23 => "X".to_string(),
                        24 => "Y".to_string(),
                        25 => "MT".to_string(),
                        n => n.to_string(),
                    };

                    final_complex_rules.push(GroupedComplexRule {
                        locus_chr_pos: (chr_str, key.1),
                        possible_contexts: contexts,
                        score_applications: scores
                            .into_iter()
                            .map(|(sc_idx, weight, ea, oa)| ScoreInfo {
                                effect_allele: ea,
                                other_allele: oa,
                                weight,
                                score_column_index: sc_idx,
                            })
                            .collect(),
                    });
                }

                // Emit CSR rows and required variant metadata for this key in sorted order.
                // This preserves global row ordering while avoiding a global index set.
                for bim_row_index in key_complex_indices {
                    required_bim_indices.push(bim_row_index);
                    required_is_complex.push(1);
                    csr_builder.finish_variant()?;
                }
            }
        }
    }
    for result in bim_iter.by_ref() {
        match result {
            Ok(record) => {
                diagnostics.add_bim_key(record.key);
                diagnostics.total_bim_variants_processed += 1;
            }
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
    if let Some(unsorted_bim) = bim_rows.descended_in() {
        // The merge-join may already have walked past rows it should have
        // matched. Redo it over every row sorted by key; the genotype files
        // stay as they are.
        eprintln!(
            "> Variants in {} are not sorted by chromosome and position. Matching them in sorted order...",
            unsorted_bim.display()
        );
        return prepare_for_computation_with_retry(
            fileset_prefixes,
            sorted_score_files,
            keep_file,
            score_regions,
            BimRowOrder::Sorted,
        );
    }
    if rows_sorted_by_key {
        // Rows were emitted in key order; readers visit them in file order.
        csr_builder.sort_rows_by_bim_index(&mut required_bim_indices, &mut required_is_complex)?;
    }

    let region_filter_hits = score_iterator.take_region_filter_hits();
    if let (Some(filters), Some(hit_flags)) = (region_filters.as_ref(), region_filter_hits.as_ref())
    {
        for (idx, region_opt) in filters.iter().enumerate() {
            if let Some(region) = region_opt
                && !hit_flags.get(idx).copied().unwrap_or(false)
            {
                let score_name = &score_names[idx];
                eprintln!(
                    "Warning: Score '{score_name}' has no variants within the requested region {region}."
                );
                diagnostics.record_region_without_hits(score_name);
            }
        }
    }

    let (total_variants_in_bim, bim_boundaries) = match (parsed_layout, bim_iterator) {
        (Some(layout), _) => layout,
        (None, Some(iterator)) => (iterator.total_variants(), iterator.boundaries),
        (None, None) => {
            return Err(PrepError::Invariant(
                "Stage 3 had neither parsed nor streamed .bim rows.".to_string(),
            ));
        }
    };
    eprintln!(
        "> TIMING: Stage 3 (Data Collection) took {:.2?}",
        overall_start_time.elapsed()
    );

    // After iterating, sum malformed line counts from all streams
    let total_malformed_lines = score_iterator.malformed_lines();

    if total_malformed_lines > 0 {
        eprintln!(
            "> Warning: Skipped {total_malformed_lines} lines from score files due to missing columns (variant_id, effect_allele, other_allele)."
        );
    }

    // --- Stage 4: Verifying data and finalizing matrix metadata ---
    eprintln!("> Stage 4: Verifying data and building final matrices...");

    if required_bim_indices.is_empty() {
        match conduct_post_mortem(&fileset_paths, sorted_score_files)? {
            PostMortemAction::Fatal(err) => return Err(err),
            PostMortemAction::None => return Err(PrepError::NoOverlappingVariants(diagnostics)),
        }
    }

    let num_reconciled_variants = required_bim_indices.len();

    let (sparse_weights, sparse_missing_corrections, sparse_score_columns, sparse_row_offsets) =
        csr_builder.into_parts();

    if sparse_weights.len() != sparse_missing_corrections.len()
        || sparse_weights.len() != sparse_score_columns.len()
    {
        return Err(PrepError::Invariant(format!(
            "CSR vector length mismatch: weights={}, missing_corrections={}, score_columns={}",
            sparse_weights.len(),
            sparse_missing_corrections.len(),
            sparse_score_columns.len()
        )));
    }

    if sparse_row_offsets.len() != num_reconciled_variants + 1 {
        return Err(PrepError::Invariant(format!(
            "CSR row offset length mismatch: got {}, expected {}",
            sparse_row_offsets.len(),
            num_reconciled_variants + 1
        )));
    }
    if sparse_row_offsets.first().copied().unwrap_or_default() != 0 {
        return Err(PrepError::Invariant(
            "CSR row offsets must start at 0.".to_string(),
        ));
    }
    if sparse_row_offsets.windows(2).any(|w| w[1] < w[0]) {
        return Err(PrepError::Invariant(
            "CSR row offsets must be non-decreasing.".to_string(),
        ));
    }
    let expected_tail = u64::try_from(sparse_weights.len()).map_err(|_| {
        PrepError::Invariant(format!(
            "CSR non-zero count {} exceeds u64::MAX during final validation.",
            sparse_weights.len()
        ))
    })?;
    if sparse_row_offsets.last().copied().unwrap_or_default() != expected_tail {
        return Err(PrepError::Invariant(format!(
            "CSR row offset tail mismatch: tail={}, expected={expected_tail}",
            sparse_row_offsets.last().copied().unwrap_or_default()
        )));
    }
    if sparse_score_columns
        .iter()
        .any(|&col| col as usize >= score_names.len())
    {
        return Err(PrepError::Invariant(format!(
            "CSR score column contains out-of-range index for {} scores.",
            score_names.len()
        )));
    }

    let clean = total_malformed_lines == 0
        && seen_invalid_bim_chrs.is_empty()
        && seen_invalid_score_chrs.is_empty()
        && region_filters
            .as_ref()
            .zip(region_filter_hits.as_ref())
            .is_none_or(|(filters, hits)| {
                filters
                    .iter()
                    .zip(hits)
                    .all(|(region, hit)| region.is_none() || *hit)
            });
    let plan = cache::VariantPlan {
        weights: sparse_weights,
        corrections: sparse_missing_corrections,
        columns: sparse_score_columns,
        offsets: sparse_row_offsets,
        baseline: baseline_missing_sum_by_score,
        required: required_bim_indices,
        complex: final_complex_rules,
        names: score_names,
        counts: score_variant_counts,
        flags: required_is_complex,
        total_variants: total_variants_in_bim,
        starts: bim_boundaries
            .iter()
            .map(|b| b.starting_global_index)
            .collect(),
    };
    Ok((
        assemble_preparation(
            plan,
            &fileset_paths,
            person_subset,
            final_person_iids,
            total_people_in_fam,
            &iid_to_original_idx,
        )?,
        clean,
    ))
}

fn assemble_preparation(
    plan: cache::VariantPlan,
    fileset_paths: &[FilesetPaths],
    person_subset: PersonSubset,
    final_person_iids: Vec<String>,
    total_people_in_fam: usize,
    iid_to_original_idx: &AHashMap<String, u32>,
) -> Result<PreparationResult, PrepError> {
    if plan.starts.len() != fileset_paths.len() {
        return Err(PrepError::Invariant(
            "Variant plan fileset count mismatch.".into(),
        ));
    }
    let num_people_to_score = final_person_iids.len();
    let num_reconciled_variants = plan.required.len();
    let stride = plan
        .names
        .len()
        .div_ceil(LANE_COUNT)
        .checked_mul(LANE_COUNT)
        .ok_or_else(|| PrepError::Invariant("Score stride overflow.".into()))?;
    let bytes_per_variant = (total_people_in_fam as u64).div_ceil(4);
    let bytes_per_variant_usize = bytes_per_variant as usize;
    let (spool_compact_byte_index, spool_dense_map) =
        build_spool_maps(&person_subset, bytes_per_variant_usize);
    let spool_bytes_per_variant = spool_compact_byte_index.len() as u64;
    debug_assert_eq!(
        spool_bytes_per_variant as usize,
        spool_compact_byte_index.len(),
        "spool bytes per variant must equal compact index length"
    );
    let mut output_idx_to_fam_idx = Vec::with_capacity(num_people_to_score);
    let mut person_fam_to_output_idx = vec![None; total_people_in_fam];

    for (output_idx, iid) in final_person_iids.iter().enumerate() {
        let original_fam_idx = *iid_to_original_idx.get(iid).unwrap();
        output_idx_to_fam_idx.push(OriginalPersonIndex(original_fam_idx));
        let output_idx_u32 = u32::try_from(output_idx).map_err(|_| {
            PrepError::Invariant(format!(
                "Output person index {output_idx} exceeds u32::MAX."
            ))
        })?;
        person_fam_to_output_idx[original_fam_idx as usize] =
            Some(OutputPersonIndex(output_idx_u32));
    }

    let pipeline_kind = if fileset_paths.len() <= 1 {
        PipelineKind::SingleFile(fileset_paths[0].bed.clone())
    } else {
        PipelineKind::MultiFile(
            fileset_paths
                .iter()
                .zip(&plan.starts)
                .map(|(files, &start)| FilesetBoundary {
                    bed_path: files.bed.clone(),
                    bim_path: files.bim.clone(),
                    fam_path: files.fam.clone(),
                    starting_global_index: start,
                })
                .collect(),
        )
    };

    Ok(PreparationResult::new(
        plan.weights,
        plan.corrections,
        plan.columns,
        plan.offsets,
        stride,
        plan.baseline,
        plan.required,
        plan.complex,
        plan.names,
        plan.counts,
        person_subset,
        final_person_iids,
        num_people_to_score,
        total_people_in_fam,
        plan.total_variants,
        num_reconciled_variants,
        bytes_per_variant,
        person_fam_to_output_idx,
        output_idx_to_fam_idx,
        plan.flags,
        spool_compact_byte_index,
        spool_dense_map,
        spool_bytes_per_variant,
        pipeline_kind,
    ))
}

// ========================================================================================
//                             Private implementation helpers
// ========================================================================================

fn build_fileset_paths(prefixes: &[PathBuf]) -> Result<Vec<FilesetPaths>, PrepError> {
    prefixes
        .iter()
        .map(|prefix| {
            // A prefix may name either a PLINK 1.9 triple (.bed/.bim/.fam) or a
            // PLINK 2 one (.pgen/.pvar/.psam). Everything downstream consumes
            // the PLINK 1.9 shape; the PLINK 2 files are adapted on read.
            let bed = apply_extension(prefix, "bed")?;
            if uses_pgen_fileset(prefix, &bed) {
                return Ok(FilesetPaths {
                    bed: apply_extension(prefix, "pgen")?,
                    bim: apply_extension(prefix, "pvar")?,
                    fam: apply_extension(prefix, "psam")?,
                });
            }
            let bim = apply_extension(prefix, "bim")?;
            let fam = apply_extension(prefix, "fam")?;
            Ok(FilesetPaths { bed, bim, fam })
        })
        .collect()
}

/// Whether `prefix` should be read as a PLINK 2 fileset.
///
/// PLINK 1.9 always wins when present, so existing filesets behave exactly as
/// before and PLINK 2 is selected only when there is no `.bed` to read.
/// Remote prefixes are probed by opening the `.bim`, which is metadata-only for
/// the object stores we support — one request per fileset, not a download.
fn uses_pgen_fileset(prefix: &Path, bed: &Path) -> bool {
    if is_remote_path(prefix) {
        let Ok(bim) = apply_extension(prefix, "bim") else {
            return false;
        };
        return open_text_source(&bim).is_err();
    }
    !bed.is_file() && apply_extension(prefix, "pgen").is_ok_and(|p| p.is_file())
}

fn is_remote_path(path: &Path) -> bool {
    path.to_str().is_some_and(|s| {
        s.starts_with("gs://") || s.starts_with("http://") || s.starts_with("https://")
    })
}

/// Builds the compacted spool index structures for the selected cohort subset.
///
/// The returned `Vec<u32>` is guaranteed to be sorted and unique, providing the
/// exact byte positions (in the original PLINK layout) that contain at least one
/// kept individual. The dense map mirrors the original byte indices and either
/// contains the compacted index or `-1` if no kept person resides in that byte.
fn build_spool_maps(
    person_subset: &PersonSubset,
    bytes_per_variant_usize: usize,
) -> (Vec<u32>, Vec<i32>) {
    match person_subset {
        PersonSubset::All => {
            let mut compact = Vec::with_capacity(bytes_per_variant_usize);
            let mut dense = Vec::with_capacity(bytes_per_variant_usize);
            for i in 0..bytes_per_variant_usize {
                compact.push(
                    u32::try_from(i).expect(
                        "Too many bytes per variant to represent in spool_compact_byte_index",
                    ),
                );
                dense.push(
                    i32::try_from(i)
                        .expect("Too many bytes per variant to represent in spool_dense_map"),
                );
            }
            (compact, dense)
        }
        PersonSubset::Indices(indices) => {
            let mut unique_bytes: BTreeSet<u32> = BTreeSet::new();
            for &fam_idx in indices {
                unique_bytes.insert(fam_idx / 4);
            }
            let compact: Vec<u32> = unique_bytes.into_iter().collect();
            let mut dense = vec![-1i32; bytes_per_variant_usize];
            for (compact_idx, &orig_byte_idx) in compact.iter().enumerate() {
                if let Some(slot) = dense.get_mut(orig_byte_idx as usize) {
                    *slot = i32::try_from(compact_idx)
                        .expect("Too many kept individuals to compact into spool_dense_map");
                }
            }
            (compact, dense)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_reordering_preserves_empty_rows_and_weight_bits() {
        let mut csr = CsrBuilder::new().unwrap();
        for entries in [
            vec![(2, -0.0f32, 2.0f32)],
            vec![],
            vec![(0, 0.1f32, -0.0f32), (3, 0.2f32, 3.0f32)],
        ] {
            for (column, dosage_weight, missing_correction) in entries {
                csr.push_contribution(
                    ScoreColumnIndex(column),
                    SimpleScoreAssignment {
                        dosage_weight,
                        missing_correction,
                    },
                )
                .unwrap();
            }
            csr.finish_variant().unwrap();
        }
        let mut rows = vec![BimRowIndex(10), BimRowIndex(1), BimRowIndex(5)];
        let mut flags = vec![0, 1, 0];
        csr.sort_rows_by_bim_index(&mut rows, &mut flags).unwrap();
        assert_eq!(rows, [BimRowIndex(1), BimRowIndex(5), BimRowIndex(10)]);
        assert_eq!(flags, [1, 0, 0]);
        assert_eq!(csr.sparse_row_offsets, [0, 0, 2, 3]);
        assert_eq!(csr.sparse_score_columns, [0, 3, 2]);
        assert_eq!(
            csr.sparse_weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [0.1f32, 0.2, -0.0].map(f32::to_bits),
        );
        assert_eq!(
            csr.sparse_missing_corrections.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [-0.0f32, 3.0, 2.0].map(f32::to_bits),
        );
    }

    fn write_bim_fileset(prefix: &Path, rows: &[&str], people: usize) {
        std::fs::write(prefix.with_extension("bim"), rows.concat()).unwrap();
        let fam: String = (0..people).map(|i| format!("F I{i} 0 0 0 -9\n")).collect();
        std::fs::write(prefix.with_extension("fam"), fam).unwrap();
        let mut bed = vec![0x6c, 0x1b, 0x01];
        bed.resize(3 + rows.len() * people.div_ceil(4), 0);
        std::fs::write(prefix.with_extension("bed"), bed).unwrap();
    }

    type RowPlan = (String, u8, Vec<(u32, u32, u32)>);
    type RulePlan = (
        (String, u32),
        Vec<(String, String, String)>,
        Vec<(String, String, u32, usize)>,
    );

    /// What a plan says about each matched row and complex rule, keyed by `.bim`
    /// row text instead of row position, so plans over different layouts compare.
    fn plan_by_row_text(prep: &PreparationResult, rows: &[&str]) -> (Vec<RowPlan>, Vec<RulePlan>) {
        let offsets = prep.sparse_row_offsets();
        let mut row_plans: Vec<RowPlan> = prep
            .required_bim_indices
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let entries = (offsets[i] as usize..offsets[i + 1] as usize)
                    .map(|e| {
                        (
                            prep.sparse_score_columns()[e],
                            prep.sparse_weights()[e].to_bits(),
                            prep.sparse_missing_corrections()[e].to_bits(),
                        )
                    })
                    .collect();
                let text = rows[row.0 as usize].to_string();
                (text, prep.required_is_complex()[i], entries)
            })
            .collect();
        row_plans.sort();
        let mut rule_plans: Vec<RulePlan> = prep
            .complex_rules
            .iter()
            .map(|rule| {
                let contexts = rule
                    .possible_contexts
                    .iter()
                    .map(|(row, a1, a2)| (rows[row.0 as usize].to_string(), a1.clone(), a2.clone()))
                    .collect();
                let applications = rule
                    .score_applications
                    .iter()
                    .map(|s| {
                        let (ea, oa) = (s.effect_allele.clone(), s.other_allele.clone());
                        (ea, oa, s.weight.to_bits(), s.score_column_index.0)
                    })
                    .collect();
                (rule.locus_chr_pos.clone(), contexts, applications)
            })
            .collect();
        rule_plans.sort();
        (row_plans, rule_plans)
    }

    #[test]
    fn bim_order_does_not_change_which_variants_match() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.tsv");
        std::fs::write(
            &weights,
            "variant_id\teffect_allele\tother_allele\tS1\tS2\n\
             1:100\tG\tA\t0.5\t-1.25\n\
             1:200\tT\tC\t0.125\t\n\
             1:300\tC\tA\t2\t0.75\n\
             1:300\tG\tA\t-0.5\t1\n\
             2:50\tAT\tA\t1.5\t2.5\n\
             3:7\tG\tT\t\t3\n\
             22:1000\tA\tG\t0.25\t0.375\n\
             X:5\tT\tC\t1\t1\n",
        )
        .unwrap();
        // Unsorted rows, and the same rows stably sorted by chromosome and
        // position: a flip (rs200), an indel, a split multiallelic locus, a row
        // with no matching allele pair and one with an unparsable chromosome.
        let unsorted = [
            "22 rs1000 0 1000 G A\n",
            "1 rs300c 0 300 A C\n",
            "X rsX 0 5 C T\n",
            "2 rs50 0 50 A AT\n",
            "chrUn_gl000220 rsUn 0 9 A C\n",
            "1 rs100 0 100 A G\n",
            "3 rs7 0 7 T A\n",
            "1 rs300g 0 300 A G\n",
            "1 rs200 0 200 T C\n",
        ];
        let sorted = [
            "1 rs100 0 100 A G\n",
            "1 rs200 0 200 T C\n",
            "1 rs300c 0 300 A C\n",
            "1 rs300g 0 300 A G\n",
            "2 rs50 0 50 A AT\n",
            "3 rs7 0 7 T A\n",
            "22 rs1000 0 1000 G A\n",
            "X rsX 0 5 C T\n",
            "chrUn_gl000220 rsUn 0 9 A C\n",
        ];
        let mut plans = Vec::new();
        for (name, rows) in [("unsorted", &unsorted), ("sorted", &sorted)] {
            let prefix = dir.path().join(name);
            write_bim_fileset(&prefix, rows, 5);
            let prep = prepare_for_computation(
                std::slice::from_ref(&prefix),
                std::slice::from_ref(&weights),
                None,
                None,
            )
            .unwrap();
            assert!(
                prep.required_bim_indices.windows(2).all(|w| w[0] < w[1]),
                "{name}"
            );
            assert_eq!(prep.score_variant_counts, vec![7, 6], "{name}");
            assert_eq!(prep.total_variants_in_bim, 9, "{name}");
            let baseline: Vec<u64> = prep
                .baseline_missing_sum_by_score()
                .iter()
                .map(|value| value.to_bits())
                .collect();
            plans.push((baseline, plan_by_row_text(&prep, rows)));
        }
        assert_eq!(plans[0], plans[1]);
    }

    #[test]
    fn bim_order_across_filesets_does_not_change_which_variants_match() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.tsv");
        std::fs::write(
            &weights,
            "variant_id\teffect_allele\tother_allele\tS\n\
             2:10\tA\tG\t1\n\
             2:20\tC\tT\t2\n\
             X:30\tG\tA\t4\n",
        )
        .unwrap();
        // Each fileset is sorted, but listing chromosome X first makes the
        // concatenated rows descend at the boundary, as natural file-name order
        // does for chrMT before chrX.
        let x_rows = ["X x30 0 30 A G\n"];
        let two_rows = ["2 r10 0 10 A G\n", "2 r20 0 20 T C\n"];
        let x = dir.path().join("x");
        let two = dir.path().join("two");
        write_bim_fileset(&x, &x_rows, 3);
        write_bim_fileset(&two, &two_rows, 3);
        let mut plans = Vec::new();
        for (filesets, second_start, rows) in [
            (
                [x.clone(), two.clone()],
                1,
                [x_rows[0], two_rows[0], two_rows[1]],
            ),
            (
                [two.clone(), x.clone()],
                2,
                [two_rows[0], two_rows[1], x_rows[0]],
            ),
        ] {
            let prep =
                prepare_for_computation(&filesets, std::slice::from_ref(&weights), None, None)
                    .unwrap();
            assert_eq!(prep.score_variant_counts, vec![3]);
            assert!(prep.required_bim_indices.windows(2).all(|w| w[0] < w[1]));
            let PipelineKind::MultiFile(boundaries) = &prep.pipeline_kind else {
                panic!("two filesets must keep a multi-file layout");
            };
            let starts: Vec<(PathBuf, u64)> = boundaries
                .iter()
                .map(|b| (b.bed_path.clone(), b.starting_global_index))
                .collect();
            assert_eq!(
                starts,
                vec![
                    (filesets[0].with_extension("bed"), 0),
                    (filesets[1].with_extension("bed"), second_start),
                ]
            );
            plans.push(plan_by_row_text(&prep, &rows));
        }
        assert_eq!(plans[0], plans[1]);
    }

    #[test]
    fn unusable_saved_plans_are_compiled_again_instead_of_failing() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.tsv");
        std::fs::write(
            &weights,
            "variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.5\n",
        )
        .unwrap();
        let prefix = dir.path().join("unusable_plan_panel");
        std::fs::write(prefix.with_extension("bim"), "1 a 0 100 A G\n").unwrap();
        std::fs::write(
            prefix.with_extension("fam"),
            "F I0 0 0 0 -9\nF I1 0 0 0 -9\n",
        )
        .unwrap();
        std::fs::write(prefix.with_extension("bed"), [0x6c, 0x1b, 0x01, 0x00]).unwrap();
        let files = build_fileset_paths(std::slice::from_ref(&prefix)).unwrap();
        let Some(key) =
            cache::PlanCache::discover(&files, std::slice::from_ref(&weights), None).unwrap()
        else {
            return;
        };
        let prepare = || {
            prepare_for_computation(
                std::slice::from_ref(&prefix),
                std::slice::from_ref(&weights),
                None,
                None,
            )
        };
        let first = prepare().unwrap();
        let zeros = [0u8; 4096];
        for garbage in [&b"truncated"[..], &zeros[..]] {
            std::fs::create_dir_all(key.path().parent().unwrap()).unwrap();
            std::fs::write(key.path(), garbage).unwrap();
            assert!(key.load().is_err());
            let again = prepare().unwrap();
            assert_eq!(again.num_people_to_score, 2);
            assert_eq!(again.required_bim_indices, first.required_bim_indices);
            assert_eq!(again.score_names, first.score_names);
            assert!(
                key.load().unwrap().is_some(),
                "the recompiled plan replaces the bad one"
            );
        }
    }

    #[test]
    fn keep_files_name_people_by_iid_or_by_plink_fid_iid_rows() {
        let dir = tempfile::tempdir().unwrap();
        let iids: Vec<String> = ["I0", "I1", "I2", "I3"].map(String::from).to_vec();
        let lookup: AHashMap<String, u32> = iids
            .iter()
            .enumerate()
            .map(|(idx, iid)| (iid.clone(), idx as u32))
            .collect();
        let keep = |text: &str| {
            let path = dir.path().join("keep.txt");
            std::fs::write(&path, text).unwrap();
            resolve_person_subset(Some(&path), &iids, &lookup)
        };
        let indices = |subset: PersonSubset| match subset {
            PersonSubset::Indices(indices) => indices,
            PersonSubset::All => panic!("a keep file selects a subset"),
        };

        let (by_iid, by_iid_names) = keep("I2\nI0\n").unwrap();
        assert_eq!(indices(by_iid), vec![0, 2]);
        assert_eq!(by_iid_names, vec!["I0", "I2"]);

        // plink2's header, tab- and space-separated FID IID rows, and one person twice.
        let (plink, plink_names) = keep("#FID\tIID\nF2\tI2\nF0 I0\nI2\n").unwrap();
        assert_eq!(indices(plink), vec![0, 2]);
        assert_eq!(plink_names, vec!["I0", "I2"]);

        let error = keep("F9\tI9\n").unwrap_err().to_string();
        assert!(error.contains("I9"), "the unmatched row is named: {error}");
    }

    #[test]
    fn cached_variants_rebind_people_paths_and_keep_layout() {
        let dir = tempfile::tempdir().unwrap();
        let weights = dir.path().join("weights.tsv");
        std::fs::write(
            &weights,
            "variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.25\n",
        )
        .unwrap();
        let mut original_key = None;
        for people in [1usize, 17] {
            let prefix = dir.path().join(format!("panel{people}"));
            std::fs::write(prefix.with_extension("bim"), "1 a 0 100 A G\n").unwrap();
            std::fs::write(
                prefix.with_extension("fam"),
                (0..people)
                    .map(|i| format!("F I{i} 0 0 0 -9\n"))
                    .collect::<String>(),
            )
            .unwrap();
            let mut bed = vec![0x6c, 0x1b, 0x01];
            bed.resize(3 + people.div_ceil(4), 0);
            std::fs::write(prefix.with_extension("bed"), bed).unwrap();
            let files = build_fileset_paths(std::slice::from_ref(&prefix)).unwrap();
            let key = cache::PlanCache::discover(&files, std::slice::from_ref(&weights), None)
                .unwrap()
                .unwrap();
            if let Some(original) = &original_key {
                assert!(key.same_inputs(original));
            }
            let prep = prepare_for_computation(
                std::slice::from_ref(&prefix),
                std::slice::from_ref(&weights),
                None,
                None,
            )
            .unwrap();
            assert_eq!(prep.num_people_to_score, people);
            assert_eq!(prep.bytes_per_variant, people.div_ceil(4) as u64);
            assert_eq!(
                prep.output_idx_to_fam_idx.last().unwrap().0,
                (people - 1) as u32
            );
            assert!(
                matches!(prep.pipeline_kind, PipelineKind::SingleFile(ref p) if *p == prefix.with_extension("bed"))
            );
            assert!(key.load().unwrap().is_some());
            if people == 17 {
                let keep = dir.path().join("keep.txt");
                std::fs::write(&keep, "I16\nI3\n").unwrap();
                let prep = prepare_for_computation(
                    &[prefix],
                    std::slice::from_ref(&weights),
                    Some(&keep),
                    None,
                )
                .unwrap();
                assert_eq!(prep.total_people_in_fam, 17);
                assert_eq!(prep.num_people_to_score, 2);
                assert_eq!(prep.spool_compact_byte_index(), &[0, 4]);
                assert_eq!(
                    prep.output_idx_to_fam_idx,
                    vec![OriginalPersonIndex(3), OriginalPersonIndex(16)]
                );
            }
            original_key = Some(key);
        }
    }

    #[test]
    fn allele_storage_preserves_exact_text_and_shares_long_sequences() {
        for value in [
            "A",
            "C",
            "G",
            "T",
            "a",
            "n",
            "0",
            "I",
            "D",
            "-",
            ".",
            "ACGTACGTACGT",
            "<DEL>",
            "é",
            "",
        ] {
            let allele = Allele::new(value);
            assert_eq!(allele.as_str(), value);
            assert_eq!(allele.clone().to_string(), value);
            if let Allele::Shared(original) = allele {
                let Allele::Shared(copy) = Allele::Shared(Arc::clone(&original)).clone() else {
                    panic!("shared allele changed representation");
                };
                assert!(Arc::ptr_eq(&original, &copy));
            }
        }
        assert!(matches!(Allele::new("A"), Allele::Literal("A")));
    }

    #[test]
    fn build_spool_maps_all_people_identity_mapping() {
        let bytes_per_variant = 3;
        let (compact, dense) = build_spool_maps(&PersonSubset::All, bytes_per_variant);
        assert_eq!(compact, vec![0, 1, 2]);
        assert_eq!(dense, vec![0, 1, 2]);
    }

    #[test]
    fn wide_singleton_join_preserves_duplicate_order_and_resets_sparse_columns() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("panel");
        std::fs::write(prefix.with_extension("bed"), [0x6c, 0x1b, 0x01, 0, 0, 0]).unwrap();
        std::fs::write(prefix.with_extension("fam"), "F I 0 0 0 -9\n").unwrap();
        std::fs::write(
            prefix.with_extension("bim"),
            "1 a 0 100 A G\n1 b 0 200 C T\n1 c 0 300 AC A\n",
        )
        .unwrap();
        let first = dir.path().join("first.tsv");
        let second = dir.path().join("second.tsv");
        std::fs::write(&first, "variant_id\teffect_allele\tother_allele\tZ\tA\n1:100\tG\tA\t16777216\t1\n1:100\tG\tA\t1\t0\n1:100\tG\tA\t-16777216\t0\n1:100\tA\tG\t0\t0.5\n1:200\tA\tG\t99\t99\n1:300\tA\tAC\t2\t3\n").unwrap();
        std::fs::write(
            &second,
            "variant_id\teffect_allele\tother_allele\tM\n1:100\tA\tG\t0.25\n1:200\tT\tC\t4\n",
        )
        .unwrap();
        let (prep, clean) = prepare_for_computation_with_retry(
            &[prefix],
            &[first, second],
            None,
            None,
            BimRowOrder::Streamed,
        )
        .unwrap();
        assert!(clean);
        let columns: Vec<_> = ["Z", "A", "M"]
            .map(|name| prep.score_names.iter().position(|s| s == name).unwrap())
            .into();
        assert_eq!(prep.required_bim_indices, [0, 1, 2].map(BimRowIndex));
        assert_eq!(prep.sparse_row_offsets(), &[0, 3, 4, 6]);
        for (row, expected) in [
            vec![
                (columns[0], 0.0f32, 0.0f32),
                (columns[1], 0.5, 1.0),
                (columns[2], -0.25, 0.5),
            ],
            vec![(columns[2], 4.0, 0.0)],
            vec![(columns[0], 2.0, 0.0), (columns[1], 3.0, 0.0)],
        ]
        .into_iter()
        .enumerate()
        {
            let mut expected = expected;
            expected.sort_unstable_by_key(|x| x.0);
            let start = prep.sparse_row_offsets()[row] as usize;
            for (offset, (col, weight, correction)) in expected.into_iter().enumerate() {
                assert_eq!(prep.sparse_score_columns()[start + offset], col as u32);
                assert_eq!(
                    prep.sparse_weights()[start + offset].to_bits(),
                    weight.to_bits()
                );
                assert_eq!(
                    prep.sparse_missing_corrections()[start + offset].to_bits(),
                    correction.to_bits()
                );
            }
        }
        for column in columns {
            assert_eq!(prep.score_variant_counts[column], 2);
        }
        assert_eq!(
            prep.baseline_missing_sum_by_score()
                [prep.score_names.iter().position(|s| s == "A").unwrap()],
            1.0
        );
        assert_eq!(
            prep.baseline_missing_sum_by_score()
                [prep.score_names.iter().position(|s| s == "M").unwrap()],
            0.5
        );
    }

    #[test]
    fn singleton_join_preserves_swaps_duplicates_indels_and_complex_loci() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("panel");
        std::fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0, 0, 0, 0, 0, 0],
        )
        .unwrap();
        std::fs::write(prefix.with_extension("fam"), "F I 0 0 0 -9\n").unwrap();
        std::fs::write(prefix.with_extension("bim"), "1 a 0 100 A G\n1 b 0 150 C T\n1 c 0 200 AC A\n1 d 0 250 A C\n1 e 0 250 G C\n1 f 0 300 <DEL> A\n").unwrap();
        let weights = dir.path().join("weights.tsv");
        std::fs::write(&weights, "variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.25\n1:150\tC\tT\t0.5\n1:200\tA\tAC\t-0.75\n1:250\tC\tA\t0.125\n1:300\t<DEL>\tA\t-0.125\n1:300\t<DEL>\tA\t0.375\n").unwrap();
        let prep = prepare_for_computation(&[prefix], &[weights], None, None).unwrap();
        assert_eq!(prep.required_bim_indices, [0, 1, 2, 3, 5].map(BimRowIndex));
        assert_eq!(prep.sparse_row_offsets(), &[0, 1, 2, 3, 3, 4]);
        assert_eq!(prep.sparse_weights(), &[0.25, -0.5, -0.75, -0.25]);
        assert_eq!(prep.sparse_missing_corrections(), &[0.0, 1.0, 0.0, 0.5]);
        assert_eq!(prep.baseline_missing_sum_by_score(), &[1.5]);
        assert_eq!(prep.score_variant_counts, [5]);
        assert_eq!(prep.required_is_complex(), &[0, 0, 0, 1, 0]);
        assert_eq!(prep.complex_rules.len(), 1);
        assert_eq!(
            prep.complex_rules[0].possible_contexts,
            [(BimRowIndex(3), "A".into(), "C".into())]
        );
    }

    #[test]
    fn build_spool_maps_subset_compacts_sorted_unique_bytes() {
        let indices = vec![0, 1, 8, 9];
        let bytes_per_variant = 4; // enough room for indices up to 3
        let (compact, dense) =
            build_spool_maps(&PersonSubset::Indices(indices.clone()), bytes_per_variant);

        // The compact list must stay sorted and unique regardless of input order.
        assert_eq!(compact, vec![0, 2]);
        assert!(compact.windows(2).all(|w| w[0] < w[1]));

        // Dense map should point to the compact slots for kept bytes and -1 elsewhere.
        assert_eq!(dense.len(), bytes_per_variant);
        assert_eq!(dense[0], 0);
        assert_eq!(dense[1], -1);
        assert_eq!(dense[2], 1);
        assert_eq!(dense[3], -1);

        // Every kept fam index should resolve to a compact entry.
        for fam_idx in indices {
            let orig_byte = (fam_idx / 4) as usize;
            assert!(dense[orig_byte] >= 0);
        }
    }

    #[test]
    fn kway_merge_continues_after_filtered_lines() {
        use std::io::Write;

        let dir = tempfile::tempdir().expect("tempdir");
        let score_path = dir.path().join("score.tsv");
        {
            let mut file = std::fs::File::create(&score_path).expect("create score");
            writeln!(file, "variant_id\teffect_allele\tother_allele\tScoreA")
                .expect("write header");
            writeln!(file, "1:100\tA\tG\t0.5").expect("write first line");
            writeln!(file, "1:150\tC\tT\t0.7").expect("write second line");
            writeln!(file, "1:180\tC\tT\t0.2").expect("write third line");
        }

        let mut score_name_to_col_index = AHashMap::new();
        score_name_to_col_index.insert("ScoreA".to_string(), ScoreColumnIndex(0));

        let region = GenomicRegion {
            chromosome: 1,
            start: 140,
            end: 200,
        };
        let file_paths = vec![score_path];
        let mut iter = KWayMergeIterator::new(
            &file_paths,
            &score_name_to_col_index,
            Some(vec![Some(region)]),
        )
        .expect("iterator");

        let mut keys = Vec::new();
        while let Some(result) = iter.next() {
            let record = result.expect("record ok");
            keys.push(record.key);
        }

        assert_eq!(keys, vec![(1, 150), (1, 180)]);
    }

    #[test]
    fn duplicate_aggregation_same_orientation_sums_weights() {
        let mut agg = SimpleScoreAssignment {
            dosage_weight: 0.0,
            missing_correction: 0.0,
        };
        apply_simple_score_assignment(&mut agg, 0.25, false);
        apply_simple_score_assignment(&mut agg, -0.10, false);

        assert!((agg.dosage_weight - 0.15).abs() < 1e-6);
        assert!(agg.missing_correction.abs() < 1e-6);
    }

    #[test]
    fn duplicate_aggregation_swapped_orientation_tracks_missing_correction() {
        let mut agg = SimpleScoreAssignment {
            dosage_weight: 0.0,
            missing_correction: 0.0,
        };
        apply_simple_score_assignment(&mut agg, 0.40, true);

        assert!((agg.dosage_weight - (-0.40)).abs() < 1e-6);
        assert!((agg.missing_correction - 0.80).abs() < 1e-6);
    }

    #[test]
    fn duplicate_aggregation_matches_row_by_row_for_all_dosages() {
        let rows = [(0.35f32, false), (0.10f32, true), (-0.05f32, false)];
        let mut agg = SimpleScoreAssignment {
            dosage_weight: 0.0,
            missing_correction: 0.0,
        };
        for (w, is_flipped) in rows {
            apply_simple_score_assignment(&mut agg, w, is_flipped);
        }

        for dosage in [0.0f32, 1.0, 2.0] {
            let row_by_row = rows
                .iter()
                .map(|(w, is_flipped)| {
                    if *is_flipped {
                        w * (2.0 - dosage)
                    } else {
                        w * dosage
                    }
                })
                .sum::<f32>();
            let aggregated = agg.missing_correction + (agg.dosage_weight * dosage);
            assert!(
                (row_by_row - aggregated).abs() < 1e-6,
                "dosage={dosage} row_by_row={row_by_row} aggregated={aggregated}"
            );
        }

        // Missing genotype contributes nothing after baseline correction.
        let missing_after_baseline = agg.missing_correction - agg.missing_correction;
        assert!(missing_after_baseline.abs() < 1e-6);
    }
}

/// The file extensions that make up a PLINK 1.9 or PLINK 2 fileset.
pub const FILESET_EXTENSIONS: [&str; 6] = ["bed", "bim", "fam", "pgen", "pvar", "psam"];

/// Strips a trailing fileset extension, if present, leaving the shared prefix.
///
/// Only the extensions above are stripped. Fileset prefixes routinely contain
/// dots of their own — All of Us ships `acaf_threshold.chr22.pgen`, whose
/// prefix is `acaf_threshold.chr22` — so truncating at the last dot would turn
/// a sibling lookup into `acaf_threshold.pvar` and silently miss the file.
pub fn strip_fileset_extension(path_str: &str) -> &str {
    if let Some((base, ext)) = path_str.rsplit_once('.')
        && FILESET_EXTENSIONS.contains(&ext)
        && !base.ends_with('/')
    {
        return base;
    }
    path_str
}

fn apply_extension(path: &Path, extension: &str) -> Result<PathBuf, PrepError> {
    let Some(path_str) = path.to_str() else {
        return Err(PrepError::Parse("Invalid UTF-8 in path".to_string()));
    };
    Ok(PathBuf::from(format!(
        "{}.{extension}",
        strip_fileset_extension(path_str)
    )))
}

fn map_pipeline_error(err: PipelineError, path: PathBuf) -> PrepError {
    PrepError::PipelineIo {
        path,
        message: err.to_string(),
    }
}

/// Extracts the malformed chromosome name from a `PrepError::Parse` message.
fn extract_chr_from_parse_error(msg: &str) -> Option<&str> {
    if let Some(rest) = msg.strip_prefix("Invalid chromosome format '")
        && let Some(end_pos) = rest.find('\'')
    {
        return Some(&rest[..end_pos]);
    }
    None
}

fn parse_key(chr_str: &str, pos_str: &str) -> Result<(u8, u32), PrepError> {
    let chr_num = parse_chromosome_label(chr_str).map_err(PrepError::Parse)?;
    let pos_trimmed = pos_str.trim();
    let pos_num: u32 = pos_trimmed
        .parse()
        .map_err(|e| PrepError::Parse(format!("Invalid position '{pos_str}': {e}")))?;

    Ok((chr_num, pos_num))
}

fn conduct_post_mortem(
    fileset_paths: &[FilesetPaths],
    score_files: &[PathBuf],
) -> Result<PostMortemAction, PrepError> {
    let mut bim_chromosomes: AHashSet<u8> = AHashSet::new();
    let mut score_chromosomes: AHashSet<u8> = AHashSet::new();

    // Row order is irrelevant here: the join matches `.bim` rows in any order.
    for next_item in BimIterator::new(fileset_paths)? {
        match next_item {
            Ok(record) => {
                bim_chromosomes.insert(record.key.0);
            }
            Err(PrepError::Parse(_)) => continue,
            Err(err) => return Err(err),
        }
    }

    for path in score_files {
        let file = File::open(path).map_err(|e| PrepError::Io(e, path.clone()))?;
        let reader = BufReader::new(file);
        let mut previous_key: Option<VariantKey> = None;

        // Score files are already normalized by `reformat_pgs_file`, which guarantees
        // the tab-separated layout: variant_id, effect_allele, other_allele, weight....
        // The quick parser below intentionally relies on that invariant to keep this
        // post-mortem check minimal and fast to implement.
        for (line_index, line_result) in reader.lines().enumerate() {
            let line_number = line_index as u64 + 1;
            let line = line_result.map_err(|e| PrepError::Io(e, path.clone()))?;
            let trimmed = line.trim();

            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut parts = trimmed.split('\t');
            let variant_id = match parts.next() {
                Some(id) if !id.is_empty() => id,
                _ => continue,
            };

            if variant_id.eq_ignore_ascii_case("variant_id") {
                continue;
            }

            let effect_allele = parts.next();
            let other_allele = parts.next();

            if effect_allele.is_none()
                || other_allele.is_none()
                || effect_allele.is_some_and(|a| a.is_empty())
                || other_allele.is_some_and(|a| a.is_empty())
            {
                continue;
            }

            let mut key_parts = variant_id.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");
            let key = parse_key(chr_str, pos_str)?;

            score_chromosomes.insert(key.0);

            if let Some(prev_key) = previous_key
                && key < prev_key
            {
                return Ok(PostMortemAction::Fatal(PrepError::UnsortedInput {
                    source: "score",
                    path: path.clone(),
                    line_number,
                    previous_key: prev_key,
                    current_key: key,
                }));
            }

            previous_key = Some(key);
        }
    }

    let has_bim_chromosomes = !bim_chromosomes.is_empty();
    let has_score_chromosomes = !score_chromosomes.is_empty();

    if has_bim_chromosomes
        && has_score_chromosomes
        && bim_chromosomes
            .iter()
            .any(|chr| score_chromosomes.contains(chr))
    {
        return Ok(PostMortemAction::Fatal(PrepError::GenomeBuildMismatch));
    }

    if has_bim_chromosomes && has_score_chromosomes {
        return Ok(PostMortemAction::Fatal(PrepError::DisjointChromosomes));
    }

    Ok(PostMortemAction::None)
}

impl<'a> BimIterator<'a> {
    fn new(filesets: &'a [FilesetPaths]) -> Result<Self, PrepError> {
        let mut iter = Self {
            filesets: filesets.iter(),
            current_reader: None,
            global_offset: 0,
            local_line_num: 0,
            current_path: PathBuf::new(),
            boundaries: Vec::with_capacity(filesets.len()),
            total_variants: 0,
        };
        iter.next_file()?;
        Ok(iter)
    }

    fn next_file(&mut self) -> Result<bool, PrepError> {
        self.global_offset += self.local_line_num;
        self.local_line_num = 0;

        if let Some(fileset) = self.filesets.next() {
            self.boundaries.push(FilesetBoundary {
                bed_path: fileset.bed.clone(),
                bim_path: fileset.bim.clone(),
                fam_path: fileset.fam.clone(),
                starting_global_index: self.global_offset,
            });

            self.current_path = fileset.bim.clone();
            let reader = open_plink_text_source(&fileset.bim)
                .map_err(|e| map_pipeline_error(e, fileset.bim.clone()))?;
            self.current_reader = Some(reader);
            Ok(true)
        } else {
            self.current_reader = None;
            self.total_variants = self.global_offset;
            Ok(false)
        }
    }

    fn total_variants(&self) -> u64 {
        self.total_variants
    }
}

impl<'a> Iterator for BimIterator<'a> {
    type Item = Result<KeyedBimRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let reader = self.current_reader.as_mut()?;

            match reader.next_line() {
                Ok(Some(line_bytes)) => {
                    self.local_line_num += 1;
                    self.total_variants = self.global_offset + self.local_line_num;
                    let row = BimRowIndex(self.global_offset + self.local_line_num - 1);
                    if let Some(item) = parse::parse_bim_row(line_bytes, row, &self.current_path) {
                        return Some(item);
                    }
                }
                Ok(None) => {
                    if let Ok(false) = self.next_file() {
                        return None;
                    }
                }
                Err(err) => {
                    return Some(Err(map_pipeline_error(err, self.current_path.clone())));
                }
            }
        }
    }
}

/// How Stage 3 reads `.bim` rows for the merge-join, which needs them in key order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BimRowOrder {
    /// Stream rows straight from the files, as most `.bim` files are sorted.
    Streamed,
    /// Read every row and sort by key, ties in file order.
    Sorted,
}

/// `.bim` rows in key order, as the merge-join consumes them.
enum BimRows<'i, 'a> {
    /// Rows straight from the files, ending at the first row whose key sorts
    /// before its predecessor's: joining past it would silently miss matches.
    Streamed {
        rows: &'i mut BimIterator<'a>,
        previous_key: Option<VariantKey>,
        descended_in: Option<PathBuf>,
    },
    Sorted(std::vec::IntoIter<KeyedBimRecord>),
    /// Rows parsed from whole local files, in file order, with each unparsable
    /// row as its error at its place: `errors` holds (rows yielded before, error).
    Parsed {
        records: std::vec::IntoIter<KeyedBimRecord>,
        errors: std::iter::Peekable<std::vec::IntoIter<(usize, PrepError)>>,
        yielded: usize,
    },
}

impl<'i, 'a> BimRows<'i, 'a> {
    fn streamed(rows: &'i mut BimIterator<'a>) -> Self {
        Self::Streamed {
            rows,
            previous_key: None,
            descended_in: None,
        }
    }

    /// Reads every row. Unparsable rows are reported and skipped exactly as the
    /// streaming join reports and skips them.
    fn sorted(
        rows: &mut BimIterator<'a>,
        seen_invalid_bim_chrs: &mut AHashSet<String>,
    ) -> Result<Self, PrepError> {
        let mut records = Vec::new();
        for result in rows {
            match result {
                Ok(record) => records.push(record),
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
        records.par_sort_unstable_by_key(|record| (record.key, record.bim_row_index));
        Ok(Self::Sorted(records.into_iter()))
    }

    /// Rows parsed whole. Rows in key order are yielded as they are. Otherwise
    /// the unparsable rows are reported and every row is sorted by key, as
    /// `sorted` does, without reading the files again.
    fn parsed(
        mut records: Vec<KeyedBimRecord>,
        errors: Vec<(usize, PrepError)>,
        boundaries: &[FilesetBoundary],
        seen_invalid_bim_chrs: &mut AHashSet<String>,
    ) -> Self {
        let Some(descent) = records
            .windows(2)
            .position(|pair| pair[1].key < pair[0].key)
        else {
            return Self::Parsed {
                records: records.into_iter(),
                errors: errors.into_iter().peekable(),
                yielded: 0,
            };
        };
        let row = records[descent + 1].bim_row_index.0;
        let fileset = boundaries.partition_point(|b| b.starting_global_index <= row) - 1;
        eprintln!(
            "> Variants in {} are not sorted by chromosome and position. Matching them in sorted order...",
            boundaries[fileset].bim_path.display()
        );
        for (_, error) in errors {
            if let PrepError::Parse(msg) = error
                && let Some(chr_name) = extract_chr_from_parse_error(&msg)
                && seen_invalid_bim_chrs.insert(chr_name.to_string())
            {
                eprintln!(
                    "Warning: Skipping variant(s) in BIM file due to unparsable chromosome name: '{chr_name}'."
                );
            }
        }
        records.par_sort_unstable_by_key(|record| (record.key, record.bim_row_index));
        Self::Sorted(records.into_iter())
    }

    /// The `.bim` file in which streamed rows stopped ascending, if they did.
    fn descended_in(&self) -> Option<&Path> {
        match self {
            Self::Streamed { descended_in, .. } => descended_in.as_deref(),
            Self::Sorted(_) | Self::Parsed { .. } => None,
        }
    }
}

impl Iterator for BimRows<'_, '_> {
    type Item = Result<KeyedBimRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Streamed {
                rows,
                previous_key,
                descended_in,
            } => {
                if descended_in.is_some() {
                    return None;
                }
                let item = rows.next()?;
                if let Ok(record) = &item {
                    if previous_key.is_some_and(|previous| record.key < previous) {
                        *descended_in = Some(rows.current_path.clone());
                        return None;
                    }
                    *previous_key = Some(record.key);
                }
                Some(item)
            }
            Self::Sorted(records) => records.next().map(Ok),
            Self::Parsed {
                records,
                errors,
                yielded,
            } => {
                if let Some((_, error)) = errors.next_if(|(before, _)| *before == *yielded) {
                    return Some(Err(error));
                }
                let record = records.next()?;
                *yielded += 1;
                Some(Ok(record))
            }
        }
    }
}

impl KWayMergeIterator {
    fn new(
        file_paths: &[PathBuf],
        score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
        region_filters: Option<Vec<Option<GenomicRegion>>>,
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
                if !header_line.trim().is_empty() && !header_line.starts_with('#') {
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
                file_line_number: 0,
                malformed_lines_count: 0,
            });
        }

        let region_filter_hits = region_filters
            .as_ref()
            .map(|_| vec![false; score_name_to_col_index.len()]);

        let mut iter = Self {
            streams,
            heap: BinaryHeap::new(),
            file_column_maps,
            next_error: None,
            region_filters,
            region_filter_hits,
        };

        for i in 0..iter.streams.len() {
            iter.replenish_from_stream(i)?
        }

        Ok(iter)
    }

    fn replenish_from_stream(&mut self, file_idx: usize) -> Result<(), PrepError> {
        let column_map = &self.file_column_maps[file_idx];

        if self.region_filters.is_none() {
            loop {
                let outcome = {
                    let stream = &mut self.streams[file_idx];
                    if !stream.line_buffer.is_empty() {
                        Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap)?;
                        return Ok(());
                    }

                    Self::read_line_into_buffer(stream, column_map, None, None)?
                };

                match outcome {
                    LineReadOutcome::Pushed => {
                        let stream = &mut self.streams[file_idx];
                        Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap)?;
                        return Ok(());
                    }
                    LineReadOutcome::Skipped => continue,
                    LineReadOutcome::Eof => return Ok(()),
                }
            }
        }

        let region_filters = self.region_filters.as_deref();
        let mut region_hits = self.region_filter_hits.take();

        loop {
            let outcome = {
                let stream = &mut self.streams[file_idx];
                if !stream.line_buffer.is_empty() {
                    Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap)?;
                    self.region_filter_hits = region_hits;
                    return Ok(());
                }

                let region_hits_slice = region_hits.as_deref_mut();

                Self::read_line_into_buffer(stream, column_map, region_filters, region_hits_slice)?
            };

            match outcome {
                LineReadOutcome::Pushed => {
                    let stream = &mut self.streams[file_idx];
                    Self::push_next_from_buffer_to_heap(stream, file_idx, &mut self.heap)?;
                    self.region_filter_hits = region_hits;
                    return Ok(());
                }
                LineReadOutcome::Skipped => continue,
                LineReadOutcome::Eof => {
                    self.region_filter_hits = region_hits;
                    return Ok(());
                }
            }
        }
    }

    fn read_line_into_buffer(
        stream: &mut FileStream,
        column_map: &[ScoreColumnIndex],
        region_filters: Option<&[Option<GenomicRegion>]>,
        mut region_hits: Option<&mut [bool]>,
    ) -> Result<LineReadOutcome, PrepError> {
        stream.line_buffer.clear();
        stream.current_line_info = None;

        loop {
            stream.line_string_buffer.clear();
            let bytes_read = stream
                .reader
                .read_line(&mut stream.line_string_buffer)
                .map_err(|e| PrepError::Io(e, PathBuf::new()))?;

            if bytes_read == 0 {
                return Ok(LineReadOutcome::Eof);
            }
            stream.file_line_number += 1;

            if stream.line_string_buffer.trim().is_empty()
                || stream.line_string_buffer.starts_with('#')
            {
                continue;
            }

            let mut parts = stream.line_string_buffer.split('\t');
            let (variant_id, effect_allele, other_allele) =
                match (parts.next(), parts.next(), parts.next()) {
                    (Some(v), Some(e), Some(o))
                        if !v.is_empty() && !e.is_empty() && !o.is_empty() =>
                    {
                        (v, e, o)
                    } // Ensure other_allele is also not empty
                    _ => {
                        stream.malformed_lines_count += 1;
                        continue; // Line doesn't have the required three non-empty columns
                    }
                };
            if other_allele == "N" {
                return Err(PrepError::Parse(format!(
                    "Score file line {} has unknown other_allele 'N'. Scores must provide an explicit allele pair.",
                    stream.file_line_number
                )));
            }

            let mut key_parts = variant_id.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");
            let key = parse_key(chr_str, pos_str)?;
            stream.current_line_info =
                Some((key, Allele::new(effect_allele), Allele::new(other_allele)));

            for (i, weight_str) in parts.enumerate() {
                let weight_str = weight_str.trim();
                if weight_str.is_empty() {
                    continue;
                }
                let Some(&score_column_index) = column_map.get(i) else {
                    continue;
                };
                let weight = weight_str.parse::<f32>().map_err(|err| {
                    PrepError::Parse(format!(
                        "Invalid weight '{}' in score file line {}, column {}: {}",
                        weight_str,
                        stream.file_line_number,
                        i + 4,
                        err
                    ))
                })?;
                if let Some(filters) = region_filters {
                    if let Some(Some(region)) = filters.get(score_column_index.0)
                        && !region.contains(key)
                    {
                        continue;
                    }
                    if let Some(hit_flags) = region_hits.as_deref_mut() {
                        hit_flags[score_column_index.0] = true;
                    }
                } else if let Some(hit_flags) = region_hits.as_deref_mut() {
                    hit_flags[score_column_index.0] = true;
                }
                stream.line_buffer.push_back((weight, score_column_index));
            }
            if stream.line_buffer.is_empty() {
                stream.current_line_info = None;
                return Ok(LineReadOutcome::Skipped);
            } else {
                return Ok(LineReadOutcome::Pushed);
            }
        }
    }

    fn push_next_from_buffer_to_heap(
        stream: &mut FileStream,
        file_idx: usize,
        heap: &mut BinaryHeap<HeapItem>,
    ) -> Result<(), PrepError> {
        if let Some((weight, score_column_index)) = stream.line_buffer.pop_front() {
            let (key, effect_allele, other_allele) = stream.current_line_info.as_ref().ok_or_else(
                || {
                    PrepError::Invariant(
                        "Score stream invariant violated: non-empty line buffer without current line info."
                            .to_string(),
                    )
                },
            )?;
            let record = KeyedScoreRecord {
                key: *key,
                effect_allele: effect_allele.clone(),
                other_allele: other_allele.clone(),
                score_column_index,
                weight,
            };
            heap.push(HeapItem { record, file_idx });
        }
        Ok(())
    }

    fn take_region_filter_hits(&mut self) -> Option<Vec<bool>> {
        self.region_filter_hits.take()
    }
}

impl Iterator for KWayMergeIterator {
    type Item = Result<KeyedScoreRecord, PrepError>;

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

/// Score records in merged order for the merge-join: parsed whole on the pool,
/// or streamed by the k-way merge.
enum ScoreRows {
    Streamed(KWayMergeIterator),
    Parsed(scores::ParsedScores),
}

impl ScoreRows {
    /// Lines skipped for missing columns among those read so far.
    fn malformed_lines(&self) -> usize {
        match self {
            Self::Streamed(merge) => merge.streams.iter().map(|s| s.malformed_lines_count).sum(),
            Self::Parsed(parsed) => parsed.malformed_lines(),
        }
    }

    fn take_region_filter_hits(&mut self) -> Option<Vec<bool>> {
        match self {
            Self::Streamed(merge) => merge.take_region_filter_hits(),
            // Parsed only without region filters, which record no hits.
            Self::Parsed(_) => None,
        }
    }
}

impl Iterator for ScoreRows {
    type Item = Result<KeyedScoreRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Streamed(merge) => merge.next(),
            Self::Parsed(parsed) => parsed.next(),
        }
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
        let lines_to_keep: AHashSet<String> = reader
            .lines()
            .filter_map(Result::ok)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && !is_keep_header(s))
            .collect();

        let mut found_people = Vec::with_capacity(lines_to_keep.len());
        let mut missing_ids = Vec::new();

        for line in lines_to_keep {
            match resolve_keep_line(&line, iid_to_original_idx) {
                Some((original_idx, iid)) => found_people.push((original_idx, iid.to_string())),
                None => missing_ids.push(line),
            }
        }

        if !missing_ids.is_empty() {
            return Err(PrepError::InconsistentKeepId(format_missing_ids_error(
                missing_ids,
            )));
        }

        found_people.sort_unstable_by_key(|(idx, _)| *idx);
        // Someone listed both by IID and as `FID IID` is still one person.
        found_people.dedup_by_key(|(idx, _)| *idx);
        let final_person_iids = found_people.iter().map(|(_, iid)| iid.clone()).collect();
        let subset_indices = found_people.into_iter().map(|(idx, _)| idx).collect();
        Ok((PersonSubset::Indices(subset_indices), final_person_iids))
    } else {
        Ok((PersonSubset::All, all_person_iids.to_vec()))
    }
}

/// Resolves one keep-file line to a person. A line that is itself an IID is taken
/// as always. Any other line of two or more fields is read as PLINK's
/// `FID IID ...`, the layout `plink2 --keep` files use, and matched by its IID.
fn resolve_keep_line<'a>(
    line: &'a str,
    iid_to_original_idx: &AHashMap<String, u32>,
) -> Option<(u32, &'a str)> {
    if let Some(&idx) = iid_to_original_idx.get(line) {
        return Some((idx, line));
    }
    let mut fields = line.split_whitespace();
    fields.next()?;
    let iid = fields.next()?;
    iid_to_original_idx.get(iid).map(|&idx| (idx, iid))
}

/// The `#FID IID` or `#IID` header line plink2 writes above sample lists.
fn is_keep_header(line: &str) -> bool {
    line.starts_with("#FID") || line.starts_with("#IID")
}

fn parse_fam_and_build_lookup(
    fileset_paths: &[FilesetPaths],
) -> Result<(Vec<String>, AHashMap<String, u32>), PrepError> {
    let mut iid_to_idx = AHashMap::new();
    let mut canonical_iids: Option<Vec<String>> = None;
    let mut canonical_path: Option<PathBuf> = None;
    let mut seen_paths: AHashSet<PathBuf> = AHashSet::new();

    for fileset in fileset_paths {
        if !seen_paths.insert(fileset.fam.clone()) {
            continue;
        }

        let iids = read_fam_file(&fileset.fam)?;
        if let Some(existing) = &canonical_iids {
            if *existing != iids {
                let canonical = canonical_path
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                return Err(PrepError::PipelineIo {
                    path: fileset.fam.clone(),
                    message: format!(
                        "FAM file '{}' does not match canonical FAM '{}'.",
                        fileset.fam.display(),
                        canonical
                    ),
                });
            }
        } else {
            let mut seen_iids = AHashSet::new();
            for (idx, iid) in iids.iter().enumerate() {
                if !seen_iids.insert(iid.clone()) {
                    return Err(PrepError::Parse(format!(
                        "Duplicate IID '{}' in FAM file '{}'. gnomon requires unique output IIDs.",
                        iid,
                        fileset.fam.display()
                    )));
                }
                let idx_u32 = u32::try_from(idx).map_err(|_| {
                    PrepError::Invariant(format!(
                        "FAM index {idx} exceeds u32::MAX while building lookup."
                    ))
                })?;
                iid_to_idx.insert(iid.clone(), idx_u32);
            }
            canonical_path = Some(fileset.fam.clone());
            canonical_iids = Some(iids);
        }
    }

    let person_iids = canonical_iids.ok_or_else(|| {
        PrepError::Parse("No individuals found in provided .fam files.".to_string())
    })?;
    Ok((person_iids, iid_to_idx))
}

fn read_fam_file(path: &Path) -> Result<Vec<String>, PrepError> {
    let mut source =
        open_plink_text_source(path).map_err(|e| map_pipeline_error(e, path.to_path_buf()))?;
    let mut iids = Vec::new();
    let mut line_number = 0usize;

    while let Some(line) = source
        .next_line()
        .map_err(|e| map_pipeline_error(e, path.to_path_buf()))?
    {
        line_number += 1;
        if line.is_empty() {
            continue;
        }
        let line_str = std::str::from_utf8(line).map_err(|e| {
            PrepError::Parse(format!(
                "Invalid UTF-8 in .fam file '{}' on line {}: {e}",
                path.display(),
                line_number
            ))
        })?;
        let iid = line_str
            .split_whitespace()
            .nth(1)
            .ok_or_else(|| {
                PrepError::Parse(format!(
                    "Missing IID in .fam file '{}' on line {}",
                    path.display(),
                    line_number
                ))
            })?
            .to_string();
        iids.push(iid);
    }

    Ok(iids)
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
    let per_file_score_names: Vec<(PathBuf, Vec<String>)> = score_files
        .par_iter()
        .map(|path| -> Result<(PathBuf, Vec<String>), PrepError> {
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

            Ok((path.clone(), score_names))
        })
        .collect::<Result<Vec<_>, PrepError>>()?;

    let mut owner_by_name = HashMap::<String, PathBuf>::new();
    let mut all_score_names = Vec::new();
    for (path, names) in per_file_score_names {
        for name in names {
            if name.is_empty() {
                return Err(PrepError::Header(format!(
                    "Empty score name in \"{}\".",
                    path.display()
                )));
            }
            if let Some(existing_path) = owner_by_name.insert(name.clone(), path.clone()) {
                return Err(PrepError::Header(format!(
                    "Duplicate Score ID '{}' detected.\n  File 1: '{}'\n  File 2: '{}'\nPlease ensure each score column has a unique identifier.",
                    name,
                    existing_path.display(),
                    path.display()
                )));
            }
            all_score_names.push(name);
        }
    }
    all_score_names.sort();
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
            PrepError::PipelineIo { path, message } => {
                write!(f, "I/O Error for file {}: {}", path.display(), message)
            }
            PrepError::UnsortedInput {
                source,
                path,
                line_number,
                previous_key,
                current_key,
            } => {
                writeln!(f, "Detected unsorted {source} data in {}.", path.display())?;
                writeln!(
                    f,
                    "Encountered key {}:{} at line {}, which comes after {}:{}.",
                    current_key.0, current_key.1, line_number, previous_key.0, previous_key.1
                )?;
                writeln!(
                    f,
                    "Please sort your input by chromosome and position before running gnomon."
                )
            }
            PrepError::GenomeBuildMismatch => {
                writeln!(
                    f,
                    "No overlapping variants found even though both inputs are sorted and share chromosomes."
                )?;
                writeln!(
                    f,
                    "This suggests a genome build mismatch (for example, GRCh37 vs GRCh38) or a formatting issue."
                )
            }
            PrepError::DisjointChromosomes => {
                writeln!(
                    f,
                    "The genotype BIM files and score files do not share any chromosomes."
                )?;
                writeln!(
                    f,
                    "Verify that you are using compatible datasets (e.g., human vs. mouse or differing chromosome naming schemes)."
                )
            }
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

                if !diag.active_region_filters.is_empty() {
                    writeln!(f, "\nRegion filters requested:")?;
                    for (score, region) in &diag.active_region_filters {
                        writeln!(f, "  - {score} -> {region}")?;
                    }

                    if !diag.region_filters_without_hits.is_empty() {
                        writeln!(
                            f,
                            "\nNo score records were observed within the requested region for:"
                        )?;
                        for score in &diag.region_filters_without_hits {
                            writeln!(f, "  - {score}")?;
                        }
                    }

                    if diag.total_score_records_processed == 0 {
                        writeln!(
                            f,
                            "\nAll score records were filtered out by the requested region restriction(s)."
                        )?;
                        writeln!(
                            f,
                            "Please verify that the specified coordinates exist in the score file(s)."
                        )?;
                    }
                } else if diag.total_score_records_processed == 0 {
                    writeln!(f, "\nNo score records were processed.")?;
                    writeln!(
                        f,
                        "This can happen if the score files are empty or contain only unparsable entries."
                    )?;
                }

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
            PrepError::Invariant(s) => write!(f, "Internal invariant violation: {s}"),
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
