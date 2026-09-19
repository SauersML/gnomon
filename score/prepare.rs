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
use crate::score::cells::{ExactPlan, PlanError};
use crate::score::io::{TextSource, open_plink_text_source, open_text_source};
use crate::score::site::{RowMatch, Site, SiteAllele, names_no_single_other_allele, row_side};
use crate::score::types::{
    BimRowIndex, FilesetBoundary, GenomicRegion, GroupedComplexRule, PersonSubset, PipelineKind,
    PreparationResult, ScoreColumnIndex, ScoreInfo, parse_chromosome_label,
};
use crate::score::types::{OriginalPersonIndex, OutputPersonIndex};
use ahash::{AHashMap, AHashSet};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::{Path, PathBuf};
use std::str::Utf8Error;
use std::sync::Arc;
use std::time::Instant;

#[path = "prepare_cache.rs"]
mod cache;

#[path = "prepare_parse.rs"]
mod parse;

#[path = "prepare_scores.rs"]
mod scores;

#[path = "prepare_blocks.rs"]
pub mod blocks;

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
    /// Whether allele 2 is the REF as the genotypes write it: a `.pvar` does, and the virtual
    /// `.bim` rows of a `.pgen` carry it there; a `.bim` writes no REF.
    reference_declared: bool,
}

/// A parsed record from a score file.
#[derive(Debug, Clone)]
struct KeyedScoreRecord {
    key: VariantKey,
    effect_allele: Allele,
    other_allele: Allele,
    score_column_index: ScoreColumnIndex,
    weight: f64,
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

// Manual implementation to handle f64 comparison correctly.
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
    line_buffer: std::collections::VecDeque<(f64, ScoreColumnIndex)>,
    /// The key and alleles for the current buffered line.
    current_line_info: Option<(VariantKey, Allele, Allele)>,
    // Temporary buffer reused for reading raw line data from the file.
    line_string_buffer: String,
    /// 1-based line number in the currently read file.
    file_line_number: u64,
    /// Counter for malformed lines in this specific file stream.
    malformed_lines_count: usize,
    /// The file, named in messages about its rows.
    path: PathBuf,
    /// Lines through the header, so messages can give physical line numbers.
    header_lines: u64,
    /// Rows and weight fields skipped because they cannot be used.
    rejected: RejectedScoreRows,
}

#[derive(Debug, Copy, Clone)]
struct SimpleScoreAssignment {
    // Weight applied to effect-allele dosage in canonical (BIM allele2) space.
    dosage_weight: f64,
    // Correction to subtract for missing calls at this variant/score cell.
    missing_correction: f64,
}

/// Lock-step CSR builder that guarantees aligned sparse vectors and valid row offsets.
struct CsrBuilder {
    sparse_weights: Vec<f64>,
    sparse_missing_corrections: Vec<f64>,
    sparse_score_columns: Vec<u32>,
    sparse_row_offsets: Vec<u64>,
}

impl CsrBuilder {
    fn new() -> Result<Self, PrepError> {
        let mut sparse_row_offsets = Vec::<u64>::new();
        sparse_row_offsets
            .try_reserve_exact(1)
            .map_err(|e| PrepError::Invariant(format!("Cannot allocate CSR row offsets: {e}")))?;
        sparse_row_offsets.push(0);
        Ok(Self {
            sparse_weights: Vec::new(),
            sparse_missing_corrections: Vec::new(),
            sparse_score_columns: Vec::new(),
            sparse_row_offsets,
        })
    }

    #[inline(always)]
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
        if self.sparse_score_columns.len() == self.sparse_score_columns.capacity()
            || self.sparse_weights.len() == self.sparse_weights.capacity()
            || self.sparse_missing_corrections.len() == self.sparse_missing_corrections.capacity()
        {
            Self::reserve_entry(&mut self.sparse_score_columns, "score columns")?;
            Self::reserve_entry(&mut self.sparse_weights, "weights")?;
            Self::reserve_entry(&mut self.sparse_missing_corrections, "missing corrections")?;
        }
        // SAFETY: the capacity checks or successful reservations above establish
        // room in every array. Nothing between those checks and these writes can
        // consume that room. Avoid repeating Vec::push's infallible growth path.
        unsafe {
            Self::push_reserved(&mut self.sparse_score_columns, col_u32);
            Self::push_reserved(&mut self.sparse_weights, assignment.dosage_weight);
            Self::push_reserved(
                &mut self.sparse_missing_corrections,
                assignment.missing_correction,
            );
        }
        Ok(())
    }

    /// The caller must establish `values.len() < values.capacity()` first.
    #[inline(always)]
    unsafe fn push_reserved<T>(values: &mut Vec<T>, value: T) {
        let len = values.len();
        debug_assert!(len < values.capacity());
        // SAFETY: the caller reserves this uninitialized element; writing it
        // before extending the length preserves Vec's initialization invariant.
        unsafe {
            values.as_mut_ptr().add(len).write(value);
            values.set_len(len + 1);
        }
    }

    #[cold]
    #[inline(never)]
    fn reserve_entry<T>(values: &mut Vec<T>, name: &'static str) -> Result<(), PrepError> {
        values
            .try_reserve(1)
            .map_err(|e| PrepError::Invariant(format!("Cannot grow CSR {name}: {e}")))
    }

    #[inline(always)]
    fn finish_variant(&mut self) -> Result<(), PrepError> {
        let offset_u64 = u64::try_from(self.sparse_score_columns.len()).map_err(|_| {
            PrepError::Invariant(format!(
                "CSR non-zero count {} exceeds u64::MAX while building row offsets.",
                self.sparse_score_columns.len()
            ))
        })?;
        if self.sparse_row_offsets.len() == self.sparse_row_offsets.capacity() {
            Self::reserve_entry(&mut self.sparse_row_offsets, "row offsets")?;
        }
        // SAFETY: the capacity check or successful reservation provides one slot.
        unsafe { Self::push_reserved(&mut self.sparse_row_offsets, offset_u64) };
        Ok(())
    }

    fn into_parts(self) -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u64>) {
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
        row_keys: Option<&mut Vec<VariantKey>>,
    ) -> Result<(), PrepError> {
        if row_keys
            .as_ref()
            .is_some_and(|keys| keys.len() != required.len())
        {
            return Err(PrepError::Invariant(
                "Row keys and matched variants disagree before reordering.".to_string(),
            ));
        }
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
        if let Some(keys) = row_keys {
            let mut sorted_keys = Vec::new();
            sorted_keys.try_reserve_exact(order.len()).map_err(|e| {
                PrepError::Invariant(format!("Cannot allocate reordered row keys: {e}"))
            })?;
            sorted_keys.extend(order.iter().map(|&row| keys[row]));
            *keys = sorted_keys;
        }
        Ok(())
    }
}

/// Everything the merge-join builds, locus by locus in key order.
struct JoinOutputs {
    required_bim_indices: Vec<BimRowIndex>,
    required_is_complex: Vec<u8>,
    csr_builder: CsrBuilder,
    baseline_missing_sum_by_score: Vec<f64>,
    baseline_errors: Vec<f64>,
    score_variant_counts: Vec<u32>,
    final_complex_rules: Vec<GroupedComplexRule>,
    /// The entries of the row being reconciled, reused across loci.
    row_entries: Vec<(ScoreColumnIndex, SimpleScoreAssignment)>,
    /// The locus of every row, kept only for a block expansion after the join.
    row_keys: Option<Vec<VariantKey>>,
}

impl JoinOutputs {
    fn new(num_scores: usize) -> Result<Self, PrepError> {
        let csr_builder = CsrBuilder::new()?;
        let mut baseline_missing_sum_by_score = Vec::new();
        baseline_missing_sum_by_score
            .try_reserve_exact(num_scores)
            .map_err(|e| PrepError::Invariant(format!("Cannot allocate score baselines: {e}")))?;
        baseline_missing_sum_by_score.resize(num_scores, 0.0f64);
        let mut baseline_errors = Vec::new();
        baseline_errors.try_reserve_exact(num_scores).map_err(|e| {
            PrepError::Invariant(format!("Cannot allocate baseline compensation: {e}"))
        })?;
        baseline_errors.resize(num_scores, 0.0f64);
        Ok(Self {
            required_bim_indices: Vec::new(),
            required_is_complex: Vec::new(),
            csr_builder,
            baseline_missing_sum_by_score,
            baseline_errors,
            score_variant_counts: vec![0u32; num_scores],
            final_complex_rules: Vec::new(),
            row_entries: Vec::new(),
            row_keys: None,
        })
    }

    /// Keeps each row's locus so [`blocks::expand_plan`] can place it.
    fn record_row_keys(&mut self) {
        self.row_keys = Some(Vec::new());
    }

    #[inline(always)]
    fn push_row_key(&mut self, key: VariantKey) {
        if let Some(keys) = &mut self.row_keys {
            keys.push(key);
        }
    }

    /// Builds the plan rows of one locus from the `.bim` rows and score records that share `key`,
    /// each group in input order, by the site rule of [`crate::score::site`]: a `.bim` row's
    /// allele 2 is its REF and allele 1 its ALT, as plink2 reads a `.bim`.
    fn reconcile_locus(
        &mut self,
        key: VariantKey,
        bim_group: &[KeyedBimRecord],
        score_group: &[KeyedScoreRecord],
    ) -> Result<(), PrepError> {
        // A single marker and a single weight need no temporary trees,
        // sets, context vectors, or match lists. Emit their CSR row in
        // exactly the same arithmetic and record order as grouped loci.
        if let ([bim], [score]) = (bim_group, score_group) {
            if let Some(effect_is_ref) = row_side(
                score.effect_allele.as_str(),
                score.other_allele.as_str(),
                bim.allele2.as_str(),
                bim.allele1.as_str(),
            ) {
                let mut assignment = SimpleScoreAssignment {
                    dosage_weight: 0.0,
                    missing_correction: 0.0,
                };
                apply_simple_score_assignment(&mut assignment, score.weight, !effect_is_ref);
                self.required_bim_indices.push(bim.bim_row_index);
                self.required_is_complex.push(0);
                self.push_row_key(key);
                self.csr_builder
                    .push_contribution(score.score_column_index, assignment)?;
                self.csr_builder.finish_variant()?;
                accumulate_baseline(
                    &mut self.baseline_missing_sum_by_score[score.score_column_index.0],
                    &mut self.baseline_errors[score.score_column_index.0],
                    assignment.missing_correction,
                );
                self.score_variant_counts[score.score_column_index.0] += 1;
            }
            return Ok(());
        }

        if let [bim] = bim_group {
            self.row_entries.clear();
            for score in score_group {
                let Some(effect_is_ref) = row_side(
                    score.effect_allele.as_str(),
                    score.other_allele.as_str(),
                    bim.allele2.as_str(),
                    bim.allele1.as_str(),
                ) else {
                    continue;
                };
                let mut assignment = SimpleScoreAssignment {
                    dosage_weight: 0.0,
                    missing_correction: 0.0,
                };
                apply_simple_score_assignment(&mut assignment, score.weight, !effect_is_ref);
                self.row_entries.push((score.score_column_index, assignment));
            }
            if !self.row_entries.is_empty() {
                // Duplicate score lines stay separate entries, each one parsed weight, so the
                // exact plan sums the written decimals rather than an f64 sum of them.
                self.row_entries.sort_by_key(|&(column, _)| column);
                self.required_bim_indices.push(bim.bim_row_index);
                self.required_is_complex.push(0);
                self.push_row_key(key);
                let mut previous = None;
                for &(column, assignment) in &self.row_entries {
                    self.csr_builder.push_contribution(column, assignment)?;
                    accumulate_baseline(
                        &mut self.baseline_missing_sum_by_score[column.0],
                        &mut self.baseline_errors[column.0],
                        assignment.missing_correction,
                    );
                    if previous != Some(column) {
                        self.score_variant_counts[column.0] += 1;
                    }
                    previous = Some(column);
                }
                self.csr_builder.finish_variant()?;
            }
            return Ok(());
        }

        let site = bim_site(bim_group);
        let chromosome = chromosome_label(key.0);
        // Every scored record with its site allele, in input order.
        let mut scored: Vec<(SiteAllele, &KeyedScoreRecord)> = Vec::new();
        for score in score_group {
            let (effect, other) = (score.effect_allele.as_str(), score.other_allele.as_str());
            match site.match_row(effect, other) {
                RowMatch::NoVariant => {}
                RowMatch::Several(first, second) => {
                    return Err(PrepError::AmbiguousReconciliation(format!(
                        "{}: remove the other's .bim row from the genotypes, or the row from the score file.",
                        site.several_error(&chromosome, key.1, effect, other, (first, second), ".bim row")
                    )));
                }
                RowMatch::Unread(_) => {
                    return Err(PrepError::AmbiguousReconciliation(format!(
                        "{} Score genotypes that declare their REF instead, a .pgen fileset or a VCF or BCF file, or remove the variant the row does not mean.",
                        site.unread_error(&chromosome, key.1, effect, other, ".bim rows")
                    )));
                }
                RowMatch::Scores(allele) => scored.push((allele, score)),
            }
        }
        if scored.is_empty() {
            return Ok(());
        }

        // A (variant, score) pair is one matched variant of the score. It adds from its variant's
        // own row alone when one row measures the variant and the pair does not name the site's
        // REF. Otherwise it is resolved per person from the site's rows: several rows of a variant
        // are measurements that must agree, and the site's REF dose is the ploidy less every
        // ALT's copies.
        scored.sort_by_key(|(allele, score)| (allele.variant(), score.score_column_index));
        let mut simple: Vec<Vec<(ScoreColumnIndex, SimpleScoreAssignment)>> =
            vec![Vec::new(); bim_group.len()];
        let mut complex = Vec::new();
        for group in scored.chunk_by(|(a, x), (b, y)| {
            (a.variant(), x.score_column_index) == (b.variant(), y.score_column_index)
        }) {
            let variant = group[0].0.variant();
            self.score_variant_counts[group[0].1.score_column_index.0] += 1;
            let whole_site = group.iter().any(|(allele, _)| site.reads_whole_site(*allele));
            let mut rows = (0..bim_group.len()).filter(|&row| site.row_variants()[row] == variant);
            match (rows.next(), rows.next()) {
                (Some(row), None) if !whole_site => {
                    for (allele, score) in group {
                        let mut assignment = SimpleScoreAssignment {
                            dosage_weight: 0.0,
                            missing_correction: 0.0,
                        };
                        // The effect allele is allele 1 when it is the ALT of a row whose REF is
                        // allele 2, or the REF of one whose REF is allele 1.
                        let alternate = matches!(allele, SiteAllele::Alternate(_));
                        let flipped = alternate == site.first_is_reference()[row];
                        apply_simple_score_assignment(&mut assignment, score.weight, flipped);
                        simple[row].push((score.score_column_index, assignment));
                    }
                }
                _ => complex.extend(group.iter().map(|(_, score)| ScoreInfo {
                    effect_allele: score.effect_allele.to_string(),
                    other_allele: score.other_allele.to_string(),
                    weight: score.weight,
                    score_column_index: score.score_column_index,
                })),
            }
        }
        let has_complex = !complex.is_empty();
        if has_complex {
            self.final_complex_rules.push(GroupedComplexRule {
                locus_chr_pos: (chromosome, key.1),
                possible_contexts: bim_group
                    .iter()
                    .map(|bim| (bim.bim_row_index, bim.allele1.to_string(), bim.allele2.to_string()))
                    .collect(),
                score_applications: complex,
                reference_declared: reference_declared(bim_group),
            });
        }
        // Every row of a site resolved per person is spooled for the complex pass.
        for (bim, mut entries) in bim_group.iter().zip(simple) {
            if entries.is_empty() && !has_complex {
                continue;
            }
            entries.sort_by_key(|&(column, _)| column);
            self.required_bim_indices.push(bim.bim_row_index);
            self.required_is_complex.push(u8::from(has_complex));
            self.push_row_key(key);
            for &(column, assignment) in &entries {
                self.csr_builder.push_contribution(column, assignment)?;
                accumulate_baseline(
                    &mut self.baseline_missing_sum_by_score[column.0],
                    &mut self.baseline_errors[column.0],
                    assignment.missing_correction,
                );
            }
            self.csr_builder.finish_variant()?;
        }
        Ok(())
    }

    /// Reorders the plan's rows so their `.bim` indices ascend.
    fn sort_rows_by_bim_index(&mut self) -> Result<(), PrepError> {
        self.csr_builder.sort_rows_by_bim_index(
            &mut self.required_bim_indices,
            &mut self.required_is_complex,
            self.row_keys.as_mut(),
        )
    }
}

/// The merge-join over rows streamed in key order. Unparsable rows are reported as
/// the join meets them, and what it walks past is kept for the diagnostics of a
/// join that matches nothing.
fn join_streams<B, S>(
    bim_iter: &mut std::iter::Peekable<B>,
    score_iter: &mut std::iter::Peekable<S>,
    outputs: &mut JoinOutputs,
    diagnostics: &mut MergeDiagnosticInfo,
    seen_invalid_bim_chrs: &mut AHashSet<String>,
    seen_invalid_score_chrs: &mut AHashSet<String>,
    effect_only_matches: &mut EffectOnlyMatches,
) -> Result<(), PrepError>
where
    B: Iterator<Item = Result<KeyedBimRecord, PrepError>>,
    S: Iterator<Item = Result<KeyedScoreRecord, PrepError>>,
{
    let mut bim_group = Vec::new();
    let mut score_group = Vec::new();
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
                    // Only a row on a contig gnomon cannot key is skipped here; any
                    // other unusable score row fails the run.
                    let Some(chr_name) = extract_chr_from_parse_error(&msg) else {
                        return Err(PrepError::Parse(msg));
                    };
                    if seen_invalid_score_chrs.insert(chr_name.to_string()) {
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

                if score_group
                    .iter()
                    .any(|record| names_no_single_other_allele(record.other_allele.as_str()))
                {
                    resolve_effect_only_records(
                        key,
                        &bim_group,
                        &mut score_group,
                        effect_only_matches,
                    );
                }
                outputs.reconcile_locus(key, &bim_group, &score_group)?;
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
    Ok(())
}

/// The merge-join over rows already in memory, each side sorted by key with no row
/// errors between: the loci `join_streams` reconciles, in the same order, without
/// stepping through non-matching rows one at a time or recording diagnostics.
fn join_sorted_slices(
    bim: &[KeyedBimRecord],
    scores: &[KeyedScoreRecord],
    outputs: &mut JoinOutputs,
    effect_only_matches: &mut EffectOnlyMatches,
) -> Result<(), PrepError> {
    // Records are borrowed; only a locus with a record to resolve is copied.
    let mut resolved = Vec::new();
    let (mut b, mut s) = (0, 0);
    while b < bim.len() && s < scores.len() {
        let (bim_key, score_key) = (bim[b].key, scores[s].key);
        match bim_key.cmp(&score_key) {
            Ordering::Less => b += leading_count(&bim[b..], |row| row.key < score_key),
            Ordering::Greater => s += leading_count(&scores[s..], |record| record.key < bim_key),
            Ordering::Equal => {
                let bim_end = b + leading_count(&bim[b..], |row| row.key == bim_key);
                let score_end = s + leading_count(&scores[s..], |record| record.key == bim_key);
                let (bim_group, score_group) = (&bim[b..bim_end], &scores[s..score_end]);
                if score_group
                    .iter()
                    .any(|record| names_no_single_other_allele(record.other_allele.as_str()))
                {
                    resolved.clear();
                    resolved.extend_from_slice(score_group);
                    resolve_effect_only_records(
                        bim_key,
                        bim_group,
                        &mut resolved,
                        effect_only_matches,
                    );
                    outputs.reconcile_locus(bim_key, bim_group, &resolved)?;
                } else {
                    outputs.reconcile_locus(bim_key, bim_group, score_group)?;
                }
                (b, s) = (bim_end, score_end);
            }
        }
    }
    Ok(())
}

/// A record that names no single other allele becomes the pair of the one allele of its site
/// that its effect allele names, or is counted and dropped. Records naming their other allele go
/// on unchanged.
fn resolve_effect_only_records(
    key: VariantKey,
    bim_group: &[KeyedBimRecord],
    score_group: &mut Vec<KeyedScoreRecord>,
    effect_only_matches: &mut EffectOnlyMatches,
) {
    let site = bim_site(bim_group);
    score_group.retain_mut(|record| {
        if !names_no_single_other_allele(record.other_allele.as_str()) {
            return true;
        }
        let decision = site.match_row(record.effect_allele.as_str(), record.other_allele.as_str());
        effect_only_matches.record(decision, key);
        let RowMatch::Scores(allele) = decision else {
            return false;
        };
        let variant = site.variants()[allele.variant()];
        record.other_allele = Allele::new(match allele {
            SiteAllele::Alternate(_) => variant.reference,
            SiteAllele::Reference(_) => variant.alternate,
        });
        true
    });
}

/// How many leading `items` satisfy `before`, given that it holds for the first item
/// and for a prefix only. Steps double first, so a short run costs about a scan.
fn leading_count<T>(items: &[T], before: impl Fn(&T) -> bool) -> usize {
    let mut step = 1;
    while step < items.len() && before(&items[step]) {
        step *= 2;
    }
    let low = step / 2;
    low + items[low..step.min(items.len())].partition_point(|item| before(item))
}

#[cfg(test)]
thread_local! {
    /// Set by tests to run the streaming join over rows the slice join would take.
    static FORCE_STREAMING_JOIN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn streaming_join_forced() -> bool {
    FORCE_STREAMING_JOIN.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn streaming_join_forced() -> bool {
    false
}

#[inline(always)]
fn apply_simple_score_assignment(entry: &mut SimpleScoreAssignment, weight: f64, is_flipped: bool) {
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

/// Retain low bits while summing the cohort-independent flipped-allele baseline.
/// This runs once per contribution, not once per person.
#[inline]
fn accumulate_baseline(sum: &mut f64, error: &mut f64, value: f64) {
    if value == 0.0 {
        return;
    }
    let next = *sum + value;
    *error += if sum.abs() >= value.abs() {
        (*sum - next) + value
    } else {
        (value - next) + *sum
    };
    *sum = next;
}

/// The site of the `.bim` rows at one locus. Allele 2 is the REF when every row declares it, as the
/// rows of a `.pvar` do; otherwise the site reads its REF from the alleles, allele 2 first where
/// they leave the choice free.
fn bim_site(bim_group: &[KeyedBimRecord]) -> Site<'_> {
    let rows: Vec<(usize, &str, &str)> = bim_group
        .iter()
        .enumerate()
        .map(|(row, bim)| (row, bim.allele2.as_str(), bim.allele1.as_str()))
        .collect();
    Site::new(&rows, reference_declared(bim_group))
}

/// Whether every row at a locus declares its REF.
fn reference_declared(bim_group: &[KeyedBimRecord]) -> bool {
    bim_group.iter().all(|bim| bim.reference_declared)
}

/// A chromosome code as gnomon writes it in messages.
fn chromosome_label(code: u8) -> String {
    match code {
        23 => "X".to_string(),
        24 => "Y".to_string(),
        25 => "XY".to_string(),
        26 => "MT".to_string(),
        n => n.to_string(),
    }
}

/// Weights from score rows that name no single other allele, by how they matched the
/// variants at their loci. Each weight is one score column of one score file row.
#[derive(Debug, Default)]
pub(crate) struct EffectOnlyMatches {
    matched: u64,
    several_rows: u64,
    no_row: u64,
    /// The first skipped loci, with why each was skipped.
    examples: Vec<(VariantKey, RowMatch)>,
}

impl EffectOnlyMatches {
    const EXAMPLES: usize = 5;

    pub(crate) fn record(&mut self, decision: RowMatch, key: VariantKey) {
        match decision {
            RowMatch::Scores(_) => {
                self.matched += 1;
                return;
            }
            RowMatch::Several(..) | RowMatch::Unread(_) => self.several_rows += 1,
            RowMatch::NoVariant => self.no_row += 1,
        }
        if self.examples.len() < Self::EXAMPLES {
            self.examples.push((key, decision));
        }
    }

    /// Whether no row needed its other allele resolved. A plan built otherwise is not
    /// cached, so every run reports these counts again.
    fn is_empty(&self) -> bool {
        self.matched + self.several_rows + self.no_row == 0
    }

    pub(crate) fn report(&self) {
        if self.matched > 0 {
            eprintln!(
                "> Matched {} weight(s) from score rows that name no single other allele on their effect allele, at loci where it names one allele of the variants there.",
                self.matched
            );
        }
        let skipped = self.several_rows + self.no_row;
        if skipped == 0 {
            return;
        }
        eprintln!(
            "> Warning: Skipped {skipped} weight(s) from score rows that name no single other allele: {} at loci where the effect allele may name more than one allele of the variants there, {} where no variant carries it with a listed other allele. They contribute nothing to any score.",
            self.several_rows, self.no_row
        );
        eprintln!("> Examples (first {}):", self.examples.len());
        for ((chr, pos), decision) in &self.examples {
            let reason = if matches!(decision, RowMatch::Several(..) | RowMatch::Unread(_)) {
                "the effect allele may name more than one allele of the variants there"
            } else {
                "no variant carries the effect allele"
            };
            eprintln!(">   - {}:{pos}: {reason}", chromosome_label(*chr));
        }
    }
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
    /// Errors to yield before the next record: rows on contigs gnomon cannot key,
    /// each skipped on its own, and then any error that ends a file.
    pending_errors: std::collections::VecDeque<PrepError>,
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
    /// `--blocks` names a partition the run cannot use.
    Blocks(String),
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
    prepare_for_computation_with_blocks(
        fileset_prefixes,
        sorted_score_files,
        keep_file,
        score_regions,
        None,
        None,
    )
}

/// [`prepare_for_computation`], expanding every score over `blocks` when one is
/// given (`--blocks`); `blocks_max` is the `--blocks-max` limit.
pub fn prepare_for_computation_with_blocks(
    fileset_prefixes: &[PathBuf],
    sorted_score_files: &[PathBuf],
    keep_file: Option<&Path>,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    blocks: Option<&blocks::BlockPartition>,
    blocks_max: Option<usize>,
) -> Result<PreparationResult, PrepError> {
    let filesets = build_fileset_paths(fileset_prefixes)?;
    // The plan cache only saves time. A plan that cannot be hashed, read, trusted or
    // held within this machine's memory budget is compiled again instead.
    let cache =
        match cache::PlanCache::discover(&filesets, sorted_score_files, score_regions, blocks) {
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
                let all_iids = index_people(&filesets)?;
                let total_people = all_iids.len();
                let (subset, iids, output_to_fam) = resolve_person_subset(keep_file, all_iids)?;
                if let Some(partition) = blocks {
                    // A cached plan under this partition's key is already expanded.
                    let base_scores = plan.names.len() / partition.columns_per_score();
                    partition.check_block_budget(blocks_max, base_scores, iids.len())?;
                }
                return assemble_preparation(
                    plan,
                    &filesets,
                    subset,
                    iids,
                    output_to_fam,
                    total_people,
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
        blocks,
        blocks_max,
        BimRowOrder::Streamed,
    )?;
    if clean && let Some(cache) = &cache {
        // Rehash after compilation so a changed source cannot be published
        // under the digest taken before the compiler opened its readers. A
        // rehash that cannot run now, such as when memory has become short,
        // publishes nothing; only a different digest means the inputs changed.
        match cache::PlanCache::discover(&filesets, sorted_score_files, score_regions, blocks) {
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
    blocks: Option<&blocks::BlockPartition>,
    blocks_max: Option<usize>,
    bim_row_order: BimRowOrder,
) -> Result<(PreparationResult, bool), PrepError> {
    // --- Stage 1: Initial setup ---
    eprintln!("> Stage 1: Indexing subject data...");
    let fileset_paths = build_fileset_paths(fileset_prefixes)?;
    let all_person_iids = index_people(&fileset_paths)?;
    let total_people_in_fam = all_person_iids.len();

    let (person_subset, final_person_iids, output_idx_to_fam_idx) =
        resolve_person_subset(keep_file, all_person_iids)?;

    // --- Stage 2: Global metadata discovery ---
    eprintln!("> Stage 2: Discovering all score columns...");
    let score_names = parse_score_file_headers_only(sorted_score_files)?;
    if let Some(partition) = blocks {
        partition.check_block_budget(blocks_max, score_names.len(), final_person_iids.len())?;
    }
    let score_name_to_col_index: AHashMap<String, ScoreColumnIndex> = score_names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), ScoreColumnIndex(i)))
        .collect();

    // --- Stage 3: Single-pass data collection ---
    eprintln!("> Stage 3: Streaming and collecting data from all input files...");
    let overall_start_time = Instant::now();

    let mut diagnostics = MergeDiagnosticInfo::default();
    let mut effect_only_matches = EffectOnlyMatches::default();
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

    // Build final artifacts incrementally during Stage 3 to avoid materializing
    // genome-scale intermediate maps that duplicate the final CSR/rule structures.
    let mut outputs = JoinOutputs::new(score_names.len())?;
    if blocks.is_some() {
        outputs.record_row_keys();
    }

    // Rows already in memory, in key order and with nothing to report between
    // them, are joined as slices. Anything else walks the streams. Nothing checks
    // that a score file ascends, so the slice join checks it for itself: over a
    // descending file the streaming join's one-row steps decide what matches.
    let plain_rows = if streaming_join_forced() {
        None
    } else {
        bim_rows.plain_records().zip(
            score_iterator
                .plain_records()
                .filter(|records| records.windows(2).all(|pair| pair[0].key <= pair[1].key)),
        )
    };
    if let Some((bim, scores)) = plain_rows {
        join_sorted_slices(bim, scores, &mut outputs, &mut effect_only_matches)?;
        if outputs.required_bim_indices.is_empty() {
            // Only a join that matched nothing reports what it walked past, so
            // walk the same rows the streaming way to describe them, counting
            // resolved score rows afresh.
            outputs = JoinOutputs::new(score_names.len())?;
            if blocks.is_some() {
                outputs.record_row_keys();
            }
            effect_only_matches = EffectOnlyMatches::default();
            join_streams(
                &mut bim.iter().cloned().map(Ok::<_, PrepError>).peekable(),
                &mut scores.iter().cloned().map(Ok::<_, PrepError>).peekable(),
                &mut outputs,
                &mut diagnostics,
                &mut seen_invalid_bim_chrs,
                &mut seen_invalid_score_chrs,
                &mut effect_only_matches,
            )?;
        }
        if rows_sorted_by_key {
            // Rows were emitted in key order; readers visit them in file order.
            outputs.sort_rows_by_bim_index()?;
        }
    } else {
        let mut bim_iter = bim_rows.by_ref().peekable();
        let mut score_iter = score_iterator.by_ref().peekable();
        join_streams(
            &mut bim_iter,
            &mut score_iter,
            &mut outputs,
            &mut diagnostics,
            &mut seen_invalid_bim_chrs,
            &mut seen_invalid_score_chrs,
            &mut effect_only_matches,
        )?;
        drop(bim_iter);
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
                blocks,
                blocks_max,
                BimRowOrder::Sorted,
            );
        }
        if rows_sorted_by_key {
            // Rows were emitted in key order; readers visit them in file order.
            outputs.sort_rows_by_bim_index()?;
        }
        // A score row the join peeked at but never took can still be one that fails
        // the run; finish() below reads the rows beyond it the same way.
        if let Some(Err(_)) = score_iter.peek()
            && let Some(Err(error)) = score_iter.next()
            && !is_unkeyable_contig(&error)
        {
            return Err(error);
        }
    }
    let JoinOutputs {
        required_bim_indices,
        required_is_complex,
        csr_builder,
        mut baseline_missing_sum_by_score,
        baseline_errors,
        score_variant_counts,
        final_complex_rules,
        row_keys,
        ..
    } = outputs;

    let region_filter_hits = score_iterator.take_region_filter_hits();
    let mut rejected_score_rows = score_iterator.finish()?;
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
    rejected_score_rows.report();
    effect_only_matches.report();

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
        && rejected_score_rows.total() == 0
        && effect_only_matches.is_empty()
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
    for (sum, error) in baseline_missing_sum_by_score.iter_mut().zip(baseline_errors) {
        *sum += error;
    }
    let mut plan = cache::VariantPlan {
        weights: cache::PlanWeights::Parsed {
            weights: sparse_weights,
            corrections: sparse_missing_corrections,
            baseline: baseline_missing_sum_by_score,
        },
        columns: sparse_score_columns,
        offsets: sparse_row_offsets,
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
    if let Some(partition) = blocks {
        let expansion_start = Instant::now();
        let row_keys = row_keys.ok_or_else(|| {
            PrepError::Invariant("The join recorded no row keys for the block expansion.".into())
        })?;
        plan = blocks::expand_plan(plan, &row_keys, partition)?;
        eprintln!(
            "> Expanded {} score(s) into {} columns over {} in {:.2?}.",
            plan.names.len() / partition.columns_per_score(),
            plan.names.len(),
            partition.describe(),
            expansion_start.elapsed()
        );
    }
    Ok((
        assemble_preparation(
            plan,
            &fileset_paths,
            person_subset,
            final_person_iids,
            output_idx_to_fam_idx,
            total_people_in_fam,
        )?,
        clean,
    ))
}

fn assemble_preparation(
    mut plan: cache::VariantPlan,
    fileset_paths: &[FilesetPaths],
    person_subset: PersonSubset,
    final_person_iids: Vec<String>,
    output_idx_to_fam_idx: Vec<OriginalPersonIndex>,
    total_people_in_fam: usize,
) -> Result<PreparationResult, PrepError> {
    if plan.starts.len() != fileset_paths.len() {
        return Err(PrepError::Invariant(
            "Variant plan fileset count mismatch.".into(),
        ));
    }
    let num_people_to_score = final_person_iids.len();
    let num_reconciled_variants = plan.required.len();
    // The join grows its arrays as it reads them, and a run keeps them to its end: the memory
    // budget charges what they hold (csr_heap_bytes counts capacity), so a compiled plan gives back
    // the capacity it grew past its length, which a loaded plan never had. On 140,000 variants and
    // 256 scores that was 358.77 MiB of a cold run's floor, the weights' 238.56 of it once they
    // became the exact plan's integers in place.
    plan.columns.shrink_to_fit();
    plan.offsets.shrink_to_fit();
    // A saved plan holds its exact plan; a compiled one's weights become theirs in place.
    let exact = match plan.weights {
        cache::PlanWeights::Exact(exact) => exact,
        cache::PlanWeights::Parsed {
            mut weights,
            corrections,
            ..
        } => {
            weights.shrink_to_fit();
            ExactPlan::new(
                weights,
                &corrections,
                &plan.columns,
                &plan.offsets,
                &plan.complex,
                &plan.names,
            )
            .map_err(|error| match error {
                PlanError::Invariant(message) => PrepError::Invariant(message),
                PlanError::Unrepresentable(message) => PrepError::Parse(message),
            })?
        }
    };
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
    if output_idx_to_fam_idx.len() != num_people_to_score {
        return Err(PrepError::Invariant(
            "Person index mapping does not cover every scored person.".into(),
        ));
    }
    let mut person_fam_to_output_idx = vec![None; total_people_in_fam];

    for (output_idx, &OriginalPersonIndex(original_fam_idx)) in
        output_idx_to_fam_idx.iter().enumerate()
    {
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
        exact,
        plan.columns,
        plan.offsets,
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
            if uses_pgen_fileset(prefix, &bed)? {
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
fn uses_pgen_fileset(prefix: &Path, bed: &Path) -> Result<bool, PrepError> {
    if is_remote_path(prefix) {
        let bim = apply_extension(prefix, "bim")?;
        let Err(bim_error) = open_text_source(&bim) else {
            return Ok(false);
        };
        if open_text_source(&apply_extension(prefix, "pvar")?).is_ok() {
            return Ok(true);
        }
        return Err(map_pipeline_error(bim_error, bim));
    }
    Ok(!bed.is_file() && apply_extension(prefix, "pgen").is_ok_and(|p| p.is_file()))
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
    fn flipped_allele_baseline_retains_small_contributions_after_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("panel");
        write_bim_fileset(
            &prefix,
            &["1 a 0 100 A G\n", "1 b 0 200 A G\n", "1 c 0 300 A G\n"],
            1,
        );
        let weights = dir.path().join("weights.tsv");
        std::fs::write(
            &weights,
            "variant_id\teffect_allele\tother_allele\tS\n1:100\tA\tG\t4503599627370496\n1:200\tA\tG\t0.5\n1:300\tA\tG\t-4503599627370496\n",
        )
        .unwrap();
        let prep = prepare_for_computation(&[prefix], &[weights], None, None).unwrap();
        assert_eq!(prep.baseline_missing_sum_by_score(), &[1.0]);
        assert_eq!(prep.score_variant_counts, [3]);
    }

    #[test]
    fn score_weights_retain_f64_precision_and_require_finite_values() {
        for (text, expected) in [
            ("1.0000000000000002", 1.0 + f64::EPSILON),
            ("1e-200", 1e-200f64),
            ("1e200", 1e200f64),
            ("-0", -0.0f64),
        ] {
            assert_eq!(
                parse_weight(text).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        for text in ["NaN", "inf", "-inf", "1e999", "not-a-weight"] {
            let error = parse_weight(text).unwrap_err().to_string();
            assert!(!error.is_empty());
        }
    }

    #[test]
    fn a_remote_prefix_without_a_bim_names_the_bim() {
        // Nothing listens on this port once the listener is dropped, so the .bim and
        // the .pvar probes both fail at once.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let prefix = PathBuf::from(format!("http://{}/cohort", listener.local_addr().unwrap()));
        drop(listener);
        let error = build_fileset_paths(std::slice::from_ref(&prefix))
            .err()
            .expect("no fileset behind the prefix")
            .to_string();
        assert!(error.contains("cohort.bim"), "{error}");
        assert!(!error.contains("psam"), "{error}");
    }


    #[test]
    fn csr_reordering_preserves_empty_rows_and_weight_bits() {
        let mut csr = CsrBuilder::new().unwrap();
        for entries in [
            vec![(2, -0.0f64, 2.0f64)],
            vec![],
            vec![(0, 0.1f64, -0.0f64), (3, 0.2f64, 3.0f64)],
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
        csr.sort_rows_by_bim_index(&mut rows, &mut flags, None).unwrap();
        assert_eq!(rows, [BimRowIndex(1), BimRowIndex(5), BimRowIndex(10)]);
        assert_eq!(flags, [1, 0, 0]);
        assert_eq!(csr.sparse_row_offsets, [0, 0, 2, 3]);
        assert_eq!(csr.sparse_score_columns, [0, 3, 2]);
        assert_eq!(
            csr.sparse_weights
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [0.1f64, 0.2, -0.0].map(f64::to_bits),
        );
        assert_eq!(
            csr.sparse_missing_corrections
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            [-0.0f64, 3.0, 2.0].map(f64::to_bits),
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

    type RowPlan = (String, u8, Vec<(u32, u64, u64)>);
    type RulePlan = (
        (String, u32),
        Vec<(String, String, String)>,
        Vec<(String, String, u64, usize)>,
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
    fn slice_join_builds_what_the_streaming_join_builds() {
        let dir = tempfile::tempdir().unwrap();
        // A locus before any score, a flip, one row weighted twice in one column, a
        // split multiallelic locus, a mismatch, runs of rows on either side alone,
        // an indel, and three chromosomes.
        let mixed_rows = [
            "1 a 0 50 C T\n",
            "1 b 0 100 A G\n",
            "1 c 0 200 A G\n",
            "1 d 0 300 A C\n",
            "1 e 0 300 A T\n",
            "1 f 0 400 G T\n",
            "1 g 0 450 G T\n",
            "1 h 0 460 G T\n",
            "2 i 0 100 AT A\n",
            "X j 0 700 C G\n",
        ];
        let mixed_weights = "variant_id\teffect_allele\tother_allele\tS1\tS2\n\
            1:10\tA\tG\t9\t9\n\
            1:100\tA\tG\t0.5\t\n\
            1:200\tG\tA\t-1.25\t2\n\
            1:200\tA\tG\t\t0.125\n\
            1:300\tA\tC\t3\t\n\
            1:300\tA\tT\t\t4\n\
            1:400\tC\tA\t1\t1\n\
            1:470\tG\tT\t5\t5\n\
            2:100\tAT\tA\t0.75\t\n\
            2:900\tA\tG\t1\t1\n\
            X:700\tG\tC\t2.5\t-2.5\n";
        let unsorted_rows = ["1 b 0 200 A G\n", "1 a 0 100 A G\n", "1 c 0 300 A G\n"];
        // Rows naming no single other allele: a simple locus, a split locus where one
        // row carries the effect allele, a candidate list picking one of two rows, a
        // locus several rows carry, one no row carries, and a written pair.
        let effect_only_rows = [
            "1 k 0 100 A G\n",
            "1 m 0 300 C T\n",
            "1 n 0 300 G A\n",
            "1 p 0 400 A G\n",
            "1 q 0 400 A AT\n",
            "1 r 0 500 G C\n",
            "1 s 0 600 A G\n",
        ];
        let cases: [(&str, &[&str], &str); 6] = [
            ("mixed", &mixed_rows, mixed_weights),
            (
                "disjoint",
                &mixed_rows,
                "variant_id\teffect_allele\tother_allele\tS1\n1:999\tA\tG\t1\n3:5\tA\tG\t1\n",
            ),
            (
                "no_allele_matches",
                &mixed_rows,
                "variant_id\teffect_allele\tother_allele\tS1\n1:100\tC\tT\t1\n1:300\tG\tC\t1\n",
            ),
            (
                "unsorted_bim",
                &unsorted_rows,
                "variant_id\teffect_allele\tother_allele\tS1\n1:100\tA\tG\t1\n1:200\tG\tA\t2\n1:300\tA\tG\t3\n",
            ),
            (
                "descending_scores",
                &mixed_rows,
                "variant_id\teffect_allele\tother_allele\tS1\n1:300\tA\tC\t1\n1:100\tA\tG\t2\n2:100\tAT\tA\t3\n",
            ),
            (
                "effect_only",
                &effect_only_rows,
                "variant_id\teffect_allele\tother_allele\tS1\n1:100\tA\t.\t1\n1:300\tA\t.\t2\n1:400\tA\tAT/C\t3\n1:400\tA\t.\t4\n1:500\tT\t.\t5\n1:600\tG\tA\t6\n",
            ),
        ];
        for (name, rows, weights_text) in cases {
            let prefix = dir.path().join(name);
            write_bim_fileset(&prefix, rows, 4);
            let weights = dir.path().join(format!("{name}.tsv"));
            std::fs::write(&weights, weights_text).unwrap();
            // The retry entry point compiles without the plan cache.
            let describe = |forced: bool| {
                FORCE_STREAMING_JOIN.with(|force| force.set(forced));
                let result = prepare_for_computation_with_retry(
                    std::slice::from_ref(&prefix),
                    std::slice::from_ref(&weights),
                    None,
                    None,
                    None,
                    None,
                    BimRowOrder::Streamed,
                );
                FORCE_STREAMING_JOIN.with(|force| force.set(false));
                match result {
                    Ok(prep) => format!("{prep:?}"),
                    Err(error) => format!("error {error}"),
                }
            };
            let sliced = describe(false);
            assert_eq!(sliced, describe(true), "{name}");
            let expect_plan = matches!(
                name,
                "mixed" | "unsorted_bim" | "descending_scores" | "effect_only"
            );
            assert_eq!(
                !sliced.starts_with("error"),
                expect_plan,
                "{name}: {sliced}"
            );
        }
    }

    #[test]
    fn leading_count_finds_every_prefix_length() {
        for len in 1..40usize {
            let items: Vec<usize> = (0..len).collect();
            for prefix in 1..=len {
                assert_eq!(
                    leading_count(&items, |&i| i < prefix),
                    prefix,
                    "{len} {prefix}"
                );
            }
        }
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
    fn unusable_score_rows_are_skipped_alone_or_fail_the_run() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("panel");
        write_bim_fileset(
            &prefix,
            &[
                "1 a 0 100 A G\n",
                "1 b 0 200 C T\n",
                "1 c 0 300 A C\n",
                "1 d 0 400 G T\n",
            ],
            3,
        );
        // Region filters send score files through the streaming merge; without them
        // the files are parsed whole. Both paths must agree.
        let whole_chromosome: HashMap<String, GenomicRegion> = [(
            "S".to_string(),
            GenomicRegion {
                chromosome: 1,
                start: 0,
                end: u32::MAX,
            },
        )]
        .into_iter()
        .collect();
        let weights = dir.path().join("weights.tsv");
        std::fs::write(
            &weights,
            "variant_id\teffect_allele\tother_allele\tS\n\
             1:100\tG\tA\t0.5\n\
             chrUn_x:5\tA\tC\t1\n\
             1:300\tC\tN\t2\n\
             1:400\tT\tG\t0.25\n",
        )
        .unwrap();
        for regions in [None, Some(&whole_chromosome)] {
            let prep = prepare_for_computation(
                std::slice::from_ref(&prefix),
                std::slice::from_ref(&weights),
                None,
                regions,
            )
            .unwrap();
            // Before, the unkeyable contig silently ended the file after 1:100.
            assert_eq!(
                prep.score_variant_counts,
                vec![2],
                "region filters {}",
                regions.is_some()
            );
        }
        for (name, row, problem) in [
            ("NA inside", "1:200\tT\tC\tNA\n", "Invalid weight 'NA'"),
            ("nan beyond", "2:5\tA\tG\tnan\n", "Invalid weight 'nan'"),
            ("position inside", "1:x\tA\tG\t1\n", "Invalid position"),
            ("position beyond", "2:bad\tA\tG\t1\n", "Invalid position"),
        ] {
            let bad = dir.path().join(format!("{}.tsv", name.replace(' ', "_")));
            std::fs::write(
                &bad,
                format!("variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.5\n{row}"),
            )
            .unwrap();
            for regions in [None, Some(&whole_chromosome)] {
                let error = prepare_for_computation(
                    std::slice::from_ref(&prefix),
                    std::slice::from_ref(&bad),
                    None,
                    regions,
                )
                .err()
                .unwrap()
                .to_string();
                assert!(
                    error.contains("line 3") && error.contains(problem),
                    "{name}, region filters {}: {error}",
                    regions.is_some()
                );
            }
        }
    }

    /// The site of `.bim` rows given as (allele 1, allele 2).
    fn site_of<'a>(rows: &[(&'a str, &'a str)]) -> Site<'a> {
        let rows: Vec<(usize, &str, &str)> = rows
            .iter()
            .enumerate()
            .map(|(row, &(allele1, allele2))| (row, allele2, allele1))
            .collect();
        Site::new(&rows, false)
    }

    #[test]
    fn explicit_other_alleles_are_left_to_the_pair_rule() {
        let site = site_of(&[("A", "G")]);
        assert_eq!(site.match_row("A", "T"), RowMatch::NoVariant);
        assert_eq!(
            site.match_row("A", "G"),
            RowMatch::Scores(SiteAllele::Alternate(0))
        );
    }

    #[test]
    fn a_missing_other_allele_resolves_to_the_one_allele_its_effect_allele_names() {
        // A split multiallelic locus where only the second row carries the effect allele.
        let site = site_of(&[("C", "T"), ("G", "A")]);
        assert_eq!(
            site.match_row("A", "."),
            RowMatch::Scores(SiteAllele::Reference(1))
        );
        assert_eq!(
            site.match_row("A", "G/T"),
            RowMatch::Scores(SiteAllele::Reference(1))
        );
        assert_eq!(site.match_row("A", "C/T"), RowMatch::NoVariant);
        assert_eq!(site_of(&[("C", "T")]).match_row("G", "."), RowMatch::NoVariant);

        // Two rows carry the effect allele: ambiguous. The candidates pick the second, whose REF a
        // `.bim` does not write: A is its ALT if AT is its REF, and the site's REF otherwise.
        let site = site_of(&[("A", "G"), ("A", "AT")]);
        assert_eq!(site.match_row("A", "."), RowMatch::Several(0, 1));
        assert_eq!(site.match_row("A", "AT/C"), RowMatch::Unread(1));
        // Both carry it as the REF, the one REF of the site.
        let site = site_of(&[("G", "A"), ("T", "A")]);
        assert_eq!(
            site.match_row("A", "."),
            RowMatch::Scores(SiteAllele::Reference(0))
        );
    }

    #[test]
    fn rows_naming_no_single_other_allele_plan_like_their_written_pairs() {
        let dir = tempfile::tempdir().unwrap();
        // A simple locus, a flip, a split multiallelic locus where both rows carry A,
        // one where only the second row carries T, and one more simple locus.
        let rows = [
            "1 rs100 0 100 A G\n",
            "1 rs200 0 200 C T\n",
            "1 rs300a 0 300 A C\n",
            "1 rs300b 0 300 A T\n",
            "1 rs400a 0 400 G C\n",
            "1 rs400b 0 400 T C\n",
            "1 rs500 0 500 G A\n",
        ];
        let prefix = dir.path().join("effect_only_panel");
        write_bim_fileset(&prefix, &rows, 4);
        let prepare = |name: &str, body: &str| {
            let weights = dir.path().join(name);
            std::fs::write(
                &weights,
                format!("variant_id\teffect_allele\tother_allele\tS\n{body}"),
            )
            .unwrap();
            prepare_for_computation(
                std::slice::from_ref(&prefix),
                std::slice::from_ref(&weights),
                None,
                None,
            )
            .unwrap()
        };

        // 1:300 A is the REF both rows there carry, named through either pair.
        let pairs = prepare(
            "pairs.tsv",
            "1:100\tA\tG\t0.5\n1:200\tT\tC\t-0.25\n1:300\tA\tC\t9\n1:400\tT\tC\t2\n1:500\tA\tG\t0.125\n",
        );
        // The same weights with the other allele unknown or listed as candidates, plus a locus the
        // genotypes lack.
        let effect_only = prepare(
            "effect_only.tsv",
            "1:100\tA\t.\t0.5\n1:200\tT\t.\t-0.25\n1:300\tA\t.\t9\n1:400\tT\tC/G\t2\n1:500\tA\tG/T\t0.125\n1:600\tA\t.\t7\n",
        );
        assert_eq!(
            plan_by_row_text(&effect_only, &rows),
            plan_by_row_text(&pairs, &rows)
        );
        assert_eq!(effect_only.score_variant_counts, pairs.score_variant_counts);

        // No variant carries C at 1:100, and at 1:400 the only row carrying T pairs it with an
        // unlisted allele: both dropped; the explicit pair stays.
        let skipped = prepare(
            "skipped.tsv",
            "1:100\tC\t.\t1\n1:400\tT\tA/G\t1\n1:500\tA\tG\t1\n",
        );
        let (matched, rules) = plan_by_row_text(&skipped, &rows);
        assert_eq!(
            matched
                .iter()
                .map(|(text, _, _)| text.as_str())
                .collect::<Vec<_>>(),
            vec![rows[6]]
        );
        assert!(rules.is_empty());
        assert_eq!(skipped.score_variant_counts, vec![1]);
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
            cache::PlanCache::discover(&files, std::slice::from_ref(&weights), None, None).unwrap()
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
    fn a_compiled_plan_charges_the_memory_its_saved_plan_does() {
        // More entries and rows than the join's first capacities hold, so its arrays grow past
        // their lengths. Inputs no other test compiles, so no other test writes this plan.
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("charged_plan_panel");
        let rows: Vec<String> = (0..37).map(|i| format!("1 v{i} 0 {} A G\n", 1000 + i)).collect();
        write_bim_fileset(&prefix, &rows.iter().map(String::as_str).collect::<Vec<_>>(), 3);
        let weights = dir.path().join("charged.tsv");
        let lines: String = (0..37)
            .map(|i| format!("1:{}\tG\tA\t0.{}\t-{i}.5\t{}e-3\n", 1000 + i, i + 1, i + 7))
            .collect();
        std::fs::write(&weights, format!("variant_id\teffect_allele\tother_allele\tS\tT\tU\n{lines}")).unwrap();
        let files = build_fileset_paths(std::slice::from_ref(&prefix)).unwrap();
        let Some(key) =
            cache::PlanCache::discover(&files, std::slice::from_ref(&weights), None, None).unwrap()
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
            .unwrap()
        };
        let compiled = prepare();
        assert!(key.load().unwrap().is_some(), "the compiled plan was saved");
        let loaded = prepare();
        assert_eq!(compiled.sparse_score_columns().len(), 111);
        assert_eq!(compiled.csr_heap_bytes(), loaded.csr_heap_bytes());
    }

    #[test]
    fn keep_files_name_people_by_iid_or_by_plink_fid_iid_rows() {
        let dir = tempfile::tempdir().unwrap();
        let iids: Vec<String> = ["I0", "I1", "I2", "I3"].map(String::from).to_vec();
        let keep = |text: &str| {
            let path = dir.path().join("keep.txt");
            std::fs::write(&path, text).unwrap();
            resolve_person_subset(Some(&path), iids.clone())
        };
        let indices = |subset: PersonSubset| match subset {
            PersonSubset::Indices(indices) => indices,
            PersonSubset::All => panic!("a keep file selects a subset"),
        };

        let (by_iid, by_iid_names, by_iid_rows) = keep("I2\nI0\n").unwrap();
        assert_eq!(indices(by_iid), vec![0, 2]);
        assert_eq!(by_iid_names, vec!["I0", "I2"]);
        assert_eq!(
            by_iid_rows,
            vec![OriginalPersonIndex(0), OriginalPersonIndex(2)]
        );

        // plink2's header, tab- and space-separated FID IID rows, and one person twice.
        let (plink, plink_names, _) = keep("#FID\tIID\nF2\tI2\nF0 I0\nI2\n").unwrap();
        assert_eq!(indices(plink), vec![0, 2]);
        assert_eq!(plink_names, vec!["I0", "I2"]);

        let error = keep("F9\tI9\n").unwrap_err().to_string();
        assert!(error.contains("I9"), "the unmatched row is named: {error}");

        // Without a keep file everyone is scored in .fam order.
        let (everyone, names, rows) = resolve_person_subset(None, iids.clone()).unwrap();
        assert!(matches!(everyone, PersonSubset::All));
        assert_eq!(names, iids);
        assert_eq!(rows, (0..4).map(OriginalPersonIndex).collect::<Vec<_>>());
    }

    #[test]
    fn people_are_indexed_from_filesets_that_agree_on_unique_iids() {
        let dir = tempfile::tempdir().unwrap();
        let fileset = |name: &str, fam: &str| {
            let prefix = dir.path().join(name);
            std::fs::write(prefix.with_extension("fam"), fam).unwrap();
            FilesetPaths {
                bed: prefix.with_extension("bed"),
                bim: prefix.with_extension("bim"),
                fam: prefix.with_extension("fam"),
            }
        };
        let shared = "F A 0 0 1 -9\nF B 0 0 2 -9\n";

        // A .fam named twice is read once; another with the same IIDs agrees even
        // when its other columns differ.
        let people = index_people(&[
            fileset("chr1", shared),
            fileset("chr2", "G\tA\t0\t0\t0\t1\r\nG\tB\t0\t0\t0\t1\r\n"),
            fileset("chr1", shared),
        ])
        .unwrap();
        assert_eq!(people, vec!["A", "B"]);

        let error = index_people(&[
            fileset("chr1", shared),
            fileset("chr3", "F A 0 0 1 -9\nF C 0 0 2 -9\n"),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("chr3.fam"), "{error}");

        // The first repeat in file order is the one named.
        let error = index_people(&[fileset("dup", "F A\nF B\nF C\nF B\nF A\n")])
            .unwrap_err()
            .to_string();
        assert!(error.contains("Duplicate IID 'B'"), "{error}");

        let error = index_people(&[]).unwrap_err().to_string();
        assert!(error.contains("No individuals found"), "{error}");
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
            let key = cache::PlanCache::discover(&files, std::slice::from_ref(&weights), None, None)
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
        assert_eq!(prep.sparse_row_offsets(), &[0, 9, 10, 12]);
        for (row, expected) in [
            vec![
                // Duplicate lines stay separate entries, in input order, so exact sums use the
                // written weights: 2^24 + 1 - 2^24 is 1 whatever f64 would retain.
                (columns[0], 16777216.0f64, 0.0f64),
                (columns[0], 1.0, 0.0),
                (columns[0], -16777216.0, 0.0),
                (columns[0], 0.0, 0.0),
                (columns[1], 1.0, 0.0),
                (columns[1], 0.0, 0.0),
                (columns[1], 0.0, 0.0),
                (columns[1], -0.5, 1.0),
                (columns[2], -0.25, 0.5),
            ],
            vec![(columns[2], 4.0, 0.0)],
            vec![(columns[0], 2.0, 0.0), (columns[1], 3.0, 0.0)],
        ]
        .into_iter()
        .enumerate()
        {
            let mut expected = expected;
            expected.sort_by_key(|x| x.0);
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
        assert_eq!(prep.required_bim_indices, [0, 1, 2, 3, 4, 5].map(BimRowIndex));
        // The two 1:300 lines stay two entries, each one written weight, flipped.
        assert_eq!(prep.sparse_row_offsets(), &[0, 1, 2, 3, 3, 3, 5]);
        assert_eq!(prep.sparse_weights(), &[0.25, -0.5, -0.75, 0.125, -0.375]);
        assert_eq!(prep.sparse_missing_corrections(), &[0.0, 1.0, 0.0, -0.25, 0.75]);
        assert_eq!(prep.baseline_missing_sum_by_score(), &[1.5]);
        assert_eq!(prep.score_variant_counts, [5]);
        // 1:250 C is the REF of both rows there, so its dose is two less both ALTs' copies:
        // the site's two rows are resolved per person, and both are spooled.
        assert_eq!(prep.required_is_complex(), &[0, 0, 0, 1, 1, 0]);
        assert_eq!(prep.complex_rules.len(), 1);
        assert_eq!(
            prep.complex_rules[0].possible_contexts,
            [
                (BimRowIndex(3), "A".into(), "C".into()),
                (BimRowIndex(4), "G".into(), "C".into())
            ]
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
        let rows = [(0.35f64, false), (0.10f64, true), (-0.05f64, false)];
        let mut agg = SimpleScoreAssignment {
            dosage_weight: 0.0,
            missing_correction: 0.0,
        };
        for (w, is_flipped) in rows {
            apply_simple_score_assignment(&mut agg, w, is_flipped);
        }

        for dosage in [0.0f64, 1.0, 2.0] {
            let row_by_row = rows
                .iter()
                .map(|(w, is_flipped)| {
                    if *is_flipped {
                        w * (2.0 - dosage)
                    } else {
                        w * dosage
                    }
                })
                .sum::<f64>();
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

/// Whether an error names a chromosome label gnomon cannot key. A score or .bim row
/// on such a contig is skipped on its own, with one warning per name.
fn is_unkeyable_contig(error: &PrepError) -> bool {
    matches!(error, PrepError::Parse(msg) if extract_chr_from_parse_error(msg).is_some())
}

/// Parses a non-empty weight field. Only a finite decimal is usable.
fn parse_weight(text: &str) -> Result<f64, String> {
    match text.parse::<f64>() {
        Ok(weight) if weight.is_finite() => Ok(weight),
        Ok(_) => Err("not a finite number".to_string()),
        Err(err) => Err(err.to_string()),
    }
}

/// The error for a weight field that is not a finite number. `plink2 --score` stops
/// on the same coefficients ("Invalid coefficient 'na' on line 11 of …"); before,
/// gnomon silently dropped every later row of the file, or scored NaN or inf.
fn unusable_weight_error(text: &str, line_number: u64, path: &Path, problem: &str) -> PrepError {
    PrepError::Parse(format!(
        "Invalid weight '{text}' on line {line_number} of score file '{}': {problem}. Weights must be finite numbers; leave the field empty for a score that does not use the variant.",
        path.display()
    ))
}

/// Score file rows whose other_allele is 'N'. No .bim record pairs with an unknown
/// allele, so each such row is skipped on its own, and they are reported together
/// after Stage 3.
#[derive(Debug, Default)]
struct RejectedScoreRows {
    unknown_other_allele: u64,
    /// The earliest rows of each file, as (file, physical line).
    examples: Vec<(PathBuf, u64)>,
}

impl RejectedScoreRows {
    const EXAMPLES: usize = 5;

    fn record(&mut self, path: &Path, line: u64) {
        self.unknown_other_allele += 1;
        if self
            .examples
            .iter()
            .filter(|(example_path, _)| example_path == path)
            .count()
            < Self::EXAMPLES
        {
            self.examples.push((path.to_path_buf(), line));
        }
    }

    fn absorb(&mut self, other: RejectedScoreRows) {
        self.unknown_other_allele += other.unknown_other_allele;
        self.examples.extend(other.examples);
    }

    fn total(&self) -> u64 {
        self.unknown_other_allele
    }

    fn report(&mut self) {
        if self.total() == 0 {
            return;
        }
        self.examples.sort();
        eprintln!(
            "> Warning: Skipped {} score file row(s) whose other_allele is 'N'. No .bim record can pair with an unknown allele, so they contribute nothing to any score.",
            self.unknown_other_allele
        );
        eprintln!(
            "> Examples (first {}):",
            self.examples.len().min(Self::EXAMPLES)
        );
        for (path, line) in self.examples.iter().take(Self::EXAMPLES) {
            eprintln!(">   - {}: line {line}", path.display());
        }
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

    /// Every row still to come, when all of them are in memory in key order with no
    /// row error between them.
    fn plain_records(&self) -> Option<&[KeyedBimRecord]> {
        match self {
            Self::Sorted(records) => Some(records.as_slice()),
            Self::Parsed {
                records, errors, ..
            } if errors.len() == 0 => Some(records.as_slice()),
            Self::Streamed { .. } | Self::Parsed { .. } => None,
        }
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
            let mut header_lines = 0u64;

            loop {
                header_line.clear();
                if reader
                    .read_line(&mut header_line)
                    .map_err(|e| PrepError::Io(e, path.clone()))?
                    == 0
                {
                    break;
                }
                header_lines += 1;
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
                path: path.clone(),
                header_lines,
                rejected: RejectedScoreRows::default(),
            });
        }

        let region_filter_hits = region_filters
            .as_ref()
            .map(|_| vec![false; score_name_to_col_index.len()]);

        let mut iter = Self {
            streams,
            heap: BinaryHeap::new(),
            file_column_maps,
            pending_errors: std::collections::VecDeque::new(),
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

                    Self::read_line_into_buffer(
                        stream,
                        column_map,
                        None,
                        None,
                        &mut self.pending_errors,
                    )?
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

                Self::read_line_into_buffer(
                    stream,
                    column_map,
                    region_filters,
                    region_hits_slice,
                    &mut self.pending_errors,
                )?
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
        pending_errors: &mut std::collections::VecDeque<PrepError>,
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
            let line_number = stream.header_lines + stream.file_line_number;
            if other_allele == "N" {
                // No .bim record pairs with an unknown other allele. The row is
                // skipped on its own and reported; the rest of the file still counts.
                stream.rejected.record(&stream.path, line_number);
                continue;
            }

            let mut key_parts = variant_id.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");
            let key = match parse_key(chr_str, pos_str) {
                Ok(key) => key,
                Err(error) if is_unkeyable_contig(&error) => {
                    pending_errors.push_back(error);
                    continue;
                }
                Err(PrepError::Parse(msg)) => {
                    return Err(PrepError::Parse(format!(
                        "Score file '{}' line {line_number}: {msg}",
                        stream.path.display()
                    )));
                }
                Err(error) => return Err(error),
            };
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
                let weight = parse_weight(weight_str).map_err(|problem| {
                    unusable_weight_error(weight_str, line_number, &stream.path, &problem)
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
        if let Some(e) = self.pending_errors.pop_front() {
            return Some(Err(e));
        }

        let top_item = self.heap.pop()?;
        let record_to_return = top_item.record;
        let file_idx = top_item.file_idx;

        if let Err(e) = self.replenish_from_stream(file_idx) {
            self.pending_errors.push_back(e);
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
    /// Every record still to come, when all of them are in memory in merge order
    /// with nothing to report between them.
    fn plain_records(&self) -> Option<&[KeyedScoreRecord]> {
        match self {
            Self::Parsed(parsed) => parsed.plain_records(),
            Self::Streamed(_) => None,
        }
    }

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

    /// Reads whatever the merge-join left unread, so a row that fails the run fails
    /// it wherever it sits, and returns every rejected row for the report. The
    /// malformed-line count keeps its meaning: lines the join read.
    fn finish(&mut self) -> Result<RejectedScoreRows, PrepError> {
        match self {
            Self::Streamed(merge) => {
                let malformed: Vec<usize> = merge
                    .streams
                    .iter()
                    .map(|s| s.malformed_lines_count)
                    .collect();
                while let Some(item) = merge.next() {
                    if let Err(error) = item
                        && !is_unkeyable_contig(&error)
                    {
                        return Err(error);
                    }
                }
                let mut rejected = RejectedScoreRows::default();
                for (stream, count) in merge.streams.iter_mut().zip(malformed) {
                    stream.malformed_lines_count = count;
                    rejected.absorb(std::mem::take(&mut stream.rejected));
                }
                Ok(rejected)
            }
            Self::Parsed(parsed) => parsed.finish(),
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

/// The people to score, their IIDs, and the `.fam` row of each, all in output order.
/// Without a keep file everyone is scored in `.fam` order: the IIDs are moved rather
/// than copied, and no IID is hashed.
fn resolve_person_subset(
    keep_file: Option<&Path>,
    all_person_iids: Vec<String>,
) -> Result<(PersonSubset, Vec<String>, Vec<OriginalPersonIndex>), PrepError> {
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
        // index_people has checked that every row index fits u32.
        let iid_to_original_idx: AHashMap<&str, u32> = all_person_iids
            .iter()
            .enumerate()
            .map(|(idx, iid)| (iid.as_str(), idx as u32))
            .collect();

        let mut found_people = Vec::with_capacity(lines_to_keep.len());
        let mut missing_ids = Vec::new();

        for line in lines_to_keep {
            match resolve_keep_line(&line, &iid_to_original_idx) {
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
        let output_to_fam = found_people
            .iter()
            .map(|&(idx, _)| OriginalPersonIndex(idx))
            .collect();
        let subset_indices = found_people.into_iter().map(|(idx, _)| idx).collect();
        Ok((
            PersonSubset::Indices(subset_indices),
            final_person_iids,
            output_to_fam,
        ))
    } else {
        // index_people has checked that every row index fits u32.
        let output_to_fam = (0..all_person_iids.len())
            .map(|idx| OriginalPersonIndex(idx as u32))
            .collect();
        Ok((PersonSubset::All, all_person_iids, output_to_fam))
    }
}

/// Resolves one keep-file line to a person. A line that is itself an IID is taken
/// as always. Any other line of two or more fields is read as PLINK's
/// `FID IID ...`, the layout `plink2 --keep` files use, and matched by its IID.
fn resolve_keep_line<'a>(
    line: &'a str,
    iid_to_original_idx: &AHashMap<&str, u32>,
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

/// Every IID of the filesets' shared `.fam`, in file order. Filesets must agree on
/// the people, and IIDs must be unique.
fn index_people(fileset_paths: &[FilesetPaths]) -> Result<Vec<String>, PrepError> {
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
            {
                // Borrowed keys: checking uniqueness copies no IID.
                let mut seen_iids: AHashSet<&str> = AHashSet::with_capacity(iids.len());
                for (idx, iid) in iids.iter().enumerate() {
                    if !seen_iids.insert(iid.as_str()) {
                        return Err(PrepError::Parse(format!(
                            "Duplicate IID '{}' in FAM file '{}'. gnomon requires unique output IIDs.",
                            iid,
                            fileset.fam.display()
                        )));
                    }
                    u32::try_from(idx).map_err(|_| {
                        PrepError::Invariant(format!(
                            "FAM index {idx} exceeds u32::MAX while building lookup."
                        ))
                    })?;
                }
            }
            canonical_path = Some(fileset.fam.clone());
            canonical_iids = Some(iids);
        }
    }

    canonical_iids
        .ok_or_else(|| PrepError::Parse("No individuals found in provided .fam files.".to_string()))
}

fn read_fam_file(path: &Path) -> Result<Vec<String>, PrepError> {
    if let Some(parsed) = parse::parse_local_fam(path) {
        return parsed;
    }
    stream_fam_file(path)
}

fn stream_fam_file(path: &Path) -> Result<Vec<String>, PrepError> {
    let mut source =
        open_plink_text_source(path).map_err(|e| map_pipeline_error(e, path.to_path_buf()))?;
    let mut iids = Vec::new();
    let mut line_number = 0u64;

    while let Some(line) = source
        .next_line()
        .map_err(|e| map_pipeline_error(e, path.to_path_buf()))?
    {
        line_number += 1;
        if let Some(iid) = parse::fam_row_iid(line, line_number, path)? {
            iids.push(iid.to_string());
        }
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
            PrepError::Blocks(s) => write!(f, "--blocks: {s}"),
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
