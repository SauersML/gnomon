// ========================================================================================
//
//               Exact score kernels for batches of packed PLINK rows
//
// ========================================================================================
//
// This module contains the synchronous, CPU-bound core of the compute pipeline. It is
// designed to be called from a higher-level orchestrator within a parallel context. Its
// sole responsibility is to add the terms of variant rows to people's integer cells: a
// dense batch through four-variant tables, a sparse variant by walking the calls that are
// not homozygous reference. It performs ZERO scientific logic or reconciliation, and
// neither path can change a cell's value, only how fast it is reached.

use crate::score::cells::ExactPlan;
use crate::score::kernel_exact::{
    People, TableScratch, apply_table_rows, for_each_call, for_each_missing, grow,
};
use crate::score::types::{
    OriginalPersonIndex, PersonSubset, PreparationResult, ReconciledVariantIndex, VariantCsrView,
};
use std::collections::TryReserveError;
use std::error::Error;

/// Where each scored person's call sits in a packed row.
pub struct PersonLayout {
    complete: bool,
    count: usize,
    /// Byte offset and shift of each kept person's call; empty for a complete cohort.
    bytes: Vec<u32>,
    shifts: Vec<u8>,
}

impl PersonLayout {
    pub fn new(prep: &PreparationResult) -> Self {
        match &prep.person_subset {
            PersonSubset::All => Self {
                complete: true,
                count: prep.num_people_to_score,
                bytes: Vec::new(),
                shifts: Vec::new(),
            },
            PersonSubset::Indices(_) => {
                let (bytes, shifts) = prep
                    .output_idx_to_fam_idx
                    .iter()
                    .map(|&OriginalPersonIndex(fam)| (fam / 4, (2 * (fam % 4)) as u8))
                    .unzip();
                Self {
                    complete: false,
                    count: prep.num_people_to_score,
                    bytes,
                    shifts,
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn people(&self) -> People<'_> {
        if self.complete {
            People::All(self.count)
        } else {
            People::Gathered {
                bytes: &self.bytes,
                shifts: &self.shifts,
            }
        }
    }
}

fn check_buffers(
    prep: &PreparationResult,
    cells: &[i64],
    counts: &[u32],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let people = prep.num_people_to_score;
    if cells.len() != people * prep.exact().stride() || counts.len() != people * prep.score_names.len()
    {
        return Err(Box::from(format!(
            "Mismatched score buffers: {} cells and {} counts for {people} people",
            cells.len(),
            counts.len()
        )));
    }
    Ok(())
}

/// Reusable per-thread scratch for [`run_dense_batch`].
#[derive(Default)]
pub struct DenseScratch {
    terms: Vec<i64>,
    tables: TableScratch,
}

/// Adds one batch of dense packed rows (each a complete `.bed` row) to every scored person's
/// cells and missing counts. `cells` holds people × stride lanes and `counts` people × scores.
pub fn run_dense_batch(
    data: &[u8],
    rows: &[ReconciledVariantIndex],
    prep: &PreparationResult,
    layout: &PersonLayout,
    scratch: &mut DenseScratch,
    cells: &mut [i64],
    counts: &mut [u32],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    check_buffers(prep, cells, counts)?;
    let exact = prep.exact();
    let stride = exact.stride();
    let num_scores = prep.score_names.len();
    let row_bytes = prep.bytes_per_variant as usize;
    if data.len() < rows.len() * row_bytes {
        return Err(Box::from("Dense batch holds fewer bytes than its rows need."));
    }
    // Term rows for PLINK codes [00, 01, 10, 11], padded to whole four-variant groups. The buffer
    // grows on a cold path and is zeroed in place, so no resize sits in the batch's code.
    let terms_len = rows.len().div_ceil(4) * 4 * 4 * stride;
    if scratch.terms.len() < terms_len {
        grow_terms(&mut scratch.terms, terms_len)?;
    }
    let terms = &mut scratch.terms[..terms_len];
    terms.fill(0);
    for (r, &index) in rows.iter().enumerate() {
        let view = prep.variant_csr_view(index);
        add_code_rows(exact, &view, stride, &mut terms[r * 4 * stride..(r + 1) * 4 * stride]);
        for_each_missing(
            &data[r * row_bytes..(r + 1) * row_bytes],
            layout.people(),
            |person| {
                for contribution in view.iter() {
                    if exact.counts_missing(contribution.entry) {
                        counts[person * num_scores + contribution.score_column.0] += 1;
                    }
                }
            },
        );
    }
    apply_table_rows(
        data,
        row_bytes,
        rows.len(),
        terms,
        stride,
        layout.people(),
        &mut scratch.tables,
        cells,
    );
    Ok(())
}

/// Grows the batch's term rows to `len`, off the batch's own code: a refused allocation is the
/// batch's error rather than an abort.
#[cold]
#[inline(never)]
fn grow_terms(terms: &mut Vec<i64>, len: usize) -> Result<(), TryReserveError> {
    terms.try_reserve_exact(len - terms.len())?;
    terms.resize(len, 0);
    Ok(())
}

/// Adds `view`'s entries to one variant's four code rows, `rows` holding 4 × stride lanes that start
/// at zero: what calls 00, 01, 10 and 11 add. Code 00 adds nothing. When every score is one lane,
/// score `s` is lane `s`, so an entry adds its weight to the 10 row and, flipped, twice it to the 01
/// row, and the 11 row is twice the 10 row, as a wrapping sum of doubled weights is.
fn add_code_rows(exact: &ExactPlan, view: &VariantCsrView<'_>, stride: usize, rows: &mut [i64]) {
    let (missing_row, doses) = rows[stride..4 * stride].split_at_mut(stride);
    let (one_dose, two_doses) = doses.split_at_mut(stride);
    if exact.one_lane_per_score() {
        for contribution in view.iter() {
            let score = contribution.score_column.0;
            let (weight, flipped) = exact.narrow_entry(contribution.entry);
            one_dose[score] = one_dose[score].wrapping_add(weight);
            if flipped {
                missing_row[score] = missing_row[score].wrapping_add(weight.wrapping_mul(2));
            }
        }
        for (two, &one) in two_doses.iter_mut().zip(one_dose.iter()) {
            *two = one.wrapping_mul(2);
        }
    } else {
        for contribution in view.iter() {
            let target = exact.entry_target(contribution.entry, contribution.score_column.0);
            let [_, missing, one, two] = exact.entry_terms(contribution.entry);
            exact.add(target, missing, missing_row);
            exact.add(target, one, one_dose);
            exact.add(target, two, two_doses);
        }
    }
}

/// One variant's entries summed per call: for each code 00..11, what the call adds to every lane
/// of a person's cell, and what a missing call adds to each score's missing count. The sums are
/// a table of one variant, so they stay inside the plan's bounds as the dense tables do.
#[derive(Default)]
pub struct VariantTerms {
    table: Vec<i64>,
    missing: Vec<u32>,
    stride: usize,
}

impl VariantTerms {
    /// Loads the terms of `index`'s entries.
    pub fn load(&mut self, prep: &PreparationResult, index: ReconciledVariantIndex) {
        let exact = prep.exact();
        let stride = exact.stride();
        self.stride = stride;
        // Both grow once, on a cold path, and are zeroed in place for every variant.
        let scores = prep.score_names.len();
        grow(&mut self.table, 4 * stride);
        grow(&mut self.missing, scores);
        let table = &mut self.table[..4 * stride];
        table.fill(0);
        let missing = &mut self.missing[..scores];
        missing.fill(0);
        let view = prep.variant_csr_view(index);
        add_code_rows(exact, &view, stride, table);
        for contribution in view.iter() {
            if exact.counts_missing(contribution.entry) {
                missing[contribution.score_column.0] += 1;
            }
        }
    }

    /// Adds one person's call to their cell and missing counts.
    #[inline(always)]
    pub fn apply(&self, code: u8, cell: &mut [i64], counts: &mut [u32]) {
        let code = usize::from(code & 3);
        let terms = &self.table[code * self.stride..(code + 1) * self.stride];
        for (lane, &term) in cell.iter_mut().zip(terms) {
            *lane = lane.wrapping_add(term);
        }
        if code == 1 {
            for (count, &add) in counts.iter_mut().zip(&self.missing) {
                *count += add;
            }
        }
    }
}

/// Adds one sparse variant's row to the scored people by walking the calls that differ from
/// homozygous reference.
pub fn run_variant_major_path(
    variant_data: &[u8],
    prep: &PreparationResult,
    layout: &PersonLayout,
    scratch: &mut VariantTerms,
    cells: &mut [i64],
    counts: &mut [u32],
    reconciled_variant_index: ReconciledVariantIndex,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    check_buffers(prep, cells, counts)?;
    let exact = prep.exact();
    let stride = exact.stride();
    let num_scores = prep.score_names.len();
    scratch.load(prep, reconciled_variant_index);
    for_each_call(variant_data, layout.people(), |person, code| {
        scratch.apply(
            code,
            &mut cells[person * stride..(person + 1) * stride],
            &mut counts[person * num_scores..(person + 1) * num_scores],
        );
    });
    Ok(())
}

/// Classifies non-reference density for the path dispatcher using `popcnt`.
///
/// This function is a key performance enabler. It leverages the `popcnt` (population
/// count) CPU instruction, which is very fast. The decision tree's highest
/// density threshold is 0.0894, so common variants stop scanning as soon as they
/// cross it; only variants that may take the sparse path need an exact result.
#[inline]
pub fn assess_variant_density_for_dispatch(variant_data: &[u8], total_people: usize) -> f32 {
    if total_people == 0 {
        return 0.0;
    }

    const CHUNK_SIZE: usize = std::mem::size_of::<u64>();
    const HIGHEST_DISPATCH_THRESHOLD: f32 = 0.0894;
    let dense_cutoff = (total_people as f32 * HIGHEST_DISPATCH_THRESHOLD) as u64;
    let mut set_bits = 0u64;

    // Process full 8-byte chunks using `chunks_exact` for safety and performance.
    let chunks = variant_data.chunks_exact(CHUNK_SIZE);
    let remainder = chunks.remainder();
    for chunk in chunks {
        // This conversion is safe because chunks_exact guarantees the slice length is CHUNK_SIZE.
        let val = u64::from_ne_bytes(chunk.try_into().unwrap());
        set_bits += u64::from(val.count_ones());
        if set_bits > dense_cutoff {
            return 1.0;
        }
    }

    // Process the remainder byte by byte.
    for &byte in remainder {
        set_bits += u64::from(byte.count_ones());
    }

    // Normalize by the number of people to get a comparable frequency.
    // The homozygous-reference genotype (0b00) has a popcnt of 0. All others
    // have a popcnt > 0. This gives a reliable, self-contained metric.
    set_bits as f32 / total_people as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::cells::ExactPlan;
    use crate::score::types::{BimRowIndex, OutputPersonIndex, PipelineKind};
    use std::path::PathBuf;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    struct Panel {
        prep: PreparationResult,
        data: Vec<u8>,
        /// Entry weight and correction at scale 10^6, independent of the exact plan.
        weights6: Vec<i64>,
        corrections6: Vec<i64>,
        columns: Vec<u32>,
        offsets: Vec<u64>,
        kept: Vec<usize>,
    }

    /// A random panel: six-decimal weights, some entries flipped, some score lines duplicated,
    /// every PLINK code present, and optionally a kept subset in reversed order. `wide` weights
    /// are fifteen-digit decimals at three places, about 10^12, so a score of a few thousand
    /// entries needs two limbs.
    fn panel(rng: &mut Rng, people: usize, scores: usize, rows: usize, keep: bool, wide: bool) -> Panel {
        let (mut weights, mut corrections, mut columns, mut offsets) =
            (Vec::new(), Vec::new(), Vec::new(), vec![0u64]);
        let (mut weights6, mut corrections6) = (Vec::new(), Vec::new());
        for _ in 0..rows {
            let mut row: Vec<u32> = (0..scores as u32).filter(|_| rng.next() % 3 != 0).collect();
            if !row.is_empty() && rng.next() % 4 == 0 {
                let duplicate = row[rng.next() as usize % row.len()];
                row.push(duplicate);
                row.sort_unstable();
            }
            for column in row {
                let micro = if wide {
                    let milli = 500_000_000_000_000 + (rng.next() % 500_000_000_000_000) as i64;
                    if rng.next() % 2 == 0 { milli * 1000 } else { -milli * 1000 }
                } else {
                    (rng.next() % 2_000_001) as i64 - 1_000_000
                };
                let flipped = rng.next() % 3 == 0;
                let weight6 = if flipped { -micro } else { micro };
                weights6.push(weight6);
                corrections6.push(if flipped { -2 * weight6 } else { 0 });
                let weight: f64 = format!("{weight6}e-6").parse().unwrap();
                weights.push(weight);
                corrections.push(if flipped { -2.0 * weight } else { 0.0 });
                columns.push(column);
            }
            offsets.push(columns.len() as u64);
        }
        let names: Vec<String> = (0..scores).map(|s| format!("S{s}")).collect();
        let exact = ExactPlan::new(weights, &corrections, &columns, &offsets, &[], &names)
            .expect("exact plan");
        let row_bytes = people.div_ceil(4);
        let data: Vec<u8> = (0..rows * row_bytes).map(|_| rng.next() as u8).collect();
        let kept: Vec<usize> = if keep {
            (0..people).rev().filter(|p| p % 3 != 1).collect()
        } else {
            (0..people).collect()
        };
        let subset = if keep {
            let mut sorted: Vec<u32> = kept.iter().map(|&p| p as u32).collect();
            sorted.sort_unstable();
            PersonSubset::Indices(sorted)
        } else {
            PersonSubset::All
        };
        let mut fam_to_output = vec![None; people];
        for (output, &fam) in kept.iter().enumerate() {
            fam_to_output[fam] = Some(OutputPersonIndex(output as u32));
        }
        let prep = PreparationResult::new(
            exact,
            columns.clone(),
            offsets.clone(),
            (0..rows as u64).map(BimRowIndex).collect(),
            Vec::new(),
            names,
            vec![rows as u32; scores],
            subset,
            kept.iter().map(|p| format!("P{p}")).collect(),
            kept.len(),
            people,
            rows as u64,
            rows,
            row_bytes as u64,
            fam_to_output,
            kept.iter().map(|&p| OriginalPersonIndex(p as u32)).collect(),
            vec![0; rows],
            Vec::new(),
            Vec::new(),
            0,
            PipelineKind::SingleFile(PathBuf::from("panel.bed")),
        );
        Panel {
            prep,
            data,
            weights6,
            corrections6,
            columns,
            offsets,
            kept,
        }
    }

    /// Correctly rounded sums and missing counts from integers at scale 10^6.
    fn oracle(panel: &Panel) -> (Vec<u64>, Vec<u32>) {
        let scores = panel.prep.score_names.len();
        let row_bytes = panel.prep.bytes_per_variant as usize;
        let people = panel.kept.len();
        let (mut sums6, mut counts) = (vec![0i128; people * scores], vec![0u32; people * scores]);
        for (output, &fam) in panel.kept.iter().enumerate() {
            for row in 0..panel.offsets.len() - 1 {
                let code = (panel.data[row * row_bytes + fam / 4] >> (2 * (fam % 4))) & 3;
                let range = panel.offsets[row] as usize..panel.offsets[row + 1] as usize;
                let mut counted = Vec::new();
                for i in range {
                    let cell = output * scores + panel.columns[i] as usize;
                    match code {
                        1 => {
                            if !counted.contains(&cell) {
                                counted.push(cell);
                                counts[cell] += 1;
                            }
                        }
                        dose => {
                            let dose = [0, 0, 1, 2][dose as usize];
                            sums6[cell] +=
                                i128::from(panel.corrections6[i] + dose * panel.weights6[i]);
                        }
                    }
                }
            }
        }
        let rounded = sums6
            .iter()
            .map(|&v| format!("{v}e-6").parse::<f64>().unwrap().to_bits())
            .collect();
        (rounded, counts)
    }

    #[test]
    fn dense_and_sparse_paths_give_the_correctly_rounded_exact_sums() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        // One dense scratch and one variant scratch for every panel: they only grow, so later, smaller
        // panels run over stale terms, tables and keys from earlier ones, which must not reach a cell.
        let (mut scratch, mut terms) = (DenseScratch::default(), VariantTerms::default());
        for (people, scores, rows, keep, wide) in [
            (5, 1, 3, false, false),
            (64, 3, 7, false, false),
            (100, 7, 9, true, false),
            (33, 70, 5, false, false),
            (257, 16, 256, true, false),
            // Two-limb scores: code rows go through the plan's targets, not one lane a score.
            (40, 2, 12_000, false, true),
        ] {
            let panel = panel(&mut rng, people, scores, rows, keep, wide);
            let prep = &panel.prep;
            assert_eq!(prep.exact().one_lane_per_score(), !wide);
            let (want_sums, want_counts) = oracle(&panel);
            let layout = PersonLayout::new(prep);
            let stride = prep.exact().stride();
            let n = prep.num_people_to_score;
            let row_bytes = prep.bytes_per_variant as usize;
            let indices: Vec<ReconciledVariantIndex> =
                (0..rows as u32).map(ReconciledVariantIndex).collect();

            let (mut dense_cells, mut dense_counts) = (vec![0i64; n * stride], vec![0u32; n * scores]);
            for chunk in (0..rows).step_by(8) {
                let end = (chunk + 8).min(rows);
                run_dense_batch(
                    &panel.data[chunk * row_bytes..end * row_bytes],
                    &indices[chunk..end],
                    prep,
                    &layout,
                    &mut scratch,
                    &mut dense_cells,
                    &mut dense_counts,
                )
                .unwrap();
            }
            let (mut sparse_cells, mut sparse_counts) = (vec![0i64; n * stride], vec![0u32; n * scores]);
            for row in 0..rows {
                run_variant_major_path(
                    &panel.data[row * row_bytes..(row + 1) * row_bytes],
                    prep,
                    &layout,
                    &mut terms,
                    &mut sparse_cells,
                    &mut sparse_counts,
                    indices[row],
                )
                .unwrap();
            }
            for (cells, counts) in [(&dense_cells, &dense_counts), (&sparse_cells, &sparse_counts)] {
                let sums: Vec<u64> = (0..n * scores)
                    .map(|cell| {
                        let person = cell / scores;
                        prep.exact()
                            .sum(cell % scores, &cells[person * stride..(person + 1) * stride])
                            .to_bits()
                    })
                    .collect();
                assert_eq!(sums, want_sums, "people {people} scores {scores} rows {rows} keep {keep}");
                assert_eq!(counts, &want_counts);
            }
        }
    }
}
