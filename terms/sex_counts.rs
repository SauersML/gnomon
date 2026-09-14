//! Sex-evidence counts taken straight from packed PLINK 1 genotype rows.
//!
//! Sex inference needs eight integers per sample: valid and heterozygous calls on
//! the autosomes, the X PAR and the X non-PAR, and valid calls on the Y PAR and
//! non-PAR. A `.bed` row stores a missing call as code `0b01` and a heterozygous
//! call as `0b10`, so both counts can be read off the packed bytes, without
//! decoding genotypes to dosages or feeding loci to `SexInferenceAccumulator`
//! one sample at a time.
//!
//! Only the selected rows are read: from the memory map for a local file, and in
//! batches of ranged reads for any other byte source. The kernel expands each
//! packed byte into one flag byte per sample and adds the flags into byte-wide
//! accumulators, which are folded into 64-bit totals every 255 rows. Samples are
//! split into contiguous byte ranges that rayon counts in parallel, each range
//! with its own accumulators, so no counter is shared between threads.

use std::ops::Range;

use infer_sex::{
    AlgorithmConstants, Chromosome, DecisionThresholds, EvidenceReport, InferenceConfig,
    InferenceError, InferenceResult, InferredSex,
};
#[cfg(unix)]
use memmap2::{Advice, UncheckedAdvice};
use rayon::prelude::*;

use crate::pipeline_error::PipelineError;
use crate::shared::files::BedSource;

/// Bytes before the first variant row of a `.bed` file (magic number and mode).
const BED_HEADER_LEN: usize = 3;

/// PLINK 1 genotype codes that the counts distinguish.
const MISSING_CODE: u8 = 0b01;
const HET_CODE: u8 = 0b10;

/// Upper bound on the bytes fetched per batch from a source without a memory map.
const READ_BATCH_BYTES: usize = 64 << 20;

/// Rows between folds of the byte accumulators into the 64-bit totals. Each row
/// adds at most one to a sample's byte, so 255 rows cannot overflow it.
const ROWS_PER_FOLD: u32 = u8::MAX as u32;

/// Packed bytes per parallel sample range, before rounding. Small enough that a
/// range's accumulators stay in a core's cache.
const TARGET_RANGE_BYTES: usize = 16 << 10;

/// The counter a locus feeds in `SexInferenceAccumulator::process_variant`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LocusClass {
    Autosome,
    XPar,
    XNonPar,
    YPar,
    YNonPar,
}

impl LocusClass {
    const ALL: [LocusClass; 5] = [
        LocusClass::Autosome,
        LocusClass::XPar,
        LocusClass::XNonPar,
        LocusClass::YPar,
        LocusClass::YNonPar,
    ];

    /// The class `process_variant` files this locus under, or `None` for an X or
    /// Y locus outside both the PAR and non-PAR intervals, which it ignores.
    pub(super) fn of(
        constants: &AlgorithmConstants,
        chrom: Chromosome,
        position: u64,
    ) -> Option<Self> {
        match chrom {
            Chromosome::Autosome => Some(Self::Autosome),
            Chromosome::X if constants.is_in_x_par(position) => Some(Self::XPar),
            Chromosome::X if constants.is_in_x_non_par(position) => Some(Self::XNonPar),
            Chromosome::Y if constants.is_in_y_par(position) => Some(Self::YPar),
            Chromosome::Y if constants.is_in_y_non_par(position) => Some(Self::YNonPar),
            Chromosome::X | Chromosome::Y => None,
        }
    }
}

/// The counters `SexInferenceAccumulator` keeps for one sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct EvidenceCounts {
    pub(super) auto_valid: u64,
    pub(super) auto_het: u64,
    pub(super) x_par_valid: u64,
    pub(super) x_par_het: u64,
    pub(super) x_non_par_valid: u64,
    pub(super) x_non_par_het: u64,
    pub(super) y_par_valid: u64,
    pub(super) y_non_par_valid: u64,
}

/// `SexInferenceAccumulator::finish` from infer_sex 0.1.2, applied to counts
/// gathered in bulk.
///
/// The accumulator can only be fed one locus at a time, which costs one call per
/// sample per locus. This mirrors `finish` operation for operation, so every
/// metric is bit-identical to the crate's;
/// `finish_counts_matches_the_infer_sex_accumulator` checks that against the
/// crate itself.
pub(super) fn finish_counts(
    config: &InferenceConfig,
    counts: &EvidenceCounts,
) -> Result<InferenceResult, InferenceError> {
    let constants = AlgorithmConstants::from_build(config.build);
    let platform = config.platform;
    if platform.n_attempted_autosomes == 0 {
        return Err(InferenceError::InvalidPlatformCounts(
            "n_attempted_autosomes must be > 0",
        ));
    }
    if counts.auto_valid > platform.n_attempted_autosomes {
        return Err(InferenceError::ObservedExceedsAttempted(
            "observed autosomal variants exceed platform definition",
        ));
    }
    if counts.y_non_par_valid > platform.n_attempted_y_nonpar {
        return Err(InferenceError::ObservedExceedsAttempted(
            "observed Y non-PAR variants exceed platform definition",
        ));
    }

    let mut report = EvidenceReport {
        auto_valid_count: counts.auto_valid,
        auto_het_count: counts.auto_het,
        x_non_par_valid_count: counts.x_non_par_valid,
        x_non_par_het_count: counts.x_non_par_het,
        x_par_valid_count: counts.x_par_valid,
        x_par_het_count: counts.x_par_het,
        y_non_par_valid_count: counts.y_non_par_valid,
        y_par_valid_count: counts.y_par_valid,
        ..EvidenceReport::default()
    };

    let total_sex_observed =
        counts.x_non_par_valid + counts.x_par_valid + counts.y_non_par_valid + counts.y_par_valid;
    if total_sex_observed == 0 {
        return Ok(InferenceResult {
            final_call: InferredSex::Indeterminate,
            report,
        });
    }

    let y_density = if counts.auto_valid == 0 || platform.n_attempted_y_nonpar == 0 {
        None
    } else {
        let auto_rate =
            (counts.auto_valid as f64 + constants.epsilon) / platform.n_attempted_autosomes as f64;
        let y_rate = counts.y_non_par_valid as f64 / platform.n_attempted_y_nonpar as f64;
        Some(y_rate / auto_rate)
    };

    let x_auto_ratio = if counts.auto_valid == 0 || counts.x_non_par_valid == 0 {
        None
    } else {
        let auto_het_rate = counts.auto_het as f64 / (counts.auto_valid as f64 + constants.epsilon);
        let x_het_rate =
            counts.x_non_par_het as f64 / (counts.x_non_par_valid as f64 + constants.epsilon);
        Some(x_het_rate / (auto_het_rate + constants.epsilon))
    };

    let composite = match (y_density, x_auto_ratio) {
        (Some(y), Some(x)) => Some(y / (x + constants.epsilon)),
        _ => None,
    };

    report.y_genome_density = y_density;
    report.x_autosome_het_ratio = x_auto_ratio;
    report.composite_sex_index = composite;

    let thresholds = config.thresholds.unwrap_or_default();
    let final_call = classify_sex(y_density, x_auto_ratio, thresholds);

    Ok(InferenceResult { final_call, report })
}

/// infer_sex 0.1.2's private `classify_sex`.
fn classify_sex(
    y_density: Option<f64>,
    x_auto_ratio: Option<f64>,
    thresholds: DecisionThresholds,
) -> InferredSex {
    let y = y_density.unwrap_or(0.0);
    let x = x_auto_ratio.unwrap_or(0.0);
    let calculated_threshold = (thresholds.slope * x) + thresholds.intercept;

    if y > calculated_threshold {
        InferredSex::Male
    } else {
        InferredSex::Female
    }
}

/// The variant rows of a `.bed` payload.
pub(super) struct BedRows<'a> {
    source: &'a BedSource,
    bytes_per_variant: usize,
    n_variants: usize,
    n_samples: usize,
}

impl<'a> BedRows<'a> {
    pub(super) fn new(
        source: &'a BedSource,
        bytes_per_variant: usize,
        n_variants: usize,
        n_samples: usize,
    ) -> Self {
        Self {
            source,
            bytes_per_variant,
            n_variants,
            n_samples,
        }
    }

    /// Calls `f` with the rows at `indices`, in order, a batch at a time. A memory
    /// map is sliced in one batch when the selected rows fit in available memory,
    /// and in batches sized from that headroom when they do not; any other source
    /// is read in batches of at most `batch_bytes`, one ranged read per run of
    /// consecutive rows.
    fn for_each_batch(
        &self,
        indices: &[usize],
        batch_bytes: usize,
        available_bytes: u64,
        mut f: impl FnMut(&[&[u8]]),
    ) -> Result<(), PipelineError> {
        let row_len = self.bytes_per_variant;
        if let Some(payload) = self
            .source
            .mmap_slice(BED_HEADER_LEN, row_len * self.n_variants)
        {
            #[cfg(unix)]
            let map = self.source.mmap();
            let batch_rows = mapped_batch_rows(indices.len(), row_len, available_bytes);
            for batch in indices.chunks(batch_rows) {
                #[cfg(unix)]
                let span = mapped_span(batch, row_len);
                // Paging a selection larger than memory in through every counting
                // thread at once thrashes: read each batch ahead, and drop its pages
                // from this mapping once it is counted, so the resident set stays
                // near one batch.
                #[cfg(unix)]
                if let (Some(map), true) = (&map, batch.len() < indices.len()) {
                    let _ = map.advise_range(Advice::WillNeed, span.start, span.len());
                }
                let rows: Vec<&[u8]> = batch
                    .iter()
                    .map(|&index| &payload[index * row_len..(index + 1) * row_len])
                    .collect();
                f(&rows);
                #[cfg(unix)]
                if let (Some(map), true) = (&map, batch.len() < indices.len()) {
                    // SAFETY: the map is a read-only shared mapping of the `.bed`, and
                    // `rows` has been dropped. Dropping these pages only discards this
                    // process's view; the next access faults them back in from the
                    // file, with the same bytes.
                    drop(rows);
                    let _ = unsafe {
                        map.unchecked_advise_range(
                            UncheckedAdvice::DontNeed,
                            span.start,
                            span.len(),
                        )
                    };
                }
            }
            return Ok(());
        }

        let mut buffer = Vec::new();
        for batch in indices.chunks((batch_bytes / row_len).max(1)) {
            buffer.resize(batch.len() * row_len, 0);
            let mut start = 0;
            while start < batch.len() {
                let mut end = start + 1;
                while end < batch.len() && batch[end] == batch[end - 1] + 1 {
                    end += 1;
                }
                let offset = (BED_HEADER_LEN + batch[start] * row_len) as u64;
                self.source
                    .read_at(offset, &mut buffer[start * row_len..end * row_len])?;
                start = end;
            }
            let rows: Vec<&[u8]> = buffer.chunks_exact(row_len).collect();
            f(&rows);
        }
        Ok(())
    }
}

/// Rows per batch when slicing a memory map. Every selected row goes in one batch
/// while they take at most a quarter of the memory available to this process;
/// otherwise a batch takes that quarter, and at least one row.
fn mapped_batch_rows(n_rows: usize, row_len: usize, available_bytes: u64) -> usize {
    let budget = usize::try_from(available_bytes / 4).unwrap_or(usize::MAX);
    if n_rows.saturating_mul(row_len) <= budget {
        n_rows.max(1)
    } else {
        (budget / row_len.max(1)).max(1)
    }
}

/// The byte range of the `.bed` file covering `batch`, whose indices ascend.
fn mapped_span(batch: &[usize], row_len: usize) -> Range<usize> {
    match (batch.first(), batch.last()) {
        (Some(&first), Some(&last)) => {
            BED_HEADER_LEN + first * row_len..BED_HEADER_LEN + (last + 1) * row_len
        }
        _ => BED_HEADER_LEN..BED_HEADER_LEN,
    }
}

/// Counts sex evidence for every sample from the selected rows of a `.bed` payload.
///
/// `loci` pairs each selected row index with the counter it feeds; rows that feed
/// no counter are left out. `progress` receives the number of rows counted so far.
pub(super) fn count_evidence(
    rows: &BedRows<'_>,
    loci: &[(usize, LocusClass)],
    progress: impl FnMut(usize),
) -> Result<Vec<EvidenceCounts>, PipelineError> {
    let (_, available_bytes) = crate::memory::memory_bytes();
    count_evidence_batched(rows, loci, READ_BATCH_BYTES, available_bytes, progress)
}

/// [`count_evidence`] with the read batch size, and the memory available for
/// slicing a map, given explicitly.
fn count_evidence_batched(
    rows: &BedRows<'_>,
    loci: &[(usize, LocusClass)],
    batch_bytes: usize,
    available_bytes: u64,
    mut progress: impl FnMut(usize),
) -> Result<Vec<EvidenceCounts>, PipelineError> {
    let mut evidence = vec![EvidenceCounts::default(); rows.n_samples];
    let mut counted = 0;
    for class in LocusClass::ALL {
        let indices: Vec<usize> = loci
            .iter()
            .filter(|(_, locus_class)| *locus_class == class)
            .map(|&(index, _)| index)
            .collect();
        if indices.is_empty() {
            continue;
        }
        let calls = count_calls(rows, &indices, batch_bytes, available_bytes)?;
        let n_rows = indices.len() as u64;
        for ((sample, &missing), &het) in evidence.iter_mut().zip(&calls.missing).zip(&calls.het) {
            let valid = n_rows - missing;
            match class {
                LocusClass::Autosome => {
                    sample.auto_valid = valid;
                    sample.auto_het = het;
                }
                LocusClass::XPar => {
                    sample.x_par_valid = valid;
                    sample.x_par_het = het;
                }
                LocusClass::XNonPar => {
                    sample.x_non_par_valid = valid;
                    sample.x_non_par_het = het;
                }
                // The accumulator keeps no heterozygous count on Y.
                LocusClass::YPar => sample.y_par_valid = valid,
                LocusClass::YNonPar => sample.y_non_par_valid = valid,
            }
        }
        counted += indices.len();
        progress(counted);
    }
    Ok(evidence)
}

/// Per-sample missing and heterozygous call counts over a set of rows.
struct CallCounts {
    missing: Vec<u64>,
    het: Vec<u64>,
}

fn count_calls(
    rows: &BedRows<'_>,
    indices: &[usize],
    batch_bytes: usize,
    available_bytes: u64,
) -> Result<CallCounts, PipelineError> {
    let kernel = Kernel::detect();
    let mut counters: Vec<RangeCounter> = sample_ranges(
        rows.n_samples,
        rows.bytes_per_variant,
        rayon::current_num_threads(),
    )
    .into_iter()
    .map(|(bytes, n_samples)| RangeCounter::new(bytes, n_samples))
    .collect();

    rows.for_each_batch(indices, batch_bytes, available_bytes, |batch| {
        counters
            .par_iter_mut()
            .for_each(|counter| counter.add_rows(kernel, batch));
    })?;
    counters.par_iter_mut().for_each(RangeCounter::fold);

    let mut missing = Vec::with_capacity(rows.n_samples);
    let mut het = Vec::with_capacity(rows.n_samples);
    for counter in counters {
        missing.extend_from_slice(&counter.missing);
        het.extend_from_slice(&counter.het);
    }
    Ok(CallCounts { missing, het })
}

/// Splits a row's packed bytes into contiguous ranges, returning each range with
/// the number of samples it holds. Ranges are aligned to eight bytes, so the
/// vector kernel covers every range but the last in whole words, and there are at
/// least `parts` of them.
fn sample_ranges(
    n_samples: usize,
    bytes_per_variant: usize,
    parts: usize,
) -> Vec<(Range<usize>, usize)> {
    let parts = parts
        .max(bytes_per_variant.div_ceil(TARGET_RANGE_BYTES))
        .max(1);
    let range_len = bytes_per_variant.div_ceil(parts).next_multiple_of(8);
    (0..bytes_per_variant)
        .step_by(range_len)
        .map(|start| {
            let end = (start + range_len).min(bytes_per_variant);
            let samples = (4 * end).min(n_samples) - 4 * start;
            (start..end, samples)
        })
        .collect()
}

/// Missing and heterozygous call counts for the samples in one byte range.
struct RangeCounter {
    bytes: Range<usize>,
    pending_rows: u32,
    /// Four byte accumulators per packed byte, one per sample slot.
    pending_missing: Vec<u8>,
    pending_het: Vec<u8>,
    missing: Vec<u64>,
    het: Vec<u64>,
}

impl RangeCounter {
    fn new(bytes: Range<usize>, n_samples: usize) -> Self {
        let slots = 4 * bytes.len();
        Self {
            bytes,
            pending_rows: 0,
            pending_missing: vec![0; slots],
            pending_het: vec![0; slots],
            missing: vec![0; n_samples],
            het: vec![0; n_samples],
        }
    }

    fn add_rows(&mut self, kernel: Kernel, rows: &[&[u8]]) {
        for row in rows {
            kernel.add_row(
                &row[self.bytes.clone()],
                &mut self.pending_missing,
                &mut self.pending_het,
            );
            self.pending_rows += 1;
            if self.pending_rows == ROWS_PER_FOLD {
                self.fold();
            }
        }
    }

    /// Moves the byte accumulators into the totals. Accumulators past the last
    /// sample count the padding codes of the final byte and are discarded.
    fn fold(&mut self) {
        for (total, &pending) in self.missing.iter_mut().zip(&self.pending_missing) {
            *total += u64::from(pending);
        }
        for (total, &pending) in self.het.iter_mut().zip(&self.pending_het) {
            *total += u64::from(pending);
        }
        self.pending_missing.fill(0);
        self.pending_het.fill(0);
        self.pending_rows = 0;
    }
}

/// For each packed byte, a little-endian `u32` with byte `k` set to 1 when the
/// genotype in slot `k` equals `code`.
const fn slot_flags(code: u8) -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut slot = 0;
        while slot < 4 {
            if (byte >> (2 * slot)) & 0b11 == code as usize {
                table[byte] |= 1 << (8 * slot);
            }
            slot += 1;
        }
        byte += 1;
    }
    table
}

static MISSING_FLAGS: [u32; 256] = slot_flags(MISSING_CODE);
static HET_FLAGS: [u32; 256] = slot_flags(HET_CODE);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    /// Lookup tables.
    Scalar,
    /// Flags formed from the code bits, without tables.
    Bits,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

impl Kernel {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
        }
        // LLVM vectorizes the table-free loop to NEON, which every aarch64 CPU
        // has, and cannot vectorize the table gathers. On x86 the table loop is the
        // faster of the two.
        if cfg!(target_arch = "aarch64") {
            return Self::Bits;
        }
        Self::Scalar
    }

    /// Adds one row's missing and heterozygous flags to the byte accumulators,
    /// which must hold four slots per packed byte of `packed` and be at most 254.
    fn add_row(self, packed: &[u8], missing: &mut [u8], het: &mut [u8]) {
        match self {
            Self::Scalar => add_row_scalar(packed, missing, het),
            Self::Bits => add_row_bits(packed, missing, het),
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `detect` returns `Avx2` only when the CPU reports AVX2.
            Self::Avx2 => unsafe { add_row_avx2(packed, missing, het) },
        }
    }
}

fn add_row_scalar(packed: &[u8], missing: &mut [u8], het: &mut [u8]) {
    assert!(missing.len() >= 4 * packed.len() && het.len() >= 4 * packed.len());
    let (missing, _) = missing.as_chunks_mut::<4>();
    let (het, _) = het.as_chunks_mut::<4>();
    for ((&byte, missing), het) in packed.iter().zip(missing).zip(het) {
        // Every accumulator byte is at most 254 before the add, so the u32 sum never
        // carries from one sample's byte into the next.
        *missing = (u32::from_le_bytes(*missing) + MISSING_FLAGS[usize::from(byte)]).to_le_bytes();
        *het = (u32::from_le_bytes(*het) + HET_FLAGS[usize::from(byte)]).to_le_bytes();
    }
}

/// [`add_row_scalar`] without the tables. A slot holds the missing code `01`
/// when its low bit is set and its high bit clear, and the heterozygous code
/// `10` the other way round. With no gathers, LLVM vectorizes the loop.
fn add_row_bits(packed: &[u8], missing: &mut [u8], het: &mut [u8]) {
    assert!(missing.len() >= 4 * packed.len() && het.len() >= 4 * packed.len());
    let (missing, _) = missing.as_chunks_mut::<4>();
    let (het, _) = het.as_chunks_mut::<4>();
    for ((&byte, missing), het) in packed.iter().zip(missing).zip(het) {
        let low = byte & 0x55;
        let high = (byte >> 1) & 0x55;
        let missing_bits = low & !high;
        let het_bits = high & !low;
        missing[0] += missing_bits & 1;
        missing[1] += (missing_bits >> 2) & 1;
        missing[2] += (missing_bits >> 4) & 1;
        missing[3] += (missing_bits >> 6) & 1;
        het[0] += het_bits & 1;
        het[1] += (het_bits >> 2) & 1;
        het[2] += (het_bits >> 4) & 1;
        het[3] += (het_bits >> 6) & 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// # Safety
/// The caller must ensure the CPU supports AVX2; `Kernel::detect` checks it with
/// `std::arch::is_x86_feature_detected!("avx2")`.
unsafe fn add_row_avx2(packed: &[u8], missing: &mut [u8], het: &mut [u8]) {
    use std::arch::x86_64::*;

    assert!(missing.len() >= 4 * packed.len() && het.len() >= 4 * packed.len());
    let words = packed.len() / 8;
    // SAFETY: AVX2 is available (the caller's contract). For `word < words` the
    // eight bytes loaded at `8 * word` lie inside `packed`, and the 32 accumulators
    // read and written at `32 * word` lie inside `missing` and `het`, which hold at
    // least `4 * packed.len()` bytes (asserted above). The loads and stores are
    // unaligned.
    unsafe {
        // Each loaded byte repeated four times, once per sample slot.
        let repeat = _mm256_setr_epi8(
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3,
            3, 3, 3,
        );
        // Copy k keeps slot k's two bits: bytes 0x03, 0x0c, 0x30, 0xc0.
        let slot_mask = _mm256_set1_epi32(0xc030_0c03_u32 as i32);
        let missing_codes = _mm256_set1_epi32(0x4010_0401);
        let het_codes = _mm256_set1_epi32(0x8020_0802_u32 as i32);
        for word in 0..words {
            let bytes = _mm_loadl_epi64(packed.as_ptr().add(8 * word).cast());
            let lanes =
                _mm256_inserti128_si256(_mm256_castsi128_si256(bytes), _mm_srli_si128(bytes, 4), 1);
            let codes = _mm256_and_si256(_mm256_shuffle_epi8(lanes, repeat), slot_mask);
            // A matching slot compares to 0xff, so subtracting it adds one.
            let pending = missing.as_mut_ptr().add(32 * word).cast::<__m256i>();
            _mm256_storeu_si256(
                pending,
                _mm256_sub_epi8(
                    _mm256_loadu_si256(pending),
                    _mm256_cmpeq_epi8(codes, missing_codes),
                ),
            );
            let pending = het.as_mut_ptr().add(32 * word).cast::<__m256i>();
            _mm256_storeu_si256(
                pending,
                _mm256_sub_epi8(
                    _mm256_loadu_si256(pending),
                    _mm256_cmpeq_epi8(codes, het_codes),
                ),
            );
        }
    }
    let done = 8 * words;
    add_row_scalar(
        &packed[done..],
        &mut missing[4 * done..],
        &mut het[4 * done..],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::files::ByteRangeSource;
    use infer_sex::{GenomeBuild, PlatformDefinition, SexInferenceAccumulator, VariantInfo};
    use std::sync::Arc;

    /// xorshift64, deterministic across platforms.
    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    /// Random packed rows, padding slots included, with every code frequent.
    fn random_rows(rng: &mut TestRng, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.next() as u8).collect()
    }

    struct MemorySource(Vec<u8>);

    impl ByteRangeSource for MemorySource {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
            let start = offset as usize;
            dst.copy_from_slice(&self.0[start..start + dst.len()]);
            Ok(())
        }
    }

    fn code(row: &[u8], sample: usize) -> u8 {
        (row[sample / 4] >> (2 * (sample % 4))) & 0b11
    }

    fn naive_evidence(
        payload: &[u8],
        row_len: usize,
        n_samples: usize,
        loci: &[(usize, LocusClass)],
    ) -> Vec<EvidenceCounts> {
        let mut evidence = vec![EvidenceCounts::default(); n_samples];
        for &(index, class) in loci {
            let row = &payload[index * row_len..(index + 1) * row_len];
            for (sample, counts) in evidence.iter_mut().enumerate() {
                let valid = u64::from(code(row, sample) != MISSING_CODE);
                let het = u64::from(code(row, sample) == HET_CODE);
                match class {
                    LocusClass::Autosome => {
                        counts.auto_valid += valid;
                        counts.auto_het += het;
                    }
                    LocusClass::XPar => {
                        counts.x_par_valid += valid;
                        counts.x_par_het += het;
                    }
                    LocusClass::XNonPar => {
                        counts.x_non_par_valid += valid;
                        counts.x_non_par_het += het;
                    }
                    LocusClass::YPar => counts.y_par_valid += valid,
                    LocusClass::YNonPar => counts.y_non_par_valid += valid,
                }
            }
        }
        evidence
    }

    fn kernels() -> Vec<Kernel> {
        let mut kernels = vec![Kernel::Scalar, Kernel::Bits];
        if !kernels.contains(&Kernel::detect()) {
            kernels.push(Kernel::detect());
        }
        kernels
    }

    #[test]
    fn slot_flags_mark_each_sample_code() {
        // Slots hold codes 01, 10, 11, 00 from the low bits up.
        let byte = 0b00_11_10_01u8;
        assert_eq!(MISSING_FLAGS[byte as usize].to_le_bytes(), [1, 0, 0, 0]);
        assert_eq!(HET_FLAGS[byte as usize].to_le_bytes(), [0, 1, 0, 0]);
        assert_eq!(MISSING_FLAGS[0x55], 0x0101_0101);
        assert_eq!(HET_FLAGS[0xaa], 0x0101_0101);
        assert_eq!(MISSING_FLAGS[0xff] | HET_FLAGS[0xff] | HET_FLAGS[0x00], 0);
    }

    /// A selection that fits in a quarter of available memory is one batch;
    /// a larger one is cut to that quarter, and never below one row.
    #[test]
    fn mapped_batches_are_sized_from_available_memory() {
        assert_eq!(mapped_batch_rows(55_000, 12_800, 64 << 30), 55_000);
        assert_eq!(
            mapped_batch_rows(55_000, 100_000, 4 << 30),
            (1 << 30) / 100_000
        );
        assert_eq!(mapped_batch_rows(55_000, 100_000, 0), 1);
        assert_eq!(mapped_batch_rows(10, 1 << 40, 4 << 30), 1);
        assert_eq!(mapped_batch_rows(0, 800, 4 << 30), 1);
        assert_eq!(
            mapped_span(&[3, 4, 9], 25),
            BED_HEADER_LEN + 75..BED_HEADER_LEN + 250
        );
        assert_eq!(mapped_span(&[], 25), BED_HEADER_LEN..BED_HEADER_LEN);
    }

    #[test]
    fn kernels_accumulate_the_packed_codes_of_every_slot() {
        let mut rng = TestRng(0x2545_f491_4f6c_dd1d);
        for packed_len in [1usize, 7, 8, 9, 15, 16, 17, 64, 250, 1001] {
            let rows = 254;
            let payload = random_rows(&mut rng, rows * packed_len);
            let mut expected_missing = vec![0u8; 4 * packed_len];
            let mut expected_het = vec![0u8; 4 * packed_len];
            for row in payload.chunks_exact(packed_len) {
                for slot in 0..4 * packed_len {
                    expected_missing[slot] += u8::from(code(row, slot) == MISSING_CODE);
                    expected_het[slot] += u8::from(code(row, slot) == HET_CODE);
                }
            }
            for kernel in kernels() {
                let mut missing = vec![0u8; 4 * packed_len];
                let mut het = vec![0u8; 4 * packed_len];
                for row in payload.chunks_exact(packed_len) {
                    kernel.add_row(row, &mut missing, &mut het);
                }
                assert_eq!(
                    missing, expected_missing,
                    "{kernel:?} missing, {packed_len} bytes"
                );
                assert_eq!(het, expected_het, "{kernel:?} het, {packed_len} bytes");
            }
        }
    }

    #[test]
    fn sample_ranges_cover_every_sample_once_in_word_aligned_ranges() {
        for n_samples in [1usize, 3, 4, 5, 31, 32, 33, 3197, 51_201, 400_001] {
            let row_len = n_samples.div_ceil(4);
            for parts in [1usize, 2, 7, 8, 64] {
                let ranges = sample_ranges(n_samples, row_len, parts);
                let mut next_byte = 0;
                let mut samples = 0;
                for (index, (bytes, range_samples)) in ranges.iter().enumerate() {
                    assert_eq!(bytes.start, next_byte);
                    if index + 1 < ranges.len() {
                        assert_eq!(bytes.len() % 8, 0);
                    }
                    assert!(*range_samples > 0 && *range_samples <= 4 * bytes.len());
                    next_byte = bytes.end;
                    samples += range_samples;
                }
                assert_eq!(next_byte, row_len);
                assert_eq!(samples, n_samples);
            }
        }
    }

    #[test]
    fn evidence_matches_naive_decoding_for_any_batching() {
        let mut rng = TestRng(0x9e37_79b9_7f4a_7c15);
        for n_samples in [1usize, 3, 7, 64, 65, 3197] {
            let row_len = n_samples.div_ceil(4);
            let n_variants = 900;
            let payload = random_rows(&mut rng, n_variants * row_len);
            // Runs, gaps, a duplicated row and more than 255 rows in one class, so the
            // fold, the run coalescing and the batch boundaries are all crossed.
            let mut loci = Vec::new();
            for index in (0..n_variants).step_by(3) {
                loci.push((index, LocusClass::Autosome));
            }
            for index in 100..700 {
                loci.push((index, LocusClass::XNonPar));
            }
            loci.push((350, LocusClass::XNonPar));
            for index in 700..760 {
                loci.push((index, LocusClass::XPar));
            }
            for index in 760..840 {
                loci.push((index, LocusClass::YNonPar));
            }
            loci.push((899, LocusClass::YPar));
            loci.sort_by_key(|&(index, _)| index);
            let expected = naive_evidence(&payload, row_len, n_samples, &loci);

            let mut file = vec![0x6c, 0x1b, 0x01];
            file.extend_from_slice(&payload);
            let source = BedSource::from_byte_source(Arc::new(MemorySource(file)));
            let rows = BedRows::new(&source, row_len, n_variants, n_samples);
            for batch_bytes in [1, 3 * row_len, READ_BATCH_BYTES] {
                let mut reported = Vec::new();
                let evidence =
                    count_evidence_batched(&rows, &loci, batch_bytes, u64::MAX, |done| {
                        reported.push(done)
                    })
                    .unwrap();
                assert_eq!(
                    evidence, expected,
                    "{n_samples} samples, batch {batch_bytes}"
                );
                assert_eq!(reported.last().copied(), Some(loci.len()));
            }
        }
    }

    /// A mapped `.bed` counted with every selected row in one batch, and in the
    /// small batches too little available memory forces, must give the naive
    /// decoding's counts.
    #[test]
    fn mapped_rows_count_the_same_in_any_number_of_batches() {
        let mut rng = TestRng(0x3c6e_f372_fe94_f82b);
        let dir = tempfile::tempdir().unwrap();
        for n_samples in [1usize, 7, 64, 3197] {
            let row_len = n_samples.div_ceil(4);
            let n_variants = 700;
            let payload = random_rows(&mut rng, n_variants * row_len);
            let path = dir.path().join(format!("mapped_{n_samples}.bed"));
            let mut file = vec![0x6c, 0x1b, 0x01];
            file.extend_from_slice(&payload);
            std::fs::write(&path, &file).unwrap();
            let source = crate::shared::files::open_bed_source(&path, None).unwrap();
            assert!(source.mmap().is_some(), "a local .bed is mapped");
            let rows = BedRows::new(&source, row_len, n_variants, n_samples);

            let mut loci: Vec<(usize, LocusClass)> = (0..n_variants)
                .step_by(2)
                .map(|index| (index, LocusClass::Autosome))
                .collect();
            loci.extend((301..600).map(|index| (index, LocusClass::XNonPar)));
            loci.extend((650..680).map(|index| (index, LocusClass::YNonPar)));
            loci.sort_by_key(|&(index, _)| index);
            loci.dedup_by_key(|&mut (index, _)| index);
            let expected = naive_evidence(&payload, row_len, n_samples, &loci);

            // 4 bytes available is a one-row batch; 4 * 25 rows a 25-row one.
            for available_bytes in [u64::MAX, 4 * 25 * row_len as u64, 4] {
                let evidence =
                    count_evidence_batched(&rows, &loci, READ_BATCH_BYTES, available_bytes, |_| {})
                        .unwrap();
                assert_eq!(
                    evidence, expected,
                    "{n_samples} samples, {available_bytes} bytes available"
                );
            }
        }
    }

    fn feed(
        accumulator: &mut SexInferenceAccumulator,
        chrom: Chromosome,
        first_position: u64,
        valid: u64,
        het: u64,
    ) {
        for offset in 0..valid {
            accumulator.process_variant(&VariantInfo {
                chrom,
                pos: first_position + offset,
                is_heterozygous: offset < het,
            });
        }
    }

    fn metric_bits(result: &InferenceResult) -> [Option<u64>; 3] {
        [
            result.report.y_genome_density.map(f64::to_bits),
            result.report.x_autosome_het_ratio.map(f64::to_bits),
            result.report.composite_sex_index.map(f64::to_bits),
        ]
    }

    #[test]
    fn finish_counts_matches_the_infer_sex_accumulator() {
        let mut rng = TestRng(0xd1b5_4a32_d192_ed03);
        for case in 0..3000u64 {
            let build = if case % 2 == 0 {
                GenomeBuild::Build37
            } else {
                GenomeBuild::Build38
            };
            let scale = if case % 10 == 0 { 20_000 } else { 60 };
            let mut counts = EvidenceCounts {
                auto_valid: rng.below(scale + 1),
                x_par_valid: rng.below(scale + 1),
                x_non_par_valid: rng.below(scale + 1),
                y_par_valid: rng.below(scale / 4 + 1),
                y_non_par_valid: rng.below(scale / 2 + 1),
                ..EvidenceCounts::default()
            };
            // Zero whole regions so every early return and `None` branch is reached.
            match case % 6 {
                0 => counts.auto_valid = 0,
                1 => counts.x_non_par_valid = 0,
                2 => counts.y_non_par_valid = 0,
                3 => {
                    counts.x_par_valid = 0;
                    counts.x_non_par_valid = 0;
                    counts.y_par_valid = 0;
                    counts.y_non_par_valid = 0;
                }
                _ => {}
            }
            counts.auto_het = rng.below(counts.auto_valid + 1);
            counts.x_par_het = rng.below(counts.x_par_valid + 1);
            counts.x_non_par_het = rng.below(counts.x_non_par_valid + 1);
            // Mostly consistent platforms, with some too small for the observations.
            let platform = PlatformDefinition {
                n_attempted_autosomes: counts.auto_valid + rng.below(scale + 1)
                    - rng.below(3).min(counts.auto_valid),
                n_attempted_y_nonpar: (counts.y_non_par_valid + rng.below(scale / 2 + 1))
                    .saturating_sub(rng.below(3) * (case % 5)),
            };
            let thresholds = (case % 3 == 0).then(|| DecisionThresholds {
                slope: rng.below(1000) as f64 / 250.0,
                intercept: rng.below(1000) as f64 / 500.0,
            });
            let config = InferenceConfig {
                build,
                platform,
                thresholds,
            };

            let constants = build.algorithm_constants();
            let mut accumulator = SexInferenceAccumulator::new(config);
            feed(
                &mut accumulator,
                Chromosome::Autosome,
                1_000_000,
                counts.auto_valid,
                counts.auto_het,
            );
            feed(
                &mut accumulator,
                Chromosome::X,
                constants.par1_x.0,
                counts.x_par_valid,
                counts.x_par_het,
            );
            feed(
                &mut accumulator,
                Chromosome::X,
                constants.non_par_x.0,
                counts.x_non_par_valid,
                counts.x_non_par_het,
            );
            feed(
                &mut accumulator,
                Chromosome::Y,
                constants.par1_y.0,
                counts.y_par_valid,
                0,
            );
            feed(
                &mut accumulator,
                Chromosome::Y,
                constants.non_par_y.0,
                counts.y_non_par_valid,
                0,
            );
            let expected = accumulator.finish();

            let actual = finish_counts(&config, &counts);
            assert_eq!(actual, expected, "case {case}: {counts:?} {platform:?}");
            if let (Ok(actual), Ok(expected)) = (&actual, &expected) {
                assert_eq!(metric_bits(actual), metric_bits(expected), "case {case}");
            }
        }
    }

    #[test]
    fn locus_classes_follow_the_accumulator_intervals() {
        for build in [GenomeBuild::Build37, GenomeBuild::Build38] {
            let constants = build.algorithm_constants();
            let boundaries = [
                constants.par1_x,
                constants.non_par_x,
                constants.par2_x,
                constants.par1_y,
                constants.non_par_y,
                constants.par2_y,
            ];
            for (start, end) in boundaries {
                for position in [start - 1, start, start + 1, end - 1, end, end + 1] {
                    for chrom in [Chromosome::Autosome, Chromosome::X, Chromosome::Y] {
                        let class = LocusClass::of(&constants, chrom, position);
                        let config = InferenceConfig {
                            build,
                            platform: PlatformDefinition {
                                n_attempted_autosomes: 1,
                                n_attempted_y_nonpar: 1,
                            },
                            thresholds: None,
                        };
                        let mut accumulator = SexInferenceAccumulator::new(config);
                        feed(&mut accumulator, chrom, position, 1, 1);
                        let report = accumulator.finish().unwrap().report;
                        let expected = if report.auto_valid_count == 1 {
                            Some(LocusClass::Autosome)
                        } else if report.x_par_valid_count == 1 {
                            Some(LocusClass::XPar)
                        } else if report.x_non_par_valid_count == 1 {
                            Some(LocusClass::XNonPar)
                        } else if report.y_par_valid_count == 1 {
                            Some(LocusClass::YPar)
                        } else if report.y_non_par_valid_count == 1 {
                            Some(LocusClass::YNonPar)
                        } else {
                            None
                        };
                        assert_eq!(class, expected, "{build:?} {chrom:?} {position}");
                    }
                }
            }
        }
    }
}
