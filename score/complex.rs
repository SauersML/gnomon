use crate::pipeline_error::PipelineError;
use crate::score::cells::{ExactPlan, Target};
use crate::score::io::BedSource;
use crate::score::site::{RowMatch, Site, SiteAllele};
use crate::score::types::{
    BimRowIndex, FilesetBoundary, GroupedComplexRule, OutputPersonIndex, PreparationResult,
    ScoreInfo,
};
use ahash::AHashMap;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

/// A read-only resolver for fetching complex variant genotypes.
///
/// This enum is initialized ONCE at the start of the pipeline run. It holds
/// either a single memory map or a collection of them, avoiding the massive
/// performance penalty of re-opening and re-mapping files inside a parallel loop.
pub enum ComplexVariantResolver {
    SingleFile(BedSource),
    MultiFile {
        sources: Vec<BedSource>,
        boundaries: Vec<FilesetBoundary>,
    },
    Spool {
        mmap: Arc<Mmap>,
        offsets: AHashMap<BimRowIndex, u64>,
        bytes_per_spooled_variant: u64,
        dense_map: Arc<Vec<i32>>,
    },
}

impl ComplexVariantResolver {
    pub fn from_single_source(source: BedSource) -> Self {
        Self::SingleFile(source)
    }

    pub fn from_multi_sources(
        sources: Vec<BedSource>,
        boundaries: Vec<FilesetBoundary>,
    ) -> Result<Self, PipelineError> {
        if sources.len() != boundaries.len() {
            return Err(PipelineError::Io(
                "Mismatched number of byte sources and fileset boundaries".to_string(),
            ));
        }
        Ok(Self::MultiFile {
            sources,
            boundaries,
        })
    }

    pub fn from_spool(
        mmap: Arc<Mmap>,
        offsets: AHashMap<BimRowIndex, u64>,
        bytes_per_spooled_variant: u64,
        dense_map: Arc<Vec<i32>>,
    ) -> Self {
        Self::Spool {
            mmap,
            offsets,
            bytes_per_spooled_variant,
            dense_map,
        }
    }

    /// Whether every genotype source is memory-mapped, so rows are borrowed in
    /// place rather than read into buffers.
    fn is_mapped(&self) -> bool {
        match self {
            ComplexVariantResolver::SingleFile(source) => source.mmap().is_some(),
            ComplexVariantResolver::MultiFile { sources, .. } => {
                sources.iter().all(|source| source.mmap().is_some())
            }
            ComplexVariantResolver::Spool { .. } => true,
        }
    }

    /// Finds where one variant's packed row starts. The resolver calls this once
    /// per rule context, so the fileset search and spool lookup never run per genotype.
    fn locate_row(
        &self,
        bytes_per_variant: u64,
        bim_row_index: BimRowIndex,
    ) -> Result<RowLocation, PipelineError> {
        let (source, local_bim_index) = match self {
            ComplexVariantResolver::SingleFile(_) => (0, bim_row_index.0),
            ComplexVariantResolver::MultiFile { boundaries, .. } => {
                let fileset_idx =
                    boundaries.partition_point(|b| b.starting_global_index <= bim_row_index.0) - 1;
                let boundary = &boundaries[fileset_idx];
                (fileset_idx, bim_row_index.0 - boundary.starting_global_index)
            }
            ComplexVariantResolver::Spool { offsets, .. } => {
                let offset = offsets.get(&bim_row_index).copied().ok_or_else(|| {
                    PipelineError::Io(format!(
                        "Missing spool offset for BIM row {} while resolving complex variant.",
                        bim_row_index.0
                    ))
                })?;
                return Ok(RowLocation {
                    bim_row_index,
                    source: 0,
                    row_start: offset,
                });
            }
        };

        // The +3 skips the PLINK .bed file magic number (0x6c, 0x1b, 0x01).
        let row_start = local_bim_index
            .checked_mul(bytes_per_variant)
            .and_then(|offset| offset.checked_add(3))
            .ok_or_else(|| {
                PipelineError::Io(format!(
                    "BED offset of BIM row {} overflows u64.",
                    bim_row_index.0
                ))
            })?;
        Ok(RowLocation {
            bim_row_index,
            source,
            row_start,
        })
    }

    fn bed_source(&self, location: &RowLocation) -> Option<&BedSource> {
        match self {
            ComplexVariantResolver::SingleFile(source) => Some(source),
            ComplexVariantResolver::MultiFile { sources, .. } => sources.get(location.source),
            ComplexVariantResolver::Spool { .. } => None,
        }
    }

    /// Borrows the scored span of a row from its memory map.
    fn mapped_span(&self, location: &RowLocation, span: RowSpan) -> Result<&[u8], PipelineError> {
        let start = location
            .row_start
            .checked_add(span.start)
            .and_then(|start| usize::try_from(start).ok());
        let bytes = match self {
            ComplexVariantResolver::Spool { mmap, .. } => {
                start.and_then(|start| mmap.get(start..start.checked_add(span.len)?))
            }
            _ => start.and_then(|start| self.bed_source(location)?.mmap_slice(start, span.len)),
        };
        bytes.ok_or_else(|| span_out_of_range(location))
    }

    /// Reads the scored span of a row from a source without a memory map.
    fn read_span(
        &self,
        location: &RowLocation,
        span: RowSpan,
        dst: &mut [u8],
    ) -> Result<(), PipelineError> {
        let source = self
            .bed_source(location)
            .ok_or_else(|| span_out_of_range(location))?;
        let start = location
            .row_start
            .checked_add(span.start)
            .ok_or_else(|| span_out_of_range(location))?;
        source.read_at(start, dst)
    }
}

fn span_out_of_range(location: &RowLocation) -> PipelineError {
    PipelineError::Io(format!(
        "Genotypes for BIM row {} lie outside the genotype data.",
        location.bim_row_index.0
    ))
}

/// Where one rule context's packed row starts.
#[derive(Clone, Copy)]
struct RowLocation {
    bim_row_index: BimRowIndex,
    source: usize,
    row_start: u64,
}

/// The bytes of a row that hold scored people, relative to the row start.
#[derive(Clone, Copy)]
struct RowSpan {
    start: u64,
    len: usize,
}

/// Where each scored person's two genotype bits sit inside a row, in output order.
/// Built once per run, so decoding a row is a gather with no per-genotype lookups.
struct PersonLayout {
    /// Each person's byte, relative to `span.start`.
    bytes: Vec<u32>,
    /// Each person's bit shift within that byte.
    shifts: Vec<u8>,
    /// People whose byte was pruned from the spool, in increasing order.
    forced_missing: Vec<usize>,
    span: RowSpan,
    /// Whether person `p` sits at byte `p / 4` with shift `2 * (p % 4)`, as when every
    /// sample is scored in .fam order, so rows can be read a byte at a time.
    contiguous: bool,
}

impl PersonLayout {
    fn new(
        resolver: &ComplexVariantResolver,
        prep_result: &PreparationResult,
        num_people: usize,
    ) -> Result<Self, PipelineError> {
        let mut row_bytes = Vec::with_capacity(num_people);
        let mut shifts = Vec::with_capacity(num_people);
        let mut forced_missing = Vec::new();
        for person_output_idx in 0..num_people {
            let output_person_idx = u32::try_from(person_output_idx)
                .map(OutputPersonIndex)
                .map_err(|_| {
                    PipelineError::Io(format!(
                        "Output person index {} exceeds u32::MAX.",
                        person_output_idx
                    ))
                })?;
            let fam_index = prep_result
                .original_person_index_for_output(output_person_idx)
                .0;
            let byte = match resolver {
                ComplexVariantResolver::Spool {
                    bytes_per_spooled_variant,
                    dense_map,
                    ..
                } => {
                    let orig_byte_idx = (fam_index / 4) as usize;
                    let Some(&compact_idx) = dense_map.get(orig_byte_idx) else {
                        return Err(fetch_error(PipelineError::Io(format!(
                            "Family index {} out of bounds for complex spool lookup.",
                            fam_index
                        ))));
                    };
                    if compact_idx < 0 {
                        // Defensive: queries should target kept individuals only, but if we miss
                        // a guard higher up we fall back to a standard PLINK "missing" genotype.
                        forced_missing.push(person_output_idx);
                        None
                    } else if compact_idx as u64 >= *bytes_per_spooled_variant {
                        return Err(PipelineError::Io(format!(
                            "compact index {} exceeds spool stride {}",
                            compact_idx, bytes_per_spooled_variant
                        )));
                    } else {
                        Some(compact_idx as u32)
                    }
                }
                _ => Some(fam_index / 4),
            };
            row_bytes.push(byte);
            shifts.push(((fam_index % 4) * 2) as u8);
        }

        let first = row_bytes.iter().flatten().min().copied();
        let last = row_bytes.iter().flatten().max().copied();
        let span = match (first, last) {
            (Some(first), Some(last)) => RowSpan {
                start: u64::from(first),
                len: (last - first) as usize + 1,
            },
            _ => RowSpan { start: 0, len: 0 },
        };
        let bytes: Vec<u32> = row_bytes
            .into_iter()
            .map(|byte| byte.map_or(0, |byte| byte - span.start as u32))
            .collect();
        let contiguous = forced_missing.is_empty()
            && bytes.iter().zip(&shifts).enumerate().all(|(person, (&byte, &shift))| {
                byte as usize == person / 4 && usize::from(shift) == 2 * (person % 4)
            });
        Ok(Self {
            bytes,
            shifts,
            forced_missing,
            span,
            contiguous,
        })
    }
}

/// Unpacks one context's genotypes for a run of people.
#[inline]
fn decode_genotypes(row: &[u8], bytes: &[u32], shifts: &[u8], out: &mut [u8]) {
    if row.is_empty() {
        // No scored person has a byte in the row: every one was pruned from the spool.
        out.fill(0b01);
        return;
    }
    for ((genotype, &byte), &shift) in out.iter_mut().zip(bytes).zip(shifts) {
        *genotype = (row[byte as usize] >> shift) & 0b11;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::io::{ByteRangeSource, open_bed_source};
    use crate::score::types::{OriginalPersonIndex, PersonSubset, PipelineKind, ScoreColumnIndex};
    use memmap2::MmapOptions;
    use std::path::{Path, PathBuf};

    /// A PreparationResult carrying only what the complex resolver reads.
    fn test_prep_result(
        rules: Vec<GroupedComplexRule>,
        total_people: usize,
        kept: &[u32],
        num_scores: usize,
        total_variants: u64,
    ) -> PreparationResult {
        let bytes_per_variant = (total_people as u64).div_ceil(4);
        let output_idx_to_fam_idx = kept.iter().copied().map(OriginalPersonIndex).collect();
        let mut person_fam_to_output_idx = vec![None; total_people];
        for (output_idx, &fam_idx) in kept.iter().enumerate() {
            person_fam_to_output_idx[fam_idx as usize] = Some(OutputPersonIndex(output_idx as u32));
        }
        let mut sorted_kept = kept.to_vec();
        sorted_kept.sort_unstable();
        let mut compact: Vec<u32> = sorted_kept.iter().map(|fam_idx| fam_idx / 4).collect();
        compact.dedup();
        let mut dense = vec![-1i32; bytes_per_variant as usize];
        for (compact_idx, &byte) in compact.iter().enumerate() {
            dense[byte as usize] = compact_idx as i32;
        }
        let spool_bytes_per_variant = compact.len() as u64;
        let names: Vec<String> = (0..num_scores).map(|i| format!("S{i}")).collect();
        let exact = crate::score::cells::ExactPlan::new(Vec::new(), &[], &[], &[0], &rules, &names)
            .expect("exact plan");
        PreparationResult::new(
            exact,
            Vec::new(),
            vec![0],
            Vec::new(),
            rules,
            names,
            vec![0; num_scores],
            PersonSubset::Indices(sorted_kept),
            kept.iter().map(|fam_idx| format!("IID{fam_idx}")).collect(),
            kept.len(),
            total_people,
            total_variants,
            0,
            bytes_per_variant,
            person_fam_to_output_idx,
            output_idx_to_fam_idx,
            Vec::new(),
            compact,
            dense,
            spool_bytes_per_variant,
            PipelineKind::SingleFile(PathBuf::from("test")),
        )
    }

    /// Reads every scored person's genotype for one variant through the row-major path.
    fn read_row_genotypes(
        resolver: &ComplexVariantResolver,
        prep_result: &PreparationResult,
        bim_row_index: BimRowIndex,
    ) -> Vec<u8> {
        let layout = PersonLayout::new(resolver, prep_result, prep_result.num_people_to_score)
            .expect("person layout");
        let location = resolver
            .locate_row(prep_result.bytes_per_variant, bim_row_index)
            .expect("row location");
        let mut storage = vec![0u8; layout.span.len];
        let row = if resolver.is_mapped() {
            resolver
                .mapped_span(&location, layout.span)
                .expect("mapped span")
        } else {
            resolver
                .read_span(&location, layout.span, &mut storage)
                .expect("read span");
            &storage
        };
        let mut genotypes = vec![0u8; prep_result.num_people_to_score];
        decode_genotypes(row, &layout.bytes, &layout.shifts, &mut genotypes);
        for &person in &layout.forced_missing {
            genotypes[person] = 0b01;
        }
        genotypes
    }

    #[test]
    fn spool_resolver_returns_expected_genotypes() {
        let mut offsets = AHashMap::new();
        offsets.insert(BimRowIndex(0), 0);
        offsets.insert(BimRowIndex(1), 2);

        let spool_bytes_per_variant = 2;
        let spool_data = vec![0xD2, 0x1B, 0x55, 0xF0];
        let mut mmap_mut = MmapOptions::new()
            .len(spool_data.len())
            .map_anon()
            .expect("failed to allocate test spool buffer");
        mmap_mut.copy_from_slice(&spool_data);
        let mmap = Arc::new(
            mmap_mut
                .make_read_only()
                .expect("failed to convert test spool mapping to read-only"),
        );
        let dense_map = Arc::new(vec![0, -1, 1]);
        let resolver = ComplexVariantResolver::from_spool(
            Arc::clone(&mmap),
            offsets.clone(),
            spool_bytes_per_variant,
            Arc::clone(&dense_map),
        );
        // People 0, 8, 5 and 9 of twelve. Person 5's byte was pruned from the spool.
        let prep_result = test_prep_result(Vec::new(), 12, &[0, 8, 5, 9], 1, 2);

        // Variant 0 pulls bytes from offsets 0 and 1.
        let variant_0 = read_row_genotypes(&resolver, &prep_result, BimRowIndex(0));
        assert_eq!(variant_0[0], 0b10);
        assert_eq!(variant_0[1], 0b11);
        // Person 5 falls back to a missing genotype.
        assert_eq!(variant_0[2], 0b01);

        // Variant 1 pulls bytes from offsets 2 and 3.
        let variant_1 = read_row_genotypes(&resolver, &prep_result, BimRowIndex(1));
        assert_eq!(variant_1[0], 0b01);
        assert_eq!(variant_1[3], 0b00);

        // When every scored person was pruned, rows have no scored bytes to read.
        let pruned_only = test_prep_result(Vec::new(), 12, &[5], 1, 2);
        assert_eq!(
            read_row_genotypes(&resolver, &pruned_only, BimRowIndex(1)),
            vec![0b01]
        );
    }

    /// A small deterministic generator, so the scenarios need no seeding API.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    struct VecSource(Vec<u8>);

    impl ByteRangeSource for VecSource {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
            let start = offset as usize;
            let bytes = self.0.get(start..start + dst.len()).ok_or_else(|| {
                PipelineError::Io(format!(
                    "read of {} bytes at {offset} is past the end",
                    dst.len()
                ))
            })?;
            dst.copy_from_slice(bytes);
            Ok(())
        }
    }

    fn bed_bytes(rows: &[Vec<u8>]) -> Vec<u8> {
        let mut data = vec![0x6c, 0x1b, 0x01];
        for row in rows {
            data.extend_from_slice(row);
        }
        data
    }

    /// Random packed genotypes (missing calls and padding bits included) and random sites: a REF
    /// written in one of two frames, ALTs drawn from a small vocabulary so a variant is often
    /// measured by several rows, and score rows naming an ALT or the REF of a variant. Rows are
    /// often shared between sites, so the calls of one variant's rows agree only by chance.
    struct Scenario {
        total_people: usize,
        rows: Vec<Vec<u8>>,
        rules: Vec<GroupedComplexRule>,
        kept: Vec<u32>,
        num_scores: usize,
    }

    impl Scenario {
        fn random(seed: u64, total_people: usize, keep_all: bool) -> Self {
            // Each ALT beside the REF C, and the same variant written in the frame of the REF CA.
            const VARIANTS: [[(&str, &str); 2]; 3] = [
                [("C", "T"), ("CA", "TA")],
                [("C", "G"), ("CA", "GA")],
                [("C", "A"), ("CA", "AA")],
            ];
            let mut rng = SplitMix64(seed);
            let num_variants = 40;
            // Rows that often agree: each row is a copy of an earlier one with a few calls changed.
            let mut rows: Vec<Vec<u8>> = Vec::new();
            for _ in 0..num_variants {
                let row = match rows.len() {
                    0 => (0..total_people.div_ceil(4)).map(|_| rng.next() as u8).collect(),
                    known => {
                        let mut row = rows[rng.below(known)].clone();
                        for byte in &mut row {
                            if rng.below(4) == 0 {
                                *byte ^= 1 << (2 * rng.below(4));
                            }
                        }
                        row
                    }
                };
                rows.push(row);
            }
            let num_scores = 1 + rng.below(4);
            let mut rules = Vec::new();
            for rule_idx in 0..60 {
                let num_contexts = 2 + rng.below(5);
                let possible_contexts: Vec<(BimRowIndex, String, String)> = (0..num_contexts)
                    .map(|_| {
                        let (reference, alternate) = VARIANTS[rng.below(VARIANTS.len())][rng.below(2)];
                        // A `.bim` writes its alleles in either order.
                        let (allele1, allele2) = if rng.below(2) == 0 {
                            (alternate, reference)
                        } else {
                            (reference, alternate)
                        };
                        (
                            BimRowIndex(rng.below(num_variants) as u64),
                            allele1.to_string(),
                            allele2.to_string(),
                        )
                    })
                    .collect();
                let score_applications = (0..1 + rng.below(4))
                    .map(|_| {
                        let (_, allele1, allele2) = &possible_contexts[rng.below(num_contexts)];
                        let (effect_allele, other_allele) = if rng.below(2) == 0 {
                            (allele1.clone(), allele2.clone())
                        } else {
                            (allele2.clone(), allele1.clone())
                        };
                        ScoreInfo {
                            effect_allele,
                            other_allele,
                            weight: rng.unit() * 3.0 - 1.5,
                            score_column_index: ScoreColumnIndex(rng.below(num_scores)),
                        }
                    })
                    .collect();
                rules.push(GroupedComplexRule {
                    locus_chr_pos: ("22".to_string(), 1000 + rule_idx),
                    possible_contexts,
                    score_applications,
                    reference_declared: false,
                });
            }
            // A site with more rows than a table covers, one variant measured by all of them.
            rules.push(GroupedComplexRule {
                locus_chr_pos: ("22".to_string(), 5000),
                possible_contexts: (0..6)
                    .map(|variant| (BimRowIndex(variant), "G".to_string(), "A".to_string()))
                    .collect(),
                score_applications: vec![
                    ScoreInfo {
                        effect_allele: "G".to_string(),
                        other_allele: "A".to_string(),
                        weight: 0.7,
                        score_column_index: ScoreColumnIndex(0),
                    },
                    ScoreInfo {
                        effect_allele: "A".to_string(),
                        other_allele: "G".to_string(),
                        weight: -0.3,
                        score_column_index: ScoreColumnIndex(0),
                    },
                ],
                reference_declared: false,
            });

            let mut kept: Vec<u32> = (0..total_people as u32)
                .filter(|_| keep_all || rng.below(3) != 0)
                .collect();
            if kept.is_empty() {
                kept.push(total_people as u32 - 1);
            }
            if !keep_all {
                for i in (1..kept.len()).rev() {
                    kept.swap(i, rng.below(i + 1));
                }
            }
            Self {
                total_people,
                rows,
                rules,
                kept,
                num_scores,
            }
        }

        fn genotype(&self, bim_row_index: BimRowIndex, fam_idx: usize) -> u8 {
            (self.rows[bim_row_index.0 as usize][fam_idx / 4] >> ((fam_idx % 4) * 2)) & 0b11
        }

        /// Accumulators as the fast path might leave them: arbitrary lanes and counts.
        fn initial_accumulators(&self, seed: u64, stride: usize) -> (Vec<i64>, Vec<u32>) {
            let mut rng = SplitMix64(seed ^ 0xA5A5);
            let cells = self.kept.len() * self.num_scores;
            let scores = (0..self.kept.len() * stride)
                .map(|_| match rng.below(3) {
                    0 => 0,
                    _ => rng.next() as i64,
                })
                .collect();
            let counts = (0..cells).map(|_| rng.below(7) as u32).collect();
            (scores, counts)
        }

        fn prep_result(&self) -> PreparationResult {
            test_prep_result(
                self.rules.clone(),
                self.total_people,
                &self.kept,
                self.num_scores,
                self.rows.len() as u64,
            )
        }

        fn single_file(&self, dir: &Path) -> ComplexVariantResolver {
            let path = dir.join("all.bed");
            std::fs::write(&path, bed_bytes(&self.rows)).expect("write test bed");
            ComplexVariantResolver::from_single_source(
                open_bed_source(&path, None).expect("open test bed"),
            )
        }

        fn multi_file(&self, dir: &Path) -> ComplexVariantResolver {
            let starts = [0, 13, 29];
            let mut sources = Vec::new();
            let mut boundaries = Vec::new();
            for (part, &start) in starts.iter().enumerate() {
                let end = starts.get(part + 1).copied().unwrap_or(self.rows.len());
                let bed_path = dir.join(format!("part{part}.bed"));
                std::fs::write(&bed_path, bed_bytes(&self.rows[start..end]))
                    .expect("write test bed part");
                sources.push(open_bed_source(&bed_path, None).expect("open test bed part"));
                boundaries.push(FilesetBoundary {
                    bim_path: bed_path.with_extension("bim"),
                    fam_path: bed_path.with_extension("fam"),
                    bed_path,
                    starting_global_index: start as u64,
                });
            }
            ComplexVariantResolver::from_multi_sources(sources, boundaries)
                .expect("matching sources and boundaries")
        }

        fn streamed(&self) -> ComplexVariantResolver {
            ComplexVariantResolver::from_single_source(BedSource::from_byte_source(Arc::new(
                VecSource(bed_bytes(&self.rows)),
            )))
        }

        fn spool(&self, prep_result: &PreparationResult) -> ComplexVariantResolver {
            let compact = prep_result.spool_compact_byte_index();
            let mut offsets = AHashMap::new();
            let mut data = Vec::new();
            // Spool rows in reverse, so offsets do not follow BIM order.
            for variant in (0..self.rows.len()).rev() {
                offsets.insert(BimRowIndex(variant as u64), data.len() as u64);
                data.extend(compact.iter().map(|&byte| self.rows[variant][byte as usize]));
            }
            let mut mmap_mut = MmapOptions::new()
                .len(data.len())
                .map_anon()
                .expect("allocate test spool");
            mmap_mut.copy_from_slice(&data);
            ComplexVariantResolver::from_spool(
                Arc::new(mmap_mut.make_read_only().expect("read-only test spool")),
                offsets,
                compact.len() as u64,
                Arc::new(prep_result.spool_dense_map().to_vec()),
            )
        }
    }

    /// The site rule person by person: every person, every rule, every application, in that
    /// order, with each variant's copies taken from the list of its rows' called copies. People
    /// whose byte is `pruned_byte` have missing calls, as a spool that pruned it gives them.
    /// Returns the conflicts per (rule, application).
    fn reference_resolve(
        scenario: &Scenario,
        prep_result: &PreparationResult,
        pruned_byte: Option<usize>,
        scores: &mut [i64],
        counts: &mut [u32],
    ) -> Vec<((usize, usize), u64)> {
        let num_scores = prep_result.score_names.len();
        let exact = prep_result.exact();
        let stride = exact.stride();
        let mut conflicts: AHashMap<(usize, usize), u64> = AHashMap::new();
        for (person, fam_idx) in prep_result.output_idx_to_fam_idx.iter().enumerate() {
            let fam_idx = fam_idx.0 as usize;
            for (rule_idx, rule) in prep_result.complex_rules.iter().enumerate() {
                let rows: Vec<(usize, &str, &str)> = rule
                    .possible_contexts
                    .iter()
                    .enumerate()
                    .map(|(context, (_, allele1, allele2))| (context, allele2.as_str(), allele1.as_str()))
                    .collect();
                let site = Site::new(&rows, rule.reference_declared);
                let variants = site.variants().len();
                // Every variant's called copies of its ALT, over its rows.
                let mut called: Vec<Vec<u32>> = vec![Vec::new(); variants];
                for (context, (bim_row, _, _)) in rule.possible_contexts.iter().enumerate() {
                    let bits = if pruned_byte == Some(fam_idx / 4) {
                        0b01
                    } else {
                        scenario.genotype(*bim_row, fam_idx)
                    };
                    let allele1 = match bits {
                        0b00 => 2,
                        0b10 => 1,
                        0b11 => 0,
                        _ => continue,
                    };
                    // A row whose REF is allele 2 has allele 1 as its ALT.
                    let copies = if site.first_is_reference()[context] { allele1 } else { 2 - allele1 };
                    called[site.row_variants()[context]].push(copies);
                }
                // Each variant's copies: Some(Ok) when its calls agree, Some(Err) when they do not.
                let copies: Vec<Option<Result<u32, ()>>> = called
                    .iter()
                    .map(|calls| {
                        let first = *calls.first()?;
                        Some(if calls.iter().all(|&c| c == first) { Ok(first) } else { Err(()) })
                    })
                    .collect();
                let alleles: Vec<SiteAllele> = rule
                    .score_applications
                    .iter()
                    .map(|info| match site.match_row(&info.effect_allele, &info.other_allele) {
                        RowMatch::Scores(allele) => allele,
                        other => panic!("scenario rule names no one allele: {other:?}"),
                    })
                    .collect();
                let key = |index: usize| {
                    (
                        alleles[index].variant(),
                        rule.score_applications[index].score_column_index.0,
                    )
                };
                for (index, score_info) in rule.score_applications.iter().enumerate() {
                    let column = score_info.score_column_index.0;
                    let first_of_pair = (0..index).all(|earlier| key(earlier) != key(index));
                    let whole_site = (0..alleles.len())
                        .any(|other| key(other) == key(index) && site.reads_whole_site(alleles[other]));
                    let dose: Result<u32, bool> = if whole_site {
                        if copies.iter().any(|c| matches!(c, Some(Err(())))) {
                            Err(true)
                        } else if copies.iter().any(Option::is_none) {
                            Err(false)
                        } else {
                            let alternates: u32 = copies.iter().map(|c| c.unwrap().unwrap()).sum();
                            match (alleles[index], alternates) {
                                (_, total) if total > 2 => Err(true),
                                (SiteAllele::Reference(_), total) => Ok(2 - total),
                                (SiteAllele::Alternate(variant), _) => Ok(copies[variant].unwrap().unwrap()),
                            }
                        }
                    } else {
                        // A variant's own REF copies are two less its ALT's.
                        match (alleles[index], copies[alleles[index].variant()]) {
                            (SiteAllele::Alternate(_), Some(Ok(copies))) => Ok(copies),
                            (SiteAllele::Reference(_), Some(Ok(copies))) => Ok(2 - copies),
                            (_, Some(Err(()))) => Err(true),
                            (_, None) => Err(false),
                        }
                    };
                    match dose {
                        Ok(dose) => exact.add(
                            exact.complex_target(column, score_info.weight),
                            exact.complex_term(column, score_info.weight, dose),
                            &mut scores[person * stride..(person + 1) * stride],
                        ),
                        Err(conflict) => {
                            if first_of_pair {
                                counts[person * num_scores + column] += 1;
                                if conflict {
                                    *conflicts.entry((rule_idx, index)).or_default() += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut conflicts: Vec<((usize, usize), u64)> = conflicts.into_iter().collect();
        conflicts.sort_unstable();
        conflicts
    }

    #[test]
    fn row_major_resolver_matches_person_major_reference() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (mut conflicting, mut reference_scored) = (0u64, false);
        for (seed, total_people, keep_all) in [
            (1, 1, true),
            (2, 3, true),
            (3, 5, false),
            (4, 37, true),
            (5, 37, false),
            (6, 1001, false),
            (7, 1003, true),
        ] {
            let scenario = Scenario::random(seed, total_people, keep_all);
            let prep_result = scenario.prep_result();
            let (initial_scores, initial_counts) = scenario.initial_accumulators(seed, prep_result.exact().stride());
            let mut expected_scores = initial_scores.clone();
            let mut expected_counts = initial_counts.clone();
            let expected_conflicts = reference_resolve(
                &scenario,
                &prep_result,
                None,
                &mut expected_scores,
                &mut expected_counts,
            );
            conflicting += expected_conflicts.iter().map(|(_, count)| count).sum::<u64>();
            reference_scored |= prep_result.complex_rules.iter().any(|rule| {
                let rows: Vec<(usize, &str, &str)> = rule
                    .possible_contexts
                    .iter()
                    .enumerate()
                    .map(|(context, (_, allele1, allele2))| (context, allele2.as_str(), allele1.as_str()))
                    .collect();
                let site = Site::new(&rows, false);
                rule.score_applications.iter().any(|info| {
                    matches!(site.match_row(&info.effect_allele, &info.other_allele), RowMatch::Scores(allele) if site.reads_whole_site(allele))
                })
            });

            let scenario_dir = dir.path().join(format!("seed{seed}"));
            std::fs::create_dir_all(&scenario_dir).expect("scenario dir");
            let resolvers = [
                ("single file", scenario.single_file(&scenario_dir)),
                ("multi file", scenario.multi_file(&scenario_dir)),
                ("streamed", scenario.streamed()),
                ("spool", scenario.spool(&prep_result)),
            ];
            for (label, resolver) in &resolvers {
                for limits in [
                    ResolveLimits {
                        block_people: 1,
                        streamed_group_bytes: 1,
                    },
                    ResolveLimits {
                        block_people: 7,
                        streamed_group_bytes: 97,
                    },
                    ResolveLimits::for_people(scenario.kept.len()),
                ] {
                    let mut scores = initial_scores.clone();
                    let mut counts = initial_counts.clone();
                    let Ok(report) = resolve_rows(
                        resolver,
                        &prep_result,
                        &mut scores,
                        &mut counts,
                        limits,
                        &ProgressBar::hidden(),
                    ) else {
                        panic!("resolution failed for seed {seed}, {label}");
                    };
                    let context = format!(
                        "seed {seed}, {total_people} people, {label}, {} per block",
                        limits.block_people
                    );
                    assert_eq!(scores, expected_scores, "{context}");
                    assert_eq!(counts, expected_counts, "{context}");
                    assert_eq!(report.conflicts, expected_conflicts, "{context}");
                }
            }
        }
        // The scenarios must reach disagreeing measurements and REF doses, not only lone ALTs.
        assert!(conflicting > 0 && reference_scored);
    }

    /// The rule with `rows` at one site, given as (allele 1, allele 2) with the calls of four
    /// people each, and `applications` as (effect, other, weight) of one score.
    fn site_prep(rows: &[(&str, &str, [u8; 4])], applications: &[(&str, &str, f64)]) -> (PreparationResult, Vec<Vec<u8>>) {
        let rule = GroupedComplexRule {
            locus_chr_pos: ("1".to_string(), 100),
            possible_contexts: rows
                .iter()
                .enumerate()
                .map(|(row, (allele1, allele2, _))| (BimRowIndex(row as u64), allele1.to_string(), allele2.to_string()))
                .collect(),
            score_applications: applications
                .iter()
                .map(|&(effect, other, weight)| ScoreInfo {
                    effect_allele: effect.to_string(),
                    other_allele: other.to_string(),
                    weight,
                    score_column_index: ScoreColumnIndex(0),
                })
                .collect(),
            reference_declared: false,
        };
        let bed_rows = rows
            .iter()
            .map(|(_, _, calls)| {
                vec![calls.iter().enumerate().fold(0u8, |byte, (person, call)| byte | (call << (2 * person)))]
            })
            .collect();
        (test_prep_result(vec![rule], 4, &[0, 1, 2, 3], 1, rows.len() as u64), bed_rows)
    }

    /// Each person's sum and missing count for the one score of `prep_result` over `bed_rows`.
    fn resolved_site(prep_result: &PreparationResult, bed_rows: &[Vec<u8>]) -> (Vec<f64>, Vec<u32>, u64) {
        let resolver = ComplexVariantResolver::from_single_source(BedSource::from_byte_source(Arc::new(
            VecSource(bed_bytes(bed_rows)),
        )));
        let stride = prep_result.exact().stride();
        let mut scores = vec![0i64; 4 * stride];
        let mut counts = vec![0u32; 4];
        let report = resolve_rows(
            &resolver,
            prep_result,
            &mut scores,
            &mut counts,
            ResolveLimits::for_people(4),
            &ProgressBar::hidden(),
        )
        .expect("resolution");
        let sums = (0..4)
            .map(|person| prep_result.exact().sum(0, &scores[person * stride..(person + 1) * stride]))
            .collect();
        (sums, counts, report.conflicts.iter().map(|(_, count)| count).sum())
    }

    // PLINK codes: 00 two copies of allele 1, 10 one, 11 none, 01 missing.
    const HOM_ALT: u8 = 0b00;
    const HET: u8 = 0b10;
    const HOM_REF: u8 = 0b11;
    const MISSING: u8 = 0b01;

    #[test]
    fn the_ref_of_a_split_site_is_the_ploidy_less_every_alt() {
        // A>G and A>T split: people G/A, T/A, G/T and T/T.
        let rows = [
            ("G", "A", [HET, HOM_REF, HET, HOM_REF]),
            ("T", "A", [HOM_REF, HET, HET, HOM_ALT]),
        ];
        let (prep_result, bed_rows) = site_prep(&rows, &[("A", "G", 1.0)]);
        assert_eq!(resolved_site(&prep_result, &bed_rows), (vec![1.0, 1.0, 0.0, 0.0], vec![0; 4], 0));
        // Named through the other ALT, it is the same allele.
        let (prep_result, bed_rows) = site_prep(&rows, &[("A", "T", 1.0)]);
        assert_eq!(resolved_site(&prep_result, &bed_rows).0, vec![1.0, 1.0, 0.0, 0.0]);
        // A missing call on either row leaves the REF's dose unknown; ALTs past the ploidy conflict.
        let rows = [
            ("G", "A", [MISSING, HOM_ALT, HET, HOM_REF]),
            ("T", "A", [HOM_REF, HET, MISSING, HOM_REF]),
        ];
        let (prep_result, bed_rows) = site_prep(&rows, &[("A", "G", 1.0), ("G", "A", 0.5)]);
        assert_eq!(resolved_site(&prep_result, &bed_rows), (vec![0.0, 0.0, 0.0, 2.0], vec![1, 1, 1, 0], 1));
    }

    #[test]
    fn repeated_rows_of_one_variant_score_the_dose_their_calls_agree_on() {
        let rows = [
            ("G", "A", [HET, HOM_ALT, MISSING, HET]),
            ("G", "A", [HET, HET, HOM_REF, MISSING]),
        ];
        let (prep_result, bed_rows) = site_prep(&rows, &[("G", "A", 1.0)]);
        // Agreeing calls give their dose, a missing call is no measurement, and calls that
        // disagree leave the dose missing and are reported.
        assert_eq!(resolved_site(&prep_result, &bed_rows), (vec![1.0, 0.0, 0.0, 1.0], vec![0, 1, 0, 0], 1));
        // One missing dose per (variant, score), however many rows of the score name the variant.
        let (prep_result, bed_rows) = site_prep(&rows, &[("G", "A", 1.0), ("A", "G", 0.25)]);
        assert_eq!(resolved_site(&prep_result, &bed_rows), (vec![1.25, 0.0, 0.5, 1.25], vec![0, 1, 0, 0], 1));
    }

    #[test]
    fn pruned_spool_bytes_resolve_as_missing_calls() {
        for (seed, total_people, keep_all) in [(21, 37, true), (22, 1001, false)] {
            let scenario = Scenario::random(seed, total_people, keep_all);
            let prep_result = scenario.prep_result();
            let ComplexVariantResolver::Spool {
                mmap,
                offsets,
                bytes_per_spooled_variant,
                dense_map,
            } = scenario.spool(&prep_result)
            else {
                unreachable!("scenario spool resolver");
            };
            // The spool loses the byte of the first scored person, so everyone in it
            // must resolve as missing calls, whichever kernel their block uses.
            let pruned_byte = scenario.kept[0] as usize / 4;
            let mut dense_map = dense_map.to_vec();
            dense_map[pruned_byte] = -1;
            let resolver = ComplexVariantResolver::from_spool(
                mmap,
                offsets,
                bytes_per_spooled_variant,
                Arc::new(dense_map),
            );
            let (initial_scores, initial_counts) = scenario.initial_accumulators(seed, prep_result.exact().stride());
            let mut expected_scores = initial_scores.clone();
            let mut expected_counts = initial_counts.clone();
            let expected_conflicts = reference_resolve(
                &scenario,
                &prep_result,
                Some(pruned_byte),
                &mut expected_scores,
                &mut expected_counts,
            );
            for block_people in [1, 7, 256] {
                let mut scores = initial_scores.clone();
                let mut counts = initial_counts.clone();
                let Ok(report) = resolve_rows(
                    &resolver,
                    &prep_result,
                    &mut scores,
                    &mut counts,
                    ResolveLimits {
                        block_people,
                        streamed_group_bytes: STREAMED_GROUP_BYTES,
                    },
                    &ProgressBar::hidden(),
                ) else {
                    panic!("resolution failed for seed {seed}, {block_people} per block");
                };
                let context = format!("seed {seed}, {block_people} per block");
                assert_eq!(scores, expected_scores, "{context}");
                assert_eq!(counts, expected_counts, "{context}");
                assert_eq!(report.conflicts, expected_conflicts, "{context}");
            }
        }
    }

    #[test]
    fn missing_spool_offset_reports_the_person_major_error_text() {
        let scenario = Scenario::random(11, 9, true);
        let prep_result = scenario.prep_result();
        let ComplexVariantResolver::Spool {
            mmap,
            mut offsets,
            bytes_per_spooled_variant,
            dense_map,
        } = scenario.spool(&prep_result)
        else {
            unreachable!("scenario spool resolver");
        };
        offsets.remove(&BimRowIndex(0));
        let resolver =
            ComplexVariantResolver::from_spool(mmap, offsets, bytes_per_spooled_variant, dense_map);
        let (mut scores, mut counts) = scenario.initial_accumulators(11, prep_result.exact().stride());
        let Err(error) = resolve_rows(
            &resolver,
            &prep_result,
            &mut scores,
            &mut counts,
            ResolveLimits::for_people(scenario.kept.len()),
            &ProgressBar::hidden(),
        ) else {
            panic!("a missing spool offset must fail resolution");
        };
        assert_eq!(
            error.to_string(),
            "I/O error during pipeline execution: I/O error during pipeline execution: \
             Missing spool offset for BIM row 0 while resolving complex variant."
        );
    }

    #[test]
    fn truncated_bed_row_is_an_error_not_a_read_past_the_map() {
        let scenario = Scenario::random(12, 10, true);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("truncated.bed");
        let mut data = bed_bytes(&scenario.rows);
        data.truncate(data.len() - 1);
        std::fs::write(&path, data).expect("write truncated bed");
        let resolver = ComplexVariantResolver::from_single_source(
            open_bed_source(&path, None).expect("open truncated bed"),
        );
        let last_variant = scenario.rows.len() as u64 - 1;
        let rules = vec![GroupedComplexRule {
            locus_chr_pos: ("22".to_string(), 1),
            possible_contexts: vec![(BimRowIndex(last_variant), "A".to_string(), "G".to_string())],
            score_applications: vec![ScoreInfo {
                effect_allele: "A".to_string(),
                other_allele: "G".to_string(),
                weight: 1.0,
                score_column_index: ScoreColumnIndex(0),
            }],
            reference_declared: false,
        }];
        let prep_result = test_prep_result(rules, 10, &scenario.kept, 1, last_variant + 1);
        let mut scores = vec![0i64; scenario.kept.len() * prep_result.exact().stride()];
        let mut counts = vec![0; scenario.kept.len()];
        let Err(error) = resolve_rows(
            &resolver,
            &prep_result,
            &mut scores,
            &mut counts,
            ResolveLimits::for_people(scenario.kept.len()),
            &ProgressBar::hidden(),
        ) else {
            panic!("a truncated row must fail resolution");
        };
        assert!(
            error
                .to_string()
                .contains("Genotypes for BIM row 39 lie outside the genotype data."),
            "{error}"
        );
    }
}

//========================================================================================
//
//                      The site rule, resolved per person
//
//========================================================================================

/// Genotype fetch failures surface as the person-major resolver reported them:
/// the fetch error's text, wrapped as an I/O error.
fn fetch_error(error: PipelineError) -> PipelineError {
    PipelineError::Io(error.to_string())
}

/// The most contexts an application may read and still be tabulated (4^4 genotype
/// combinations). Wider applications resolve per person.
const TABULATED_MAX_CONTEXTS: usize = 4;

/// Loci a warning names as examples.
const MAX_CONFLICT_EXAMPLES: usize = 5;

/// People per evaluation block: few enough that a block's decoded genotypes and
/// score rows stay in cache, enough to amortize the per-block setup.
const MIN_BLOCK_PEOPLE: usize = 256;
const MAX_BLOCK_PEOPLE: usize = 4096;

/// Row bytes a resolver without memory maps reads for one group of rules.
const STREAMED_GROUP_BYTES: usize = 64 << 20;

/// The copies of allele 1, a `.bim` row's ALT, in each PLINK code [00, 01, 10, 11]: `None` for the
/// missing call.
const ALT_COPIES: [Option<u32>; 4] = [Some(2), None, Some(1), Some(0)];

/// What one combination of genotypes on an application's contexts does to one score of one person.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    /// The dose is known: its exact term.
    Add(i128),
    /// A call the dose needs is missing.
    Missing,
    /// Calls the dose needs disagree: rows measuring one variant give different copies, or the
    /// ALTs' copies at the site pass the ploidy. No genotype has these calls, so the dose is unknown.
    Conflict,
}

/// The branch-free form of an `Outcome`, applied to every person.
#[derive(Clone, Copy)]
struct TableEntry {
    /// The exact term added to the score; zero for outcomes that add nothing.
    value: i128,
    /// Added to the missing count.
    missing: u32,
    /// Whether the outcome is a conflict to report.
    conflict: bool,
}

impl TableEntry {
    /// The entry of `outcome` for an application that does, or does not, count its variant's
    /// missing dose. Each (variant, score) counts one missing dose however many rows name it.
    fn new(outcome: Outcome, counts_missing: bool) -> Self {
        match outcome {
            Outcome::Add(value) => Self {
                value,
                missing: 0,
                conflict: false,
            },
            Outcome::Missing => Self {
                value: 0,
                missing: u32::from(counts_missing),
                conflict: false,
            },
            Outcome::Conflict => Self {
                value: 0,
                missing: u32::from(counts_missing),
                conflict: counts_missing,
            },
        }
    }
}

/// How one application reads its site.
#[derive(Clone, Copy, Debug)]
struct SiteReading {
    /// The allele it scores.
    allele: SiteAllele,
    /// Whether its dose waits on every variant of the site: its (variant, score) scores the REF.
    whole_site: bool,
    /// Whether it counts its (variant, score)'s missing dose: the first of them does.
    counts_missing: bool,
}

/// One variant's copies of its ALT over the rows measuring it, so far.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Measured {
    /// No row has a call yet.
    Uncalled,
    /// Every called row gave these copies.
    Copies(u32),
    /// Called rows gave different copies.
    Disagreeing,
}

/// A site's contexts as the site rule reads them.
struct SiteRows<'p> {
    /// The variant each context measures.
    variants: &'p [usize],
    /// Whether each context's ALT is its allele 1: its REF is allele 2.
    alternate_is_allele1: &'p [bool],
}

/// The outcome of one application for `genotypes` on its `matching` contexts. A variant's copies
/// of its ALT is the one value its called rows agree on: its rows are measurements of one
/// quantity, and a missing call is no measurement. The site's REF copies are the ploidy, two on a
/// `.bed`, less every ALT's at the site; a variant's own REF copies are two less its ALT's.
fn resolve_outcome(
    exact: &ExactPlan,
    score_info: &ScoreInfo,
    reading: SiteReading,
    rows: &SiteRows<'_>,
    matching: &[usize],
    genotypes: &[u8],
    measured: &mut Vec<Measured>,
) -> Outcome {
    measured.clear();
    measured.resize(rows.variants.iter().max().map_or(0, |&last| last + 1), Measured::Uncalled);
    for (&context, &bits) in matching.iter().zip(genotypes) {
        let Some(allele1) = ALT_COPIES[usize::from(bits & 0b11)] else {
            continue;
        };
        let copies = if rows.alternate_is_allele1[context] { allele1 } else { 2 - allele1 };
        let slot = &mut measured[rows.variants[context]];
        *slot = match *slot {
            Measured::Uncalled => Measured::Copies(copies),
            Measured::Copies(earlier) if earlier == copies => Measured::Copies(copies),
            _ => Measured::Disagreeing,
        };
    }
    let variant = reading.allele.variant();
    let dose = if reading.whole_site {
        // Every variant of the site is among the contexts read.
        let mut alternates = 0u32;
        let mut missing = false;
        for &state in measured.iter() {
            match state {
                Measured::Disagreeing => return Outcome::Conflict,
                Measured::Uncalled => missing = true,
                Measured::Copies(copies) => alternates += copies,
            }
        }
        if missing {
            return Outcome::Missing;
        }
        let Some(reference) = 2u32.checked_sub(alternates) else {
            return Outcome::Conflict;
        };
        match (reading.allele, measured[variant]) {
            (SiteAllele::Reference(_), _) => reference,
            (SiteAllele::Alternate(_), Measured::Copies(copies)) => copies,
            (SiteAllele::Alternate(_), _) => return Outcome::Missing,
        }
    } else {
        match (reading.allele, measured[variant]) {
            (SiteAllele::Alternate(_), Measured::Copies(copies)) => copies,
            (SiteAllele::Reference(_), Measured::Copies(copies)) => 2 - copies,
            (_, Measured::Uncalled) => return Outcome::Missing,
            (_, Measured::Disagreeing) => return Outcome::Conflict,
        }
    };
    Outcome::Add(exact.complex_term(score_info.score_column_index.0, score_info.weight, dose))
}

/// One score row's view of its site: the contexts it reads and what every combination of
/// genotypes on them does.
struct ApplicationPlan {
    column: usize,
    /// Where the application's terms go in a person's lanes.
    target: Target,
    reading: SiteReading,
    /// The contexts it reads, in context order: its variant's rows, or every row of the site.
    matching: Vec<usize>,
    kind: ApplicationKind,
}

enum ApplicationKind {
    /// One context and nothing to report: each person's genotype on that context selects what
    /// is added, read straight from the row.
    Direct { values: [i128; 4], missing: [u32; 4] },
    /// Indexed by the packed genotype code over `matching`, two bits per context with the first
    /// context lowest.
    Table { entries: Vec<TableEntry> },
    /// Too many contexts to tabulate: resolved per person.
    PerPerson,
}

impl ApplicationKind {
    fn new(
        exact: &ExactPlan,
        score_info: &ScoreInfo,
        reading: SiteReading,
        rows: &SiteRows<'_>,
        matching: &[usize],
    ) -> Self {
        if matching.len() > TABULATED_MAX_CONTEXTS {
            return Self::PerPerson;
        }
        let mut measured = Vec::new();
        let mut entry = |genotypes: &[u8]| {
            let outcome = resolve_outcome(
                exact,
                score_info,
                reading,
                rows,
                matching,
                genotypes,
                &mut measured,
            );
            TableEntry::new(outcome, reading.counts_missing)
        };
        if matching.len() == 1 {
            let entries: [TableEntry; 4] = std::array::from_fn(|code| entry(&[code as u8]));
            if !entries.iter().any(|entry| entry.conflict) {
                return Self::Direct {
                    values: entries.map(|entry| entry.value),
                    missing: entries.map(|entry| entry.missing),
                };
            }
        }
        let mut genotypes = [0u8; TABULATED_MAX_CONTEXTS];
        let genotypes = &mut genotypes[..matching.len()];
        let entries = (0..1usize << (2 * matching.len()))
            .map(|code| {
                for (position, bits) in genotypes.iter_mut().enumerate() {
                    *bits = ((code >> (2 * position)) & 0b11) as u8;
                }
                entry(genotypes)
            })
            .collect();
        Self::Table { entries }
    }
}

struct RulePlan {
    /// Contexts decoded into genotypes for a block: first those tabulated and
    /// per-person applications read, then those only direct applications read.
    decoded_contexts: Vec<usize>,
    /// How many of `decoded_contexts` tabulated and per-person applications read.
    /// The rest are decoded only when some person's calls are forced missing.
    table_contexts: usize,
    /// The variant each context measures, by the site rule.
    row_variants: Vec<usize>,
    /// Whether each context's ALT is its allele 1.
    alternate_is_allele1: Vec<bool>,
    applications: Vec<ApplicationPlan>,
}

impl RulePlan {
    fn rows(&self) -> SiteRows<'_> {
        SiteRows {
            variants: &self.row_variants,
            alternate_is_allele1: &self.alternate_is_allele1,
        }
    }
}

impl RulePlan {
    /// The plan of `rule`, one site's rows and the score rows resolved per person there. The join
    /// built it by the site rule, so every score row names one allele of the site.
    fn new(exact: &ExactPlan, rule: &GroupedComplexRule) -> Result<Self, PipelineError> {
        let num_contexts = rule.possible_contexts.len();
        let rows: Vec<(usize, &str, &str)> = rule
            .possible_contexts
            .iter()
            .enumerate()
            .map(|(context, (_, allele1, allele2))| (context, allele2.as_str(), allele1.as_str()))
            .collect();
        let site = Site::new(&rows, rule.reference_declared);
        let alleles = rule
            .score_applications
            .iter()
            .map(|score_info| match site.match_row(&score_info.effect_allele, &score_info.other_allele) {
                RowMatch::Scores(allele) => Ok(allele),
                RowMatch::NoVariant | RowMatch::Several(..) | RowMatch::Unread(_) => Err(PipelineError::Compute(format!(
                    "A plan's complex rule at {}:{} holds a score row naming {} and {}, which name no one allele of its site.",
                    rule.locus_chr_pos.0, rule.locus_chr_pos.1, score_info.effect_allele, score_info.other_allele
                ))),
            })
            .collect::<Result<Vec<SiteAllele>, PipelineError>>()?;
        let row_variants = site.row_variants().to_vec();
        // A row whose first written allele, allele 2, is its REF has allele 1 as its ALT.
        let alternate_is_allele1 = site.first_is_reference().to_vec();
        let site_rows = SiteRows {
            variants: &row_variants,
            alternate_is_allele1: &alternate_is_allele1,
        };
        let group = |index: usize| {
            (
                alleles[index].variant(),
                rule.score_applications[index].score_column_index,
            )
        };
        let mut tabulated = vec![false; num_contexts];
        let mut direct = vec![false; num_contexts];
        let applications = rule
            .score_applications
            .iter()
            .enumerate()
            .map(|(index, score_info)| {
                let allele = alleles[index];
                let whole_site = (0..alleles.len()).any(|other| {
                    group(other) == group(index) && site.reads_whole_site(alleles[other])
                });
                let reading = SiteReading {
                    allele,
                    whole_site,
                    counts_missing: (0..index).all(|earlier| group(earlier) != group(index)),
                };
                let matching: Vec<usize> = (0..num_contexts)
                    .filter(|&context| whole_site || row_variants[context] == allele.variant())
                    .collect();
                let kind = ApplicationKind::new(exact, score_info, reading, &site_rows, &matching);
                let used = match kind {
                    ApplicationKind::Direct { .. } => &mut direct,
                    _ => &mut tabulated,
                };
                for &context in &matching {
                    used[context] = true;
                }
                ApplicationPlan {
                    column: score_info.score_column_index.0,
                    target: exact.complex_target(score_info.score_column_index.0, score_info.weight),
                    reading,
                    matching,
                    kind,
                }
            })
            .collect();
        let mut decoded_contexts: Vec<usize> = (0..num_contexts)
            .filter(|&context| tabulated[context])
            .collect();
        let table_contexts = decoded_contexts.len();
        decoded_contexts
            .extend((0..num_contexts).filter(|&context| direct[context] && !tabulated[context]));
        Ok(Self {
            decoded_contexts,
            table_contexts,
            row_variants,
            alternate_is_allele1,
            applications,
        })
    }
}

/// Conflicts from one block of people: per (rule, application) that counts its variant's missing
/// dose, the people whose calls there disagree.
#[derive(Default)]
struct BlockReport {
    conflicts: AHashMap<(usize, usize), u64>,
}

/// Everything a block needs to apply one group of rules.
struct GroupPass<'a> {
    rules: &'a [GroupedComplexRule],
    plans: &'a [RulePlan],
    /// Prefix sums of the rules' context counts.
    context_offsets: &'a [usize],
    group: Range<usize>,
    /// The scored span of every context of the group's rules, in rule then context order.
    rows: &'a [&'a [u8]],
    max_contexts: usize,
    layout: &'a PersonLayout,
    exact: &'a ExactPlan,
    stride: usize,
    num_scores: usize,
}

/// What a direct application adds to one block of people.
struct DirectApplication<'a> {
    values: &'a [i128; 4],
    missing: &'a [u32; 4],
    exact: &'a ExactPlan,
    target: Target,
    column: usize,
    stride: usize,
    num_scores: usize,
}

impl DirectApplication<'_> {
    /// Adds the outcome of one person's genotype code (its low two bits).
    #[inline(always)]
    fn add(&self, person: usize, code: u8, scores: &mut [i64], counts: &mut [u32]) {
        let code = usize::from(code & 0b11);
        self.exact.add(
            self.target,
            self.values[code],
            &mut scores[person * self.stride..(person + 1) * self.stride],
        );
        counts[person * self.num_scores + self.column] += self.missing[code];
    }

    fn add_decoded(&self, codes: &[u8], scores: &mut [i64], counts: &mut [u32]) {
        for (person, &code) in codes.iter().enumerate() {
            self.add(person, code, scores, counts);
        }
    }

    /// For people who occupy consecutive two-bit slots of the row, the first of them at
    /// slot `first_slot`: whole bytes are read four people at a time.
    fn add_packed(&self, row: &[u8], first_slot: usize, scores: &mut [i64], counts: &mut [u32]) {
        let num_people = counts.len() / self.num_scores;
        let slot = |person: usize| {
            let position = first_slot + person;
            row[position / 4] >> (2 * (position % 4))
        };
        let head = ((4 - first_slot % 4) % 4).min(num_people);
        let body_bytes = (num_people - head) / 4;
        for person in 0..head {
            self.add(person, slot(person), scores, counts);
        }
        let first_byte = (first_slot + head) / 4;
        for (index, &byte) in row[first_byte..first_byte + body_bytes].iter().enumerate() {
            let person = head + 4 * index;
            self.add(person, byte, scores, counts);
            self.add(person + 1, byte >> 2, scores, counts);
            self.add(person + 2, byte >> 4, scores, counts);
            self.add(person + 3, byte >> 6, scores, counts);
        }
        for person in head + 4 * body_bytes..num_people {
            self.add(person, slot(person), scores, counts);
        }
    }

    fn add_gathered(
        &self,
        row: &[u8],
        bytes: &[u32],
        shifts: &[u8],
        scores: &mut [i64],
        counts: &mut [u32],
    ) {
        for (person, (&byte, &shift)) in bytes.iter().zip(shifts).enumerate() {
            self.add(person, row[byte as usize] >> shift, scores, counts);
        }
    }
}

/// Applies a group's rules to one block of people. Rules run in order, and each
/// rule's applications in order, so every accumulator receives the same additions
/// in the same order as under the person-major resolver, and ends bit-identical.
fn evaluate_block(
    pass: &GroupPass,
    first_person: usize,
    scores: &mut [i64],
    counts: &mut [u32],
) -> BlockReport {
    let num_scores = pass.num_scores;
    let num_people = counts.len() / num_scores;
    let people = first_person..first_person + num_people;
    let bytes = &pass.layout.bytes[people.clone()];
    let shifts = &pass.layout.shifts[people.clone()];
    let forced_missing = {
        let all = &pass.layout.forced_missing;
        &all[all.partition_point(|&person| person < people.start)
            ..all.partition_point(|&person| person < people.end)]
    };
    // Direct applications read rows in place, unless some person's calls are forced
    // missing: then they read decoded genotypes, as tabulated applications do.
    let rows_in_place = forced_missing.is_empty();
    let first_context = pass.context_offsets[pass.group.start];
    let mut genotypes = vec![0u8; pass.max_contexts * num_people];
    let mut code_buffer = vec![0u8; num_people];
    let mut tuple = Vec::new();
    let mut measured = Vec::new();
    let mut report = BlockReport::default();

    for rule_idx in pass.group.clone() {
        let rule = &pass.rules[rule_idx];
        let plan = &pass.plans[rule_idx];
        let rows = &pass.rows[pass.context_offsets[rule_idx] - first_context
            ..pass.context_offsets[rule_idx + 1] - first_context];
        let decoded_contexts = if rows_in_place {
            &plan.decoded_contexts[..plan.table_contexts]
        } else {
            &plan.decoded_contexts[..]
        };
        for &context in decoded_contexts {
            let decoded = &mut genotypes[context * num_people..(context + 1) * num_people];
            decode_genotypes(rows[context], bytes, shifts, decoded);
            for &person in forced_missing {
                decoded[person - first_person] = 0b01;
            }
        }

        for (application_idx, application) in plan.applications.iter().enumerate() {
            let column = application.column;
            let mut conflicts = 0u64;
            let entries = match &application.kind {
                ApplicationKind::Direct { values, missing } => {
                    let context = application.matching[0];
                    let direct = DirectApplication {
                        values,
                        missing,
                        exact: pass.exact,
                        target: application.target,
                        column,
                        stride: pass.stride,
                        num_scores,
                    };
                    if !rows_in_place {
                        let decoded = &genotypes[context * num_people..(context + 1) * num_people];
                        direct.add_decoded(decoded, scores, counts);
                    } else if pass.layout.contiguous {
                        direct.add_packed(rows[context], first_person, scores, counts);
                    } else {
                        direct.add_gathered(rows[context], bytes, shifts, scores, counts);
                    }
                    continue;
                }
                ApplicationKind::Table { entries } => entries,
                ApplicationKind::PerPerson => {
                    let score_info = &rule.score_applications[application_idx];
                    let people_rows = scores
                        .chunks_exact_mut(pass.stride)
                        .zip(counts.chunks_exact_mut(num_scores))
                        .enumerate();
                    for (person, (person_scores, person_counts)) in people_rows {
                        tuple.clear();
                        tuple.extend(
                            application
                                .matching
                                .iter()
                                .map(|&context| genotypes[context * num_people + person]),
                        );
                        let outcome = resolve_outcome(
                            pass.exact,
                            score_info,
                            application.reading,
                            &plan.rows(),
                            &application.matching,
                            &tuple,
                            &mut measured,
                        );
                        let entry = TableEntry::new(outcome, application.reading.counts_missing);
                        pass.exact.add(application.target, entry.value, person_scores);
                        person_counts[column] += entry.missing;
                        conflicts += u64::from(entry.conflict);
                    }
                    if conflicts > 0 {
                        *report.conflicts.entry((rule_idx, application_idx)).or_default() += conflicts;
                    }
                    continue;
                }
            };

            let codes: &[u8] = match application.matching.as_slice() {
                [context] => &genotypes[context * num_people..(context + 1) * num_people],
                matching => {
                    code_buffer.fill(0);
                    for (position, &context) in matching.iter().enumerate() {
                        let decoded = &genotypes[context * num_people..(context + 1) * num_people];
                        for (code, &bits) in code_buffer.iter_mut().zip(decoded) {
                            *code |= bits << (2 * position);
                        }
                    }
                    &code_buffer
                }
            };
            let people_rows = scores
                .chunks_exact_mut(pass.stride)
                .zip(counts.chunks_exact_mut(num_scores));
            for ((person_scores, person_counts), &code) in people_rows.zip(codes) {
                let entry = entries[code as usize];
                pass.exact.add(application.target, entry.value, person_scores);
                person_counts[column] += entry.missing;
                conflicts += u64::from(entry.conflict);
            }
            if conflicts > 0 {
                *report.conflicts.entry((rule_idx, application_idx)).or_default() += conflicts;
            }
        }
    }
    report
}

/// Block size and read budget for one resolution.
#[derive(Clone, Copy)]
struct ResolveLimits {
    block_people: usize,
    streamed_group_bytes: usize,
}

impl ResolveLimits {
    fn for_people(num_people: usize) -> Self {
        let threads = rayon::current_num_threads().max(1);
        Self {
            block_people: num_people
                .div_ceil(threads * 4)
                .clamp(MIN_BLOCK_PEOPLE, MAX_BLOCK_PEOPLE),
            streamed_group_bytes: STREAMED_GROUP_BYTES,
        }
    }
}

/// Splits the rules into runs whose rows fit a read budget. Every run holds at
/// least one rule.
fn streamed_groups(context_offsets: &[usize], span_len: usize, budget: usize) -> Vec<Range<usize>> {
    let num_rules = context_offsets.len() - 1;
    let mut groups = Vec::new();
    let mut start = 0;
    let mut group_bytes = 0usize;
    for rule in 0..num_rules {
        let rule_bytes = (context_offsets[rule + 1] - context_offsets[rule]).saturating_mul(span_len);
        if rule > start && group_bytes.saturating_add(rule_bytes) > budget {
            groups.push(start..rule);
            start = rule;
            group_bytes = 0;
        }
        group_bytes = group_bytes.saturating_add(rule_bytes);
    }
    groups.push(start..num_rules);
    groups
}

/// People whose calls disagreed, per (rule, application) that counts its variant's missing dose,
/// in rule then application order.
struct ResolutionReport {
    conflicts: Vec<((usize, usize), u64)>,
}

/// The row-major resolver. Each rule context's row is located once and its scored
/// span borrowed from the memory map, or read once per group of rules when there
/// is no map. People are processed in parallel blocks; inside a block each rule
/// decodes its rows for the block and applies per-rule outcome tables, so no
/// genotype pays for a source dispatch, a fileset search or a hash lookup.
fn resolve_rows(
    resolver: &ComplexVariantResolver,
    prep_result: &PreparationResult,
    final_scores: &mut [i64],
    final_missing_counts: &mut [u32],
    limits: ResolveLimits,
    pb: &ProgressBar,
) -> Result<ResolutionReport, PipelineError> {
    let rules = &prep_result.complex_rules;
    let num_scores = prep_result.score_names.len();
    let exact = prep_result.exact();
    let stride = exact.stride();
    let num_people = final_scores
        .len()
        .checked_div(stride)
        .unwrap_or(0)
        .min(final_missing_counts.len().checked_div(num_scores).unwrap_or(0));
    let mut conflicts: AHashMap<(usize, usize), u64> = AHashMap::new();
    if num_people == 0 || rules.is_empty() {
        return Ok(ResolutionReport {
            conflicts: Vec::new(),
        });
    }

    let layout = PersonLayout::new(resolver, prep_result, num_people)?;
    let mut context_offsets = Vec::with_capacity(rules.len() + 1);
    let mut locations =
        Vec::with_capacity(rules.iter().map(|rule| rule.possible_contexts.len()).sum());
    context_offsets.push(0);
    for rule in rules {
        for (bim_row_index, _, _) in &rule.possible_contexts {
            locations.push(
                resolver
                    .locate_row(prep_result.bytes_per_variant, *bim_row_index)
                    .map_err(fetch_error)?,
            );
        }
        context_offsets.push(locations.len());
    }

    let plans: Vec<RulePlan> = rules
        .par_iter()
        .map(|rule| RulePlan::new(exact, rule))
        .collect::<Result<_, _>>()?;

    let mapped = resolver.is_mapped();
    let groups = if mapped {
        vec![0..rules.len()]
    } else {
        streamed_groups(
            &context_offsets,
            layout.span.len,
            limits.streamed_group_bytes,
        )
    };
    pb.set_length((num_people * groups.len()) as u64);

    let mut storage = Vec::<u8>::new();
    let people_scores = &mut final_scores[..num_people * stride];
    let people_counts = &mut final_missing_counts[..num_people * num_scores];
    let block_lanes = limits.block_people * stride;
    let block_cells = limits.block_people * num_scores;

    for group in groups {
        let group_locations = &locations[context_offsets[group.start]..context_offsets[group.end]];
        let rows: Vec<&[u8]> = if mapped {
            group_locations
                .iter()
                .map(|location| resolver.mapped_span(location, layout.span).map_err(fetch_error))
                .collect::<Result<_, _>>()?
        } else if layout.span.len == 0 {
            vec![&[][..]; group_locations.len()]
        } else {
            storage.clear();
            storage.resize(group_locations.len() * layout.span.len, 0);
            storage
                .par_chunks_mut(layout.span.len)
                .zip(group_locations.par_iter())
                .try_for_each(|(dst, location)| {
                    resolver
                        .read_span(location, layout.span, dst)
                        .map_err(fetch_error)
                })?;
            storage.chunks_exact(layout.span.len).collect()
        };

        let pass = GroupPass {
            rules,
            plans: &plans,
            context_offsets: &context_offsets,
            max_contexts: group
                .clone()
                .map(|rule| rules[rule].possible_contexts.len())
                .max()
                .unwrap_or(0),
            group,
            rows: &rows,
            layout: &layout,
            exact,
            stride,
            num_scores,
        };
        let block_reports: Vec<BlockReport> = people_scores
            .par_chunks_mut(block_lanes)
            .zip(people_counts.par_chunks_mut(block_cells))
            .enumerate()
            .map(|(block, (scores, counts))| {
                let block_report =
                    evaluate_block(&pass, block * limits.block_people, scores, counts);
                pb.inc((counts.len() / num_scores) as u64);
                block_report
            })
            .collect();
        for block_report in block_reports {
            for (key, count) in block_report.conflicts {
                *conflicts.entry(key).or_default() += count;
            }
        }
    }

    let mut conflicts: Vec<((usize, usize), u64)> = conflicts.into_iter().collect();
    conflicts.sort_unstable();
    Ok(ResolutionReport { conflicts })
}

// The "slow path" resolver for complex variants.
///
/// This function runs *after* the main high-performance pipeline is complete and
/// adds every person's score contributions at the sites whose doses read several
/// rows: a variant measured by more than one row, and the REF of a site with several
/// variants. It is row-major (see `resolve_rows`), and the progress bar advances as
/// blocks of people finish.
pub fn resolve_complex_variants(
    resolver: &ComplexVariantResolver,
    prep_result: &Arc<PreparationResult>,
    final_scores: &mut [i64],
    final_missing_counts: &mut [u32],
) -> Result<(), PipelineError> {
    let num_rules = prep_result.complex_rules.len();
    if num_rules == 0 {
        return Ok(());
    }

    eprintln!("> Resolving {num_rules} sites whose doses read several rows...");

    let pb = ProgressBar::new(prep_result.num_people_to_score as u64);
    let progress_style = ProgressStyle::with_template(
        "> Resolving complex variants [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
    )
    .expect("Internal Error: Invalid progress bar template string.");
    pb.set_style(progress_style.progress_chars("█▉▊▋▌▍▎▏ "));

    let resolution = resolve_rows(
        resolver,
        prep_result,
        final_scores,
        final_missing_counts,
        ResolveLimits::for_people(prep_result.num_people_to_score),
        &pb,
    );

    pb.finish_with_message("Done.");

    let report = resolution?;
    if let Some(text) = conflict_report(&report, prep_result) {
        eprint!("{text}");
    }
    eprintln!("> Complex variant resolution complete.");
    Ok(())
}

/// The warning for doses left missing because calls disagreed, naming the first loci. None when
/// every call agreed.
fn conflict_report(report: &ResolutionReport, prep_result: &PreparationResult) -> Option<String> {
    use std::fmt::Write;
    if report.conflicts.is_empty() {
        return None;
    }
    let total: u64 = report.conflicts.iter().map(|(_, count)| count).sum();
    let mut text = String::new();
    writeln!(
        text,
        "> Warning: {total} dose(s) at {} (variant, score) pair(s) are missing because the calls they need disagree: rows measuring one variant give different copies, or the ALTs' copies at a site pass the ploidy. Examples:",
        report.conflicts.len()
    )
    .unwrap();
    for &((rule_idx, application_idx), count) in report.conflicts.iter().take(MAX_CONFLICT_EXAMPLES) {
        let rule = &prep_result.complex_rules[rule_idx];
        let score_info = &rule.score_applications[application_idx];
        writeln!(
            text,
            ">   - {}:{} effect allele {} (other {}) in score '{}': {count} people",
            rule.locus_chr_pos.0,
            rule.locus_chr_pos.1,
            score_info.effect_allele,
            score_info.other_allele,
            prep_result.score_names[score_info.score_column_index.0]
        )
        .unwrap();
    }
    Some(text)
}

