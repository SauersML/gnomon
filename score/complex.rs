use crate::pipeline_error::PipelineError;
use crate::score::io::BedSource;
use crate::score::types::{
    BimRowIndex, FilesetBoundary, GroupedComplexRule, OutputPersonIndex, PreparationResult,
    ScoreInfo,
};
use ahash::{AHashMap, AHashSet};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
        let bytes = row_bytes
            .into_iter()
            .map(|byte| byte.map_or(0, |byte| byte - span.start as u32))
            .collect();
        Ok(Self {
            bytes,
            shifts,
            forced_missing,
            span,
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

// ========================================================================================
//                            Complex Variant Resolution Types
// ========================================================================================

// A type alias for the final, merged collector.
pub type FinalAggregatedCollector = HashMap<Heuristic, (u64, Vec<CriticalIntegrityWarningInfo>)>;

/// An enum representing the complete, ordered set of resolution strategies.
/// This approach uses static dispatch for zero-cost abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Heuristic {
    /// Tries to find one BIM entry that perfectly matches both score file alleles.
    ExactScoreAlleleMatch,
    /// Tries to find one interpretation composed ONLY of score file alleles.
    PrioritizeUnambiguousGenotype,
    /// Prefers an interpretation where allele lengths match the score file.
    PreferMatchingAlleleStructure,
    /// Checks if all conflicting interpretations result in the same dosage.
    ConsistentDosage,
    /// As a last resort, prefers a single heterozygous call over homozygous ones.
    PreferHeterozygous,
    /// Infers a heterozygous genotype if conflicting homozygous calls involve alleles
    /// where one is a prefix of the other (e.g. C/C vs CAGA/CAGA -> C/CAGA).
    IndelAnchorBase,
    /// Final fallback: if both homozygous states conflict (00 and 11), infer a
    /// heterozygous genotype from the observed homozygous alleles.
    FallbackOpposingHomozygousAsHet,
    /// Absolute last fallback: average dosage across all remaining conflicts.
    FallbackAverageDosageAcrossConflicts,
}

/// Describes the specific heuristic used to resolve a critical data ambiguity,
/// holding the data needed for transparent reporting.
#[derive(Debug, Clone)]
pub enum ResolutionMethod {
    /// All conflicting sources yielded the same effect allele dosage.
    ConsistentDosage {
        dosage: f64,
    },
    /// Exactly one heterozygous call was found alongside one or more homozygous
    /// calls, and the heterozygous call was chosen.
    PreferHeterozygous {
        chosen_dosage: f64,
    },
    /// A single BIM entry's alleles perfectly matched the score file alleles.
    ExactScoreAlleleMatch {
        chosen_dosage: f64,
    },
    /// A single interpretation was composed of standard alleles from the score file.
    PrioritizeUnambiguousGenotype {
        chosen_dosage: f64,
    },
    PreferMatchingAlleleStructure {
        chosen_dosage: f64,
    },
    IndelAnchorBase {
        chosen_dosage: f64,
    },
    FallbackOpposingHomozygousAsHet {
        chosen_dosage: f64,
    },
    FallbackAverageDosageAcrossConflicts {
        chosen_dosage: f64,
    },
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
        PreparationResult::new(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![0],
            1,
            vec![0.0; num_scores],
            Vec::new(),
            rules,
            (0..num_scores).map(|i| format!("S{i}")).collect(),
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

    #[test]
    fn heuristic_table_is_indexed_by_discriminant() {
        for (index, method) in HEURISTICS.iter().enumerate() {
            assert_eq!(*method as usize, index);
        }
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

    /// Random packed genotypes (missing calls and padding bits included) and random
    /// rules over a small allele vocabulary, so contexts collide, duplicate, swap and
    /// share prefixes often enough to reach every heuristic.
    struct Scenario {
        total_people: usize,
        rows: Vec<Vec<u8>>,
        rules: Vec<GroupedComplexRule>,
        kept: Vec<u32>,
        num_scores: usize,
    }

    impl Scenario {
        fn random(seed: u64, total_people: usize, keep_all: bool) -> Self {
            const ALLELES: [&str; 6] = ["A", "G", "C", "T", "CA", "CAGA"];
            let mut rng = SplitMix64(seed);
            let num_variants = 40;
            let rows = (0..num_variants)
                .map(|_| {
                    (0..total_people.div_ceil(4))
                        .map(|_| rng.next() as u8)
                        .collect()
                })
                .collect();
            let num_scores = 1 + rng.below(4);
            let mut rules = Vec::new();
            for rule_idx in 0..60 {
                let num_contexts = 1 + rng.below(6);
                let possible_contexts: Vec<(BimRowIndex, String, String)> = (0..num_contexts)
                    .map(|_| {
                        (
                            BimRowIndex(rng.below(num_variants) as u64),
                            ALLELES[rng.below(ALLELES.len())].to_string(),
                            ALLELES[rng.below(ALLELES.len())].to_string(),
                        )
                    })
                    .collect();
                let score_applications = (0..1 + rng.below(4))
                    .map(|_| {
                        let (effect_allele, other_allele) = if rng.below(4) != 0 {
                            let (_, a1, a2) = &possible_contexts[rng.below(num_contexts)];
                            if rng.below(2) == 0 {
                                (a1.clone(), a2.clone())
                            } else {
                                (a2.clone(), a1.clone())
                            }
                        } else {
                            (
                                ALLELES[rng.below(ALLELES.len())].to_string(),
                                ALLELES[rng.below(ALLELES.len())].to_string(),
                            )
                        };
                        ScoreInfo {
                            effect_allele,
                            other_allele,
                            weight: (rng.unit() * 3.0 - 1.5) as f32,
                            score_column_index: ScoreColumnIndex(rng.below(num_scores)),
                        }
                    })
                    .collect();
                rules.push(GroupedComplexRule {
                    locus_chr_pos: ("22".to_string(), 1000 + rule_idx),
                    possible_contexts,
                    score_applications,
                });
            }
            // A locus with more duplicate contexts than a table covers.
            rules.push(GroupedComplexRule {
                locus_chr_pos: ("22".to_string(), 5000),
                possible_contexts: (0..6)
                    .map(|variant| (BimRowIndex(variant), "A".to_string(), "G".to_string()))
                    .collect(),
                score_applications: vec![ScoreInfo {
                    effect_allele: "G".to_string(),
                    other_allele: "A".to_string(),
                    weight: 0.7,
                    score_column_index: ScoreColumnIndex(0),
                }],
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

        /// Accumulators as the fast path might leave them, signed zeros included.
        fn initial_accumulators(&self, seed: u64) -> (Vec<f64>, Vec<u32>) {
            let mut rng = SplitMix64(seed ^ 0xA5A5);
            let cells = self.kept.len() * self.num_scores;
            let scores = (0..cells)
                .map(|_| match rng.below(5) {
                    0 => -0.0,
                    1 => 0.0,
                    _ => rng.unit() * 1e3 - 500.0,
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

    /// The person-major resolver this module replaced, reduced to its arithmetic:
    /// every person, every rule, every application, in that order.
    fn reference_resolve(
        scenario: &Scenario,
        prep_result: &PreparationResult,
        scores: &mut [f64],
        counts: &mut [u32],
    ) -> FinalAggregatedCollector {
        let pipeline = ResolverPipeline::new();
        let num_scores = prep_result.score_names.len();
        let mut collector = FinalAggregatedCollector::new();
        for (person, fam_idx) in prep_result.output_idx_to_fam_idx.iter().enumerate() {
            for rule in &prep_result.complex_rules {
                let valid: Vec<(u8, &(BimRowIndex, String, String))> = rule
                    .possible_contexts
                    .iter()
                    .map(|context| (scenario.genotype(context.0, fam_idx.0 as usize), context))
                    .filter(|(bits, _)| *bits != 0b01)
                    .collect();
                for score_info in &rule.score_applications {
                    let cell = person * num_scores + score_info.score_column_index.0;
                    let matching: Vec<_> = valid
                        .iter()
                        .copied()
                        .filter(|(_, context)| {
                            score_allele_pair_matches(score_info, &context.1, &context.2)
                        })
                        .collect();
                    if matching.is_empty() {
                        counts[cell] += 1;
                        continue;
                    }
                    if matching.len() == 1 {
                        let (bits, (_, bim_a1, bim_a2)) = matching[0];
                        match Heuristic::calculate_score_dosage(bits, bim_a1, bim_a2, score_info) {
                            Some(dosage) => scores[cell] += dosage * score_info.weight as f64,
                            None => counts[cell] += 1,
                        }
                        continue;
                    }
                    let context = ResolutionContext {
                        score_info,
                        conflicting_interpretations: &matching,
                    };
                    let resolution = pipeline
                        .resolve(&context)
                        .expect("the average fallback resolves every conflict");
                    scores[cell] += resolution.chosen_dosage * score_info.weight as f64;
                    let (count, samples) = collector
                        .entry(resolution.method_used)
                        .or_insert((0, Vec::new()));
                    *count += 1;
                    if samples.len() < MAX_WARNING_SAMPLES {
                        samples.push(CriticalIntegrityWarningInfo {
                            iid: prep_result.final_person_iids[person].clone(),
                            locus_chr_pos: rule.locus_chr_pos.clone(),
                            score_name: prep_result.score_names[score_info.score_column_index.0]
                                .clone(),
                            conflicts: matching
                                .iter()
                                .map(|(bits, context)| ConflictSource {
                                    bim_row: context.0,
                                    alleles: (context.1.clone(), context.2.clone()),
                                    genotype_bits: *bits,
                                })
                                .collect(),
                            resolution_method: resolution_method(
                                resolution.method_used,
                                resolution.chosen_dosage,
                            ),
                            score_effect_allele: score_info.effect_allele.clone(),
                            score_other_allele: score_info.other_allele.clone(),
                        });
                    }
                }
            }
        }
        collector
    }

    fn rendered(collector: &FinalAggregatedCollector) -> Vec<String> {
        let mut categories: Vec<String> = collector
            .iter()
            .map(|(method, (count, samples))| {
                let samples: Vec<String> = samples
                    .iter()
                    .map(format_critical_integrity_warning)
                    .collect();
                format!("{method:?} {count}\n{}", samples.join("\n---\n"))
            })
            .collect();
        categories.sort();
        categories
    }

    #[test]
    fn row_major_resolver_matches_person_major_reference() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut methods_seen = AHashSet::new();
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
            let (initial_scores, initial_counts) = scenario.initial_accumulators(seed);
            let mut expected_scores = initial_scores.clone();
            let mut expected_counts = initial_counts.clone();
            let expected_warnings = reference_resolve(
                &scenario,
                &prep_result,
                &mut expected_scores,
                &mut expected_counts,
            );
            methods_seen.extend(expected_warnings.keys().copied());

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
                    assert!(report.unresolvable.is_none(), "{context}");
                    assert_eq!(
                        scores.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        expected_scores
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "{context}"
                    );
                    assert_eq!(counts, expected_counts, "{context}");
                    assert_eq!(
                        rendered(&report.warnings),
                        rendered(&expected_warnings),
                        "{context}"
                    );
                }
            }
        }
        // The scenarios must reach the heuristic chain, not only single interpretations.
        assert!(methods_seen.len() >= 3, "heuristics reached: {methods_seen:?}");
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
        let (mut scores, mut counts) = scenario.initial_accumulators(11);
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
        }];
        let prep_result = test_prep_result(rules, 10, &scenario.kept, 1, last_variant + 1);
        let mut scores = vec![0.0; scenario.kept.len()];
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

    #[test]
    fn resolver_pipeline_uses_heterozygous_fallback_last() {
        let score_info = ScoreInfo {
            effect_allele: "T".to_string(),
            other_allele: "A".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        let ctx_a = (BimRowIndex(10), "A".to_string(), "T".to_string());
        let ctx_b = (BimRowIndex(11), "A".to_string(), "T".to_string());
        let conflicting_interpretations = vec![(0b00, &ctx_a), (0b11, &ctx_b)];
        let context = ResolutionContext {
            score_info: &score_info,
            conflicting_interpretations: &conflicting_interpretations,
        };

        let pipeline = ResolverPipeline::new();
        let resolution = pipeline
            .resolve(&context)
            .expect("fallback heuristic should resolve remaining ambiguity");

        assert_eq!(
            resolution.method_used,
            Heuristic::FallbackOpposingHomozygousAsHet
        );
        assert!((resolution.chosen_dosage - 1.0).abs() < 1e-9);
    }

    #[test]
    fn resolver_pipeline_preserves_existing_priority_over_fallback() {
        let score_info = ScoreInfo {
            effect_allele: "A".to_string(),
            other_allele: "G".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        let exact_ctx = (BimRowIndex(20), "A".to_string(), "G".to_string());
        let non_exact_ctx = (BimRowIndex(21), "A".to_string(), "T".to_string());
        let conflicting_interpretations = vec![(0b10, &exact_ctx), (0b11, &non_exact_ctx)];
        let context = ResolutionContext {
            score_info: &score_info,
            conflicting_interpretations: &conflicting_interpretations,
        };

        let pipeline = ResolverPipeline::new();
        let resolution = pipeline
            .resolve(&context)
            .expect("exact match heuristic should resolve ambiguity first");

        assert_eq!(resolution.method_used, Heuristic::ExactScoreAlleleMatch);
    }

    #[test]
    fn fallback_het_requires_both_homozygous_states() {
        let score_info = ScoreInfo {
            effect_allele: "A".to_string(),
            other_allele: "G".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        let ctx_a = (BimRowIndex(30), "A".to_string(), "C".to_string());
        let ctx_b = (BimRowIndex(31), "A".to_string(), "T".to_string());
        let conflicting_interpretations = vec![(0b00, &ctx_a), (0b10, &ctx_b)];
        let context = ResolutionContext {
            score_info: &score_info,
            conflicting_interpretations: &conflicting_interpretations,
        };

        let resolution = Heuristic::FallbackOpposingHomozygousAsHet.try_resolve(&context);
        assert!(resolution.is_none());
    }

    #[test]
    fn fallback_het_dosage_matches_inferred_allele_pair() {
        let ctx_a = (BimRowIndex(40), "C".to_string(), "T".to_string());
        let ctx_b = (BimRowIndex(41), "C".to_string(), "T".to_string());
        let conflicting_interpretations = vec![(0b00, &ctx_a), (0b11, &ctx_b)];

        let incompatible_score_info = ScoreInfo {
            effect_allele: "G".to_string(),
            other_allele: "A".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        let incompatible_context = ResolutionContext {
            score_info: &incompatible_score_info,
            conflicting_interpretations: &conflicting_interpretations,
        };
        assert!(
            Heuristic::FallbackOpposingHomozygousAsHet
                .try_resolve(&incompatible_context)
                .is_none()
        );

        let score_info_one = ScoreInfo {
            effect_allele: "C".to_string(),
            other_allele: "T".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        let context_one = ResolutionContext {
            score_info: &score_info_one,
            conflicting_interpretations: &conflicting_interpretations,
        };
        let res_one = Heuristic::FallbackOpposingHomozygousAsHet
            .try_resolve(&context_one)
            .expect("fallback should resolve");
        assert!((res_one.chosen_dosage - 1.0).abs() < 1e-9);

        let score_info_t = ScoreInfo {
            effect_allele: "T".to_string(),
            other_allele: "C".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        let context_t = ResolutionContext {
            score_info: &score_info_t,
            conflicting_interpretations: &conflicting_interpretations,
        };
        let res_t = Heuristic::FallbackOpposingHomozygousAsHet
            .try_resolve(&context_t)
            .expect("fallback should resolve");
        assert!((res_t.chosen_dosage - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fallback_average_dosage_supports_three_plus_conflicts() {
        let score_info = ScoreInfo {
            effect_allele: "T".to_string(),
            other_allele: "A".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        let c1 = (BimRowIndex(50), "A".to_string(), "T".to_string()); // 00 => 0
        let c2 = (BimRowIndex(51), "A".to_string(), "T".to_string()); // 10 => 1
        let c3 = (BimRowIndex(52), "A".to_string(), "T".to_string()); // 11 => 2
        let conflicting_interpretations = vec![(0b00, &c1), (0b10, &c2), (0b11, &c3)];
        let context = ResolutionContext {
            score_info: &score_info,
            conflicting_interpretations: &conflicting_interpretations,
        };

        let resolution = Heuristic::FallbackAverageDosageAcrossConflicts
            .try_resolve(&context)
            .expect("average fallback should resolve");
        assert!((resolution.chosen_dosage - 1.0).abs() < 1e-9);
    }

    #[test]
    fn resolver_pipeline_uses_average_only_as_absolute_last_resort() {
        let score_info = ScoreInfo {
            effect_allele: "T".to_string(),
            other_allele: "A".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        // 00 + 11 should be consumed by the earlier opposing-homozygous fallback,
        // not by the average fallback.
        let c1 = (BimRowIndex(60), "A".to_string(), "T".to_string());
        let c2 = (BimRowIndex(61), "A".to_string(), "T".to_string());
        let conflicting_interpretations = vec![(0b00, &c1), (0b11, &c2)];
        let context = ResolutionContext {
            score_info: &score_info,
            conflicting_interpretations: &conflicting_interpretations,
        };

        let pipeline = ResolverPipeline::new();
        let resolution = pipeline
            .resolve(&context)
            .expect("pipeline should resolve with prior fallback");
        assert_eq!(
            resolution.method_used,
            Heuristic::FallbackOpposingHomozygousAsHet
        );
    }

    #[test]
    fn fallback_average_dosage_handles_two_conflicts() {
        let score_info = ScoreInfo {
            effect_allele: "T".to_string(),
            other_allele: "A".to_string(),
            weight: 1.0,
            score_column_index: ScoreColumnIndex(0),
        };
        // Crafted to avoid all earlier heuristics:
        // - no exact score allele match
        // - no unambiguous genotype from score allele set
        // - not a single heterozygous-vs-homozygous tie
        // - not opposing 00/11 fallback
        // Dosages are 0 (00 with A/T) and 2 (00 with T/A), average = 1.0.
        let c1 = (BimRowIndex(70), "A".to_string(), "T".to_string());
        let c2 = (BimRowIndex(71), "T".to_string(), "A".to_string());
        let conflicting_interpretations = vec![(0b00, &c1), (0b00, &c2)];
        let context = ResolutionContext {
            score_info: &score_info,
            conflicting_interpretations: &conflicting_interpretations,
        };

        let pipeline = ResolverPipeline::new();
        let resolution = pipeline
            .resolve(&context)
            .expect("pipeline should resolve with average fallback");
        assert_eq!(
            resolution.method_used,
            Heuristic::FallbackAverageDosageAcrossConflicts
        );
        assert!((resolution.chosen_dosage - 1.0).abs() < 1e-9);
    }
}

/// A private struct holding the raw data for one conflicting source of evidence.
/// This is used exclusively for building the final fatal error report.
#[derive(Debug, Clone)]
pub struct ConflictSource {
    pub bim_row: BimRowIndex,
    pub alleles: (String, String),
    pub genotype_bits: u8,
}

/// A private struct holding the data for a critical but non-fatal integrity warning.
/// This is used when multiple data sources conflict but lead to a consistent outcome.
#[derive(Debug, Clone)]
pub struct CriticalIntegrityWarningInfo {
    pub iid: String,
    pub locus_chr_pos: (String, u32),
    pub score_name: String,
    pub conflicts: Vec<ConflictSource>,
    pub resolution_method: ResolutionMethod,
    pub score_effect_allele: String,
    pub score_other_allele: String,
}
//========================================================================================
//
//                      The Zero-Cost Heuristic Pipeline
//
//========================================================================================

/// The data required for any heuristic to make a decision.
/// It is created once per conflict and passed down the chain.
pub struct ResolutionContext<'a> {
    pub score_info: &'a ScoreInfo,
    pub conflicting_interpretations: &'a [(u8, &'a (BimRowIndex, String, String))],
}

/// The successful outcome of a resolution, specifying the dosage and the rule that won.
pub struct Resolution {
    pub chosen_dosage: f64,
    pub method_used: Heuristic,
}

#[inline(always)]
fn score_allele_pair_matches(score_info: &ScoreInfo, bim_a1: &str, bim_a2: &str) -> bool {
    (score_info.effect_allele == bim_a1 && score_info.other_allele == bim_a2)
        || (score_info.effect_allele == bim_a2 && score_info.other_allele == bim_a1)
}

impl Heuristic {
    /// The main dispatcher for the enum. It calls the appropriate private method
    /// for the specific heuristic variant.
    pub fn try_resolve(&self, context: &ResolutionContext) -> Option<Resolution> {
        match self {
            Heuristic::ExactScoreAlleleMatch => self.resolve_exact_match(context),
            Heuristic::PrioritizeUnambiguousGenotype => self.resolve_unambiguous_genotype(context),
            Heuristic::PreferMatchingAlleleStructure => {
                self.resolve_prefer_matching_allele_structure(context)
            }
            Heuristic::ConsistentDosage => self.resolve_consistent_dosage(context),
            Heuristic::PreferHeterozygous => self.resolve_prefer_het(context),
            Heuristic::IndelAnchorBase => self.resolve_indel_anchor_base(context),
            Heuristic::FallbackOpposingHomozygousAsHet => {
                self.resolve_fallback_opposing_homozygous_as_het(context)
            }
            Heuristic::FallbackAverageDosageAcrossConflicts => {
                self.resolve_fallback_average_dosage_across_conflicts(context)
            }
        }
    }

    /// Heuristic 1: The most stringent rule. Succeeds only if exactly one
    /// BIM entry's alleles are identical to the score file's alleles.
    fn resolve_exact_match(&self, context: &ResolutionContext) -> Option<Resolution> {
        let score_eff_allele = &context.score_info.effect_allele;
        let score_oth_allele = &context.score_info.other_allele;

        let exact_matches: Vec<_> = context
            .conflicting_interpretations
            .iter()
            .filter(|(_, (_, bim_a1, bim_a2))| {
                (bim_a1 == score_eff_allele && bim_a2 == score_oth_allele)
                    || (bim_a1 == score_oth_allele && bim_a2 == score_eff_allele)
            })
            .collect();

        if exact_matches.len() == 1 {
            let (packed_geno, (_, bim_a1, bim_a2)) = exact_matches[0];
            let dosage =
                Self::calculate_score_dosage(*packed_geno, bim_a1, bim_a2, context.score_info)?;
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// Heuristic 3: Solves conflicts by preferring an interpretation where the resulting
    /// genotype's allele lengths match the allele lengths from the score file.
    fn resolve_prefer_matching_allele_structure(
        &self,
        context: &ResolutionContext,
    ) -> Option<Resolution> {
        let score_a1_len = context.score_info.effect_allele.len();
        let score_a2_len = context.score_info.other_allele.len();

        let matching_structure_interpretations: Vec<_> = context
            .conflicting_interpretations
            .iter()
            .filter(|(packed_geno, (_, bim_a1, bim_a2))| {
                // Interpret the person's actual alleles first.
                let (person_allele_1, person_allele_2) =
                    Self::interpret_person_alleles(*packed_geno, bim_a1, bim_a2);

                // Get the lengths of the person's alleles.
                let person_a1_len = person_allele_1.len();
                let person_a2_len = person_allele_2.len();

                // If the genotype was invalid/missing, the lengths will be 0, so this will not match.
                if person_a1_len == 0 {
                    return false;
                }

                // Check for a match in either direction to handle swapped alleles.
                (person_a1_len == score_a1_len && person_a2_len == score_a2_len)
                    || (person_a1_len == score_a2_len && person_a2_len == score_a1_len)
            })
            .collect();

        // If we found exactly one interpretation with a matching allele structure, it's our winner.
        if matching_structure_interpretations.len() == 1 {
            let (packed_geno, (_, bim_a1, bim_a2)) = matching_structure_interpretations[0];
            let dosage =
                Self::calculate_score_dosage(*packed_geno, bim_a1, bim_a2, context.score_info)?;
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// Heuristic 2: Solves the `A/A` vs `AGA/AGA` conflict. Succeeds if
    /// exactly one interpretation is composed solely of standard alleles found
    /// in the score file, while others use non-standard/complex alleles.
    fn resolve_unambiguous_genotype(&self, context: &ResolutionContext) -> Option<Resolution> {
        let score_eff_allele = &context.score_info.effect_allele;
        let score_oth_allele = &context.score_info.other_allele;

        let mut unambiguous_interpretations = Vec::new();
        for &interpretation in context.conflicting_interpretations {
            let (_, (_, bim_a1, bim_a2)) = interpretation;
            let is_a1_valid = bim_a1 == score_eff_allele || bim_a1 == score_oth_allele;
            let is_a2_valid = bim_a2 == score_eff_allele || bim_a2 == score_oth_allele;

            if is_a1_valid && is_a2_valid {
                unambiguous_interpretations.push(interpretation);
            }
        }

        if unambiguous_interpretations.len() == 1 {
            let (packed_geno, (_, bim_a1, bim_a2)) = unambiguous_interpretations[0];
            let dosage =
                Self::calculate_score_dosage(packed_geno, bim_a1, bim_a2, context.score_info)?;
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// Heuristic 3: Succeeds if all conflicting interpretations, despite
    /// having different allele definitions, coincidentally result in the same
    /// final effect allele dosage.
    fn resolve_consistent_dosage(&self, context: &ResolutionContext) -> Option<Resolution> {
        let dosages: Vec<f64> = context
            .conflicting_interpretations
            .iter()
            .map(|(packed_geno, (_, bim_a1, bim_a2))| {
                Self::calculate_score_dosage(*packed_geno, bim_a1, bim_a2, context.score_info)
            })
            .collect::<Option<Vec<_>>>()?;

        let first_dosage = dosages[0];
        if dosages.iter().all(|&d| (d - first_dosage).abs() < 1e-9) {
            Some(Resolution {
                chosen_dosage: first_dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// Heuristic 4: A final tie-breaker that prefers a single heterozygous
    /// call if it conflicts with one or more homozygous calls.
    fn resolve_prefer_het(&self, context: &ResolutionContext) -> Option<Resolution> {
        let heterozygous_calls: Vec<_> = context
            .conflicting_interpretations
            .iter()
            .filter(|(packed_geno, _)| *packed_geno == 0b10)
            .collect();

        let homozygous_calls_exist = context
            .conflicting_interpretations
            .iter()
            .any(|(packed_geno, _)| *packed_geno != 0b10);

        if heterozygous_calls.len() == 1 && homozygous_calls_exist {
            let (packed_geno, (_, bim_a1, bim_a2)) = heterozygous_calls[0];
            let dosage =
                Self::calculate_score_dosage(*packed_geno, bim_a1, bim_a2, context.score_info)?;
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// Heuristic 5: Resolves conflicts where one row says homozygous Ref and another
    /// says homozygous Alt, but the alleles share an anchor base (one is a prefix of the other).
    /// This implies the array detected both the short (Ref) and long (Alt) alleles,
    /// so the true genotype is Heterozygous.
    fn resolve_indel_anchor_base(&self, context: &ResolutionContext) -> Option<Resolution> {
        let mut homozygous_alleles = Vec::new();

        for (packed_geno, (_, bim_a1, bim_a2)) in context.conflicting_interpretations {
            // Check for Homozygous calls (00 = Hom A1, 11 = Hom A2)
            match packed_geno {
                0b00 => homozygous_alleles.push(bim_a1.as_str()),
                0b11 => homozygous_alleles.push(bim_a2.as_str()),
                _ => {} // Ignore heterozygous or missing for this check
            }
        }

        if homozygous_alleles.len() < 2 {
            return None;
        }

        // Check if we have valid conflicting homozygous alleles
        let first_allele = homozygous_alleles[0];
        let has_conflict = homozygous_alleles.iter().any(|&a| a != first_allele);

        if !has_conflict {
            return None; // All homozygous calls agree, so this heuristic doesn't apply
        }

        // We have conflicting homozygous calls. Check if they form a prefix relationship.
        // We require that ALL observed homozygous alleles can be explained by a single
        // pair of (Short, Long) alleles where Short is a prefix of Long.

        // Find the shortest and longest alleles
        let shortest = homozygous_alleles.iter().min_by_key(|a| a.len()).unwrap();
        let longest = homozygous_alleles.iter().max_by_key(|a| a.len()).unwrap();

        // The prefix condition must hold
        if !longest.starts_with(shortest) {
            return None;
        }

        // Also strictly require that every homozygous allele found is EITHER the short or the long one.
        // (No third unrelated allele allowed)
        let clean_evidence = homozygous_alleles
            .iter()
            .all(|&a| a == *shortest || a == *longest);
        if !clean_evidence {
            return None;
        }

        // If we get here, we have evidence for both Short and Long alleles, and one is a prefix of the other.
        // This strongly suggests a Heterozygous genotype (Short/Long).
        // Since we are creating a synthetic Het call, the dosage of the effect allele is always 1.0,
        // (assuming the effect allele is one of the two).

        // Sanity check: is the effect allele even involved?
        let effect = &context.score_info.effect_allele;
        let other = &context.score_info.other_allele;
        if !((effect == *shortest && other == *longest)
            || (effect == *longest && other == *shortest))
        {
            return None;
        }

        // The inferred genotype is Short/Long.
        // If Effect == Short or Effect == Long, dosage is 1.0.
        // If Effect is neither (weird?), dosage is 0.0.
        // Derived from logic: if we infer Het (Short/Long), and one of them is the effect allele, dosage is 1.

        // Inferred Genotype: { *shortest, *longest }
        // Dosage = count of Effect Allele in that set.
        let mut final_dosage = 0.0;
        if effect == *shortest {
            final_dosage += 1.0;
        }
        if effect == *longest {
            final_dosage += 1.0;
        }

        Some(Resolution {
            chosen_dosage: final_dosage,
            method_used: *self,
        })
    }

    /// Heuristic 6: Final fallback for the specific opposing-homozygous pattern.
    /// If both homozygous states (00 and 11) are observed among conflicting
    /// interpretations, infer a heterozygous genotype from the observed alleles
    /// and score dosage from that inferred pair.
    fn resolve_fallback_opposing_homozygous_as_het(
        &self,
        context: &ResolutionContext,
    ) -> Option<Resolution> {
        let mut hom_a1_alleles = AHashSet::new();
        let mut hom_a2_alleles = AHashSet::new();

        for (packed_geno, (_, bim_a1, bim_a2)) in context.conflicting_interpretations {
            match packed_geno {
                0b00 => {
                    hom_a1_alleles.insert(bim_a1.as_str());
                }
                0b11 => {
                    hom_a2_alleles.insert(bim_a2.as_str());
                }
                _ => {}
            }
        }

        // This fallback applies only when each side points to a single concrete
        // allele and those alleles disagree (e.g., C/C vs T/T => infer C/T).
        if hom_a1_alleles.len() != 1 || hom_a2_alleles.len() != 1 {
            return None;
        }
        let allele_from_00 = *hom_a1_alleles.iter().next().unwrap();
        let allele_from_11 = *hom_a2_alleles.iter().next().unwrap();
        if allele_from_00 == allele_from_11 {
            return None;
        }

        let effect = context.score_info.effect_allele.as_str();
        let other = context.score_info.other_allele.as_str();
        if !((effect == allele_from_00 && other == allele_from_11)
            || (effect == allele_from_11 && other == allele_from_00))
        {
            return None;
        }
        let mut inferred_dosage = 0.0;
        if effect == allele_from_00 {
            inferred_dosage += 1.0;
        }
        if effect == allele_from_11 {
            inferred_dosage += 1.0;
        }

        Some(Resolution {
            chosen_dosage: inferred_dosage,
            method_used: *self,
        })
    }

    /// Heuristic 7: Absolute last resort. Compute dosage for each remaining
    /// conflicting interpretation and use the arithmetic mean.
    fn resolve_fallback_average_dosage_across_conflicts(
        &self,
        context: &ResolutionContext,
    ) -> Option<Resolution> {
        if context.conflicting_interpretations.is_empty() {
            return None;
        }
        let sum: f64 = context
            .conflicting_interpretations
            .iter()
            .map(|(packed_geno, (_, bim_a1, bim_a2))| {
                Self::calculate_score_dosage(*packed_geno, bim_a1, bim_a2, context.score_info)
            })
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .sum();
        let avg = sum / context.conflicting_interpretations.len() as f64;
        Some(Resolution {
            chosen_dosage: avg,
            method_used: *self,
        })
    }

    /// A private helper to compute dosage from raw PLINK bits.
    /// This function is now fully safe and self-contained, returning 0.0 if the
    /// effect allele is not one of the two alleles from the BIM entry.
    #[inline(always)]
    fn calculate_score_dosage(
        packed_geno: u8,
        bim_a1: &str,
        bim_a2: &str,
        score_info: &ScoreInfo,
    ) -> Option<f64> {
        if !score_allele_pair_matches(score_info, bim_a1, bim_a2) {
            return None;
        }
        // Decodes the genotype with respect to the BIM alleles.
        let dosage_wrt_a1 = match packed_geno {
            0b00 => 2.0, // Homozygous for A1
            0b10 => 1.0, // Heterozygous (one A1, one A2)
            0b11 => 0.0, // Homozygous for A2
            _ => 0.0,    // Missing or invalid
        };

        if bim_a1 == score_info.effect_allele {
            // Case 1: The effect allele is A1. The dosage is the count of A1.
            Some(dosage_wrt_a1)
        } else if bim_a2 == score_info.effect_allele {
            // Case 2: The effect allele is A2. The dosage is the count of A2,
            // which is the inverse of the A1 count.
            Some(2.0 - dosage_wrt_a1)
        } else {
            None
        }
    }

    /// A private helper to determine a person's actual alleles from their genotype bits.
    #[inline(always)]
    fn interpret_person_alleles<'a>(
        packed_geno: u8,
        bim_a1: &'a str,
        bim_a2: &'a str,
    ) -> (&'a str, &'a str) {
        match packed_geno {
            0b00 => (bim_a1, bim_a1), // Homozygous for A1
            0b10 => (bim_a1, bim_a2), // Heterozygous (one A1, one A2)
            0b11 => (bim_a2, bim_a2), // Homozygous for A2
            _ => ("", ""),            // Represents a missing or invalid genotype
        }
    }
}

/// The pipeline orchestrator that holds and runs the heuristic chain.
pub struct ResolverPipeline {
    heuristics: Vec<Heuristic>,
}

impl Default for ResolverPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl ResolverPipeline {
    /// Creates a new pipeline with the heuristics in their correct order of priority.
    pub fn new() -> Self {
        // The order is explicit
        let heuristics = vec![
            Heuristic::ExactScoreAlleleMatch,
            Heuristic::PrioritizeUnambiguousGenotype,
            Heuristic::PreferMatchingAlleleStructure,
            Heuristic::ConsistentDosage,
            Heuristic::IndelAnchorBase,
            Heuristic::PreferHeterozygous,
            Heuristic::FallbackOpposingHomozygousAsHet,
            Heuristic::FallbackAverageDosageAcrossConflicts,
        ];
        Self { heuristics }
    }

    /// Executes the heuristic chain, returning the first successful resolution.
    pub fn resolve(&self, context: &ResolutionContext) -> Option<Resolution> {
        for heuristic in &self.heuristics {
            if let Some(resolution) = heuristic.try_resolve(context) {
                return Some(resolution); // Success! Stop the chain.
            }
        }
        None // All heuristics failed.
    }
}

/// A private struct holding the complete, raw payload for a fatal ambiguity error.
/// Collecting this data first and formatting it once at the end is a key optimization.
struct FatalAmbiguityData {
    iid: String,
    locus_chr_pos: (String, u32),
    score_name: String,
    conflicts: Vec<ConflictSource>,
}

/// Genotype fetch failures surface as the person-major resolver reported them:
/// the fetch error's text, wrapped as an I/O error.
fn fetch_error(error: PipelineError) -> PipelineError {
    PipelineError::Io(error.to_string())
}

/// The most matching contexts a score application may have and still be
/// tabulated (4^4 genotype combinations). Wider applications resolve per person.
const TABULATED_MAX_CONTEXTS: usize = 4;

/// Sample warnings reported per heuristic.
const MAX_WARNING_SAMPLES: usize = 5;

const HEURISTIC_COUNT: usize = 8;

/// Every heuristic, indexed by `Heuristic as usize`.
const HEURISTICS: [Heuristic; HEURISTIC_COUNT] = [
    Heuristic::ExactScoreAlleleMatch,
    Heuristic::PrioritizeUnambiguousGenotype,
    Heuristic::PreferMatchingAlleleStructure,
    Heuristic::ConsistentDosage,
    Heuristic::PreferHeterozygous,
    Heuristic::IndelAnchorBase,
    Heuristic::FallbackOpposingHomozygousAsHet,
    Heuristic::FallbackAverageDosageAcrossConflicts,
];

/// People per evaluation block: few enough that a block's decoded genotypes and
/// score rows stay in cache, enough to amortize the per-block setup.
const MIN_BLOCK_PEOPLE: usize = 256;
const MAX_BLOCK_PEOPLE: usize = 4096;

/// Row bytes a resolver without memory maps reads for one group of rules.
const STREAMED_GROUP_BYTES: usize = 64 << 20;

/// What one combination of genotypes does to one score of one person.
#[derive(Clone, Copy)]
enum Outcome {
    /// No interpretation carries the score's alleles.
    Missing,
    /// Exactly one interpretation: its dosage times the weight.
    Add(f64),
    /// Several interpretations, reconciled by a heuristic.
    Resolved {
        method: Heuristic,
        dosage: f64,
        value: f64,
    },
    /// Several interpretations and no heuristic applies.
    Unresolvable,
}

/// The branch-free form of an `Outcome`, applied to every person.
#[derive(Clone, Copy)]
struct TableEntry {
    /// Added to the score. Outcomes that add nothing carry -0.0, the exact IEEE 754
    /// additive identity, so the accumulator keeps its bits, sign of zero included.
    value: f64,
    /// Added to the missing count.
    missing: u32,
    /// Whether the outcome is a heuristic event to report.
    reported: bool,
}

impl From<Outcome> for TableEntry {
    fn from(outcome: Outcome) -> Self {
        match outcome {
            Outcome::Missing => Self {
                value: -0.0,
                missing: 1,
                reported: false,
            },
            Outcome::Add(value) => Self {
                value,
                missing: 0,
                reported: false,
            },
            Outcome::Resolved { value, .. } => Self {
                value,
                missing: 0,
                reported: true,
            },
            Outcome::Unresolvable => Self {
                value: -0.0,
                missing: 0,
                reported: true,
            },
        }
    }
}

/// Resolves one score application for one combination of genotypes on its
/// matching contexts, by the person-major resolver's rules: drop missing calls,
/// then take the single interpretation left or run the heuristic chain.
fn resolve_outcome(
    pipeline: &ResolverPipeline,
    rule: &GroupedComplexRule,
    score_info: &ScoreInfo,
    matching: &[usize],
    genotypes: &[u8],
) -> Outcome {
    let interpretations: Vec<(u8, &(BimRowIndex, String, String))> = matching
        .iter()
        .zip(genotypes)
        .filter(|&(_, &bits)| bits != 0b01)
        .map(|(&context, &bits)| (bits, &rule.possible_contexts[context]))
        .collect();
    match interpretations.as_slice() {
        [] => Outcome::Missing,
        [(packed_geno, (_, bim_a1, bim_a2))] => {
            match Heuristic::calculate_score_dosage(*packed_geno, bim_a1, bim_a2, score_info) {
                Some(dosage) => Outcome::Add(dosage * score_info.weight as f64),
                None => Outcome::Missing,
            }
        }
        _ => {
            let context = ResolutionContext {
                score_info,
                conflicting_interpretations: &interpretations,
            };
            match pipeline.resolve(&context) {
                Some(resolution) => Outcome::Resolved {
                    method: resolution.method_used,
                    dosage: resolution.chosen_dosage,
                    value: resolution.chosen_dosage * score_info.weight as f64,
                },
                None => Outcome::Unresolvable,
            }
        }
    }
}

/// One score's view of a rule: the contexts that can carry its allele pair and
/// what every combination of genotypes on them does.
struct ApplicationPlan {
    column: usize,
    /// The rule's contexts whose allele pair matches the score's, in context order.
    matching: Vec<usize>,
    /// Indexed by the packed genotype code over `matching`, two bits per context
    /// with the first context lowest. Empty when `matching` is too wide to tabulate.
    entries: Vec<TableEntry>,
    outcomes: Vec<Outcome>,
}

struct RulePlan {
    /// Contexts some application reads; no other context is decoded.
    decoded_contexts: Vec<usize>,
    applications: Vec<ApplicationPlan>,
}

impl RulePlan {
    fn new(pipeline: &ResolverPipeline, rule: &GroupedComplexRule) -> Self {
        let mut decoded = vec![false; rule.possible_contexts.len()];
        let applications = rule
            .score_applications
            .iter()
            .map(|score_info| {
                let matching: Vec<usize> = rule
                    .possible_contexts
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, bim_a1, bim_a2))| {
                        score_allele_pair_matches(score_info, bim_a1, bim_a2)
                    })
                    .map(|(context, _)| context)
                    .collect();
                for &context in &matching {
                    decoded[context] = true;
                }
                let outcomes: Vec<Outcome> = if matching.len() <= TABULATED_MAX_CONTEXTS {
                    let mut genotypes = vec![0u8; matching.len()];
                    (0..1usize << (2 * matching.len()))
                        .map(|code| {
                            for (position, bits) in genotypes.iter_mut().enumerate() {
                                *bits = ((code >> (2 * position)) & 0b11) as u8;
                            }
                            resolve_outcome(pipeline, rule, score_info, &matching, &genotypes)
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                ApplicationPlan {
                    column: score_info.score_column_index.0,
                    entries: outcomes.iter().map(|&outcome| outcome.into()).collect(),
                    matching,
                    outcomes,
                }
            })
            .collect();
        let decoded_contexts = decoded
            .iter()
            .enumerate()
            .filter(|&(_, &is_decoded)| is_decoded)
            .map(|(context, _)| context)
            .collect();
        Self {
            decoded_contexts,
            applications,
        }
    }
}

/// A reported heuristic event, kept compact until the report is built.
struct Sample {
    /// (person, rule, application): the order the person-major resolver met events in.
    order: (usize, usize, usize),
    /// Genotype bits on the application's matching contexts.
    genotypes: Vec<u8>,
    dosage: f64,
}

/// Heuristic events from one block of people.
#[derive(Default)]
struct BlockReport {
    counts: [u64; HEURISTIC_COUNT],
    /// The earliest events per heuristic, sorted, at most `MAX_WARNING_SAMPLES` each.
    samples: [Vec<Sample>; HEURISTIC_COUNT],
    unresolvable: Option<Sample>,
}

impl BlockReport {
    /// Records a reported outcome. Returns false when the outcome is unresolvable
    /// and the pass must stop.
    fn record(
        &mut self,
        outcome: Outcome,
        order: (usize, usize, usize),
        genotypes: impl FnOnce() -> Vec<u8>,
    ) -> bool {
        match outcome {
            Outcome::Resolved { method, dosage, .. } => {
                self.counts[method as usize] += 1;
                let samples = &mut self.samples[method as usize];
                if samples.len() < MAX_WARNING_SAMPLES
                    || samples.last().is_some_and(|last| order < last.order)
                {
                    let position = samples.partition_point(|existing| existing.order < order);
                    samples.insert(
                        position,
                        Sample {
                            order,
                            genotypes: genotypes(),
                            dosage,
                        },
                    );
                    samples.truncate(MAX_WARNING_SAMPLES);
                }
                true
            }
            Outcome::Unresolvable => {
                self.unresolvable = Some(Sample {
                    order,
                    genotypes: genotypes(),
                    dosage: 0.0,
                });
                false
            }
            Outcome::Missing | Outcome::Add(_) => true,
        }
    }
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
    pipeline: &'a ResolverPipeline,
    num_scores: usize,
    stop: &'a AtomicBool,
}

/// Applies a group's rules to one block of people. Rules run in order, and each
/// rule's applications in order, so every accumulator receives the same additions
/// in the same order as under the person-major resolver, and ends bit-identical.
fn evaluate_block(
    pass: &GroupPass,
    first_person: usize,
    scores: &mut [f64],
    counts: &mut [u32],
) -> BlockReport {
    let num_scores = pass.num_scores;
    let num_people = scores.len() / num_scores;
    let people = first_person..first_person + num_people;
    let bytes = &pass.layout.bytes[people.clone()];
    let shifts = &pass.layout.shifts[people.clone()];
    let forced_missing = {
        let all = &pass.layout.forced_missing;
        &all[all.partition_point(|&person| person < people.start)
            ..all.partition_point(|&person| person < people.end)]
    };
    let first_context = pass.context_offsets[pass.group.start];
    let mut genotypes = vec![0u8; pass.max_contexts * num_people];
    let mut code_buffer = vec![0u8; num_people];
    let mut tuple = Vec::new();
    let mut report = BlockReport::default();

    for rule_idx in pass.group.clone() {
        if pass.stop.load(Ordering::Relaxed) {
            break;
        }
        let rule = &pass.rules[rule_idx];
        let plan = &pass.plans[rule_idx];
        let rows = &pass.rows[pass.context_offsets[rule_idx] - first_context
            ..pass.context_offsets[rule_idx + 1] - first_context];
        for &context in &plan.decoded_contexts {
            let decoded = &mut genotypes[context * num_people..(context + 1) * num_people];
            decode_genotypes(rows[context], bytes, shifts, decoded);
            for &person in forced_missing {
                decoded[person - first_person] = 0b01;
            }
        }

        for (application_idx, application) in plan.applications.iter().enumerate() {
            let column = application.column;
            let people_rows = scores
                .chunks_exact_mut(num_scores)
                .zip(counts.chunks_exact_mut(num_scores))
                .enumerate();

            if application.entries.is_empty() {
                let score_info = &rule.score_applications[application_idx];
                for (person, (person_scores, person_counts)) in people_rows {
                    tuple.clear();
                    tuple.extend(
                        application
                            .matching
                            .iter()
                            .map(|&context| genotypes[context * num_people + person]),
                    );
                    let outcome = resolve_outcome(
                        pass.pipeline,
                        rule,
                        score_info,
                        &application.matching,
                        &tuple,
                    );
                    let entry = TableEntry::from(outcome);
                    person_scores[column] += entry.value;
                    person_counts[column] += entry.missing;
                    if entry.reported
                        && !report.record(
                            outcome,
                            (first_person + person, rule_idx, application_idx),
                            || tuple.clone(),
                        )
                    {
                        pass.stop.store(true, Ordering::Relaxed);
                        return report;
                    }
                }
                continue;
            }

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
            for ((person, (person_scores, person_counts)), &code) in people_rows.zip(codes) {
                let entry = application.entries[code as usize];
                person_scores[column] += entry.value;
                person_counts[column] += entry.missing;
                if entry.reported {
                    let width = application.matching.len();
                    let unpack = || {
                        (0..width)
                            .map(|position| (code >> (2 * position)) & 0b11)
                            .collect()
                    };
                    let order = (first_person + person, rule_idx, application_idx);
                    if !report.record(application.outcomes[code as usize], order, unpack) {
                        pass.stop.store(true, Ordering::Relaxed);
                        return report;
                    }
                }
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

fn conflict_sources(
    rule: &GroupedComplexRule,
    matching: &[usize],
    genotypes: &[u8],
) -> Vec<ConflictSource> {
    matching
        .iter()
        .zip(genotypes)
        .filter(|&(_, &bits)| bits != 0b01)
        .map(|(&context, &bits)| {
            let (bim_row, bim_a1, bim_a2) = &rule.possible_contexts[context];
            ConflictSource {
                bim_row: *bim_row,
                alleles: (bim_a1.clone(), bim_a2.clone()),
                genotype_bits: bits,
            }
        })
        .collect()
}

fn resolution_method(method: Heuristic, chosen_dosage: f64) -> ResolutionMethod {
    match method {
        Heuristic::ExactScoreAlleleMatch => ResolutionMethod::ExactScoreAlleleMatch { chosen_dosage },
        Heuristic::PrioritizeUnambiguousGenotype => {
            ResolutionMethod::PrioritizeUnambiguousGenotype { chosen_dosage }
        }
        Heuristic::PreferMatchingAlleleStructure => {
            ResolutionMethod::PreferMatchingAlleleStructure { chosen_dosage }
        }
        Heuristic::ConsistentDosage => ResolutionMethod::ConsistentDosage {
            dosage: chosen_dosage,
        },
        Heuristic::PreferHeterozygous => ResolutionMethod::PreferHeterozygous { chosen_dosage },
        Heuristic::IndelAnchorBase => ResolutionMethod::IndelAnchorBase { chosen_dosage },
        Heuristic::FallbackOpposingHomozygousAsHet => {
            ResolutionMethod::FallbackOpposingHomozygousAsHet { chosen_dosage }
        }
        Heuristic::FallbackAverageDosageAcrossConflicts => {
            ResolutionMethod::FallbackAverageDosageAcrossConflicts { chosen_dosage }
        }
    }
}

/// Heuristic warnings, and the unresolvable ambiguity that stopped resolution, if any.
struct ResolutionReport {
    warnings: FinalAggregatedCollector,
    unresolvable: Option<FatalAmbiguityData>,
}

/// The row-major resolver. Each rule context's row is located once and its scored
/// span borrowed from the memory map, or read once per group of rules when there
/// is no map. People are processed in parallel blocks; inside a block each rule
/// decodes its rows for the block and applies per-rule outcome tables, so no
/// genotype pays for a source dispatch, a fileset search or a hash lookup.
fn resolve_rows(
    resolver: &ComplexVariantResolver,
    prep_result: &PreparationResult,
    final_scores: &mut [f64],
    final_missing_counts: &mut [u32],
    limits: ResolveLimits,
    pb: &ProgressBar,
) -> Result<ResolutionReport, PipelineError> {
    let rules = &prep_result.complex_rules;
    let num_scores = prep_result.score_names.len();
    let num_people = final_scores
        .len()
        .checked_div(num_scores)
        .unwrap_or(0)
        .min(final_missing_counts.len().checked_div(num_scores).unwrap_or(0));
    let mut report = ResolutionReport {
        warnings: FinalAggregatedCollector::new(),
        unresolvable: None,
    };
    if num_people == 0 || rules.is_empty() {
        return Ok(report);
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

    let pipeline = ResolverPipeline::new();
    let plans: Vec<RulePlan> = rules
        .par_iter()
        .map(|rule| RulePlan::new(&pipeline, rule))
        .collect();

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

    let stop = AtomicBool::new(false);
    let mut counts = [0u64; HEURISTIC_COUNT];
    let mut samples: [Vec<Sample>; HEURISTIC_COUNT] = Default::default();
    let mut unresolvable: Option<Sample> = None;
    let mut storage = Vec::<u8>::new();
    let people_scores = &mut final_scores[..num_people * num_scores];
    let people_counts = &mut final_missing_counts[..num_people * num_scores];
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
            pipeline: &pipeline,
            num_scores,
            stop: &stop,
        };
        let block_reports: Vec<BlockReport> = people_scores
            .par_chunks_mut(block_cells)
            .zip(people_counts.par_chunks_mut(block_cells))
            .enumerate()
            .map(|(block, (scores, counts))| {
                let block_report =
                    evaluate_block(&pass, block * limits.block_people, scores, counts);
                pb.inc((scores.len() / num_scores) as u64);
                block_report
            })
            .collect();

        for block_report in block_reports {
            let BlockReport {
                counts: block_counts,
                samples: block_samples,
                unresolvable: block_unresolvable,
            } = block_report;
            for (method_idx, method_samples) in block_samples.into_iter().enumerate() {
                counts[method_idx] += block_counts[method_idx];
                samples[method_idx].extend(method_samples);
            }
            unresolvable = match (unresolvable, block_unresolvable) {
                (Some(current), Some(candidate)) if candidate.order < current.order => {
                    Some(candidate)
                }
                (current, candidate) => current.or(candidate),
            };
        }
        for method_samples in &mut samples {
            method_samples.sort_by_key(|sample| sample.order);
            method_samples.truncate(MAX_WARNING_SAMPLES);
        }
        if stop.load(Ordering::Relaxed) {
            break;
        }
    }

    for (method_idx, method_samples) in samples.iter().enumerate() {
        if counts[method_idx] == 0 {
            continue;
        }
        let method = HEURISTICS[method_idx];
        let infos = method_samples
            .iter()
            .map(|sample| {
                let (person, rule_idx, application_idx) = sample.order;
                let rule = &rules[rule_idx];
                let score_info = &rule.score_applications[application_idx];
                CriticalIntegrityWarningInfo {
                    iid: prep_result.final_person_iids[person].clone(),
                    locus_chr_pos: rule.locus_chr_pos.clone(),
                    score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
                    conflicts: conflict_sources(
                        rule,
                        &plans[rule_idx].applications[application_idx].matching,
                        &sample.genotypes,
                    ),
                    resolution_method: resolution_method(method, sample.dosage),
                    score_effect_allele: score_info.effect_allele.clone(),
                    score_other_allele: score_info.other_allele.clone(),
                }
            })
            .collect();
        report
            .warnings
            .insert(method, (counts[method_idx], infos));
    }
    report.unresolvable = unresolvable.map(|sample| {
        let (person, rule_idx, application_idx) = sample.order;
        let rule = &rules[rule_idx];
        let score_info = &rule.score_applications[application_idx];
        FatalAmbiguityData {
            iid: prep_result.final_person_iids[person].clone(),
            locus_chr_pos: rule.locus_chr_pos.clone(),
            score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
            conflicts: conflict_sources(
                rule,
                &plans[rule_idx].applications[application_idx].matching,
                &sample.genotypes,
            ),
        }
    });
    Ok(report)
}

// The "slow path" resolver for complex variants.
///
/// This function runs *after* the main high-performance pipeline is complete and
/// adds every person's score contributions for the small set of variants that
/// could not be handled by the fast path. It is row-major (see `resolve_rows`),
/// and the progress bar advances as blocks of people finish.
pub fn resolve_complex_variants(
    resolver: &ComplexVariantResolver,
    prep_result: &Arc<PreparationResult>,
    final_scores: &mut [f64],
    final_missing_counts: &mut [u32],
) -> Result<(), PipelineError> {
    let num_rules = prep_result.complex_rules.len();
    if num_rules == 0 {
        return Ok(());
    }

    eprintln!("> Resolving {num_rules} complex variant rules...");

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

    let ResolutionReport {
        warnings: all_warnings_for_reporting,
        unresolvable,
    } = resolution?;

    if !all_warnings_for_reporting.is_empty() {
        eprintln!(
            "\n\n========================= CRITICAL DATA INTEGRITY WARNINGS ========================="
        );
        eprintln!(
            "Gnomon detected loci with ambiguous data that were resolved via heuristics.\nWhile computation continued, the underlying data should be investigated."
        );

        for (heuristic, (total_count, samples)) in &all_warnings_for_reporting {
            eprintln!(
                "\n==================== WARNING CATEGORY: {:?} ====================",
                heuristic
            );
            eprintln!("Total Occurrences: {}", total_count);
            eprintln!("Showing up to 5 samples:");

            if samples.is_empty() {
                eprintln!("  (No samples collected)");
            } else {
                for (i, info) in samples.iter().enumerate() {
                    if i > 0 {
                        eprintln!(
                            "---------------------------------------------------------------------------------"
                        );
                    }
                    eprintln!("{}", format_critical_integrity_warning(info));
                }
            }
        }
        eprintln!(
            "\n=================================================================================\n"
        );
    }

    if let Some(data) = unresolvable {
        return Err(PipelineError::Compute(format_fatal_ambiguity_report(&data)));
    }

    eprintln!("> Complex variant resolution complete.");
    Ok(())
}

/// A private helper function to format the final, dense data report for a fatal ambiguity.
/// This is called only once, on the main thread, after a fatal error is confirmed.
fn format_fatal_ambiguity_report(data: &FatalAmbiguityData) -> String {
    use std::fmt::Write;
    let mut report = String::with_capacity(512);

    // Helper to interpret genotype bits is still useful.
    let interpret_genotype = |bits: u8, a1: &str, a2: &str| -> String {
        match bits {
            0b00 => format!("{a1}/{a1}"),
            0b01 => "Missing".to_string(),
            0b10 => format!("{a1}/{a2}"),
            0b11 => format!("{a2}/{a2}"),
            _ => "Invalid Bits".to_string(),
        }
    };

    // Build the final report string.
    writeln!(
        report,
        "Fatal: Unresolvable ambiguity for individual '{}'.\n",
        data.iid
    )
    .unwrap();
    writeln!(report, "Individual:   {}", data.iid).unwrap();
    writeln!(
        report,
        "Locus:        {}:{}",
        data.locus_chr_pos.0, data.locus_chr_pos.1
    )
    .unwrap();
    writeln!(report, "Score:        {}\n", data.score_name).unwrap();
    writeln!(report, "Conflicting Sources:").unwrap();

    for conflict in &data.conflicts {
        writeln!(report, "  - BIM Row: {}", conflict.bim_row.0).unwrap();
        writeln!(
            report,
            "    Alleles (A1,A2): ({}, {})",
            conflict.alleles.0, conflict.alleles.1
        )
        .unwrap();

        let bits_str = match conflict.genotype_bits {
            0b00 => "00",
            0b01 => "01",
            0b10 => "10",
            0b11 => "11",
            _ => "??",
        };
        let interpretation = interpret_genotype(
            conflict.genotype_bits,
            &conflict.alleles.0,
            &conflict.alleles.1,
        );
        writeln!(
            report,
            "    Genotype Bits:   {bits_str} (Interpreted as {interpretation})"
        )
        .unwrap();
    }

    report
}

/// A private helper function to format a critical integrity warning.
/// This is called only once, on the main thread, to report benign ambiguities.
fn format_critical_integrity_warning(data: &CriticalIntegrityWarningInfo) -> String {
    use std::fmt::Write;
    let mut report = String::with_capacity(512);

    // Helper to interpret genotype bits and alleles into a human-readable string.
    let interpret_genotype = |bits: u8, a1: &str, a2: &str| -> String {
        match bits {
            0b00 => format!("{a1}/{a1}"),
            0b01 => "Missing".to_string(),
            0b10 => format!("{a1}/{a2}"),
            0b11 => format!("{a2}/{a2}"),
            _ => "Invalid Bits".to_string(),
        }
    };

    // Update the `is_chosen` helper
    let is_chosen = |method: &ResolutionMethod,
                     conflict: &ConflictSource,
                     score_ea: &str,
                     score_oa: &str|
     -> bool {
        let bim_a1 = &conflict.alleles.0;
        let bim_a2 = &conflict.alleles.1;
        match method {
            ResolutionMethod::ExactScoreAlleleMatch { .. } => {
                (bim_a1 == score_ea && bim_a2 == score_oa)
                    || (bim_a1 == score_oa && bim_a2 == score_ea)
            }
            ResolutionMethod::PrioritizeUnambiguousGenotype { .. } => {
                (bim_a1 == score_ea || bim_a1 == score_oa)
                    && (bim_a2 == score_ea || bim_a2 == score_oa)
            }
            ResolutionMethod::PreferMatchingAlleleStructure { .. } => {
                let score_a1_len = score_ea.len();
                let score_a2_len = score_oa.len();

                // Re-create the logic from the actual heuristic
                let (person_allele_1, person_allele_2) = Heuristic::interpret_person_alleles(
                    conflict.genotype_bits,
                    &conflict.alleles.0,
                    &conflict.alleles.1,
                );
                let person_a1_len = person_allele_1.len();
                let person_a2_len = person_allele_2.len();

                if person_a1_len == 0 {
                    return false;
                }

                (person_a1_len == score_a1_len && person_a2_len == score_a2_len)
                    || (person_a1_len == score_a2_len && person_a2_len == score_a1_len)
            }
            ResolutionMethod::PreferHeterozygous { .. } => conflict.genotype_bits == 0b10,
            ResolutionMethod::ConsistentDosage { .. } => true,
            ResolutionMethod::IndelAnchorBase { .. } => {
                // For indel anchor base, we "chose" the homozygous rows that contributed
                // to the inference.
                match conflict.genotype_bits {
                    0b00 => true, // Contributed evidence if hom
                    0b11 => true,
                    _ => false,
                }
            }
            ResolutionMethod::FallbackOpposingHomozygousAsHet { .. } => true,
            ResolutionMethod::FallbackAverageDosageAcrossConflicts { .. } => true,
        }
    };

    // Add the individual, locus, and score information to the report
    writeln!(
        report,
        "Ambiguity resolved for Individual '{}' at Locus {}:{}",
        data.iid, data.locus_chr_pos.0, data.locus_chr_pos.1
    )
    .unwrap();
    writeln!(report, "  While calculating score: '{}'", data.score_name).unwrap();
    writeln!(report).unwrap();

    // Update the rationale generation
    let method_name: &str;
    let mut rationale = String::new();

    match &data.resolution_method {
        ResolutionMethod::ExactScoreAlleleMatch { .. } => {
            method_name = "'Exact Score Allele Match' Heuristic";
            if let Some(chosen) = data.conflicts.iter().find(|c| {
                is_chosen(
                    &data.resolution_method,
                    c,
                    &data.score_effect_allele,
                    &data.score_other_allele,
                )
            }) {
                write!(rationale, "The interpretation with alleles ({}, {}) was chosen because it perfectly matches the score file.", chosen.alleles.0, chosen.alleles.1).unwrap();
            }
        }
        ResolutionMethod::PrioritizeUnambiguousGenotype { .. } => {
            method_name = "'Prioritize Unambiguous Genotype' Heuristic";
            if let Some(chosen) = data.conflicts.iter().find(|c| {
                is_chosen(
                    &data.resolution_method,
                    c,
                    &data.score_effect_allele,
                    &data.score_other_allele,
                )
            }) {
                write!(rationale, "The interpretation with alleles ({}, {}) was chosen because both alleles are present in the score file's required set.", chosen.alleles.0, chosen.alleles.1).unwrap();
            }
        }
        ResolutionMethod::PreferMatchingAlleleStructure { .. } => {
            method_name = "'Prefer Matching Allele Structure' Heuristic";
            if let Some(c) = data.conflicts.iter().find(|c| {
                is_chosen(
                    &data.resolution_method,
                    c,
                    &data.score_effect_allele,
                    &data.score_other_allele,
                )
            }) {
                // Re-interpret the genotype to get the person's actual alleles for the rationale.
                let (pa1, pa2) = Heuristic::interpret_person_alleles(
                    c.genotype_bits,
                    &c.alleles.0,
                    &c.alleles.1,
                );
                let person_geno_str = format!("{}/{}", pa1, pa2);
                write!(rationale, "The interpretation from BIM Row {} (resulting in genotype '{}') was chosen because its allele lengths ({}, {}) structurally match the score file.", c.bim_row.0, person_geno_str, pa1.len(), pa2.len()).unwrap();
            }
        }
        ResolutionMethod::PreferHeterozygous { .. } => {
            method_name = "'Prefer Heterozygous' Heuristic";
            let chosen = data.conflicts.iter().find(|c| c.genotype_bits == 0b10);
            let rejected = data.conflicts.iter().find(|c| c.genotype_bits != 0b10);
            if let (Some(c), Some(r)) = (chosen, rejected) {
                let chosen_geno = interpret_genotype(c.genotype_bits, &c.alleles.0, &c.alleles.1);
                let rejected_geno = interpret_genotype(r.genotype_bits, &r.alleles.0, &r.alleles.1);
                write!(rationale, "The heterozygous interpretation ({}) was chosen over a conflicting homozygous interpretation ({}).", chosen_geno, rejected_geno).unwrap();
            }
        }
        ResolutionMethod::ConsistentDosage { dosage } => {
            method_name = "'Consistent Dosage' Heuristic";

            write!(rationale, "All conflicting sources yielded a consistent effect allele dosage of {}, so computation continued.", dosage).unwrap();
        }
        ResolutionMethod::IndelAnchorBase { .. } => {
            method_name = "'Indel Anchor Base' Heuristic";
            write!(rationale, "Conflicting homozygous calls were found for alleles sharing an anchor base (one is a prefix of the other). This implies the array detected both alleles, so a Heterozygous genotype was inferred.").unwrap();
        }
        ResolutionMethod::FallbackOpposingHomozygousAsHet { .. } => {
            method_name = "'Fallback Opposing Homozygous As Het' Heuristic";
            write!(rationale, "All higher-priority ambiguity rules failed, and conflicting sources contained opposing homozygous states (00 and 11). A synthetic heterozygous genotype was inferred from those alleles, and dosage was computed from that inferred pair.").unwrap();
        }
        ResolutionMethod::FallbackAverageDosageAcrossConflicts { chosen_dosage } => {
            method_name = "'Fallback Average Dosage Across Conflicts' Heuristic";
            write!(rationale, "All higher-priority ambiguity rules failed. Dosage was computed independently for each conflicting interpretation and averaged as a final fallback (mean dosage = {}).", chosen_dosage).unwrap();
        }
    };

    writeln!(report, "  Method: {}", method_name).unwrap();
    writeln!(
        report,
        "  Score File requires: Effect={}, Other={}",
        data.score_effect_allele, data.score_other_allele
    )
    .unwrap();

    writeln!(report, "\n  Conflicting Sources Considered:").unwrap();

    for conflict in &data.conflicts {
        let prefix = if is_chosen(
            &data.resolution_method,
            conflict,
            &data.score_effect_allele,
            &data.score_other_allele,
        ) {
            "-> Chosen:  "
        } else {
            "   Rejected:"
        };
        let interpretation = interpret_genotype(
            conflict.genotype_bits,
            &conflict.alleles.0,
            &conflict.alleles.1,
        );
        writeln!(
            report,
            "{} BIM Row {}: Alleles=({}, {}), Genotype={} ({})",
            prefix,
            conflict.bim_row.0,
            conflict.alleles.0,
            conflict.alleles.1,
            conflict.genotype_bits,
            interpretation
        )
        .unwrap();
    }

    if !rationale.is_empty() {
        writeln!(report, "\n  Rationale: {}", rationale).unwrap();
    }

    report.trim_end().to_string()
}
