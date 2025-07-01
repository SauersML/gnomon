use crate::types::{
    BimRowIndex, FilesetBoundary, PipelineKind, PreparationResult,
    ScoreColumnIndex, ScoreInfo
};
use ahash::AHashSet;
use dashmap::DashSet;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
use crate::pipeline::{PipelineError, ScopeGuard};

/// A performant, read-only resolver for fetching complex variant genotypes.
///
/// This enum is initialized ONCE at the start of the pipeline run. It holds
/// either a single memory map or a collection of them, avoiding the massive
/// performance penalty of re-opening and re-mapping files inside a parallel loop.
pub enum ComplexVariantResolver {
    SingleFile(Arc<Mmap>),
    MultiFile {
        // A vector of all memory maps for the multi-file case.
        mmaps: Vec<Mmap>,
        // A copy of the boundaries to map a global index to the correct mmap.
        boundaries: Vec<FilesetBoundary>,
    },
}

impl ComplexVariantResolver {
    /// Creates a new resolver based on the pipeline strategy.
    pub fn new(prep_result: &PreparationResult) -> Result<Self, PipelineError> {
        match &prep_result.pipeline_kind {
            PipelineKind::SingleFile(bed_path) => {
                let file = File::open(bed_path).map_err(|e| {
                    PipelineError::Io(format!("Opening {}: {}", bed_path.display(), e))
                })?;
                let mmap = Arc::new(unsafe {
                    Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string()))?
                });
                Ok(Self::SingleFile(mmap))
            }
            PipelineKind::MultiFile(boundaries) => {
                let mmaps = boundaries
                    .iter()
                    .map(|b| {
                        let file = File::open(&b.bed_path).map_err(|e| {
                            PipelineError::Io(format!("Opening {}: {}", b.bed_path.display(), e))
                        })?;
                        unsafe { Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string())) }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::MultiFile {
                    mmaps,
                    boundaries: boundaries.clone(),
                })
            }
        }
    }

    /// Fetches a packed genotype for a given person and global variant index.
    /// This is the fast, central lookup method used by the parallel resolver.
    #[inline(always)]
    fn get_packed_genotype(
        &self,
        bytes_per_variant: u64,
        bim_row_index: BimRowIndex,
        fam_index: u32,
    ) -> u8 {
        let (mmap, local_bim_index) = match self {
            ComplexVariantResolver::SingleFile(mmap) => (mmap.as_ref(), bim_row_index.0),
            ComplexVariantResolver::MultiFile { mmaps, boundaries } => {
                // Find which fileset contains this global index using a fast binary search.
                let fileset_idx =
                    boundaries.partition_point(|b| b.starting_global_index <= bim_row_index.0) - 1;
                let boundary = &boundaries[fileset_idx];
                let local_index = bim_row_index.0 - boundary.starting_global_index;
                // This unsafe is acceptable because the number of mmaps is tied to the
                // number of boundaries, and the index is derived from it.
                (unsafe { mmaps.get_unchecked(fileset_idx) }, local_index)
            }
        };

        // The +3 skips the PLINK .bed file magic number (0x6c, 0x1b, 0x01).
        let variant_start_offset = 3 + local_bim_index * bytes_per_variant;
        let person_byte_offset = fam_index as u64 / 4;
        let final_byte_offset = (variant_start_offset + person_byte_offset) as usize;

        let bit_offset_in_byte = (fam_index % 4) * 2;

        // This indexing is safe because the preparation phase guarantees all indices are valid.
        let packed_byte = unsafe { *mmap.get_unchecked(final_byte_offset) };
        (packed_byte >> bit_offset_in_byte) & 0b11
    }
}

//========================================================================================
//
//                      The Zero-Cost Heuristic Pipeline
//
//========================================================================================

use crate::types::{BimRowIndex, ScoreInfo};

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

/// An enum representing the complete, ordered set of resolution strategies.
/// This approach uses static dispatch for zero-cost abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heuristic {
    /// Tries to find one BIM entry that perfectly matches both score file alleles.
    ExactScoreAlleleMatch,
    /// Tries to find one interpretation composed ONLY of score file alleles.
    PrioritizeUnambiguousGenotype,
    /// Checks if all conflicting interpretations result in the same dosage.
    ConsistentDosage,
    /// As a last resort, prefers a single heterozygous call over homozygous ones.
    PreferHeterozygous,
}

impl Heuristic {
    /// The main dispatcher for the enum. It calls the appropriate private method
    /// for the specific heuristic variant.
    pub fn try_resolve(&self, context: &ResolutionContext) -> Option<Resolution> {
        match self {
            Heuristic::ExactScoreAlleleMatch => self.resolve_exact_match(context),
            Heuristic::PrioritizeUnambiguousGenotype => self.resolve_unambiguous_genotype(context),
            Heuristic::ConsistentDosage => self.resolve_consistent_dosage(context),
            Heuristic::PreferHeterozygous => self.resolve_prefer_het(context),
        }
    }

    /// **Heuristic 1:** The most stringent rule. Succeeds only if exactly one
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
            let (packed_geno, (_, bim_a1, _)) = exact_matches[0];
            let dosage = Self::calculate_dosage(*packed_geno, bim_a1, score_eff_allele);
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// **Heuristic 2 (NEW):** Solves the `A/A` vs `AGA/AGA` conflict. Succeeds if
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
            let (packed_geno, (_, bim_a1, _)) = unambiguous_interpretations[0];
            let dosage = Self::calculate_dosage(*packed_geno, bim_a1, score_eff_allele);
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// **Heuristic 3:** Succeeds if all conflicting interpretations, despite
    /// having different allele definitions, coincidentally result in the same
    /// final effect allele dosage.
    fn resolve_consistent_dosage(&self, context: &ResolutionContext) -> Option<Resolution> {
        let dosages: Vec<f64> = context
            .conflicting_interpretations
            .iter()
            .map(|(packed_geno, (_, bim_a1, _))| {
                Self::calculate_dosage(*packed_geno, bim_a1, &context.score_info.effect_allele)
            })
            .collect();

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

    /// **Heuristic 4:** A final tie-breaker that prefers a single heterozygous
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
            let (packed_geno, (_, bim_a1, _)) = heterozygous_calls[0];
            let dosage = Self::calculate_dosage(*packed_geno, bim_a1, &context.score_info.effect_allele);
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// A private helper to compute dosage from raw PLINK bits.
    #[inline(always)]
    fn calculate_dosage(packed_geno: u8, bim_a1: &str, effect_allele: &str) -> f64 {
        let dosage_wrt_a1 = match packed_geno {
            0b00 => 2.0, // Homozygous for A1
            0b10 => 1.0, // Heterozygous
            0b11 => 0.0, // Homozygous for A2
            _ => 0.0,    // Should not happen for valid data
        };

        if bim_a1 == effect_allele {
            dosage_wrt_a1
        } else {
            2.0 - dosage_wrt_a1
        }
    }
}

/// The pipeline orchestrator that holds and runs the heuristic chain.
pub struct ResolverPipeline {
    heuristics: Vec<Heuristic>,
}

impl ResolverPipeline {
    /// Creates a new pipeline with the heuristics in their correct order of priority.
    pub fn new() -> Self {
        // The order is explicit
        let heuristics = vec![
            Heuristic::ExactScoreAlleleMatch,
            Heuristic::PrioritizeUnambiguousGenotype,
            Heuristic::ConsistentDosage,
            Heuristic::PreferHeterozygous,
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

/// A private helper struct to hold the raw components of a warning message.
/// This avoids heap allocations (`format!`) inside the hot parallel loop.
struct WarningInfo {
    person_output_idx: usize,
    locus_id: BimRowIndex,
    winning_a1: String,
    winning_a2: String,
    score_col_idx: ScoreColumnIndex,
}

/// A private struct holding the raw data for one conflicting source of evidence.
/// This is used exclusively for building the final fatal error report.
struct ConflictSource {
    bim_row: BimRowIndex,
    alleles: (String, String),
    genotype_bits: u8,
}

/// A private struct holding the complete, raw payload for a fatal ambiguity error.
/// Collecting this data first and formatting it once at the end is a key optimization.
struct FatalAmbiguityData {
    iid: String,
    locus_chr_pos: (String, u32),
    score_name: String,
    conflicts: Vec<ConflictSource>,
}

/// Describes the specific heuristic used to resolve a critical data ambiguity,
/// holding the data needed for transparent reporting.
enum ResolutionMethod {
    /// All conflicting sources yielded the same effect allele dosage.
    ConsistentDosage { dosage: f64 },
    /// Exactly one heterozygous call was found alongside one or more homozygous
    /// calls, and the heterozygous call was chosen.
    PreferHeterozygous { chosen_dosage: f64 },
    /// A single BIM entry's alleles perfectly matched the score file alleles.
    ExactScoreAlleleMatch { chosen_dosage: f64 },
}

/// A private struct holding the data for a critical but non-fatal integrity warning.
/// This is used when multiple data sources conflict but lead to a consistent outcome.
struct CriticalIntegrityWarningInfo {
    iid: String,
    locus_chr_pos: (String, u32),
    score_name: String,
    conflicts: Vec<ConflictSource>,
    /// The specific heuristic that was successfully applied and its outcome.
    resolution_method: ResolutionMethod,
}

/// A private helper enum to represent the outcome of processing one person for one rule.
/// This decouples the core logic from the side-effects (like I/O or setting global flags).
enum ResolutionOutcome {
    Success,
    Warning(WarningInfo),
    CriticalIntegrityWarning(CriticalIntegrityWarningInfo),
    Fatal(FatalAmbiguityData),
}

/// Processes a single person for a single complex rule.
///
/// This is a "pure" function: it contains only the core business logic and is free
/// of I/O, locks, or other side-effects, making it easy to test and reason about.
#[inline]
fn process_person_for_rule(
    resolver: &ComplexVariantResolver,
    prep_result: &Arc<PreparationResult>,
    group_rule: &crate::types::GroupedComplexRule,
    person_output_idx: usize,
    original_fam_idx: u32,
    person_scores_slice: &mut [f64],
    person_counts_slice: &mut [u32],
) -> ResolutionOutcome {
    // --- Step 1: Gather Evidence ---
    let mut valid_interpretations = Vec::with_capacity(group_rule.possible_contexts.len());
    for context in &group_rule.possible_contexts {
        let (bim_idx, ..) = context;
        let packed_geno =
            resolver.get_packed_genotype(prep_result.bytes_per_variant, *bim_idx, original_fam_idx);
        if packed_geno != 0b01 {
            // 0b01 is the PLINK "missing" code
            valid_interpretations.push((packed_geno, context));
        }
    }

    // --- Step 2: Apply Decision Policy ---
    match valid_interpretations.len() {
        0 => {
            // Case A: No valid genotypes found for this locus.
            let mut counted_cols: AHashSet<ScoreColumnIndex> = AHashSet::new();
            for score_info in &group_rule.score_applications {
                counted_cols.insert(score_info.score_column_index);
            }
            for score_col in counted_cols {
                person_counts_slice[score_col.0] += 1;
            }
            ResolutionOutcome::Success
        }
        1 => {
            // Case B: Exactly one valid genotype found. Unambiguous happy path.
            let (winning_geno, winning_context) = valid_interpretations[0];
            let (_bim_idx, winning_a1, winning_a2) = winning_context;
            for score_info in &group_rule.score_applications {
                let effect_allele = &score_info.effect_allele;
                let score_col = score_info.score_column_index.0;
                let weight = score_info.weight as f64;
                if effect_allele != winning_a1 && effect_allele != winning_a2 {
                    continue;
                }
                let dosage: f64 = if effect_allele == winning_a1 {
                    match winning_geno {
                        0b00 => 2.0,
                        0b10 => 1.0,
                        0b11 => 0.0,
                        _ => unreachable!(),
                    }
                } else {
                    match winning_geno {
                        0b00 => 0.0,
                        0b10 => 1.0,
                        0b11 => 2.0,
                        _ => unreachable!(),
                    }
                };
                person_scores_slice[score_col] += dosage * weight;
            }
            ResolutionOutcome::Success
        }
        _ => {
            // Case C: A data conflict was detected.
            for score_info in &group_rule.score_applications {
                let matching_interpretations: Vec<_> = valid_interpretations
                    .iter()
                    .filter(|(_, context)| {
                        &score_info.effect_allele == &context.1
                            || &score_info.effect_allele == &context.2
                    })
                    .collect();
                match matching_interpretations.len() {
                    1 => {
                        let (winning_geno, winning_context) = matching_interpretations[0];
                        let (locus_id, winning_a1, winning_a2) = &**winning_context;
                        let dosage: f64 = if &score_info.effect_allele == winning_a1 {
                            match *winning_geno {
                                0b00 => 2.0,
                                0b10 => 1.0,
                                0b11 => 0.0,
                                _ => unreachable!(),
                            }
                        } else {
                            match *winning_geno {
                                0b00 => 0.0,
                                0b10 => 1.0,
                                0b11 => 2.0,
                                _ => unreachable!(),
                            }
                        };
                        person_scores_slice[score_info.score_column_index.0] +=
                            dosage * score_info.weight as f64;
                        // Return the raw data for the warning, not the formatted string.
                        return ResolutionOutcome::Warning(WarningInfo {
                            person_output_idx,
                            locus_id: *locus_id,
                            winning_a1: winning_a1.clone(),
                            winning_a2: winning_a2.clone(),
                            score_col_idx: score_info.score_column_index,
                        });
                    }
                    0 => continue,
                    // This is the fatal ambiguity case. We collect all necessary
                    // This case handles multiple, conflicting, non-missing genotype
                    // records that are all relevant to the current score.
                    _ => {
                        // --- HEURISTIC 1: Exact Score Allele Match ---
                        // Before checking dosages, see if there's one BIM entry that
                        // perfectly matches both alleles from the score file. This is
                        // the highest-confidence resolution strategy.
                        let score_eff_allele = &score_info.effect_allele;
                        let score_oth_allele = &score_info.other_allele;

                        let exact_matches: Vec<_> = matching_interpretations
                            .iter()
                            .filter(|(_, context)| {
                                let bim_a1 = &context.1;
                                let bim_a2 = &context.2;
                                (bim_a1 == score_eff_allele && bim_a2 == score_oth_allele)
                                    || (bim_a1 == score_oth_allele && bim_a2 == score_eff_allele)
                            })
                            .collect();

                        if exact_matches.len() == 1 {
                            // SUCCESS: Found a single, unambiguous BIM entry matching the score.
                            let (winning_geno, winning_context) = exact_matches[0];
                            let winning_a1 = &winning_context.1;

                            // Calculate dosage from this winning interpretation.
                            let chosen_dosage: f64 = if score_eff_allele == winning_a1 {
                                match *winning_geno {
                                    0b00 => 2.0,
                                    0b10 => 1.0,
                                    0b11 => 0.0,
                                    _ => unreachable!(),
                                }
                            } else {
                                match *winning_geno {
                                    0b00 => 0.0,
                                    0b10 => 1.0,
                                    0b11 => 2.0,
                                    _ => unreachable!(),
                                }
                            };
                            person_scores_slice[score_info.score_column_index.0] +=
                                chosen_dosage * score_info.weight as f64;

                            // Collect all original conflicting sources for the warning report.
                            let conflicts: Vec<ConflictSource> = matching_interpretations
                                .iter()
                                .map(|(packed_geno, context)| ConflictSource {
                                    bim_row: context.0,
                                    alleles: (context.1.clone(), context.2.clone()),
                                    genotype_bits: *packed_geno,
                                })
                                .collect();

                            return ResolutionOutcome::CriticalIntegrityWarning(
                                CriticalIntegrityWarningInfo {
                                    iid: prep_result.final_person_iids[person_output_idx].clone(),
                                    locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                    score_name: prep_result.score_names
                                        [score_info.score_column_index.0]
                                        .clone(),
                                    conflicts,
                                    resolution_method: ResolutionMethod::ExactScoreAlleleMatch {
                                        chosen_dosage,
                                    },
                                },
                            );
                        }

                        // --- HEURISTIC 2: Consistent Dosage (Fallback) ---
                        // If the exact match heuristic fails, proceed with existing heuristics.
                        // Calculate the dosage from each conflicting source to see if they are consistent.
                        let mut dosages = Vec::with_capacity(matching_interpretations.len());
                        for (geno, context) in &matching_interpretations {
                            let a1 = &context.1;
                            let dosage: f64 = if &score_info.effect_allele == a1 {
                                match geno {
                                    0b00 => 2.0,
                                    0b10 => 1.0,
                                    0b11 => 0.0,
                                    _ => unreachable!(),
                                }
                            } else {
                                match geno {
                                    0b00 => 0.0,
                                    0b10 => 1.0,
                                    0b11 => 2.0,
                                    _ => unreachable!(),
                                }
                            };
                            dosages.push(dosage);
                        }

                        // Check if all calculated dosages are identical.
                        let first_dosage = dosages[0];
                        if dosages.iter().all(|&d| (d - first_dosage).abs() < 1e-9) {
                            // BENIGN AMBIGUITY: The data is messy, but the outcome is the same.
                            // We can safely apply the score and issue a critical warning.
                            person_scores_slice[score_info.score_column_index.0] +=
                                first_dosage * score_info.weight as f64;

                            let conflicts: Vec<ConflictSource> = matching_interpretations
                                .iter()
                                .map(|(packed_geno, context)| ConflictSource {
                                    bim_row: context.0,
                                    alleles: (context.1.clone(), context.2.clone()),
                                    genotype_bits: *packed_geno,
                                })
                                .collect();

                            return ResolutionOutcome::CriticalIntegrityWarning(
                                CriticalIntegrityWarningInfo {
                                    iid: prep_result.final_person_iids[person_output_idx].clone(),
                                    locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                    score_name: prep_result.score_names
                                        [score_info.score_column_index.0]
                                        .clone(),
                                    conflicts,
                                    resolution_method: ResolutionMethod::ConsistentDosage {
                                        dosage: first_dosage,
                                    },
                                },
                            );
                        } else {
                            // The dosages were not consistent. As a final attempt, apply
                            // the "Prefer Heterozygous" heuristic.
                            let heterozygous_calls: Vec<_> = matching_interpretations
                                .iter()
                                .filter(|(geno, _)| *geno == 0b10)
                                .collect();
                            let homozygous_calls: Vec<_> = matching_interpretations
                                .iter()
                                .filter(|(geno, _)| *geno != 0b10)
                                .collect();

                            if heterozygous_calls.len() == 1 && !homozygous_calls.is_empty() {
                                // HEURISTIC APPLIED: A single het call is preferred over any
                                // number of conflicting homozygous calls.
                                let (winning_geno, winning_context) = heterozygous_calls[0];
                                let a1 = &winning_context.1;
                                let chosen_dosage: f64 = if &score_info.effect_allele == a1 {
                                    // This logic is duplicated from the simple path for clarity.
                                    match *winning_geno {
                                        0b10 => 1.0,
                                        _ => unreachable!(),
                                    }
                                } else {
                                    match *winning_geno {
                                        0b10 => 1.0,
                                        _ => unreachable!(),
                                    }
                                };

                                person_scores_slice[score_info.score_column_index.0] +=
                                    chosen_dosage * score_info.weight as f64;

                                let conflicts: Vec<ConflictSource> = matching_interpretations
                                    .iter()
                                    .map(|(packed_geno, context)| ConflictSource {
                                        bim_row: context.0,
                                        alleles: (context.1.clone(), context.2.clone()),
                                        genotype_bits: *packed_geno,
                                    })
                                    .collect();

                                return ResolutionOutcome::CriticalIntegrityWarning(
                                    CriticalIntegrityWarningInfo {
                                        iid: prep_result.final_person_iids[person_output_idx]
                                            .clone(),
                                        locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                        score_name: prep_result.score_names
                                            [score_info.score_column_index.0]
                                            .clone(),
                                        conflicts,
                                        resolution_method: ResolutionMethod::PreferHeterozygous {
                                            chosen_dosage,
                                        },
                                    },
                                );
                            } else {
                                // MALIGNANT AMBIGUITY: No further heuristics apply. The program MUST fail.
                                let conflicts: Vec<ConflictSource> = matching_interpretations
                                    .iter()
                                    .map(|(packed_geno, context)| ConflictSource {
                                        bim_row: context.0,
                                        alleles: (context.1.clone(), context.2.clone()),
                                        genotype_bits: *packed_geno,
                                    })
                                    .collect();

                                let data = FatalAmbiguityData {
                                    iid: prep_result.final_person_iids[person_output_idx].clone(),
                                    locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                    score_name: prep_result.score_names
                                        [score_info.score_column_index.0]
                                        .clone(),
                                    conflicts,
                                };
                                return ResolutionOutcome::Fatal(data);
                            }
                        }
                    }
                }
            }
            ResolutionOutcome::Success
        }
    }
}

/// The "slow path" resolver for complex, multiallelic variants.
///
/// This function runs *after* the main high-performance pipeline is complete. It
/// iterates through each person and resolves their score contributions for the small
/// set of variants that could not be handled by the fast path. It uses a rule-major
/// outer loop with a person-major parallel inner loop to provide granular progress.
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

    eprintln!("> Resolving {} complex variant rules...", num_rules);

    // This state must persist across all iterations of the rules loop. These are
    // thread-safe types that can be safely shared between threads.
    let fatal_error_occurred = Arc::new(AtomicBool::new(false));
    let fatal_error_storage = Mutex::new(None::<FatalAmbiguityData>);
    let warned_pairs = DashSet::<(usize, BimRowIndex)>::new();
    let all_warnings_to_print = Mutex::new(Vec::<WarningInfo>::new());
    let all_critical_warnings_to_print = Mutex::new(Vec::<CriticalIntegrityWarningInfo>::new());

    // Iterate through rules one by one to provide clear, sequential progress to the user.
    for (rule_idx, group_rule) in prep_result.complex_rules.iter().enumerate() {
        if fatal_error_occurred.load(Ordering::Relaxed) {
            break;
        }

        // This state is specific to the processing of a single rule.
        let pb = ProgressBar::new(prep_result.num_people_to_score as u64);
        let progress_style = ProgressStyle::with_template(&format!(
            ">  - Rule {:2}/{} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{eta}})",
            rule_idx + 1,
            num_rules
        ))
        .expect("Internal Error: Invalid progress bar template string.");
        pb.set_style(progress_style.progress_chars("█▉▊▋▌▍▎▏ "));

        let progress_counter = Arc::new(AtomicU64::new(0));

        // Use a scoped thread block to ensure the main thread waits for both the
        // workers and the progress updater to finish before proceeding. `thread::scope`
        // guarantees that any threads spawned within it will complete before the scope
        // exits, allowing safe borrowing of data from the parent stack.
        thread::scope(|s| {
            // Spawner #1: The dedicated progress bar updater thread.
            // This closure uses `move` because it only needs to own its copies of the
            // `Arc` pointers, which is a cheap and correct way to pass them.
            s.spawn({
                let pb_updater = pb.clone();
                let counter_for_updater = Arc::clone(&progress_counter);
                let error_flag_for_updater = Arc::clone(&fatal_error_occurred);
                let total_people = prep_result.num_people_to_score as u64;

                move || {
                    // This loop terminates under two conditions:
                    // 1. All work is complete (counter reaches total).
                    // 2. A fatal error has been signaled by a worker thread.
                    // This prevents the updater from hanging if workers stop early.
                    while counter_for_updater.load(Ordering::Relaxed) < total_people
                        && !error_flag_for_updater.load(Ordering::Relaxed)
                    {
                        pb_updater.set_position(counter_for_updater.load(Ordering::Relaxed));
                        thread::sleep(Duration::from_millis(200));
                    }
                    // Perform one final update to show the terminal state. If an error
                    // occurred, this accurately reflects how many items were processed
                    // before the operation was aborted.
                    pb_updater.set_position(counter_for_updater.load(Ordering::Relaxed));
                }
            });

            // Spawner #2: The worker threads (managed by Rayon).
            // This closure does NOT use `move`. It correctly borrows data from the
            // parent scope. This is safe because `thread::scope` guarantees this thread
            // cannot outlive the borrowed data (like `final_scores`, `warned_pairs`, etc.).
            s.spawn(|| {
                final_scores
                    .par_chunks_mut(prep_result.score_names.len())
                    .zip(final_missing_counts.par_chunks_mut(prep_result.score_names.len()))
                    .enumerate()
                    .for_each(
                        |(person_output_idx, (person_scores_slice, person_counts_slice))| {
                            // This guard ensures the progress counter is always incremented
                            // when the closure for a person finishes, regardless of how it exits
                            let _progress_guard = ScopeGuard::new(|| {
                                progress_counter.fetch_add(1, Ordering::Relaxed);
                            });

                            // This check provides a fast-fail mechanism, preventing new work
                            // from being done after a fatal error has been detected.
                            if fatal_error_occurred.load(Ordering::Relaxed) {
                                return;
                            }

                            let original_fam_idx =
                                prep_result.output_idx_to_fam_idx[person_output_idx];

                            let outcome = process_person_for_rule(
                                resolver,
                                prep_result,
                                group_rule,
                                person_output_idx,
                                original_fam_idx,
                                person_scores_slice,
                                person_counts_slice,
                            );

                            match outcome {
                                ResolutionOutcome::Success => {}
                                ResolutionOutcome::Warning(info) => {
                                    if warned_pairs.insert((info.person_output_idx, info.locus_id))
                                    {
                                        all_warnings_to_print.lock().unwrap().push(info);
                                    }
                                }
                                ResolutionOutcome::CriticalIntegrityWarning(info) => {
                                    // This is a non-fatal but severe warning. We collect it
                                    // to report at the end. We do not set the fatal error flag.
                                    all_critical_warnings_to_print.lock().unwrap().push(info);
                                }
                                ResolutionOutcome::Fatal(data) => {
                                    // Use compare_exchange to ensure only the FIRST fatal error
                                    // payload is stored. This prevents race conditions where multiple
                                    // threads might fail on different individuals simultaneously.
                                    if fatal_error_occurred
                                        .compare_exchange(
                                            false,
                                            true,
                                            Ordering::AcqRel,
                                            Ordering::Relaxed,
                                        )
                                        .is_ok()
                                    {
                                        *fatal_error_storage.lock().unwrap() = Some(data);
                                    }
                                    // `return` ensures we stop processing for this thread.
                                    return;
                                }
                            }
                        },
                    );
            });
        }); // Scope ends, all spawned threads are joined.

        pb.finish_with_message("Done.");
    }

    // After all rules are processed, drain and print a summary of warnings. This is
    // performed once by the main thread to avoid lock contention on stderr.
    let collected_warnings = std::mem::take(&mut *all_warnings_to_print.lock().unwrap());
    let collected_critical_warnings =
        std::mem::take(&mut *all_critical_warnings_to_print.lock().unwrap());
    const MAX_WARNINGS_TO_PRINT: usize = 10;

    let total_critical_warnings = collected_critical_warnings.len();
    if total_critical_warnings > 0 {
        eprintln!(
            "\n\n========================= CRITICAL DATA INTEGRITY ISSUE ========================="
        );
        eprintln!(
            "Gnomon detected one or more loci with conflicting genotype data that were\n\
                  resolved using a heuristic. While computation was able to continue, the\n\
                  underlying genotype data is ambiguous and should be investigated."
        );
        eprintln!(
            "---------------------------------------------------------------------------------"
        );
        for (i, info) in collected_critical_warnings.into_iter().enumerate() {
            if i >= MAX_WARNINGS_TO_PRINT {
                break;
            }
            if i > 0 {
                eprintln!(
                    "---------------------------------------------------------------------------------"
                );
            }
            eprintln!("{}", format_critical_integrity_warning(&info));
        }
        if total_critical_warnings > MAX_WARNINGS_TO_PRINT {
            eprintln!(
                "\n... and {} more similar critical warnings.",
                total_critical_warnings - MAX_WARNINGS_TO_PRINT
            );
        }
        eprintln!(
            "=================================================================================\n"
        );
    }

    let total_warnings = collected_warnings.len();
    if total_warnings > 0 {
        eprintln!("\n--- Data Inconsistency Warning Summary ---");
        for (i, info) in collected_warnings.into_iter().enumerate() {
            if i >= MAX_WARNINGS_TO_PRINT {
                break;
            }
            let iid = &prep_result.final_person_iids[info.person_output_idx];
            let score_name = &prep_result.score_names[info.score_col_idx.0];
            eprintln!(
                "WARNING: Resolved data inconsistency for IID '{}' at locus corresponding to BIM row {}. Multiple non-missing genotypes found. Used the one matching score '{}' (alleles: {}, {}).",
                iid, info.locus_id.0, score_name, info.winning_a1, info.winning_a2
            );
        }
        if total_warnings > MAX_WARNINGS_TO_PRINT {
            eprintln!(
                "... and {} more similar warnings.",
                total_warnings - MAX_WARNINGS_TO_PRINT
            );
        }
    }

    // After all rules are processed, check if a fatal error was ever stored.
    // If so, retrieve it from the mutex, format the final report, and propagate it.
    if fatal_error_occurred.load(Ordering::Relaxed) {
        if let Ok(mut guard) = fatal_error_storage.lock() {
            if let Some(data) = guard.take() {
                let report = format_fatal_ambiguity_report(&data);
                return Err(PipelineError::Compute(report));
            }
        }
        // As a fallback, return a generic error if the specific one can't be retrieved.
        return Err(PipelineError::Compute(
            "A fatal, unspecified error occurred in a parallel task and the detailed report could not be generated.".to_string(),
        ));
    }

    eprintln!("> Complex variant resolution complete.");
    Ok(())
}

/// A private helper function to format the final, dense data report for a fatal ambiguity.
/// This is called only once, on the main thread, after a fatal error is confirmed.
fn format_fatal_ambiguity_report(data: &FatalAmbiguityData) -> String {
    use std::fmt::Write;
    let mut report = String::with_capacity(512);

    // Helper to interpret genotype bits and alleles into a human-readable string.
    let interpret_genotype = |bits: u8, a1: &str, a2: &str| -> String {
        match bits {
            0b00 => format!("{}/{}", a1, a1),
            0b01 => "Missing".to_string(),
            0b10 => format!("{}/{}", a1, a2),
            0b11 => format!("{}/{}", a2, a2),
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
            "    Genotype Bits:   {} (Interpreted as {})",
            bits_str, interpretation
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
            0b00 => format!("{}/{}", a1, a1),
            0b01 => "Missing".to_string(),
            0b10 => format!("{}/{}", a1, a2),
            0b11 => format!("{}/{}", a2, a2),
            _ => "Invalid Bits".to_string(),
        }
    };

    // Build the final report string.
    writeln!(report, "Ambiguity Resolved for Individual '{}'", data.iid).unwrap();
    writeln!(
        report,
        "  Locus:  {}:{}",
        data.locus_chr_pos.0, data.locus_chr_pos.1
    )
    .unwrap();
    writeln!(report, "  Score:  {}", data.score_name).unwrap();

    match &data.resolution_method {
        ResolutionMethod::ConsistentDosage { dosage } => {
            writeln!(report, "  Method: 'Consistent Dosage' Heuristic").unwrap();
            writeln!(report, "  Outcome: All conflicting sources yielded a consistent dosage of {}, so computation continued.", dosage).unwrap();
        }
        ResolutionMethod::PreferHeterozygous { chosen_dosage } => {
            writeln!(report, "  Method: 'Prefer Heterozygous' Heuristic").unwrap();
            writeln!(report, "  Outcome: A single heterozygous call was chosen over conflicting homozygous call(s), yielding a dosage of {}.", chosen_dosage).unwrap();
        }
        ResolutionMethod::ExactScoreAlleleMatch { chosen_dosage } => {
            writeln!(report, "  Method: 'Exact Score Allele Match' Heuristic").unwrap();
            writeln!(report, "  Outcome: A single BIM entry's alleles matched the score file perfectly, yielding a dosage of {}.", chosen_dosage).unwrap();
        }
    }

    writeln!(report, "  \n  Conflicting Sources Found:").unwrap();

    for conflict in &data.conflicts {
        let interpretation = interpret_genotype(
            conflict.genotype_bits,
            &conflict.alleles.0,
            &conflict.alleles.1,
        );
        writeln!(
            report,
            "    - BIM Row {}: Alleles=({}, {}), Genotype={} ({})",
            conflict.bim_row.0,
            conflict.alleles.0,
            conflict.alleles.1,
            conflict.genotype_bits,
            interpretation
        )
        .unwrap();
    }

    report.trim_end().to_string()
}
