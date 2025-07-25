use crate::pipeline::{PipelineError, ScopeGuard};
use crate::types::{BimRowIndex, FilesetBoundary, PipelineKind, PreparationResult, ScoreInfo};
use ahash::AHashSet;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

/// A read-only resolver for fetching complex variant genotypes.
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
            let (packed_geno, (_, bim_a1, bim_a2)) = exact_matches[0];
            let dosage = Self::calculate_dosage(*packed_geno, bim_a1, bim_a2, score_eff_allele);
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// **Heuristic 2:** Solves the `A/A` vs `AGA/AGA` conflict. Succeeds if
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
            let dosage = Self::calculate_dosage(packed_geno, bim_a1, bim_a2, score_eff_allele);
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
            .map(|(packed_geno, (_, bim_a1, bim_a2))| {
                Self::calculate_dosage(
                    *packed_geno,
                    bim_a1,
                    bim_a2,
                    &context.score_info.effect_allele,
                )
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
            let (packed_geno, (_, bim_a1, bim_a2)) = heterozygous_calls[0];
            let dosage = Self::calculate_dosage(
                *packed_geno,
                bim_a1,
                bim_a2,
                &context.score_info.effect_allele,
            );
            Some(Resolution {
                chosen_dosage: dosage,
                method_used: *self,
            })
        } else {
            None
        }
    }

    /// A private helper to compute dosage from raw PLINK bits.
    /// This function is now fully safe and self-contained, returning 0.0 if the
    /// effect allele is not one of the two alleles from the BIM entry.
    #[inline(always)]
    fn calculate_dosage(packed_geno: u8, bim_a1: &str, bim_a2: &str, effect_allele: &str) -> f64 {
        // Decodes the genotype with respect to the BIM alleles.
        let dosage_wrt_a1 = match packed_geno {
            0b00 => 2.0, // Homozygous for A1
            0b10 => 1.0, // Heterozygous (one A1, one A2)
            0b11 => 0.0, // Homozygous for A2
            _ => 0.0,    // Missing or invalid
        };

        if bim_a1 == effect_allele {
            // Case 1: The effect allele is A1. The dosage is the count of A1.
            dosage_wrt_a1
        } else if bim_a2 == effect_allele {
            // Case 2: The effect allele is A2. The dosage is the count of A2,
            // which is the inverse of the A1 count.
            2.0 - dosage_wrt_a1
        } else {
            // Case 3: The effect allele is neither A1 nor A2. The dosage must be 0.
            0.0
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
pub enum ResolutionMethod {
    /// All conflicting sources yielded the same effect allele dosage.
    ConsistentDosage { dosage: f64 },
    /// Exactly one heterozygous call was found alongside one or more homozygous
    /// calls, and the heterozygous call was chosen.
    PreferHeterozygous { chosen_dosage: f64 },
    /// A single BIM entry's alleles perfectly matched the score file alleles.
    ExactScoreAlleleMatch { chosen_dosage: f64 },
    /// A single interpretation was composed of standard alleles from the score file.
    PrioritizeUnambiguousGenotype { chosen_dosage: f64 },
}

/// A private struct holding the data for a critical but non-fatal integrity warning.
/// This is used when multiple data sources conflict but lead to a consistent outcome.
struct CriticalIntegrityWarningInfo {
    iid: String,
    locus_chr_pos: (String, u32),
    score_name: String,
    conflicts: Vec<ConflictSource>,
    resolution_method: ResolutionMethod,
    score_effect_allele: String,
    score_other_allele: String,
}

// The "slow path" resolver for complex variants.
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

    eprintln!("> Resolving {num_rules} complex variant rules...");

    let fatal_error_occurred = Arc::new(AtomicBool::new(false));
    let fatal_error_storage = Mutex::new(None::<FatalAmbiguityData>);
    let all_critical_warnings_to_print = Mutex::new(Vec::<CriticalIntegrityWarningInfo>::new());

    for (rule_idx, group_rule) in prep_result.complex_rules.iter().enumerate() {
        if fatal_error_occurred.load(Ordering::Relaxed) {
            break;
        }

        let pb = ProgressBar::new(prep_result.num_people_to_score as u64);
        let progress_style = ProgressStyle::with_template(&format!(
            ">  - Rule {:2}/{} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{eta}})",
            rule_idx + 1,
            num_rules
        ))
        .expect("Internal Error: Invalid progress bar template string.");
        pb.set_style(progress_style.progress_chars("█▉▊▋▌▍▎▏ "));
        let progress_counter = Arc::new(AtomicU64::new(0));

        let pipeline = ResolverPipeline::new();

        thread::scope(|s| {
            s.spawn({
                let pb_updater = pb.clone();
                let counter_for_updater = Arc::clone(&progress_counter);
                let error_flag_for_updater = Arc::clone(&fatal_error_occurred);
                let total_people = prep_result.num_people_to_score as u64;
                move || {
                    while counter_for_updater.load(Ordering::Relaxed) < total_people
                        && !error_flag_for_updater.load(Ordering::Relaxed)
                    {
                        pb_updater.set_position(counter_for_updater.load(Ordering::Relaxed));
                        thread::sleep(Duration::from_millis(200));
                    }
                    pb_updater.set_position(counter_for_updater.load(Ordering::Relaxed));
                }
            });

            s.spawn(|| {
                final_scores
                    .par_chunks_mut(prep_result.score_names.len())
                    .zip(final_missing_counts.par_chunks_mut(prep_result.score_names.len()))
                    .enumerate()
                    .for_each(|(person_output_idx, (person_scores_slice, person_counts_slice))| {
                        // Create a guard that will increment the counter when dropped
                        let guard = ScopeGuard::new(|| {
                            progress_counter.fetch_add(1, Ordering::Relaxed);
                        });
                        // Use the guard to avoid unused variable warning
                        let _ = &guard;

                        if fatal_error_occurred.load(Ordering::Relaxed) {
                            return;
                        }

                        let original_fam_idx = prep_result.output_idx_to_fam_idx[person_output_idx];

                        let valid_interpretations: Vec<_> = group_rule
                            .possible_contexts
                            .iter()
                            .filter_map(|context| {
                                let packed_geno = resolver.get_packed_genotype(
                                    prep_result.bytes_per_variant,
                                    context.0,
                                    original_fam_idx,
                                );
                                (packed_geno != 0b01).then_some((packed_geno, context))
                            })
                            .collect();

                        match valid_interpretations.len() {
                            0 => {
                                // Ensure we only increment the missing count ONCE per unique score
                                // column, even if a locus is used in multiple score file rows that
                                // map to the same final score.
                                let mut counted_cols = AHashSet::new();
                                for score_info in &group_rule.score_applications {
                                    counted_cols.insert(score_info.score_column_index);
                                }
                                for &score_col_idx in counted_cols.iter() {
                                    person_counts_slice[score_col_idx.0] += 1;
                                }
                            }
                            1 => {
                                let (packed_geno, context) = valid_interpretations[0];
                                let (_, bim_a1, bim_a2) = context;

                                for score_info in &group_rule.score_applications {
                                    // Add a guard to ensure the score's effect allele is actually
                                    // present in the single genotype context we found. If not, this
                                    // score rule is irrelevant, so we skip it. This prevents calls
                                    // to calculate_dosage with bad data.
                                    let effect_allele = &score_info.effect_allele;
                                    if effect_allele != bim_a1 && effect_allele != bim_a2 {
                                        continue;
                                    }

                                    let dosage = Heuristic::calculate_dosage(packed_geno, bim_a1, bim_a2, effect_allele);
                                    person_scores_slice[score_info.score_column_index.0] += dosage * score_info.weight as f64;
                                }
                            }
                            _ => {
                                for score_info in &group_rule.score_applications {
                                    // Performance: Filter interpretations to only those relevant to the current score.
                                    let matching_interpretations: Vec<_> = valid_interpretations.iter().copied().filter(|(_, context)| {
                                        score_info.effect_allele == context.1 || score_info.effect_allele == context.2
                                    }).collect();
                                    // If no interpretations match this score's alleles, skip to the next score.
                                    if matching_interpretations.is_empty() {
                                        continue;
                                    }

                                    let context = ResolutionContext {
                                        score_info,
                                        conflicting_interpretations: &matching_interpretations,
                                    };

                                    // Collect conflict details once before branching.
                                    let conflicts = matching_interpretations.iter().map(|(bits, ctx)| ConflictSource {
                                        bim_row: ctx.0,
                                        alleles: (ctx.1.clone(), ctx.2.clone()),
                                        genotype_bits: *bits,
                                    }).collect();

                                    if let Some(resolution) = pipeline.resolve(&context) {
                                        person_scores_slice[score_info.score_column_index.0] +=
                                            resolution.chosen_dosage * score_info.weight as f64;
                                        let resolution_method = match resolution.method_used {
                                            Heuristic::ExactScoreAlleleMatch => ResolutionMethod::ExactScoreAlleleMatch { chosen_dosage: resolution.chosen_dosage },
                                            Heuristic::PrioritizeUnambiguousGenotype => ResolutionMethod::PrioritizeUnambiguousGenotype { chosen_dosage: resolution.chosen_dosage },
                                            Heuristic::ConsistentDosage => ResolutionMethod::ConsistentDosage { dosage: resolution.chosen_dosage },
                                            Heuristic::PreferHeterozygous => ResolutionMethod::PreferHeterozygous { chosen_dosage: resolution.chosen_dosage },
                                        };

                                        all_critical_warnings_to_print.lock().unwrap().push(CriticalIntegrityWarningInfo {
                                            iid: prep_result.final_person_iids[person_output_idx].clone(),
                                            locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                            score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
                                            conflicts,
                                            resolution_method,
                                            score_effect_allele: score_info.effect_allele.clone(),
                                            score_other_allele: score_info.other_allele.clone(),
                                        });

                                    } else {
                                        // The entire pipeline failed. This is a fatal error.
                                        let data = FatalAmbiguityData {
                                            iid: prep_result.final_person_iids[person_output_idx].clone(),
                                            locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                            score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
                                            conflicts,
                                        };

                                        if fatal_error_occurred.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                                            *fatal_error_storage.lock().unwrap() = Some(data);
                                        }
                                        // Use `return` to exit the entire closure for this person, preventing further work.
                                        return;
                                    }
                                }
                            }
                        }
                    });
            });
        });

        pb.finish_with_message("Done.");
    }

    let collected_critical_warnings =
        std::mem::take(&mut *all_critical_warnings_to_print.lock().unwrap());
    const MAX_WARNINGS_TO_PRINT: usize = 10;

    if !collected_critical_warnings.is_empty() {
        eprintln!(
            "\n\n========================= CRITICAL DATA INTEGRITY ISSUE ========================="
        );
        eprintln!(
            "Gnomon detected loci with conflicting genotype data that were resolved\nusing a heuristic. While computation continued, the underlying data is\nambiguous and should be investigated."
        );
        eprintln!(
            "---------------------------------------------------------------------------------"
        );

        for (i, info) in collected_critical_warnings
            .iter()
            .enumerate()
            .take(MAX_WARNINGS_TO_PRINT)
        {
            if i > 0 {
                eprintln!(
                    "---------------------------------------------------------------------------------"
                );
            }
            eprintln!("{}", format_critical_integrity_warning(info));
        }

        if collected_critical_warnings.len() > MAX_WARNINGS_TO_PRINT {
            eprintln!(
                "\n... and {} more similar critical warnings.",
                collected_critical_warnings.len() - MAX_WARNINGS_TO_PRINT
            );
        }
        eprintln!(
            "=================================================================================\n"
        );
    }

    if fatal_error_occurred.load(Ordering::Relaxed) {
        if let Some(data) = fatal_error_storage.lock().unwrap().take() {
            return Err(PipelineError::Compute(format_fatal_ambiguity_report(&data)));
        }
        return Err(PipelineError::Compute(
            "A fatal, unspecified error occurred in a parallel task.".to_string(),
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

    // A helper to determine if a given conflict was the one chosen by the heuristic.
    let is_chosen = |method: &ResolutionMethod, conflict: &ConflictSource, score_ea: &str, score_oa: &str| -> bool {
        let bim_a1 = &conflict.alleles.0;
        let bim_a2 = &conflict.alleles.1;

        match method {
            ResolutionMethod::ExactScoreAlleleMatch {..} => {
                (bim_a1 == score_ea && bim_a2 == score_oa) || (bim_a1 == score_oa && bim_a2 == score_ea)
            }
            ResolutionMethod::PrioritizeUnambiguousGenotype {..} => {
                let is_a1_valid = bim_a1 == score_ea || bim_a1 == score_oa;
                let is_a2_valid = bim_a2 == score_ea || bim_a2 == score_oa;
                is_a1_valid && is_a2_valid
            }
            ResolutionMethod::PreferHeterozygous {..} => conflict.genotype_bits == 0b10,
            // The consistent dosage rule applies to all conflicts equally.
            ResolutionMethod::ConsistentDosage {..} => true,
        }
    };


    // Build the final report string.
    writeln!(report, "Ambiguity Resolved for Individual '{}'", data.iid).unwrap();
    writeln!(report, "  Locus:  {}:{}", data.locus_chr_pos.0, data.locus_chr_pos.1).unwrap();
    writeln!(report, "  Score:  {}", data.score_name).unwrap();

    let (method_name, rationale) = match &data.resolution_method {
        ResolutionMethod::ConsistentDosage { dosage } => ("'Consistent Dosage' Heuristic", format!("All conflicting sources yielded a consistent dosage of {dosage}, so computation continued.")),
        ResolutionMethod::PreferHeterozygous { .. } => ("'Prefer Heterozygous' Heuristic", "A single heterozygous call was chosen over conflicting homozygous call(s).".to_string()),
        ResolutionMethod::ExactScoreAlleleMatch { .. } => ("'Exact Score Allele Match' Heuristic", "A single BIM entry's alleles matched the score file perfectly.".to_string()),
        ResolutionMethod::PrioritizeUnambiguousGenotype { .. } => ("'Prioritize Unambiguous Genotype' Heuristic", "A single interpretation was chosen because its alleles matched the score file's standard alleles.".to_string()),
    };
    writeln!(report, "  Method: {}", method_name).unwrap();
    writeln!(report, "  Score File requires: Effect={}, Other={}", data.score_effect_allele, data.score_other_allele).unwrap();


    writeln!(report, "\n  Conflicting Sources Considered:").unwrap();

    for conflict in &data.conflicts {
        let prefix = if is_chosen(&data.resolution_method, conflict, &data.score_effect_allele, &data.score_other_allele) {
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

    writeln!(report, "\n  Rationale: {}", rationale).unwrap();


    report.trim_end().to_string()
}
