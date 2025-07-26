use crate::pipeline::{PipelineError, ScopeGuard};
use crate::types::{
    BimRowIndex, ConflictSource, CriticalIntegrityWarningInfo, FilesetBoundary, Heuristic,
    PipelineKind, PreparationResult, ResolutionMethod, ScoreInfo, PerThreadCollector,
    FinalAggregatedCollector
};
use ahash::AHashSet;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
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

impl Heuristic {
    /// The main dispatcher for the enum. It calls the appropriate private method
    /// for the specific heuristic variant.
    pub fn try_resolve(&self, context: &ResolutionContext) -> Option<Resolution> {
        match self {
            Heuristic::ExactScoreAlleleMatch => self.resolve_exact_match(context),
            Heuristic::PrioritizeUnambiguousGenotype => self.resolve_unambiguous_genotype(context),
            Heuristic::PreferMatchingAlleleStructure => self.resolve_prefer_matching_allele_structure(context),
            Heuristic::ConsistentDosage => self.resolve_consistent_dosage(context),
            Heuristic::PreferHeterozygous => self.resolve_prefer_het(context),
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
            let dosage = Self::calculate_dosage(*packed_geno, bim_a1, bim_a2, score_eff_allele);
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
    fn resolve_prefer_matching_allele_structure(&self, context: &ResolutionContext) -> Option<Resolution> {
        let score_a1_len = context.score_info.effect_allele.len();
        let score_a2_len = context.score_info.other_allele.len();

        let matching_structure_interpretations: Vec<_> = context
            .conflicting_interpretations
            .iter()
            .filter(|(packed_geno, (_, bim_a1, bim_a2))| {
                // Interpret the person's actual alleles first.
                let (person_allele_1, person_allele_2) = Self::interpret_person_alleles(*packed_geno, bim_a1, bim_a2);

                // Get the lengths of the person's alleles.
                let person_a1_len = person_allele_1.len();
                let person_a2_len = person_allele_2.len();

                // If the genotype was invalid/missing, the lengths will be 0, so this will not match.
                if person_a1_len == 0 { return false; }

                // Check for a match in either direction to handle swapped alleles.
                (person_a1_len == score_a1_len && person_a2_len == score_a2_len)
                    || (person_a1_len == score_a2_len && person_a2_len == score_a1_len)
            })
            .collect();

        // If we found exactly one interpretation with a matching allele structure, it's our winner.
        if matching_structure_interpretations.len() == 1 {
            let (packed_geno, (_, bim_a1, bim_a2)) = matching_structure_interpretations[0];
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
            let dosage = Self::calculate_dosage(packed_geno, bim_a1, bim_a2, score_eff_allele);
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
            _ => ("", ""),           // Represents a missing or invalid genotype
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

/// A private struct holding the complete, raw payload for a fatal ambiguity error.
/// Collecting this data first and formatting it once at the end is a key optimization.
struct FatalAmbiguityData {
    iid: String,
    locus_chr_pos: (String, u32),
    score_name: String,
    conflicts: Vec<ConflictSource>,
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
    let mut all_warnings_for_reporting: FinalAggregatedCollector = HashMap::new();

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

        let rule_level_warnings = thread::scope(|s| {
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
                    .try_fold(
                        || PerThreadCollector::new(),
                        |mut local_collector, (person_output_idx, (person_scores_slice, person_counts_slice))| {
                            let guard = ScopeGuard::new(|| {
                                progress_counter.fetch_add(1, Ordering::Relaxed);
                            });
                            let _ = &guard;

                            if fatal_error_occurred.load(Ordering::Relaxed) {
                                return Err(());
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
                                        let matching_interpretations: Vec<_> = valid_interpretations.iter().copied().filter(|(_, context)| {
                                            score_info.effect_allele == context.1 || score_info.effect_allele == context.2
                                        }).collect();
                                        if matching_interpretations.is_empty() {
                                            continue;
                                        }

                                        let context = ResolutionContext {
                                            score_info,
                                            conflicting_interpretations: &matching_interpretations,
                                        };

                                        if let Some(resolution) = pipeline.resolve(&context) {
                                            person_scores_slice[score_info.score_column_index.0] +=
                                                resolution.chosen_dosage * score_info.weight as f64;

                                            let (count, samples) = local_collector.entry(resolution.method_used).or_insert((0, Vec::new()));
                                            *count += 1;
                                            if samples.len() < 5 {
                                                let resolution_method = match resolution.method_used {
                                                    Heuristic::ExactScoreAlleleMatch => ResolutionMethod::ExactScoreAlleleMatch { chosen_dosage: resolution.chosen_dosage },
                                                    Heuristic::PrioritizeUnambiguousGenotype => ResolutionMethod::PrioritizeUnambiguousGenotype { chosen_dosage: resolution.chosen_dosage },
                                                    Heuristic::PreferMatchingAlleleStructure => ResolutionMethod::PreferMatchingAlleleStructure { chosen_dosage: resolution.chosen_dosage },
                                                    Heuristic::ConsistentDosage => ResolutionMethod::ConsistentDosage { dosage: resolution.chosen_dosage },
                                                    Heuristic::PreferHeterozygous => ResolutionMethod::PreferHeterozygous { chosen_dosage: resolution.chosen_dosage },
                                                };

                                                let conflicts = valid_interpretations.iter().map(|(bits, ctx)| ConflictSource {
                                                    bim_row: ctx.0,
                                                    alleles: (ctx.1.clone(), ctx.2.clone()),
                                                    genotype_bits: *bits,
                                                }).collect();

                                                samples.push(CriticalIntegrityWarningInfo {
                                                    iid: prep_result.final_person_iids[person_output_idx].clone(),
                                                    locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                                    score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
                                                    conflicts,
                                                    resolution_method,
                                                    score_effect_allele: score_info.effect_allele.clone(),
                                                    score_other_allele: score_info.other_allele.clone(),
                                                });
                                            }
                                        } else {
                                            let conflicts = valid_interpretations.iter().map(|(bits, ctx)| ConflictSource {
                                                bim_row: ctx.0,
                                                alleles: (ctx.1.clone(), ctx.2.clone()),
                                                genotype_bits: *bits,
                                            }).collect();

                                            let data = FatalAmbiguityData {
                                                iid: prep_result.final_person_iids[person_output_idx].clone(),
                                                locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                                score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
                                                conflicts,
                                            };

                                            if fatal_error_occurred.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                                                *fatal_error_storage.lock().unwrap() = Some(data);
                                            }
                                            return Err(());
                                        }
                                    }
                                }
                            }
                            Ok(local_collector)
                        },
                    )
                    .map(|collector_opt| collector_opt.unwrap_or_default())
                    .try_reduce(
                        || PerThreadCollector::new(),
                        |mut main_collector, thread_collector| {
                            for (heuristic, (count, samples)) in thread_collector {
                                let (main_count, main_samples) = main_collector.entry(heuristic).or_insert((0, Vec::new()));
                                *main_count += count;
                                if main_samples.len() < 5 {
                                    main_samples.extend(samples.into_iter().take(5 - main_samples.len()));
                                }
                            }
                            Ok(main_collector)
                        },
                    )
            })
        });

        pb.finish_with_message("Done.");

        if let Ok(Ok(warnings)) = rule_level_warnings {
            for (heuristic, (count, samples)) in warnings {
                let (total_count, total_samples) =
                    all_warnings_for_reporting.entry(heuristic).or_insert((0, Vec::new()));
                *total_count += count;
                if total_samples.len() < 5 {
                    total_samples.extend(samples.into_iter().take(5 - total_samples.len()));
                }
            }
        }
    }

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
    let is_chosen = |method: &ResolutionMethod, conflict: &ConflictSource, score_ea: &str, score_oa: &str| -> bool {
        let bim_a1 = &conflict.alleles.0;
        let bim_a2 = &conflict.alleles.1;
        match method {
            ResolutionMethod::ExactScoreAlleleMatch {..} => (bim_a1 == score_ea && bim_a2 == score_oa) || (bim_a1 == score_oa && bim_a2 == score_ea),
            ResolutionMethod::PrioritizeUnambiguousGenotype {..} => (bim_a1 == score_ea || bim_a1 == score_oa) && (bim_a2 == score_ea || bim_a2 == score_oa),
            ResolutionMethod::PreferMatchingAlleleStructure {..} => {
                let score_a1_len = score_ea.len();
                let score_a2_len = score_oa.len();

                // Re-create the logic from the actual heuristic
                let (person_allele_1, person_allele_2) = Heuristic::interpret_person_alleles(conflict.genotype_bits, &conflict.alleles.0, &conflict.alleles.1);
                let person_a1_len = person_allele_1.len();
                let person_a2_len = person_allele_2.len();

                if person_a1_len == 0 { return false; }

                (person_a1_len == score_a1_len && person_a2_len == score_a2_len)
                    || (person_a1_len == score_a2_len && person_a2_len == score_a1_len)
            }
            ResolutionMethod::PreferHeterozygous {..} => conflict.genotype_bits == 0b10,
            ResolutionMethod::ConsistentDosage {..} => true,
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
            if let Some(chosen) = data.conflicts.iter().find(|c| is_chosen(&data.resolution_method, c, &data.score_effect_allele, &data.score_other_allele)) {
                write!(rationale, "The interpretation with alleles ({}, {}) was chosen because it perfectly matches the score file.", chosen.alleles.0, chosen.alleles.1).unwrap();
            }
        }
        ResolutionMethod::PrioritizeUnambiguousGenotype { .. } => {
            method_name = "'Prioritize Unambiguous Genotype' Heuristic";
             if let Some(chosen) = data.conflicts.iter().find(|c| is_chosen(&data.resolution_method, c, &data.score_effect_allele, &data.score_other_allele)) {
                write!(rationale, "The interpretation with alleles ({}, {}) was chosen because both alleles are present in the score file's required set.", chosen.alleles.0, chosen.alleles.1).unwrap();
            }
        }
        ResolutionMethod::PreferMatchingAlleleStructure { .. } => {
            method_name = "'Prefer Matching Allele Structure' Heuristic";
            if let Some(c) = data.conflicts.iter().find(|c| is_chosen(&data.resolution_method, c, &data.score_effect_allele, &data.score_other_allele)) {
                // Re-interpret the genotype to get the person's actual alleles for the rationale.
                let (pa1, pa2) = Heuristic::interpret_person_alleles(c.genotype_bits, &c.alleles.0, &c.alleles.1);
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
        let interpretation = interpret_genotype(conflict.genotype_bits, &conflict.alleles.0, &conflict.alleles.1);
        writeln!(
            report, "{} BIM Row {}: Alleles=({}, {}), Genotype={} ({})",
            prefix, conflict.bim_row.0, conflict.alleles.0, conflict.alleles.1,
            conflict.genotype_bits, interpretation
        ).unwrap();
    }

    if !rationale.is_empty() {
        writeln!(report, "\n  Rationale: {}", rationale).unwrap();
    }
    
    report.trim_end().to_string()
}

