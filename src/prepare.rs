// ========================================================================================
//
//               THE PRE-COMPUTATION, RECONCILIATION & PREPARATION ENGINE
//
// ========================================================================================
//
// ### 1. Mission and Philosophy ###
//
// This module is the **sole authority on pre-computation**. Its mission is to be the
// application's "setup phase," transforming raw, un-validated user inputs (file
// paths) into a single, cohesive, and scientifically correct set of data structures
// ready for the high-performance pipeline.
//
// It serves as a strict "airlock" between the messy reality of file formats and
// the clean, logical world of the concurrent processing engine. It is responsible
// for not only structuring data for performance but also for preserving the
// metadata (individual and score IDs) required for the output.
//
// The core design principle is to produce a single, immutable "proof token"—the
// `PreparationResult` struct. The successful creation of this struct is a
// run-time guarantee that all data has been loaded, validated,
// scientifically reconciled, and structured for maximum performance.

use ahash::AHashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::Path;

// ========================================================================================
//                                  PUBLIC API
// ========================================================================================

/// An instruction for the I/O engine on how to handle a SNP's dosage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconciliation {
    /// Use the dosage as-is (effect allele is Allele 2).
    Identity,
    /// Flip the dosage: `2 - dosage` (effect allele is Allele 1).
    Flip,
}

/// A comprehensive error type for the preparation phase.
#[derive(Debug)]
pub enum PrepError {
    Io(io::Error),
    Parse(String),
    Header(String),
    InconsistentScores(String),
    ParseFloat(ParseFloatError),
}

/// The final, validated, and pipeline-ready data produced by this module.
/// This struct is a "proof token" that all data and metadata required for
/// the computation and final output are present and consistent.
#[derive(Debug)]
pub struct PreparationResult {
    /// The weight matrix, interleaved into a SNP-major layout for the compute kernel.
    pub interleaved_weights: Vec<f32>,
    /// The sorted list of physical row indices in the .bed file for `io.rs` to read.
    pub required_snp_indices: Vec<usize>,
    /// The parallel list of `Identity`/`Flip` instructions for `io.rs`.
    pub reconciliation_instructions: Vec<Reconciliation>,
    /// The total number of SNPs in the original .bim file, for validation.
    pub total_snps_in_bim: usize,
    /// The number of SNPs that were successfully reconciled between files.
    pub num_reconciled_snps: usize,
    /// A vector of Individual IDs (IID), preserving person identity.
    pub person_ids: Vec<String>,
    /// The names of the scores, extracted from the score file header.
    pub score_names: Vec<String>,
}

/// The single entry point for the preparation phase. It orchestrates file parsing,
/// scientific reconciliation, sorting, and data structuring.
pub fn prepare_for_computation(
    plink_prefix: &Path,
    score_path: &Path,
) -> Result<PreparationResult, PrepError> {
    // Formulate paths using type-safe, allocation-free methods.
    let fam_path = plink_prefix.with_extension("fam");
    let bim_path = plink_prefix.with_extension("bim");

    // --- PHASE 1: PARSE METADATA AND FLATTEN WEIGHTS ---
    // The order of parsing is intentional: parse metadata first to establish context.
    let person_ids = parse_fam_file(&fam_path)?;
    let (score_names, flat_weights, mut score_map, score_file_skipped_snps) =
        parse_scores_and_flatten(score_path)?;

    // --- PHASE 2: BUILD RECONCILIATION PLAN ---
    let (mut tasks, total_snps_in_bim, bim_file_skipped_snps) =
        build_reconciliation_plan(&bim_path, &mut score_map)?;

    // Report summary of skipped SNPs to the user. This is a non-negotiable contract.
    let total_skipped = score_file_skipped_snps + bim_file_skipped_snps;
    if total_skipped > 0 {
        eprintln!(
            "\nInfo: A total of {} SNPs were skipped due to data inconsistencies. See warnings above for details.",
            total_skipped
        );
    }

    let num_reconciled_snps = tasks.len();
    if num_reconciled_snps == 0 {
        return Err(PrepError::Parse(
            "No overlapping SNPs found between the score file and the genotype data.".to_string(),
        ));
    }

    // --- PHASE 3: THE CRITICAL SORT ---
    tasks.sort_unstable_by_key(|task| task.bed_row_index);

    // --- PHASE 4: THE FUSED FINALIZATION PASS ---
    let (interleaved_weights, required_snp_indices, reconciliation_instructions) =
        finalize_from_sorted_plan(tasks, &flat_weights, score_names.len());

    Ok(PreparationResult {
        interleaved_weights,
        required_snp_indices,
        reconciliation_instructions,
        total_snps_in_bim,
        num_reconciled_snps,
        person__ids: person_ids,
        score_names,
    })
}

// ========================================================================================
//                           PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

/// A temporary data structure holding the plan for a single SNP.
struct ReconciliationTask {
    bed_row_index: usize,
    instruction: Reconciliation,
    weights_start_index: usize,
}

/// Parses the .fam file to extract the Individual ID (IID) for each person.
fn parse_fam_file(fam_path: &Path) -> Result<Vec<String>, PrepError> {
    let file = File::open(fam_path)?;
    BufReader::new(file)
        .lines()
        .map(|line_result| {
            let line = line_result?;
            let mut parts = line.split_whitespace();
            // Skip the first field (FID).
            let _fid = parts
                .next()
                .ok_or_else(|| PrepError::Parse("Missing FID in .fam file".to_string()))?;
            // The second field is the IID, which we need.
            let iid = parts
                .next()
                .ok_or_else(|| PrepError::Parse("Missing IID in .fam file".to_string()))?;
            Ok(iid.to_string())
        })
        .collect()
}

/// Parses the score file, extracting score names and creating a flattened weight vector.
fn parse_scores_and_flatten(
    score_path: &Path,
) -> Result<(Vec<String>, Vec<f32>, AHashMap<String, (usize, u8)>, usize), PrepError> {
    let file = File::open(score_path)?;
    let mut reader = BufReader::new(file);
    let mut header_line = String::new();
    reader.read_line(&mut header_line)?;

    let mut header_parts = header_line.split_whitespace();
    // The first two columns ('snp_id', 'effect_allele') are metadata, not scores.
    // We confirm their presence but do not store them as score names.
    if header_parts.next().is_none() || header_parts.next().is_none() {
        return Err(PrepError::Header(
            "Score file header must contain at least 'snp_id' and 'effect_allele' columns."
                .to_string(),
        ));
    }

    // The rest of the header parts are the names of the scores.
    let score_names: Vec<String> = header_parts.map(|s| s.to_string()).collect();
    if score_names.is_empty() {
        return Err(PrepError::Header(
            "No score columns found in score file header after 'snp_id' and 'effect_allele'."
                .to_string(),
        ));
    }
    let num_scores = score_names.len();

    let mut flat_weights = Vec::new();
    let mut score_map = AHashMap::new();
    let mut skipped_snp_count = 0;

    for line_result in reader.lines() {
        let line = line_result?;
        let mut parts = line.split_whitespace();
        let snp_id = match parts.next() {
            Some(id) => id,
            None => continue,
        };
        let effect_allele_str = match parts.next() {
            Some(a) => a,
            None => continue,
        };

        if effect_allele_str.len() != 1 {
            eprintln!("Warning: Skipping SNP '{}' in score file due to invalid allele format (must be a single character).", snp_id);
            skipped_snp_count += 1;
            continue;
        }
        let effect_allele_u8 = effect_allele_str.as_bytes()[0];
        let weights_start_index = flat_weights.len();

        let mut weights_count = 0;
        for part in &mut parts {
            weights_count += 1;
            flat_weights.push(part.parse()?);
        }

        if weights_count != num_scores {
            return Err(PrepError::InconsistentScores(format!(
                "SNP '{}' has {} weights, but header implies {}.",
                snp_id, weights_count, num_scores
            )));
        }

        score_map.insert(
            snp_id.to_string(),
            (weights_start_index, effect_allele_u8),
        );
    }
    Ok((score_names, flat_weights, score_map, skipped_snp_count))
}

/// Reads the .bim file, compares against the score map, and builds a list of tasks.
fn build_reconciliation_plan(
    bim_path: &Path,
    score_map: &mut AHashMap<String, (usize, u8)>,
) -> Result<(Vec<ReconciliationTask>, usize, usize), PrepError> {
    let file = File::open(bim_path)?;
    let reader = BufReader::new(file);

    let mut tasks = Vec::new();
    let mut total_bim_snps = 0;
    let mut skipped_snp_count = 0;

    for (line_number, line_result) in reader.lines().enumerate() {
        total_bim_snps += 1;
        let line = line_result?;
        let mut parts = line.split_whitespace();

        let _chromosome = parts.next();
        let snp_id = parts.next();
        let _cm_pos = parts.next();
        let _bp_pos = parts.next();
        let allele1_str = parts.next();
        let allele2_str = parts.next();

        if let (Some(id), Some(a1_str), Some(a2_str)) = (snp_id, allele1_str, allele2_str) {
            // All alleles must be a single character to be valid. This is the hot path.
            if a1_str.len() == 1 && a2_str.len() == 1 {
                if let Some((weights_start_index, effect_allele_u8)) = score_map.remove(id) {
                    let bim_a1 = a1_str.as_bytes()[0];
                    let bim_a2 = a2_str.as_bytes()[0];

                    let instruction = if effect_allele_u8 == bim_a2 {
                        Reconciliation::Identity
                    } else if effect_allele_u8 == bim_a1 {
                        Reconciliation::Flip
                    } else {
                        eprintln!("Warning: Skipping SNP '{}' due to allele mismatch (effect: '{}', bim: '{}/{}').", id, effect_allele_u8 as char, bim_a1 as char, bim_a2 as char);
                        skipped_snp_count += 1;
                        continue;
                    };

                    tasks.push(ReconciliationTask {
                        bed_row_index: line_number,
                        instruction,
                        weights_start_index,
                    });
                }
            } else {
                eprintln!("Warning: Skipping SNP '{}' in .bim file due to invalid allele format (alleles: '{}'/'{}'). Alleles must be a single character.", id, a1_str, a2_str);
                skipped_snp_count += 1;
            }
        } else {
            return Err(PrepError::Parse(format!(
                "Malformed line #{} in .bim file: {}",
                line_number + 1,
                line
            )));
        }
    }
    Ok((tasks, total_bim_snps, skipped_snp_count))
}

/// Finalizes the data structures from the sorted plan. This is the "gather-scatter" pass.
fn finalize_from_sorted_plan(
    tasks: Vec<ReconciliationTask>,
    flat_weights: &[f32],
    num_scores: usize,
) -> (Vec<f32>, Vec<usize>, Vec<Reconciliation>) {
    let num_reconciled_snps = tasks.len();
    let mut interleaved_weights = vec![0.0f32; num_reconciled_snps * num_scores];
    let mut required_snp_indices = Vec::with_capacity(num_reconciled_snps);
    let mut reconciliation_instructions = Vec::with_capacity(num_reconciled_snps);

    for (snp_i, task) in tasks.into_iter().enumerate() {
        required_snp_indices.push(task.bed_row_index);
        reconciliation_instructions.push(task.instruction);
        let source_slice =
            &flat_weights[task.weights_start_index..task.weights_start_index + num_scores];
        let dest_start = snp_i * num_scores;
        let dest_end = dest_start + num_scores;
        let dest_slice = &mut interleaved_weights[dest_start..dest_end];
        dest_slice.copy_from_slice(source_slice);
    }
    (
        interleaved_weights,
        required_snp_indices,
        reconciliation_instructions,
    )
}

// ========================================================================================
//                                 ERROR HANDLING
// ========================================================================================

impl fmt::Display for PrepError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PrepError::Io(e) => write!(f, "I/O Error: {}", e),
            PrepError::Parse(s) => write!(f, "Parse Error: {}", s),
            PrepError::Header(s) => write!(f, "Invalid Header: {}", s),
            PrepError::InconsistentScores(s) => write!(f, "Inconsistent Data: {}", s),
            PrepError::ParseFloat(e) => write!(f, "Float Parse Error: {}", e),
        }
    }
}

impl Error for PrepError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PrepError::Io(e) => Some(e),
            PrepError::ParseFloat(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for PrepError {
    fn from(err: io::Error) -> Self {
        PrepError::Io(err)
    }
}

impl From<ParseFloatError> for PrepError {
    fn from(err: ParseFloatError) -> Self {
        PrepError::ParseFloat(err)
    }
}
