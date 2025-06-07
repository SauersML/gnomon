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
// the clean, logical world of the concurrent processing engine. `main.rs` remains a
// simple conductor, completely insulated from the complexity of parsing, validation,
// and data transformation.
//
// The core design principle is to produce a single, immutable "proof token"—the
// `PreparationResult` struct. The successful creation of this struct is a
// run-time guarantee that all data has been loaded, validated,
// scientifically reconciled, and structured for maximum performance.
//
// ---
//
// ### 2. Public API Specification ###
//
// This module exposes a minimal, powerful, and clear public interface.
//
// #### `pub enum Reconciliation`
// This is the public "instruction set" that this module generates. It dictates
// how the raw dosage from the genotype file must be handled to produce the
// scientifically correct dosage for the scoring formula.
//
// - `Identity`: Indicates the raw dosage can be used as-is (Genotype Allele #2 is
//   the Effect Allele).
// - `Flip`: Indicates the raw dosage must be inverted (`2 - dosage`) because
//   Genotype Allele #1 is the Effect Allele.
//
// #### `pub struct PreparationResult`
// This struct is the "proof token" returned on success. Its fields are a set of
// parallel vectors, all sorted identically by the physical row index of the SNPs
// in the `.bed` file. This sorting is a non-negotiable architectural contract
// required for the I/O engine's performance.
//
// - `interleaved_weights: Vec<f32>`: The final, memory-contiguous weight matrix,
//   transposed and sorted for the kernel. Weights are stored as-is, without any
//   mathematical adjustments.
//
// - `required_snp_indices: Vec<usize>`: A simple, sorted vector of the row indices
//   in the `.bed` file to be read. This is the exact, direct input that `io.rs`
//   needs to perform its physically sequential reads.
//
// - `reconciliation_instructions: Vec<Reconciliation>`: A vector of `Reconciliation`
//   enums, in the same sorted order. This is the instruction set for the **`io.rs`**
//   module. It allows the I/O engine to be the single source of truth for the
//   final, reconciled dosage.
//
// - `num_people`, `total_snps_in_bim`, `num_reconciled_snps`, `num_scores`:
//   Metadata fields required for allocating memory and controlling loops in the
//   main application logic.

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
#[derive(Debug)]
pub struct PreparationResult {
    /// The weight matrix, interleaved into a SNP-major layout for the compute kernel.
    pub interleaved_weights: Vec<f32>,
    /// The sorted list of physical row indices in the `.bed` file for `io.rs` to read.
    pub required_snp_indices: Vec<usize>,
    /// The parallel list of `Identity`/`Flip` instructions for `io.rs`.
    pub reconciliation_instructions: Vec<Reconciliation>,
    /// The total number of individuals in the genotype file.
    pub num_people: usize,
    /// The total number of SNPs in the `.bim` file, for validation.
    pub total_snps_in_bim: usize,
    /// The number of SNPs that were successfully reconciled between files.
    pub num_reconciled_snps: usize,
    /// The number of scores (i.e., weight columns) per SNP.
    pub num_scores: usize,
}

/// The single entry point for the preparation phase. It orchestrates file parsing,
/// scientific reconciliation, sorting, and data structuring.
pub fn prepare_for_computation(
    plink_prefix: &str,
    score_path: &str,
) -> Result<PreparationResult, PrepError> {
    // Formulate paths
    let fam_path = Path::new(&format!("{}.fam", plink_prefix));
    let bim_path = Path::new(&format!("{}.bim", plink_prefix));
    let score_path = Path::new(score_path);

    // "Parse" phase
    let num_people = parse_fam_file(fam_path)?;
    let (bim_map, total_snps_in_bim) = parse_bim_file(bim_path)?;

    // "Reconcile & Collect" phase
    let (num_scores, temp_reconciled) = reconcile_and_collect(score_path, &bim_map)?;
    let num_reconciled_snps = temp_reconciled.len();
    if num_reconciled_snps == 0 {
        return Err(PrepError::Parse(
            "No overlapping SNPs found between the score file and the genotype data.".to_string(),
        ));
    }

    // "Sort & Finalize" phase
    let (interleaved_weights, required_snp_indices, reconciliation_instructions) =
        sort_and_interleave(temp_reconciled, num_scores);

    // Assemble and return the final "proof token"
    Ok(PreparationResult {
        interleaved_weights,
        required_snp_indices,
        reconciliation_instructions,
        num_people,
        total_snps_in_bim,
        num_reconciled_snps,
        num_scores,
    })
}

// ========================================================================================
//                           PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

/// A temporary, intermediate representation of a reconciled SNP before sorting.
struct TempRecon {
    row_index: usize,
    instruction: Reconciliation,
    weights: Vec<f32>,
}

/// Helper 1: Parses the `.fam` file to get the number of individuals.
fn parse_fam_file(fam_path: &Path) -> Result<usize, PrepError> {
    let file = File::open(fam_path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines().count())
}

/// Helper 2: Parses the `.bim` file to create a fast SNP-lookup map and count total SNPs.
fn parse_bim_file(
    bim_path: &Path,
) -> Result<(AHashMap<String, (usize, u8, u8)>, usize), PrepError> {
    let file = File::open(bim_path)?;
    let reader = BufReader::new(file);

    let mut bim_map = AHashMap::new();
    let mut line_count = 0;

    for (line_number, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let mut parts = line.split_whitespace();

        // .bim format: <chromosome> <variant-id> <cm-position> <bp-position> <allele1> <allele2>
        let _chromosome = parts.next();
        let snp_id = parts.next();
        let _cm_pos = parts.next();
        let _bp_pos = parts.next();
        let allele1 = parts.next();
        let allele2 = parts.next();

        if let (Some(id), Some(a1_str), Some(a2_str)) = (snp_id, allele1, allele2) {
            // Optimization: Convert allele strings to single u8 bytes. This handles
            // the vast majority of SNPs ('A', 'C', 'T', 'G'). We treat multi-char
            // alleles (indels) as un-reconcilable for this high-performance path.
            // Consider addressing indels later.
            let a1_byte = a1_str.as_bytes().get(0).copied();
            let a2_byte = a2_str.as_bytes().get(0).copied();

            if let (Some(a1), Some(a2)) = (a1_byte, a2_byte) {
                if a1_str.len() == 1 && a2_str.len() == 1 {
                    bim_map.insert(id.to_string(), (line_number, a1, a2));
                }
            }
        } else {
            return Err(PrepError::Parse(format!(
                "Malformed line #{} in .bim file: {}",
                line_number + 1,
                line
            )));
        }
        line_count += 1;
    }

    Ok((bim_map, line_count))
}

/// Helper 3: Streams the score file, reconciles against the `.bim` map, and collects results.
fn reconcile_and_collect(
    score_path: &Path,
    bim_map: &AHashMap<String, (usize, u8, u8)>,
) -> Result<(usize, Vec<TempRecon>), PrepError> {
    let file = File::open(score_path)?;
    let mut reader = BufReader::new(file);

    // Read and parse the header to find the number of scores.
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let header_parts: Vec<&str> = header.split_whitespace().collect();
    if header_parts.len() < 2 || !header_parts.contains(&"effect_allele") {
        return Err(PrepError::Header(
            "Score file header is invalid. Must contain at least 'snp_id' and 'effect_allele' columns.".to_string(),
        ));
    }
    let num_scores = header_parts.len() - 2; // - snp_id, effect_allele

    let mut temp_reconciled = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let mut parts = line.split_whitespace();
        let snp_id = match parts.next() {
            Some(id) => id,
            None => continue, // Skip empty lines
        };
        let effect_allele_str = match parts.next() {
            Some(a) => a,
            None => continue, // Skip lines without an effect allele
        };

        // Skip non-single-char effect alleles
        if effect_allele_str.len() != 1 {
            continue;
        }
        let effect_allele_u8 = effect_allele_str.as_bytes()[0];

        // Look up the SNP in our BIM map. If it's not there, we can't use it.
        if let Some(&(row_index, bim_a1, bim_a2)) = bim_map.get(snp_id) {
            let weights: Vec<f32> = parts.map(|s| s.parse()).collect::<Result<_, _>>()?;
            if weights.len() != num_scores {
                return Err(PrepError::InconsistentScores(format!(
                    "SNP '{}' has {} weights, but header implies {}.",
                    snp_id,
                    weights.len(),
                    num_scores
                )));
            }

            let instruction = if effect_allele_u8 == bim_a2 {
                Reconciliation::Identity
            } else if effect_allele_u8 == bim_a1 {
                Reconciliation::Flip
            } else {
                // Alleles do not match, or are ambiguous. Skip this SNP.
                eprintln!(
                    "Warning: Skipping SNP '{}' due to allele mismatch (effect: {}, bim: {}/{})",
                    snp_id, effect_allele_str, bim_a1 as char, bim_a2 as char
                );
                continue;
            };

            temp_reconciled.push(TempRecon {
                row_index,
                instruction,
                weights,
            });
        }
    }
    Ok((num_scores, temp_reconciled))
}

/// Helper 4: Sorts the reconciled data and interleaves the weights into a final,
/// kernel-ready, SNP-major memory layout.
fn sort_and_interleave(
    mut temp_reconciled: Vec<TempRecon>,
    num_scores: usize,
) -> (Vec<f32>, Vec<usize>, Vec<Reconciliation>) {
    // This sort is the critical optimization that enables physically sequential reads in `io.rs`.
    temp_reconciled.sort_unstable_by_key(|item| item.row_index);

    let num_reconciled_snps = temp_reconciled.len();

    // Pre-allocate final vectors to their known sizes to avoid reallocations.
    let mut interleaved_weights = vec![0.0f32; num_reconciled_snps * num_scores];
    let mut required_snp_indices = Vec::with_capacity(num_reconciled_snps);
    let mut reconciliation_instructions = Vec::with_capacity(num_reconciled_snps);

    // Perform a single pass to "unzip" the temporary struct into the final, parallel vectors
    // and correctly interleave the weights for the compute kernel.
    for (snp_i, item) in temp_reconciled.into_iter().enumerate() {
        required_snp_indices.push(item.row_index);
        reconciliation_instructions.push(item.instruction);

        for (score_j, weight) in item.weights.into_iter().enumerate() {
            // The Correct SNP-Major Interleaving Formula:
            // All weights for a single SNP are contiguous in memory.
            let dest_idx = (snp_i * num_scores) + score_j;
            interleaved_weights[dest_idx] = weight;
        }
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
