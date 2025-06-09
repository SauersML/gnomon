// ========================================================================================
//
//                       CORE DATA TYPES FOR THE GNOMON ENGINE
//
// ========================================================================================
//
// This module serves as the canonical dictionary for all data structures and types that
// are shared across the major architectural boundaries of the application (e.g.,
// `prepare`, `batch`, `main`).
//
// By centralizing these definitions, we create a single source of truth and enforce
// a clean, one-way dependency graph where high-level modules can depend on these
// core types, but not on each other's implementation details.

use crate::kernel;
use std::cell::RefCell;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::num::ParseFloatError;
use std::path::PathBuf;
use thread_local::ThreadLocal;

// ========================================================================================
//                            High-Level Data Contracts
// ========================================================================================

/// Defines the subset of individuals to be processed by the engine.
///
/// This enum makes the program's control flow explicit and type-safe. It is
/// impossible to ambiguously handle the "all vs. subset" cases.
#[derive(Debug)]
pub enum PersonSubset {
    /// Process all individuals from the .fam file.
    All,
    /// Process only a specific subset of individuals.
    /// Contains the sorted, 0-based indices from the original .fam file.
    Indices(Vec<u32>),
}

/// A "proof token" containing the complete, validated, and reconciled blueprint
/// for a polygenic score calculation.
///
/// The successful creation of this struct guarantees that all input files were
/// valid, consistent, and that there is a scientifically valid set of overlapping
/// markers to be processed. The engine takes this as an immutable reference and
/// can therefore operate with correct configuration.
#[derive(Debug)]
pub struct PreparationResult {
    /// The flattened, interleaved weight matrix, structured for extreme performance.
    /// Layout: `[S1_w1, S1_w2, ..., S2_w1, S2_w2, ...]`.
    pub interleaved_weights: Vec<f32>,
    /// The 0-based indices of the SNPs from the original .bim file that are
    /// required for the calculation, sorted in file order for sequential access.
    pub required_snp_indices: Vec<usize>,
    /// The corresponding set of reconciliation instructions for each required SNP.
    pub reconciliation_instructions: Vec<Reconciliation>,
    /// The names of the scores being calculated.
    pub score_names: Vec<String>,
    /// The total number of SNPs that were successfully reconciled.
    pub num_reconciled_snps: usize,
    /// The total number of SNPs found in the original .bim file.
    pub total_snps_in_bim: usize,
    /// The exact subset of individuals to be processed.
    pub person_subset: PersonSubset,
    /// The total number of individuals found in the original .fam file.
    pub total_people_in_fam: usize,
    /// The number of individuals that will actually be scored in this run.
    pub num_people_to_score: usize,
    /// The list of Individual IDs (IIDs) for the individuals being scored,
    /// in the final output order.
    pub final_person_iids: Vec<String>,
}

/// An instruction for how to handle a SNP's dosage based on its effect allele.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconciliation {
    /// Use the dosage as-is (effect allele is Allele 2).
    Identity,
    /// Flip the dosage: `2 - dosage` (effect allele is Allele 1).
    Flip,
}

// ========================================================================================
//                             Error and Type Definitions
// ========================================================================================

/// A comprehensive, production-grade error type for the preparation phase.
#[derive(Debug)]
pub enum PrepError {
    /// An error occurred during file I/O, with the associated file path.
    Io(io::Error, PathBuf),
    /// An error occurred parsing a text file (e.g., malformed lines).
    Parse(String),
    /// The header of a file was invalid or missing required columns.
    Header(String),
    /// The number of scores for a SNP did not match the header.
    InconsistentScores(String),
    /// An error occurred parsing a floating-point number.
    ParseFloat(ParseFloatError),
    /// One or more individual IDs from the keep file were not found in the .fam file.
    InconsistentKeepId(String),
}

impl Display for PrepError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            PrepError::Io(e, path) => write!(f, "I/O Error for file '{}': {}", path.display(), e),
            PrepError::Parse(s) => write!(f, "Parse Error: {}", s),
            PrepError::Header(s) => write!(f, "Invalid Header: {}", s),
            PrepError::InconsistentScores(s) => write!(f, "Inconsistent Data: {}", s),
            PrepError::ParseFloat(e) => write!(f, "Numeric Parse Error: {}", e),
            PrepError::InconsistentKeepId(s) => write!(f, "Configuration Error: {}", s),
        }
    }
}

impl Error for PrepError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PrepError::Io(e, _) => Some(e),
            PrepError::ParseFloat(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ParseFloatError> for PrepError {
    fn from(err: ParseFloatError) -> Self {
        PrepError::ParseFloat(err)
    }
}

/// An index into the original, full .fam file (e.g., one of 150,000).
///
/// This prevents confusion between different index spaces at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct OriginalPersonIndex(pub u32);

/// A `#[repr(transparent)]` wrapper for a dosage value, guaranteeing it is <= 2.
///
/// This type can only be constructed via its smart constructor, which ensures that
/// an invalid dosage value is an unrepresentable state within the engine.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectAlleleDosage(pub u8);

impl EffectAlleleDosage {
    /// Creates a new dosage, asserting the value is valid in debug builds.
    #[inline(always)]
    pub fn new(value: u8) -> Self {
        debug_assert!(value <= 2, "Invalid dosage value created: {}", value);
        Self(value)
    }
}

// ========================================================================================
//                                Shared Resource Pools
// ========================================================================================

/// A pool of reusable, thread-local buffers for the compute kernel.
///
/// This avoids allocator contention in the hot `rayon` loops by ensuring each
/// thread has its own set of scratch buffers. Created in `main`, consumed by `batch`.
pub type KernelDataPool =
    ThreadLocal<RefCell<(Vec<kernel::SimdVec>, Vec<usize>, Vec<usize>)>>;
