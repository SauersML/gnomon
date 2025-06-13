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

/// A "proof token" containing the complete, validated, and "compiled" blueprint for a
/// polygenic score calculation.
///
/// The successful creation of this struct is a guarantee that all input files were
/// valid and consistent. It contains the hyper-optimized data matrices that the
/// compute kernel (the "Virtual Machine") will execute.
#[derive(Debug)]
pub struct PreparationResult {
    // --- COMPILED DATA MATRICES for the VM ---
    /// A flat, padded matrix of aligned weights (W'). Rows correspond to variants,
    /// and columns correspond to scores. This matrix is the result of resolving all
    /// allele differences into a single, canonical representation.
    pub aligned_weights_matrix: Vec<f32>,
    /// A vector where each element is the sum of all dosage-independent correction
    /// constants for the corresponding score. This value is used to initialize
    /// each person's score before the dosage-dependent calculations begin.
    pub base_scores: Vec<f32>,
    /// The padded width of one row in the matrices. This is always a multiple of
    /// the SIMD lane count and is essential for correct, safe memory access.
    pub stride: usize,
    /// A fast, O(1) lookup map to connect a sparse, original `.bim` file row
    /// index to its dense row index in our compiled matrices. This is a critical
    /// performance component for the batching module.
    pub bim_row_to_matrix_row: Vec<Option<usize>>,
    /// The sorted list of original `.bim` row indices that are required for this
    /// calculation. This is used by the orchestrator to identify relevant chunks
    /// of the `.bed` file.
    pub required_bim_indices: Vec<usize>,

    // --- PRESERVED METADATA & FINAL CALCULATION DATA ---
    /// The names of the scores being calculated, corresponding to matrix columns.
    pub score_names: Vec<String>,
    /// A vector containing the total number of variants that contribute to each score.
    /// This is used as the denominator for per-variant-average and missing % calculations.
    /// The order matches `score_names`.
    pub score_variant_counts: Vec<u32>,
    /// A "missingness blueprint" mapping each dense matrix row (variant) to the
    /// score column indices it affects. This enables efficient missingness tracking.
    pub variant_to_scores_map: Vec<Vec<u16>>,
    /// The exact subset of individuals to be processed.
    pub person_subset: PersonSubset,
    /// The list of Individual IDs (IIDs) for the individuals being scored, in the
    /// final output order.
    pub final_person_iids: Vec<String>,
    /// The number of individuals that will actually be scored in this run.
    pub num_people_to_score: usize,
    /// The total number of individuals found in the original .fam file.
    pub total_people_in_fam: usize,
    /// The total number of variants found in the original .bim file.
    pub total_snps_in_bim: usize,
    /// The number of variants successfully reconciled and included in the matrices.
    /// This corresponds to the number of rows in the compiled matrices.
    pub num_reconciled_variants: usize,
    /// The number of bytes per SNP in the .bed file, calculated as CEIL(total_people_in_fam / 4).
    pub bytes_per_snp: u64,
}

// ========================================================================================
//                            Primitive Type Definitions
// ========================================================================================

/// An index into the original, full .fam file (e.g., one of 150,000).
///
/// This prevents confusion between different index spaces at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct OriginalPersonIndex(pub u32);

/// A `#[repr(transparent)]` wrapper for a dosage value.
///
/// This type is used in the pivoted tile buffer. Its `u8` representation is
/// compact and efficient for the pivoting process.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EffectAlleleDosage(pub u8);

impl EffectAlleleDosage {
    /// Creates a new dosage, asserting the value is valid in debug builds.
    #[inline(always)]
    pub fn new(value: u8) -> Self {
        debug_assert!(value <= 3, "Invalid dosage value created: {}", value);
        Self(value)
    }
}
