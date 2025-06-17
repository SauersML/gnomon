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
//
// This file is ONLY for types that are SHARED BETWEEN FILES, not types that only are used in one file.

use std::ops::{Deref, DerefMut};

// ========================================================================================
//                             ZERO-COST BUFFER STATE TYPES
// ========================================================================================

// These wrapper types encode a buffer's state (Clean or Dirty) into the type system,
// allowing the compiler to enforce correct usage at zero runtime cost. The `#[repr(transparent)]`
// attribute ensures they are identical to a `Vec` in memory layout.
/// A buffer that is guaranteed by the type system to have been zeroed.
#[derive(Debug)]
#[repr(transparent)]
pub struct CleanScores(pub Vec<f64>);
#[derive(Debug)]
#[repr(transparent)]
pub struct CleanCounts(pub Vec<u32>);

/// A buffer that may contain non-zero data from a previous computation.
#[derive(Debug)]
#[repr(transparent)]
pub struct DirtyScores(pub Vec<f64>);
#[derive(Debug)]
#[repr(transparent)]
pub struct DirtyCounts(pub Vec<u32>);

// --- State Transition & Ergonomics ---

impl DirtyScores {
    /// Consumes a dirty buffer and returns a clean one after zeroing its contents.
    /// This is the ONLY way to transition from a Dirty to a Clean state.
    pub fn into_clean(mut self) -> CleanScores {
        self.0.fill(0.0);
        CleanScores(self.0)
    }
}

impl Deref for DirtyScores {
    type Target = [f64];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<'a> IntoIterator for &'a DirtyScores {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    /// Enables direct iteration over the wrapped data.
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl DirtyCounts {
    /// Consumes a dirty buffer and returns a clean one after zeroing its contents.
    pub fn into_clean(mut self) -> CleanCounts {
        self.0.fill(0);
        CleanCounts(self.0)
    }
}

impl Deref for DirtyCounts {
    type Target = [u32];
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<'a> IntoIterator for &'a DirtyCounts {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    /// Enables direct iteration over the wrapped data.
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl CleanScores {
    /// Consumes a clean buffer, acknowledging it is now dirty after use.
    /// This is a zero-cost type change.
    pub fn into_dirty(self) -> DirtyScores {
        DirtyScores(self.0)
    }
}
impl Deref for CleanScores {
    type Target = [f64];
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl DerefMut for CleanScores {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

impl CleanCounts {
    /// Consumes a clean buffer, acknowledging it is now dirty after use.
    pub fn into_dirty(self) -> DirtyCounts {
        DirtyCounts(self.0)
    }
}
impl Deref for CleanCounts {
    type Target = [u32];
    fn deref(&self) -> &Self::Target { &self.0 }
}
impl DerefMut for CleanCounts {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

// ========================================================================================
//                            High-Level Data Contracts
// ========================================================================================

/// A buffer containing the raw, un-decoded genotype data for a single variant,
/// matching the PLINK .bed file's variant-major layout.
#[repr(transparent)]
#[derive(Debug)]
pub struct PackedVariantGenotypes(pub Vec<u8>);

/// The data payload of a non-empty `DenseVariantBatch`. This struct is public
/// to allow the pipeline orchestrator to own and move the data out of a
/// `DenseVariantBatch::Buffering` state. It contains all data required for the
/// person-major path, pre-gathered into contiguous buffers.
#[derive(Debug)]
pub struct DenseVariantBatchData {
    /// A contiguous buffer of packed variant genotype data.
    pub data: Vec<u8>,
    /// A contiguous buffer of pre-gathered, padded weights for the variants in this batch.
    pub weights: Vec<f32>,
    /// A contiguous buffer of pre-gathered, padded flips for the variants in this batch.
    pub flips: Vec<u8>,
    /// The number of variants in this batch. This is the source of truth for the batch size.
    pub variant_count: usize,
    /// The reconciled variant indices for each corresponding variant in the batch.
    /// This is used by the compute engine to look up global metadata for rare
    /// events like missingness handling.
    pub reconciled_variant_indices: Vec<ReconciledVariantIndex>,
}

/// A batch of raw variant data, curated to contain only "dense" variants
/// suitable for the high-throughput person-major compute path.
///
/// This enum enforces a state machine at compile time: a batch is either
/// `Empty` or it is `Buffering`. This makes it impossible to dispatch an empty
/// batch or add data to a batch that has already been dispatched, preventing a
/// class of logic errors.
#[derive(Debug)]
pub enum DenseVariantBatch {
    /// Represents a batch that has no variants. This is the initial state.
    Empty,
    /// Represents a batch that is actively accumulating variants.
    Buffering(DenseVariantBatchData),
}

impl DenseVariantBatch {
    /// Consumes the batch and returns the inner data if it was buffering,
    /// otherwise panics. This is used when a batch is known to be full.
    pub fn into_data(self) -> DenseVariantBatchData {
        match self {
            DenseVariantBatch::Buffering(data) => data,
            DenseVariantBatch::Empty => panic!("called `into_data()` on an `Empty` DenseVariantBatch"),
        }
    }
}

/// Represents the dispatcher's decision for which compute path to use for a
/// given unit of work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputePath {
    VariantMajor,
    PersonMajor,
}

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
/// compute engine will execute.
#[derive(Debug)]
pub struct PreparationResult {
    // --- COMPILED DATA & METADATA ---
    // The `variant_data` field stores weight/flip info per-variant, which is
    // efficient for the gather-on-assemble strategy used by the pipeline.
    // The fields are public to be read by downstream modules.

    /// A vector where each element is a tuple containing the padded `(weights, flips)`
    /// for a single reconciled variant. The index of this vector corresponds to the
    /// `ReconciledVariantIndex`.
    pub variant_data: Vec<(Vec<f32>, Vec<u8>)>,
    /// The padded width of a single variant's score data, rounded up to the
    /// nearest multiple of the SIMD vector width.
    pub padded_score_count: usize,

    // --- PUBLIC METADATA & LOOKUP TABLES ---
    /// The sorted list of original `.bim` row indices that are required for this
    /// calculation. This is used by the I/O producer to filter the `.bed` file.
    pub required_bim_indices: Vec<BimRowIndex>,
    /// A map from a person's original .fam index to their compact output index.
    /// `None` if the person is not in the scored subset.
    /// Essential for O(1) lookups in the variant-major path.
    pub person_fam_to_output_idx: Vec<Option<u32>>,
    /// The names of the scores being calculated, corresponding to matrix columns.
    pub score_names: Vec<String>,
    /// A vector containing the total number of variants that contribute to each score.
    /// This is used as the denominator for per-variant-average and missing % calculations.
    /// The order matches `score_names`.
    pub score_variant_counts: Vec<u32>,
    /// A "missingness blueprint" mapping each dense matrix row (variant) to the
    /// score column indices it affects.
    pub variant_to_scores_map: Vec<Vec<ScoreColumnIndex>>,
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
    pub total_variants_in_bim: usize,
    /// The number of variants successfully reconciled and included in the matrices.
    /// This corresponds to the number of rows in the compiled matrices.
    pub num_reconciled_variants: usize,
    /// The number of bytes per variant in the .bed file, calculated as CEIL(total_people_in_fam / 4).
    pub bytes_per_variant: u64,
}

impl PreparationResult {
    /// The constructor is crate-private, enforcing the "Airlock" pattern.
    /// Only the `prepare` module can construct this "proof token".
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        variant_data: Vec<(Vec<f32>, Vec<u8>)>,
        padded_score_count: usize,
        required_bim_indices: Vec<BimRowIndex>,
        score_names: Vec<String>,
        score_variant_counts: Vec<u32>,
        variant_to_scores_map: Vec<Vec<ScoreColumnIndex>>,
        person_subset: PersonSubset,
        final_person_iids: Vec<String>,
        num_people_to_score: usize,
        total_people_in_fam: usize,
        total_variants_in_bim: usize,
        num_reconciled_variants: usize,
        bytes_per_variant: u64,
        person_fam_to_output_idx: Vec<Option<u32>>,
    ) -> Self {
        Self {
            variant_data,
            padded_score_count,
            required_bim_indices,
            score_names,
            score_variant_counts,
            variant_to_scores_map,
            person_subset,
            final_person_iids,
            num_people_to_score,
            total_people_in_fam,
            total_variants_in_bim,
            num_reconciled_variants,
            bytes_per_variant,
            person_fam_to_output_idx,
        }
    }
}

// ========================================================================================
//                            Primitive Type Definitions
// ========================================================================================

/// An index into the original, full .fam file (e.g., one of 150,000).
///
/// This newtype prevents confusion between different index spaces at compile time.
/// The `#[repr(transparent)]` attribute guarantees this is a zero-cost abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct OriginalPersonIndex(pub u32);

/// An index into the original, full .bim file.
/// This wraps a u32, which is sufficient for >4 billion variants and halves the
/// memory usage of index vectors on 64-bit systems compared to usize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct BimRowIndex(pub u32);

/// An index into the dense, reconciled matrices (`weights_matrix`, `flip_mask_matrix`).
/// This wraps a u32, which is sufficient for >4 billion variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ReconciledVariantIndex(pub u32);

/// An index corresponding to a specific score (a column in the conceptual matrix).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ScoreColumnIndex(pub usize);

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
