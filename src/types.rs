// ========================================================================================
//                             High-Level Data Contracts
// ========================================================================================

// This file is ONLY for types that are SHARED BETWEEN FILES, not types that only are used in one file.

/// The payload sent from the I/O producer to the compute consumers. It contains
/// the raw data for one variant and the necessary metadata to process it.
#[derive(Debug)]
pub struct WorkItem {
    pub data: Vec<u8>,
    pub reconciled_variant_index: ReconciledVariantIndex,
}

/// Contains the score-specific information for a complex variant.
/// This is the part of a rule that changes for each score file.
#[derive(Debug, Clone)]
pub struct ScoreInfo {
    /// The effect allele as defined in the score file. This could theoretically
    /// differ between scores for the same variant.
    pub effect_allele: String,
    /// The effect weight for this specific score.
    pub weight: f32,
    /// The global column index for the score this rule applies to.
    pub score_column_index: ScoreColumnIndex,
}

/// A self-contained, grouped unit of work for a single unique complex variant.
/// This struct holds information that is shared across all scores that use this variant.
#[derive(Debug, Clone)]
pub struct GroupedComplexRule {
    /// A list of all plausible BIM contexts (genotype definitions). This data is
    /// now stored only once per complex variant, not duplicated for every score.
    pub possible_contexts: Vec<(BimRowIndex, String, String)>, // (BimRowIndex, allele1, allele2)

    /// A list of all scores that apply to this set of contexts.
    pub score_applications: Vec<ScoreInfo>,
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
    // --- PRIVATE, COMPILED DATA MATRICES ---
    // These fields are private to guarantee their invariants. They are created once
    // by the `prepare` module and can only be read by downstream modules.
    weights_matrix: Vec<f32>,
    flip_mask_matrix: Vec<u8>,
    stride: usize,

    // --- PUBLIC METADATA & LOOKUP TABLES ---
    /// The sorted list of original `.bim` row indices for the "fast path."
    /// This is used by the I/O producer to filter the `.bed` file for all
    /// simple, unambiguous variants.
    pub required_bim_indices: Vec<BimRowIndex>,
    /// A list of self-contained, grouped rules for variants that require complex,
    /// deferred resolution (the "slow path"). Each rule represents one unique
    /// complex variant and all scores that use it.
    pub complex_rules: Vec<GroupedComplexRule>,
    /// A map from a person's original .fam index to their compact output index.
    /// `None` if the person is not in the scored subset.
    pub person_fam_to_output_idx: Vec<Option<u32>>,
    /// A map from the final, compact output index back to the original .fam file index.
    /// This is a critical optimization for the variant-major path.
    pub output_idx_to_fam_idx: Vec<u32>,
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
    pub fn new(
        weights_matrix: Vec<f32>,
        flip_mask_matrix: Vec<u8>,
        stride: usize,
        required_bim_indices: Vec<BimRowIndex>,
        complex_rules: Vec<GroupedComplexRule>,
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
        output_idx_to_fam_idx: Vec<u32>,
    ) -> Self {
        Self {
            weights_matrix,
            flip_mask_matrix,
            stride,
            required_bim_indices,
            complex_rules,
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
            output_idx_to_fam_idx,
        }
    }

    // --- Public Getters for Private Data ---

    #[inline(always)]
    pub fn weights_matrix(&self) -> &[f32] {
        &self.weights_matrix
    }

    #[inline(always)]
    pub fn flip_mask_matrix(&self) -> &[u8] {
        &self.flip_mask_matrix
    }

    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
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
