// ========================================================================================
//
//                      THE PRE-COMPUTATION & RECONCILIATION ENGINE
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
// compile-time and run-time guarantee that all data has been loaded, validated,
// scientifically reconciled, and structured for maximum performance.
//
// ---
//
// ### 2. Public API Specification ###
//
// This module exposes a minimal, powerful, and clear public interface.
//
// #### `pub enum Reconciliation`
// This is the public "instruction set" that this module generates for other parts
// of the system. It dictates how the raw dosage from the genotype file must be
// handled to produce the scientifically correct dosage for the scoring formula.
//
// - `Identity`: Indicates that the `.bed` file's dosage (a count of Allele #2)
//   can be used as-is because Allele #2 is the Effect Allele.
// - `Flip`: Indicates that the raw dosage must be inverted (`2 - dosage`) because
//   the genotype file's Allele #1 is the Effect Allele.
//
// #### `pub struct PreparationResult`
// This struct is the "proof token" returned on success. Its fields are a set of
// parallel vectors, all sorted identically by the physical row index of the SNPs
// in the `.bed` file. This sorting is a non-negotiable architectural contract
// required for the I/O engine's performance.
//
// - `interleaved_weights: Vec<f32>`: The final, memory-contiguous weight matrix,
//   transposed and sorted for the kernel. The signs of the weights have already
//   been adjusted (negated) for any SNP requiring a `Flip`.
//
// - `required_snp_indices: Vec<usize>`: A simple, sorted vector of the row indices
//   in the `.bed` file to be read. This is the exact, direct input that `io.rs`
//   needs to perform its physically sequential reads.
//
// - `reconciliation_instructions: Vec<Reconciliation>`: A vector of `Reconciliation`
//   enums, in the same sorted order as the other vectors. This is the instruction
//   set for `batch.rs`, telling it how to transform the raw dosage from `io.rs`.
//
// - `constant_score_offset: Vec<f32>`: A vector containing the constant offset
//   (`sum of 2*w` for all flipped SNPs) that must be added to each person's
//   final calculated scores. There is one offset value for each score. This final
//   addition is performed by `main.rs` after the pipeline completes.
//
// - `num_people`, `num_reconciled_snps`, `num_scores`: Metadata fields required
//   for allocating memory and controlling loops in the main application logic.
//
// #### `pub enum PrepError`
// A dedicated, comprehensive error type that provides clear, actionable feedback
// if any part of the preparation fails.
//
// #### `pub fn prepare_for_computation(...) -> Result<PreparationResult, PrepError>`
// The single public entry point into the module. `main.rs` calls this function
// once at startup.
//
// ---
//
// ### 3. Detailed Internal Implementation Blueprint ###
//
// The public `prepare_for_computation` function will achieve its result by
// orchestrating a series of private, single-responsibility helper functions.
// The core strategy is **"Reconcile First, Sort Second, Unzip Last."**
//
// #### Helper 1: `parse_fam_file(fam_path: &Path) -> Result<usize, PrepError>`
// - **Responsibility:** Determine the number of individuals.
// - **Logic:**
//   1. Open the `.fam` file.
//   2. Wrap in a `BufReader`.
//   3. Return `reader.lines().count()`.
//
// #### Helper 2: `parse_bim_file(bim_path: &Path) -> Result<(HashMap<String, (usize, String, String)>, usize), PrepError>`
// - **Responsibility:** Index all SNPs in the genotype data by their ID.
// - **Logic:**
//   1. Initialize an empty `HashMap<String, (usize, String, String)>` called `bim_map`.
//   2. Initialize `total_snp_count = 0`.
//   3. Open and stream the `.bim` file line-by-line. For each line:
//      - Increment `total_snp_count`.
//      - Parse the line (tab-separated).
//      - Extract the SNP ID (column 2), Allele 1 (col 5), and Allele 2 (col 6).
//      - Insert into `bim_map`: `key = SNP_ID`, `value = (line_number - 1, allele_1, allele_2)`.
//   4. Return `Ok((bim_map, total_snp_count))`.
//
// #### Helper 3: `reconcile_and_collect(score_path: &Path, bim_map: &HashMap<...>) -> Result<Vec<TempRecon>, PrepError>`
// - **Responsibility:** Iterate through the required scores, perform scientific
//   reconciliation against the available genotypes, and collect the results into a
//   temporary, *unsorted* intermediate representation.
// - **Logic:**
//   1. Define a private temporary struct:
//      `struct TempRecon { row_index: usize, instruction: Reconciliation, weights: Vec<f32> }`
//   2. Initialize `let mut temp_reconciled = Vec::new();`.
//   3. Open and stream the score file line-by-line (or use a CSV reader).
//   4. For each line (representing one SNP):
//      - Parse the SNP ID, Effect Allele, and vector of weights.
//      - Look up the SNP ID in the `bim_map`. If it doesn't exist, log a warning and `continue`.
//      - Get `(row_index, allele1, allele2)` from the map.
//      - **Perform Reconciliation:**
//        - If `effect_allele == allele2`, the instruction is `Reconciliation::Identity`.
//        - If `effect_allele == allele1`, the instruction is `Reconciliation::Flip`.
//        - If neither matches (or alleles are ambiguous, e.g., A/T vs. T/A), log a
//          warning and `continue` (discarding the SNP).
//      - If reconciliation was successful, create a `TempRecon` struct with the
//        `row_index`, the determined `instruction`, and the parsed `weights`, and
//        push it into the `temp_reconciled` vector.
//   5. Return `Ok(temp_reconciled)`.
//
// #### Helper 4: `sort_and_unzip(mut temp_reconciled: Vec<TempRecon>, num_scores: usize) -> (Vec<usize>, Vec<Reconciliation>, Vec<Vec<f32>>, Vec<f32>)`
// - **Responsibility:** Take the unsorted, reconciled data, sort it by physical
//   file location, and then "unzip" it into the final, parallel vectors required
//   by `PreparationResult`. This is where the mathematical adjustments happen.
// - **Logic:**
//   1. **The Critical Sort:** Perform the single most important optimization:
//      `temp_reconciled.sort_unstable_by_key(|item| item.row_index);`
//   2. **Initialize Final Vectors:** Create the empty vectors that will be returned,
//      including `let mut constant_score_offset = vec![0.0f32; num_scores];`.
//   3. **Single Pass Unzip:** Iterate through the now-sorted `temp_reconciled` vector once.
//      For each `item`:
//      a. Push `item.row_index` into `required_snp_indices`.
//      b. Push `item.instruction` into `reconciliation_instructions`.
//      c. **Apply Math:** Check `item.instruction`.
//         - If `Identity`, push the `item.weights` vector as-is into a temporary
//           `final_uninterleaved_weights` vector.
//         - If `Flip`:
//           i. For each `weight` in `item.weights`, add `2.0 * weight` to the
//              corresponding element in `constant_score_offset`.
//           ii. Create a new `flipped_weights` vector where each weight is negated.
//           iii. Push the `flipped_weights` vector into `final_uninterleaved_weights`.
//   4. Return the tuple of populated final vectors.
//
// #### Helper 5: `build_interleaved_weights(final_uninterleaved_weights: Vec<Vec<f32>>, num_reconciled_snps: usize, num_scores: usize) -> Vec<f32>`
// - **Responsibility:** Perform the final data layout transformation for the kernel.
// - **Logic:**
//   1. Allocate the final `interleaved_weights: Vec<f32>` with size
//      `num_reconciled_snps * num_scores`.
//   2. This is a standard matrix transpose. Iterate through the input vector of
//      vectors (which is SNP-major) and write the data into the output `Vec`
//      in an interleaved (score-major per SNP) format.
//
// #### Final Assembly in `prepare_for_computation`
// 1. Call `parse_fam_file`.
// 2. Call `parse_bim_file`.
// 3. Call `reconcile_and_collect`.
// 4. Call `sort_and_unzip`.
// 5. Call `build_interleaved_weights`.
// 6. Assemble the `PreparationResult` struct from the outputs of the helpers.
// 7. Return `Ok(PreparationResult)`.
