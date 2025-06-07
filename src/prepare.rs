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
//   transposed and sorted for the kernel. The signs of the weights have already
//   been adjusted (negated) for any SNP requiring a `Flip`.
//
// - `required_snp_indices: Vec<usize>`: A simple, sorted vector of the row indices
//   in the `.bed` file to be read. This is the exact, direct input that `io.rs`
//   needs to perform its physically sequential reads.
//
// - `reconciliation_instructions: Vec<Reconciliation>`: A vector of `Reconciliation`
//   enums, in the same sorted order. This is the instruction set for the **`io.rs`**
//   module. It allows the I/O engine to be the single source of truth for the
//   final, reconciled dosage. `batch.rs` will be completely unaware of this logic.
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
// if any part of the preparation fails (e.g., file not found, parse error,
// allele mismatch).
//
// #### `pub fn prepare_for_computation(...) -> Result<PreparationResult, PrepError>`
// The single public entry point into the module. `main.rs` calls this function
// once at startup.
//
// ---
//
// ### 3. Detailed Internal Implementation Blueprint ###
//
// The public `prepare_for_computation` function will orchestrate a series of private,
// single-responsibility helpers. The core strategy is:
// **"Parse, Reconcile, Sort, Finalize."**
//
// #### Helper 1: `parse_fam_file(fam_path: &Path) -> Result<usize, PrepError>`
// - **Responsibility:** Determine the number of individuals.
// - **Logic:** Open the `.fam` file, wrap in a `BufReader`, and return `reader.lines().count()`.
//
// #### Helper 2: `parse_bim_and_get_count(bim_path: &Path) -> Result<(HashMap<String, (usize, u8, u8)>, usize), PrepError>`
// - **Responsibility:** Efficiently parse the entire `.bim` file **once** to create a
//   fast lookup map and get the total SNP count.
// - **Logic:**
//   1. Initialize `bim_map: HashMap<String, (usize, u8, u8)>` and `total_snp_count = 0`.
//   2. Stream the `.bim` file line-by-line. For each line:
//      - Increment `total_snp_count`.
//      - Parse the SNP ID (string), Allele 1, and Allele 2.
//      - **CRITICAL OPTIMIZATION:** Convert Allele 1 and 2 to `u8` bytes (e.g., `b'A'`) before storing.
//      - Insert into `bim_map`: `key = SNP_ID`, `value = (line_number - 1, allele1_u8, allele2_u8)`.
//   3. Return `Ok((bim_map, total_snp_count))`.
//
// #### Helper 3: `reconcile_and_collect(score_path: &Path, bim_map: &HashMap<...>) -> Result<(Vec<TempRecon>, Vec<f32>), PrepError>`
// - **Responsibility:** Iterate through the scores, perform scientific reconciliation,
//   make all necessary mathematical adjustments upfront, and collect the results
//   into a temporary, *unsorted* intermediate representation.
// - **Logic:**
//   1. Define a private struct: `struct TempRecon { row_index: usize, instruction: Reconciliation, adjusted_weights: Vec<f32> }`.
//   2. Initialize `temp_reconciled: Vec<TempRecon>` and `constant_score_offset: Vec<f32>`.
//   3. Stream the score file. For each SNP:
//      - Parse its ID, Effect Allele (as `u8`), and vector of weights.
//      - Look up the SNP in `bim_map`. If not found, skip it.
//      - Get `(row_index, allele1_u8, allele2_u8)` from the map.
//      - **Perform Reconciliation & Math:**
//        - If `effect_allele_u8 == allele2_u8`: instruction is `Identity`. Push a `TempRecon`
//          with the original weights.
//        - If `effect_allele_u8 == allele1_u8`: instruction is `Flip`.
//          - For each `weight` in the parsed weights: add `2.0 * weight` to the `constant_score_offset` vector.
//          - Create a new `adjusted_weights` vector where each weight is negated (`-w`).
//          - Push a `TempRecon` with `Reconciliation::Flip` and the `adjusted_weights`.
//        - If neither matches or alleles are ambiguous, log a warning and discard the SNP.
//   4. Return `Ok((temp_reconciled, constant_score_offset))`.
//
// #### Helper 4: `sort_and_interleave(mut temp_reconciled: Vec<TempRecon>, num_reconciled_snps: usize, num_scores: usize) -> (Vec<f32>, Vec<usize>, Vec<Reconciliation>)`
// - **Responsibility:** Take the unsorted reconciled data, sort it by physical file location,
//   and produce the final, parallel, interleaved data structures in a **single,
//   cache-efficient pass.** This function is the ultimate performance finalizer.
// - **Logic:**
//   1. **The Critical Sort:** `temp_reconciled.sort_unstable_by_key(|item| item.row_index);`
//   2. **Pre-Allocate Final Vectors:** Allocate `required_snp_indices`,
//      `reconciliation_instructions`, and `interleaved_weights` to their final, known sizes.
//   3. **Single-Pass "Unzip and Transpose":**
//      - Iterate through the sorted `temp_reconciled` with an index:
//        `for (snp_i, item) in temp_reconciled.iter().enumerate()`
//      - Inside the loop:
//        a. `required_snp_indices.push(item.row_index);`
//        b. `reconciliation_instructions.push(item.instruction);`
//        c. **Direct Interleaved Write:** Loop through the `item.adjusted_weights` and
//           write them directly to their final, transposed positions in the single
//           `interleaved_weights` vector. The formula for the destination index is:
//           `let dest_idx = score_j * num_reconciled_snps + snp_i;`
//           `interleaved_weights[dest_idx] = weight;`
//   4. Return the tuple of three completed, sorted, parallel vectors.
//
// #### Final Assembly in `prepare_for_computation`
// 1. Call `parse_fam_file` to get `num_people`.
// 2. Call `parse_bim_and_get_count` to get the `bim_map` and `total_bim_snps`.
// 3. Call `reconcile_and_collect` to get `temp_reconciled` and `constant_score_offset`.
// 4. Call `sort_and_interleave` with the results to get the final `interleaved_weights`,
//    `required_snp_indices`, and `reconciliation_instructions`.
// 5. Assemble the `PreparationResult` struct with all the final, prepared data.
// 6. Return `Ok(PreparationResult)`. Any error from the helpers is propagated up.

// TODO: CRITICAL BUG! The indexing formula `score_j * num_reconciled_snps + snp_i`
// described produces a SCORE-MAJOR memory layout. This is incorrect.
// The high-performance SIMD kernel (`kernel.rs`) is architected specifically for
// a SNP-MAJOR layout to enable linear memory reads. The correct formula, which enables
// the kernel to function as designed, MUST be `(snp_i * num_scores) + score_j`.
// Leaving this as-is will cause the kernel to read incorrect weights and produce
// completely invalid results.
