// ========================================================================================
//
//               THE LOGICALLY-RANDOM, PHYSICALLY-SEQUENTIAL I/O ENGINE
//
// ========================================================================================
//
// ### 1. Mission and Philosophy ###
//
// This module is the **sole authority on reading genotype data from disk**. It serves
// as a strict "airlock" between the physical file formats (`.bed`) and the logical
// processing pipeline in `batch.rs`.
//
// Its architecture is dictated by two non-negotiable principles:
//
//  a. **Logically Random, Physically Sequential Access:** The module provides the *illusion*
//     of random access, allowing the application to select any arbitrary subset of
//     SNPs. However, its internal design guarantees that all reads from the underlying
//     file are performed in a physically sequential, forward-only manner. This is
//     achieved by receiving a pre-sorted list of SNP indices from the `prepare.rs`
//     module and is the cornerstone of the engine's high-throughput performance.
//
//  b. **Abstraction (The Sparse, Reconciled Event Generator):** This
//     module's primary responsibility is to act as an intelligent "event generator".
//     It does not simply return raw data. It performs the complex, stateful work of:
//       1. Decoding the raw 2-bit genotype data.
//       2. Applying the scientifically critical "Flip" or "Identity" logic based on
//          the `Reconciliation` instructions provided by `prepare.rs`.
//       3. Emitting events **only for non-zero, reconciled dosages**.
//
//     By performing this work upfront,
//     it eliminates far more expensive operations in the `batch.rs` module, namely
//     the need to zero-out multi-megabyte buffers.
//
// ---
//
// ### 2. Public API Specification ###
//
// This module defines the `BedReader` trait and a "proof-carrying" newtype that
// form the unbreakable contract between the I/O layer and the batch processor.
//
// #### `pub trait BedReader`
// The central abstraction. It defines a source of reconciled genotype events.
//
// - `fn next_snp_events(&mut self, snp_idx_in_chunk: usize, num_snps_in_chunk: usize, events_buffer: &mut Vec<(usize, EffectAlleleDosage)>) -> IoResult<bool>`
//   - This is the entire contract. It is the responsibility of the caller
//     (`batch.rs`) to provide the correct dimensions of the destination chunk.
//     The I/O engine then calculates the final destination indices for its events.
//
// #### Proof-Carrying Newtype
// A zero-cost wrapper that uses the type system to enforce correctness.
//
// - `#[repr(transparent)] pub struct EffectAlleleDosage(pub u8);`
//   - Its existence proves that its internal `u8` is the final, scientifically
//     correct dosage, already reconciled according to the effect allele. It makes
//     it impossible for the batching layer to accidentally receive a raw,
//     un-reconciled dosage.
//
// ---
//
// ### 3. Concrete Implementation: `MmapBedReader` ###
//
// This is the struct that will implement the `BedReader` trait.
//
// #### Struct Definition
// - `mmap: Mmap`: The memory map of the entire `.bed` file.
// - `required_snp_indices: Vec<usize>`: The **sorted list** of physical row indices
//   to read from the `.bed` file, provided by `prepare.rs`.
// - `reconciliation_instructions: Vec<Reconciliation>`: A parallel vector, also
//   from `prepare.rs`, containing the `Identity` or `Flip` instruction for each
//   SNP in `required_snp_indices`.
// - `cursor: usize`: Tracks our position in the two parallel vectors above.
// - `num_people: usize`: The total number of individuals, provided by `prepare.rs`.
// - `bytes_per_snp: usize`: The pre-calculated size of one SNP's data.
//
// #### Public Constructor
// - `pub fn new(bed_path: &Path, num_people: usize, total_snps_in_bim: usize, required_indices: Vec<usize>, instructions: Vec<Reconciliation>) -> IoResult<Self>`
//   - This constructor is simpler and more efficient. It does no file parsing itself;
//     it receives all necessary metadata from the orchestrator (`main.rs`), which
//     gets it from `prepare.rs`.
//   - **Logic:**
//     1. Calculate `bytes_per_snp` from the provided `num_people`.
//     2. Open the `.bed` file at `bed_path`.
//     3. Validate the 3-byte header to ensure it's a valid, SNP-major file.
//     4. Validate the file size against the provided `num_people` and `total_snps_in_bim`
//        to ensure consistency and data integrity.
//     5. Create the `Mmap` of the file inside a tight `unsafe` block.
//     6. Instantiate and return `Self`, storing the `required_indices`, `instructions`,
//        and other metadata provided by the caller.
//
// #### Trait Implementation (`impl BedReader for MmapBedReader`)
// - `fn next_snp_events(...) -> IoResult<bool>`
//   - This is the core logic of the event generator.
//   - **Logic:**
//     1. **Check Cursor:** If `self.cursor` is at the end of the `required_snp_indices`
//        vector, return `Ok(false)` (EOF).
//     2. **Get Next Task:**
//        - `let bed_file_snp_index = self.required_snp_indices[self.cursor];`
//        - `let instruction = self.reconciliation_instructions[self.cursor];`
//     3. **Access Data:** Calculate the byte `offset` and get the `snp_data_slice`
//        from the `mmap`. This is a fast pointer operation.
//     4. **Decode, Reconcile, and Generate Events:**
//        a. Define the **correct `DOSAGE_LOOKUP` table** locally: `[0, MISSING_SENTINEL, 1, 2]`.
//        b. Loop from `person_idx = 0` to `self.num_people`.
//        c. Unpack the 2-bit value to get the `raw_dosage`.
//        d. If `raw_dosage == MISSING_SENTINEL`, `continue`.
//        e. **Apply Reconciliation:** `let final_dosage = match instruction { Reconciliation::Identity => raw_dosage, Reconciliation::Flip => 2 - raw_dosage };`
//        f. **Sparsity Check:** `if final_dosage > 0`.
//        g. **Calculate Index:** `let dest_idx = person_idx * num_snps_in_chunk + snp_idx_in_chunk;`
//        h. **Mint and Push:** Create the proof-carrying type and push to the buffer:
//           `events_buffer.push((dest_idx, EffectAlleleDosage(final_dosage)));`
//     5. **Advance State:** Increment `self.cursor` and return `Ok(true)`.
