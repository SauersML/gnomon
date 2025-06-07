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


// ========================================================================================
//
//        THE HIGH-PERFORMANCE, MEMORY-MAPPED, SPARSE I/O ENGINE
//
// ========================================================================================
//
// ### 1. File-Level Purpose and Philosophy ###
//
// This module is the sole and exclusive point of contact with the filesystem for
// reading genotype data. It provides a high-level, clean abstraction to the rest of
// the application, which operates on logical data structures, not raw file bytes.
//
// ### 2. The "Negative Cost" Abstraction ###
//
// The core component, `MmapBedReader`, is not a simple file reader. It is an
// intelligent **Sparse Event Generator**. Instead of producing dense arrays of
// genotypes (which are mostly zero), it performs the work of identifying, decoding,
// and providing coordinates for only the necessary non-zero genotypes.
//
// This is a **"Negative Cost" Abstraction** because, by doing this more efficient
// work upfront, it ELIMINATES far more expensive operations in the calling module
// (`batch.rs`), namely:
//
//  1. **Eliminates Buffer Zeroing:** The pipeline buffers no longer need to be
//     zeroed-out on every chunk, saving a massive amount of memory bandwidth.
//  2. **Eliminates Branching:** The data processing logic in `batch.rs` is
//     transformed from a loop with conditional writes into a tight, branch-free
//     loop of simple memory writes, which is ideal for modern CPU pipelining.
//
// ### 3. Core Technology and Access Pattern ###
//
// This engine uses `memmap2` to handle potentially terabyte-scale `.bed` files with
// zero-copy I/O. It provides **logically random access** to any user-specified
// subset of SNPs while ensuring **physically sequential reads** from the disk. This
// is achieved by resolving all required SNP indices from the `.bim` file, sorting
// them, and then iterating through the memory-mapped file in a forward-only manner,
// maximizing the effectiveness of the OS's page cache and read-ahead algorithms.

use crate::batch::BedReader;
use memmap2::Mmap;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Result as IoResult};
use std::path::Path;

// ========================================================================================
//                              PUBLIC STRUCT DEFINITION
// ========================================================================================

pub struct MmapBedReader {
    /// A memory map of the entire .bed file. This is a zero-cost "portal" to the
    /// file on disk, not data loaded into RAM.
    mmap: Mmap,

    /// A sorted list of the ROW INDICES in the .bed file for the SNPs we actually need.
    /// Sorting is the key to quasi-sequential reads even with logical random access.
    required_snp_indices: Vec<usize>,

    /// Our current position in the `required_snp_indices` list.
    cursor: usize,

    /// The total number of individuals in the dataset, from the .fam file.
    num_people: usize,

    /// The number of bytes for one SNP's data, pre-calculated as `ceil(num_people / 4)`.
    bytes_per_snp: usize,
}

impl MmapBedReader {
    /// The "smart constructor" that encapsulates all setup and validation logic.
    /// It memory-maps the `.bed` file and prepares a sorted list of SNP indices
    /// to be read.
    ///
    /// # Arguments
    /// * `plink_prefix`: The path prefix for the `.bed`, `.bim`, and `.fam` files.
    /// * `required_snp_ids`: A `HashSet` of the SNP IDs required for the computation.
    ///
    /// # Errors
    /// Returns an `IoResult` if any file is missing, cannot be parsed, or if the
    /// file contents are inconsistent (e.g., file size does not match metadata).
    pub fn new(
        plink_prefix: &str,
        required_snp_ids: &HashSet<String>,
    ) -> IoResult<Self> {
        // --- Step 1: Path Construction & FAM Parsing ---
        let fam_path = Path::new(&format!("{}.fam", plink_prefix));
        let bim_path = Path::new(&format!("{}.bim", plink_prefix));
        let bed_path = Path::new(&format!("{}.bed", plink_prefix));

        let num_people = count_lines_in_file(fam_path)?;
        if num_people == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "FAM file is empty or could not be read."));
        }
        let bytes_per_snp = (num_people + 3) / 4;

        // --- Step 2: BIM Intersection and Index Building ---
        // This helper performs the efficient streaming intersection to find the indices
        // of the SNPs we need and returns them pre-sorted.
        let (required_snp_indices, total_snps_in_bim) =
            intersect_and_build_indices(bim_path, required_snp_ids)?;

        // --- Step 3: BED File Validation & mmap ---
        let bed_file = File::open(bed_path)?;
        let bed_metadata = bed_file.metadata()?;

        // Validate the 3-byte header.
        let mut header = [0u8; 3];
        // We use a local reader here just for the header, not for the main data.
        let mut reader = BufReader::new(&bed_file);
        reader.read_exact(&mut header)?;
        if header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid .bed file magic number or mode. Must be SNP-major."));
        }

        // Validate file size.
        let expected_size = 3 + total_snps_in_bim as u64 * bytes_per_snp as u64;
        if bed_metadata.len() != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "BED file size mismatch. Expected {} bytes based on FAM/BIM, but found {} bytes.",
                    expected_size,
                    bed_metadata.len()
                ),
            ));
        }

        // This is the only unsafe block, justified by the prior validation.
        let mmap = unsafe { Mmap::map(&bed_file)? };

        // --- Step 4: Instantiate and return the reader ---
        Ok(MmapBedReader {
            mmap,
            required_snp_indices,
            cursor: 0,
            num_people,
            bytes_per_snp,
        })
    }
}

// ========================================================================================
//                             TRAIT IMPLEMENTATION
// ========================================================================================

impl BedReader for MmapBedReader {
    /// Fulfills the `BedReader` contract by generating sparse events for the next
    /// required SNP in our sorted list.
    fn next_snp_events(
        &mut self,
        events_buffer: &mut Vec<(usize, u8)>,
        snp_idx_in_chunk: usize,
        num_snps_in_chunk: usize,
        num_people: usize,
    ) -> IoResult<bool> {
        // This assertion ensures the caller (`batch.rs`) and the reader (`io.rs`) agree
        // on the number of individuals. A mismatch is a fatal logic error.
        assert_eq!(self.num_people, num_people, "Logic error: mismatched number of people between I/O and batch layers.");

        // --- Step 1: Check if we have processed all required SNPs ---
        if self.cursor >= self.required_snp_indices.len() {
            return Ok(false); // Signal EOF to the producer.
        }

        // --- Step 2: Get the next physical row index and its data slice ---
        let bed_file_snp_index = self.required_snp_indices[self.cursor];
        let offset = 3 + bed_file_snp_index * self.bytes_per_snp;
        let snp_data_slice = &self.mmap[offset..offset + self.bytes_per_snp];

        // --- Step 3: Decode the slice and generate sparse events ---
        events_buffer.clear(); // Reuse the allocation.

        // This lookup table correctly decodes the 2-bit packed genotype data according
        // to the PLINK .bed specification.
        const MISSING_SENTINEL: u8 = u8::MAX;
        const DOSAGE_LOOKUP: [u8; 4] = [
            0,                // 00 -> Homozygous for allele #1 (A1/A1)
            MISSING_SENTINEL, // 01 -> Missing Genotype
            1,                // 10 -> Heterozygous (A1/A2)
            2,                // 11 -> Homozygous for allele #2 (A2/A2)
        ];

        for i in 0..self.num_people {
            let byte_index = i / 4;
            let bit_offset = (i % 4) * 2;
            let packed_val = (snp_data_slice[byte_index] >> bit_offset) & 0b11;
            let dosage = DOSAGE_LOOKUP[packed_val as usize];

            // Only create events for non-missing, non-zero dosages.
            if dosage == 1 || dosage == 2 {
                let dest_idx = i * num_snps_in_chunk + snp_idx_in_chunk;
                events_buffer.push((dest_idx, dosage));
            }
        }

        // --- Step 4: Advance cursor and signal success ---
        self.cursor += 1;
        Ok(true)
    }
}

// ========================================================================================
//                              PRIVATE HELPER FUNCTIONS
// ========================================================================================

/// Opens the specified file and counts the number of lines.
fn count_lines_in_file(path: &Path) -> IoResult<usize> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    Ok(reader.lines().count())
}

/// Streams through the `.bim` file to find the intersection of SNPs required by the
/// caller and available in the file. Returns a sorted `Vec` of the required row
//  indices and the total number of SNPs found in the file.
fn intersect_and_build_indices(
    bim_path: &Path,
    required_snp_ids: &HashSet<String>,
) -> IoResult<(Vec<usize>, usize)> {
    let file = File::open(bim_path)?;
    let reader = BufReader::new(file);

    let mut indices = Vec::new();
    let mut total_lines = 0;

    for (line_number, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        total_lines += 1;

        // The SNP ID is the second column, separated by whitespace.
        if let Some(snp_id) = line.split_whitespace().nth(1) {
            if required_snp_ids.contains(snp_id) {
                indices.push(line_number);
            }
        }
    }

    // This sort is the critical optimization that enables physically sequential reads.
    indices.sort_unstable();
    
    // Deduplication is necessary in case the input required_snp_ids contains duplicates
    // that resolve to the same index.
    indices.dedup();

    Ok((indices, total_lines))
}



