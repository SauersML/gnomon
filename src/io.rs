// ========================================================================================
//
//               THE HARDENED I/O AND VALIDATION ENGINE
//
// ========================================================================================
//
// ### 1. Mission and Philosophy ###
//
// This module is the "airlock" between the raw, untrusted data on disk and our
// validated, type-safe pipeline. It is the first and last line of defense against
// data corruption. Its purpose is not merely to read bytes, but to be the **sole
// authority on the validity of raw genotype data**.
//
// ### 2. Architectural Mandates ###
//
//   - **Zero-Trust Validation:** The `MmapBedReader` operates on a zero-trust
//     principle. It assumes input files may be corrupt. It meticulously validates
//     every 2-bit genotype code against the PLINK1 specification. Any deviation
//     is treated as an unrecoverable error, causing an immediate panic.
//
//   - **Immutable Contract Enforcement:** This module is the loyal implementation of
//     the `BedReader` trait defined in `batch.rs`. It respects the sealed,
//     type-safe `EffectAlleleDosage` newtype, using its fallible constructor as
//     a validation checkpoint. This ensures that only provably valid dosage
//     values can ever enter the pipeline.
//
//   - **Clarity over Terse-ness:** The decoding logic favors explicit,
//     self-documenting `match` statements over opaque lookup tables. The flow of
//     control must be undeniable at a glance.

use crate::batch::{BedReader, EffectAlleleDosage};
use crate::prepare::Reconciliation;
use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Read, Result as IoResult};
use std::path::Path;

// ========================================================================================
//                              PUBLIC STRUCT DEFINITION
// ========================================================================================

pub struct MmapBedReader {
    /// A memory map of the entire .bed file. This is a zero-cost "portal" to the
    /// file on disk, not data loaded into RAM.
    mmap: Mmap,

    /// A sorted list of the ROW INDICES in the .bed file for the SNPs we actually need.
    /// This list is provided, pre-sorted, by the `prepare` module.
    required_snp_indices: Vec<usize>,

    /// A parallel vector containing the `Identity` or `Flip` instruction for each
    /// corresponding SNP in `required_snp_indices`.
    reconciliation_instructions: Vec<Reconciliation>,

    /// Our current position in the `required_snp_indices` and `reconciliation_instructions` vectors.
    cursor: usize,

    /// The total number of individuals in the dataset, provided by the `prepare` module.
    num_people: usize,

    /// The number of bytes for one SNP's data, pre-calculated as `ceil(num_people / 4)`.
    bytes_per_snp: usize,
}

impl MmapBedReader {
    /// The constructor for the I/O engine. It performs no file parsing itself besides
    /// validating and mapping the `.bed` file. It relies on the caller to provide all
    /// necessary, pre-computed metadata.
    ///
    /// # Arguments
    /// * `bed_path`: The path to the `.bed` file.
    /// * `num_people`: The total number of individuals in the study.
    /// * `total_snps_in_bim`: The total number of SNPs in the corresponding `.bim` file, used for validation.
    /// * `required_snp_indices`: A **sorted** `Vec` of the physical row indices to read.
    /// * `reconciliation_instructions`: A parallel `Vec` with the `Flip`/`Identity` instruction for each index.
    ///
    /// # Errors
    /// Returns an `IoResult` if the `.bed` file is missing, malformed, or if its size is inconsistent
    /// with the provided metadata.
    pub fn new(
        bed_path: &Path,
        num_people: usize,
        total_snps_in_bim: usize,
        required_snp_indices: Vec<usize>,
        reconciliation_instructions: Vec<Reconciliation>,
    ) -> IoResult<Self> {
        // This assertion is a critical guard against logic errors in the calling module.
        assert_eq!(
            required_snp_indices.len(),
            reconciliation_instructions.len(),
            "Logic error: Mismatched lengths for SNP indices and reconciliation instructions."
        );

        let bytes_per_snp = (num_people + 3) / 4;

        // --- BED File Validation & mmap ---
        let bed_file = File::open(bed_path)?;
        let bed_metadata = bed_file.metadata()?;

        // Validate the 3-byte header.
        let mut header = [0u8; 3];
        let mut header_reader = bed_file.try_clone()?;
        header_reader.read_exact(&mut header)?;
        if header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid .bed file magic number or mode. Must be SNP-major.",
            ));
        }

        // Validate file size against the provided metadata.
        let expected_size = 3 + total_snps_in_bim as u64 * bytes_per_snp as u64;
        if bed_metadata.len() != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "BED file size mismatch. Expected {} bytes based on metadata, but found {} bytes.",
                    expected_size,
                    bed_metadata.len()
                ),
            ));
        }

        // This is the only unsafe block, justified by the prior validation of the file handle and size.
        let mmap = unsafe { Mmap::map(&bed_file)? };

        Ok(MmapBedReader {
            mmap,
            required_snp_indices,
            reconciliation_instructions,
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
    /// Fulfills the `BedReader` contract. For the next required SNP, it decodes the
    /// raw data, applies the correct reconciliation logic, and generates sparse
    /// events containing the final, scientifically-correct dosage. This function
    /// acts as a validation gatekeeper.
    ///
    /// # Panics
    /// This function **will panic** if it encounters a genotype code in the `.bed`
    /// file that does not conform to the PLINK1 specification (i.e., a raw dosage
    /// value greater than 2). This indicates unrecoverable data corruption in the
    /// input file, and immediate termination is the only correct action.
    fn next_snp_events(
        &mut self,
        events_buffer: &mut Vec<(usize, EffectAlleleDosage)>,
        snp_idx_in_chunk: usize,
        num_snps_in_chunk: usize,
        num_people: usize,
    ) -> IoResult<bool> {
        // This assertion ensures the caller (`batch.rs`) and the reader (`io.rs`) agree
        // on the number of individuals. A mismatch is a fatal logic error.
        assert_eq!(
            self.num_people, num_people,
            "Logic error: mismatched number of people between I/O and batch layers."
        );

        if self.cursor >= self.required_snp_indices.len() {
            return Ok(false); // Signal EOF to the producer.
        }

        let bed_file_snp_index = self.required_snp_indices[self.cursor];
        let instruction = self.reconciliation_instructions[self.cursor];

        let offset = 3 + bed_file_snp_index * self.bytes_per_snp;
        let snp_data_slice = &self.mmap[offset..offset + self.bytes_per_snp];

        events_buffer.clear();

        for i in 0..self.num_people {
            let byte_index = i / 4;
            let bit_offset = (i % 4) * 2;
            let packed_val = (snp_data_slice[byte_index] >> bit_offset) & 0b11;

            // Directly and explicitly decode the 2-bit genotype code. This replaces
            // the opaque lookup table with undeniable, self-documenting logic.
            let raw_dosage = match packed_val {
                0b00 => 0, // Homozygous for allele #1 (A1/A1)
                0b10 => 1, // Heterozygous (A1/A2)
                0b11 => 2, // Homozygous for allele #2 (A2/A2)
                0b01 => continue, // Missing Genotype. Skip this person for this SNP.
                _ => unreachable!(), // Masking with `& 0b11` makes other values impossible.
            };

            let final_dosage = match instruction {
                Reconciliation::Identity => raw_dosage,
                Reconciliation::Flip => 2 - raw_dosage,
            };

            if final_dosage > 0 {
                let dest_idx = i * num_snps_in_chunk + snp_idx_in_chunk;

                // Use the fallible constructor for EffectAlleleDosage as a validation gate.
                // The PLINK spec guarantees a valid dosage is <= 2. If this is not the
                // case, the file is corrupt, and we must panic immediately.
                let valid_dosage = EffectAlleleDosage::new(final_dosage).expect(
                    "Fatal Error: Corrupt .bed file detected. Found impossible dosage value (>2).",
                );

                events_buffer.push((dest_idx, valid_dosage));
            }
        }

        self.cursor += 1;
        Ok(true)
    }
}
