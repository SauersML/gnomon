// ========================================================================================
//
//          HIGH-PERFORMANCE, SYNCHRONOUS, SINGLE-SNP MEMORY-MAPPED READER
//
// ========================================================================================
//
// ### Purpose ###
//
// This module provides a high-performance, synchronous reader for moving raw,
// sequential, single-SNP data rows from a memory-mapped .bed file. It operates
// on a `mmap`'d region for zero-copy reads from the OS page cache, which is
// ideal for the producer-consumer pipeline in the main orchestrator.

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::Path;

/// A high-performance reader for moving raw, sequential, single-SNP data rows
/// from a memory-mapped .bed file. It operates on a `mmap`'d region for zero-copy
/// reads from the OS page cache.
pub struct BedReader {
    /// A memory map of the entire .bed file. This provides a zero-cost "view" of
    /// the file on disk, not data loaded into RAM.
    mmap: Mmap,

    /// The current read position (in bytes) within the memory map. It is initialized
    /// to 3 to skip the PLINK magic number.
    pub cursor: u64,

    /// The total size of the file in bytes, cached at creation time.
    file_size: u64,

    /// The number of bytes per SNP, calculated once at construction. This is a
    /// critical piece of encapsulated state that prevents this logic from
    /// leaking into other modules.
    bytes_per_variant: u64,
}

impl BedReader {
    /// Creates a new `SnpReader` after performing critical validation.
    ///
    /// This constructor is the "airlock" for the raw .bed file. It guarantees that the
    /// file exists, is a valid PLINK .bed file, and has a size consistent with the
    /// metadata from the `prepare` phase, using overflow-safe arithmetic.
    pub fn new(bed_path: &Path, bytes_per_variant_arg: u64, num_variants: usize) -> io::Result<Self> {
        let bed_file = File::open(bed_path)?;
        let metadata = bed_file.metadata()?;
        let file_size = metadata.len();

        // The only unsafe block, justified because we will immediately validate the
        // mapped memory before it can be used.
        let mmap = unsafe { Mmap::map(&bed_file)? };

        // --- Validation Step 1: Magic Number (from the mmap itself) ---
        if mmap.get(0..3) != Some(&[0x6c, 0x1b, 0x01]) {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "Invalid .bed file magic number. The file may be corrupt or not a valid PLINK .bed file in SNP-major mode.",
            ));
        }

        // --- Validation Step 2: File Size (with checked arithmetic) ---
        // bytes_per_snp is now passed as an argument
        let total_variant_bytes = (num_variants as u64).checked_mul(bytes_per_variant_arg)
            .ok_or_else(|| {
                io::Error::new(ErrorKind::InvalidData, "Theoretical file size calculation overflowed (exceeds u64::MAX).")
            })?;

        let expected_size = 3u64.checked_add(total_variant_bytes)
            .ok_or_else(|| {
                io::Error::new(ErrorKind::InvalidData, "Theoretical file size calculation overflowed (exceeds u64::MAX).")
            })?;

        if file_size != expected_size {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!(
                    "BED file size mismatch. Metadata implies {} bytes, but file is {} bytes.",
                    expected_size, file_size
                ),
            ));
        }

        Ok(BedReader {
            mmap,
            cursor: 3, // Start reading after the 3-byte magic number.
            file_size,
            // Store the passed-in value.
            bytes_per_variant: bytes_per_variant_arg,
        })
    }

    /// Reads the data for the next single SNP.
    ///
    /// This function is designed for high-throughput, sequential reading. It takes an
    /// empty buffer from a pool, fills it with the data for exactly one SNP, and
    /// returns it. This allows the calling context to reuse buffer allocations.
    ///
    /// # Arguments
    /// * `buf`: A mutable `Vec` whose allocation will be used for the read. The
    ///          method takes ownership of the buffer's memory via `std::mem::take`,
    ///          leaving an empty `Vec` in its place.
    ///
    /// # Returns
    /// * `Ok(Some(Vec<u8>))`: On a successful read, returns the buffer now filled with SNP data.
    /// * `Ok(None)`: If the end of the file is reached.
    /// * `Err(e)`: On an I/O error.
    pub fn read_next_variant(&mut self, buf: &mut Vec<u8>) -> io::Result<Option<Vec<u8>>> {
        if self.cursor >= self.file_size {
            return Ok(None); // Graceful EOF
        }

        let bytes_per_variant = self.bytes_per_variant as usize;

        // Make sure there is enough data left in the file for one full SNP.
        if self.file_size - self.cursor < self.bytes_per_variant {
            // This case handles trailing bytes in a file that are not a full SNP.
            self.cursor = self.file_size;
            return Ok(None);
        }

        // Take ownership of the buffer's allocation, leaving an empty vector behind.
        let mut owned_buf = std::mem::take(buf);
        // We are about to fill the buffer completely.
        // We guarantee the length is correct before the copy.
        // This is safe because we just checked that enough bytes remain.
        unsafe {
            owned_buf.set_len(bytes_per_variant);
        }

        let src_end = self.cursor as usize + bytes_per_variant;
        let src_slice = &self.mmap[self.cursor as usize..src_end];

        owned_buf.copy_from_slice(src_slice);

        self.cursor += bytes_per_variant as u64;

        Ok(Some(owned_buf))
    }
}
