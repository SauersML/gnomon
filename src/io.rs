// ========================================================================================
//
//              HIGH-PERFORMANCE, SYNCHRONOUS, ENCAPSULATED BLOCK READER
//
// ========================================================================================
//
// ### Purpose ###
//
// This module provides a high-performance, synchronous reader for moving raw,
// sequential chunks of data from a memory-mapped .bed file.

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::Path;

/// A self-contained, validated chunk of SNP-major data.
///
/// The existence of this struct is a guarantee that `buffer.len()` is a
/// multiple of `bytes_per_snp` and that `num_snps` is correct. This makes
/// inconsistent states unrepresentable in the rest of the application.
pub struct SnpChunk {
    pub buffer: Vec<u8>,
    pub num_snps: usize,
}

/// A high-performance reader for moving raw, sequential chunks of data from a
/// memory-mapped .bed file. It acts as a factory for producing valid `SnpChunk`s.
pub struct SnpChunkReader {
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
    bytes_per_snp: u64,
}

impl SnpChunkReader {
    /// Creates a new `SnpChunkReader` after performing critical validation.
    ///
    /// This constructor is the "airlock" for the raw .bed file. It guarantees that the
    /// file exists, is a valid PLINK .bed file, and has a size consistent with the
    /// metadata from the `prepare` phase, using overflow-safe arithmetic.
    pub fn new(bed_path: &Path, bytes_per_snp_arg: u64, num_snps: usize) -> io::Result<Self> {
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
        let total_snp_bytes = (num_snps as u64).checked_mul(bytes_per_snp_arg)
            .ok_or_else(|| {
                io::Error::new(ErrorKind::InvalidData, "Theoretical file size calculation overflowed (exceeds u64::MAX).")
            })?;

        let expected_size = 3u64.checked_add(total_snp_bytes)
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

        Ok(SnpChunkReader {
            mmap,
            cursor: 3, // Start reading after the 3-byte magic number.
            file_size,
            // Store the passed-in value.
            bytes_per_snp: bytes_per_snp_arg,
        })
    }

    /// Reads the next chunk of SNP data, producing a validated `SnpChunk`.
    ///
    /// This is a "smart factory" method. It takes a mutable buffer to use for I/O,
    /// fills it with a non-partial amount of SNP data, and returns a type-safe
    /// struct that guarantees correctness.
    ///
    /// # Arguments
    /// * `buf`: A mutable `Vec` which will be used for the read. Its capacity
    ///          is respected. The method takes ownership of the buffer's memory
    ///          via `std::mem::take`, leaving an empty `Vec` in its place.
    ///
    /// # Returns
    /// * `Ok(Some(SnpChunk))`: On a successful read.
    /// * `Ok(None)`: If the end of the file is reached.
    /// * `Err(e)`: On an I/O error.
    pub fn read_next_chunk(&mut self, buf: &mut Vec<u8>) -> io::Result<Option<SnpChunk>> {
        if self.cursor >= self.file_size {
            return Ok(None); // Graceful EOF
        }

        let bytes_remaining = self.file_size - self.cursor;
        let max_bytes_to_read = (buf.capacity() as u64).min(bytes_remaining) as usize;

        if max_bytes_to_read == 0 {
            return Ok(None);
        }

        // Round down to the nearest whole SNP. This is the core of the correctness guarantee.
        let num_snps = if self.bytes_per_snp > 0 {
            max_bytes_to_read / self.bytes_per_snp as usize
        } else {
            0
        };

        if num_snps == 0 {
            // Not enough space in the remaining file for even one full SNP.
            return Ok(None);
        }

        let final_bytes_to_copy = num_snps * self.bytes_per_snp as usize;

        // Take ownership of the buffer's allocation, leaving an empty vector behind.
        let mut owned_buf = std::mem::take(buf);
        // Use unsafe set_len because we are about to fill it. This is faster than resize.
        unsafe {
            owned_buf.set_len(final_bytes_to_copy);
        }

        let src_end = self.cursor as usize + final_bytes_to_copy;
        let src_slice = &self.mmap[self.cursor as usize..src_end];

        owned_buf.copy_from_slice(src_slice);

        self.cursor += final_bytes_to_copy as u64;

        Ok(Some(SnpChunk {
            buffer: owned_buf,
            num_snps,
        }))
    }
}
