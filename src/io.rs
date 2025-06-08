// ========================================================================================
//
//                      HIGH-THROUGHPUT BLOCK READER
//
// ========================================================================================

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::Path;

// ========================================================================================
//                                  PUBLIC API
// ========================================================================================

/// A high-performance reader for moving raw, sequential chunks of data from a
/// memory-mapped .bed file.
pub struct SnpChunkReader {
    /// A memory map of the entire .bed file. This provides a zero-cost "view" of
    /// the file on disk, not data loaded into RAM.
    mmap: Mmap,

    /// The current read position (in bytes) within the memory map. It is initialized
    /// to 3 to skip the PLINK magic number.
    cursor: u64,

    /// The total size of the file in bytes, cached at creation time.
    file_size: u64,
}

impl SnpChunkReader {
    /// Creates a new `SnpChunkReader` after performing critical validation.
    ///
    /// This constructor is the "airlock" for the raw .bed file. It guarantees that the
    /// file exists, is a valid PLINK .bed file, and has a size consistent with the
    /// metadata from the `prepare` phase, using overflow-safe arithmetic.
    ///
    /// # Arguments
    /// * `bed_path`: The path to the `.bed` file.
    /// * `num_people`: The total number of individuals in the study, from `.fam` file.
    /// * `num_snps`: The total number of SNPs in the study, from `.bim` file.
    ///
    /// # Errors
    /// Returns an `io::Error` if the file is missing, malformed, or if its size is
    /// inconsistent with the provided metadata, preventing the engine from ever
    /// operating on corrupt or mismatched data.
    pub fn new(bed_path: &Path, num_people: usize, num_snps: usize) -> io::Result<Self> {
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
        let bytes_per_snp = (num_people as u64 + 3) / 4;
        let total_snp_bytes = (num_snps as u64).checked_mul(bytes_per_snp)
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
        })
    }

    /// Reads the next chunk of raw SNP data into the provided buffer.
    ///
    /// This method is allocation-free. The caller owns and provides the buffer,
    /// and this function simply fills it with bytes from the memory map.
    ///
    /// # Arguments
    /// * `buf`: A mutable slice representing the destination buffer.
    ///
    /// # Returns
    /// A `Result` containing the number of bytes read. A value of `Ok(0)`
    /// indicates that the end of the file has been reached (EOF).
    pub fn read_chunk(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // If the cursor is at or past the end, we are done.
        if self.cursor >= self.file_size {
            return Ok(0);
        }

        // Determine how many bytes to copy. It's the smaller of the buffer's
        // capacity or the number of bytes remaining in the file.
        let bytes_remaining = self.file_size - self.cursor;
        let bytes_to_copy = (buf.len() as u64).min(bytes_remaining) as usize;

        // Define the source and destination slices for the memory copy.
        let src_end = self.cursor as usize + bytes_to_copy;
        let src_slice = &self.mmap[self.cursor as usize..src_end];
        let dest_slice = &mut buf[..bytes_to_copy];

        // Perform the single, fast memory copy.
        dest_slice.copy_from_slice(src_slice);

        // Advance the cursor for the next read.
        self.cursor += bytes_to_copy as u64;

        Ok(bytes_to_copy)
    }
}
