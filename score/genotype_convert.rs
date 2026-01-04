// ========================================================================================
//
//               VCF/BCF/DTC to PLINK Conversion Module
//
// ========================================================================================
//
// This module provides transparent conversion of VCF/BCF/DTC text files to PLINK format
// for use with the gnomon scoring pipeline. It includes caching to avoid re-conversion
// on repeated runs, and automatic reference genome downloading for DTC files.

use convert_genome::input::InputFormat as ConvertInputFormat;
use convert_genome::{ConversionConfig, OutputFormat, convert_dtc_file};
use flate2::read::GzDecoder;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Reference genome download URLs with fallbacks (tried in order)
const GRCH37_URLS: &[&str] = &[
    // Hail (Google Cloud) - fast, reliable, compressed
    "https://storage.googleapis.com/hail-common/references/human_g1k_v37.fasta.gz",
    // Gencode/EBI - authoritative source, compressed
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/GRCh37.p13.genome.fa.gz",
    // UCSC - compressed
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/latest/hg19.fa.gz",
    // Illumina DRAGEN S3 - uncompressed but reliable
    "https://ilmn-dragen-giab-samples.s3.amazonaws.com/FASTA/GRCh37.fa",
];

const GRCH38_URLS: &[&str] = &[
    // UCSC - fast, compressed
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz",
    // Illumina DRAGEN S3 - uncompressed but reliable
    "https://ilmn-dragen-giab-samples.s3.amazonaws.com/FASTA/hg38.fa",
    // Ensembl - primary assembly, compressed
    "https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    // Illumina DRAGEN S3 - alt-aware version
    "https://ilmn-dragen-giab-samples.s3.amazonaws.com/FASTA/hg38_alt_aware_nohla.fa",
    // Ensembl current - toplevel with index available
    "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna_index/Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
];

/// Supported input formats for genotype data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// PLINK binary format (.bed/.bim/.fam)
    Plink,
    /// Variant Call Format text (.vcf, .vcf.gz)
    Vcf,
    /// Binary Call Format (.bcf)
    Bcf,
    /// Direct-to-consumer text format (23andMe, AncestryDNA, etc.)
    Dtc,
}

/// Detects the input format based on file extension.
///
/// Returns `None` if the format cannot be determined.
pub fn detect_input_format(path: &Path) -> Option<InputFormat> {
    let path_str = path.to_string_lossy().to_lowercase();

    // Check for PLINK format (by extension or existence of .bed/.bim/.fam)
    if path_str.ends_with(".bed") {
        return Some(InputFormat::Plink);
    }

    // Check for DTC text format (must check before PLINK prefix detection)
    if path_str.ends_with(".txt") || path_str.ends_with(".csv") {
        return Some(InputFormat::Dtc);
    }

    // Check if it's a prefix pointing to PLINK files
    if !path_str.ends_with(".vcf") && !path_str.ends_with(".vcf.gz") && !path_str.ends_with(".bcf")
    {
        // Check if PLINK files exist
        let bed_path = path.with_extension("bed");
        let bim_path = path.with_extension("bim");
        let fam_path = path.with_extension("fam");

        if bed_path.exists() && bim_path.exists() && fam_path.exists() {
            return Some(InputFormat::Plink);
        }
    }

    // Check for VCF format
    if path_str.ends_with(".vcf") || path_str.ends_with(".vcf.gz") {
        return Some(InputFormat::Vcf);
    }

    // Check for BCF format
    if path_str.ends_with(".bcf") {
        return Some(InputFormat::Bcf);
    }

    // Directory (could contain PLINK files)
    if path.is_dir() {
        return Some(InputFormat::Plink);
    }

    None
}

/// Returns the gnomon cache directory (~/.gnomon or fallback to current dir)
fn get_gnomon_cache_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".gnomon"))
        .unwrap_or_else(|| PathBuf::from(".gnomon"))
}

/// Ensures a reference genome is available, downloading if necessary.
///
/// Tries multiple mirror URLs with fallback if one fails.
/// Returns the path to the uncompressed reference FASTA file.
fn ensure_reference_genome(build: &str) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let cache_dir = get_gnomon_cache_dir().join("refs");
    fs::create_dir_all(&cache_dir)?;

    let (urls, canonical_filename) = if build.contains("37") || build.to_lowercase().contains("hg19") {
        (GRCH37_URLS, "human_g1k_v37.fasta")
    } else {
        (GRCH38_URLS, "GRCh38_reference.fa")
    };

    let ref_path = cache_dir.join(canonical_filename);

    // Check for existing cached reference (uncompressed)
    if ref_path.exists() {
        eprintln!("> Using cached reference genome: {}", ref_path.display());
        return Ok(ref_path);
    }

    eprintln!("> Reference genome not found locally.");
    eprintln!("> Downloading {} reference (~900MB)...", build);

    // Try each URL until one works
    let mut last_error = String::new();
    for (i, url) in urls.iter().enumerate() {
        eprintln!("> Trying mirror {}/{}: {}", i + 1, urls.len(), url);

        // Download to temp file first
        let temp_path = if url.ends_with(".gz") {
            cache_dir.join(format!("{}.gz.tmp", canonical_filename))
        } else {
            cache_dir.join(format!("{}.tmp", canonical_filename))
        };

        match download_with_progress(url, &temp_path) {
            Ok(()) => {
                // Decompress if needed
                if url.ends_with(".gz") {
                    eprintln!("> Decompressing reference genome...");
                    match decompress_gz(&temp_path, &ref_path) {
                        Ok(()) => {
                            let _ = fs::remove_file(&temp_path);
                            eprintln!("> Reference genome cached at: {}", ref_path.display());
                            return Ok(ref_path);
                        }
                        Err(e) => {
                            let _ = fs::remove_file(&temp_path);
                            last_error = format!("Decompression failed: {}", e);
                            eprintln!("> {}. Trying next mirror...", last_error);
                            continue;
                        }
                    }
                } else {
                    // Just rename the temp file
                    fs::rename(&temp_path, &ref_path)?;
                    eprintln!("> Reference genome cached at: {}", ref_path.display());
                    return Ok(ref_path);
                }
            }
            Err(e) => {
                let _ = fs::remove_file(&temp_path);
                last_error = e.to_string();
                eprintln!("> Mirror failed: {}. Trying next...", last_error);
            }
        }
    }

    Err(format!(
        "Failed to download reference genome from all {} mirrors. Last error: {}",
        urls.len(),
        last_error
    ).into())
}

/// Decompress a gzipped file
fn decompress_gz(src: &Path, dest: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let input = File::open(src)?;
    let decoder = GzDecoder::new(BufReader::new(input));
    let mut reader = BufReader::new(decoder);

    let output = File::create(dest)?;
    let mut writer = BufWriter::new(output);

    let mut buffer = [0u8; 65536];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        writer.write_all(&buffer[..bytes_read])?;
    }
    writer.flush()?;
    Ok(())
}

/// Downloads a file with progress indication
fn download_with_progress(url: &str, dest: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let response = ureq::get(url).call().map_err(|e| format!("Download failed: {}", e))?;

    let content_length = response
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    let mut reader = response.into_reader();

    // Write to a temporary file first, then rename
    let temp_path = dest.with_extension("tmp");
    let file = File::create(&temp_path)?;
    let mut writer = BufWriter::new(file);

    let mut buffer = [0u8; 65536]; // 64KB buffer
    let mut downloaded: u64 = 0;
    let mut last_percent = 0;

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        writer.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        // Update progress every 5%
        if let Some(total) = content_length {
            let percent = ((downloaded as f64 / total as f64) * 100.0) as u64;
            if percent >= last_percent + 5 {
                eprint!("\r> Downloading... {}%", percent);
                last_percent = percent;
            }
        }
    }

    writer.flush()?;
    eprintln!("\r> Download complete.          ");

    // Atomic rename
    fs::rename(&temp_path, dest)?;

    Ok(())
}

/// Computes the cache directory path for a VCF/BCF/DTC file.
///
/// The cache directory is created alongside the input file as:
/// `{parent}/{stem}.gnomon_cache/`
fn get_cache_dir(input_path: &Path) -> PathBuf {
    let parent = input_path.parent().unwrap_or(Path::new("."));
    let stem = input_path
        .file_stem()
        .map(|s| {
            // Handle .vcf.gz by stripping the .vcf part too
            let s_str = s.to_string_lossy();
            if s_str.ends_with(".vcf") {
                s_str[..s_str.len() - 4].to_string()
            } else {
                s_str.to_string()
            }
        })
        .unwrap_or_else(|| "converted".to_string());

    parent.join(format!("{}.gnomon_cache", stem))
}

/// Checks if the cache is valid (source file not modified since conversion).
fn is_cache_valid(source_path: &Path, cache_dir: &Path) -> bool {
    let cache_bed = cache_dir.join("genotypes.bed");

    if !cache_bed.exists() {
        return false;
    }

    // Compare modification times
    let source_mtime = match fs::metadata(source_path).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };

    let cache_mtime = match fs::metadata(&cache_bed).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return false,
    };

    // Cache is valid if it was created after the source was last modified
    cache_mtime > source_mtime
}

/// Ensures the input is in PLINK format, converting from VCF/BCF/DTC if necessary.
///
/// # Arguments
/// * `input_path` - Path to the input genotype file (PLINK prefix, VCF, BCF, or DTC text)
/// * `reference` - Optional path to reference genome FASTA (required for DTC, auto-downloaded if None)
/// * `build` - Optional genome build override (auto-detected or defaults to GRCh38)
///
/// # Returns
/// * `Ok(PathBuf)` - Path to the PLINK prefix (either original or converted)
/// * `Err` - If conversion fails
pub fn ensure_plink_format(
    input_path: &Path,
    reference: Option<&Path>,
    build: Option<&str>,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let format = detect_input_format(input_path).ok_or_else(|| {
        format!(
            "Could not determine input format for '{}'. \
             Expected PLINK (.bed/.bim/.fam), VCF (.vcf, .vcf.gz), BCF (.bcf), or DTC text (.txt).",
            input_path.display()
        )
    })?;

    match format {
        InputFormat::Plink => {
            // Already PLINK format, return the prefix
            let prefix = if input_path.extension().is_some_and(|ext| ext == "bed") {
                input_path.with_extension("")
            } else {
                input_path.to_path_buf()
            };
            Ok(prefix)
        }
        InputFormat::Vcf | InputFormat::Bcf => {
            // Check cache validity
            let cache_dir = get_cache_dir(input_path);
            let cache_prefix = cache_dir.join("genotypes");

            if is_cache_valid(input_path, &cache_dir) {
                eprintln!(
                    "> Using cached PLINK conversion from '{}'",
                    cache_dir.display()
                );
                return Ok(cache_prefix);
            }

            // Create cache directory
            fs::create_dir_all(&cache_dir)?;

            eprintln!("> Converting {} to PLINK format...", input_path.display());

            // Set up conversion config
            let input_format = match format {
                InputFormat::Vcf => ConvertInputFormat::Vcf,
                InputFormat::Bcf => ConvertInputFormat::Bcf,
                _ => unreachable!(),
            };

            let assembly = build.unwrap_or("GRCh38").to_string();

            // No reference needed for VCF/BCF - they have embedded reference info
            let config = ConversionConfig {
                input: input_path.to_path_buf(),
                input_format,
                input_origin: input_path.display().to_string(),
                reference_fasta: None,
                reference_origin: None,
                reference_fai: None,
                reference_fai_origin: None,
                output: cache_prefix.clone(),
                output_dir: Some(cache_dir.clone()),
                output_format: OutputFormat::Plink,
                sample_id: "sample".to_string(),
                assembly,
                include_reference_sites: false,
                sex: None,
                par_boundaries: None,
                standardize: false,
                panel: None,
            };

            // Run conversion
            convert_dtc_file(config)?;

            eprintln!(
                "> Conversion complete. Cache stored at '{}'",
                cache_dir.display()
            );

            Ok(cache_prefix)
        }
        InputFormat::Dtc => {
            // Check cache validity
            let cache_dir = get_cache_dir(input_path);
            let cache_prefix = cache_dir.join("genotypes");

            if is_cache_valid(input_path, &cache_dir) {
                eprintln!(
                    "> Using cached PLINK conversion from '{}'",
                    cache_dir.display()
                );
                return Ok(cache_prefix);
            }

            // Determine build (default to GRCh38 for modern DTC tests)
            let assembly = build.unwrap_or("GRCh38").to_string();

            eprintln!("> Detected format: DTC text file");
            eprintln!("> Using genome build: {}", assembly);

            // Get reference genome - use provided path or auto-download
            let reference_path = match reference {
                Some(path) => {
                    eprintln!("> Using provided reference: {}", path.display());
                    path.to_path_buf()
                }
                None => {
                    // Magic: auto-download reference genome
                    ensure_reference_genome(&assembly)?
                }
            };

            // Create cache directory
            fs::create_dir_all(&cache_dir)?;

            eprintln!("> Converting {} to PLINK format...", input_path.display());

            let config = ConversionConfig {
                input: input_path.to_path_buf(),
                input_format: ConvertInputFormat::Dtc,
                input_origin: input_path.display().to_string(),
                reference_fasta: Some(reference_path.clone()),
                reference_origin: Some(reference_path.display().to_string()),
                reference_fai: None,
                reference_fai_origin: None,
                output: cache_prefix.clone(),
                output_dir: Some(cache_dir.clone()),
                output_format: OutputFormat::Plink,
                sample_id: input_path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "sample".to_string()),
                assembly,
                include_reference_sites: false,
                sex: None,
                par_boundaries: None,
                standardize: false,
                panel: None,
            };

            // Run conversion
            convert_dtc_file(config)?;

            eprintln!(
                "> Conversion complete. Cache stored at '{}'",
                cache_dir.display()
            );
            eprintln!("> Note: Using raw genotyped data. Missing variants will be mean-imputed during scoring.");

            Ok(cache_prefix)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_detect_plink_bed() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.bed")),
            Some(InputFormat::Plink)
        );
    }

    #[test]
    fn test_detect_vcf() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.vcf")),
            Some(InputFormat::Vcf)
        );
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.vcf.gz")),
            Some(InputFormat::Vcf)
        );
    }

    #[test]
    fn test_detect_bcf() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.bcf")),
            Some(InputFormat::Bcf)
        );
    }

    #[test]
    fn test_detect_dtc_txt() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/23andme_data.txt")),
            Some(InputFormat::Dtc)
        );
        assert_eq!(
            detect_input_format(Path::new("/path/to/ancestry_data.csv")),
            Some(InputFormat::Dtc)
        );
    }

    #[test]
    fn test_cache_dir_vcf() {
        let cache = get_cache_dir(Path::new("/data/sample.vcf"));
        assert_eq!(cache, Path::new("/data/sample.gnomon_cache"));
    }

    #[test]
    fn test_cache_dir_vcf_gz() {
        let cache = get_cache_dir(Path::new("/data/sample.vcf.gz"));
        assert_eq!(cache, Path::new("/data/sample.gnomon_cache"));
    }

    #[test]
    fn test_cache_dir_dtc() {
        let cache = get_cache_dir(Path::new("/data/23andme_raw.txt"));
        assert_eq!(cache, Path::new("/data/23andme_raw.gnomon_cache"));
    }
}
