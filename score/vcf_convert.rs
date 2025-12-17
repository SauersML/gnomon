// ========================================================================================
//
//               VCF/BCF to PLINK Conversion Module
//
// ========================================================================================
//
// This module provides transparent conversion of VCF/BCF files to PLINK format for
// use with the gnomon scoring pipeline. It includes caching to avoid re-conversion
// on repeated runs.

use convert_genome::input::InputFormat as ConvertInputFormat;
use convert_genome::{ConversionConfig, OutputFormat, convert_dtc_file};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

/// Supported input formats for genotype data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// PLINK binary format (.bed/.bim/.fam)
    Plink,
    /// Variant Call Format text (.vcf, .vcf.gz)
    Vcf,
    /// Binary Call Format (.bcf)
    Bcf,
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

/// Computes the cache directory path for a VCF/BCF file.
///
/// The cache directory is created alongside the input file as:
/// `{parent}/{stem}.gnomon_cache/`
fn get_cache_dir(vcf_path: &Path) -> PathBuf {
    let parent = vcf_path.parent().unwrap_or(Path::new("."));
    let stem = vcf_path
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

/// Ensures the input is in PLINK format, converting from VCF/BCF if necessary.
///
/// # Arguments
/// * `input_path` - Path to the input genotype file (PLINK prefix, VCF, or BCF)
///
/// # Returns
/// * `Ok(PathBuf)` - Path to the PLINK prefix (either original or converted)
/// * `Err` - If conversion fails
pub fn ensure_plink_format(input_path: &Path) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let format = detect_input_format(input_path).ok_or_else(|| {
        format!(
            "Could not determine input format for '{}'. \
             Expected PLINK (.bed/.bim/.fam), VCF (.vcf, .vcf.gz), or BCF (.bcf).",
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

            // No reference needed - convert_genome uses natural chromosome ordering
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
                assembly: "GRCh38".to_string(),
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
    fn test_cache_dir_vcf() {
        let cache = get_cache_dir(Path::new("/data/sample.vcf"));
        assert_eq!(cache, Path::new("/data/sample.gnomon_cache"));
    }

    #[test]
    fn test_cache_dir_vcf_gz() {
        let cache = get_cache_dir(Path::new("/data/sample.vcf.gz"));
        assert_eq!(cache, Path::new("/data/sample.gnomon_cache"));
    }
}
