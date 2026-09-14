// ========================================================================================
//
//               VCF/BCF/DTC to PLINK Conversion Module
//
// ========================================================================================
//
// This module provides transparent conversion of VCF/BCF/DTC text files to PLINK format
// for use with the gnomon scoring pipeline. It includes caching to avoid re-conversion
// on repeated runs, and automatic reference genome downloading for DTC files.

use convert_genome::cli::Sex;
use convert_genome::input::InputFormat as ConvertInputFormat;
use convert_genome::conversion::{
    DEFAULT_MAX_PARSE_ERROR_RATIO, DEFAULT_MIN_BUILD_CONFIDENCE, DEFAULT_MIN_EMITTED_VARIANTS,
};
use convert_genome::{ConversionConfig, OutputFormat, convert_dtc_file};

// Re-export the underlying Sex enum so callers (e.g. `score/main.rs`) can
// construct `EnsurePlinkOptions { inferred_sex: Some(...) }` without a
// direct dependency on `convert_genome`.
pub use convert_genome::cli::Sex as ConvertSex;
use flate2::read::GzDecoder;
use infer_sex::{GenomeBuild, InferredSex};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::terms::infer_first_sample_sex;

/// Reference genome sources: (FASTA URL, optional FAI URL)
/// Sources with pre-built .fai indexes are preferred as they're more reliable
const GRCH37_SOURCES: &[(&str, Option<&str>)] = &[
    // 1000 Genomes - has pre-built index (most reliable)
    (
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.gz",
        Some(
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.fai",
        ),
    ),
    // Hail (Google Cloud) - fast, has index
    (
        "https://storage.googleapis.com/hail-common/references/human_g1k_v37.fasta.gz",
        Some("https://storage.googleapis.com/hail-common/references/human_g1k_v37.fasta.fai"),
    ),
    // Illumina DRAGEN S3 - uncompressed, no index
    (
        "https://ilmn-dragen-giab-samples.s3.amazonaws.com/FASTA/GRCh37.fa",
        None,
    ),
];

const GRCH38_SOURCES: &[(&str, Option<&str>)] = &[
    // Ensembl indexed - has pre-built index (most reliable)
    (
        "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna_index/Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
        Some(
            "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna_index/Homo_sapiens.GRCh38.dna.toplevel.fa.gz.fai",
        ),
    ),
    // UCSC - fast, no index
    (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz",
        None,
    ),
    // Illumina DRAGEN S3 - uncompressed, no index
    (
        "https://ilmn-dragen-giab-samples.s3.amazonaws.com/FASTA/hg38.fa",
        None,
    ),
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

fn parse_genome_build_hint(build: &str) -> Option<GenomeBuild> {
    let lower = build.to_ascii_lowercase();
    if lower.contains("37") || lower.contains("hg19") || lower.contains("grch37") {
        Some(GenomeBuild::Build37)
    } else if lower.contains("38") || lower.contains("hg38") || lower.contains("grch38") {
        Some(GenomeBuild::Build38)
    } else {
        None
    }
}

fn to_convert_sex(inferred: InferredSex) -> Sex {
    match inferred {
        InferredSex::Male => Sex::Male,
        InferredSex::Female => Sex::Female,
        InferredSex::Indeterminate => Sex::Unknown,
    }
}

/// Detects the input format based on file extension.
///
/// Returns `None` if the format cannot be determined.
pub fn detect_input_format(path: &Path) -> Option<InputFormat> {
    let path_str = path.to_string_lossy().to_lowercase();

    // Check for PLINK format (by extension or existence of the fileset).
    // `.pgen` is PLINK 2's genotype table and is read through the same path.
    if path_str.ends_with(".bed") || path_str.ends_with(".pgen") {
        return Some(InputFormat::Plink);
    }

    // Remote PLINK directories cannot be inspected with `Path::exists`. The score
    // resolver validates every discovered bed/bim/fam or pgen/pvar/psam triad after
    // format dispatch, so a GCS directory is unambiguously the multi-fileset PLINK
    // form accepted by `gnomon score`.
    if path_str.starts_with("gs://")
        && (path_str.ends_with('/') || path_str.ends_with("/*"))
    {
        return Some(InputFormat::Plink);
    }

    // Check for DTC text format (must check before PLINK prefix detection)
    if path_str.ends_with(".txt") || path_str.ends_with(".csv") {
        return Some(InputFormat::Dtc);
    }

    // Check if it's a prefix pointing to a PLINK fileset, of either version.
    if !path_str.ends_with(".vcf") && !path_str.ends_with(".vcf.gz") && !path_str.ends_with(".bcf")
    {
        // Not `with_extension`: a prefix may carry dots of its own
        // (`acaf_threshold.chr22`), which that would truncate.
        let stem = crate::score::prepare::strip_fileset_extension(&path.to_string_lossy())
            .to_string();
        let exists = |ext: &str| Path::new(&format!("{stem}.{ext}")).exists();
        if (exists("bed") && exists("bim") && exists("fam"))
            || (exists("pgen") && exists("pvar") && exists("psam"))
        {
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
/// Downloads both FASTA and .fai index when available.
/// Returns the path to the uncompressed reference FASTA file.
fn ensure_reference_genome(build: &str) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let cache_dir = get_gnomon_cache_dir().join("refs");
    fs::create_dir_all(&cache_dir)?;

    let (sources, canonical_filename) =
        if build.contains("37") || build.to_lowercase().contains("hg19") {
            (GRCH37_SOURCES, "human_g1k_v37.fasta")
        } else {
            (GRCH38_SOURCES, "GRCh38_reference.fa")
        };

    let ref_path = cache_dir.join(canonical_filename);
    let fai_path = cache_dir.join(format!("{}.fai", canonical_filename));

    // Check for existing cached reference (uncompressed)
    if ref_path.exists() {
        eprintln!("> Using cached reference genome: {}", ref_path.display());
        return Ok(ref_path);
    }

    eprintln!("> Reference genome not found locally.");
    eprintln!("> Downloading {} reference (~900MB)...", build);

    // Try each source until one works
    let mut last_error = String::new();
    for (i, (fasta_url, fai_url)) in sources.iter().enumerate() {
        eprintln!("> Trying source {}/{}: {}", i + 1, sources.len(), fasta_url);

        // Download .fai index first if available (small file, fast)
        if let Some(fai_url) = fai_url {
            eprintln!("> Downloading index file...");
            if let Err(e) = download_file(fai_url, &fai_path) {
                eprintln!("> Warning: Failed to download index: {}", e);
                // Continue anyway - we'll try to create the index later
            }
        }

        // Download FASTA to temp file
        let temp_path = if fasta_url.ends_with(".gz") {
            cache_dir.join(format!("{}.gz.tmp", canonical_filename))
        } else {
            cache_dir.join(format!("{}.tmp", canonical_filename))
        };

        match download_with_progress(fasta_url, &temp_path) {
            Ok(()) => {
                // Decompress if needed
                if fasta_url.ends_with(".gz") {
                    eprintln!("> Decompressing reference genome...");
                    match decompress_gz(&temp_path, &ref_path) {
                        Ok(()) => {
                            let _ = fs::remove_file(&temp_path);
                            eprintln!("> Reference genome cached at: {}", ref_path.display());
                            return Ok(ref_path);
                        }
                        Err(e) => {
                            let _ = fs::remove_file(&temp_path);
                            let _ = fs::remove_file(&fai_path);
                            last_error = format!("Decompression failed: {}", e);
                            eprintln!("> {}. Trying next source...", last_error);
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
                let _ = fs::remove_file(&fai_path);
                last_error = e.to_string();
                eprintln!("> Source failed: {}. Trying next...", last_error);
            }
        }
    }

    Err(format!(
        "Failed to download reference genome from all {} sources. Last error: {}",
        sources.len(),
        last_error
    )
    .into())
}

/// Simple file download without progress (for small files like .fai)
fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {}", e))?;
    let mut reader = response.into_reader();
    let file = File::create(dest)?;
    let mut writer = BufWriter::new(file);
    std::io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    Ok(())
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
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {}", e))?;

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

/// Computes the default cache directory path for a VCF/BCF/DTC file.
///
/// The cache directory is created alongside the input file as:
/// `{parent}/{stem}.gnomon_cache/`
fn get_cache_dir(input_path: &Path) -> PathBuf {
    let parent = input_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}.gnomon_cache", cache_stem(input_path)))
}

/// The input's file name without its extension, and without `.vcf` for `.vcf.gz`.
fn cache_stem(input_path: &Path) -> String {
    input_path
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
        .unwrap_or_else(|| "converted".to_string())
}

/// Where the conversion cache for `input_path` lives: beside the input, or under
/// `cache_root` (the directory of `--out PREFIX`) when one is given. A shared root holds
/// caches for inputs from many directories, so there the directory name also carries a
/// key over the input's canonical path.
fn conversion_cache_dir(input_path: &Path, cache_root: Option<&Path>) -> PathBuf {
    let Some(root) = cache_root else {
        return get_cache_dir(input_path);
    };
    let canonical = fs::canonicalize(input_path).unwrap_or_else(|_| input_path.to_path_buf());
    let key = hex::encode(&Sha256::digest(canonical.as_os_str().as_encoded_bytes())[..8]);
    root.join(format!("{}.{key}.gnomon_cache", cache_stem(input_path)))
}

/// The prefix a default run names its results after when `input_path` needs conversion:
/// `{parent}/{stem}.gnomon_cache/genotypes`, where those results have always landed. It
/// is not the converted fileset, which sits in a generation directory below it. `None`
/// for PLINK inputs, which name results after their own prefix.
pub fn default_output_prefix(input_path: &Path) -> Option<PathBuf> {
    match detect_input_format(input_path)? {
        InputFormat::Plink => None,
        InputFormat::Vcf | InputFormat::Bcf | InputFormat::Dtc => {
            Some(get_cache_dir(input_path).join("genotypes"))
        }
    }
}

/// File recording the conversion parameters a cache was produced under.
const CACHE_PARAMS_FILE: &str = "conversion_params.txt";

/// Fingerprint of the conversion parameters that change the cached output.
///
/// The mtime check alone is unsafe: re-running the same input with a different
/// genome build, strand-harmonization panel, or reference produces different
/// genotypes, but the source file is untouched, so a stale cache would be
/// silently reused. Including these in a fingerprint forces re-conversion when
/// any of them changes. `build` is resolved through the same `GRCh38` default the
/// conversion uses, so `None` and an explicit `GRCh38` share a cache.
fn cache_params_fingerprint(
    build: Option<&str>,
    panel: Option<&Path>,
    reference: Option<&Path>,
) -> String {
    let assembly = build.unwrap_or("GRCh38");
    let panel = panel.map(|p| p.display().to_string()).unwrap_or_default();
    let reference = reference.map(|p| p.display().to_string()).unwrap_or_default();
    format!("v1\nbuild={assembly}\npanel={panel}\nreference={reference}\n")
}

/// Staging directory names tried before giving up. A collision needs another
/// conversion of the same generation with the same pid and clock reading.
const STAGING_NAME_ATTEMPTS: u32 = 32;

/// Name prefix of a generation converted from a source whose metadata could not be read.
const UNIDENTIFIED_GENERATION_PREFIX: &str = "g-u";

/// The directory holding one conversion inside a cache directory. Its name is a key
/// over the conversion parameters and the source's size and modification time, so an
/// edited source, or another `--build`, `--panel` or `--reference`, gets a generation
/// of its own and no run reads a fileset converted from something else. A source whose
/// metadata cannot be read, such as a remote one, cannot be identified, so its
/// generation (`g-u…`, keyed by the parameters alone) is never served from the cache.
/// Every run converts it again and replaces it, as before generations existed.
fn generation_dir(cache_dir: &Path, source_path: &Path, fingerprint: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update((fingerprint.len() as u64).to_le_bytes());
    hasher.update(fingerprint.as_bytes());
    let identity = fs::metadata(source_path).ok().and_then(|metadata| {
        let modified = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
        Some((metadata.len(), modified.as_nanos()))
    });
    let prefix = match identity {
        Some((len, modified_nanos)) => {
            hasher.update(len.to_le_bytes());
            hasher.update(modified_nanos.to_le_bytes());
            "g-"
        }
        None => UNIDENTIFIED_GENERATION_PREFIX,
    };
    cache_dir.join(format!("{prefix}{}", hex::encode(&hasher.finalize()[..8])))
}

/// Whether `generation` holds a complete conversion made under `fingerprint` that may be
/// served. A generation is published whole, by renaming its directory into place, so
/// its files are complete whenever it exists; the parameters are still compared as a
/// guard, and an unidentified source's generation is never served.
fn is_generation_valid(generation: &Path, fingerprint: &str) -> bool {
    let unidentified = generation
        .file_name()
        .is_some_and(|name| name.to_string_lossy().starts_with(UNIDENTIFIED_GENERATION_PREFIX));
    !unidentified
        && ["genotypes.bed", "genotypes.bim", "genotypes.fam"]
            .iter()
            .all(|name| generation.join(name).is_file())
        && fs::read_to_string(generation.join(CACHE_PARAMS_FILE))
            .is_ok_and(|found| found == fingerprint)
}

/// Converts into a private staging directory inside the cache directory, fsyncs every
/// file the converter wrote, and renames the directory into place as `generation`.
/// Readers therefore see no generation or a complete one, never files still being
/// written. A failed conversion removes its staging directory. When runs convert the
/// same source at once, the first rename wins and the others use that generation.
fn publish_generation<F>(
    generation: &Path,
    fingerprint: &str,
    convert: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: FnOnce(&Path) -> Result<(), Box<dyn Error + Send + Sync>>,
{
    let cache_dir = generation.parent().ok_or_else(|| {
        format!(
            "Conversion cache path '{}' has no parent directory.",
            generation.display()
        )
    })?;
    fs::create_dir_all(cache_dir)?;
    let staging = create_staging_dir(cache_dir, generation)?;
    let staged = (|| -> Result<(), Box<dyn Error + Send + Sync>> {
        convert(&staging)?;
        fs::write(staging.join(CACHE_PARAMS_FILE), fingerprint)?;
        // The converter does not fsync. Make every file durable before the rename can
        // publish it, so a crash cannot leave a generation of truncated files.
        for entry in fs::read_dir(&staging)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                // Opened for writing: Windows cannot flush a read-only handle.
                OpenOptions::new()
                    .write(true)
                    .open(entry.path())?
                    .sync_all()?;
            }
        }
        Ok(())
    })();
    if let Err(err) = staged {
        let _ = fs::remove_dir_all(&staging);
        return Err(err);
    }
    if let Err(first_err) = fs::rename(&staging, generation) {
        if is_generation_valid(generation, fingerprint) {
            // Another run published this generation first.
            let _ = fs::remove_dir_all(&staging);
            return Ok(());
        }
        let replaced = if generation.is_dir() {
            // A published generation that may not be served, such as an unidentified
            // source's, is replaced: moved aside under a fresh private name, then
            // removed once the new one is in place. A run still reading it keeps its
            // open files on Unix.
            create_staging_dir(cache_dir, generation).and_then(|aside| {
                fs::remove_dir(&aside)?;
                fs::rename(generation, &aside)?;
                let published = fs::rename(&staging, generation);
                let _ = fs::remove_dir_all(&aside);
                published
            })
        } else {
            Err(first_err)
        };
        if let Err(err) = replaced {
            let _ = fs::remove_dir_all(&staging);
            return Err(format!(
                "Could not publish the PLINK conversion cache '{}': {err}",
                generation.display()
            )
            .into());
        }
    }
    Ok(())
}

/// Creates `.{generation}.{pid}.{nanos}.tmp` in `cache_dir` exclusively, so concurrent
/// conversions never share a staging directory.
fn create_staging_dir(cache_dir: &Path, generation: &Path) -> io::Result<PathBuf> {
    let name = generation
        .file_name()
        .map_or_else(|| "generation".into(), |name| name.to_string_lossy());
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    for attempt in 0..STAGING_NAME_ATTEMPTS {
        let candidate =
            cache_dir.join(format!(".{name}.{pid}.{}.tmp", nanos + u128::from(attempt)));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "Failed to allocate a staging directory in '{}'.",
            cache_dir.display()
        ),
    ))
}

/// Removes the generations in `cache_dir` made under `fingerprint`, other than
/// `current`. They were converted from an earlier version of the source, so they can
/// never be served again. Generations made under other parameters stay for the runs
/// that use them, and files from the previous cache layout are left alone. Best effort.
fn prune_superseded_generations(cache_dir: &Path, current: &Path, fingerprint: &str) {
    let Ok(entries) = fs::read_dir(cache_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == current || !entry.file_name().to_string_lossy().starts_with("g-") {
            continue;
        }
        if fs::read_to_string(path.join(CACHE_PARAMS_FILE)).is_ok_and(|found| found == fingerprint)
        {
            let _ = fs::remove_dir_all(&path);
        }
    }
}

/// Options controlling how `ensure_plink_format_with_options` performs its work.
#[derive(Debug, Default, Clone, Copy)]
pub struct EnsurePlinkOptions {
    /// If true, skip the pre-conversion `infer_first_sample_sex` pass on VCF/BCF
    /// inputs. The FAM file will then be written with `Sex::Unknown`. Callers that
    /// run sex inference downstream (e.g. `gnomon all` invoking `terms`) can set
    /// this to avoid a redundant whole-VCF scan.
    pub skip_sex_inference: bool,
    /// Pre-computed sex to use instead of running `infer_first_sample_sex` on
    /// VCF/BCF inputs. When `Some`, the VCF-scan is skipped and this value is
    /// written into the FAM file directly. Takes precedence over
    /// `skip_sex_inference`. Intended for pipelines (e.g. pgsEngine gather)
    /// that already ran sex inference upstream on the pre-imputed VCF and
    /// would otherwise pay the ~4min full-VCF scan again.
    pub inferred_sex: Option<Sex>,
}

/// Ensures the input is in PLINK format, converting from VCF/BCF/DTC if necessary.
///
/// # Arguments
/// * `input_path` - Path to the input genotype file (PLINK prefix, VCF, BCF, or DTC text)
/// * `reference` - Optional path to reference genome FASTA (required for DTC, auto-downloaded if None)
/// * `build` - Optional genome build override (auto-detected or defaults to GRCh38)
/// * `panel` - Optional reference panel VCF for strand harmonization (flips alleles to match panel)
///
/// # Returns
/// * `Ok(PathBuf)` - Path to the PLINK prefix (either original or converted)
/// * `Err` - If conversion fails
pub fn ensure_plink_format(
    input_path: &Path,
    reference: Option<&Path>,
    build: Option<&str>,
    panel: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    ensure_plink_format_with_options(
        input_path,
        reference,
        build,
        panel,
        EnsurePlinkOptions::default(),
    )
}

/// Variant of [`ensure_plink_format`] that accepts additional runtime options.
///
/// Behaves identically when `options` is left at its default; in particular,
/// the cache directory, cache filenames, and PLINK prefix layout are unchanged,
/// so a conversion produced with options is interchangeable with one produced
/// by the plain `ensure_plink_format`.
pub fn ensure_plink_format_with_options(
    input_path: &Path,
    reference: Option<&Path>,
    build: Option<&str>,
    panel: Option<&Path>,
    options: EnsurePlinkOptions,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    ensure_plink_format_in(input_path, reference, build, panel, options, None)
}

/// Variant of [`ensure_plink_format_with_options`] that keeps the conversion cache
/// under `cache_root` (the directory of `--out PREFIX`) instead of beside the input.
///
/// Converted filesets are published whole. Each lives in a generation directory
/// keyed by the conversion parameters and the source's size and modification time,
/// and is written under a private staging name, then renamed into place once
/// complete. A reader, a concurrent run, or a run after a killed conversion
/// therefore never reads a partial or mixed fileset.
pub fn ensure_plink_format_in(
    input_path: &Path,
    reference: Option<&Path>,
    build: Option<&str>,
    panel: Option<&Path>,
    options: EnsurePlinkOptions,
    cache_root: Option<&Path>,
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
            let cache_fingerprint = cache_params_fingerprint(build, panel, reference);
            let cache_dir = conversion_cache_dir(input_path, cache_root);
            let generation = generation_dir(&cache_dir, input_path, &cache_fingerprint);
            let cache_prefix = generation.join("genotypes");

            if is_generation_valid(&generation, &cache_fingerprint) {
                eprintln!(
                    "> Using cached PLINK conversion from '{}'",
                    generation.display()
                );
                return Ok(cache_prefix);
            }

            eprintln!("> Converting {} to PLINK format...", input_path.display());

            // Set up conversion config
            let input_format = match format {
                InputFormat::Vcf => ConvertInputFormat::Vcf,
                InputFormat::Bcf => ConvertInputFormat::Bcf,
                _ => unreachable!(),
            };

            let assembly = build.unwrap_or("GRCh38").to_string();
            let inferred_sex = if let Some(preset) = options.inferred_sex {
                eprintln!(
                    "> Using caller-provided sex {:?}; skipping pre-conversion sex inference pass.",
                    preset
                );
                preset
            } else if options.skip_sex_inference {
                eprintln!(
                    "> Skipping pre-conversion sex inference pass (sex will be written as Unknown; downstream `terms` pass will recompute)."
                );
                Sex::Unknown
            } else {
                let build_hint = parse_genome_build_hint(&assembly);
                match infer_first_sample_sex(input_path, build_hint) {
                    Ok(Some(sex)) => {
                        let convert_sex = to_convert_sex(sex);
                        eprintln!("> Inferred sample sex from input: {:?}", convert_sex);
                        convert_sex
                    }
                    Ok(None) => {
                        eprintln!(
                            "> Sex inference produced no sample calls; defaulting to Unknown"
                        );
                        Sex::Unknown
                    }
                    Err(err) => {
                        eprintln!("> Sex inference unavailable ({err}); defaulting to Unknown");
                        Sex::Unknown
                    }
                }
            };

            publish_generation(&generation, &cache_fingerprint, |staging| {
                // No reference needed for VCF/BCF - they have embedded reference info
                let config = ConversionConfig {
                    input: input_path.to_path_buf(),
                    input_format,
                    input_origin: input_path.display().to_string(),
                    reference_fasta: None,
                    reference_origin: None,
                    reference_fai: None,
                    reference_fai_origin: None,
                    output: staging.join("genotypes"),
                    output_dir: Some(staging.to_path_buf()),
                    output_format: OutputFormat::Plink,
                    sample_id: "sample".to_string(),
                    assembly,
                    input_build: build.map(|b| b.to_string()),
                    include_reference_sites: false,
                    sex: Some(inferred_sex),
                    par_boundaries: None,
                    standardize: false,
                    panel: panel.map(|p| p.to_path_buf()),
                    // Clinical-safety gates: reject a conversion that silently
                    // produces almost nothing, cannot confidently identify the
                    // source build, or fails to parse much of its input. Upstream
                    // owns these thresholds, so track its defaults rather than
                    // pinning our own copies.
                    min_emitted_variants: DEFAULT_MIN_EMITTED_VARIANTS,
                    min_build_confidence: DEFAULT_MIN_BUILD_CONFIDENCE,
                    max_parse_error_ratio: DEFAULT_MAX_PARSE_ERROR_RATIO,
                };

                // Run conversion
                convert_dtc_file(config)?;
                Ok(())
            })?;
            prune_superseded_generations(&cache_dir, &generation, &cache_fingerprint);

            eprintln!(
                "> Conversion complete. Cache stored at '{}'",
                generation.display()
            );

            Ok(cache_prefix)
        }
        InputFormat::Dtc => {
            // Check cache validity
            let cache_fingerprint = cache_params_fingerprint(build, panel, reference);
            let cache_dir = conversion_cache_dir(input_path, cache_root);
            let generation = generation_dir(&cache_dir, input_path, &cache_fingerprint);
            let cache_prefix = generation.join("genotypes");

            if is_generation_valid(&generation, &cache_fingerprint) {
                eprintln!(
                    "> Using cached PLINK conversion from '{}'",
                    generation.display()
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

            eprintln!("> Converting {} to PLINK format...", input_path.display());

            publish_generation(&generation, &cache_fingerprint, |staging| {
                let config = ConversionConfig {
                    input: input_path.to_path_buf(),
                    input_format: ConvertInputFormat::Dtc,
                    input_origin: input_path.display().to_string(),
                    reference_fasta: Some(reference_path.clone()),
                    reference_origin: Some(reference_path.display().to_string()),
                    reference_fai: None,
                    reference_fai_origin: None,
                    output: staging.join("genotypes"),
                    output_dir: Some(staging.to_path_buf()),
                    output_format: OutputFormat::Plink,
                    sample_id: input_path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "sample".to_string()),
                    assembly,
                    input_build: build.map(|b| b.to_string()),
                    include_reference_sites: false,
                    sex: None,
                    par_boundaries: None,
                    standardize: false,
                    panel: panel.map(|p| p.to_path_buf()),
                    // Clinical-safety gates: reject a conversion that silently
                    // produces almost nothing, cannot confidently identify the
                    // source build, or fails to parse much of its input. Upstream
                    // owns these thresholds, so track its defaults rather than
                    // pinning our own copies.
                    min_emitted_variants: DEFAULT_MIN_EMITTED_VARIANTS,
                    min_build_confidence: DEFAULT_MIN_BUILD_CONFIDENCE,
                    max_parse_error_ratio: DEFAULT_MAX_PARSE_ERROR_RATIO,
                };

                // Run conversion
                convert_dtc_file(config)?;
                Ok(())
            })?;
            prune_superseded_generations(&cache_dir, &generation, &cache_fingerprint);

            eprintln!(
                "> Conversion complete. Cache stored at '{}'",
                generation.display()
            );
            eprintln!(
                "> Note: Using raw genotyped data. Missing variants will be mean-imputed during scoring."
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
    fn test_detect_remote_plink_directory() {
        assert_eq!(
            detect_input_format(Path::new("gs://bucket/callset/pgen/")),
            Some(InputFormat::Plink)
        );
        assert_eq!(
            detect_input_format(Path::new("gs://bucket/callset/pgen/*")),
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

    fn fake_conversion(staging: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
        for name in ["genotypes.bed", "genotypes.bim", "genotypes.fam"] {
            fs::write(staging.join(name), name)?;
        }
        Ok(())
    }

    fn entry_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("read_dir")
            .map(|entry| {
                entry
                    .expect("directory entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_published_generation_is_complete_and_leaves_no_staging_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("sample.bcf");
        fs::write(&source, b"version one").expect("source");
        let cache_dir = conversion_cache_dir(&source, None);
        assert_eq!(cache_dir, dir.path().join("sample.gnomon_cache"));
        let generation = generation_dir(&cache_dir, &source, "v1\n");

        publish_generation(&generation, "v1\n", fake_conversion).expect("publish");

        assert!(is_generation_valid(&generation, "v1\n"));
        assert!(!is_generation_valid(&generation, "v2\n"));
        let name = generation.file_name().expect("name").to_string_lossy().into_owned();
        assert_eq!(entry_names(&cache_dir), [name]);
    }

    #[test]
    fn a_failed_conversion_publishes_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let generation = cache_dir.join("g-0");
        let err = publish_generation(&generation, "v1\n", |staging| {
            fs::write(staging.join("genotypes.bed"), b"partial")?;
            Err("converter failed".into())
        })
        .expect_err("the conversion error must propagate");
        assert_eq!(err.to_string(), "converter failed");
        assert!(!generation.exists());
        assert!(entry_names(&cache_dir).is_empty());
    }

    #[test]
    fn concurrent_conversions_of_one_source_all_succeed_with_one_generation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let generation = cache_dir.join("g-0");
        std::thread::scope(|scope| {
            let runs: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        publish_generation(&generation, "v1\n", fake_conversion)
                            .map_err(|err| err.to_string())
                    })
                })
                .collect();
            for run in runs {
                run.join().expect("thread").expect("every run must succeed");
            }
        });
        assert!(is_generation_valid(&generation, "v1\n"));
        assert_eq!(entry_names(&cache_dir), ["g-0"]);
    }

    #[test]
    fn generation_keys_follow_the_parameters_and_the_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("sample.bcf");
        fs::write(&source, b"version one").expect("source");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let first = generation_dir(&cache_dir, &source, "v1\nbuild=GRCh38\n");
        assert_eq!(first, generation_dir(&cache_dir, &source, "v1\nbuild=GRCh38\n"));
        assert_ne!(first, generation_dir(&cache_dir, &source, "v1\nbuild=GRCh37\n"));
        fs::write(&source, b"version two, longer").expect("edit the source");
        assert_ne!(first, generation_dir(&cache_dir, &source, "v1\nbuild=GRCh38\n"));
    }

    #[test]
    fn an_unidentified_source_is_never_served_and_is_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("remote.gnomon_cache");
        let missing = dir.path().join("absent.bcf");
        let generation = generation_dir(&cache_dir, &missing, "v1\n");
        assert_eq!(generation, generation_dir(&cache_dir, &missing, "v1\n"));
        for version in ["first", "second"] {
            publish_generation(&generation, "v1\n", |staging| {
                for name in ["genotypes.bed", "genotypes.bim", "genotypes.fam"] {
                    fs::write(staging.join(name), version)?;
                }
                Ok(())
            })
            .expect("publish");
            assert!(!is_generation_valid(&generation, "v1\n"));
            assert_eq!(
                fs::read_to_string(generation.join("genotypes.bed")).expect("read"),
                version
            );
        }
        assert_eq!(entry_names(&cache_dir).len(), 1, "{:?}", entry_names(&cache_dir));
    }

    #[test]
    fn pruning_removes_only_superseded_generations_of_the_same_parameters() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        for (name, fingerprint) in [("g-old", "v1\n"), ("g-new", "v1\n"), ("g-other", "v2\n")] {
            publish_generation(&cache_dir.join(name), fingerprint, fake_conversion)
                .expect("publish");
        }
        // The previous layout kept its files directly in the cache directory.
        fs::write(cache_dir.join("genotypes.bed"), b"legacy").expect("legacy");

        prune_superseded_generations(&cache_dir, &cache_dir.join("g-new"), "v1\n");

        assert_eq!(entry_names(&cache_dir), ["g-new", "g-other", "genotypes.bed"]);
    }

    #[test]
    fn a_shared_cache_root_keys_conversion_caches_by_input_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("results").join("gnomon_score_cache");
        let a = dir.path().join("a").join("sample.vcf.gz");
        let b = dir.path().join("b").join("sample.vcf.gz");
        for input in [&a, &b] {
            fs::create_dir_all(input.parent().expect("parent")).expect("mkdir");
            fs::write(input, b"source").expect("source");
        }
        let cache_a = conversion_cache_dir(&a, Some(&root));
        let cache_b = conversion_cache_dir(&b, Some(&root));
        assert_ne!(cache_a, cache_b);
        for cache in [&cache_a, &cache_b] {
            assert_eq!(cache.parent(), Some(root.as_path()));
            let name = cache.file_name().expect("name").to_string_lossy().into_owned();
            assert!(name.starts_with("sample.") && name.ends_with(".gnomon_cache"), "{name}");
        }
        assert_eq!(
            conversion_cache_dir(&a, None),
            dir.path().join("a").join("sample.gnomon_cache")
        );
    }

    #[test]
    fn converted_inputs_name_default_results_after_their_cache_directory() {
        assert_eq!(
            default_output_prefix(Path::new("/data/sample.bcf")),
            Some(PathBuf::from("/data/sample.gnomon_cache/genotypes"))
        );
        assert_eq!(default_output_prefix(Path::new("/data/arrays.bed")), None);
    }
}
