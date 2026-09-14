use std::collections::HashSet;
use std::fs::File;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use infer_sex::{
    Chromosome, EvidenceReport, GenomeBuild, InferenceConfig, InferenceError, InferenceResult,
    InferredSex, PlatformDefinition, SexInferenceAccumulator, VariantInfo,
};
use memmap2::Mmap;
use rayon::prelude::*;
use thiserror::Error;

use crate::adapt_plink2::GenomeBuild as PgenGenomeBuild;
use crate::map::fit::VariantBlockSource;
use crate::map::io::{
    DatasetBlockSource, GenotypeDataset, GenotypeIoError, PlinkDataset, PlinkIoError, SelectionPlan,
};
use crate::map::variant_filter::VariantKey;
use crate::terms::sex_counts::{
    BedRows, EvidenceCounts, LocusClass, count_evidence, finish_counts,
};

#[derive(Debug, Error)]
pub enum SexInferenceError {
    #[error("genotype I/O error: {0}")]
    Dataset(#[from] GenotypeIoError),
    #[error(
        "the PLINK filesets in {directory} list different samples: {first} and {other} do not match"
    )]
    MismatchedFilesetSamples {
        directory: String,
        first: String,
        other: String,
    },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("sex inference error: {0:?}")]
    Inference(InferenceError),
    #[error(
        "variant stream yielded more variants than expected (expected {expected}, observed {observed})"
    )]
    VariantOverflow { expected: usize, observed: usize },
    #[error(
        "variant stream terminated early (processed {observed} of {expected} expected variants)"
    )]
    VariantUnderflow { expected: usize, observed: usize },
    #[error(
        "insufficient sex-informative variants: {autosomes} autosomal and {y_non_par} Y non-PAR loci available"
    )]
    InsufficientInformativeVariants { autosomes: u64, y_non_par: u64 },
}

impl From<InferenceError> for SexInferenceError {
    fn from(value: InferenceError) -> Self {
        SexInferenceError::Inference(value)
    }
}

#[derive(Debug, Clone)]
pub struct SexInferenceRecord {
    pub individual_id: String,
    pub inference: InferenceResult,
}

#[derive(Debug, Clone)]
struct SexVariantSelection {
    keys: Vec<SelectedVariant>,
    /// Row index of each selected variant, ascending and parallel to `keys`.
    indices: Vec<usize>,
    build: GenomeBuild,
}

#[derive(Debug, Clone)]
struct SelectedVariant {
    position: u64,
    chrom: Chromosome,
}

/// How sex inference reads a variant's chromosome label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocusChromosome {
    Autosome,
    X,
    Y,
    /// The X pseudoautosomal regions under a code of their own: PLINK's `25` or
    /// `XY`, or plink2's `PAR1` and `PAR2` contigs. Positions are X coordinates.
    XPar,
}

/// Chromosome class and position of every variant, in file order.
#[derive(Debug, Default, PartialEq, Eq)]
struct VariantLoci {
    chroms: Vec<Option<LocusChromosome>>,
    positions: Vec<u64>,
}

impl VariantLoci {
    fn from_keys(keys: &[VariantKey]) -> Self {
        Self {
            chroms: keys
                .iter()
                .map(|key| classify_chromosome(&key.chromosome))
                .collect(),
            positions: keys.iter().map(|key| key.position).collect(),
        }
    }

    /// The loci of a PLINK 1 fileset, read from its `.bim` without building a key
    /// per variant. A label is classified exactly as its normalized key would be,
    /// once per run of rows that share it.
    fn from_bim(dataset: &PlinkDataset) -> Result<Self, PlinkIoError> {
        if let Some(loci) = Self::scan_local_bim(dataset, BIM_SCAN_CHUNK_BYTES) {
            return Ok(loci);
        }
        let mut loci = Self {
            chroms: Vec::with_capacity(dataset.n_variants()),
            positions: Vec::with_capacity(dataset.n_variants()),
        };
        let mut label = String::new();
        let mut class = None;
        dataset.for_each_variant_position(|chromosome, position| {
            if chromosome != label {
                label.clear();
                label.push_str(chromosome);
                class = classify_chromosome(&VariantKey::new(chromosome, position).chromosome);
            }
            loci.chroms.push(class);
            loci.positions.push(position);
        })?;
        Ok(loci)
    }

    /// [`VariantLoci::from_bim`] for a local `.bim`: the file is mapped and
    /// scanned in parallel chunks, each extended to the end of its last line.
    ///
    /// `PlinkDataset::open` has already validated every line, so this pass only
    /// looks for the label and position columns. It finds them as
    /// `split_whitespace` does, and skips the lines `PlinkVariantRecordIter`
    /// skips. It returns `None` when the path is not a local file or the scan
    /// does not reproduce the validated shape, and the caller then reads through
    /// the record reader, which reports whatever changed.
    fn scan_local_bim(dataset: &PlinkDataset, chunk_bytes: usize) -> Option<Self> {
        let file = File::open(dataset.bim_path()).ok()?;
        // SAFETY: the map is read-only and dropped before this returns. As with
        // the `.bed` map, the file must not be truncated while it is mapped.
        let map = unsafe { Mmap::map(&file) }.ok()?;
        let text: &[u8] = &map;

        let mut bounds = vec![0];
        let mut search_from = chunk_bytes.max(1);
        while search_from < text.len() {
            let Some(offset) = memchr::memchr(b'\n', &text[search_from..]) else {
                break;
            };
            let end = search_from + offset + 1;
            bounds.push(end);
            search_from = end + chunk_bytes.max(1);
        }
        if bounds.last() != Some(&text.len()) {
            bounds.push(text.len());
        }

        let chunks: Vec<Option<Self>> = bounds
            .par_windows(2)
            .map(|chunk| scan_bim_chunk(&text[chunk[0]..chunk[1]]))
            .collect();
        let mut loci = Self {
            chroms: Vec::with_capacity(dataset.n_variants()),
            positions: Vec::with_capacity(dataset.n_variants()),
        };
        for chunk in chunks {
            let chunk = chunk?;
            loci.chroms.extend_from_slice(&chunk.chroms);
            loci.positions.extend_from_slice(&chunk.positions);
        }
        (loci.positions.len() == dataset.n_variants()).then_some(loci)
    }
}

/// Bytes of `.bim` text per parallel scan chunk, before a chunk is extended to
/// the end of its last line.
const BIM_SCAN_CHUNK_BYTES: usize = 1 << 20;

/// The loci of every line in one chunk of `.bim` text, or `None` if a line has
/// no label or no `u64` position.
fn scan_bim_chunk(text: &[u8]) -> Option<VariantLoci> {
    let mut loci = VariantLoci::default();
    let mut label: &[u8] = &[];
    let mut class = None;
    let mut rest = text;
    while !rest.is_empty() {
        let line = match memchr::memchr(b'\n', rest) {
            Some(end) => {
                let line = &rest[..end];
                rest = &rest[end + 1..];
                line
            }
            None => std::mem::take(&mut rest),
        };
        // `LocalTextSource` drops the newline and one carriage return, and the
        // record reader skips lines that are only ASCII whitespace.
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let (line_label, position) = bim_label_and_position(line)?;
        if loci.positions.is_empty() || line_label != label {
            label = line_label;
            class = classify_chromosome(
                &VariantKey::new(str::from_utf8(line_label).ok()?, position).chromosome,
            );
        }
        loci.chroms.push(class);
        loci.positions.push(position);
    }
    Some(loci)
}

/// A `.bim` line's label and position columns, found as `split_whitespace`
/// finds them.
fn bim_label_and_position(line: &[u8]) -> Option<(&[u8], u64)> {
    let (label, position) = if line.is_ascii() {
        // On ASCII text `split_whitespace` splits on exactly these bytes.
        let mut fields = line
            .split(|byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r'))
            .filter(|field| !field.is_empty());
        let label = fields.next()?;
        (label, fields.nth(2)?)
    } else {
        let mut fields = str::from_utf8(line).ok()?.split_whitespace();
        let label = fields.next()?;
        (label.as_bytes(), fields.nth(2)?.as_bytes())
    };
    Some((label, str::from_utf8(position).ok()?.parse().ok()?))
}

impl SexVariantSelection {
    fn from_loci(loci: &VariantLoci, build: GenomeBuild) -> Self {
        const AUTOSOME_SAMPLE_TARGET: usize = 2000;

        let mut selected = Vec::new();
        let mut selected_autosomes = HashSet::new();
        let mut autosome_indices = Vec::new();

        for (index, (&chrom, &position)) in loci.chroms.iter().zip(&loci.positions).enumerate() {
            let chrom = match chrom {
                Some(LocusChromosome::Autosome) => {
                    autosome_indices.push(index);
                    continue;
                }
                Some(LocusChromosome::X) => Chromosome::X,
                Some(LocusChromosome::Y) => Chromosome::Y,
                // A row coded pseudoautosomal is read as X only inside the build's PAR
                // intervals. Elsewhere its label and position disagree, and filing it
                // under non-PAR X would read a male's diploid calls as heterozygous
                // X evidence.
                Some(LocusChromosome::XPar) if build.is_in_x_par(position) => Chromosome::X,
                Some(LocusChromosome::XPar) | None => continue,
            };
            selected.push((index, SelectedVariant { position, chrom }));
        }

        let autosome_sample_count = AUTOSOME_SAMPLE_TARGET.min(autosome_indices.len());
        for i in 0..autosome_sample_count {
            let source_idx = i * autosome_indices.len() / autosome_sample_count;
            let key_index = autosome_indices[source_idx];
            if selected_autosomes.insert(key_index) {
                selected.push((
                    key_index,
                    SelectedVariant {
                        position: loci.positions[key_index],
                        chrom: Chromosome::Autosome,
                    },
                ));
            }
        }

        selected.sort_by_key(|entry| entry.0);

        let mut selected_indices = Vec::with_capacity(selected.len());
        let mut selected_keys = Vec::with_capacity(selected.len());
        for (index, variant) in selected.into_iter() {
            selected_indices.push(index);
            selected_keys.push(variant);
        }

        Self {
            keys: selected_keys,
            indices: selected_indices,
            build,
        }
    }
}

const SEX_TSV_HEADER: &str = concat!(
    "IID\tBuild\tSex\tY_Density\tX_AutoHet_Ratio\tComposite_Index\tAuto_Valid\tAuto_Het\t",
    "X_NonPAR_Valid\tX_NonPAR_Het\tY_NonPAR_Valid\tY_PAR_Valid",
);

enum TermsProgress {
    Terminal(ProgressBar),
    Plain { last_percent: u64 },
    Silent,
}

impl TermsProgress {
    fn new(total_variants: usize, enabled: bool) -> Self {
        if !enabled {
            return Self::Silent;
        }
        let total = total_variants as u64;
        if std::io::stderr().is_terminal() {
            let pb =
                ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr_with_hz(20));
            let style = ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {pos:>7}/{len:7} ({percent:>3}%)",
            )
            .expect("progress template should be valid")
            .progress_chars("█▉▊▋▌▍▎▏  ");
            pb.set_style(style);
            pb.set_message("Inferring sex terms");
            Self::Terminal(pb)
        } else {
            eprintln!("Inferring sex terms: 0% complete...");
            Self::Plain { last_percent: 0 }
        }
    }

    fn update(&mut self, processed: usize, total_variants: usize) {
        let total = total_variants.max(1) as u64;
        let processed_u64 = processed as u64;
        let percent = ((processed_u64.saturating_mul(100)) / total).min(100);
        match self {
            Self::Terminal(pb) => pb.set_position(processed_u64.min(total)),
            Self::Plain { last_percent } => {
                if percent > *last_percent {
                    *last_percent = percent;
                    eprintln!("Inferring sex terms: {percent}% complete...");
                }
            }
            Self::Silent => {}
        }
    }

    fn finish(self, total_variants: usize) {
        let total = total_variants as u64;
        match self {
            Self::Terminal(pb) => {
                pb.set_position(total);
                pb.finish_with_message("Inferring sex terms: complete");
            }
            Self::Plain { last_percent } => {
                if last_percent < 100 {
                    eprintln!("Inferring sex terms: 100% complete.");
                }
            }
            Self::Silent => {}
        }
    }
}

pub fn infer_sex_to_tsv(
    genotype_path: &Path,
    force_build: Option<GenomeBuild>,
) -> Result<PathBuf, SexInferenceError> {
    let input = open_inference_input(genotype_path, force_build)?;
    // Refuse before inference when the table could not be saved, typically a
    // default location beside read-only inputs.
    let default_output = input.output_path();
    crate::output::ensure_output_writable(&default_output)?;
    let (build, records) = input.infer(force_build, true)?;

    write_results(&default_output, &records, build)?;

    Ok(default_output)
}

/// Identical to [`infer_sex_to_tsv`] but writes the sex TSV at an explicit
/// caller-provided path instead of `dataset.output_path("sex.tsv")`. Used by
/// `gnomon terms --out PREFIX`, and so by `gnomon all --out PREFIX`.
pub fn infer_sex_to_tsv_at(
    genotype_path: &Path,
    force_build: Option<GenomeBuild>,
    output_path: &Path,
) -> Result<PathBuf, SexInferenceError> {
    // Refuse before inference when the table could not be saved.
    crate::output::ensure_output_writable(output_path)?;
    let (_, build, records) = infer_records(genotype_path, force_build, true)?;

    write_results(output_path, &records, build)?;

    Ok(output_path.to_path_buf())
}

pub fn infer_first_sample_sex(
    genotype_path: &Path,
    force_build: Option<GenomeBuild>,
) -> Result<Option<InferredSex>, SexInferenceError> {
    let (_, _, records) = infer_records(genotype_path, force_build, false)?;
    Ok(records.first().map(|record| record.inference.final_call))
}

/// Infers sex for every sample of `genotype_path`, returning the table's default
/// output path, the build used and one record per sample.
fn infer_records(
    genotype_path: &Path,
    force_build: Option<GenomeBuild>,
    show_progress: bool,
) -> Result<(PathBuf, GenomeBuild, Vec<SexInferenceRecord>), SexInferenceError> {
    let input = open_inference_input(genotype_path, force_build)?;
    let (build, records) = input.infer(force_build, show_progress)?;
    Ok((input.output_path(), build, records))
}

/// What sex inference reads: one dataset, or a directory of PLINK 1 filesets,
/// such as one per chromosome, read as one.
enum InferenceInput {
    Dataset(GenotypeDataset),
    PlinkDirectory {
        directory: PathBuf,
        filesets: Vec<PlinkDataset>,
    },
}

impl InferenceInput {
    /// The table's default location. As for a directory of VCF files, a
    /// directory's table goes inside it.
    fn output_path(&self) -> PathBuf {
        match self {
            Self::Dataset(dataset) => dataset.output_path("sex.tsv"),
            Self::PlinkDirectory { directory, .. } => directory.join("sex.tsv"),
        }
    }

    fn infer(
        &self,
        force_build: Option<GenomeBuild>,
        show_progress: bool,
    ) -> Result<(GenomeBuild, Vec<SexInferenceRecord>), SexInferenceError> {
        match self {
            Self::Dataset(dataset) => infer_dataset_records(dataset, force_build, show_progress),
            Self::PlinkDirectory { filesets, .. } => {
                infer_directory_records(filesets, force_build, show_progress)
            }
        }
    }
}

fn open_inference_input(
    genotype_path: &Path,
    force_build: Option<GenomeBuild>,
) -> Result<InferenceInput, SexInferenceError> {
    if let Some(beds) = plink_fileset_directory(genotype_path)? {
        let filesets = beds
            .iter()
            .map(PlinkDataset::open)
            .collect::<Result<Vec<_>, _>>()
            .map_err(GenotypeIoError::from)?;
        ensure_same_samples(genotype_path, &filesets)?;
        return Ok(InferenceInput::PlinkDirectory {
            directory: genotype_path.to_path_buf(),
            filesets,
        });
    }
    let pgen_build = force_build.map(|build| match build {
        GenomeBuild::Build37 => PgenGenomeBuild::Grch37,
        GenomeBuild::Build38 => PgenGenomeBuild::Grch38,
    });
    Ok(InferenceInput::Dataset(GenotypeDataset::open(
        genotype_path,
        pgen_build,
    )?))
}

fn infer_dataset_records(
    dataset: &GenotypeDataset,
    force_build: Option<GenomeBuild>,
    show_progress: bool,
) -> Result<(GenomeBuild, Vec<SexInferenceRecord>), SexInferenceError> {
    let loci = match dataset {
        GenotypeDataset::Plink(plink) => {
            VariantLoci::from_bim(plink).map_err(GenotypeIoError::from)?
        }
        _ => VariantLoci::from_keys(&dataset.variant_keys_for_plan(&SelectionPlan::All)?),
    };
    let build = resolve_build(force_build, &loci);
    let selection = SexVariantSelection::from_loci(&loci, build);
    let records = match dataset {
        GenotypeDataset::Plink(plink) => {
            collect_packed_inference(&[(plink, 0)], &selection, show_progress)?
        }
        _ => collect_inference(dataset, &selection, show_progress)?,
    };
    Ok((build, records))
}

/// Infers sex across PLINK 1 filesets that list the same samples.
fn infer_directory_records(
    filesets: &[PlinkDataset],
    force_build: Option<GenomeBuild>,
    show_progress: bool,
) -> Result<(GenomeBuild, Vec<SexInferenceRecord>), SexInferenceError> {
    // The filesets read as one: rows numbered across them in natural file
    // order, so build inference and autosome sampling see every row at once.
    let mut loci = VariantLoci::default();
    let mut parts = Vec::with_capacity(filesets.len());
    for fileset in filesets {
        parts.push((fileset, loci.positions.len()));
        let fileset_loci = VariantLoci::from_bim(fileset).map_err(GenotypeIoError::from)?;
        loci.chroms.extend(fileset_loci.chroms);
        loci.positions.extend(fileset_loci.positions);
    }
    let build = resolve_build(force_build, &loci);
    let selection = SexVariantSelection::from_loci(&loci, build);
    let records = collect_packed_inference(&parts, &selection, show_progress)?;
    Ok((build, records))
}

/// The build `force_build` names, or the one the X positions imply.
fn resolve_build(force_build: Option<GenomeBuild>, loci: &VariantLoci) -> GenomeBuild {
    force_build.unwrap_or_else(|| {
        let inferred = infer_build(loci);
        eprintln!("Inferred Genome Build: {:?}", inferred);
        inferred
    })
}

/// The `.bed` files of a local directory of PLINK 1 filesets, such as one per
/// chromosome, in natural order; `None` when `path` is not a directory holding
/// any. A directory holding VCF or BCF files stays with the variant reader, as
/// before, whatever else it holds.
fn plink_fileset_directory(path: &Path) -> Result<Option<Vec<PathBuf>>, SexInferenceError> {
    if !path.is_dir() || crate::files::list_variant_paths(path).is_ok_and(|files| !files.is_empty())
    {
        return Ok(None);
    }
    let mut beds = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let bed = entry?.path();
        // `gnomon score` leaves `<stem>.sorted.*` caches beside its inputs.
        if bed.is_file()
            && bed.extension().is_some_and(|ext| ext == "bed")
            && !bed
                .file_stem()
                .and_then(|stem| stem.to_str())
                .is_some_and(|stem| stem.ends_with(".sorted"))
        {
            beds.push(bed);
        }
    }
    if beds.is_empty() {
        return Ok(None);
    }
    beds.sort_by(|a, b| natord::compare(&a.to_string_lossy(), &b.to_string_lossy()));
    Ok(Some(beds))
}

/// Filesets counted together must list the same samples, in the same order.
fn ensure_same_samples(
    directory: &Path,
    filesets: &[PlinkDataset],
) -> Result<(), SexInferenceError> {
    let Some((first, rest)) = filesets.split_first() else {
        return Ok(());
    };
    for other in rest {
        let same = first.samples().len() == other.samples().len()
            && first
                .samples()
                .iter()
                .zip(other.samples())
                .all(|(a, b)| a.family_id == b.family_id && a.individual_id == b.individual_id);
        if !same {
            return Err(SexInferenceError::MismatchedFilesetSamples {
                directory: directory.display().to_string(),
                first: first.fam_path().display().to_string(),
                other: other.fam_path().display().to_string(),
            });
        }
    }
    Ok(())
}

fn collect_inference(
    dataset: &GenotypeDataset,
    selection: &SexVariantSelection,
    show_progress: bool,
) -> Result<Vec<SexInferenceRecord>, SexInferenceError> {
    let sample_ids: Vec<String> = dataset
        .samples()
        .iter()
        .map(|record| record.individual_id.clone())
        .collect();
    let n_samples = sample_ids.len();

    let build = selection.build;
    let platform = derive_platform_definition(&selection.keys, build);
    ensure_informative_platform(&platform)?;
    let config = InferenceConfig {
        build,
        platform,
        thresholds: None,
    };
    let mut accumulators: Vec<SexInferenceAccumulator> = (0..n_samples)
        .map(|_| SexInferenceAccumulator::new(config))
        .collect();

    let mut block_source =
        dataset.block_source_with_plan(SelectionPlan::ByIndices(selection.indices.clone()))?;
    // A haploid call is hemizygous, never heterozygous. Read it as the homozygous
    // call a PLINK import makes of it, so a dosage of 1.0 below means two alleles.
    if let DatasetBlockSource::Variants(source) = &mut block_source {
        source.count_haploid_calls_as_homozygous();
    }
    let total_variants = selection.keys.len();
    let block_capacity = 256usize;
    let mut storage = vec![f64::NAN; block_capacity * n_samples];
    let mut processed = 0usize;
    let mut progress = TermsProgress::new(total_variants, show_progress);

    while processed < total_variants {
        progress.update(processed, total_variants);
        let capacity = block_capacity.min(total_variants - processed);
        let slice_len = capacity * n_samples;
        let filled = block_source.next_block_into(capacity, &mut storage[..slice_len])?;
        if filled == 0 {
            return Err(SexInferenceError::VariantUnderflow {
                expected: total_variants,
                observed: processed,
            });
        }
        if processed + filled > total_variants {
            return Err(SexInferenceError::VariantOverflow {
                expected: total_variants,
                observed: processed + filled,
            });
        }

        for local_idx in 0..filled {
            let selected = &selection.keys[processed + local_idx];
            let chrom = selected.chrom;
            let pos = selected.position;
            let column_offset = local_idx * n_samples;

            for sample_idx in 0..n_samples {
                let dosage = storage[column_offset + sample_idx];
                if dosage.is_nan() {
                    continue;
                }
                let is_het = dosage == 1.0;
                let info = VariantInfo {
                    chrom,
                    pos,
                    is_heterozygous: is_het,
                };
                accumulators[sample_idx].process_variant(&info);
            }
        }

        processed += filled;
    }
    progress.finish(total_variants);

    if processed < total_variants {
        return Err(SexInferenceError::VariantUnderflow {
            expected: total_variants,
            observed: processed,
        });
    }

    finalize_records(
        accumulators
            .into_iter()
            .zip(sample_ids)
            .map(|(accumulator, individual_id)| {
                Ok(SexInferenceRecord {
                    individual_id,
                    inference: accumulator.finish()?,
                })
            }),
        &platform,
    )
}

/// [`collect_inference`] for PLINK 1 filesets: one, or several read as one, each
/// paired with the number of rows numbered before it. The calls are counted
/// directly on the packed `.bed` rows, and only the rows that feed a counter are
/// read; see `sex_counts`.
fn collect_packed_inference(
    parts: &[(&PlinkDataset, usize)],
    selection: &SexVariantSelection,
    show_progress: bool,
) -> Result<Vec<SexInferenceRecord>, SexInferenceError> {
    let build = selection.build;
    let platform = derive_platform_definition(&selection.keys, build);
    ensure_informative_platform(&platform)?;
    let config = InferenceConfig {
        build,
        platform,
        thresholds: None,
    };

    let constants = build.algorithm_constants();
    let loci: Vec<(usize, LocusClass)> = selection
        .indices
        .iter()
        .zip(&selection.keys)
        .filter_map(|(&index, selected)| {
            LocusClass::of(&constants, selected.chrom, selected.position)
                .map(|class| (index, class))
        })
        .collect();

    let Some(&(first, _)) = parts.first() else {
        return Ok(Vec::new());
    };
    let total_variants = selection.keys.len();
    let mut progress = TermsProgress::new(total_variants, show_progress);
    let mut evidence = vec![EvidenceCounts::default(); first.n_samples()];
    let mut counted_before = 0;
    for &(dataset, first_row) in parts {
        let end_row = first_row + dataset.n_variants();
        let part_loci: Vec<(usize, LocusClass)> = loci
            .iter()
            .filter(|&&(index, _)| (first_row..end_row).contains(&index))
            .map(|&(index, class)| (index - first_row, class))
            .collect();
        if part_loci.is_empty() {
            continue;
        }
        let rows = BedRows::new(
            dataset.bed_source(),
            dataset.bytes_per_variant(),
            dataset.n_variants(),
            dataset.n_samples(),
        );
        let part_evidence = count_evidence(&rows, &part_loci, |counted| {
            progress.update(counted_before + counted, total_variants)
        })
        .map_err(|err| GenotypeIoError::from(PlinkIoError::from(err)))?;
        counted_before += part_loci.len();
        // Every counter is a count of calls, so the filesets' counts add.
        for (total, counts) in evidence.iter_mut().zip(&part_evidence) {
            total.auto_valid += counts.auto_valid;
            total.auto_het += counts.auto_het;
            total.x_par_valid += counts.x_par_valid;
            total.x_par_het += counts.x_par_het;
            total.x_non_par_valid += counts.x_non_par_valid;
            total.x_non_par_het += counts.x_non_par_het;
            total.y_par_valid += counts.y_par_valid;
            total.y_non_par_valid += counts.y_non_par_valid;
        }
    }
    progress.finish(total_variants);

    finalize_records(
        first
            .samples()
            .iter()
            .zip(&evidence)
            .map(|(sample, counts)| {
                Ok(SexInferenceRecord {
                    individual_id: sample.individual_id.clone(),
                    inference: finish_counts(&config, counts)?,
                })
            }),
        &platform,
    )
}

/// True when the Y density cannot separate the sexes in this panel: the panel
/// carries no Y non-PAR locus, or no sample missed any Y non-PAR locus it carries.
///
/// `derive_platform_definition` defines "attempted" as "present in this file",
/// so when every sample called every attempted Y locus the Y call rate is
/// `observed / observed = 1.0` by construction -- for every sample, of either
/// sex. It is not a measurement of Y coverage, it is a restatement of the
/// filtering. chrY dropout is what distinguishes the sexes, and a panel without
/// any has been filtered to called sites, or carries female chrY calls that no
/// genotyper should have made.
///
/// This belongs to the panel, not to a sample. A panel with real dropout can
/// still hold samples that called every locus -- males on WGS calls whose
/// autosomes carry no no-calls, or on a thin Y panel -- and their densities
/// measure something, because the females beside them did miss chrY.
fn y_density_is_uninformative(
    records: &[SexInferenceRecord],
    platform: &PlatformDefinition,
) -> bool {
    platform.n_attempted_y_nonpar == 0
        || records.iter().all(|record| {
            record.inference.report.y_non_par_valid_count == platform.n_attempted_y_nonpar
        })
}

/// The X/autosome heterozygosity ratio under which a sample is called male on
/// X alone. A male's non-PAR X is hemizygous, so every heterozygous call he
/// shows there is an error, while a female's X is diploid and heterozygous on
/// the order of her autosomes. PLINK's `--check-sex` calls a male when his X
/// heterozygosity is under a fifth of its expectation (F above 0.8); the ratio
/// takes the sample's own autosomal heterozygosity as that expectation.
const MALE_X_AUTOSOME_HET_RATIO: f64 = 0.2;

/// The call on X evidence alone.
fn call_on_x(report: &EvidenceReport) -> InferredSex {
    match report.x_autosome_het_ratio {
        Some(ratio) if ratio < MALE_X_AUTOSOME_HET_RATIO => InferredSex::Male,
        Some(_) => InferredSex::Female,
        None => InferredSex::Indeterminate,
    }
}

/// Collects every sample's result. Where the panel's Y density is
/// uninformative, the calls come from X alone.
fn finalize_records(
    records: impl IntoIterator<Item = Result<SexInferenceRecord, InferenceError>>,
    platform: &PlatformDefinition,
) -> Result<Vec<SexInferenceRecord>, SexInferenceError> {
    let mut records = records.into_iter().collect::<Result<Vec<_>, _>>()?;
    // infer_sex reads a missing density as zero, which calls every sample
    // female, and a saturated density is one, which calls nearly every sample
    // male -- most confidently the females, whose no-calls were the signal that
    // got removed. Neither is evidence; the X heterozygosity is.
    if y_density_is_uninformative(&records, platform) {
        for record in &mut records {
            record.inference.final_call = call_on_x(&record.inference.report);
        }
    }
    Ok(records)
}

fn write_results(
    path: &Path,
    records: &[SexInferenceRecord],
    build: GenomeBuild,
) -> Result<(), SexInferenceError> {
    crate::output::write_atomically(path, |writer| {
        writeln!(writer, "{}", SEX_TSV_HEADER)?;
        for record in records {
            let label = sex_label(record.inference.final_call);
            let report = &record.inference.report;
            let y_density = report
                .y_genome_density
                .map_or("NA".to_string(), |v| format!("{v:.6}"));
            let x_ratio = report
                .x_autosome_het_ratio
                .map_or("NA".to_string(), |v| format!("{v:.6}"));
            let composite = report
                .composite_sex_index
                .map_or("NA".to_string(), |v| format!("{v:.6}"));

            writeln!(
                writer,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                record.individual_id,
                format!("{:?}", build),
                label,
                y_density,
                x_ratio,
                composite,
                report.auto_valid_count,
                report.auto_het_count,
                report.x_non_par_valid_count,
                report.x_non_par_het_count,
                report.y_non_par_valid_count,
                report.y_par_valid_count,
            )?;
        }
        Ok(())
    })?;
    Ok(())
}

fn sex_label(sex: InferredSex) -> &'static str {
    match sex {
        InferredSex::Male => "male",
        InferredSex::Female => "female",
        InferredSex::Indeterminate => "indeterminate",
    }
}

fn infer_build(loci: &VariantLoci) -> GenomeBuild {
    const GRCH38_THRESHOLD: u64 = 155_700_000;
    const GRCH37_THRESHOLD: u64 = 154_900_000;

    let max_x = loci
        .chroms
        .iter()
        .zip(&loci.positions)
        .filter(|&(&chrom, _)| matches!(chrom, Some(LocusChromosome::X | LocusChromosome::XPar)))
        .map(|(_, &position)| position)
        .max();

    match max_x {
        Some(pos) if pos >= GRCH38_THRESHOLD => GenomeBuild::Build38,
        Some(pos) if pos >= GRCH37_THRESHOLD => GenomeBuild::Build37,
        Some(_) => GenomeBuild::Build38,
        None => GenomeBuild::Build38,
    }
}

fn classify_chromosome(label: &str) -> Option<LocusChromosome> {
    let trimmed = label.trim();
    let bytes = trimmed.as_bytes();
    let stripped = if bytes.len() >= 3
        && matches!(bytes[0] | 32, b'c')
        && matches!(bytes[1] | 32, b'h')
        && matches!(bytes[2] | 32, b'r')
    {
        &trimmed[3..]
    } else {
        trimmed
    };

    for (name, chrom) in [
        ("X", LocusChromosome::X),
        ("Y", LocusChromosome::Y),
        ("XY", LocusChromosome::XPar),
        ("PAR1", LocusChromosome::XPar),
        ("PAR2", LocusChromosome::XPar),
    ] {
        if stripped.eq_ignore_ascii_case(name) {
            return Some(chrom);
        }
    }

    match stripped.parse::<u8>().ok()? {
        23 => Some(LocusChromosome::X),
        24 => Some(LocusChromosome::Y),
        25 => Some(LocusChromosome::XPar),
        chrom if (1..=22).contains(&chrom) => Some(LocusChromosome::Autosome),
        _ => None,
    }
}

fn derive_platform_definition(keys: &[SelectedVariant], build: GenomeBuild) -> PlatformDefinition {
    let mut n_attempted_autosomes = 0u64;
    let mut n_attempted_y_nonpar = 0u64;

    for selected in keys {
        match selected.chrom {
            Chromosome::Autosome => n_attempted_autosomes += 1,
            Chromosome::Y if build.is_in_y_non_par(selected.position) => {
                n_attempted_y_nonpar += 1
            }
            _ => {}
        }
    }

    PlatformDefinition {
        n_attempted_autosomes,
        n_attempted_y_nonpar,
    }
}

fn ensure_informative_platform(platform: &PlatformDefinition) -> Result<(), SexInferenceError> {
    if platform.n_attempted_autosomes == 0 {
        return Err(SexInferenceError::InsufficientInformativeVariants {
            autosomes: platform.n_attempted_autosomes,
            y_non_par: platform.n_attempted_y_nonpar,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufWriter;
    use tempfile::tempdir;

    #[test]
    fn classify_chromosome_recognizes_common_labels() {
        assert_eq!(classify_chromosome("X"), Some(LocusChromosome::X));
        assert_eq!(classify_chromosome("chrX"), Some(LocusChromosome::X));
        assert_eq!(classify_chromosome("Y"), Some(LocusChromosome::Y));
        assert_eq!(classify_chromosome("chrY"), Some(LocusChromosome::Y));
        assert_eq!(classify_chromosome("1"), Some(LocusChromosome::Autosome));
        assert_eq!(
            classify_chromosome("chr22"),
            Some(LocusChromosome::Autosome)
        );
        for label in ["25", "XY", "xy", "chrXY", "PAR1", "PAR2", "chrPAR2", "par1"] {
            assert_eq!(
                classify_chromosome(label),
                Some(LocusChromosome::XPar),
                "{label}"
            );
        }
        assert_eq!(classify_chromosome("MT"), None);
        assert_eq!(classify_chromosome("26"), None);
        assert_eq!(classify_chromosome("PAR3"), None);
    }

    #[test]
    fn infer_build_detects_build_thresholds() {
        let loci = VariantLoci::from_keys(&[VariantKey::new("chrX", 155_800_000)]);
        assert_eq!(infer_build(&loci), GenomeBuild::Build38);

        let loci = VariantLoci::from_keys(&[VariantKey::new("X", 155_000_000)]);
        assert_eq!(infer_build(&loci), GenomeBuild::Build37);

        assert_eq!(infer_build(&VariantLoci::default()), GenomeBuild::Build38);
    }

    /// hg38 non-PAR X ends below the Build38 threshold, so with PAR2 coded apart
    /// from X the build is only visible in the PAR-coded rows. Those rows count as
    /// X PAR inside the PAR intervals and nowhere else.
    #[test]
    fn par_coded_rows_set_the_build_and_count_only_inside_the_par() {
        let keys = [
            VariantKey::new("1", 1_000),
            VariantKey::new("X", 3_000_000),
            VariantKey::new("X", 155_000_000),
            VariantKey::new("XY", 1_000_000),
            VariantKey::new("25", 3_000_000),
            VariantKey::new("PAR2", 155_800_000),
        ];
        let loci = VariantLoci::from_keys(&keys);
        assert_eq!(infer_build(&loci), GenomeBuild::Build38);

        let selection = SexVariantSelection::from_loci(&loci, GenomeBuild::Build38);
        assert_eq!(selection.indices, vec![0, 1, 2, 3, 5]);
        assert!(
            selection.keys[1..]
                .iter()
                .all(|selected| selected.chrom == Chromosome::X)
        );
    }

    #[test]
    fn platform_definition_counts_autosomes_and_y_nonpar() {
        let build = GenomeBuild::Build38;
        let keys = vec![
            SelectedVariant {
                position: 1_000,
                chrom: Chromosome::Autosome,
            },
            SelectedVariant {
                position: 2_000,
                chrom: Chromosome::Autosome,
            },
            SelectedVariant {
                position: 3_000_000,
                chrom: Chromosome::Y,
            },
            SelectedVariant {
                position: 56_887_902,
                chrom: Chromosome::Y,
            },
            SelectedVariant {
                position: 56_887_903,
                chrom: Chromosome::Y,
            },
        ];

        let platform = derive_platform_definition(&keys, build);
        assert_eq!(platform.n_attempted_autosomes, 2);
        assert_eq!(platform.n_attempted_y_nonpar, 2);
    }

    #[test]
    fn ensure_informative_platform_rejects_missing_loci() {
        let ok_platform = PlatformDefinition {
            n_attempted_autosomes: 1,
            n_attempted_y_nonpar: 1,
        };
        ensure_informative_platform(&ok_platform).expect("platform should be accepted");

        let missing_autosomes = PlatformDefinition {
            n_attempted_autosomes: 0,
            n_attempted_y_nonpar: 5,
        };
        let err = ensure_informative_platform(&missing_autosomes)
            .expect_err("should reject missing autosomes");
        match err {
            SexInferenceError::InsufficientInformativeVariants {
                autosomes,
                y_non_par,
            } => {
                assert_eq!(autosomes, 0);
                assert_eq!(y_non_par, 5);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn selection_limits_autosomes_but_keeps_sex_chromosomes() {
        let build = GenomeBuild::Build38;
        let mut keys = Vec::new();
        for i in 0..3000u64 {
            keys.push(VariantKey::new("1", 10_000 + i));
        }
        keys.push(VariantKey::new("X", 3_000_000));
        keys.push(VariantKey::new("Y", 3_000_000));
        keys.push(VariantKey::new("Y", 56_887_903));

        let selection = SexVariantSelection::from_loci(&VariantLoci::from_keys(&keys), build);

        assert_eq!(selection.keys.len(), 2003);
        let indices = &selection.indices;
        assert_eq!(indices.len(), selection.keys.len());
        assert!(indices.windows(2).all(|w| w[0] < w[1]));
        assert!(indices.contains(&0));
        assert!(indices.contains(&3000));
        assert!(indices.contains(&3001));
        assert!(indices.contains(&3002));

        let autosome_indices: Vec<_> = indices.iter().copied().filter(|idx| *idx < 3000).collect();
        assert_eq!(autosome_indices.len(), 2000);
        assert!(autosome_indices.contains(&0));
        assert!(autosome_indices.contains(&1500));
        assert!(autosome_indices.contains(&2998));
    }

    #[test]
    fn manual_accumulators_produce_expected_calls() {
        let build = GenomeBuild::Build38;
        let platform = PlatformDefinition {
            n_attempted_autosomes: 400,
            n_attempted_y_nonpar: 80,
        };
        let config = InferenceConfig {
            build,
            platform,
            thresholds: None,
        };

        let mut female_acc = SexInferenceAccumulator::new(config);
        let mut male_acc = SexInferenceAccumulator::new(config);

        for i in 0..400u64 {
            let pos = 1_000_000 + i;
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos,
                is_heterozygous: i % 2 == 0,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos,
                is_heterozygous: i % 3 == 0,
            });
        }

        for i in 0..200u64 {
            let pos = 3_000_000 + i;
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos,
                is_heterozygous: true,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos,
                is_heterozygous: false,
            });
        }

        for i in 0..80u64 {
            let pos = 3_000_000 + i;
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Y,
                pos,
                is_heterozygous: false,
            });
        }

        let female_result = female_acc.finish().unwrap();
        let male_result = male_acc.finish().unwrap();

        assert_eq!(female_result.final_call, InferredSex::Female);
        assert_eq!(male_result.final_call, InferredSex::Male);

        assert!(
            female_result
                .report
                .y_genome_density
                .expect("female density")
                < 0.2
        );
        assert!(
            female_result
                .report
                .x_autosome_het_ratio
                .expect("female ratio")
                > 0.5
        );
        assert!(male_result.report.y_genome_density.expect("male density") > 0.5);
        assert!(male_result.report.x_autosome_het_ratio.expect("male ratio") < 0.2);
    }

    /// A female genotyped on a panel filtered to called sites.
    ///
    /// Her chrY loci are absent from the input, so they are never "attempted"
    /// either, and the density she produces is 1.0 -- the same value a male
    /// produces, for the opposite reason. The evidence that separates them was
    /// removed before the file was written, and the density can no longer see
    /// the difference.
    #[test]
    fn saturated_panel_is_detected_when_no_locus_is_missing() {
        let build = GenomeBuild::Build38;
        // "Attempted" counted off a file that retains only called sites.
        let platform = PlatformDefinition {
            n_attempted_autosomes: 400,
            n_attempted_y_nonpar: 12,
        };
        let config = InferenceConfig {
            build,
            platform,
            thresholds: None,
        };
        let mut acc = SexInferenceAccumulator::new(config);

        for i in 0..400u64 {
            acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos: 1_000_000 + i,
                is_heterozygous: i % 2 == 0,
            });
        }
        // Every attempted Y locus calls, because the ones that failed are gone.
        for i in 0..12u64 {
            acc.process_variant(&VariantInfo {
                chrom: Chromosome::Y,
                pos: 3_000_000 + i,
                is_heterozygous: false,
            });
        }

        let result = acc.finish().unwrap();
        let density = result.report.y_genome_density.expect("density");
        // infer_sex adds its epsilon (1e-9) to the autosome numerator, so the
        // saturated density is 400 / (400 + 1e-9), a hair below 1.0.
        assert!(
            (density - 1.0).abs() < 1e-9,
            "a filtered panel yields a saturated density by construction, got {density}"
        );
        assert!(
            y_density_is_uninformative(&[record("F1", result)], &platform),
            "zero missingness across the panel must be recognized"
        );
    }

    fn record(individual_id: &str, inference: InferenceResult) -> SexInferenceRecord {
        SexInferenceRecord {
            individual_id: individual_id.to_string(),
            inference,
        }
    }

    /// A male genotyped at every locus of a thin chrY panel, or on WGS calls
    /// whose autosomes carry no no-calls, beside females who miss chrY. The
    /// panel has real dropout, so his density measures something and his call
    /// stands. Alone, the same male is a panel without dropout, and his call
    /// comes from his X.
    #[test]
    fn a_male_who_calls_every_locus_keeps_his_call_in_a_panel_with_dropout() {
        let platform = PlatformDefinition {
            n_attempted_autosomes: 400,
            n_attempted_y_nonpar: 12,
        };
        let config = InferenceConfig {
            build: GenomeBuild::Build38,
            platform,
            thresholds: None,
        };
        let finish = |male: bool| {
            let mut acc = SexInferenceAccumulator::new(config);
            for i in 0..400u64 {
                acc.process_variant(&VariantInfo {
                    chrom: Chromosome::Autosome,
                    pos: 1_000_000 + i,
                    is_heterozygous: i % 3 == 0,
                });
            }
            for i in 0..200u64 {
                acc.process_variant(&VariantInfo {
                    chrom: Chromosome::X,
                    pos: 3_000_000 + i,
                    is_heterozygous: !male && i % 2 == 0,
                });
            }
            if male {
                for i in 0..12u64 {
                    acc.process_variant(&VariantInfo {
                        chrom: Chromosome::Y,
                        pos: 3_000_000 + i,
                        is_heterozygous: false,
                    });
                }
            }
            acc.finish()
        };

        let panel = finalize_records(
            [(true, "M1"), (false, "F1"), (true, "M2")]
                .map(|(male, id)| Ok(record(id, finish(male)?))),
            &platform,
        )
        .unwrap();
        let calls: Vec<InferredSex> = panel
            .iter()
            .map(|record| record.inference.final_call)
            .collect();
        assert_eq!(
            calls,
            [InferredSex::Male, InferredSex::Female, InferredSex::Male]
        );

        let alone = finalize_records([Ok(record("M1", finish(true).unwrap()))], &platform).unwrap();
        assert_eq!(alone[0].inference.final_call, InferredSex::Male);
    }

    /// The ordinary case, which must keep working: some loci fail to call, so
    /// the density measures something real and the call stands.
    #[test]
    fn a_panel_with_dropout_is_not_saturated() {
        let platform = PlatformDefinition {
            n_attempted_autosomes: 400,
            n_attempted_y_nonpar: 80,
        };
        let config = InferenceConfig {
            build: GenomeBuild::Build38,
            platform,
            thresholds: None,
        };
        let mut acc = SexInferenceAccumulator::new(config);

        for i in 0..400u64 {
            acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos: 1_000_000 + i,
                is_heterozygous: i % 3 == 0,
            });
        }
        // 6 of 80 attempted Y loci call: real chrY dropout.
        for i in 0..6u64 {
            acc.process_variant(&VariantInfo {
                chrom: Chromosome::Y,
                pos: 3_000_000 + i,
                is_heterozygous: false,
            });
        }

        let result = acc.finish().unwrap();
        assert!(
            !y_density_is_uninformative(&[record("M1", result)], &platform),
            "a panel with genuine dropout must keep its call"
        );
    }

    /// With no chrY locus at all, infer_sex reads the missing density as zero
    /// and calls every sample female. The calls must come from X instead: a
    /// male's hemizygous X shows no heterozygosity, a female's about as much as
    /// her autosomes, and a sample without an X call has nothing to be called on.
    #[test]
    fn a_panel_without_any_y_locus_is_called_on_x() {
        let platform = PlatformDefinition {
            n_attempted_autosomes: 400,
            n_attempted_y_nonpar: 0,
        };
        let config = InferenceConfig {
            build: GenomeBuild::Build38,
            platform,
            thresholds: None,
        };
        // Heterozygous calls among 200 non-PAR X calls, or no X call at all.
        let finish = |x_het: Option<u64>| {
            let mut acc = SexInferenceAccumulator::new(config);
            for i in 0..400u64 {
                acc.process_variant(&VariantInfo {
                    chrom: Chromosome::Autosome,
                    pos: 1_000_000 + i,
                    is_heterozygous: i % 2 == 0,
                });
            }
            for i in 0..x_het.map_or(0, |_| 200u64) {
                acc.process_variant(&VariantInfo {
                    chrom: Chromosome::X,
                    pos: 3_000_000 + i,
                    is_heterozygous: x_het.is_some_and(|x_het| i < x_het),
                });
            }
            record("S", acc.finish().unwrap())
        };
        // Autosomal heterozygosity is 1/2, so 19 of 200 X calls heterozygous is a
        // ratio of 0.19, and 21 is 0.21.
        let panel = [Some(0), Some(19), Some(21), Some(100), None].map(finish);
        assert!(y_density_is_uninformative(&panel, &platform));
        let calls: Vec<InferredSex> = finalize_records(panel.map(Ok), &platform)
            .unwrap()
            .iter()
            .map(|record| record.inference.final_call)
            .collect();
        assert_eq!(
            calls,
            [
                InferredSex::Male,
                InferredSex::Male,
                InferredSex::Female,
                InferredSex::Female,
                InferredSex::Indeterminate
            ]
        );
    }

    /// Female chrY calls that no genotyper should have made -- no-calls filled
    /// as reference, or reads mismapped to chrY -- leave the panel without Y
    /// dropout. Her density is then the male one, and infer_sex calls her male
    /// most confidently of all; the call must come from her X.
    #[test]
    fn a_panel_without_y_dropout_is_called_on_x() {
        let platform = PlatformDefinition {
            n_attempted_autosomes: 400,
            n_attempted_y_nonpar: 12,
        };
        let config = InferenceConfig {
            build: GenomeBuild::Build38,
            platform,
            thresholds: None,
        };
        let finish = |male: bool| {
            let mut acc = SexInferenceAccumulator::new(config);
            // One autosomal no-call, so the panel is not saturated as a whole.
            for i in 1..400u64 {
                acc.process_variant(&VariantInfo {
                    chrom: Chromosome::Autosome,
                    pos: 1_000_000 + i,
                    is_heterozygous: i % 3 == 0,
                });
            }
            for i in 0..200u64 {
                acc.process_variant(&VariantInfo {
                    chrom: Chromosome::X,
                    pos: 3_000_000 + i,
                    is_heterozygous: !male && i % 2 == 0,
                });
            }
            for i in 0..12u64 {
                acc.process_variant(&VariantInfo {
                    chrom: Chromosome::Y,
                    pos: 3_000_000 + i,
                    is_heterozygous: false,
                });
            }
            acc.finish()
        };
        let female = finish(false).unwrap();
        assert_eq!(female.final_call, InferredSex::Male, "infer_sex alone");

        let panel = finalize_records(
            [(false, "F1"), (true, "M1")].map(|(male, id)| Ok(record(id, finish(male)?))),
            &platform,
        )
        .unwrap();
        assert_eq!(panel[0].inference.final_call, InferredSex::Female);
        assert_eq!(panel[1].inference.final_call, InferredSex::Male);
    }

    #[test]
    fn sex_tsv_includes_metric_columns() -> Result<(), Box<dyn std::error::Error>> {
        let build = GenomeBuild::Build38;
        let platform = PlatformDefinition {
            n_attempted_autosomes: 10,
            n_attempted_y_nonpar: 2,
        };
        let config = InferenceConfig {
            build,
            platform,
            thresholds: None,
        };

        let mut female_acc = SexInferenceAccumulator::new(config);
        let mut male_acc = SexInferenceAccumulator::new(config);

        for i in 0..10u64 {
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos: 1_000 + i,
                is_heterozygous: i % 2 == 0,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Autosome,
                pos: 1_000 + i,
                is_heterozygous: i % 3 == 0,
            });
        }
        for i in 0..2u64 {
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::Y,
                pos: 3_000_000 + i,
                is_heterozygous: false,
            });
        }
        for i in 0..4u64 {
            female_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos: 3_000_000 + i,
                is_heterozygous: true,
            });
            male_acc.process_variant(&VariantInfo {
                chrom: Chromosome::X,
                pos: 3_000_000 + i,
                is_heterozygous: false,
            });
        }

        let female = SexInferenceRecord {
            individual_id: "F1".to_string(),
            inference: female_acc.finish().unwrap(),
        };
        let male = SexInferenceRecord {
            individual_id: "M1".to_string(),
            inference: male_acc.finish().unwrap(),
        };

        let dir = tempdir()?;
        let output_path = dir.path().join("sex.tsv");
        write_results(&output_path, &[female, male], build)?;
        let tsv_contents = std::fs::read_to_string(&output_path)?;
        let mut lines = tsv_contents.lines();
        let header = lines.next().unwrap();
        assert_eq!(header, SEX_TSV_HEADER);

        for line in lines {
            let mut parts = line.split('\t');
            let _ = parts.next().unwrap();
            let build = parts.next().unwrap();
            let sex = parts.next().unwrap();
            let y_density = parts.next().unwrap();
            let x_ratio = parts.next().unwrap();
            let composite = parts.next().unwrap();
            let auto_valid = parts.next().unwrap();
            let auto_het = parts.next().unwrap();
            assert_ne!(y_density, "NA");
            assert_ne!(x_ratio, "NA");
            assert_ne!(composite, "NA");
            assert!(auto_valid.parse::<u64>().unwrap() > 0);
            assert!(auto_het.parse::<u64>().unwrap() > 0);
            let x_valid = parts.next().unwrap().parse::<u64>().unwrap();
            let _ = parts.next().unwrap();
            let y_non_par = parts.next().unwrap().parse::<u64>().unwrap();
            let y_par = parts.next().unwrap().parse::<u64>().unwrap();
            if sex == "female" {
                assert_eq!(y_non_par, 0);
                assert_eq!(y_par, 0);
            } else {
                assert!(y_non_par > 0);
                assert_eq!(y_par, 0);
            }
            assert!(x_valid > 0);
            assert_eq!(build, "Build38");
        }

        Ok(())
    }

    fn metric_bits(result: &InferenceResult) -> [Option<u64>; 3] {
        [
            result.report.y_genome_density.map(f64::to_bits),
            result.report.x_autosome_het_ratio.map(f64::to_bits),
            result.report.composite_sex_index.map(f64::to_bits),
        ]
    }

    /// Writes a PLINK 1 fileset whose rows cover every locus class, every Build38
    /// PAR boundary, loci the accumulator ignores, missing calls and dirty padding
    /// codes, and returns its `.bed` path. Odd samples look male, even samples
    /// female; with no missing calls the panel is saturated.
    fn write_sex_fixture(
        dir: &Path,
        n_samples: usize,
        missing_percent: u64,
    ) -> std::io::Result<PathBuf> {
        let mut state = 0x853c_49e6_748f_ea9b_u64 ^ n_samples as u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut rows: Vec<(String, u64)> = (0..2_600u64)
            .map(|i| ((1 + i * 22 / 2_600).to_string(), 1_000 + i * 10))
            .collect();
        let mut x_positions: Vec<u64> = (0..700u64).map(|i| 5_000 + i * 224_000).collect();
        x_positions.extend([
            10_000,
            10_001,
            2_781_479,
            2_781_480,
            155_701_382,
            155_701_383,
            156_030_895,
            156_030_896,
        ]);
        x_positions.sort_unstable();
        rows.extend(x_positions.into_iter().map(|pos| ("X".to_string(), pos)));
        let mut y_positions: Vec<u64> = (0..300u64).map(|i| 5_000 + i * 200_000).collect();
        y_positions.extend([
            10_000, 10_001, 2_781_479, 2_781_480, 56_887_902, 56_887_903, 57_217_415, 57_217_416,
        ]);
        y_positions.sort_unstable();
        rows.extend(y_positions.into_iter().map(|pos| ("Y".to_string(), pos)));
        rows.push(("MT".to_string(), 100));

        let prefix = dir.join(format!("fixture_{n_samples}_{missing_percent}"));
        let mut bim = BufWriter::new(File::create(prefix.with_extension("bim"))?);
        for (index, (chrom, pos)) in rows.iter().enumerate() {
            writeln!(bim, "{chrom}\tv{index}\t0\t{pos}\tA\tG")?;
        }
        bim.flush()?;
        let mut fam = BufWriter::new(File::create(prefix.with_extension("fam"))?);
        for sample in 0..n_samples {
            writeln!(fam, "F{sample}\tI{sample}\t0\t0\t0\t-9")?;
        }
        fam.flush()?;

        let row_len = n_samples.div_ceil(4);
        let mut bed = BufWriter::new(File::create(prefix.with_extension("bed"))?);
        bed.write_all(&[0x6c, 0x1b, 0x01])?;
        for (chrom, _) in &rows {
            let mut row = vec![0u8; row_len];
            for slot in 0..4 * row_len {
                let roll = next();
                let male = slot % 2 == 1;
                let code = if slot >= n_samples {
                    // Padding codes, which the reader must ignore.
                    (roll & 0b11) as u8
                } else if roll % 100 < missing_percent {
                    0b01
                } else {
                    match chrom.as_str() {
                        "X" if male => [0b00, 0b11][((roll >> 8) & 1) as usize],
                        "Y" if !male && missing_percent > 0 && (roll >> 16) % 10 != 0 => 0b01,
                        _ => [0b00, 0b10, 0b11][((roll >> 24) % 3) as usize],
                    }
                };
                row[slot / 4] |= code << (2 * (slot % 4));
            }
            bed.write_all(&row)?;
        }
        bed.flush()?;
        Ok(prefix.with_extension("bed"))
    }

    /// The packed path must reproduce the accumulator path record for record,
    /// down to the bits of every metric.
    #[test]
    fn packed_counts_reproduce_the_accumulator_path() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        for (n_samples, missing_percent) in [(1, 3), (7, 3), (64, 0), (203, 20)] {
            let bed = write_sex_fixture(dir.path(), n_samples, missing_percent)?;
            let dataset = GenotypeDataset::open(&bed, None)?;
            let GenotypeDataset::Plink(plink) = &dataset else {
                panic!("the fixture is a PLINK 1 fileset");
            };
            let keys = dataset.variant_keys_for_plan(&SelectionPlan::All)?;
            let loci = VariantLoci::from_keys(&keys);
            assert_eq!(VariantLoci::from_bim(plink)?, loci);
            let build = infer_build(&loci);
            assert_eq!(build, GenomeBuild::Build38);
            let selection = SexVariantSelection::from_loci(&loci, build);
            let expected = collect_inference(&dataset, &selection, false)?;
            let packed = collect_packed_inference(&[(plink, 0)], &selection, false)?;

            assert_eq!(packed.len(), n_samples);
            assert_eq!(expected.len(), n_samples);
            for (packed, expected) in packed.iter().zip(&expected) {
                assert_eq!(packed.individual_id, expected.individual_id);
                assert_eq!(
                    packed.inference, expected.inference,
                    "{n_samples} samples, {missing_percent}% missing"
                );
                assert_eq!(
                    metric_bits(&packed.inference),
                    metric_bits(&expected.inference)
                );
            }
            let calls: HashSet<&str> = packed
                .iter()
                .map(|record| sex_label(record.inference.final_call))
                .collect();
            // A saturated panel is called on X, which separates the fixture's sexes too.
            if n_samples > 1 {
                assert_eq!(calls, HashSet::from(["male", "female"]));
            }
        }
        Ok(())
    }

    /// The `.bim` scan must classify every label as the normalized keys do,
    /// including spellings that only resolve after normalization.
    #[test]
    fn bim_loci_classify_labels_like_normalized_keys() -> Result<(), Box<dyn std::error::Error>> {
        let labels = [
            "1", "01", "+1", "chr1", "CHR01", "chrchr1", "22", "23", "+23", "chr+23", "024", "x",
            "chrX", "ChRx", "chrchrX", "X", "Y", "chry", "25", "XY", "PAR1", "MT", "chrM", "0",
            "-1", "256", "chrUn_gl000220", "1", "X",
        ];
        let dir = tempdir()?;
        let prefix = dir.path().join("labels");
        let mut bim = BufWriter::new(File::create(prefix.with_extension("bim"))?);
        for (index, label) in labels.iter().enumerate() {
            writeln!(bim, "{label}\tv{index}\t0\t{}\tA\tG", 1_000 + index)?;
        }
        bim.flush()?;
        let mut fam = BufWriter::new(File::create(prefix.with_extension("fam"))?);
        writeln!(fam, "F0\tI0\t0\t0\t0\t-9")?;
        fam.flush()?;
        let mut bed = BufWriter::new(File::create(prefix.with_extension("bed"))?);
        bed.write_all(&[0x6c, 0x1b, 0x01])?;
        bed.write_all(&vec![0u8; labels.len()])?;
        bed.flush()?;

        let dataset = GenotypeDataset::open(prefix.with_extension("bed"), None)?;
        let GenotypeDataset::Plink(plink) = &dataset else {
            panic!("the fixture is a PLINK 1 fileset");
        };
        let keys = dataset.variant_keys_for_plan(&SelectionPlan::All)?;
        let from_bim = VariantLoci::from_bim(plink)?;
        assert_eq!(from_bim, VariantLoci::from_keys(&keys));
        assert_eq!(from_bim.chroms[2], Some(LocusChromosome::Autosome));
        assert_eq!(from_bim.chroms[5], Some(LocusChromosome::Autosome));
        assert_eq!(from_bim.chroms[14], Some(LocusChromosome::X));
        assert_eq!(from_bim.chroms[18], Some(LocusChromosome::XPar));
        assert_eq!(from_bim.chroms[19], Some(LocusChromosome::XPar));
        assert_eq!(from_bim.chroms[20], Some(LocusChromosome::XPar));
        assert_eq!(from_bim.chroms[21], None);
        Ok(())
    }

    /// The mapped scan must find the rows the record reader finds, on any
    /// chunking and whatever the line endings, separators and blank lines.
    #[test]
    fn mapped_bim_scan_reads_what_the_record_reader_reads() -> Result<(), Box<dyn std::error::Error>>
    {
        let lines = [
            "1\tv0\t0\t1000\tA\tG\n",
            "1 v1 0  1001 A G\r\n",
            "\n",
            "   \t\r\n",
            "chr1\x0bv2\x0b0\x0b+1002\x0bA\x0bG\n",
            "1\u{a0}v3\u{a0}0\u{a0}1003\u{a0}A\u{a0}G\n",
            "X\tv4\t0\t155800000\tA\tG\n",
            "25\tv5\t0\t155900000\tA\tG\r\n",
            "Y\tv6\t0\t3000000\tA\tG",
        ];
        let dir = tempdir()?;
        let prefix = dir.path().join("scan");
        std::fs::write(prefix.with_extension("bim"), lines.concat())?;
        std::fs::write(prefix.with_extension("fam"), "F0\tI0\t0\t0\t0\t-9\n")?;
        std::fs::write(
            prefix.with_extension("bed"),
            [0x6c, 0x1b, 0x01, 0, 0, 0, 0, 0, 0, 0],
        )?;

        let dataset = GenotypeDataset::open(prefix.with_extension("bed"), None)?;
        let GenotypeDataset::Plink(plink) = &dataset else {
            panic!("the fixture is a PLINK 1 fileset");
        };
        let mut expected = VariantLoci::default();
        plink.for_each_variant_position(|label, position| {
            expected.chroms.push(classify_chromosome(
                &VariantKey::new(label, position).chromosome,
            ));
            expected.positions.push(position);
        })?;
        assert_eq!(
            expected.positions,
            [1000, 1001, 1002, 1003, 155_800_000, 155_900_000, 3_000_000]
        );
        for chunk_bytes in [1, 2, 7, 64, BIM_SCAN_CHUNK_BYTES] {
            assert_eq!(
                VariantLoci::scan_local_bim(plink, chunk_bytes).as_ref(),
                Some(&expected),
                "{chunk_bytes}-byte chunks"
            );
        }
        assert_eq!(VariantLoci::from_bim(plink)?, expected);
        Ok(())
    }

    /// A directory of per-chromosome PLINK filesets must infer exactly what the
    /// same rows in one fileset infer, filesets whose samples differ are refused,
    /// and a directory that also holds VCF files stays with the variant reader.
    #[test]
    fn per_chromosome_filesets_infer_what_one_fileset_infers()
    -> Result<(), Box<dyn std::error::Error>> {
        let merged_dir = tempdir()?;
        let bed = write_sex_fixture(merged_dir.path(), 64, 3)?;
        let (_, expected_build, expected) = infer_records(&bed, None, false)?;

        // Split the fixture at every change of chromosome label, as per-chromosome
        // exports are, naming the parts `<stem>.chr<label>`.
        let bim = std::fs::read_to_string(bed.with_extension("bim"))?;
        let fam = std::fs::read_to_string(bed.with_extension("fam"))?;
        let payload = std::fs::read(&bed)?;
        let row_len = 64usize.div_ceil(4);
        let mut parts: Vec<(String, String, Vec<u8>)> = Vec::new();
        for (index, line) in bim.lines().enumerate() {
            let label = line.split('\t').next().unwrap_or_default();
            if parts.last().is_none_or(|(last, _, _)| last != label) {
                parts.push((label.to_string(), String::new(), vec![0x6c, 0x1b, 0x01]));
            }
            let (_, part_bim, part_bed) = parts.last_mut().expect("a part was just pushed");
            part_bim.push_str(line);
            part_bim.push('\n');
            let start = 3 + index * row_len;
            part_bed.extend_from_slice(&payload[start..start + row_len]);
        }
        let split_dir = tempdir()?;
        for (label, part_bim, part_bed) in &parts {
            let member = |ext: &str| split_dir.path().join(format!("fixture.chr{label}.{ext}"));
            std::fs::write(member("bim"), part_bim)?;
            std::fs::write(member("fam"), &fam)?;
            std::fs::write(member("bed"), part_bed)?;
        }
        assert!(parts.len() > 20, "the fixture spans many chromosomes");

        let (output, build, records) = infer_records(split_dir.path(), None, false)?;
        assert_eq!(output, split_dir.path().join("sex.tsv"));
        assert_eq!(build, expected_build);
        assert_eq!(records.len(), expected.len());
        for (record, expected) in records.iter().zip(&expected) {
            assert_eq!(record.individual_id, expected.individual_id);
            assert_eq!(record.inference, expected.inference);
            assert_eq!(
                metric_bits(&record.inference),
                metric_bits(&expected.inference)
            );
        }

        // A directory the variant reader can read stays with it.
        let vcf = split_dir.path().join("calls.vcf.gz");
        std::fs::write(&vcf, b"")?;
        assert!(plink_fileset_directory(split_dir.path())?.is_none());
        std::fs::remove_file(&vcf)?;

        std::fs::write(
            split_dir.path().join("fixture.chrY.fam"),
            fam.replacen("I0", "J0", 1),
        )?;
        match infer_records(split_dir.path(), None, false) {
            Err(SexInferenceError::MismatchedFilesetSamples { .. }) => {}
            other => panic!("filesets with different samples must be refused, got {other:?}"),
        }
        Ok(())
    }

    /// PAR rows under a code of their own -- PLINK's `25` and `XY`, plink2's
    /// `PAR1` and `PAR2` -- must infer exactly what the same rows coded `X` do.
    #[test]
    fn par_coded_rows_infer_what_the_rows_coded_x_infer() -> Result<(), Box<dyn std::error::Error>>
    {
        let reference_dir = tempdir()?;
        let (_, expected_build, expected) = infer_records(
            &write_sex_fixture(reference_dir.path(), 64, 3)?,
            None,
            false,
        )?;
        let constants = expected_build.algorithm_constants();
        for par_label in ["25", "XY", "chrXY", "PAR1/PAR2"] {
            let dir = tempdir()?;
            let bed = write_sex_fixture(dir.path(), 64, 3)?;
            let bim_path = bed.with_extension("bim");
            let mut relabelled = String::new();
            for line in std::fs::read_to_string(&bim_path)?.lines() {
                let mut fields: Vec<&str> = line.split('\t').collect();
                let position: u64 = fields[3].parse()?;
                // plink --split-x: everything outside the non-PAR interval moves to the PAR code.
                if fields[0] == "X" && !constants.is_in_x_non_par(position) {
                    fields[0] = match par_label {
                        "PAR1/PAR2" if position < constants.non_par_x.0 => "PAR1",
                        "PAR1/PAR2" => "PAR2",
                        label => label,
                    };
                }
                relabelled.push_str(&fields.join("\t"));
                relabelled.push('\n');
            }
            std::fs::write(&bim_path, relabelled)?;

            let (_, build, records) = infer_records(&bed, None, false)?;
            assert_eq!(build, expected_build, "{par_label}");
            assert_eq!(records.len(), expected.len());
            for (record, expected) in records.iter().zip(&expected) {
                assert_eq!(record.individual_id, expected.individual_id);
                assert_eq!(record.inference, expected.inference, "{par_label}");
                assert_eq!(
                    metric_bits(&record.inference),
                    metric_bits(&expected.inference)
                );
            }
        }
        Ok(())
    }
}
