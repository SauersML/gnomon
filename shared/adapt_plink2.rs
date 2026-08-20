//! A single-file, pure-Rust façade which consumes PLINK 2.0 inputs
//! (`.pgen/.pvar/.psam`) and presents virtual PLINK 1.9 outputs
//! (`.bed/.bim/.fam`) through the same streaming traits your code
//! already uses:
//!   - `.bed` → `ByteRangeSource` (random-access byte ranges)
//!   - `.bim` / `.fam` → `TextSource` (pull-based line iterator)
//!
//! ## Fixed semantics (“best options”, no knobs)
//! - Multiallelic: always split every ALT; never drop. Variant order in
//!   the virtual outputs matches `.pvar`, expanded in ALT order and with
//!   multiallelics deterministically split.
//! - Allele orientation: A1 equals ALT and A2 equals REF (per split
//!   ALT). The virtual `.bim` follows the PLINK 1.9 contract with `cM =`
//!   `0` and synthesised IDs when needed.
//! - Genotype basis: the hard-call track wins wherever it has a call. The
//!   dosage track is consulted only where that call is missing, and a dosage
//!   then becomes the nearest whole allele count within ±0.10, else missing.
//!   A dosage-valued (imputed) fileset therefore reaches consumers as hard
//!   calls, which is a different analysis from the one it looks like. What
//!   that costs is counted as the `.pgen` is read and reported: see
//!   `VirtualPlink19::dosage_coercion_report`, and the stderr warning the
//!   adapter raises on its own once the counts are decisive.
//! - Ploidy: autosomes plus pseudoautosomal regions are diploid; `X`
//!   (non-PAR), `Y`, and `MT` are treated as haploid for males.
//!   Heterozygotes in these contexts are coerced to missing before PLINK
//!   1.9 packing.
//! - `.bed` encoding: exact PLINK 1.9 2-bit codes (`00` hom ALT, `01`
//!   missing, `10` het, `11` hom REF; least-significant bit first within
//!   each byte).
//! - Split IDs: if `ID != "."` → `ID__ALT=<ALT>`; else use
//!   `chr:pos:ref:alt`.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::str;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::files::{
    // Traits
    ByteRangeSource,
    TextSource,
    // Helpers
    open_text_source,
};
/// Bring in your crate-local traits and error type.
/// These are expected to already exist (per your provided infrastructure).
use crate::score::pipeline::PipelineError;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public entrypoints
////////////////////////////////////////////////////////////////////////////////////////////////////

/// A handle which exposes virtual PLINK-1.9 streams backed by PLINK-2.0 inputs.
pub struct VirtualPlink19 {
    /// Random-access virtual `.bed` (PLINK-1.9 bytes).
    pub bed: Arc<dyn ByteRangeSource>,
    build: GenomeBuild,
    /// Reopens the backing `.pvar` so `.bim` rows can be regenerated on demand
    /// instead of held in memory. See `StreamingVirtualBim`.
    pvar: PvarFactory,
    fam_rows: Vec<FamRow>,
    n_samples: usize,
    n_variants: usize,
    /// Shared with the decoder behind `bed`, so the counts a caller reads are
    /// the ones the decode path is writing. `None` for mode 0x01, which *is* a
    /// PLINK 1.9 `.bed`: it carries no dosages, so there is nothing to lose.
    dosage: Option<Arc<DosageCoercionMeter>>,
}

/// Reopens a `.pvar` text stream. Boxed rather than a path so remote (`gs://`,
/// http) sources work identically to local files.
pub type PvarFactory = Arc<dyn Fn() -> Result<Box<dyn TextSource>, PipelineError> + Send + Sync>;

impl VirtualPlink19 {
    /// Returns the inferred genome build (`"GRCh37"` or `"GRCh38"`).
    pub fn inferred_genome_build(&self) -> &'static str {
        match self.build {
            GenomeBuild::Grch37 => "GRCh37",
            GenomeBuild::Grch38 => "GRCh38",
        }
    }

    pub fn bed_source(&self) -> Arc<dyn ByteRangeSource> {
        Arc::clone(&self.bed)
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn n_variants(&self) -> usize {
        self.n_variants
    }

    /// What presenting this fileset as PLINK 1.9 hard calls has cost so far.
    ///
    /// The counts accumulate as the virtual `.bed` is read, so exact totals
    /// mean asking after the final pass; before that they describe the records
    /// visited. A caller is not obliged to ask — the adapter warns on stderr by
    /// itself, precisely because a consumer that never asks must still not be
    /// left believing it fitted on dosages.
    pub fn dosage_coercion_report(&self) -> DosageCoercionReport {
        self.dosage
            .as_ref()
            .map(|meter| meter.report())
            .unwrap_or_default()
    }

    /// A fresh forward pass over the virtual `.bim`.
    ///
    /// Each call reopens the `.pvar` and regenerates rows as it goes; nothing
    /// is retained between calls.
    pub fn bim_source(&self) -> Result<Box<dyn TextSource>, PipelineError> {
        Ok(Box::new(StreamingVirtualBim::new(
            (self.pvar)()?,
            Some(self.n_variants as u64),
        )))
    }

    pub fn fam_source(&self) -> Box<dyn TextSource> {
        Box::new(VirtualFam::from_rows(self.fam_rows.clone()))
    }
}

/// A streaming virtual `.bim` over a `.pvar`, without touching the `.pgen`.
///
/// The transform is purely textual — split multiallelic ALTs, synthesise IDs,
/// emit A1=ALT/A2=REF — so consumers that only need variant metadata (variant
/// reconciliation, coverage QC) can avoid opening the genotype table at all.
pub fn open_virtual_bim(pvar_path: &Path) -> Result<Box<dyn TextSource>, PipelineError> {
    Ok(Box::new(StreamingVirtualBim::new(
        open_text_source(pvar_path)?,
        None,
    )))
}

/// A virtual `.fam` over a `.psam`.
///
/// Sample counts are bounded by cohort size (hundreds of thousands), not
/// variant count, so unlike the `.bim` this is materialized.
pub fn open_virtual_fam(psam_path: &Path) -> Result<Box<dyn TextSource>, PipelineError> {
    let mut psam = open_text_source(psam_path)?;
    let info = PsamInfo::from_psam(&mut *psam)?;
    Ok(Box::new(VirtualFam::from_rows(info.fam_rows)))
}

/// Open from filesystem paths. `.pvar` and `.psam` are opened with your existing
/// `open_text_source`. The `.pgen` is opened as a local file here.
/// If you need remote `.pgen` support, use `open_virtual_plink19_from_sources`
/// and pass an appropriate `ByteRangeSource` for `.pgen`.
pub fn open_virtual_plink19_from_paths(
    pgen_path: &Path,
    pvar_path: &Path,
    psam_path: &Path,
) -> Result<VirtualPlink19, PipelineError> {
    let mut psam_for_plan = open_text_source(psam_path)?;
    let pgen = Arc::new(LocalFileByteRangeSource::open(pgen_path)?);
    let pvar_path = pvar_path.to_path_buf();
    let pvar: PvarFactory = Arc::new(move || open_text_source(&pvar_path));

    open_virtual_plink19_from_sources(pgen, pvar, &mut *psam_for_plan)
}

/// Open from caller-provided sources. Callers may pass a custom/remote-capable
/// `ByteRangeSource` for `.pgen` and a `TextSource` for `.psam`.
///
/// `.pvar` is supplied as a *factory* rather than a stream: it is read once to
/// build the variant plan, and reopened whenever the virtual `.bim` is walked,
/// so its rows never have to be held in memory.
pub fn open_virtual_plink19_from_sources(
    pgen: Arc<dyn ByteRangeSource>,
    pvar: PvarFactory,
    psam_for_plan: &mut dyn TextSource,
) -> Result<VirtualPlink19, PipelineError> {
    let header = PgenHeader::parse(&*pgen)?;
    let psam_info = PsamInfo::from_psam(psam_for_plan)?;
    let plan = VariantPlan::from_pvar(&mut *pvar()?)?;
    let inferred_build = plan.inferred_build();

    if header.m_variants != 0 && header.m_variants as usize != plan.in_variants {
        return Err(PipelineError::Io(format!(
            "Variant count mismatch: .pgen header has {}, .pvar expands to {}",
            header.m_variants, plan.in_variants
        )));
    }

    if header.n_samples != 0 && header.n_samples as usize != psam_info.n_samples {
        return Err(PipelineError::Io(format!(
            "Sample-count mismatch: .pgen header has {}, .psam has {}",
            header.n_samples, psam_info.n_samples
        )));
    }

    let fam_rows = psam_info.fam_rows.clone();

    let (bed_source, dosage): (Arc<dyn ByteRangeSource>, Option<Arc<DosageCoercionMeter>>) =
        match header.mode {
            PgenMode::Bed => {
                if plan.out_variants != plan.in_variants {
                    return Err(PipelineError::Io(
                        "Mode 0x01 (.bed) cannot expand multiallelic variants; re-encode the input to mode 0x10/0x11"
                            .into(),
                    ));
                }
                let bed: Arc<dyn ByteRangeSource> = pgen.clone();
                (bed, None)
            }
            _ => {
                let decoder = PgenDecoder::new(
                    pgen.clone(),
                    header,
                    psam_info.n_samples,
                    plan.in_variants,
                    plan.alts_per_in.clone(),
                )?;
                // Taken before the decoder is moved into the block source: the
                // decoder is the only writer, and this handle the only reader.
                let meter = Arc::clone(&decoder.dosage_meter);
                let sex_by_sample_arc: Arc<[u8]> =
                    Arc::from(psam_info.sex_by_sample.clone().into_boxed_slice());
                let bed: Arc<dyn ByteRangeSource> = Arc::new(VirtualBed::new(
                    decoder,
                    plan.clone(),
                    psam_info.n_samples,
                    sex_by_sample_arc,
                )?);
                (bed, Some(meter))
            }
        };

    Ok(VirtualPlink19 {
        bed: bed_source,
        build: inferred_build,
        pvar,
        fam_rows,
        n_samples: psam_info.n_samples,
        n_variants: plan.out_variants,
        dosage,
    })
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PSAM → FAM (header semantics + row mapping)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct PsamInfo {
    n_samples: usize,
    sex_by_sample: Vec<u8>,
    fam_rows: Vec<FamRow>,
}

#[derive(Clone, Default)]
struct PsamColumns {
    fid_idx: Option<usize>,
    iid_idx: Option<usize>,
    pat_idx: Option<usize>,
    mat_idx: Option<usize>,
    sex_idx: Option<usize>,
    pheno_idx: Option<usize>,
    pheno1_idx: Option<usize>,
    sid_idx: Option<usize>,
}

#[derive(Clone, Default)]
struct FamRow {
    fid: String,
    iid: String,
    pat: String,
    mat: String,
    sex: String,
    phe: String,
}

impl FamRow {
    fn as_line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.fid, self.iid, self.pat, self.mat, self.sex, self.phe
        )
    }
}

impl PsamInfo {
    fn from_psam(source: &mut dyn TextSource) -> Result<Self, PipelineError> {
        let mut header_tokens: Option<Vec<String>> = None;
        let mut columns: Option<PsamColumns> = None;
        let mut sex_by_sample: Vec<u8> = Vec::new();
        let mut fam_rows: Vec<FamRow> = Vec::new();

        while let Some(line) = source.next_line()? {
            let s = str::from_utf8(line)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .psam: {e}")))?;

            if s.starts_with('#') {
                let cols = s.trim_start_matches('#').trim();
                if !cols.is_empty() && !cols.starts_with('#') {
                    header_tokens = Some(cols.split_whitespace().map(|t| t.to_string()).collect());
                    columns = None; // the last header wins
                }
                continue;
            }

            let trimmed = s.trim();
            if trimmed.is_empty() {
                continue;
            }

            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }

            if columns.is_none() {
                columns = Some(match header_tokens.as_ref() {
                    Some(tokens) => PsamColumns::from_header(tokens),
                    None => PsamColumns::from_headerless(fields.len()),
                }?);
            }

            let cols = columns.as_ref().unwrap();
            let fam_row = FamRow::from_fields(&fields, cols);
            if fam_row.iid == "0" {
                return Err(PipelineError::Io(
                    "IID must not be '0' (PSAM/FAM contract)".into(),
                ));
            }
            let sex_code = parse_sex_token(&fam_row.sex);
            sex_by_sample.push(sex_code);
            fam_rows.push(fam_row);
        }

        if columns.is_none() {
            let tokens = header_tokens
                .ok_or_else(|| PipelineError::Io("Missing .psam header (#FID/#IID…)".into()))?;
            PsamColumns::from_header(&tokens)?;
        }

        Ok(Self {
            n_samples: sex_by_sample.len(),
            sex_by_sample,
            fam_rows,
        })
    }
}

fn parse_sex_token(token: &str) -> u8 {
    match token.trim() {
        "1" => 1,
        "2" => 2,
        "M" | "m" => 1,
        "F" | "f" => 2,
        t if t.eq_ignore_ascii_case("male") => 1,
        t if t.eq_ignore_ascii_case("female") => 2,
        t if t.eq_ignore_ascii_case("unknown") => 0,
        t if t.eq_ignore_ascii_case("unk") => 0,
        t if t.eq_ignore_ascii_case("u") => 0,
        "0" | "NA" | "na" | "Na" | "nA" | "." | "nan" | "NaN" | "NAN" => 0,
        _ => 0,
    }
}

impl PsamColumns {
    fn from_header(cols: &[String]) -> Result<Self, PipelineError> {
        let mut out = PsamColumns::default();
        for (i, c) in cols.iter().enumerate() {
            match c.to_ascii_uppercase().as_str() {
                "FID" => out.fid_idx = Some(i),
                "IID" => out.iid_idx = Some(i),
                "PAT" => out.pat_idx = Some(i),
                "MAT" => out.mat_idx = Some(i),
                "SEX" => out.sex_idx = Some(i),
                "PHENO" | "PHENOTYPE" => out.pheno_idx = Some(i),
                "PHENO1" => out.pheno1_idx = Some(i),
                "SID" => out.sid_idx = Some(i),
                _ => {}
            }
        }
        if out.iid_idx.is_none() && out.fid_idx.is_none() {
            return Err(PipelineError::Io(
                "Invalid .psam header: need #FID or #IID".to_string(),
            ));
        }
        Ok(out)
    }

    fn from_headerless(field_count: usize) -> Result<Self, PipelineError> {
        if field_count >= 6 {
            Ok(PsamColumns {
                fid_idx: Some(0),
                iid_idx: Some(1),
                pat_idx: Some(2),
                mat_idx: Some(3),
                sex_idx: Some(4),
                pheno_idx: None,
                pheno1_idx: Some(5),
                sid_idx: None,
            })
        } else if field_count == 5 {
            Ok(PsamColumns {
                fid_idx: Some(0),
                iid_idx: Some(1),
                pat_idx: Some(2),
                mat_idx: Some(3),
                sex_idx: Some(4),
                pheno_idx: None,
                pheno1_idx: None,
                sid_idx: None,
            })
        } else {
            Err(PipelineError::Io(
                "Headerless .psam requires 5 or 6 columns".to_string(),
            ))
        }
    }
}

fn coerce_pheno_token(tok: &str) -> String {
    let t = tok.trim();
    if t.is_empty() {
        return "-9".to_string();
    }
    if t.eq_ignore_ascii_case("na")
        || t.eq_ignore_ascii_case("nan")
        || t == "."
        || t.eq_ignore_ascii_case("none")
    {
        return "-9".to_string();
    }
    if t == "0" {
        return "-9".to_string();
    }
    if t.parse::<f64>().is_ok() {
        t.to_string()
    } else {
        "-9".to_string()
    }
}

impl FamRow {
    fn from_fields(fields: &[&str], cols: &PsamColumns) -> FamRow {
        let get_clean = |idx: Option<usize>| -> Option<String> {
            idx.and_then(|i| fields.get(i))
                .map(|s| s.trim())
                .filter(|t| !t.is_empty())
                .map(|t| t.to_string())
        };

        let mut iid = get_clean(cols.iid_idx)
            .or_else(|| get_clean(cols.sid_idx))
            .or_else(|| get_clean(cols.fid_idx))
            .unwrap_or_else(|| "0".to_string());
        if iid == "0" {
            iid = get_clean(cols.sid_idx).unwrap_or_else(|| "0".to_string());
        }

        let fid = get_clean(cols.fid_idx).unwrap_or_else(|| iid.clone());
        let pat = get_clean(cols.pat_idx).unwrap_or_else(|| "0".to_string());
        let mat = get_clean(cols.mat_idx).unwrap_or_else(|| "0".to_string());
        let sex = get_clean(cols.sex_idx).unwrap_or_else(|| "0".to_string());
        // PHENO1 takes precedence when both PHENO and PHENO1 are present.
        let phe = cols
            .pheno1_idx
            .and_then(|i| fields.get(i))
            .or_else(|| cols.pheno_idx.and_then(|i| fields.get(i)))
            .map(|s| coerce_pheno_token(s))
            .unwrap_or_else(|| "-9".to_string());
        FamRow {
            fid,
            iid,
            pat,
            mat,
            sex,
            phe,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PVAR → VariantPlan (always split) + BIM streaming transform
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping from virtual BED variant index (post-split) to PGEN record index
/// and the ALT ordinal within that record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HaploidyKind {
    Diploid,
    HaploidMales,
    HaploidAll,
    HaploidMalesFemalesMissing,
}

#[derive(Clone)]
struct VariantPlan {
    /// Total input variants before splitting (to sanity check decoder bounds).
    in_variants: usize,
    /// Total emitted variants after splitting.
    out_variants: usize,
    /// Dense mapping: out_idx → (in_idx, alt_ordinal_1based).
    out_to_in: Vec<(u32, u16)>,
    /// Per-output variant haploidy behaviour.
    haploidy: Vec<HaploidyKind>,
    /// ALT allele count per input variant.
    alts_per_in: Vec<u16>,
    build: GenomeBuild,
}

#[derive(Clone)]
struct VariantRangeEntry {
    chrom: String,
    pos: u64,
    out_start: usize,
    out_end: usize,
}

#[derive(Clone, Copy)]
struct PvarCols {
    chrom: usize,
    id: usize,
    pos: usize,
    refa: usize,
    alt: usize,
}

impl PvarCols {
    fn from_header_line(line: &str) -> Result<Self, PipelineError> {
        let body = line.trim_start_matches('#').trim();
        let tokens: Vec<&str> = body.split_whitespace().collect();
        if tokens.is_empty() {
            return Err(PipelineError::Io("Empty .pvar header line".to_string()));
        }
        Self::from_tokens(&tokens)
    }

    fn from_tokens(tokens: &[&str]) -> Result<Self, PipelineError> {
        let mut chrom = None;
        let mut id = None;
        let mut pos = None;
        let mut refa = None;
        let mut alt = None;

        for (i, token) in tokens.iter().enumerate() {
            let upper = token.trim().trim_start_matches('#').to_ascii_uppercase();
            match upper.as_str() {
                "CHROM" => chrom = Some(i),
                "ID" => id = Some(i),
                "POS" | "BP" => pos = Some(i),
                "REF" => refa = Some(i),
                "ALT" => alt = Some(i),
                _ => {}
            }
        }

        let chrom = chrom.ok_or_else(|| PipelineError::Io(".pvar header missing CHROM".into()))?;
        let id = id.ok_or_else(|| PipelineError::Io(".pvar header missing ID".into()))?;
        let pos = pos.ok_or_else(|| PipelineError::Io(".pvar header missing POS".into()))?;
        let refa = refa.ok_or_else(|| PipelineError::Io(".pvar header missing REF".into()))?;
        let alt = alt.ok_or_else(|| PipelineError::Io(".pvar header missing ALT".into()))?;

        Ok(PvarCols {
            chrom,
            id,
            pos,
            refa,
            alt,
        })
    }

    fn from_headerless(field_count: usize) -> Result<Self, PipelineError> {
        if field_count >= 6 {
            Ok(PvarCols {
                chrom: 0,
                id: 1,
                pos: 3,
                refa: 4,
                alt: 5,
            })
        } else if field_count == 5 {
            Ok(PvarCols {
                chrom: 0,
                id: 1,
                pos: 2,
                refa: 3,
                alt: 4,
            })
        } else {
            Err(PipelineError::Io(
                "Headerless .pvar requires ≥5 columns".to_string(),
            ))
        }
    }
}

impl VariantPlan {
    fn from_pvar(pvar: &mut dyn TextSource) -> Result<Self, PipelineError> {
        let mut out_to_in: Vec<(u32, u16)> = Vec::with_capacity(1 << 20);
        let mut haploidy: Vec<HaploidyKind> = Vec::with_capacity(1 << 20);
        let mut per_variant: Vec<VariantRangeEntry> = Vec::with_capacity(1 << 16);
        let mut alts_per_in: Vec<u16> = Vec::with_capacity(1 << 16);
        let mut header_cols: Option<PvarCols> = None;
        let mut in_idx: u32 = 0;
        let mut in_variants: usize = 0;
        let mut max_x_pos: u64 = 0;
        let mut saw_x_par37_only = false;
        let mut saw_x_par38_only = false;
        let mut sorted_positions = PvarPositionSortState::default();

        while let Some(line) = pvar.next_line()? {
            let s = str::from_utf8(line)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .pvar: {e}")))?;
            let trimmed = s.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with("##") {
                continue;
            }
            if trimmed.starts_with('#') {
                header_cols = Some(PvarCols::from_header_line(trimmed)?);
                continue;
            }

            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }
            let cols = if let Some(cols) = header_cols {
                cols
            } else {
                let derived = PvarCols::from_headerless(fields.len())?;
                header_cols = Some(derived);
                derived
            };

            let chrom_raw = *fields
                .get(cols.chrom)
                .ok_or_else(|| ioerr(".pvar missing CHROM column"))?;
            let pos_raw = *fields
                .get(cols.pos)
                .ok_or_else(|| ioerr(".pvar missing POS column"))?;
            let id_raw = *fields
                .get(cols.id)
                .ok_or_else(|| ioerr(".pvar missing ID column"))?;
            // REF is validated by presence here; the value itself is only
            // needed when the virtual .bim rows are streamed.
            fields
                .get(cols.refa)
                .ok_or_else(|| ioerr(".pvar missing REF column"))?;
            let alt_raw = *fields
                .get(cols.alt)
                .ok_or_else(|| ioerr(".pvar missing ALT column"))?;

            let chrom = normalize_chrom(chrom_raw);
            let pos = pos_raw
                .parse::<u64>()
                .map_err(|_| ioerr("Invalid POS in .pvar (expected integer)"))?;
            if pos == 0 {
                return Err(ioerr(".pvar POS must be positive"));
            }
            sorted_positions.observe(&chrom, pos, in_variants + 1)?;
            if chrom == "X" {
                max_x_pos = max_x_pos.max(pos);
                let in37 = in_any_range(pos, GRCH37_X_PAR);
                let in38 = in_any_range(pos, GRCH38_X_PAR);
                if in37 && !in38 {
                    saw_x_par37_only = true;
                }
                if in38 && !in37 {
                    saw_x_par38_only = true;
                }
            }

            let alts: Vec<&str> = alt_raw
                .split(',')
                .map(|a| a.trim())
                .filter(|a| !a.is_empty() && *a != ".")
                .collect();

            for alt in &alts {
                if is_symbolic_alt(alt) {
                    return Err(PipelineError::Io(format!(
                        "Symbolic ALT '{}' unsupported for variant {}:{} (ID {})",
                        alt, chrom, pos, id_raw
                    )));
                }
            }

            let out_start = out_to_in.len();
            for alt_ord in 1..=alts.len() as u16 {
                out_to_in.push((in_idx, alt_ord));
                haploidy.push(HaploidyKind::Diploid);
            }
            let out_end = out_to_in.len();
            // Autosomes are diploid under every build, so there is nothing to
            // revisit once the build is known. Only the sex chromosomes and MT
            // need their coordinates retained -- which keeps this buffer empty
            // for the 22 autosomal filesets rather than holding a String per
            // variant.
            if !is_autosome(&chrom) {
                per_variant.push(VariantRangeEntry {
                    chrom,
                    pos,
                    out_start,
                    out_end,
                });
            }

            alts_per_in.push(alts.len() as u16);
            in_idx += 1;
            in_variants += 1;
        }

        if header_cols.is_none() {
            return Err(PipelineError::Io(
                "Missing .pvar header or inferable columns".to_string(),
            ));
        }

        if saw_x_par37_only && saw_x_par38_only {
            return Err(PipelineError::Io(
                "Encountered X PAR loci matching both GRCh37-only and GRCh38-only ranges; cannot infer a single build"
                    .to_string(),
            ));
        }

        let build = if saw_x_par37_only {
            GenomeBuild::Grch37
        } else if saw_x_par38_only {
            GenomeBuild::Grch38
        } else {
            infer_genome_build(max_x_pos)
        };
        for entry in per_variant {
            let hap = haploidy_for_variant(&entry.chrom, entry.pos, build);
            for idx in entry.out_start..entry.out_end {
                if let Some(slot) = haploidy.get_mut(idx) {
                    *slot = hap;
                }
            }
        }

        Ok(Self {
            in_variants,
            out_variants: out_to_in.len(),
            out_to_in,
            haploidy,
            alts_per_in,
            build,
        })
    }

    #[inline]
    fn mapping(&self, out_idx: usize) -> Option<(u32, u16)> {
        self.out_to_in.get(out_idx).copied()
    }

    #[inline]
    fn haploidy_of(&self, out_idx: usize) -> Option<HaploidyKind> {
        self.haploidy.get(out_idx).copied()
    }

    #[inline]
    fn alt_count_of_in(&self, in_idx: u32) -> u16 {
        self.alts_per_in.get(in_idx as usize).copied().unwrap_or(0)
    }

    #[inline]
    fn inferred_build(&self) -> GenomeBuild {
        self.build
    }
}

fn ioerr(msg: &str) -> PipelineError {
    PipelineError::Io(msg.to_string())
}

#[derive(Default)]
struct PvarPositionSortState {
    current_chromosome: String,
    current_position: Option<u64>,
    previous_positions_by_chrom: HashMap<String, u64>,
}

impl PvarPositionSortState {
    #[inline]
    fn observe(
        &mut self,
        chromosome: &str,
        position: u64,
        record: usize,
    ) -> Result<(), PipelineError> {
        if let Some(previous) = self.current_position
            && self.current_chromosome == chromosome
        {
            if position < previous {
                return Err(PipelineError::Io(format!(
                    ".pvar variants are not position-sorted within chromosome {chromosome}: record {record} has position {position} after position {previous}"
                )));
            }
            self.current_position = Some(position);
            return Ok(());
        }

        if let Some(previous) = self.current_position {
            self.previous_positions_by_chrom
                .insert(self.current_chromosome.clone(), previous);
        }

        if let Some(&previous) = self.previous_positions_by_chrom.get(chromosome)
            && position < previous
        {
            return Err(PipelineError::Io(format!(
                ".pvar variants are not position-sorted within chromosome {chromosome}: record {record} has position {position} after position {previous}"
            )));
        }

        self.current_chromosome.clear();
        self.current_chromosome.push_str(chromosome);
        self.current_position = Some(position);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenomeBuild {
    Grch37,
    Grch38,
}

fn infer_genome_build(max_x_pos: u64) -> GenomeBuild {
    const GRCH38_THRESHOLD: u64 = 155_700_000;
    const GRCH37_THRESHOLD: u64 = 154_900_000;
    if max_x_pos >= GRCH38_THRESHOLD {
        GenomeBuild::Grch38
    } else if max_x_pos >= GRCH37_THRESHOLD {
        GenomeBuild::Grch37
    } else {
        GenomeBuild::Grch38
    }
}

fn normalize_chrom(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut body = trimmed;
    if trimmed.len() >= 3 && trimmed[..3].eq_ignore_ascii_case("chr") {
        body = &trimmed[3..];
    }
    let upper = body.to_ascii_uppercase();
    match upper.as_str() {
        "M" => "MT".to_string(),
        _ => upper,
    }
}

fn is_symbolic_alt(alt: &str) -> bool {
    let trimmed = alt.trim();
    if trimmed == "*" {
        return true;
    }
    if trimmed.starts_with('<') && trimmed.ends_with('>') {
        return true;
    }
    trimmed.contains('[') || trimmed.contains(']')
}

const GRCH37_X_PAR: &[(u64, u64)] = &[(60_001, 2_699_520), (154_931_044, 155_260_560)];
const GRCH38_X_PAR: &[(u64, u64)] = &[(10_001, 2_781_479), (155_701_383, 156_030_895)];

/// True for a normalized chromosome label that is a plain numbered autosome.
///
/// Autosomes are diploid under every genome build, so their ploidy never
/// depends on which build we infer.
fn is_autosome(chrom: &str) -> bool {
    !chrom.is_empty() && chrom.bytes().all(|b| b.is_ascii_digit())
}

fn in_any_range(pos: u64, ranges: &[(u64, u64)]) -> bool {
    ranges
        .iter()
        .any(|(start, end)| pos >= *start && pos <= *end)
}

fn haploidy_for_variant(chrom: &str, pos: u64, build: GenomeBuild) -> HaploidyKind {
    match chrom {
        "X" => {
            let ranges = match build {
                GenomeBuild::Grch37 => GRCH37_X_PAR,
                GenomeBuild::Grch38 => GRCH38_X_PAR,
            };
            if in_any_range(pos, ranges) {
                HaploidyKind::Diploid
            } else {
                HaploidyKind::HaploidMales
            }
        }
        "Y" => HaploidyKind::HaploidMalesFemalesMissing,
        "MT" => HaploidyKind::HaploidAll,
        _ => HaploidyKind::Diploid,
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .bim (TextSource): split multiallelic, A1=ALT, A2=REF, cM=0, stable IDs
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Renders one virtual `.bim` row for a single (variant, ALT) pair.
///
/// PLINK 1.9 contract: `cM` is always 0, A1 is the ALT allele and A2 is REF.
///
/// ID selection matters for compatibility, not just cosmetics. A biallelic site
/// keeps its `.pvar` ID verbatim, so the row is byte-identical to the one the
/// equivalent `.bed`/`.bim` fileset would carry — a cohort scored via PGEN and
/// via PLINK 1.9 then presents the same variant identifiers to any downstream
/// coverage QC. Only a site that actually splits into several rows needs a
/// disambiguating ID, since the raw one would no longer be unique:
///   - `<ID>__ALT=<ALT>` when the `.pvar` carries an ID
///   - `chr:pos:ref:alt` when it does not (`.`)
///
/// gnomon's own reconciliation keys on (chrom, pos, A1, A2) rather than ID, so
/// this choice does not affect which variants gnomon matches.
fn format_bim_row(chrom: &str, id: &str, pos: u64, refa: &str, alt: &str, split: bool) -> String {
    let has_id = id != "." && !id.is_empty();
    let id_out = match (has_id, split) {
        (true, false) => id.to_string(),
        (true, true) => format!("{id}__ALT={alt}"),
        (false, _) => format!("{chrom}:{pos}:{refa}:{alt}"),
    };
    format!("{chrom}\t{id_out}\t0\t{pos}\t{alt}\t{refa}")
}

/// Streaming virtual `.bim`: re-reads the `.pvar` and emits the split rows on
/// the fly.
///
/// Materializing the rows instead would cost roughly 140 bytes per output
/// variant — about 1.2 GB for a WGS chr1 at 8.45M variants, and ~10 GB if a
/// genome-wide set of per-chromosome filesets were opened at once. Nothing
/// downstream needs random access to the rows, only a single forward pass, so
/// they are generated on demand and never retained.
struct StreamingVirtualBim {
    pvar: Box<dyn TextSource>,
    cols: Option<PvarCols>,
    /// Rows pending for the current `.pvar` line (one per ALT), reversed so
    /// `pop` yields them in ALT order.
    pending: Vec<String>,
    carry: Option<Box<[u8]>>,
    total: Option<u64>,
}

impl StreamingVirtualBim {
    fn new(pvar: Box<dyn TextSource>, total: Option<u64>) -> Self {
        Self {
            pvar,
            cols: None,
            pending: Vec::new(),
            carry: None,
            total,
        }
    }
}

impl TextSource for StreamingVirtualBim {
    fn len(&self) -> Option<u64> {
        self.total
    }

    fn next_line(&mut self) -> Result<Option<&[u8]>, PipelineError> {
        loop {
            if let Some(row) = self.pending.pop() {
                self.carry = Some(row.into_bytes().into_boxed_slice());
                return Ok(self.carry.as_deref());
            }

            let Some(line) = self.pvar.next_line()? else {
                return Ok(None);
            };
            let s = str::from_utf8(line)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .pvar: {e}")))?;
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed.starts_with("##") {
                continue;
            }
            if trimmed.starts_with('#') {
                self.cols = Some(PvarCols::from_header_line(trimmed)?);
                continue;
            }

            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }
            let cols = match self.cols {
                Some(cols) => cols,
                None => {
                    let derived = PvarCols::from_headerless(fields.len())?;
                    self.cols = Some(derived);
                    derived
                }
            };

            let chrom = normalize_chrom(
                fields
                    .get(cols.chrom)
                    .copied()
                    .ok_or_else(|| ioerr(".pvar missing CHROM column"))?,
            );
            let pos = fields
                .get(cols.pos)
                .copied()
                .ok_or_else(|| ioerr(".pvar missing POS column"))?
                .parse::<u64>()
                .map_err(|_| ioerr("Invalid POS in .pvar (expected integer)"))?;
            let id = fields
                .get(cols.id)
                .copied()
                .ok_or_else(|| ioerr(".pvar missing ID column"))?;
            let refa = fields
                .get(cols.refa)
                .copied()
                .ok_or_else(|| ioerr(".pvar missing REF column"))?;
            let alt_raw = fields
                .get(cols.alt)
                .copied()
                .ok_or_else(|| ioerr(".pvar missing ALT column"))?;

            let alts: Vec<&str> = alt_raw
                .split(',')
                .map(str::trim)
                .filter(|a| !a.is_empty() && *a != ".")
                .collect();
            let split = alts.len() > 1;
            // Reversed, so `pop` above walks ALTs in order — matching the
            // variant order the plan assigned during the indexing pass.
            self.pending.extend(
                alts.into_iter()
                    .map(|alt| format_bim_row(&chrom, id, pos, refa, alt, split))
                    .rev(),
            );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .fam (TextSource): map .psam to .fam with fixed defaults
////////////////////////////////////////////////////////////////////////////////////////////////////

struct VirtualFam {
    rows: Vec<FamRow>,
    next_idx: usize,
    carry: Option<Box<[u8]>>,
}

impl VirtualFam {
    fn from_rows(rows: Vec<FamRow>) -> Self {
        Self {
            rows,
            next_idx: 0,
            carry: None,
        }
    }
}

impl TextSource for VirtualFam {
    fn len(&self) -> Option<u64> {
        Some(self.rows.len() as u64)
    }

    fn next_line(&mut self) -> Result<Option<&[u8]>, PipelineError> {
        if self.next_idx >= self.rows.len() {
            return Ok(None);
        }
        let line = self.rows[self.next_idx].as_line();
        self.next_idx += 1;
        self.carry = Some(line.into_bytes().into_boxed_slice());
        Ok(self.carry.as_deref())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .bed (ByteRangeSource): fixed 3-byte header + B bytes per variant
////////////////////////////////////////////////////////////////////////////////////////////////////

const BED_MAGIC_0: u8 = 0x6c;
const BED_MAGIC_1: u8 = 0x1b;
const BED_MODE_SNP_MAJOR: u8 = 0x01;

#[derive(Clone)]
struct VirtualBed {
    inner: Arc<Mutex<PgenDecoder>>, // guarded decoder (seekable + scratch buffers)
    plan: VariantPlan,
    n_samples: usize,
    block_bytes: usize, // ceil(n_samples / 4)
    // small LRU of packed blocks by out-variant index
    cache: Arc<Mutex<BlockCache>>,
    sex_by_sample: Arc<[u8]>,
}

impl VirtualBed {
    fn new(
        decoder: PgenDecoder,
        plan: VariantPlan,
        n_samples: usize,
        sex_by_sample: Arc<[u8]>,
    ) -> Result<Self, PipelineError> {
        if sex_by_sample.len() != n_samples {
            return Err(PipelineError::Compute(
                "SEX column count mismatch with sample count".into(),
            ));
        }
        let block_bytes = n_samples.div_ceil(4);
        Ok(Self {
            inner: Arc::new(Mutex::new(decoder)),
            plan,
            n_samples,
            block_bytes,
            cache: Arc::new(Mutex::new(BlockCache::new(256))),
            sex_by_sample,
        })
    }

    #[inline]
    fn total_len(&self) -> u64 {
        3 + (self.plan.out_variants as u64) * (self.block_bytes as u64)
    }

    /// Pack 0/1/2/255 hard-calls (A1 dosage) into PLINK 1.9 2-bit codes.
    /// Codes (LSB-first per 2-bit field):
    ///   00 hom-A1 (A1 dosage 2)
    ///   10 het     (A1 dosage 1)
    ///   11 hom-A2 (A1 dosage 0)
    ///   01 missing
    fn pack_to_block(dst: &mut [u8], hardcalls: &[u8]) {
        debug_assert_eq!(dst.len(), hardcalls.len().div_ceil(4));
        for chunk_i in 0..dst.len() {
            let base = chunk_i * 4;
            let mut byte = 0u8;
            for j in 0..4 {
                let idx = base + j;
                let code = if idx < hardcalls.len() {
                    match hardcalls[idx] {
                        2 => 0b00,
                        1 => 0b10,
                        0 => 0b11,
                        _ => 0b01, // 255 or anything else → missing
                    }
                } else {
                    0b01
                };
                byte |= code << (2 * j);
            }
            dst[chunk_i] = byte;
        }
    }
}

fn enforce_haploidy(hardcalls: &mut [u8], sex_by_sample: &[u8], kind: HaploidyKind) {
    match kind {
        HaploidyKind::Diploid => {}
        HaploidyKind::HaploidAll => {
            for val in hardcalls.iter_mut() {
                if *val == 1 {
                    *val = 255;
                }
            }
        }
        HaploidyKind::HaploidMales => {
            let n = hardcalls.len().min(sex_by_sample.len());
            for i in 0..n {
                if sex_by_sample[i] == 1 && hardcalls[i] == 1 {
                    hardcalls[i] = 255;
                }
            }
        }
        HaploidyKind::HaploidMalesFemalesMissing => {
            let n = hardcalls.len().min(sex_by_sample.len());
            for i in 0..n {
                let sex = sex_by_sample[i];
                if sex == 2 {
                    hardcalls[i] = 255;
                    continue;
                }
                if hardcalls[i] == 1 {
                    hardcalls[i] = 255;
                }
            }
        }
    }
}

fn fill_sample_ploidy(
    buf: &mut Vec<u8>,
    kind: HaploidyKind,
    sex_by_sample: &[u8],
    n_samples: usize,
) {
    buf.clear();
    buf.resize(n_samples, 2);
    match kind {
        HaploidyKind::Diploid => {}
        HaploidyKind::HaploidAll => {
            for v in buf.iter_mut() {
                *v = 1;
            }
        }
        HaploidyKind::HaploidMales => {
            let limit = sex_by_sample.len().min(n_samples);
            for i in 0..limit {
                if sex_by_sample[i] == 1 {
                    buf[i] = 1;
                }
            }
        }
        HaploidyKind::HaploidMalesFemalesMissing => {
            let limit = sex_by_sample.len().min(n_samples);
            for i in 0..limit {
                match sex_by_sample[i] {
                    1 => buf[i] = 1,
                    2 => buf[i] = 0,
                    _ => {}
                }
            }
        }
    }
}

impl ByteRangeSource for VirtualBed {
    fn len(&self) -> u64 {
        self.total_len()
    }

    fn read_at(&self, mut offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        if dst.is_empty() {
            return Ok(());
        }

        let total = self.total_len();
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| ioerr("Overflow in read_at range"))?;
        if end > total {
            return Err(ioerr("Attempted to read past end of virtual .bed"));
        }

        let mut written = 0usize;
        let mut sample_ploidy_buf: Vec<u8> = Vec::new();
        let mut hard_buf: Vec<u8> = Vec::new();

        // 1) Serve the 3-byte header if requested.
        if offset < 3 {
            let hdr = [BED_MAGIC_0, BED_MAGIC_1, BED_MODE_SNP_MAJOR];
            while offset < 3 && written < dst.len() {
                dst[written] = hdr[offset as usize];
                offset += 1;
                written += 1;
            }
            if written == dst.len() {
                return Ok(());
            }
        }

        // 2) Serve the body: contiguous blocks of size self.block_bytes per variant.
        let body_off = offset - 3;
        let mut out_idx = (body_off / (self.block_bytes as u64)) as usize;
        let mut within_block = (body_off % (self.block_bytes as u64)) as usize;

        let mut decoder = self.inner.lock().unwrap();

        while written < dst.len() {
            if out_idx >= self.plan.out_variants {
                break;
            }

            // Copy bytes from this block.
            let remaining_in_block = self.block_bytes - within_block;
            let remaining_in_dst = dst.len() - written;
            let to_copy = remaining_in_block.min(remaining_in_dst);

            // Fetch or produce the packed block for this out-variant.
            let mut cache_hit = false;
            {
                let cache = self.cache.lock().unwrap();
                if let Some(buf) = cache.get(out_idx) {
                    let start = within_block;
                    let end = start + to_copy;
                    dst[written..written + to_copy].copy_from_slice(&buf[start..end]);
                    cache_hit = true;
                }
            }
            if !cache_hit {
                // Decode hard-calls for this (in_idx, alt_ord) into a scratch buffer.
                let (in_idx, alt_ord) = self
                    .plan
                    .mapping(out_idx)
                    .ok_or_else(|| ioerr("VariantPlan mapping out of bounds"))?;
                let alt_count = self.plan.alt_count_of_in(in_idx);
                if alt_count == 0 {
                    return Err(PipelineError::Io(format!(
                        "ALT count missing for variant {} in .pvar plan",
                        in_idx
                    )));
                }
                if alt_count != 0 && alt_ord > alt_count {
                    return Err(ioerr("ALT ordinal exceeds allele count in .pvar"));
                }
                let haploidy_kind = self.plan.haploidy_of(out_idx);
                let sample_ploidy = haploidy_kind.and_then(|kind| {
                    if matches!(kind, HaploidyKind::Diploid) {
                        None
                    } else {
                        fill_sample_ploidy(
                            &mut sample_ploidy_buf,
                            kind,
                            &self.sex_by_sample,
                            self.n_samples,
                        );
                        Some(sample_ploidy_buf.as_slice())
                    }
                });

                // Reused across variants: a scoring run decodes millions of
                // blocks, and two fresh sample-sized allocations per block is
                // pure allocator traffic.
                hard_buf.clear();
                hard_buf.resize(self.n_samples, 255); // 255 = missing
                decoder.decode_variant_hardcalls(in_idx, alt_ord, &mut hard_buf, sample_ploidy)?;

                if let Some(kind) = haploidy_kind
                    && !matches!(kind, HaploidyKind::Diploid)
                {
                    enforce_haploidy(&mut hard_buf, &self.sex_by_sample, kind);
                }

                let mut block = vec![0u8; self.block_bytes];
                Self::pack_to_block(&mut block, &hard_buf);

                let mut cache = self.cache.lock().unwrap();
                let stored = cache.put(out_idx, block);

                let start = within_block;
                let end = start + to_copy;
                dst[written..written + to_copy].copy_from_slice(&stored[start..end]);
            }

            written += to_copy;
            within_block = 0;
            out_idx += 1;
        }

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tiny LRU for packed blocks
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Fixed-capacity cache of packed variant blocks, keyed by out-variant index.
///
/// Lookups are hash-based rather than a linear scan of the MRU list: a scoring
/// run performs one lookup per matched variant, and at millions of variants an
/// O(capacity) scan per lookup is pure overhead.
struct BlockCache {
    cap: usize,
    blocks: HashMap<usize, Vec<u8>>,
    /// Insertion order, used to pick the eviction victim. Entries are appended
    /// on insert only, so this stays in sync with `blocks` without any
    /// move-to-back bookkeeping on the (common) hit path.
    order: VecDeque<usize>,
}

impl BlockCache {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            blocks: HashMap::with_capacity(cap.max(1)),
            order: VecDeque::with_capacity(cap.max(1)),
        }
    }

    fn get(&self, k: usize) -> Option<&Vec<u8>> {
        self.blocks.get(&k)
    }

    /// Inserts `v`, evicting the oldest entry when at capacity, and returns a
    /// reference to the stored block so callers can copy out of it without
    /// cloning the buffer they just built.
    fn put(&mut self, k: usize, v: Vec<u8>) -> &[u8] {
        if self.blocks.insert(k, v).is_none() {
            self.order.push_back(k);
            while self.order.len() > self.cap {
                // Skip keys already evicted or re-inserted out of band.
                if let Some(victim) = self.order.pop_front()
                    && victim != k
                {
                    self.blocks.remove(&victim);
                }
            }
        }
        &self.blocks[&k]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Minimal local ByteRangeSource for `.pgen` (seekable)
////////////////////////////////////////////////////////////////////////////////////////////////////

struct LocalFileByteRangeSource {
    file: Mutex<File>,
    len: u64,
}

impl LocalFileByteRangeSource {
    fn open(path: &Path) -> Result<Self, PipelineError> {
        let f = File::open(path)
            .map_err(|e| PipelineError::Io(format!("Opening {}: {e}", path.display())))?;
        let len = f
            .metadata()
            .map_err(|e| PipelineError::Io(format!("Metadata {}: {e}", path.display())))?
            .len();
        Ok(Self {
            file: Mutex::new(f),
            len,
        })
    }
}
impl ByteRangeSource for LocalFileByteRangeSource {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        if dst.is_empty() {
            return Ok(());
        }
        if offset.saturating_add(dst.len() as u64) > self.len {
            return Err(ioerr("Attempted to read past end of local .pgen"));
        }
        let mut f = self.file.lock().unwrap();
        f.seek(SeekFrom::Start(offset))
            .map_err(|e| PipelineError::Io(e.to_string()))?;
        f.read_exact(dst)
            .map_err(|e| PipelineError::Io(e.to_string()))
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PGEN decoder (spec-aligned subset)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PgenMode {
    Bed = 0x01,
    FixHard = 0x02,
    FixDosage = 0x03,
    FixPhDosage = 0x04,
    Var = 0x10,
    VarIgnorable = 0x11,
}

fn read_le_u32(src: &dyn ByteRangeSource, off: u64) -> Result<u32, PipelineError> {
    let mut buf = [0u8; 4];
    src.read_at(off, &mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_le_u64(src: &dyn ByteRangeSource, off: u64) -> Result<u64, PipelineError> {
    let mut buf = [0u8; 8];
    src.read_at(off, &mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_varint_from_source(
    src: &dyn ByteRangeSource,
    offset: &mut u64,
) -> Result<u64, PipelineError> {
    let mut out: u64 = 0;
    let mut shift = 0;
    loop {
        if *offset >= src.len() {
            return Err(ioerr("Unexpected EOF in header varint"));
        }
        let mut byte = [0u8; 1];
        src.read_at(*offset, &mut byte)?;
        *offset += 1;
        out |= ((byte[0] & 0x7f) as u64) << shift;
        if (byte[0] & 0x80) == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(ioerr("Header varint too large"));
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
struct PgenHeader {
    mode: PgenMode,
    m_variants: u32,
    n_samples: u32,
    fmt_byte: u8,
    block_offsets: Vec<u64>,
    rec_types: Vec<u8>,
    rec_lens: Vec<u32>,
}

impl PgenHeader {
    fn parse(src: &dyn ByteRangeSource) -> Result<Self, PipelineError> {
        if src.len() < 3 {
            return Err(ioerr("PGEN file too small"));
        }
        let mut magic = [0u8; 3];
        src.read_at(0, &mut magic)?;
        if magic[0] != 0x6c || magic[1] != 0x1b {
            return Err(ioerr("Not a PGEN (bad magic)"));
        }

        let mode = match magic[2] {
            0x01 => PgenMode::Bed,
            0x02 => PgenMode::FixHard,
            0x03 => PgenMode::FixDosage,
            0x04 => PgenMode::FixPhDosage,
            0x10 => PgenMode::Var,
            0x11 => PgenMode::VarIgnorable,
            0x20 | 0x21 => {
                return Err(ioerr(
                    "External index modes (0x20/0x21) unsupported; re-encode with `plink2 --pgen ... --make-pgen`",
                ));
            }
            other => {
                return Err(PipelineError::Io(format!(
                    "Unsupported PGEN mode 0x{other:02x}"
                )));
            }
        };

        if mode == PgenMode::Bed {
            return Ok(Self {
                mode,
                m_variants: 0,
                n_samples: 0,
                fmt_byte: 0,
                block_offsets: vec![],
                rec_types: vec![],
                rec_lens: vec![],
            });
        }

        let m_variants = read_le_u32(src, 3)?;
        let n_samples = read_le_u32(src, 7)?;

        if matches!(
            mode,
            PgenMode::FixHard | PgenMode::FixDosage | PgenMode::FixPhDosage
        ) {
            let mut b = [0u8; 1];
            src.read_at(11, &mut b)?;
            let fmt = b[0];
            return Ok(Self {
                mode,
                m_variants,
                n_samples,
                fmt_byte: fmt,
                block_offsets: vec![],
                rec_types: vec![],
                rec_lens: vec![],
            });
        }

        let fmt = {
            let mut b = [0u8; 1];
            src.read_at(11, &mut b)?;
            b[0]
        };
        let type_bits = if (fmt & 0x0f) <= 3 { 4 } else { 8 };
        let len_bytes = match fmt & 0x07 {
            0 | 4 => 1,
            1 | 5 => 2,
            2 | 6 => 3,
            3 | 7 => 4,
            _ => unreachable!(),
        };
        let ac_bytes = match (fmt >> 4) & 0x03 {
            0 => 0,
            1 => 1,
            2 => 2,
            3 => 4,
            _ => unreachable!(),
        };
        let ref_flag_mode = (fmt >> 6) & 0x03;

        let blocks = ((m_variants as u64) + ((1u64 << 16) - 1)) >> 16;
        let mut block_offsets = Vec::with_capacity(blocks as usize);
        let mut off = 12u64;
        for _ in 0..blocks {
            block_offsets.push(read_le_u64(src, off)?);
            off += 8;
        }

        let mut rec_types = vec![0u8; m_variants as usize];
        let mut rec_lens = vec![0u32; m_variants as usize];

        let mut idx = 0usize;
        for _ in 0..blocks {
            let remain = (m_variants as usize) - idx;
            let cnt = remain.min(1 << 16);

            // types per block
            if type_bits == 4 {
                let nbytes = cnt.div_ceil(2);
                let mut buf = vec![0u8; nbytes];
                src.read_at(off, &mut buf)?;
                off += nbytes as u64;
                for (i, byte) in buf.into_iter().enumerate() {
                    let base = idx + 2 * i;
                    if base < idx + cnt {
                        rec_types[base] = byte & 0x0f;
                    }
                    if base + 1 < idx + cnt {
                        rec_types[base + 1] = (byte >> 4) & 0x0f;
                    }
                }
            } else {
                let nbytes = cnt;
                src.read_at(off, &mut rec_types[idx..idx + cnt])?;
                off += nbytes as u64;
            }

            if len_bytes > 0 {
                let nbytes = cnt * (len_bytes as usize);
                let mut buf = vec![0u8; nbytes];
                if nbytes > 0 {
                    src.read_at(off, &mut buf)?;
                }
                off += nbytes as u64;
                for i in 0..cnt {
                    let start = i * (len_bytes as usize);
                    let len = match len_bytes {
                        1 => u32::from(buf[start]),
                        2 => u32::from_le_bytes([buf[start], buf[start + 1], 0, 0]),
                        3 => u32::from_le_bytes([buf[start], buf[start + 1], buf[start + 2], 0]),
                        4 => u32::from_le_bytes([
                            buf[start],
                            buf[start + 1],
                            buf[start + 2],
                            buf[start + 3],
                        ]),
                        _ => unreachable!(),
                    };
                    rec_lens[idx + i] = len;
                }
            }

            if ac_bytes > 0 {
                let nbytes = cnt * (ac_bytes as usize);
                off += nbytes as u64;
            }
            if ref_flag_mode == 3 {
                let nbytes = cnt.div_ceil(8);
                off += nbytes as u64;
            }

            idx += cnt;
        }

        if mode == PgenMode::VarIgnorable {
            let mut ext_off = off;
            let header_flags = read_varint_from_source(src, &mut ext_off)?;
            let footer_flags = read_varint_from_source(src, &mut ext_off)?;

            if footer_flags != 0 {
                if ext_off
                    .checked_add(8)
                    .ok_or_else(|| ioerr("Footer offset overflow"))?
                    > src.len()
                {
                    return Err(ioerr("EOF reading footer offset"));
                }
                ext_off += 8;
            }

            let mut lengths = Vec::new();
            let mut mask = header_flags;
            while mask != 0 {
                if (mask & 1) != 0 {
                    lengths.push(read_varint_from_source(src, &mut ext_off)?);
                }
                mask >>= 1;
            }
            let sum: u64 = lengths.into_iter().try_fold(0u64, |acc, x| {
                acc.checked_add(x)
                    .ok_or_else(|| ioerr("Header extension length overflow"))
            })?;
            if ext_off
                .checked_add(sum)
                .ok_or_else(|| ioerr("Header extension overflow"))?
                > src.len()
            {
                return Err(ioerr("Header extensions overrun file"));
            }
        }

        Ok(Self {
            mode,
            m_variants,
            n_samples,
            fmt_byte: fmt,
            block_offsets,
            rec_types,
            rec_lens,
        })
    }
}

#[inline]
fn read_base128_varint(buf: &[u8], cursor: &mut usize) -> Result<u64, PipelineError> {
    let mut out: u64 = 0;
    let mut shift = 0;
    loop {
        if *cursor >= buf.len() {
            return Err(ioerr("Unexpected EOF in varint"));
        }
        let b = buf[*cursor];
        *cursor += 1;
        out |= ((b & 0x7f) as u64) << shift;
        if (b & 0x80) == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(ioerr("Varint too large"));
        }
    }
    Ok(out)
}

#[inline]
fn read_u24_le(buf: &[u8], cursor: &mut usize) -> Result<u32, PipelineError> {
    if *cursor + 3 > buf.len() {
        return Err(ioerr("EOF in u24"));
    }
    let v = u32::from_le_bytes([buf[*cursor], buf[*cursor + 1], buf[*cursor + 2], 0]);
    *cursor += 3;
    Ok(v)
}

#[inline]
fn read_bitarray_indices(
    buf: &[u8],
    cursor: &mut usize,
    nbits: usize,
) -> Result<Vec<usize>, PipelineError> {
    let nbytes = nbits.div_ceil(8);
    if *cursor + nbytes > buf.len() {
        return Err(ioerr("EOF in bitarray"));
    }
    let mut out = Vec::with_capacity(nbits.min(1024));
    for (j, &byte) in buf[*cursor..*cursor + nbytes].iter().enumerate() {
        for b in 0..8 {
            let bit = j * 8 + b;
            if bit >= nbits {
                break;
            }
            if (byte >> b) & 1 == 1 {
                out.push(bit);
            }
        }
    }
    *cursor += nbytes;
    Ok(out)
}

#[inline]
fn read_packed_fixed_width(
    buf: &[u8],
    cursor: &mut usize,
    width_bits: usize,
    count: usize,
) -> Result<Vec<u32>, PipelineError> {
    if width_bits == 0 {
        return Ok(vec![0; count]);
    }
    let total_bits = width_bits * count;
    let nbytes = total_bits.div_ceil(8);
    if *cursor + nbytes > buf.len() {
        return Err(ioerr("EOF in packed values"));
    }
    let slice = &buf[*cursor..*cursor + nbytes];
    let mut out = Vec::with_capacity(count);
    let mut bitpos = 0usize;
    for _ in 0..count {
        let mut acc = 0u32;
        for k in 0..width_bits {
            let bp = bitpos + k;
            let byte = slice[bp >> 3];
            let bit = (byte >> (bp & 7)) & 1;
            acc |= (bit as u32) << k;
        }
        out.push(acc);
        bitpos += width_bits;
    }
    *cursor += nbytes;
    Ok(out)
}

#[inline]
fn sample_id_bytes(n_samples: usize) -> usize {
    if n_samples <= (1 << 8) {
        1
    } else if n_samples <= (1 << 16) {
        2
    } else if n_samples <= (1 << 24) {
        3
    } else {
        4
    }
}

fn difflist_ids(
    buf: &[u8],
    cursor: &mut usize,
    n_samples: usize,
) -> Result<Vec<u32>, PipelineError> {
    let l = read_base128_varint(buf, cursor)? as usize;
    if l == 0 {
        return Ok(vec![]);
    }
    let g = l.div_ceil(64);
    let sid_bytes = sample_id_bytes(n_samples);

    let mut first_ids = Vec::with_capacity(g);
    for _ in 0..g {
        let v = match sid_bytes {
            1 => {
                if *cursor >= buf.len() {
                    return Err(ioerr("EOF in u8 first-ID"));
                }
                let v = buf[*cursor] as u32;
                *cursor += 1;
                v
            }
            2 => {
                if *cursor + 2 > buf.len() {
                    return Err(ioerr("EOF in u16 first-ID"));
                }
                let v = u16::from_le_bytes([buf[*cursor], buf[*cursor + 1]]) as u32;
                *cursor += 2;
                v
            }
            3 => read_u24_le(buf, cursor)?,
            _ => {
                if *cursor + 4 > buf.len() {
                    return Err(ioerr("EOF in u32 first-ID"));
                }
                let v = u32::from_le_bytes([
                    buf[*cursor],
                    buf[*cursor + 1],
                    buf[*cursor + 2],
                    buf[*cursor + 3],
                ]);
                *cursor += 4;
                v
            }
        };
        first_ids.push(v);
    }

    let mut group_delta_bytes = Vec::with_capacity(g.saturating_sub(1));
    if g > 1 {
        if *cursor + (g - 1) > buf.len() {
            return Err(ioerr("EOF in difflist group byte-lengths"));
        }
        for raw in &buf[*cursor..*cursor + g - 1] {
            group_delta_bytes.push((*raw as usize) + 63);
        }
        *cursor += g - 1;
    }

    let mut ids = Vec::with_capacity(l);
    let mut delta_cur = *cursor;

    for gi in 0..g {
        let group_elems = if gi < g - 1 { 64 } else { l - 64 * (g - 1) };
        if group_elems == 0 {
            return Err(ioerr("Empty difflist group"));
        }

        ids.push(first_ids[gi]);

        let start = delta_cur;
        for _ in 1..group_elems {
            let d = read_base128_varint(buf, &mut delta_cur)? as u32;
            let last = *ids.last().unwrap();
            ids.push(
                last.checked_add(d)
                    .ok_or_else(|| ioerr("Difflist delta overflow"))?,
            );
        }
        if gi < g - 1 {
            let used = delta_cur - start;
            let expected = group_delta_bytes[gi];
            if used != expected {
                return Err(ioerr("Difflist group byte-length mismatch"));
            }
        }
    }

    *cursor = delta_cur;
    if ids.len() != l {
        return Err(ioerr("Difflist decode count mismatch"));
    }
    Ok(ids)
}

fn difflist_pairs(
    buf: &[u8],
    cursor: &mut usize,
    n_samples: usize,
) -> Result<Vec<(u32, u8)>, PipelineError> {
    let l = read_base128_varint(buf, cursor)? as usize;
    if l == 0 {
        return Ok(vec![]);
    }
    let g = l.div_ceil(64);
    let sid_bytes = sample_id_bytes(n_samples);

    let mut first_ids = Vec::with_capacity(g);
    for _ in 0..g {
        let v = match sid_bytes {
            1 => {
                if *cursor >= buf.len() {
                    return Err(ioerr("EOF in u8 first-ID"));
                }
                let v = buf[*cursor] as u32;
                *cursor += 1;
                v
            }
            2 => {
                if *cursor + 2 > buf.len() {
                    return Err(ioerr("EOF in u16 first-ID"));
                }
                let v = u16::from_le_bytes([buf[*cursor], buf[*cursor + 1]]) as u32;
                *cursor += 2;
                v
            }
            3 => read_u24_le(buf, cursor)?,
            _ => {
                if *cursor + 4 > buf.len() {
                    return Err(ioerr("EOF in u32 first-ID"));
                }
                let v = u32::from_le_bytes([
                    buf[*cursor],
                    buf[*cursor + 1],
                    buf[*cursor + 2],
                    buf[*cursor + 3],
                ]);
                *cursor += 4;
                v
            }
        };
        first_ids.push(v);
    }

    let mut group_delta_bytes = Vec::with_capacity(g.saturating_sub(1));
    if g > 1 {
        if *cursor + (g - 1) > buf.len() {
            return Err(ioerr("EOF in difflist group byte-lengths"));
        }
        for raw in &buf[*cursor..*cursor + g - 1] {
            group_delta_bytes.push((*raw as usize) + 63);
        }
        *cursor += g - 1;
    }

    let vals_packed = l.div_ceil(4);
    if *cursor + vals_packed > buf.len() {
        return Err(ioerr("EOF in difflist values"));
    }
    let mut vals = Vec::with_capacity(l);
    for i in 0..vals_packed {
        let b = buf[*cursor + i];
        vals.push(b & 0b11);
        if vals.len() == l {
            break;
        }
        vals.push((b >> 2) & 0b11);
        if vals.len() == l {
            break;
        }
        vals.push((b >> 4) & 0b11);
        if vals.len() == l {
            break;
        }
        vals.push((b >> 6) & 0b11);
    }
    *cursor += vals_packed;

    let mut ids = Vec::with_capacity(l);
    let mut delta_cur = *cursor;

    for gi in 0..g {
        let group_elems = if gi < g - 1 { 64 } else { l - 64 * (g - 1) };
        if group_elems == 0 {
            return Err(ioerr("Empty difflist group"));
        }

        ids.push(first_ids[gi]);

        let start = delta_cur;
        for _ in 1..group_elems {
            let d = read_base128_varint(buf, &mut delta_cur)? as u32;
            let last = *ids.last().unwrap();
            ids.push(
                last.checked_add(d)
                    .ok_or_else(|| ioerr("Difflist delta overflow"))?,
            );
        }
        if gi < g - 1 {
            let used = delta_cur - start;
            let expected = group_delta_bytes[gi];
            if used != expected {
                return Err(ioerr("Difflist group byte-length mismatch"));
            }
        }
    }

    *cursor = delta_cur;
    if ids.len() != l || vals.len() != l {
        return Err(ioerr("Difflist decode count mismatch"));
    }
    Ok(ids.into_iter().zip(vals.into_iter()).collect())
}

/// Record spacing of the sparse offset index. A lookup sums at most
/// `OFFSET_STRIDE - 1` record lengths instead of walking to the start of the
/// 2^16-record variant block, which would be up to 65535 additions per variant.
/// The table itself costs `8 * m_variants / OFFSET_STRIDE` bytes — about 1 MB
/// for a WGS chr1 with 8.45M variants.
const OFFSET_STRIDE: usize = 64;

/// Offset of the first fixed-width record body, past the header and the
/// optional provisional-REF bitarray.
fn fixhard_body_offset(hdr: &PgenHeader) -> Result<u64, PipelineError> {
    let ref_flag_mode = (hdr.fmt_byte >> 6) & 0x03;
    let mut base = 12u64 + 1; // 11-byte header + format byte
    if ref_flag_mode == 3 {
        base = base
            .checked_add((hdr.m_variants as u64).div_ceil(8))
            .ok_or_else(|| ioerr("Header overflow"))?;
    }
    Ok(base)
}

/// Precomputes the absolute file offset of every `OFFSET_STRIDE`-th record.
///
/// Record offsets are only recoverable by summing record lengths from the start
/// of the enclosing 2^16-record variant block. Doing that per lookup makes
/// sparse access quadratic in the block size; doing it once here makes every
/// later lookup bounded by `OFFSET_STRIDE`.
fn build_stride_offsets(hdr: &PgenHeader, n_samples: usize) -> Result<Vec<u64>, PipelineError> {
    let m = hdr.m_variants as usize;
    if m == 0 {
        return Ok(Vec::new());
    }
    let entries = m.div_ceil(OFFSET_STRIDE);

    match hdr.mode {
        PgenMode::FixHard => {
            let base = fixhard_body_offset(hdr)?;
            let rec_len = n_samples.div_ceil(4) as u64;
            Ok((0..entries)
                .map(|k| base + (k * OFFSET_STRIDE) as u64 * rec_len)
                .collect())
        }
        PgenMode::Var | PgenMode::VarIgnorable => {
            if hdr.rec_lens.len() < m {
                return Err(ioerr("Header record-length table is short"));
            }
            let mut out = Vec::with_capacity(entries);
            let mut off = 0u64;
            for idx in 0..m {
                // Every 2^16 records the file gives an authoritative offset;
                // restarting from it keeps this exact rather than cumulative.
                if idx % 65536 == 0 {
                    off = *hdr
                        .block_offsets
                        .get(idx >> 16)
                        .ok_or_else(|| ioerr("Missing block offset"))?;
                }
                if idx % OFFSET_STRIDE == 0 {
                    out.push(off);
                }
                off = off
                    .checked_add(hdr.rec_lens[idx] as u64)
                    .ok_or_else(|| ioerr("Record offset overflow"))?;
            }
            Ok(out)
        }
        _ => Ok(Vec::new()),
    }
}

struct PgenDecoder {
    src: Arc<dyn ByteRangeSource>,
    hdr: PgenHeader,
    n: usize,
    scratch: Vec<u8>,
    /// Absolute file offset of record `k * OFFSET_STRIDE`.
    stride_offsets: Vec<u64>,
    /// Raw (pre-projection) genotype categories of the LD anchor identified by
    /// `anchor_idx`. An LD-compressed record diffs against the most recent
    /// *non*-LD-compressed record in its block, so decoding record `i` out of
    /// order requires locating and decoding that anchor rather than trusting
    /// whatever was decoded last.
    anchor_idx: Option<usize>,
    anchor_cats: Vec<u8>,
    /// Reused per-decode scratch so a scoring run doing millions of variant
    /// reads does not allocate two sample-sized buffers per variant.
    cats_buf: Vec<u8>,
    alt_counts: Vec<u16>,
    /// Tallies what the hard-call projection discards, and says so out loud
    /// once it has seen enough records to mean it. Shared with the
    /// `VirtualPlink19` handle rather than owned outright, because the caller
    /// holding that handle never sees this decoder.
    dosage_meter: Arc<DosageCoercionMeter>,
}

impl PgenDecoder {
    fn new(
        src: Arc<dyn ByteRangeSource>,
        hdr: PgenHeader,
        n_samples_from_psam: usize,
        in_variants: usize,
        alt_counts: Vec<u16>,
    ) -> Result<Self, PipelineError> {
        match hdr.mode {
            PgenMode::Bed => return Err(ioerr("Mode 0x01 passthrough handled elsewhere")),
            PgenMode::FixDosage | PgenMode::FixPhDosage => {
                return Err(ioerr("Fixed-width dosage modes carry no hard-calls"));
            }
            _ => {}
        }

        if hdr.m_variants as usize != in_variants {
            return Err(PipelineError::Io(format!(
                "Variant count mismatch: .pgen header {0} vs .pvar {1}",
                hdr.m_variants, in_variants
            )));
        }

        if hdr.n_samples as usize != n_samples_from_psam && hdr.n_samples != 0 {
            return Err(PipelineError::Io(format!(
                "Sample count mismatch: .pgen {0} vs .psam {1}",
                hdr.n_samples, n_samples_from_psam
            )));
        }

        let stride_offsets = build_stride_offsets(&hdr, n_samples_from_psam)?;

        Ok(Self {
            src,
            hdr,
            n: n_samples_from_psam,
            scratch: Vec::new(),
            stride_offsets,
            anchor_idx: None,
            anchor_cats: Vec::new(),
            cats_buf: Vec::new(),
            alt_counts,
            dosage_meter: Arc::new(DosageCoercionMeter::new(
                n_samples_from_psam,
                in_variants,
            )),
        })
    }

    fn record_offset_len(&self, idx: usize) -> Result<(u64, usize, u8), PipelineError> {
        match self.hdr.mode {
            PgenMode::FixHard => {
                let rec_len = self.n.div_ceil(4);
                let base = fixhard_body_offset(&self.hdr)?;
                Ok((base + (idx as u64) * (rec_len as u64), rec_len, 0))
            }
            PgenMode::Var | PgenMode::VarIgnorable => {
                // Start from the nearest indexed record rather than the start of
                // the 2^16 variant block, so a lookup costs at most
                // `OFFSET_STRIDE - 1` additions regardless of where in the block
                // `idx` falls.
                let anchor = idx / OFFSET_STRIDE;
                let mut off = *self
                    .stride_offsets
                    .get(anchor)
                    .ok_or_else(|| ioerr("Missing stride offset"))?;
                let mut cursor = anchor * OFFSET_STRIDE;
                while cursor < idx {
                    off += *self
                        .hdr
                        .rec_lens
                        .get(cursor)
                        .ok_or_else(|| ioerr("Missing rec_len"))? as u64;
                    cursor += 1;
                }
                let rec_len = *self
                    .hdr
                    .rec_lens
                    .get(idx)
                    .ok_or_else(|| ioerr("Missing rec_len"))?
                    as usize;
                let rec_ty = *self
                    .hdr
                    .rec_types
                    .get(idx)
                    .ok_or_else(|| ioerr("Missing rec_type"))?;
                Ok((off, rec_len, rec_ty))
            }
            _ => Err(ioerr("Unsupported PGEN mode")),
        }
    }

    /// Reads record `idx` into `self.scratch`, returning its byte length and
    /// record type.
    fn load_record(&mut self, idx: usize) -> Result<(usize, u8), PipelineError> {
        let (off, len, rec_ty) = self.record_offset_len(idx)?;
        if self.scratch.len() < len {
            self.scratch.resize(len, 0);
        }
        self.src.read_at(off, &mut self.scratch[..len])?;
        Ok((len, rec_ty))
    }

    /// Locates the record an LD-compressed record at `idx` diffs against: the
    /// most recent record in the same 2^16 variant block whose main track is not
    /// itself LD-compressed.
    fn ld_anchor_index(&self, idx: usize) -> Result<usize, PipelineError> {
        let block_start = idx & !0xffff;
        let mut j = idx;
        while j > block_start {
            j -= 1;
            let ty = *self
                .hdr
                .rec_types
                .get(j)
                .ok_or_else(|| ioerr("Missing rec_type while resolving LD anchor"))?;
            if !matches!(ty & 0x07, 2 | 3) {
                return Ok(j);
            }
        }
        Err(ioerr(
            "LD-compressed record has no anchor in its variant block",
        ))
    }

    /// Ensures `self.anchor_cats` holds the raw categories of the LD anchor for
    /// `target`, decoding that anchor record if it is not already cached.
    ///
    /// Sparse scoring runs touch variants in arbitrary order, so the anchor can
    /// never be assumed to be whatever was decoded most recently.
    fn ensure_anchor(&mut self, target: usize) -> Result<(), PipelineError> {
        let anchor_idx = self.ld_anchor_index(target)?;
        if self.anchor_idx == Some(anchor_idx) && self.anchor_cats.len() == self.n {
            return Ok(());
        }

        // Invalidate first: if decoding fails midway we must not leave a stale
        // index pointing at partially-written categories.
        self.anchor_idx = None;

        let (len, rec_ty) = self.load_record(anchor_idx)?;
        let main_kind = rec_ty & 0x07;
        if matches!(main_kind, 2 | 3) {
            return Err(ioerr("LD anchor is itself LD-compressed"));
        }
        let n = self.n;
        let mut cursor = 0usize;
        let Self {
            scratch,
            anchor_cats,
            ..
        } = self;
        decode_main_track_into(&scratch[..len], &mut cursor, n, main_kind, None, anchor_cats)?;

        self.anchor_idx = Some(anchor_idx);
        Ok(())
    }

    fn decode_variant_hardcalls(
        &mut self,
        in_idx: u32,
        alt_ord_1b: u16,
        dst: &mut [u8],
        sample_ploidy: Option<&[u8]>,
    ) -> Result<(), PipelineError> {
        if dst.len() != self.n {
            return Err(ioerr("Hardcall buffer length must equal sample count"));
        }
        if let Some(ploidy) = sample_ploidy
            && ploidy.len() != self.n
        {
            return Err(ioerr("Sample ploidy length mismatch"));
        }
        let idx = in_idx as usize;
        if idx >= self.hdr.m_variants as usize {
            return Err(ioerr("Variant index out of bounds"));
        }

        let n = self.n;
        let alt_count = self.alt_counts.get(idx).copied().unwrap_or(0);

        // Peek at the record type first: an LD-compressed record needs its
        // anchor decoded before we overwrite the scratch buffer with the target
        // record.
        let (_, _, rec_ty) = self.record_offset_len(idx)?;
        let main_kind = rec_ty & 0x07;
        if matches!(main_kind, 2 | 3) {
            if (idx & 0xffff) == 0 {
                return Err(ioerr("LD-compressed record at block start"));
            }
            self.ensure_anchor(idx)?;
        }

        let (len, rec_ty) = self.load_record(idx)?;
        let mut cursor = 0usize;

        // Disjoint field borrows: the record bytes, the anchor categories, the
        // category output buffer and the coercion meter are all owned by
        // `self`.
        let Self {
            scratch,
            anchor_cats,
            cats_buf,
            dosage_meter,
            ..
        } = self;
        let buf = &scratch[..len];
        let anchor = if matches!(main_kind, 2 | 3) {
            Some(anchor_cats.as_slice())
        } else {
            None
        };
        decode_main_track_into(buf, &mut cursor, n, main_kind, anchor, cats_buf)?;
        let cats = cats_buf.as_mut_slice();

        let has_multiallelic = (rec_ty & 0b0000_1000) != 0;
        let mut a1dosage = vec![255u8; n];
        if has_multiallelic {
            apply_multiallelic_and_project(
                buf,
                &mut cursor,
                n,
                cats,
                alt_count,
                alt_ord_1b,
                &mut a1dosage,
            )?;
        } else {
            cats_to_a1dosage(&mut a1dosage, cats);
        }

        if (rec_ty & 0b0001_0000) != 0 {
            if cursor >= len {
                return Err(PipelineError::Io(format!(
                    "EOF in phase header (variant #{idx})"
                )));
            }
            let start = cursor;
            let mut bit_cursor = 0usize;
            let phasepresent = (buf[start] & 1) == 1;
            bit_cursor += 1;
            let h = cats.iter().filter(|&&c| c == 1).count();
            let mut phased_count = h;
            if phasepresent {
                let mut present_count = 0usize;
                for _ in 0..h {
                    let bit_idx = bit_cursor;
                    let byte_idx = start + (bit_idx >> 3);
                    if byte_idx >= len {
                        return Err(PipelineError::Io(format!(
                            "EOF in phase presence (variant #{idx})"
                        )));
                    }
                    let byte = buf[byte_idx];
                    if (byte >> (bit_idx & 7)) & 1 == 1 {
                        present_count += 1;
                    }
                    bit_cursor += 1;
                }
                phased_count = present_count;
                if bit_cursor & 7 != 0 {
                    bit_cursor += 8 - (bit_cursor & 7);
                }
            }
            for _ in 0..phased_count {
                let bit_idx = bit_cursor;
                let byte_idx = start + (bit_idx >> 3);
                if byte_idx >= len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phase info (variant #{idx})"
                    )));
                }
                bit_cursor += 1;
            }
            let bytes_needed = bit_cursor.div_ceil(8);
            if start + bytes_needed > len {
                return Err(PipelineError::Io(format!(
                    "EOF in phase track (variant #{idx})"
                )));
            }
            cursor = start + bytes_needed;
        }

        let has_dosage = (rec_ty & 0b0110_0000) != 0;

        // Multiallelic dosage tracks (#5-#10) are intentionally ignored for
        // alternate alleles beyond the first; keep hard-call derived values
        // (which may remain missing).
        let decode_dosage = alt_count <= 1 || alt_ord_1b == 1;

        // Account for this input variant at most once for the life of the
        // process, and only on a visit that actually reads its dosage track.
        // Both halves matter: a multiallelic record is re-decoded once per ALT,
        // and a multi-pass consumer (block Lanczos re-streams the matrix every
        // iteration) re-decodes every record once per pass. A visit that skips
        // the dosage track leaves the variant unclaimed, so a later
        // representative visit can still count it.
        let meter: &DosageCoercionMeter = &**dosage_meter;
        let accounting = (!has_dosage || decode_dosage) && meter.claim(idx);
        if accounting {
            meter.note_variant(has_dosage);
        }
        let entry_meter = accounting.then_some(meter);

        let mut dosage_entries = 0usize;
        if has_dosage {
            let b5 = (rec_ty & 0b0010_0000) != 0;
            let b6 = (rec_ty & 0b0100_0000) != 0;

            if b5 && !b6 {
                let ids = difflist_ids(buf, &mut cursor, n)?;
                let cnt = ids.len();
                let need = cnt * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in dosage values (variant #{idx})"
                    )));
                }
                if decode_dosage {
                    for (i, &sid) in ids.iter().enumerate() {
                        let v = u16::from_le_bytes([buf[cursor + 2 * i], buf[cursor + 2 * i + 1]]);
                        absorb_dosage_entry(
                            &mut a1dosage,
                            sid as usize,
                            v,
                            sample_ploidy,
                            entry_meter,
                        );
                    }
                }
                cursor += need;
                dosage_entries = cnt;
            } else if !b5 && b6 {
                let need = n * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in dense dosage values (variant #{idx})"
                    )));
                }
                if decode_dosage {
                    for s in 0..n {
                        let v = u16::from_le_bytes([buf[cursor + 2 * s], buf[cursor + 2 * s + 1]]);
                        absorb_dosage_entry(&mut a1dosage, s, v, sample_ploidy, entry_meter);
                    }
                }
                cursor += need;
                dosage_entries = n;
            } else {
                let present = read_bitarray_indices(buf, &mut cursor, n)?;
                let cnt = present.len();
                let need = cnt * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in sparse dosage values (variant #{idx})"
                    )));
                }
                if decode_dosage {
                    for (i, &s) in present.iter().enumerate() {
                        let v = u16::from_le_bytes([buf[cursor + 2 * i], buf[cursor + 2 * i + 1]]);
                        absorb_dosage_entry(&mut a1dosage, s, v, sample_ploidy, entry_meter);
                    }
                }
                cursor += need;
                dosage_entries = cnt;
            }
        }

        if (rec_ty & 0b1000_0000) != 0 {
            if !has_dosage {
                return Err(PipelineError::Io(format!(
                    "Phased dosage track present without dosage (variant #{idx})"
                )));
            }
            let b5 = (rec_ty & 0b0010_0000) != 0;
            let b6 = (rec_ty & 0b0100_0000) != 0;
            if !b5 && b6 {
                let need = n * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phased dense dosage values (variant #{idx})"
                    )));
                }
                cursor += need;
            } else {
                let d = dosage_entries;
                let nbytes = d.div_ceil(8);
                if cursor + nbytes > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phased dosage presence (variant #{idx})"
                    )));
                }
                let mut present_count = 0usize;
                for bit in 0..d {
                    let byte = buf[cursor + (bit >> 3)];
                    if (byte >> (bit & 7)) & 1 == 1 {
                        present_count += 1;
                    }
                }
                cursor += nbytes;
                let need = present_count * 2;
                if cursor + need > len {
                    return Err(PipelineError::Io(format!(
                        "EOF in phased dosage values (variant #{idx})"
                    )));
                }
                cursor += need;
            }
        }

        match len.checked_sub(cursor) {
            Some(0) => {}
            Some(remaining) => {
                return Err(PipelineError::Io(format!(
                    "Trailing data ({remaining} bytes) after variant #{idx}"
                )));
            }
            None => {
                return Err(PipelineError::Io(format!(
                    "Cursor advanced beyond end of record for variant #{idx}"
                )));
            }
        }

        // Only on a visit that changed the counts: the verdict cannot move
        // otherwise, and a scoring run re-reads records far more often than it
        // discovers them.
        if accounting {
            meter.maybe_report();
        }

        dst.copy_from_slice(&a1dosage);
        Ok(())
    }
}

/// Fills `cats` with the raw 2-bit genotype categories encoded by the main data
/// track at `cursor`, advancing `cursor` past it.
///
/// "Raw" means pre-projection: no multiallelic patching, ploidy coercion, or
/// dosage overlay is applied. That is exactly the form an LD-compressed record
/// diffs against, so the same output can be cached and reused as an anchor.
fn decode_main_track_into(
    buf: &[u8],
    cursor: &mut usize,
    n: usize,
    main_kind: u8,
    anchor: Option<&[u8]>,
    cats: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let len = buf.len();
    cats.clear();
    cats.resize(n, 3);

    match main_kind {
        0 => {
            let need = n.div_ceil(4);
            if *cursor + need > len {
                return Err(ioerr("Truncated type-0 main track"));
            }
            unpack_pgen2bit_to_categories(&buf[*cursor..*cursor + need], cats, n);
            *cursor += need;
        }
        1 => {
            if *cursor >= len {
                return Err(ioerr("Truncated type-1 header byte"));
            }
            let pair = buf[*cursor];
            *cursor += 1;
            let (low, high) = match pair {
                1 => (0u8, 1),
                2 => (0, 2),
                3 => (0, 3),
                5 => (1, 2),
                6 => (1, 3),
                9 => (2, 3),
                _ => return Err(ioerr("Invalid 1-bit pair code")),
            };
            let idxs = read_bitarray_indices(buf, cursor, n)?;
            cats.fill(low);
            for bit in idxs {
                if bit < n {
                    cats[bit] = high;
                }
            }
            for (sid, val) in difflist_pairs(buf, cursor, n)? {
                if (sid as usize) < n {
                    cats[sid as usize] = val;
                }
            }
        }
        2 | 3 => {
            let anchor = anchor.ok_or_else(|| ioerr("Missing LD anchor"))?;
            if anchor.len() != n {
                return Err(ioerr("LD anchor sample-count mismatch"));
            }
            cats.copy_from_slice(anchor);
            for (sid, val) in difflist_pairs(buf, cursor, n)? {
                if (sid as usize) < n {
                    cats[sid as usize] = val;
                }
            }
            if main_kind == 3 {
                // Type 3 is "LD-compressed, inverted": the diff is patched in
                // first, then REF/ALT homozygotes swap.
                for c in cats.iter_mut() {
                    if *c == 0 {
                        *c = 2;
                    } else if *c == 2 {
                        *c = 0;
                    }
                }
            }
        }
        4 | 6 | 7 => {
            let x = match main_kind {
                4 => 0u8,
                6 => 2,
                _ => 3,
            };
            cats.fill(x);
            for (sid, val) in difflist_pairs(buf, cursor, n)? {
                if (sid as usize) < n {
                    cats[sid as usize] = val;
                }
            }
        }
        _ => {
            return Err(PipelineError::Io(format!(
                "Unsupported main-track type {main_kind}"
            )));
        }
    }
    Ok(())
}

fn unpack_pgen2bit_to_categories(block: &[u8], dst: &mut [u8], n: usize) {
    let mut i = 0usize;
    for &byte in block {
        for shift in 0..4 {
            if i >= n {
                return;
            }
            dst[i] = (byte >> (2 * shift)) & 0b11;
            i += 1;
        }
    }
}

#[inline]
fn cats_to_a1dosage(dst: &mut [u8], cats: &[u8]) {
    for (d, c) in dst.iter_mut().zip(cats) {
        *d = match *c {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => 255,
        };
    }
}

#[inline]
fn collect_cat_ids(cats: &[u8], cat: u8) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, &c) in cats.iter().enumerate() {
        if c == cat {
            out.push(i as u32);
        }
    }
    out
}

fn apply_multiallelic_and_project(
    record: &[u8],
    cursor: &mut usize,
    n: usize,
    cats: &mut [u8],
    alt_count: u16,
    alt_ord_1b: u16,
    out: &mut [u8],
) -> Result<(), PipelineError> {
    if alt_count <= 1 {
        cats_to_a1dosage(out, cats);
        return Ok(());
    }

    if *cursor >= record.len() {
        return Err(ioerr("EOF before multiallelic patch header"));
    }
    let fmt_byte = record[*cursor];
    *cursor += 1;
    let cat1_fmt = fmt_byte & 0x0f;
    let cat2_fmt = (fmt_byte >> 4) & 0x0f;

    let cat1_ids = collect_cat_ids(cats, 1);
    let cat2_ids = collect_cat_ids(cats, 2);

    let mut cat1_override: Vec<(u32, u16)> = Vec::new();
    if cat1_fmt != 15 {
        match cat1_fmt {
            0 => {
                let set_indices = read_bitarray_indices(record, cursor, cat1_ids.len())?;
                let k = set_indices.len();
                let width = match alt_count {
                    2 => 0,
                    3 => 1,
                    4..=5 => 2,
                    6..=17 => 4,
                    18..=257 => 8,
                    258..=65535 => 16,
                    _ => 24,
                } as usize;
                let vals = read_packed_fixed_width(record, cursor, width, k)?;
                for (idx_in_list, v) in set_indices.into_iter().zip(vals.into_iter()) {
                    let sid = cat1_ids[idx_in_list];
                    let altj = if width == 0 { 2 } else { (v as u16) + 2 };
                    cat1_override.push((sid, altj));
                }
            }
            1 => {
                let sids = difflist_ids(record, cursor, n)?;
                let k = sids.len();
                let width = match alt_count {
                    2 => 0,
                    3 => 1,
                    4..=5 => 2,
                    6..=17 => 4,
                    18..=257 => 8,
                    258..=65535 => 16,
                    _ => 24,
                } as usize;
                let vals = read_packed_fixed_width(record, cursor, width, k)?;
                for (sid, v) in sids.into_iter().zip(vals.into_iter()) {
                    let altj = if width == 0 { 2 } else { (v as u16) + 2 };
                    cat1_override.push((sid, altj));
                }
            }
            _ => return Err(ioerr("Unsupported multiallelic cat1 patch format")),
        }
    }

    let mut cat2_override: Vec<(u32, (u16, u16))> = Vec::new();
    if cat2_fmt != 15 {
        match cat2_fmt {
            0 => {
                let set_indices = read_bitarray_indices(record, cursor, cat2_ids.len())?;
                let k = set_indices.len();
                if alt_count == 2 {
                    let hom2_flags = read_bitarray_indices(record, cursor, k)?;
                    let mut is_hom2 = vec![false; k];
                    for pos in hom2_flags {
                        if pos < k {
                            is_hom2[pos] = true;
                        }
                    }
                    for (flag, idx_in_list) in is_hom2.into_iter().zip(set_indices.into_iter()) {
                        let sid = cat2_ids[idx_in_list];
                        let pair = if flag { (2, 2) } else { (1, 2) };
                        cat2_override.push((sid, pair));
                    }
                } else {
                    let width = match alt_count {
                        3..=4 => 2,
                        5..=16 => 4,
                        17..=256 => 8,
                        257..=65535 => 16,
                        _ => 24,
                    } as usize;
                    let vals = read_packed_fixed_width(record, cursor, width, 2 * k)?;
                    for i in 0..k {
                        let sid = cat2_ids[set_indices[i]];
                        let lo = (vals[2 * i] as u16) + 1;
                        let hi = (vals[2 * i + 1] as u16) + 1;
                        let pair = if lo <= hi { (lo, hi) } else { (hi, lo) };
                        cat2_override.push((sid, pair));
                    }
                }
            }
            1 => {
                let sids = difflist_ids(record, cursor, n)?;
                let k = sids.len();
                if alt_count == 2 {
                    let hom2_flags = read_bitarray_indices(record, cursor, k)?;
                    let mut is_hom2 = vec![false; k];
                    for pos in hom2_flags {
                        if pos < k {
                            is_hom2[pos] = true;
                        }
                    }
                    for (flag, sid) in is_hom2.into_iter().zip(sids.into_iter()) {
                        let pair = if flag { (2, 2) } else { (1, 2) };
                        cat2_override.push((sid, pair));
                    }
                } else {
                    let width = match alt_count {
                        3..=4 => 2,
                        5..=16 => 4,
                        17..=256 => 8,
                        257..=65535 => 16,
                        _ => 24,
                    } as usize;
                    let vals = read_packed_fixed_width(record, cursor, width, 2 * k)?;
                    for i in 0..k {
                        let sid = sids[i];
                        let lo = (vals[2 * i] as u16) + 1;
                        let hi = (vals[2 * i + 1] as u16) + 1;
                        let pair = if lo <= hi { (lo, hi) } else { (hi, lo) };
                        cat2_override.push((sid, pair));
                    }
                }
            }
            _ => return Err(ioerr("Unsupported multiallelic cat2 patch format")),
        }
    }

    cat1_override.sort_unstable_by_key(|x| x.0);
    cat2_override.sort_unstable_by_key(|x| x.0);

    for i in 0..n {
        let c = cats[i];
        out[i] = match c {
            0 => 0,
            3 => 255,
            1 => {
                let mut altj = 1u16;
                if let Ok(pos) = cat1_override.binary_search_by_key(&(i as u32), |(sid, _)| *sid) {
                    altj = cat1_override[pos].1;
                }
                if altj == alt_ord_1b { 1 } else { 0 }
            }
            2 => {
                let mut pair = (1u16, 1u16);
                if let Ok(pos) = cat2_override.binary_search_by_key(&(i as u32), |(sid, _)| *sid) {
                    pair = cat2_override[pos].1;
                }
                let mut dose = 0u8;
                if pair.0 == alt_ord_1b {
                    dose += 1;
                }
                if pair.1 == alt_ord_1b {
                    dose += 1;
                }
                dose
            }
            _ => 255,
        };
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dosage → hard-call coercion, and the cost of it
////////////////////////////////////////////////////////////////////////////////////////////////////

/// How far a dosage may sit from a whole allele count and still be read as that
/// count. Past it the genotype is dropped rather than guessed at.
const DOSAGE_HARDCALL_TOLERANCE: f32 = 0.10;

/// Input variants that must be seen before the coercion counts are allowed to
/// say anything. Carrying a dosage track is a property of each record, so a few
/// hundred records already separate an imputed fileset from an array one; a
/// fileset smaller than this is judged on the whole of it instead.
const DOSAGE_VERDICT_AFTER_VARIANTS: u64 = 256;

/// The share of seen variants that must carry a dosage track before this stops
/// being a footnote and becomes a wrong-analysis warning. Compared as integer
/// percent so no float rounding decides whether a user is told.
const DOSAGE_ALARM_PERCENT: u64 = 50;

/// What presenting a `.pgen` as PLINK 1.9 hard calls has discarded.
///
/// Counted over *input* variants, each one accounted exactly once however many
/// times a consumer re-reads it, so these numbers describe the dataset and not
/// the reads. Fields are cumulative over the life of the handle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DosageCoercionReport {
    /// Input variants whose record has been decoded at least once.
    pub variants_examined: u64,
    /// How many of those carried a dosage track at all.
    pub variants_with_dosage: u64,
    /// Dosage entries read. Missing dosages and samples with no alleles at the
    /// site are excluded: neither loses anything to a hard call.
    pub dosage_values: u64,
    /// Entries that are not whole allele counts, and so cannot survive the
    /// hard-call representation however they were resolved. This is the
    /// headline "the input really is dosage-valued" number.
    pub fractional_values: u64,
    /// Fractional entries snapped to the nearest whole count, because the
    /// record's hard-call track had no call for that sample.
    pub rounded_to_hardcall: u64,
    /// Entries turned into missing genotypes because no whole count was within
    /// `DOSAGE_HARDCALL_TOLERANCE`.
    pub dropped_off_tolerance: u64,
    /// Fractional entries never consulted at all, because the record's own
    /// hard-call track already had a call for that sample and it wins. On a
    /// fileset PLINK 2 wrote with both tracks this is where nearly all of the
    /// loss lives, and the tolerance above never enters into it.
    pub discarded_for_existing_hardcall: u64,
}

/// Accumulates a `DosageCoercionReport` as the `.pgen` is decoded, and raises
/// the alarm once the counts are decisive.
///
/// Shared between the decoder and the `VirtualPlink19` handle, so the counters
/// are atomics rather than a second mutex on the decode path.
struct DosageCoercionMeter {
    n_samples: usize,
    in_variants: usize,
    /// One bit per input variant, set the first time that variant is counted.
    /// Without it a multi-pass fit would multiply every total below by the
    /// number of passes and report a fiction.
    accounted: Vec<AtomicU64>,
    variants_examined: AtomicU64,
    variants_with_dosage: AtomicU64,
    dosage_values: AtomicU64,
    fractional_values: AtomicU64,
    rounded_to_hardcall: AtomicU64,
    dropped_off_tolerance: AtomicU64,
    discarded_for_existing_hardcall: AtomicU64,
    /// Latches, so a message cannot be printed once per variant. The note and
    /// the warning latch separately: a fileset whose dosage records only begin
    /// part-way through is upgraded from one to the other rather than being
    /// stuck with whatever the first few hundred records suggested.
    noted: AtomicBool,
    alarmed: AtomicBool,
}

impl DosageCoercionMeter {
    fn new(n_samples: usize, in_variants: usize) -> Self {
        Self {
            n_samples,
            in_variants,
            accounted: (0..in_variants.div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect(),
            variants_examined: AtomicU64::new(0),
            variants_with_dosage: AtomicU64::new(0),
            dosage_values: AtomicU64::new(0),
            fractional_values: AtomicU64::new(0),
            rounded_to_hardcall: AtomicU64::new(0),
            dropped_off_tolerance: AtomicU64::new(0),
            discarded_for_existing_hardcall: AtomicU64::new(0),
            noted: AtomicBool::new(false),
            alarmed: AtomicBool::new(false),
        }
    }

    /// Claims input variant `idx` for counting, returning `false` if some
    /// earlier decode of the same record already counted it.
    ///
    /// The bitset is allocated in 64-bit words, so the length check is against
    /// the variant count rather than the word count: past the end of the
    /// fileset there is no variant to account for, even where a spare bit
    /// exists to record one.
    fn claim(&self, idx: usize) -> bool {
        if idx >= self.in_variants {
            return false;
        }
        let Some(word) = self.accounted.get(idx >> 6) else {
            return false;
        };
        let bit = 1u64 << (idx & 63);
        (word.fetch_or(bit, Ordering::Relaxed) & bit) == 0
    }

    fn note_variant(&self, has_dosage: bool) {
        self.variants_examined.fetch_add(1, Ordering::Relaxed);
        if has_dosage {
            self.variants_with_dosage.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records one dosage entry and what the hard-call projection did to it.
    ///
    /// The tolerance verdict comes from `u16_to_hardcall_biallelic` itself
    /// rather than being re-derived here, so the counts cannot drift away from
    /// the conversion they claim to describe.
    fn record_entry(&self, v: u16, ploidy: u8, had_hardcall: bool) {
        // A missing dosage, or a sample with no alleles at this site, loses
        // nothing to the hard-call form: it was already absent.
        if v == 65535 || ploidy == 0 {
            return;
        }
        self.dosage_values.fetch_add(1, Ordering::Relaxed);

        // Decided on the stored integer rather than the float: the scale is a
        // power of two, so a whole allele count is exactly a multiple of one
        // copy's worth and no epsilon is involved.
        let per_copy: u32 = if ploidy <= 1 { 32768 } else { 16384 };
        let fractional = (v as u32) % per_copy != 0;
        if fractional {
            self.fractional_values.fetch_add(1, Ordering::Relaxed);
        }

        if had_hardcall {
            if fractional {
                self.discarded_for_existing_hardcall
                    .fetch_add(1, Ordering::Relaxed);
            }
        } else if u16_to_hardcall_biallelic(v, ploidy) == 255 {
            self.dropped_off_tolerance.fetch_add(1, Ordering::Relaxed);
        } else if fractional {
            self.rounded_to_hardcall.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn report(&self) -> DosageCoercionReport {
        DosageCoercionReport {
            variants_examined: self.variants_examined.load(Ordering::Relaxed),
            variants_with_dosage: self.variants_with_dosage.load(Ordering::Relaxed),
            dosage_values: self.dosage_values.load(Ordering::Relaxed),
            fractional_values: self.fractional_values.load(Ordering::Relaxed),
            rounded_to_hardcall: self.rounded_to_hardcall.load(Ordering::Relaxed),
            dropped_off_tolerance: self.dropped_off_tolerance.load(Ordering::Relaxed),
            discarded_for_existing_hardcall: self
                .discarded_for_existing_hardcall
                .load(Ordering::Relaxed),
        }
    }

    /// Tells the user, at most one note and at most one warning for the life of
    /// the handle. The fileset's own size bounds the evidence threshold, so a
    /// fileset of three variants is still judged — on all three of them.
    ///
    /// Written to stderr rather than through `log`, because nothing in this
    /// crate installs a logger: a `warn!` here would be discarded, which for
    /// this particular message is the same as not writing it at all.
    fn maybe_report(&self) {
        let report = self.report();
        if report.variants_with_dosage == 0 {
            return;
        }
        let decisive = DOSAGE_VERDICT_AFTER_VARIANTS
            .min(self.in_variants as u64)
            .max(1);
        if report.variants_examined < decisive {
            return;
        }

        // Short-circuit order matters: below the alarm share, `alarmed` must
        // stay unlatched so a later, more dosage-heavy stretch can still raise
        // it. Above the share but already latched, the note is suppressed too,
        // because the warning subsumes it.
        if report.variants_with_dosage * 100 >= report.variants_examined * DOSAGE_ALARM_PERCENT
            && !self.alarmed.swap(true, Ordering::Relaxed)
        {
            self.noted.store(true, Ordering::Relaxed);
            eprint!("{}", format_dosage_alarm(&report, self.n_samples));
        } else if !self.noted.swap(true, Ordering::Relaxed) {
            eprint!("{}", format_dosage_note(&report));
        }
    }
}

/// The quiet form, for a fileset where dosage records are the minority: worth
/// saying once, not worth a banner.
fn format_dosage_note(report: &DosageCoercionReport) -> String {
    let tolerance = DOSAGE_HARDCALL_TOLERANCE;
    format!(
        "> Note: {}/{} .pgen variants read so far carry a dosage track. Those dosages are \
         read as hard calls (nearest whole allele count within ±{tolerance}, otherwise \
         missing); {} values so far were not whole allele counts.\n",
        report.variants_with_dosage, report.variants_examined, report.fractional_values,
    )
}

/// The loud form, for a fileset that is substantially dosage-valued. This is
/// the case where the analysis the user believes they asked for and the
/// analysis they are getting are two different analyses.
fn format_dosage_alarm(report: &DosageCoercionReport, n_samples: usize) -> String {
    use std::fmt::Write;

    let percent = |num: u64, den: u64| -> f64 {
        if den == 0 {
            0.0
        } else {
            (num as f64) * 100.0 / (den as f64)
        }
    };
    let rule = "=".repeat(81);
    let tolerance = DOSAGE_HARDCALL_TOLERANCE;

    let mut out = String::with_capacity(1024);
    // Built whole and written in one call: several decode threads may share
    // this meter, and a half-interleaved banner would be worse than none.
    let _ = writeln!(out, "\n{rule}");
    let _ = writeln!(
        out,
        " WARNING: this .pgen holds dosages, and gnomon is reading it as hard calls."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        " {} of {} variants read so far ({:.1}%) carry a dosage track, over {} samples.",
        report.variants_with_dosage,
        report.variants_examined,
        percent(report.variants_with_dosage, report.variants_examined),
        n_samples,
    );
    let _ = writeln!(
        out,
        " Of {} dosage values read, {} ({:.1}%) are not whole allele counts, and a hard",
        report.dosage_values,
        report.fractional_values,
        percent(report.fractional_values, report.dosage_values),
    );
    let _ = writeln!(out, " call cannot carry them:");
    let _ = writeln!(
        out,
        "   {} were never consulted: the record's own hard-call track already had a",
        report.discarded_for_existing_hardcall,
    );
    let _ = writeln!(out, "     call for that sample, and that call wins.");
    let _ = writeln!(
        out,
        "   {} were snapped to the nearest whole count (within ±{tolerance}).",
        report.rounded_to_hardcall,
    );
    let _ = writeln!(
        out,
        "   {} became MISSING: no whole count was within ±{tolerance}.",
        report.dropped_off_tolerance,
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        " Everything computed downstream (allele frequencies, the MAF screen, LD pruning,"
    );
    let _ = writeln!(
        out,
        " the PCA fit itself) is computed from those hard calls. If this fileset is"
    );
    let _ = writeln!(
        out,
        " imputed, the result is a hard-call fit and not the dosage fit it resembles."
    );
    let _ = writeln!(out, "{rule}\n");
    out
}

/// Applies one PGEN dosage entry to the hard-call vector, and records what that
/// cost.
///
/// The hard-call track wins wherever it has a call. PLINK 2 normally writes
/// both tracks, so on an imputed fileset that branch is the common one and the
/// dosage is dropped without the tolerance ever being consulted — the reason
/// `meter` counts it separately. Pass `None` for `meter` on a repeat visit to a
/// record already accounted for, so re-reads cannot inflate the totals.
fn absorb_dosage_entry(
    a1dosage: &mut [u8],
    s: usize,
    v: u16,
    sample_ploidy: Option<&[u8]>,
    meter: Option<&DosageCoercionMeter>,
) {
    let Some(slot) = a1dosage.get_mut(s) else {
        // A sample ID past the end of the cohort: not this variant's genotype,
        // and not this variant's loss either.
        return;
    };
    let had_hardcall = *slot != 255;
    if had_hardcall && meter.is_none() {
        // The hard call stands and nobody is counting what that cost; skip the
        // ploidy lookup entirely, since this is the common case on a re-read.
        return;
    }
    let ploidy = sample_ploidy.and_then(|p| p.get(s)).copied().unwrap_or(2);
    if !had_hardcall && v != 65535 {
        let hc = u16_to_hardcall_biallelic(v, ploidy);
        if hc != 255 {
            *slot = hc;
        }
    }
    if let Some(meter) = meter {
        meter.record_entry(v, ploidy, had_hardcall);
    }
}

fn u16_to_hardcall_biallelic(v: u16, ploidy: u8) -> u8 {
    if v == 65535 || ploidy == 0 {
        return 255;
    }
    if ploidy <= 1 {
        let ds = (v as f32) * (1.0 / 32768.0) * 1.0;
        let candidates = [0.0f32, 1.0];
        let mut best = 255u8;
        let mut best_d = f32::INFINITY;
        for (i, &c) in candidates.iter().enumerate() {
            let d = (ds - c).abs();
            if d < best_d {
                best_d = d;
                best = i as u8;
            }
        }
        if best_d <= DOSAGE_HARDCALL_TOLERANCE {
            match best {
                0 => 0,
                1 => 2,
                _ => 255,
            }
        } else {
            255
        }
    } else {
        let ds = (v as f32) * (1.0 / 32768.0) * 2.0;
        let candidates = [0.0f32, 1.0, 2.0];
        let mut best = 255u8;
        let mut best_d = f32::INFINITY;
        for (i, &c) in candidates.iter().enumerate() {
            let d = (ds - c).abs();
            if d < best_d {
                best_d = d;
                best = i as u8;
            }
        }
        if best_d <= DOSAGE_HARDCALL_TOLERANCE {
            best
        } else {
            255
        }
    }
}

#[cfg(test)]
fn unpack_plink1_block(block: &[u8], dst: &mut [u8], n: usize) {
    let mut i = 0usize;
    for &byte in block {
        for shift in 0..4 {
            if i >= n {
                return;
            }
            let code = (byte >> (2 * shift)) & 0b11;
            dst[i] = match code {
                0b00 => 2,
                0b10 => 1,
                0b11 => 0,
                _ => 255,
            };
            i += 1;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (subset) – validates packing and basic decode scaffolding behavior
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;
    use std::sync::Arc;

    struct VecSource {
        data: Vec<u8>,
    }

    impl VecSource {
        fn new(data: Vec<u8>) -> Self {
            Self { data }
        }
    }

    impl ByteRangeSource for VecSource {
        fn len(&self) -> u64 {
            self.data.len() as u64
        }

        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
            let off = usize::try_from(offset).map_err(|_| ioerr("Offset too large"))?;
            let end = off + dst.len();
            if end > self.data.len() {
                return Err(ioerr("Read past end"));
            }
            dst.copy_from_slice(&self.data[off..end]);
            Ok(())
        }
    }

    struct LineSource {
        lines: Vec<&'static str>,
        index: usize,
        carry: Option<Box<[u8]>>,
    }

    impl LineSource {
        fn new(lines: Vec<&'static str>) -> Self {
            Self {
                lines,
                index: 0,
                carry: None,
            }
        }
    }

    impl TextSource for LineSource {
        fn len(&self) -> Option<u64> {
            Some(self.lines.len() as u64)
        }

        fn next_line(&mut self) -> Result<Option<&[u8]>, PipelineError> {
            let Some(line) = self.lines.get(self.index) else {
                return Ok(None);
            };
            self.index += 1;
            self.carry = Some(line.as_bytes().to_vec().into_boxed_slice());
            Ok(self.carry.as_deref())
        }
    }

    fn encode_varint(mut v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
        out
    }

    fn pack_twobit_values(values: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for chunk in values.chunks(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                byte |= (val & 0b11) << (2 * i);
            }
            out.push(byte);
        }
        out
    }

    fn push_sid(buf: &mut Vec<u8>, sid: u32, bytes: usize) {
        let le = sid.to_le_bytes();
        buf.extend_from_slice(&le[..bytes]);
    }

    #[test]
    fn pvar_plan_rejects_unsorted_positions_within_chromosome() {
        let mut pvar = LineSource::new(vec![
            "#CHROM\tPOS\tID\tREF\tALT",
            "1\t200\tv1\tA\tG",
            "1\t100\tv2\tC\tT",
        ]);

        let err = match VariantPlan::from_pvar(&mut pvar) {
            Ok(_) => panic!("expected unsorted .pvar to fail"),
            Err(err) => err,
        };
        match err {
            PipelineError::Io(message) => {
                assert!(message.contains("not position-sorted"));
                assert!(message.contains("position 100"));
                assert!(message.contains("position 200"));
            }
            other => panic!("expected PipelineError::Io, got {other:?}"),
        }
    }

    #[test]
    fn pack_contract_smoke() {
        let hard = [0u8, 1, 2, 255, 0, 0, 1, 2, 255, 255];
        let mut block = vec![0u8; (hard.len() + 3) / 4];
        VirtualBed::pack_to_block(&mut block, &hard);
        assert_eq!(block[0], 0x4B);
        assert_eq!(block[1], 0x2F);
        assert_eq!(block[2], 0x55);

        let mut round = vec![0u8; hard.len()];
        unpack_plink1_block(&block, &mut round, hard.len());
        assert_eq!(&hard, &round[..]);
    }

    #[test]
    fn dosage_u16_rounding() {
        let vals = [
            (((0.05 / 2.0) * 32768.0) as u16, 0u8),
            (((1.00 / 2.0) * 32768.0) as u16, 1u8),
            (((1.09 / 2.0) * 32768.0) as u16, 1u8),
            (((2.00 / 2.0) * 32768.0) as u16, 2u8),
        ];
        for (v, expect) in vals {
            assert_eq!(u16_to_hardcall_biallelic(v, 2), expect);
        }

        let hap_vals = [
            (((0.02 / 1.0) * 32768.0) as u16, 0u8),
            (((0.50 / 1.0) * 32768.0) as u16, 255u8),
            (((0.95 / 1.0) * 32768.0) as u16, 2u8),
        ];
        for (v, expect) in hap_vals {
            assert_eq!(u16_to_hardcall_biallelic(v, 1), expect);
        }
    }

    /// The tolerance is a contract with the user, not an implementation
    /// detail: a genuine hard call survives untouched, a value just inside the
    /// tolerance becomes a genotype, and a value just outside it becomes
    /// missing rather than being guessed at. Values are written as raw `u16`
    /// so the assertion is about the boundary and not about how some decimal
    /// literal happened to round on the way in.
    #[test]
    fn dosage_hardcall_tolerance_boundary() {
        // Diploid: one allele copy is 16384 units, so 1638/16384 = 0.0999 sits
        // inside the ±0.10 tolerance and 1639/16384 = 0.1000366 sits outside.
        assert_eq!(u16_to_hardcall_biallelic(0, 2), 0);
        assert_eq!(u16_to_hardcall_biallelic(16384, 2), 1);
        assert_eq!(u16_to_hardcall_biallelic(32768, 2), 2);
        assert_eq!(u16_to_hardcall_biallelic(16384 + 1638, 2), 1);
        assert_eq!(u16_to_hardcall_biallelic(16384 + 1639, 2), 255);
        assert_eq!(u16_to_hardcall_biallelic(16384 - 1638, 2), 1);
        assert_eq!(u16_to_hardcall_biallelic(16384 - 1639, 2), 255);
        // Dosage 1.5: the case a hard call has no honest answer for.
        assert_eq!(u16_to_hardcall_biallelic(24576, 2), 255);
        assert_eq!(u16_to_hardcall_biallelic(65535, 2), 255);

        // Haploid: one copy is the whole 32768, so the same absolute tolerance
        // covers half as much of the scale.
        assert_eq!(u16_to_hardcall_biallelic(0, 1), 0);
        assert_eq!(u16_to_hardcall_biallelic(32768, 1), 2);
        assert_eq!(u16_to_hardcall_biallelic(3276, 1), 0);
        assert_eq!(u16_to_hardcall_biallelic(3277, 1), 255);
        assert_eq!(u16_to_hardcall_biallelic(16384, 1), 255);

        // Ploidy 0 (a female sample on chrY) has no genotype to round to.
        assert_eq!(u16_to_hardcall_biallelic(0, 0), 255);
    }

    /// The counters must describe exactly the conversion above, including the
    /// outcome that never reaches the tolerance at all: where the record's own
    /// hard-call track has a call, the dosage is discarded without being
    /// rounded, and that is the bulk of the loss on a PLINK 2-written imputed
    /// fileset.
    #[test]
    fn dosage_meter_classifies_each_outcome() {
        let meter = DosageCoercionMeter::new(4, 1);
        meter.record_entry(16384, 2, false); // exactly 1.0: nothing is lost
        meter.record_entry(16384 + 1638, 2, false); // inside tolerance: rounded
        meter.record_entry(16384 + 1639, 2, false); // outside tolerance: missing
        meter.record_entry(16384 + 1638, 2, true); // hard call wins: discarded
        meter.record_entry(65535, 2, false); // dosage already missing
        meter.record_entry(16384 + 1638, 0, false); // no alleles at this site

        let report = meter.report();
        assert_eq!(report.dosage_values, 4);
        assert_eq!(report.fractional_values, 3);
        assert_eq!(report.rounded_to_hardcall, 1);
        assert_eq!(report.dropped_off_tolerance, 1);
        assert_eq!(report.discarded_for_existing_hardcall, 1);
    }

    /// A record is decoded once per ALT of a multiallelic and once per pass of
    /// a multi-pass fit. The report has to describe the dataset, not the reads,
    /// or a warning about "millions of discarded dosages" would be a statement
    /// about the number of Lanczos iterations.
    #[test]
    fn dosage_meter_counts_each_variant_once() {
        let meter = DosageCoercionMeter::new(2, 100);
        assert!(meter.claim(1));
        assert!(!meter.claim(1));
        assert!(meter.claim(0));
        // Bits are independent across the 64-variant words the set is built
        // from: claiming one variant must not claim its neighbours.
        assert!(meter.claim(64));
        assert!(!meter.claim(64));
        assert!(meter.claim(65));
        // Past the end of the fileset, even though a spare bit exists in the
        // final word: there is no variant there to account for.
        assert!(!meter.claim(100));
        assert!(!meter.claim(127));
    }

    /// End to end through a real dense-dosage record: the hard-call track wins
    /// where it has a call, the dosage fills the gaps it can and drops the ones
    /// it cannot, and every one of those outcomes is counted exactly once no
    /// matter how often the record is re-read.
    #[test]
    fn dense_dosage_record_is_hardcalled_and_counted() {
        let n = 4usize;
        // Categories 0 and 1 are calls; 3 is missing, and only there does the
        // dosage track get a say.
        let mut rec = pack_twobit_values(&[0u8, 1, 3, 3]);
        // 18022 is dosage 1.09997 (inside tolerance), 18023 is 1.10004 (outside).
        for v in [0u16, 18022, 18022, 18023] {
            rec.extend_from_slice(&v.to_le_bytes());
        }

        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(rec.clone()));
        let hdr = PgenHeader {
            mode: PgenMode::Var,
            m_variants: 1,
            n_samples: n as u32,
            fmt_byte: 0,
            // Bit 6 alone: a dense dosage track, unphased, biallelic.
            rec_types: vec![0b0100_0000],
            rec_lens: vec![rec.len() as u32],
            block_offsets: vec![0],
        };
        let mut decoder = PgenDecoder::new(src, hdr, n, 1, vec![1]).unwrap();
        let meter = Arc::clone(&decoder.dosage_meter);

        let mut out = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(0, 1, &mut out, None)
            .unwrap();
        // Sample 1 keeps its hard call of 1 even though its dosage is 1.09997;
        // sample 2 is rounded into one; sample 3 is dropped.
        assert_eq!(out, vec![0, 1, 1, 255]);

        let first = meter.report();
        assert_eq!(first.variants_examined, 1);
        assert_eq!(first.variants_with_dosage, 1);
        assert_eq!(first.dosage_values, 4);
        assert_eq!(first.fractional_values, 3);
        assert_eq!(first.discarded_for_existing_hardcall, 1);
        assert_eq!(first.rounded_to_hardcall, 1);
        assert_eq!(first.dropped_off_tolerance, 1);

        decoder
            .decode_variant_hardcalls(0, 1, &mut out, None)
            .unwrap();
        assert_eq!(out, vec![0, 1, 1, 255]);
        assert_eq!(
            meter.report(),
            first,
            "a re-read must not inflate the counts"
        );
    }

    /// A hard-call fileset must stay silent: the whole point of the counters is
    /// that a microarray `.pgen` is served exactly as before, with nothing to
    /// report and no warning to ignore.
    #[test]
    fn hardcall_only_record_reports_no_dosage_loss() {
        let n = 8usize;
        let rec = pack_twobit_values(&[0u8, 1, 2, 0, 2, 1, 3, 2]);
        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(rec.clone()));
        let hdr = PgenHeader {
            mode: PgenMode::Var,
            m_variants: 1,
            n_samples: n as u32,
            fmt_byte: 0,
            rec_types: vec![0],
            rec_lens: vec![rec.len() as u32],
            block_offsets: vec![0],
        };
        let mut decoder = PgenDecoder::new(src, hdr, n, 1, vec![1]).unwrap();
        let meter = Arc::clone(&decoder.dosage_meter);

        let mut out = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(0, 1, &mut out, None)
            .unwrap();
        assert_eq!(out, vec![0, 1, 2, 0, 2, 1, 255, 2]);

        // The variant is seen; nothing about it is dosage-valued, so every
        // other counter stays at zero and nothing is ever printed.
        let expected = DosageCoercionReport {
            variants_examined: 1,
            ..DosageCoercionReport::default()
        };
        assert_eq!(meter.report(), expected);
    }

    #[test]
    fn difflist_two_group_round_trip() {
        let n_samples = 2_000_000usize;
        let sid_bytes = sample_id_bytes(n_samples);
        assert_eq!(sid_bytes, 3);

        let mut group0_ids = Vec::with_capacity(64);
        let mut group0_deltas = Vec::with_capacity(63);
        let mut cur = 1_000u32;
        group0_ids.push(cur);
        let pattern0 = [1u32, 200, 20_000, 2, 3, 150, 4_000];
        for i in 0..63 {
            let delta = pattern0[i % pattern0.len()];
            cur += delta;
            group0_ids.push(cur);
            group0_deltas.push(delta);
        }

        let mut group1_ids = Vec::with_capacity(15);
        let mut group1_deltas = Vec::with_capacity(14);
        cur = 1_200_000u32;
        group1_ids.push(cur);
        let pattern1 = [
            2u32, 5_000, 180_000, 1, 2, 7_000, 3, 1, 400, 80_000, 2, 1, 1, 2,
        ];
        for &delta in &pattern1 {
            cur += delta;
            group1_ids.push(cur);
            group1_deltas.push(delta);
        }

        assert_eq!(group0_ids.len(), 64);
        assert_eq!(group1_ids.len(), 15);
        assert!(group0_ids.last().unwrap() < group1_ids.first().unwrap());
        assert!(*group1_ids.last().unwrap() < n_samples as u32);

        let expected_ids: Vec<u32> = group0_ids
            .iter()
            .chain(group1_ids.iter())
            .copied()
            .collect();

        let delta_bytes_g0: Vec<u8> = group0_deltas
            .iter()
            .flat_map(|&d| encode_varint(d as u64))
            .collect();
        let delta_bytes_g1: Vec<u8> = group1_deltas
            .iter()
            .flat_map(|&d| encode_varint(d as u64))
            .collect();
        assert!(delta_bytes_g0.len() > 63);
        let sentinel = u8::try_from(delta_bytes_g0.len() - 63).unwrap();

        let mut buf_ids = Vec::new();
        buf_ids.extend_from_slice(&encode_varint(expected_ids.len() as u64));
        push_sid(&mut buf_ids, group0_ids[0], sid_bytes);
        push_sid(&mut buf_ids, group1_ids[0], sid_bytes);
        buf_ids.push(sentinel);
        buf_ids.extend_from_slice(&delta_bytes_g0);
        buf_ids.extend_from_slice(&delta_bytes_g1);

        let mut cursor = 0usize;
        let decoded_ids = difflist_ids(&buf_ids, &mut cursor, n_samples).unwrap();
        assert_eq!(decoded_ids, expected_ids);
        assert_eq!(cursor, buf_ids.len());

        let expected_vals: Vec<u8> = (0..expected_ids.len()).map(|i| (i as u8) & 0b11).collect();
        let mut buf_pairs = Vec::new();
        buf_pairs.extend_from_slice(&encode_varint(expected_ids.len() as u64));
        push_sid(&mut buf_pairs, group0_ids[0], sid_bytes);
        push_sid(&mut buf_pairs, group1_ids[0], sid_bytes);
        buf_pairs.push(sentinel);
        buf_pairs.extend_from_slice(&pack_twobit_values(&expected_vals));
        buf_pairs.extend_from_slice(&delta_bytes_g0);
        buf_pairs.extend_from_slice(&delta_bytes_g1);

        let mut cursor_pairs = 0usize;
        let decoded_pairs = difflist_pairs(&buf_pairs, &mut cursor_pairs, n_samples).unwrap();
        let (ids_again, vals_again): (Vec<_>, Vec<_>) = decoded_pairs.into_iter().unzip();
        assert_eq!(ids_again, expected_ids);
        assert_eq!(vals_again, expected_vals);
        assert_eq!(cursor_pairs, buf_pairs.len());
    }

    #[test]
    fn type3_ld_record_inverts_after_patch() {
        let n = 8usize;
        let anchor_cats = [0u8, 1, 2, 0, 2, 1, 3, 2];
        let rec0 = pack_twobit_values(&anchor_cats);
        assert_eq!(rec0.len(), 2);

        let mut rec1 = Vec::new();
        let difflist_vals = [2u8, 0, 2];
        rec1.extend_from_slice(&encode_varint(difflist_vals.len() as u64));
        rec1.push(0); // first sample ID
        rec1.extend_from_slice(&pack_twobit_values(&difflist_vals));
        rec1.extend_from_slice(&encode_varint(2));
        rec1.extend_from_slice(&encode_varint(3));
        assert_eq!(rec1.len(), 5);

        let mut data = Vec::new();
        data.extend_from_slice(&rec0);
        data.extend_from_slice(&rec1);

        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(data));
        let hdr = PgenHeader {
            mode: PgenMode::Var,
            m_variants: 2,
            n_samples: n as u32,
            fmt_byte: 0,
            block_offsets: vec![0],
            rec_types: vec![0, 3],
            rec_lens: vec![rec0.len() as u32, rec1.len() as u32],
        };
        let mut decoder = PgenDecoder::new(src, hdr, n, 2, vec![1, 1]).unwrap();

        let mut out0 = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(0, 1, &mut out0, None)
            .unwrap();
        assert_eq!(out0, vec![0, 1, 2, 0, 2, 1, 255, 2]);

        let mut out1 = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(1, 1, &mut out1, None)
            .unwrap();
        assert_eq!(out1, vec![0, 1, 2, 2, 0, 0, 255, 0]);

        // The type-3 record's anchor is record 0, decoded and cached on demand.
        assert_eq!(decoder.anchor_idx, Some(0));
        assert_eq!(decoder.anchor_cats, anchor_cats.to_vec());

        // Re-reading in the opposite order must give the same answers: the
        // anchor is resolved from the record table, not from decode history.
        let mut again1 = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(1, 1, &mut again1, None)
            .unwrap();
        assert_eq!(again1, out1);
        let mut again0 = vec![0u8; n];
        decoder
            .decode_variant_hardcalls(0, 1, &mut again0, None)
            .unwrap();
        assert_eq!(again0, out0);
    }

    /// The sparse offset index must agree with a naive walk from the start of
    /// the variant block, including at and around the stride boundaries.
    #[test]
    fn stride_offsets_match_naive_record_walk() {
        let m = OFFSET_STRIDE * 3 + 7;
        let rec_lens: Vec<u32> = (0..m).map(|i| 5 + (i as u32 * 7) % 23).collect();
        let hdr = PgenHeader {
            mode: PgenMode::Var,
            m_variants: m as u32,
            n_samples: 4,
            fmt_byte: 0,
            block_offsets: vec![1000],
            rec_types: vec![0; m],
            rec_lens: rec_lens.clone(),
        };
        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(vec![]));
        let decoder = PgenDecoder::new(src, hdr, 4, m, vec![1; m]).unwrap();

        let mut expected = 1000u64;
        for (idx, len) in rec_lens.iter().enumerate() {
            let (got, got_len, _) = decoder.record_offset_len(idx).unwrap();
            assert_eq!(got, expected, "offset mismatch at record {idx}");
            assert_eq!(got_len, *len as usize);
            expected += *len as u64;
        }
    }

    /// An LD anchor is the nearest preceding non-LD record, regardless of how
    /// many LD records sit between it and the target.
    #[test]
    fn ld_anchor_is_nearest_preceding_non_ld_record() {
        let rec_types = vec![0u8, 2, 3, 2, 1, 2, 2, 4, 2];
        let m = rec_types.len();
        let hdr = PgenHeader {
            mode: PgenMode::Var,
            m_variants: m as u32,
            n_samples: 4,
            fmt_byte: 0,
            block_offsets: vec![0],
            rec_types,
            rec_lens: vec![2; m],
        };
        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(vec![]));
        let decoder = PgenDecoder::new(src, hdr, 4, m, vec![1; m]).unwrap();

        for (target, want) in [(1, 0), (2, 0), (3, 0), (5, 4), (6, 4), (8, 7)] {
            assert_eq!(
                decoder.ld_anchor_index(target).unwrap(),
                want,
                "anchor for record {target}"
            );
        }
        // Record 0 begins the block, so it has nothing to diff against.
        assert!(decoder.ld_anchor_index(0).is_err());
    }

    #[test]
    fn multiallelic_alt1_patches_apply() {
        let record = vec![0u8, 0x01, 0x01, 0x01, 0x0A];
        let mut cats = vec![1u8, 2, 2];
        let mut out = vec![255u8; 3];
        let mut cursor = 0usize;
        apply_multiallelic_and_project(&record, &mut cursor, 3, &mut cats, 3, 1, &mut out).unwrap();
        assert_eq!(out, vec![0, 0, 2]);
        assert_eq!(cursor, record.len());
    }

    #[test]
    fn fixed_width_offsets_include_header_components() {
        let n_samples = 10usize;
        let m_variants = 10u32;
        let rec_len = (n_samples + 3) / 4;

        let hdr_plain = PgenHeader {
            mode: PgenMode::FixHard,
            m_variants,
            n_samples: n_samples as u32,
            fmt_byte: 0,
            block_offsets: vec![],
            rec_types: vec![],
            rec_lens: vec![],
        };
        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(vec![]));
        let decoder_plain = PgenDecoder::new(
            Arc::clone(&src),
            hdr_plain,
            n_samples,
            m_variants as usize,
            vec![0; m_variants as usize],
        )
        .unwrap();
        let (off0, len0, ty0) = decoder_plain.record_offset_len(0).unwrap();
        assert_eq!(off0, 12 + 1);
        assert_eq!(len0, rec_len);
        assert_eq!(ty0, 0);
        let (off1, _, _) = decoder_plain.record_offset_len(1).unwrap();
        assert_eq!(off1, 12 + 1 + rec_len as u64);

        let hdr_ref = PgenHeader {
            mode: PgenMode::FixHard,
            m_variants,
            n_samples: n_samples as u32,
            fmt_byte: 0b1100_0000,
            block_offsets: vec![],
            rec_types: vec![],
            rec_lens: vec![],
        };
        let decoder_ref = PgenDecoder::new(
            src,
            hdr_ref,
            n_samples,
            m_variants as usize,
            vec![0; m_variants as usize],
        )
        .unwrap();
        let (off0_ref, len_ref, _) = decoder_ref.record_offset_len(0).unwrap();
        let expected_base = 12 + 1 + ((m_variants as u64 + 7) / 8);
        assert_eq!(off0_ref, expected_base);
        assert_eq!(len_ref, rec_len);
        let (off2_ref, _, _) = decoder_ref.record_offset_len(2).unwrap();
        assert_eq!(off2_ref, expected_base + (rec_len as u64) * 2);
    }

    #[test]
    fn parse_fixhard_reads_fmt_byte() {
        let m_variants = 3u32;
        let n_samples = 8u32;
        let fmt = 0b1100_0000u8;
        let mut data = vec![0u8; 12];
        data[0] = 0x6c;
        data[1] = 0x1b;
        data[2] = 0x02;
        data[3..7].copy_from_slice(&m_variants.to_le_bytes());
        data[7..11].copy_from_slice(&n_samples.to_le_bytes());
        data[11] = fmt;
        let src = VecSource::new(data);
        let header = PgenHeader::parse(&src).unwrap();
        assert_eq!(header.mode, PgenMode::FixHard);
        assert_eq!(header.m_variants, m_variants);
        assert_eq!(header.n_samples, n_samples);
        assert_eq!(header.fmt_byte, fmt);
    }

    #[test]
    fn fam_row_uses_sid_when_iid_missing() {
        let fields = ["", "unused", "sid123", "1"];
        let cols = PsamColumns {
            fid_idx: Some(0),
            iid_idx: None,
            pat_idx: None,
            mat_idx: None,
            sex_idx: Some(3),
            pheno_idx: None,
            pheno1_idx: None,
            sid_idx: Some(2),
        };
        let fam = FamRow::from_fields(&fields, &cols);
        assert_eq!(fam.iid, "sid123");
        assert_eq!(fam.fid, "sid123");
        assert_eq!(fam.sex, "1");
    }

    #[test]
    fn fam_row_defaults_fid_to_resolved_iid() {
        let fields = ["iid789", ""];
        let cols = PsamColumns {
            fid_idx: Some(1),
            iid_idx: Some(0),
            ..PsamColumns::default()
        };
        let fam = FamRow::from_fields(&fields, &cols);
        assert_eq!(fam.iid, "iid789");
        assert_eq!(fam.fid, "iid789");
    }

    #[test]
    fn parse_sex_token_supports_common_words() {
        assert_eq!(parse_sex_token("male"), 1);
        assert_eq!(parse_sex_token("FEMALE"), 2);
        assert_eq!(parse_sex_token("Unknown"), 0);
        assert_eq!(parse_sex_token("UNK"), 0);
    }

    #[test]
    fn coerce_pheno_token_handles_missing_values() {
        assert_eq!(coerce_pheno_token("   "), "-9");
        assert_eq!(coerce_pheno_token("NaN"), "-9");
        assert_eq!(coerce_pheno_token("1.5"), "1.5");
        assert_eq!(coerce_pheno_token("nonsense"), "-9");
    }
}
