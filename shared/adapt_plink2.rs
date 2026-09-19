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
//! - Ploidy: no call depends on the `.psam` SEX column. Stored hard calls pass
//!   through as they are, heterozygous haploid ones included, as plink2
//!   `--make-bed` writes them without `--set-invalid-haploid-missing`, and a
//!   dosage-only entry rounds to a hard call on the diploid scale for every
//!   sample. So a sex check or a score never reads a call its label rewrote.
//! - `.bed` encoding: exact PLINK 1.9 2-bit codes (`00` hom ALT, `01`
//!   missing, `10` het, `11` hom REF; least-significant bit first within
//!   each byte).
//! - Split IDs: if `ID != "."` → `ID__ALT=<ALT>`; else use
//!   `chr:pos:ref:alt`.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
#[cfg(not(unix))]
use std::io::{Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::str;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crossbeam_queue::SegQueue;

use crate::files::{
    // Traits
    ByteRangeSource,
    TextSource,
    // Helpers
    open_text_source,
};
/// Bring in your crate-local traits and error type.
/// These are expected to already exist (per your provided infrastructure).
use crate::pipeline_error::PipelineError;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public entrypoints
////////////////////////////////////////////////////////////////////////////////////////////////////

/// A handle which exposes virtual PLINK-1.9 streams backed by PLINK-2.0 inputs.
pub struct VirtualPlink19 {
    /// Random-access virtual `.bed` (PLINK-1.9 bytes).
    pub bed: Arc<dyn ByteRangeSource>,
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
    build: GenomeBuild,
) -> Result<VirtualPlink19, PipelineError> {
    let mut psam_for_plan = open_text_source(psam_path)?;
    let pgen = Arc::new(LocalFileByteRangeSource::open(pgen_path)?);
    let factory_path = pvar_path.to_path_buf();
    let pvar: PvarFactory = Arc::new(move || open_text_source(&factory_path));

    open_virtual_plink19_with_local_pvar(pgen, pvar, Some(pvar_path), &mut *psam_for_plan, build)
}

/// Bytes of `.pvar` data one chunk of the parallel variant-plan scan takes.
const PLAN_SCAN_CHUNK_BYTES: usize = 4 << 20;

/// The variant plan of the `.pvar` that `pvar` opens, its lines read in pieces at
/// once by one reader: the mapped file's pieces when `local_pvar` names it, and
/// otherwise pieces of the lines the stream gives, a batch at a time.
fn plan_for(pvar: &PvarFactory, local_pvar: Option<&Path>) -> Result<VariantPlan, PipelineError> {
    match local_pvar {
        Some(path) => VariantPlan::from_local_pvar(path, PLAN_SCAN_CHUNK_BYTES),
        None => VariantPlan::from_pvar(&mut *pvar()?),
    }
}

/// `text` split into pieces of whole lines of about `chunk_bytes`, every piece but
/// the last ending just after a newline.
fn newline_pieces(text: &[u8], chunk_bytes: usize) -> Vec<&[u8]> {
    let mut pieces = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let search_from = start + chunk_bytes.max(1);
        let end = if search_from >= text.len() {
            text.len()
        } else {
            memchr::memchr(b'\n', &text[search_from..])
                .map_or(text.len(), |offset| search_from + offset + 1)
        };
        pieces.push(&text[start..end]);
        start = end;
    }
    pieces
}

/// Open from caller-provided sources. Callers may pass a custom/remote-capable
/// `ByteRangeSource` for `.pgen` and a `TextSource` for `.psam`.
///
/// `.pvar` is supplied as a *factory* rather than a stream: it is read once to
/// build the variant plan, and reopened whenever the virtual `.bim` is walked,
/// so its rows never have to be held in memory.
///
/// The genome build is accepted for callers that pass `--build`; no decoded call
/// depends on it, since no call depends on ploidy or sex.
pub fn open_virtual_plink19_from_sources(
    pgen: Arc<dyn ByteRangeSource>,
    pvar: PvarFactory,
    psam_for_plan: &mut dyn TextSource,
    build: GenomeBuild,
) -> Result<VirtualPlink19, PipelineError> {
    open_virtual_plink19_with_local_pvar(pgen, pvar, None, psam_for_plan, build)
}

/// [`open_virtual_plink19_from_sources`], given the local path of the `.pvar`
/// the factory opens when there is one, so that the variant plan can come from
/// a parallel scan of the mapped file; see [`plan_for`].
pub(crate) fn open_virtual_plink19_with_local_pvar(
    pgen: Arc<dyn ByteRangeSource>,
    pvar: PvarFactory,
    local_pvar: Option<&Path>,
    psam_for_plan: &mut dyn TextSource,
    _build: GenomeBuild,
) -> Result<VirtualPlink19, PipelineError> {
    let header = PgenHeader::parse(&*pgen)?;
    let psam_info = PsamInfo::from_psam(psam_for_plan)?;
    let plan = plan_for(&pvar, local_pvar)?;

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
                let bed: Arc<dyn ByteRangeSource> =
                    Arc::new(VirtualBed::new(decoder, plan.clone(), psam_info.n_samples));
                (bed, Some(meter))
            }
        };

    Ok(VirtualPlink19 {
        bed: bed_source,
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
            fam_rows.push(fam_row);
        }

        if columns.is_none() {
            let tokens = header_tokens
                .ok_or_else(|| PipelineError::Io("Missing .psam header (#FID/#IID…)".into()))?;
            PsamColumns::from_header(&tokens)?;
        }

        Ok(Self {
            n_samples: fam_rows.len(),
            fam_rows,
        })
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
#[derive(Clone)]
struct VariantPlan {
    /// Total input variants before splitting (to sanity check decoder bounds).
    in_variants: usize,
    /// Total emitted variants after splitting.
    out_variants: usize,
    /// Dense mapping: out_idx → (in_idx, alt_ordinal_1based).
    out_to_in: Vec<(u32, u16)>,
    /// ALT allele count per input variant.
    alts_per_in: Vec<u16>,
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
    /// The plan of the `.pvar` that `pvar` streams, read in batches of pieces of
    /// about [`PLAN_SCAN_CHUNK_BYTES`] of lines, a batch as wide as the rayon pool.
    fn from_pvar(pvar: &mut dyn TextSource) -> Result<Self, PipelineError> {
        Self::from_pvar_in_pieces(pvar, PLAN_SCAN_CHUNK_BYTES)
    }

    fn from_pvar_in_pieces(
        pvar: &mut dyn TextSource,
        piece_bytes: usize,
    ) -> Result<Self, PipelineError> {
        let batch = rayon::current_num_threads().max(1);
        let mut builder = PlanBuilder::default();
        loop {
            let mut pieces = Vec::with_capacity(batch);
            // A line the source fails to give ends the read, after the plan has read
            // every line before it, as a reader of one line after another would.
            let mut stop: Option<Result<(), PipelineError>> = None;
            while pieces.len() < batch && stop.is_none() {
                let mut lines = PvarLines::default();
                while lines.bytes.len() < piece_bytes.max(1) {
                    match pvar.next_line() {
                        Ok(Some(line)) => lines.push(line),
                        Ok(None) => {
                            stop = Some(Ok(()));
                            break;
                        }
                        Err(error) => {
                            stop = Some(Err(error));
                            break;
                        }
                    }
                }
                pieces.push(PvarPiece::Lines(lines));
            }
            builder.add(&pieces)?;
            match stop {
                Some(Ok(())) => return builder.finish(),
                Some(Err(error)) => return Err(error),
                None => {}
            }
        }
    }

    /// The plan of the local `.pvar` at `pvar_path`, its lines read at once in
    /// newline-aligned pieces of about `chunk_bytes` of the mapped file.
    fn from_local_pvar(pvar_path: &Path, chunk_bytes: usize) -> Result<Self, PipelineError> {
        let file = File::open(pvar_path)
            .map_err(|e| PipelineError::Io(format!("Opening {}: {e}", pvar_path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| PipelineError::Io(format!("Metadata for {}: {e}", pvar_path.display())))?
            .len();
        let mut builder = PlanBuilder::default();
        if len > 0 {
            // SAFETY: the map is read-only and lives only as long as this read. The
            // file must not be truncated while it is mapped.
            let map = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| PipelineError::Io(format!("Mapping {}: {e}", pvar_path.display())))?;
            let pieces: Vec<PvarPiece<'_>> = newline_pieces(&map, chunk_bytes)
                .into_iter()
                .map(PvarPiece::Text)
                .collect();
            builder.add(&pieces)?;
        }
        builder.finish()
    }

    #[inline]
    fn mapping(&self, out_idx: usize) -> Option<(u32, u16)> {
        self.out_to_in.get(out_idx).copied()
    }

    #[inline]
    fn alt_count_of_in(&self, in_idx: u32) -> u16 {
        self.alts_per_in.get(in_idx as usize).copied().unwrap_or(0)
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
pub enum GenomeBuild {
    Grch37,
    Grch38,
}

impl GenomeBuild {
    pub fn parse(value: &str) -> Result<Self, PipelineError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "37" | "grch37" | "hg19" => Ok(Self::Grch37),
            "38" | "grch38" | "hg38" => Ok(Self::Grch38),
            _ => Err(PipelineError::Io(format!(
                "Unsupported genome build '{value}'; expected 37 or 38"
            ))),
        }
    }
}

#[cfg(test)]
fn normalize_chrom(raw: &str) -> String {
    let mut chrom = String::new();
    normalize_chrom_into(raw, &mut chrom);
    chrom
}

/// `normalize_chrom` into a reused buffer.
fn normalize_chrom_into(raw: &str, chrom: &mut String) {
    chrom.clear();
    let trimmed = raw.trim();
    let mut body = trimmed;
    if trimmed
        .get(..3)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("chr"))
    {
        body = &trimmed[3..];
    }
    chrom.push_str(body);
    chrom.make_ascii_uppercase();
    if chrom == "M" {
        chrom.push('T');
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
fn write_bim_row(
    row: &mut Vec<u8>,
    chrom: &str,
    id: &str,
    pos: u64,
    refa: &str,
    alt: &str,
    split: bool,
) {
    let has_id = id != "." && !id.is_empty();
    row.extend_from_slice(chrom.as_bytes());
    row.push(b'\t');
    match (has_id, split) {
        (true, false) => row.extend_from_slice(id.as_bytes()),
        (true, true) => {
            row.extend_from_slice(id.as_bytes());
            row.extend_from_slice(b"__ALT=");
            row.extend_from_slice(alt.as_bytes());
        }
        (false, _) => {
            row.extend_from_slice(chrom.as_bytes());
            row.push(b':');
            push_decimal(row, pos);
            row.push(b':');
            row.extend_from_slice(refa.as_bytes());
            row.push(b':');
            row.extend_from_slice(alt.as_bytes());
        }
    }
    row.extend_from_slice(b"\t0\t");
    push_decimal(row, pos);
    row.push(b'\t');
    row.extend_from_slice(alt.as_bytes());
    row.push(b'\t');
    row.extend_from_slice(refa.as_bytes());
}

/// Appends `value` in decimal, as `Display` writes it.
fn push_decimal(row: &mut Vec<u8>, mut value: u64) {
    let mut digits = [0u8; 20];
    let mut start = digits.len();
    loop {
        start -= 1;
        digits[start] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    row.extend_from_slice(&digits[start..]);
}

/// Whether `byte` is whitespace to `str::trim` and `str::split_whitespace`, for
/// an ASCII byte: space, tab, line feed, vertical tab, form feed and carriage
/// return, and no other.
#[inline]
fn is_ascii_text_whitespace(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// `text.trim()`, by bytes when `text` is ASCII.
fn trim_text(text: &str) -> &str {
    if !text.is_ascii() {
        return text.trim();
    }
    let bytes = text.as_bytes();
    let start = bytes
        .iter()
        .position(|&byte| !is_ascii_text_whitespace(byte))
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|&byte| !is_ascii_text_whitespace(byte))
        .map_or(start, |last| last + 1);
    &text[start..end]
}

/// The columns of one `.pvar` data line that a plan or a virtual `.bim` row
/// needs, split without collecting every field.
struct PvarFields<'a> {
    chrom: Option<&'a str>,
    id: Option<&'a str>,
    pos: Option<&'a str>,
    refa: Option<&'a str>,
    alt: Option<&'a str>,
}

impl<'a> PvarFields<'a> {
    fn split(line: &'a str, cols: PvarCols) -> Self {
        let mut fields = Self {
            chrom: None,
            id: None,
            pos: None,
            refa: None,
            alt: None,
        };
        let last = cols
            .chrom
            .max(cols.id)
            .max(cols.pos)
            .max(cols.refa)
            .max(cols.alt);
        let mut keep = |column: usize, field: &'a str| {
            if column == cols.chrom {
                fields.chrom = Some(field);
            }
            if column == cols.id {
                fields.id = Some(field);
            }
            if column == cols.pos {
                fields.pos = Some(field);
            }
            if column == cols.refa {
                fields.refa = Some(field);
            }
            if column == cols.alt {
                fields.alt = Some(field);
            }
        };
        if line.is_ascii() {
            // The fields `split_whitespace` gives, split on the same bytes.
            let bytes = line.as_bytes();
            let mut at = 0;
            for column in 0..=last {
                while at < bytes.len() && is_ascii_text_whitespace(bytes[at]) {
                    at += 1;
                }
                if at == bytes.len() {
                    break;
                }
                let start = at;
                while at < bytes.len() && !is_ascii_text_whitespace(bytes[at]) {
                    at += 1;
                }
                keep(column, &line[start..at]);
            }
        } else {
            for (column, field) in line.split_whitespace().take(last + 1).enumerate() {
                keep(column, field);
            }
        }
        fields
    }
}

/// Streaming virtual `.bim`: re-reads the `.pvar` and emits the split rows on
/// the fly.
///
/// Materializing the rows instead would cost roughly 140 bytes per output
/// variant — about 1.2 GB for a WGS chr1 at 8.45M variants, and ~10 GB if a
/// genome-wide set of per-chromosome filesets were opened at once. Nothing
/// downstream needs random access to the rows, only a single forward pass, so
/// they are generated on demand and never retained. One `.pvar` line's rows are
/// written end to end into a reused buffer, so a pass allocates nothing per row.
struct StreamingVirtualBim {
    pvar: Box<dyn TextSource>,
    lines: VirtualBimLines,
    /// The next row of the current line's rows to return.
    next_row: usize,
    total: Option<u64>,
}

impl StreamingVirtualBim {
    fn new(pvar: Box<dyn TextSource>, total: Option<u64>) -> Self {
        Self {
            pvar,
            lines: VirtualBimLines::new(PvarLayout::default()),
            next_row: 0,
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
            if let Some(row) = self.lines.row(self.next_row) {
                self.next_row += 1;
                return Ok(Some(row));
            }
            let Some(line) = self.pvar.next_line()? else {
                return Ok(None);
            };
            self.next_row = 0;
            self.lines.render(line)?;
        }
    }
}

/// The column layout of a `.pvar` read line by line in file order: the last header
/// line's, or while no header line has set one, the first data line's field count.
#[derive(Clone, Copy, Default)]
pub(crate) struct PvarLayout {
    cols: Option<PvarCols>,
}

/// A `.pvar` line as every reader of the file reads it.
enum PvarLine<'l> {
    /// A data line, trimmed, with the columns that read it.
    Data(&'l str, PvarCols),
    /// A header line, which sets the layout of the lines after it.
    Header,
    /// A comment or blank line.
    Other,
}

impl PvarLayout {
    /// Reads one line, without its line ending, in this layout, which a header line
    /// replaces and a first data line sets.
    fn read<'l>(&mut self, line: &'l [u8]) -> Result<PvarLine<'l>, PipelineError> {
        let s = str::from_utf8(line)
            .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .pvar: {e}")))?;
        let trimmed = trim_text(s);
        if trimmed.is_empty() || trimmed.starts_with("##") {
            return Ok(PvarLine::Other);
        }
        if trimmed.starts_with('#') {
            self.cols = Some(PvarCols::from_header_line(trimmed)?);
            return Ok(PvarLine::Header);
        }
        let cols = match self.cols {
            Some(cols) => cols,
            None => {
                let derived = PvarCols::from_headerless(trimmed.split_whitespace().count())?;
                self.cols = Some(derived);
                derived
            }
        };
        Ok(PvarLine::Data(trimmed, cols))
    }

    /// The layout after a piece of lines that [`piece_layouts`] summarized, read from
    /// this one.
    fn after(self, (last_header, from_unset): (Option<PvarCols>, Option<PvarCols>)) -> Self {
        Self {
            cols: match self.cols {
                Some(set) => Some(last_header.unwrap_or(set)),
                None => from_unset,
            },
        }
    }
}

/// Whole `.pvar` lines in file order, for a parallel read: a slice of the file
/// ending just after a newline, or the lines a text source has read.
enum PvarPiece<'a> {
    Text(&'a [u8]),
    Lines(PvarLines),
}

/// Lines a text source has read, end to end.
#[derive(Default)]
struct PvarLines {
    bytes: Vec<u8>,
    ends: Vec<usize>,
}

impl PvarLines {
    fn push(&mut self, line: &[u8]) {
        self.bytes.extend_from_slice(line);
        self.ends.push(self.bytes.len());
    }
}

impl PvarPiece<'_> {
    /// The piece's lines, as the text source that reads the file gives them.
    fn lines(&self) -> Box<dyn Iterator<Item = &[u8]> + '_> {
        match self {
            Self::Text(text) => Box::new(text_lines(text)),
            Self::Lines(lines) => Box::new(
                std::iter::once(0)
                    .chain(lines.ends.iter().copied())
                    .zip(lines.ends.iter().copied())
                    .map(|(start, end)| &lines.bytes[start..end]),
            ),
        }
    }
}

/// The lines of `text` as a local text source reads them: split after each
/// newline, each without its newline and one carriage return before it.
fn text_lines(text: &[u8]) -> impl Iterator<Item = &[u8]> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let (line, after) = match memchr::memchr(b'\n', rest) {
            Some(newline) => (&rest[..newline], &rest[newline + 1..]),
            None => (rest, &rest[rest.len()..]),
        };
        rest = after;
        Some(line.strip_suffix(b"\r").unwrap_or(line))
    })
}

/// Reads `pieces` of one `.pvar`, in file order, at once on the rayon pool: `read`
/// gets each piece with the layout the lines before it set, from `layout` at the
/// first piece. Returns what each piece read, in order, and the layout after the
/// last piece.
///
/// Only a header line, or while no layout is set a data line, sets a layout, so
/// each piece is first read for those lines alone, and the layout each piece
/// starts with follows from its predecessors' in one pass over the pieces.
fn for_each_pvar_piece<S: Send>(
    pieces: &[PvarPiece<'_>],
    layout: PvarLayout,
    read: impl Fn(PvarLayout, &PvarPiece<'_>) -> S + Sync,
) -> (Vec<S>, PvarLayout) {
    use rayon::prelude::*;

    let summaries: Vec<(Option<PvarCols>, Option<PvarCols>)> =
        pieces.par_iter().map(piece_layouts).collect();
    let mut after = layout;
    let starts: Vec<PvarLayout> = summaries
        .into_iter()
        .map(|summary| {
            let start = after;
            after = start.after(summary);
            start
        })
        .collect();
    let read = pieces
        .par_iter()
        .zip(starts)
        .map(|(piece, start)| read(start, piece))
        .collect();
    (read, after)
}

/// For one piece of whole `.pvar` lines: the layout of its last header line that
/// parses, and the layout [`PvarLayout::read`] holds after the piece when it starts
/// with none set.
fn piece_layouts(piece: &PvarPiece<'_>) -> (Option<PvarCols>, Option<PvarCols>) {
    let mut last_header = None;
    let mut from_unset = None;
    for line in piece.lines() {
        // `str::trim` removes exactly these six ASCII characters, so a line whose
        // first other byte is ASCII and not `#` is a data line.
        let lead = line
            .iter()
            .find(|&&byte| !matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c));
        let maybe_header = lead.is_none_or(|&byte| byte == b'#' || !byte.is_ascii());
        if !maybe_header && from_unset.is_some() {
            continue;
        }
        let mut layout = PvarLayout { cols: from_unset };
        if let Ok(read) = layout.read(line) {
            if matches!(read, PvarLine::Header) {
                last_header = layout.cols;
            }
            from_unset = layout.cols;
        }
    }
    (last_header, from_unset)
}

/// Renders `.pvar` lines as the rows of the virtual `.bim`, one line at a time in
/// file order, with the layout the lines before it set.
pub(crate) struct VirtualBimLines {
    layout: PvarLayout,
    /// The current `.pvar` line's rows, one per ALT in ALT order, end to end.
    rows: Vec<u8>,
    /// Where each row of `rows` ends.
    row_ends: Vec<usize>,
    /// The current line's normalized chromosome.
    chrom: String,
}

impl VirtualBimLines {
    fn new(layout: PvarLayout) -> Self {
        Self {
            layout,
            rows: Vec::new(),
            row_ends: Vec::new(),
            chrom: String::new(),
        }
    }

    /// Renders one `.pvar` line, without its line ending, as its rows: none for a
    /// header, comment or blank line, and none when the line is refused.
    pub(crate) fn render(&mut self, line: &[u8]) -> Result<(), PipelineError> {
        self.rows.clear();
        self.row_ends.clear();
        let PvarLine::Data(trimmed, cols) = self.layout.read(line)? else {
            return Ok(());
        };
        let fields = PvarFields::split(trimmed, cols);

        normalize_chrom_into(
            fields
                .chrom
                .ok_or_else(|| ioerr(".pvar missing CHROM column"))?,
            &mut self.chrom,
        );
        let pos = fields
            .pos
            .ok_or_else(|| ioerr(".pvar missing POS column"))?
            .parse::<u64>()
            .map_err(|_| ioerr("Invalid POS in .pvar (expected integer)"))?;
        let id = fields.id.ok_or_else(|| ioerr(".pvar missing ID column"))?;
        let refa = fields
            .refa
            .ok_or_else(|| ioerr(".pvar missing REF column"))?;
        let alt_raw = fields
            .alt
            .ok_or_else(|| ioerr(".pvar missing ALT column"))?;

        let alts = || {
            alt_raw
                .split(',')
                .map(str::trim)
                .filter(|a| !a.is_empty() && *a != ".")
        };
        let split = alts().nth(1).is_some();
        // Rows in ALT order, matching the variant order the plan assigned
        // during the indexing pass.
        for alt in alts() {
            write_bim_row(&mut self.rows, &self.chrom, id, pos, refa, alt, split);
            self.row_ends.push(self.rows.len());
        }
        Ok(())
    }

    /// Row `index` of the line rendered last, if it has that many.
    pub(crate) fn row(&self, index: usize) -> Option<&[u8]> {
        let end = *self.row_ends.get(index)?;
        let start = index
            .checked_sub(1)
            .map_or(0, |before| self.row_ends[before]);
        Some(&self.rows[start..end])
    }
}

/// Renders the `.pvar` text split into `pieces` of whole lines, in file order, as
/// the virtual `.bim` a pass over the whole file renders it, and returns one state
/// per piece. Pieces are rendered at once on the rayon pool; within a piece,
/// `each_line` is given every line in order, as the renderer holding its rows or
/// as its error, with the layout the lines before it set.
pub(crate) fn render_virtual_bim_pieces<S: Default + Send>(
    pieces: &[&[u8]],
    each_line: impl Fn(&mut S, Result<&VirtualBimLines, PipelineError>) + Sync,
) -> Vec<S> {
    let pieces: Vec<PvarPiece<'_>> = pieces.iter().map(|&text| PvarPiece::Text(text)).collect();
    let (rendered, _) = for_each_pvar_piece(&pieces, PvarLayout::default(), |layout, piece| {
        let mut lines = VirtualBimLines::new(layout);
        let mut state = S::default();
        for line in piece.lines() {
            let rendered = lines.render(line).map(|()| &lines);
            each_line(&mut state, rendered);
        }
        state
    });
    rendered
}

/// One piece of a `.pvar` read for the variant plan: each record's ALT count, the
/// chromosome runs the records form, and the line that ends the plan, if one does.
#[derive(Default)]
struct PlanPiece {
    alts: Vec<u16>,
    runs: Vec<PlanRun>,
    end: Option<PlanEnd>,
}

/// Records of one chromosome in a row, positions ascending: the first and last
/// record, as indices into their piece, and their positions.
struct PlanRun {
    chrom: String,
    first_record: usize,
    first: u64,
    last_record: usize,
    last: u64,
}

/// Where a piece's records stop: a line the plan refuses, or a record placed before
/// the one ahead of it on its chromosome, which the sort check refuses in its own
/// words once the records before it are observed.
enum PlanEnd {
    Refused(PipelineError),
    Descends { record: usize, position: u64 },
}

/// Reads one piece of a `.pvar` for the variant plan, line by line from `layout`, as
/// [`VariantPlan`] requires of every record: its columns present, a positive
/// integer position, and positions that do not descend on a chromosome.
fn plan_piece(mut layout: PvarLayout, piece: &PvarPiece<'_>) -> PlanPiece {
    let mut read = PlanPiece::default();
    // The last raw label, and the label it normalizes to.
    let mut label = None;
    let mut chrom = String::new();
    for line in piece.lines() {
        let (raw_chrom, position, alt_count) = match plan_record(&mut layout, line) {
            Ok(Some(record)) => record,
            Ok(None) => continue,
            Err(error) => {
                read.end = Some(PlanEnd::Refused(error));
                break;
            }
        };
        if label != Some(raw_chrom) {
            label = Some(raw_chrom);
            normalize_chrom_into(raw_chrom, &mut chrom);
        }
        let record = read.alts.len();
        match read.runs.last_mut() {
            Some(run) if run.chrom == chrom => {
                if position < run.last {
                    read.end = Some(PlanEnd::Descends { record, position });
                    break;
                }
                run.last_record = record;
                run.last = position;
            }
            _ => read.runs.push(PlanRun {
                chrom: chrom.clone(),
                first_record: record,
                first: position,
                last_record: record,
                last: position,
            }),
        }
        read.alts.push(alt_count);
    }
    read
}

/// One `.pvar` line as a plan record: its chromosome as written, its position and its ALT
/// count, or `None` for a line that is not a record.
fn plan_record<'l>(
    layout: &mut PvarLayout,
    line: &'l [u8],
) -> Result<Option<(&'l str, u64, u16)>, PipelineError> {
    let PvarLine::Data(trimmed, cols) = layout.read(line)? else {
        return Ok(None);
    };
    let fields = PvarFields::split(trimmed, cols);
    let chrom_raw = fields
        .chrom
        .ok_or_else(|| ioerr(".pvar missing CHROM column"))?;
    let pos_raw = fields
        .pos
        .ok_or_else(|| ioerr(".pvar missing POS column"))?;
    // ID and REF are validated by presence here; their values are only
    // needed when the virtual .bim rows are streamed.
    fields.id.ok_or_else(|| ioerr(".pvar missing ID column"))?;
    fields
        .refa
        .ok_or_else(|| ioerr(".pvar missing REF column"))?;
    let alt_raw = fields
        .alt
        .ok_or_else(|| ioerr(".pvar missing ALT column"))?;

    let pos = pos_raw
        .parse::<u64>()
        .map_err(|_| ioerr("Invalid POS in .pvar (expected integer)"))?;
    if pos == 0 {
        return Err(ioerr(".pvar POS must be positive"));
    }
    // Symbolic ALTs (`<INS>`, `<DEL:ME:ALU>`, `*`, breakends) stay as
    // ordinary allele codes, as plink2's own .bim export and the VCF
    // readers keep them: dropping or rejecting them would make a PGEN
    // disagree with the same data read as BED or VCF.
    let alt_count = alt_raw
        .split(',')
        .map(|a| a.trim())
        .filter(|a| !a.is_empty() && *a != ".")
        .count();
    Ok(Some((chrom_raw, pos, alt_count as u16)))
}

/// The variant plan of a `.pvar` read piece by piece, in file order, each batch of
/// pieces at once: [`plan_piece`] reads a piece's records, and joining the pieces
/// in order applies the sort check to every record the pieces read, as a reader
/// of one line after another applies it, and stops at the first record or line
/// it refuses.
#[derive(Default)]
struct PlanBuilder {
    layout: PvarLayout,
    sorted_positions: PvarPositionSortState,
    alts_per_in: Vec<u16>,
}

impl PlanBuilder {
    fn add(&mut self, pieces: &[PvarPiece<'_>]) -> Result<(), PipelineError> {
        let (read, layout) = for_each_pvar_piece(pieces, self.layout, plan_piece);
        self.layout = layout;
        for piece in read {
            let before = self.alts_per_in.len();
            // Positions within a run ascend already; a run's first record is what
            // the check compares with the records before it, and its last is what
            // it compares the records after it with.
            for run in &piece.runs {
                self.sorted_positions.observe(
                    &run.chrom,
                    run.first,
                    before + run.first_record + 1,
                )?;
                self.sorted_positions.observe(
                    &run.chrom,
                    run.last,
                    before + run.last_record + 1,
                )?;
            }
            match piece.end {
                Some(PlanEnd::Refused(error)) => return Err(error),
                Some(PlanEnd::Descends { record, position }) => {
                    let chrom = &piece.runs.last().expect("a descent follows a run").chrom;
                    return Err(self
                        .sorted_positions
                        .observe(chrom, position, before + record + 1)
                        .expect_err("a position below the one before it on its chromosome"));
                }
                None => {}
            }
            self.alts_per_in.extend_from_slice(&piece.alts);
        }
        Ok(())
    }

    fn finish(self) -> Result<VariantPlan, PipelineError> {
        if self.layout.cols.is_none() {
            return Err(PipelineError::Io(
                "Missing .pvar header or inferable columns".to_string(),
            ));
        }
        let mut out_to_in: Vec<(u32, u16)> = Vec::with_capacity(
            self.alts_per_in
                .iter()
                .map(|&count| usize::from(count))
                .sum(),
        );
        for (in_idx, &count) in self.alts_per_in.iter().enumerate() {
            let in_idx = u32::try_from(in_idx)
                .map_err(|_| ioerr(".pvar holds more records than a plan can index"))?;
            out_to_in.extend((1..=count).map(|alt_ord| (in_idx, alt_ord)));
        }
        Ok(VariantPlan {
            in_variants: self.alts_per_in.len(),
            out_variants: out_to_in.len(),
            out_to_in,
            alts_per_in: self.alts_per_in,
        })
    }
}

/// A local `.pvar`, mapped for a parallel scan: its column layout, where its data
/// lines start, and the bounds of data chunks of about a given size, each ending
/// on a line end.
struct LocalPvar {
    map: memmap2::Mmap,
    cols: PvarCols,
    data_start: usize,
    bounds: Vec<usize>,
}

impl LocalPvar {
    /// The header lines are read in order, as the streaming readers read them.
    /// `None` when the file cannot be mapped, a header line is not UTF-8 or not a
    /// header the readers accept, or no header or first data line sets the
    /// columns.
    fn open(pvar_path: &Path, chunk_bytes: usize) -> Option<Self> {
        let file = File::open(pvar_path).ok()?;
        // SAFETY: the map is read-only and lives only as long as the scan that
        // opened it. The file must not be truncated while it is mapped.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        let text: &[u8] = &map;

        let mut cols = None;
        let mut data_start = 0;
        while data_start < text.len() {
            let end = memchr::memchr(b'\n', &text[data_start..])
                .map_or(text.len(), |offset| data_start + offset + 1);
            let trimmed = str::from_utf8(&text[data_start..end]).ok()?.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                break;
            }
            if trimmed.starts_with('#') && !trimmed.starts_with("##") {
                cols = Some(PvarCols::from_header_line(trimmed).ok()?);
            }
            data_start = end;
        }
        let data = &text[data_start..];
        let cols = match cols {
            Some(cols) => cols,
            None => {
                let first_line = data.split(|&byte| byte == b'\n').next()?;
                PvarCols::from_headerless(
                    str::from_utf8(first_line).ok()?.split_whitespace().count(),
                )
                .ok()?
            }
        };

        let chunk_bytes = chunk_bytes.max(1);
        let mut bounds = vec![0];
        let mut search_from = chunk_bytes;
        while search_from < data.len() {
            let Some(offset) = memchr::memchr(b'\n', &data[search_from..]) else {
                break;
            };
            let end = search_from + offset + 1;
            bounds.push(end);
            search_from = end + chunk_bytes;
        }
        if bounds.last() != Some(&data.len()) {
            bounds.push(data.len());
        }
        Some(Self {
            map,
            cols,
            data_start,
            bounds,
        })
    }

    fn data(&self) -> &[u8] {
        &self.map[self.data_start..]
    }
}

/// The rows of the virtual `.bim` that share one chromosome label, with the
/// label as the `.bim` writes it and each row's position.
pub(crate) struct PvarRowRun {
    pub(crate) chrom: String,
    pub(crate) positions: Vec<u64>,
}

/// The chromosome and position of every row of the virtual `.bim` over the local
/// `.pvar` at `pvar_path`, in row order, without rendering the rows.
///
/// The file is mapped, its header lines are read in order, and its data lines are
/// scanned in parallel chunks of about `chunk_bytes`, each extended to the end of
/// its last line. A line is read as [`StreamingVirtualBim`] reads it. `None` when
/// the file cannot be mapped, its text is not UTF-8, a `#` line follows the data,
/// or a line lacks a column the `.bim` needs or an integer position; the caller
/// then reads the `.bim` rows, which report whatever is wrong.
pub(crate) fn scan_local_pvar_rows(
    pvar_path: &Path,
    chunk_bytes: usize,
) -> Option<Vec<PvarRowRun>> {
    use rayon::prelude::*;

    let local = LocalPvar::open(pvar_path, chunk_bytes)?;
    let data = local.data();
    let chunks: Vec<Option<Vec<PvarRowRun>>> = local
        .bounds
        .par_windows(2)
        .map(|chunk| scan_pvar_chunk(&data[chunk[0]..chunk[1]], local.cols))
        .collect();
    let mut runs: Vec<PvarRowRun> = Vec::new();
    for chunk in chunks {
        for run in chunk? {
            if runs.last().is_some_and(|last| last.chrom == run.chrom) {
                runs.last_mut()?.positions.extend_from_slice(&run.positions);
            } else {
                runs.push(run);
            }
        }
    }
    Some(runs)
}

/// The row runs of one chunk of `.pvar` data lines, or `None` where
/// [`scan_local_pvar_rows`] refuses a line.
///
/// A chunk of ASCII without a vertical tab is split on its bytes, which there
/// hold the same whitespace as the text: a vertical tab is whitespace to `trim`
/// and `split_whitespace` but not to their byte counterparts. Other chunks are
/// read as text.
fn scan_pvar_chunk(chunk: &[u8], cols: PvarCols) -> Option<Vec<PvarRowRun>> {
    if !chunk.is_ascii() || memchr::memchr(0x0b, chunk).is_some() {
        return scan_pvar_text_chunk(str::from_utf8(chunk).ok()?, cols);
    }
    let last = cols
        .chrom
        .max(cols.id)
        .max(cols.pos)
        .max(cols.refa)
        .max(cols.alt);
    let mut runs: Vec<PvarRowRun> = Vec::new();
    // The last raw label, and the label it normalizes to.
    let mut label: &[u8] = &[];
    let mut chrom = String::new();
    for line in chunk.split(|&byte| byte == b'\n') {
        let trimmed = line.trim_ascii();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed[0] == b'#' {
            return None;
        }
        let (mut raw_chrom, mut id, mut pos, mut refa, mut alt) = (None, None, None, None, None);
        for (column, field) in trimmed
            .split(u8::is_ascii_whitespace)
            .filter(|field| !field.is_empty())
            .take(last + 1)
            .enumerate()
        {
            if column == cols.chrom {
                raw_chrom = Some(field);
            }
            if column == cols.id {
                id = Some(field);
            }
            if column == cols.pos {
                pos = Some(field);
            }
            if column == cols.refa {
                refa = Some(field);
            }
            if column == cols.alt {
                alt = Some(field);
            }
        }
        id?;
        refa?;
        let raw_chrom = raw_chrom?;
        if raw_chrom != label {
            label = raw_chrom;
            normalize_chrom_into(str::from_utf8(raw_chrom).ok()?, &mut chrom);
        }
        let pos = str::from_utf8(pos?).ok()?.parse::<u64>().ok()?;
        let alts = alt?
            .split(|&byte| byte == b',')
            .map(<[u8]>::trim_ascii)
            .filter(|alt| !alt.is_empty() && *alt != b".")
            .count();
        if alts == 0 {
            continue;
        }
        if runs.last().is_none_or(|last| last.chrom != chrom) {
            runs.push(PvarRowRun {
                chrom: chrom.clone(),
                positions: Vec::new(),
            });
        }
        runs.last_mut()?
            .positions
            .extend(std::iter::repeat_n(pos, alts));
    }
    Some(runs)
}

/// [`scan_pvar_chunk`] for a chunk read as text.
fn scan_pvar_text_chunk(text: &str, cols: PvarCols) -> Option<Vec<PvarRowRun>> {
    let mut runs: Vec<PvarRowRun> = Vec::new();
    let mut chrom = String::new();
    for line in text.split('\n') {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('#') {
            return None;
        }
        let fields = PvarFields::split(trimmed, cols);
        fields.id?;
        fields.refa?;
        normalize_chrom_into(fields.chrom?, &mut chrom);
        let pos = fields.pos?.parse::<u64>().ok()?;
        let alts = fields
            .alt?
            .split(',')
            .map(str::trim)
            .filter(|alt| !alt.is_empty() && *alt != ".")
            .count();
        if alts == 0 {
            continue;
        }
        if runs.last().is_none_or(|last| last.chrom != chrom) {
            runs.push(PvarRowRun {
                chrom: chrom.clone(),
                positions: Vec::new(),
            });
        }
        runs.last_mut()?
            .positions
            .extend(std::iter::repeat_n(pos, alts));
    }
    Some(runs)
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
    decoders: Arc<DecoderPool>,
    plan: VariantPlan,
    n_samples: usize,
    block_bytes: usize, // ceil(n_samples / 4)
    // small LRU of packed blocks by out-variant index
    cache: Arc<Mutex<BlockCache>>,
}

/// A decoder and the hard-call scratch it decodes into.
struct Decoding {
    decoder: PgenDecoder,
    hard_buf: Vec<u8>,
}

/// The decoders no read holds, in a lock-free queue. A read takes one for the blocks it
/// decodes in order and gives it back once they are decoded, so no two reads share a
/// decoder and none waits on another; a read that finds none idle forks one, so there are
/// never more than reads ever ran at once. A decoder keeps the LD anchor it decoded last,
/// which serves the records after it in its variant block. Which decoder serves a block
/// decides nothing about its bytes.
struct DecoderPool {
    template: PgenDecoder,
    idle: SegQueue<Decoding>,
}

/// A decoder taken from a [`DecoderPool`], given back when this is dropped.
struct TakenDecoding<'a> {
    pool: &'a DecoderPool,
    decoding: Option<Decoding>,
}

impl DecoderPool {
    fn new(template: PgenDecoder) -> Self {
        Self {
            template,
            idle: SegQueue::new(),
        }
    }

    fn take(&self) -> TakenDecoding<'_> {
        let decoding = self.idle.pop().unwrap_or_else(|| Decoding {
            decoder: self.template.fork(),
            hard_buf: Vec::new(),
        });
        TakenDecoding {
            pool: self,
            decoding: Some(decoding),
        }
    }
}

impl TakenDecoding<'_> {
    fn decoding(&mut self) -> &mut Decoding {
        self.decoding
            .as_mut()
            .expect("a taken decoder is held until it is given back")
    }
}

impl Drop for TakenDecoding<'_> {
    fn drop(&mut self) {
        if let Some(decoding) = self.decoding.take() {
            self.pool.idle.push(decoding);
        }
    }
}

/// One block's share of a read of the virtual `.bed`: bytes `within..within + dst.len()`
/// of out-variant `out_idx`'s packed block, bound for `dst`.
struct BlockPiece<'d> {
    out_idx: usize,
    within: usize,
    dst: &'d mut [u8],
}

impl VirtualBed {
    fn new(decoder: PgenDecoder, plan: VariantPlan, n_samples: usize) -> Self {
        Self {
            decoders: Arc::new(DecoderPool::new(decoder)),
            plan,
            n_samples,
            block_bytes: n_samples.div_ceil(4),
            cache: Arc::new(Mutex::new(BlockCache::new(256))),
        }
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

impl ByteRangeSource for VirtualBed {
    fn len(&self) -> u64 {
        self.total_len()
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        let mut pieces = Vec::new();
        self.split_range(offset, dst, 0, &mut pieces)?;
        self.decode_pieces(&mut pieces).map_err(|(_, error)| error)
    }

    /// Every range's blocks are decoded together on the rayon pool, so a caller that reads
    /// scattered rows gets them decoded in parallel as a pass over adjacent rows does.
    fn read_ranges(
        &self,
        offsets: &[u64],
        dsts: &mut [&mut [u8]],
    ) -> Result<(), (usize, PipelineError)> {
        let mut pieces = Vec::with_capacity(dsts.len());
        for (range, (&offset, dst)) in offsets.iter().zip(dsts.iter_mut()).enumerate() {
            if let Err(error) = self.split_range(offset, dst, range, &mut pieces) {
                // Reading one range at a time fills every range before this one first.
                self.decode_pieces(&mut pieces)?;
                return Err((range, error));
            }
        }
        self.decode_pieces(&mut pieces)
    }
}

impl VirtualBed {
    /// Writes the header bytes of the range at `offset` that `dst` covers, and appends its
    /// block pieces to `pieces`, each beside `range`: a read of that range is a read of
    /// its pieces.
    fn split_range<'d>(
        &self,
        offset: u64,
        dst: &'d mut [u8],
        range: usize,
        pieces: &mut Vec<(usize, BlockPiece<'d>)>,
    ) -> Result<(), PipelineError> {
        if dst.is_empty() {
            return Ok(());
        }
        let end = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| ioerr("Overflow in read_at range"))?;
        if end > self.total_len() {
            return Err(ioerr("Attempted to read past end of virtual .bed"));
        }

        let header = [BED_MAGIC_0, BED_MAGIC_1, BED_MODE_SNP_MAJOR];
        let in_header = (header.len() as u64)
            .saturating_sub(offset)
            .min(dst.len() as u64) as usize;
        let (head, mut body) = dst.split_at_mut(in_header);
        if in_header > 0 {
            head.copy_from_slice(&header[offset as usize..offset as usize + in_header]);
        }

        let body_offset = offset.saturating_sub(header.len() as u64);
        let mut out_idx = (body_offset / self.block_bytes as u64) as usize;
        let mut within = (body_offset % self.block_bytes as u64) as usize;
        while !body.is_empty() {
            let len = (self.block_bytes - within).min(body.len());
            let (piece, rest) = std::mem::take(&mut body).split_at_mut(len);
            pieces.push((
                range,
                BlockPiece {
                    out_idx,
                    within,
                    dst: piece,
                },
            ));
            body = rest;
            out_idx += 1;
            within = 0;
        }
        Ok(())
    }

    /// Decodes `pieces` on the rayon pool, each piece with the decoder its worker took for
    /// the run of pieces it walks in order, and returns the range beside the first piece
    /// in order that failed, with its error: the error a read of one piece at a time,
    /// in order, stops at. A lone piece is decoded here, as it has nothing to share out.
    fn decode_pieces(
        &self,
        pieces: &mut [(usize, BlockPiece<'_>)],
    ) -> Result<(), (usize, PipelineError)> {
        use rayon::prelude::*;

        if let [(range, piece)] = pieces {
            return self
                .decode_piece(self.decoders.take().decoding(), piece)
                .map_err(|error| (*range, error));
        }
        match pieces
            .par_iter_mut()
            .map_init(
                || self.decoders.take(),
                |taken, (range, piece)| {
                    self.decode_piece(taken.decoding(), piece)
                        .err()
                        .map(|error| (*range, error))
                },
            )
            .find_map_first(|failure| failure)
        {
            Some(failure) => Err(failure),
            None => Ok(()),
        }
    }

    /// A whole block is decoded straight into its piece. Only a partial read, which comes
    /// back for the rest of its block, goes through the cache.
    fn decode_piece(
        &self,
        decoding: &mut Decoding,
        piece: &mut BlockPiece<'_>,
    ) -> Result<(), PipelineError> {
        if piece.within == 0 && piece.dst.len() == self.block_bytes {
            decode_virtual_block(
                self,
                &mut decoding.decoder,
                piece.out_idx,
                &mut decoding.hard_buf,
                piece.dst,
            )
        } else {
            copy_virtual_block(
                self,
                &mut decoding.decoder,
                piece.out_idx,
                piece.within,
                piece.dst,
                &mut decoding.hard_buf,
            )
        }
    }
}

/// Copies bytes `within_block..within_block + dst.len()` of out-variant
/// `out_idx`'s packed block into `dst`, from the block cache, or by decoding
/// the block and caching it.
fn copy_virtual_block(
    bed: &VirtualBed,
    decoder: &mut PgenDecoder,
    out_idx: usize,
    within_block: usize,
    dst: &mut [u8],
    hard_buf: &mut Vec<u8>,
) -> Result<(), PipelineError> {
    let end = within_block + dst.len();
    {
        let cache = bed.cache.lock().unwrap();
        if let Some(buf) = cache.get(out_idx) {
            dst.copy_from_slice(&buf[within_block..end]);
            return Ok(());
        }
    }
    let mut block = vec![0u8; bed.block_bytes];
    decode_virtual_block(bed, decoder, out_idx, hard_buf, &mut block)?;
    let mut cache = bed.cache.lock().unwrap();
    let stored = cache.put(out_idx, block);
    dst.copy_from_slice(&stored[within_block..end]);
    Ok(())
}

/// Decodes out-variant `out_idx` into its packed PLINK 1.9 `block`.
fn decode_virtual_block(
    bed: &VirtualBed,
    decoder: &mut PgenDecoder,
    out_idx: usize,
    hard_buf: &mut Vec<u8>,
    block: &mut [u8],
) -> Result<(), PipelineError> {
    // Decode hard-calls for this (in_idx, alt_ord) into a scratch buffer.
    let (in_idx, alt_ord) = bed
        .plan
        .mapping(out_idx)
        .ok_or_else(|| ioerr("VariantPlan mapping out of bounds"))?;
    let alt_count = bed.plan.alt_count_of_in(in_idx);
    if alt_count == 0 {
        return Err(PipelineError::Io(format!(
            "ALT count missing for variant {} in .pvar plan",
            in_idx
        )));
    }
    if alt_count != 0 && alt_ord > alt_count {
        return Err(ioerr("ALT ordinal exceeds allele count in .pvar"));
    }
    // The packed fast path has no ALT ordinal: it hands back ALT1's calls for
    // whichever split row asked, so it can only serve single-ALT records.
    if alt_count == 1 && decoder.try_decode_packed_block(in_idx, block) {
        return Ok(());
    }

    // Reused across variants: a scoring run decodes millions of blocks, and a
    // fresh sample-sized allocation per block is pure allocator traffic.
    hard_buf.clear();
    hard_buf.resize(bed.n_samples, 255); // 255 = missing
    // No per-sample ploidy: a dosage-only entry rounds on the diploid scale for
    // every sample, whatever the .psam records as its sex.
    decoder.decode_variant_hardcalls(in_idx, alt_ord, hard_buf, None)?;

    VirtualBed::pack_to_block(block, hard_buf);
    Ok(())
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

/// On unix every read is positional (`pread`), so threads reading different
/// records never wait on one another for a file cursor.
struct LocalFileByteRangeSource {
    #[cfg(unix)]
    file: File,
    #[cfg(not(unix))]
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
            #[cfg(unix)]
            file: f,
            #[cfg(not(unix))]
            file: Mutex::new(f),
            len,
        })
    }

    #[cfg(unix)]
    fn read_exact_positioned(&self, offset: u64, dst: &mut [u8]) -> std::io::Result<()> {
        self.file.read_exact_at(dst, offset)
    }

    #[cfg(not(unix))]
    fn read_exact_positioned(&self, offset: u64, dst: &mut [u8]) -> std::io::Result<()> {
        let mut f = self.file.lock().unwrap();
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(dst)
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
        self.read_exact_positioned(offset, dst)
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
    hdr: Arc<PgenHeader>,
    n: usize,
    scratch: Vec<u8>,
    /// Absolute file offset of record `k * OFFSET_STRIDE`.
    stride_offsets: Arc<[u64]>,
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
    /// Scratch and LD anchor of `try_decode_packed_block`.
    packed: PackedScratch,
    alt_counts: Arc<[u16]>,
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
            hdr: Arc::new(hdr),
            n: n_samples_from_psam,
            scratch: Vec::new(),
            stride_offsets: Arc::from(stride_offsets),
            anchor_idx: None,
            anchor_cats: Vec::new(),
            cats_buf: Vec::new(),
            packed: PackedScratch::default(),
            alt_counts: Arc::from(alt_counts),
            dosage_meter: Arc::new(DosageCoercionMeter::new(n_samples_from_psam, in_variants)),
        })
    }

    /// A decoder over the same file, index and coercion meter with its own
    /// scratch buffers and LD anchor, so separate threads can decode at once.
    fn fork(&self) -> Self {
        Self {
            src: Arc::clone(&self.src),
            hdr: Arc::clone(&self.hdr),
            n: self.n,
            scratch: Vec::new(),
            stride_offsets: Arc::clone(&self.stride_offsets),
            anchor_idx: None,
            anchor_cats: Vec::new(),
            cats_buf: Vec::new(),
            packed: PackedScratch::default(),
            alt_counts: Arc::clone(&self.alt_counts),
            dosage_meter: Arc::clone(&self.dosage_meter),
        }
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
        decode_main_track_into(
            &scratch[..len],
            &mut cursor,
            n,
            main_kind,
            None,
            anchor_cats,
        )?;

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
        } else if alt_count > 1 && alt_ord_1b > 1 {
            // No patch track means no sample carries any ALT past the first, so
            // a later ALT's row is zero copies wherever the sample is called and
            // missing where it is not.
            for (dosage, &cat) in a1dosage.iter_mut().zip(cats.iter()) {
                *dosage = if cat == 3 { 255 } else { 0 };
            }
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

    /// Decodes record `in_idx` straight into packed PLINK 1.9 `block` when its
    /// hard calls need no per-sample projection: no multiallelic patch track and
    /// no dosage track.
    ///
    /// Returns false, leaving `block` unspecified, for any other record and for
    /// a record this path cannot validate. `decode_variant_hardcalls` then
    /// decodes it and raises its errors, so the two paths agree on every block,
    /// every error and every coercion-meter count.
    fn try_decode_packed_block(&mut self, in_idx: u32, block: &mut [u8]) -> bool {
        let idx = in_idx as usize;
        let n = self.n;
        if idx >= self.hdr.m_variants as usize || block.len() != n.div_ceil(4) {
            return false;
        }
        let Ok((_, _, rec_ty)) = self.record_offset_len(idx) else {
            return false;
        };
        if rec_ty & 0b1110_1000 != 0 {
            return false;
        }
        let main_kind = rec_ty & 0x07;
        let ld_compressed = matches!(main_kind, 2 | 3);
        if ld_compressed && ((idx & 0xffff) == 0 || self.ensure_packed_anchor(idx).is_err()) {
            return false;
        }
        let Ok((len, _)) = self.load_record(idx) else {
            return false;
        };

        let Self {
            scratch,
            packed,
            dosage_meter,
            ..
        } = self;
        let buf = &scratch[..len];
        let mut cursor = 0usize;
        let anchor = ld_compressed.then_some(packed.anchor.as_slice());
        if decode_main_track_packed(buf, &mut cursor, n, main_kind, anchor, &mut packed.cats)
            .is_err()
        {
            return false;
        }
        if (rec_ty & 0b0001_0000) != 0
            && skip_phase_track(buf, &mut cursor, packed_het_count(&packed.cats, n)).is_err()
        {
            return false;
        }
        if cursor != len {
            return false;
        }

        let meter: &DosageCoercionMeter = dosage_meter;
        if meter.claim(idx) {
            meter.note_variant(false);
            meter.maybe_report();
        }

        write_packed_calls(&packed.cats, n, block);
        if !ld_compressed {
            // Later LD-compressed records in this variant block diff against it.
            std::mem::swap(&mut packed.cats, &mut packed.anchor);
            packed.anchor_idx = Some(idx);
        }
        true
    }

    /// `ensure_anchor` for the packed path: leaves the raw categories of the LD
    /// anchor for `target` in `self.packed.anchor`.
    fn ensure_packed_anchor(&mut self, target: usize) -> Result<(), PipelineError> {
        let anchor_idx = self.ld_anchor_index(target)?;
        if self.packed.anchor_idx == Some(anchor_idx) {
            return Ok(());
        }
        self.packed.anchor_idx = None;

        let (len, rec_ty) = self.load_record(anchor_idx)?;
        let main_kind = rec_ty & 0x07;
        if matches!(main_kind, 2 | 3) {
            return Err(ioerr("LD anchor is itself LD-compressed"));
        }
        let n = self.n;
        let Self {
            scratch, packed, ..
        } = self;
        let mut cursor = 0usize;
        decode_main_track_packed(
            &scratch[..len],
            &mut cursor,
            n,
            main_kind,
            None,
            &mut packed.anchor,
        )?;
        packed.anchor_idx = Some(anchor_idx);
        Ok(())
    }
}

/// Scratch for decoding records straight into packed PLINK blocks. Categories
/// are packed two bits per sample, sample `i` at bits `2 * (i % 32)` of word
/// `i / 32`, the layout of a type-0 main track read as little-endian words.
#[derive(Default)]
struct PackedScratch {
    /// Raw categories of the record being decoded.
    cats: Vec<u64>,
    /// Raw categories of the LD anchor `anchor_idx`.
    anchor: Vec<u64>,
    anchor_idx: Option<usize>,
}

/// The low bit of every two-bit field.
const LOW_BITS: u64 = 0x5555_5555_5555_5555;

/// `decode_main_track_into` into packed categories. Fields past the last
/// sample are left unspecified.
fn decode_main_track_packed(
    buf: &[u8],
    cursor: &mut usize,
    n: usize,
    main_kind: u8,
    anchor: Option<&[u64]>,
    cats: &mut Vec<u64>,
) -> Result<(), PipelineError> {
    let len = buf.len();
    let words = n.div_ceil(32);
    cats.clear();
    cats.resize(words, 0);

    match main_kind {
        0 => {
            let need = n.div_ceil(4);
            if *cursor + need > len {
                return Err(ioerr("Truncated type-0 main track"));
            }
            // Whole words are loaded as words; only the last, short one is copied
            // into a zeroed word.
            let (whole, rest) = buf[*cursor..*cursor + need].as_chunks::<8>();
            for (word, bytes) in cats.iter_mut().zip(whole) {
                *word = u64::from_le_bytes(*bytes);
            }
            if !rest.is_empty() {
                let mut chunk = [0u8; 8];
                chunk[..rest.len()].copy_from_slice(rest);
                cats[whole.len()] = u64::from_le_bytes(chunk);
            }
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
            let nbytes = n.div_ceil(8);
            if *cursor + nbytes > len {
                return Err(ioerr("EOF in bitarray"));
            }
            let low_fields = repeated_category(low);
            let high_fields = repeated_category(high);
            let categories = |bits: u32| {
                let set = spread_to_low_bits(bits);
                let fields = set | (set << 1);
                (low_fields & !fields) | (high_fields & fields)
            };
            let (whole, rest) = buf[*cursor..*cursor + nbytes].as_chunks::<4>();
            for (word, bytes) in cats.iter_mut().zip(whole) {
                *word = categories(u32::from_le_bytes(*bytes));
            }
            if !rest.is_empty() {
                let mut chunk = [0u8; 4];
                chunk[..rest.len()].copy_from_slice(rest);
                cats[whole.len()] = categories(u32::from_le_bytes(chunk));
            }
            *cursor += nbytes;
            patch_packed_categories(buf, cursor, n, cats)?;
        }
        2 | 3 => {
            let anchor = anchor.ok_or_else(|| ioerr("Missing LD anchor"))?;
            if anchor.len() != words {
                return Err(ioerr("LD anchor sample-count mismatch"));
            }
            cats.copy_from_slice(anchor);
            patch_packed_categories(buf, cursor, n, cats)?;
            if main_kind == 3 {
                // Swap REF and ALT homozygotes: flip the high bit where the low
                // bit is clear.
                for word in cats.iter_mut() {
                    let low = *word & LOW_BITS;
                    let high = (*word >> 1) & LOW_BITS;
                    *word = low | ((high ^ (!low & LOW_BITS)) << 1);
                }
            }
        }
        4 | 6 | 7 => {
            let category = match main_kind {
                4 => 0u8,
                6 => 2,
                _ => 3,
            };
            cats.fill(repeated_category(category));
            patch_packed_categories(buf, cursor, n, cats)?;
        }
        _ => {
            return Err(PipelineError::Io(format!(
                "Unsupported main-track type {main_kind}"
            )));
        }
    }
    Ok(())
}

#[inline]
fn repeated_category(category: u8) -> u64 {
    u64::from(category) * LOW_BITS
}

/// Moves bit `i` of `bits` to bit `2 * i`.
#[inline]
fn spread_to_low_bits(bits: u32) -> u64 {
    let mut x = u64::from(bits);
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    (x | (x << 1)) & LOW_BITS
}

fn patch_packed_categories(
    buf: &[u8],
    cursor: &mut usize,
    n: usize,
    cats: &mut [u64],
) -> Result<(), PipelineError> {
    for (sid, val) in difflist_pairs(buf, cursor, n)? {
        let sample = sid as usize;
        if sample < n {
            let shift = 2 * (sample % 32);
            let word = &mut cats[sample / 32];
            *word = (*word & !(0b11 << shift)) | (u64::from(val) << shift);
        }
    }
    Ok(())
}

/// Samples in category 1 (heterozygous).
fn packed_het_count(cats: &[u64], n: usize) -> usize {
    let mut count = 0usize;
    for (w, &word) in cats.iter().enumerate() {
        let mut het = word & !(word >> 1) & LOW_BITS;
        let samples_in_word = (n - w * 32).min(32);
        if samples_in_word < 32 {
            het &= (1u64 << (2 * samples_in_word)) - 1;
        }
        count += het.count_ones() as usize;
    }
    count
}

/// Steps `cursor` over a phase track for `het` heterozygous samples, with the
/// bounds checks `decode_variant_hardcalls` applies.
fn skip_phase_track(buf: &[u8], cursor: &mut usize, het: usize) -> Result<(), PipelineError> {
    let len = buf.len();
    let start = *cursor;
    if start >= len {
        return Err(ioerr("EOF in phase header"));
    }
    let mut bit_cursor = 1usize;
    let mut phased = het;
    if (buf[start] & 1) == 1 {
        if het > 0 && start + (het >> 3) >= len {
            return Err(ioerr("EOF in phase presence"));
        }
        phased = count_set_bits(&buf[start..], 1, het);
        bit_cursor = (1 + het).next_multiple_of(8);
    }
    if phased > 0 && start + ((bit_cursor + phased - 1) >> 3) >= len {
        return Err(ioerr("EOF in phase info"));
    }
    bit_cursor += phased;
    let bytes_needed = bit_cursor.div_ceil(8);
    if start + bytes_needed > len {
        return Err(ioerr("EOF in phase track"));
    }
    *cursor = start + bytes_needed;
    Ok(())
}

/// Set bits among bits `start..start + count` of `bytes`, least significant
/// bit of each byte first.
fn count_set_bits(bytes: &[u8], start: usize, count: usize) -> usize {
    let end = start + count;
    let mut bit = start;
    let mut set = 0usize;
    while bit < end && (bit & 7) != 0 {
        set += usize::from((bytes[bit >> 3] >> (bit & 7)) & 1);
        bit += 1;
    }
    while bit + 8 <= end {
        set += bytes[bit >> 3].count_ones() as usize;
        bit += 8;
    }
    while bit < end {
        set += usize::from((bytes[bit >> 3] >> (bit & 7)) & 1);
        bit += 1;
    }
    set
}

/// Writes packed categories as PLINK 1.9 codes for A1 = ALT, the codes
/// `cats_to_a1dosage` and `VirtualBed::pack_to_block` give together: hom REF 11,
/// het 10, hom ALT 00, missing and padding 01.
fn write_packed_calls(cats: &[u64], n: usize, block: &mut [u8]) {
    let codes = |word: u64| {
        let low = word & LOW_BITS;
        let high = (word >> 1) & LOW_BITS;
        let out_low = !(low ^ high) & LOW_BITS;
        let out_high = !high & LOW_BITS;
        (out_low | (out_high << 1)).to_le_bytes()
    };
    // Whole words are stored as words; only the last, short one is copied.
    let (whole, rest) = block.as_chunks_mut::<8>();
    for (chunk, &word) in whole.iter_mut().zip(cats) {
        *chunk = codes(word);
    }
    if let Some(&word) = cats.get(whole.len())
        && !rest.is_empty()
    {
        rest.copy_from_slice(&codes(word)[..rest.len()]);
    }
    let tail = n % 4;
    if tail != 0
        && let Some(last) = block.last_mut()
    {
        let kept = (1u8 << (2 * tail)) - 1;
        *last = (*last & kept) | (0x55 & !kept);
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

    /// Pieces of a `.pvar` rendered at once give every line the rows, and every
    /// refused line the error, that one streaming pass over the file gives it,
    /// wherever the pieces split: header lines before and among the data, a header
    /// that does not parse, a layout derived from the first data line with enough
    /// fields, and lines the renderer refuses, with every line ending and whitespace
    /// the streaming reader accepts.
    #[test]
    fn pvar_pieces_render_the_streamed_virtual_bim() {
        let texts: [&[u8]; 3] = [
            b"##fileformat=PVARv1.0\n#CHROM\tPOS\tID\tREF\tALT\nchr1\t100\trs1\tA\tG\n\
              1  200 . AC A,T\r\n\n1\t300\trs3\tA\n1\x0b400\x0crs4 G T\n2\tx\trs5\tA\tC\n\
              4\t10\trs6\t\xff\tC\n5\xc2\xa011 rs7 A C,T\n#CHROM\tID\tPOS\tREF\tALT\n\
              6\trs8\t12\tA\t.\nMT\trs9\t13\tA\tC\n#NOT\ta header\n7\trs10\t14\tA\tG\r\n\
              8\trs11\t15\tA\tG",
            b"\n##meta\n22 rs1 0 13 A G\n\n\ny rs2 0 14 C T\n",
            b"1 2 3\n\xff\xfe\n22 rs1 13 A G\n#CHROM POS ID REF ALT\n3 30 rs3 G A\n",
        ];
        let dir = tempfile::tempdir().unwrap();
        for (file, text) in texts.iter().enumerate() {
            let path = dir.path().join(format!("{file}.pvar"));
            std::fs::write(&path, text).unwrap();
            let mut bim = StreamingVirtualBim::new(open_text_source(&path).unwrap(), None);
            let mut expected = Vec::new();
            loop {
                match bim.next_line() {
                    Ok(Some(row)) => expected.push(format!("row {}", String::from_utf8_lossy(row))),
                    Ok(None) => break,
                    Err(error) => expected.push(format!("error {error}")),
                }
            }
            // The headerless file reads cleanly; the other two refuse lines.
            assert_eq!(
                expected.iter().any(|item| item.starts_with("error")),
                file != 1,
                "file {file}"
            );
            if file == 0 {
                assert!(
                    expected.contains(&"row MT\trs9\t0\t13\tC\tA".to_string()),
                    "the header among the data reorders the columns after it"
                );
            }

            let line_starts: Vec<usize> = std::iter::once(0)
                .chain(memchr::memchr_iter(b'\n', text).map(|newline| newline + 1))
                .filter(|&start| start < text.len())
                .collect();
            for (i, &first) in line_starts.iter().enumerate() {
                for &second in &line_starts[i..] {
                    let pieces = [&text[..first], &text[first..second], &text[second..]];
                    let rendered = render_virtual_bim_pieces(
                        &pieces,
                        |out: &mut Vec<String>, line| match line {
                            Ok(lines) => {
                                let mut index = 0;
                                while let Some(row) = lines.row(index) {
                                    out.push(format!("row {}", String::from_utf8_lossy(row)));
                                    index += 1;
                                }
                            }
                            Err(error) => out.push(format!("error {error}")),
                        },
                    );
                    assert_eq!(
                        rendered.concat(),
                        expected,
                        "file {file} split at bytes {first} and {second}"
                    );
                }
            }
        }
    }

    /// The `.pvar` scan must read the rows the streaming virtual `.bim` renders, on
    /// any chunking and with or without a header, across multiallelic sites, sites
    /// without an ALT, labels the `.bim` normalizes, blank lines and line endings,
    /// and on the text path, which a non-ASCII label or a vertical tab takes.
    #[test]
    fn pvar_row_scan_reads_what_the_virtual_bim_renders() {
        let with_header = concat!(
            "##fileformat=PVARv1.0\n",
            "#CHROM\tPOS\tID\tREF\tALT\n",
            "chr1\t100\trs1\tA\tG\n",
            "chr1  200 . A C,T\r\n",
            "\n",
            "1\t300\trs3\tA\t.\n",
            "Chr1\t400\trs4\tA\tG,.,T\n",
            "chrM\t10\trs5\tA\tG\n",
            "\u{1f9ec}1\t20\trs8\tA\tG\n",
            "chrM\u{0b}30\trs9\tA\t\u{0b}T\n",
            "X\t155800000\trs6\tA\tG\n",
            "PAR2\t155900000\trs7\tC\tA",
        );
        let headerless = concat!(
            "chr1 rs1 0 100 A G\n",
            "chr1 . 0 200 A C,T\n",
            "chrX rs3 0 3000000 A G\n",
            "Y rs4 0 4000000 A .\n",
            "Y rs5 0 5000000 A T\n",
        );
        let dir = tempfile::tempdir().unwrap();
        for (name, text) in [
            ("header.pvar", with_header),
            ("headerless.pvar", headerless),
        ] {
            let path = dir.path().join(name);
            std::fs::write(&path, text).unwrap();
            let mut bim = open_virtual_bim(&path).unwrap();
            let mut expected: Vec<(String, u64)> = Vec::new();
            while let Some(line) = bim.next_line().unwrap() {
                let fields: Vec<&str> = str::from_utf8(line).unwrap().split('\t').collect();
                expected.push((fields[0].to_string(), fields[3].parse().unwrap()));
            }
            assert!(expected.len() >= 5, "{name}: {expected:?}");
            for chunk_bytes in [1, 7, 64, 1 << 20] {
                let rows: Vec<(String, u64)> = scan_local_pvar_rows(&path, chunk_bytes)
                    .unwrap()
                    .into_iter()
                    .flat_map(|run| {
                        run.positions
                            .into_iter()
                            .map(move |position| (run.chrom.clone(), position))
                    })
                    .collect();
                assert_eq!(rows, expected, "{name}, {chunk_bytes}-byte chunks");
            }
        }

        // A header line after the data is left to the streaming `.bim`.
        let path = dir.path().join("late_header.pvar");
        std::fs::write(
            &path,
            format!("{headerless}#CHROM\tPOS\tID\tREF\tALT\n1\t10\trs9\tA\tG\n"),
        )
        .unwrap();
        assert!(scan_local_pvar_rows(&path, 1 << 20).is_none());
    }

    /// A variant plan read in pieces, from a mapped `.pvar` or from the lines a
    /// stream gives, must equal the plan of one line after another, or its error
    /// word for word, on any piece size: with or without a header, across
    /// multiallelic sites, sites without an ALT, blank lines and line endings,
    /// non-ASCII text and vertical tabs, a header among the data, positions out of
    /// order within a piece and across pieces, a chromosome entered again, a zero
    /// position, a line without an ALT, a header that does not parse, and a file
    /// that sets no layout at all.
    #[test]
    fn a_pvar_plan_read_in_pieces_is_the_plan_of_one_line_after_another() {
        type Parts = (usize, usize, Vec<(u32, u16)>, Vec<u16>);
        fn parts(plan: VariantPlan) -> Parts {
            (
                plan.in_variants,
                plan.out_variants,
                plan.out_to_in,
                plan.alts_per_in,
            )
        }
        fn words(plan: Result<VariantPlan, PipelineError>) -> Result<Parts, String> {
            plan.map(parts).map_err(|err| err.to_string())
        }
        fn opened(path: &Path) -> Result<Parts, String> {
            let factory_path = path.to_path_buf();
            let pvar: PvarFactory = Arc::new(move || open_text_source(&factory_path));
            words(plan_for(&pvar, Some(path)))
        }

        let cases = [
            (
                "header.pvar",
                concat!(
                    "##fileformat=PVARv1.0\n",
                    "#CHROM\tPOS\tID\tREF\tALT\n",
                    "chr1\t100\trs1\tA\tG\n",
                    "chr1  200 . A C,T\r\n",
                    "\n",
                    "1\t300\trs3\tA\t.\n",
                    "Chr1\t400\trs4\tA\tG,.,T\n",
                    "chrM\t10\trs5\tA\tG\n",
                    "X\t155800000\trs6\tA\tG\n",
                    "PAR2\t155900000\trs7\tC\tA",
                ),
            ),
            (
                "headerless.pvar",
                concat!(
                    "chr1 rs1 0 100 A G\n",
                    "chr1 . 0 200 A C,T\n",
                    "chrX rs3 0 3000000 A G\n",
                    "Y rs4 0 4000000 A .\n",
                    "Y rs5 0 5000000 A T\n",
                ),
            ),
            (
                "text.pvar",
                "#CHROM\tPOS\tID\tREF\tALT\n\u{1f9ec}1\t20\trs8\tA\tG\nchrM\u{0b}30\trs9\tA\t\u{0b}T\n",
            ),
            (
                "late_header.pvar",
                "#CHROM\tPOS\tID\tREF\tALT\n1\t10\trs1\tA\tG\n#CHROM\tID\tPOS\tREF\tALT\n1\trs2\t20\tA\tG\n",
            ),
            (
                "unsorted.pvar",
                "#CHROM\tPOS\tID\tREF\tALT\n1\t300\trs1\tA\tG\n1\t200\trs2\tA\tG\n",
            ),
            (
                "reentered.pvar",
                "#CHROM\tPOS\tID\tREF\tALT\n1\t300\trs1\tA\tG\n2\t10\trs2\tA\tG\n1\t100\trs3\tA\tG\n",
            ),
            (
                "zero_position.pvar",
                "#CHROM\tPOS\tID\tREF\tALT\n1\t0\trs1\tA\tG\n",
            ),
            (
                "missing_alt.pvar",
                "#CHROM\tPOS\tID\tREF\tALT\n1\t10\trs1\tA\n",
            ),
            (
                "unsorted_after_many.pvar",
                concat!(
                    "#CHROM\tPOS\tID\tREF\tALT\n",
                    "1\t10\ta\tA\tG\n1\t20\tb\tA\tG\n1\t30\tc\tA\tG\n1\t40\td\tA\tG\n",
                    "2\t10\te\tA\tG\n2\t20\tf\tA\tG\n2\t15\tg\tA\tG\n2\t30\th\tA\tG\n",
                ),
            ),
            (
                "refused_after_many.pvar",
                concat!(
                    "#CHROM\tPOS\tID\tREF\tALT\n",
                    "1\t10\ta\tA\tG\n1\t20\tb\tA\tG\n1\t30\tc\tA\tG\n2\t5\td\tA\tG\n",
                    "2\tx\te\tA\tG\n1\t1\tf\tA\tG\n",
                ),
            ),
            (
                "bad_header.pvar",
                "#CHROM\tPOS\tREF\tALT\n1\t10\trs1\tA\tG\n",
            ),
            (
                "invalid_text.pvar",
                "#CHROM\tPOS\tID\tREF\tALT\n1\t10\trs1\tA\tG\n1\t20\trs2\t\u{ff}\tG\n",
            ),
            ("short_headerless.pvar", "1 10 rs1 A\n"),
            ("comments_only.pvar", "##fileformat=PVARv1.0\n\n##a\n"),
            ("empty.pvar", ""),
        ];
        let dir = tempfile::tempdir().unwrap();
        let mut plans = 0;
        let mut refusals = 0;
        for (name, text) in cases {
            let path = dir.path().join(name);
            let mut bytes = text.as_bytes().to_vec();
            if name == "invalid_text.pvar" {
                // U+00FF is written as the one byte 0xff, which is not UTF-8.
                let at = bytes
                    .windows(2)
                    .position(|pair| pair == [0xc3, 0xbf])
                    .unwrap();
                bytes.splice(at..at + 2, [0xff]);
            }
            std::fs::write(&path, &bytes).unwrap();
            let expected = words(line_by_line_plan(&mut *open_text_source(&path).unwrap()));
            match &expected {
                Ok(_) => plans += 1,
                Err(_) => refusals += 1,
            }
            for piece_bytes in [1, 7, 64, 1 << 20] {
                assert_eq!(
                    words(VariantPlan::from_local_pvar(&path, piece_bytes)),
                    expected,
                    "{name}: mapped file in {piece_bytes}-byte pieces"
                );
                assert_eq!(
                    words(VariantPlan::from_pvar_in_pieces(
                        &mut *open_text_source(&path).unwrap(),
                        piece_bytes
                    )),
                    expected,
                    "{name}: streamed lines in {piece_bytes}-byte pieces"
                );
            }
            assert_eq!(opened(&path), expected, "{name}: opened plan");
        }
        assert!(
            plans >= 4 && refusals >= 8,
            "{plans} plans, {refusals} refusals"
        );
    }

    /// The plan of a `.pvar` read one line after another, each line in full
    /// before the next: the reference a plan read in pieces must match.
    fn line_by_line_plan(pvar: &mut dyn TextSource) -> Result<VariantPlan, PipelineError> {
        let mut out_to_in: Vec<(u32, u16)> = Vec::new();
        let mut alts_per_in: Vec<u16> = Vec::new();
        let mut header_cols: Option<PvarCols> = None;
        let mut in_idx: u32 = 0;
        let mut in_variants: usize = 0;
        let mut sorted_positions = PvarPositionSortState::default();
        let mut chrom = String::new();

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

            let cols = if let Some(cols) = header_cols {
                cols
            } else {
                let derived = PvarCols::from_headerless(trimmed.split_whitespace().count())?;
                header_cols = Some(derived);
                derived
            };
            let fields = PvarFields::split(trimmed, cols);

            let chrom_raw = fields
                .chrom
                .ok_or_else(|| ioerr(".pvar missing CHROM column"))?;
            let pos_raw = fields
                .pos
                .ok_or_else(|| ioerr(".pvar missing POS column"))?;
            fields.id.ok_or_else(|| ioerr(".pvar missing ID column"))?;
            fields
                .refa
                .ok_or_else(|| ioerr(".pvar missing REF column"))?;
            let alt_raw = fields
                .alt
                .ok_or_else(|| ioerr(".pvar missing ALT column"))?;

            normalize_chrom_into(chrom_raw, &mut chrom);
            let pos = pos_raw
                .parse::<u64>()
                .map_err(|_| ioerr("Invalid POS in .pvar (expected integer)"))?;
            if pos == 0 {
                return Err(ioerr(".pvar POS must be positive"));
            }
            sorted_positions.observe(&chrom, pos, in_variants + 1)?;
            let alt_count = alt_raw
                .split(',')
                .map(|a| a.trim())
                .filter(|a| !a.is_empty() && *a != ".")
                .count();
            for alt_ord in 1..=alt_count as u16 {
                out_to_in.push((in_idx, alt_ord));
            }
            alts_per_in.push(alt_count as u16);
            in_idx += 1;
            in_variants += 1;
        }

        if header_cols.is_none() {
            return Err(PipelineError::Io(
                "Missing .pvar header or inferable columns".to_string(),
            ));
        }
        Ok(VariantPlan {
            in_variants,
            out_variants: out_to_in.len(),
            out_to_in,
            alts_per_in,
        })
    }

    #[test]
    fn chromosome_normalization_accepts_utf8_without_slicing_panics() {
        assert_eq!(normalize_chrom("🧬1"), "🧬1");
        assert_eq!(normalize_chrom("éé"), "éé");
        assert_eq!(normalize_chrom("ChrX"), "X");
    }

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

    /// 1000 Genomes .pvar files carry structural variants (`<INS:ME:ALU>`),
    /// spanning deletions (`*`) and breakends. plink2's .bim export and the VCF
    /// readers keep these as allele codes, so the virtual .bim must too, with
    /// one variant per ALT in ALT order.
    #[test]
    fn pvar_plan_keeps_symbolic_alts_as_allele_codes() {
        let lines = vec![
            "#CHROM\tPOS\tID\tREF\tALT",
            "22\t10532563\tsv1\tT\t<INS:ME:ALU>",
            "22\t10600000\tv2\tA\tG,*",
            "22\t10700000\tbnd1\tG\tG]22:10800000]",
        ];
        let plan = VariantPlan::from_pvar(&mut LineSource::new(lines.clone()))
            .expect("symbolic ALTs are allele codes");
        assert_eq!(plan.in_variants, 3);
        assert_eq!(plan.out_variants, 4);
        let mappings: Vec<_> = (0..4).map(|out| plan.mapping(out).unwrap()).collect();
        assert_eq!(mappings, vec![(0, 1), (1, 1), (1, 2), (2, 1)]);
        assert_eq!(plan.alt_count_of_in(1), 2);

        let mut bim = StreamingVirtualBim::new(Box::new(LineSource::new(lines)), None);
        let mut rows = Vec::new();
        while let Some(row) = bim.next_line().unwrap() {
            let fields: Vec<String> = str::from_utf8(row)
                .unwrap()
                .split('\t')
                .map(str::to_string)
                .collect();
            rows.push((fields[3].clone(), fields[4].clone(), fields[5].clone()));
        }
        let expected = [
            ("10532563", "<INS:ME:ALU>", "T"),
            ("10600000", "G", "A"),
            ("10600000", "*", "A"),
            ("10700000", "G]22:10800000]", "G"),
        ];
        assert_eq!(rows.len(), expected.len());
        for ((pos, a1, a2), (want_pos, want_a1, want_a2)) in rows.iter().zip(expected) {
            assert_eq!(
                (pos.as_str(), a1.as_str(), a2.as_str()),
                (want_pos, want_a1, want_a2)
            );
        }
    }

    /// A biallelic row keeps its .pvar ID, the rows of a split site get
    /// `<ID>__ALT=<ALT>`, and a site without an ID gets `chr:pos:ref:alt`.
    #[test]
    fn virtual_bim_rows_keep_ids_and_disambiguate_split_sites() {
        let lines = vec![
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL",
            "chr1\t100\trs1\tA\tG\t.",
            "1\t200\trs2\tC\tT,G\t.",
            "1\t300\t.\tAT\tA,ATT\t.",
            "chrM\t400\t.\tG\tC\t.",
        ];
        let mut bim = StreamingVirtualBim::new(Box::new(LineSource::new(lines)), None);
        let mut rows = Vec::new();
        while let Some(row) = bim.next_line().unwrap() {
            rows.push(String::from_utf8(row.to_vec()).unwrap());
        }
        assert_eq!(
            rows,
            vec![
                "1\trs1\t0\t100\tG\tA",
                "1\trs2__ALT=T\t0\t200\tT\tC",
                "1\trs2__ALT=G\t0\t200\tG\tC",
                "1\t1:300:AT:A\t0\t300\tA\tAT",
                "1\t1:300:AT:ATT\t0\t300\tATT\tAT",
                "MT\tMT:400:G:C\t0\t400\tC\tG",
            ]
        );
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

    /// The packed path must give the byte path's block for every main-track
    /// type, with and without a phase track, under every haploid rule and in
    /// either read order, and must hand the records it does not cover (a
    /// dosage track, a malformed record) back to the byte path.
    #[test]
    fn packed_block_decode_matches_byte_path() {
        fn rand(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }
        fn random_cats(n: usize, state: &mut u64) -> Vec<u8> {
            (0..n).map(|_| (rand(state) & 3) as u8).collect()
        }
        /// A difflist over up to 10 samples, also applied to `cats`.
        fn difflist(n: usize, state: &mut u64, cats: &mut [u8]) -> Vec<u8> {
            let mut sids = Vec::new();
            let mut sid = (rand(state) % 3) as usize;
            while sid < n && sids.len() < 10 {
                sids.push(sid);
                sid += 1 + (rand(state) % 7) as usize;
            }
            let vals: Vec<u8> = sids.iter().map(|_| (rand(state) & 3) as u8).collect();
            let mut out = encode_varint(sids.len() as u64);
            if sids.is_empty() {
                return out;
            }
            push_sid(&mut out, sids[0] as u32, sample_id_bytes(n));
            out.extend_from_slice(&pack_twobit_values(&vals));
            for pair in sids.windows(2) {
                out.extend_from_slice(&encode_varint((pair[1] - pair[0]) as u64));
            }
            for (&sample, &val) in sids.iter().zip(&vals) {
                cats[sample] = val;
            }
            out
        }
        fn phase_track(het: usize, present: bool, state: &mut u64) -> Vec<u8> {
            let mut bits = vec![present];
            let mut phased = het;
            if present {
                phased = 0;
                for _ in 0..het {
                    let bit = rand(state) & 1 == 1;
                    phased += usize::from(bit);
                    bits.push(bit);
                }
                while bits.len() % 8 != 0 {
                    bits.push(false);
                }
            }
            for _ in 0..phased {
                bits.push(rand(state) & 1 == 1);
            }
            let mut out = vec![0u8; bits.len().div_ceil(8)];
            for (i, &bit) in bits.iter().enumerate() {
                if bit {
                    out[i >> 3] |= 1 << (i & 7);
                }
            }
            out
        }
        let het = |cats: &[u8]| cats.iter().filter(|&&c| c == 1).count();

        for n in [1usize, 2, 3, 4, 5, 31, 32, 33, 63, 64, 65, 127, 200, 255] {
            let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ n as u64;
            let mut records: Vec<(u8, Vec<u8>)> = Vec::new();

            let cats0 = random_cats(n, &mut state);
            records.push((0, pack_twobit_values(&cats0)));
            for ld_type in [2u8, 3] {
                let mut cats = cats0.clone();
                records.push((ld_type, difflist(n, &mut state, &mut cats)));
            }

            // Type 1: het everywhere, hom ALT where the bit is set, then patched.
            let mut cats3 = vec![1u8; n];
            let mut rec = vec![5u8];
            let mut bits = vec![0u8; n.div_ceil(8)];
            for sample in 0..n {
                if rand(&mut state) & 1 == 1 {
                    bits[sample >> 3] |= 1 << (sample & 7);
                    cats3[sample] = 2;
                }
            }
            rec.extend_from_slice(&bits);
            rec.extend_from_slice(&difflist(n, &mut state, &mut cats3));
            records.push((1, rec));
            let mut cats = cats3.clone();
            records.push((2, difflist(n, &mut state, &mut cats)));

            for (main_type, fill) in [(4u8, 0u8), (6, 2), (7, 3)] {
                let mut cats = vec![fill; n];
                records.push((main_type, difflist(n, &mut state, &mut cats)));
            }

            let cats8 = random_cats(n, &mut state);
            let mut rec = pack_twobit_values(&cats8);
            rec.extend_from_slice(&phase_track(het(&cats8), false, &mut state));
            records.push((0x10, rec));
            let cats9 = random_cats(n, &mut state);
            let mut rec = pack_twobit_values(&cats9);
            rec.extend_from_slice(&phase_track(het(&cats9), true, &mut state));
            records.push((0x10, rec));
            let mut cats10 = cats9.clone();
            let mut rec = difflist(n, &mut state, &mut cats10);
            rec.extend_from_slice(&phase_track(het(&cats10), true, &mut state));
            records.push((0x12, rec));

            // 11: a trailing byte; 12: a dense dosage track.
            let mut rec = pack_twobit_values(&cats0);
            rec.push(0);
            records.push((0, rec));
            let mut rec = pack_twobit_values(&cats0);
            for sample in 0..n {
                rec.extend_from_slice(&(sample as u16).wrapping_mul(977).to_le_bytes());
            }
            records.push((0x40, rec));

            let m = records.len();
            let rec_types: Vec<u8> = records.iter().map(|(ty, _)| *ty).collect();
            let rec_lens: Vec<u32> = records.iter().map(|(_, rec)| rec.len() as u32).collect();
            let data: Vec<u8> = records.iter().flat_map(|(_, rec)| rec.clone()).collect();
            let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(data));
            let decoder = || {
                let hdr = PgenHeader {
                    mode: PgenMode::Var,
                    m_variants: m as u32,
                    n_samples: n as u32,
                    fmt_byte: 0,
                    block_offsets: vec![0],
                    rec_types: rec_types.clone(),
                    rec_lens: rec_lens.clone(),
                };
                PgenDecoder::new(Arc::clone(&src), hdr, n, m, vec![1; m]).unwrap()
            };

            let forward: Vec<usize> = (0..m).collect();
            let backward: Vec<usize> = (0..m).rev().collect();
            for order in [&forward, &backward] {
                let mut fast = decoder();
                let mut slow = decoder();
                for &idx in order {
                    let mut block = vec![0u8; n.div_ceil(4)];
                    let handled = fast.try_decode_packed_block(idx as u32, &mut block);
                    let mut hard = vec![255u8; n];
                    let decoded = slow.decode_variant_hardcalls(idx as u32, 1, &mut hard, None);
                    match idx {
                        11 => {
                            assert!(!handled, "n={n}: trailing data taken by the packed path");
                            assert!(decoded.is_err());
                        }
                        12 => {
                            assert!(!handled, "n={n}: dosage track taken by the packed path");
                            decoded.unwrap();
                        }
                        _ => {
                            assert!(handled, "n={n} record {idx}: not handled");
                            decoded.unwrap();
                            let mut want = vec![0u8; n.div_ceil(4)];
                            VirtualBed::pack_to_block(&mut want, &hard);
                            assert_eq!(block, want, "n={n} record {idx}");
                        }
                    }
                }
            }
        }
    }

    /// Reads served by decoders taken from the pool must give the bytes a lone
    /// serial reader gets, one block at a time: from rayon workers reading whole
    /// blocks, partial blocks and runs of blocks that decode on the pool (whose
    /// work stealing can put another read on a worker already holding a
    /// decoder), from threads outside the pool reading at once, in pools of two
    /// widths, and as scattered ranges read together.
    #[test]
    fn concurrent_reads_through_the_decoder_pool_match_a_serial_reader() {
        use rayon::prelude::*;

        const N: usize = 37;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut records: Vec<(u8, Vec<u8>)> = Vec::new();
        for idx in 0..96usize {
            if idx % 4 == 0 {
                let cats: Vec<u8> = (0..N).map(|_| (next() & 3) as u8).collect();
                records.push((0, pack_twobit_values(&cats)));
            } else {
                // An LD record without differences: its anchor, or the anchor inverted.
                records.push((2 + (idx % 2) as u8, encode_varint(0)));
            }
        }
        let m = records.len();
        let rec_types: Vec<u8> = records.iter().map(|(ty, _)| *ty).collect();
        let rec_lens: Vec<u32> = records.iter().map(|(_, rec)| rec.len() as u32).collect();
        let data: Vec<u8> = records.iter().flat_map(|(_, rec)| rec.clone()).collect();
        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(data));
        let virtual_bed = || {
            let hdr = PgenHeader {
                mode: PgenMode::Var,
                m_variants: m as u32,
                n_samples: N as u32,
                fmt_byte: 0,
                block_offsets: vec![0],
                rec_types: rec_types.clone(),
                rec_lens: rec_lens.clone(),
            };
            let decoder = PgenDecoder::new(Arc::clone(&src), hdr, N, m, vec![1; m]).unwrap();
            let plan = VariantPlan {
                in_variants: m,
                out_variants: m,
                out_to_in: (0..m as u32).map(|idx| (idx, 1)).collect(),
                alts_per_in: vec![1; m],
            };
            VirtualBed::new(decoder, plan, N)
        };

        let block_bytes = N.div_ceil(4);
        let serial = virtual_bed();
        let mut expected = vec![0u8; 3 + m * block_bytes];
        serial.read_at(0, &mut expected[..3]).unwrap();
        for idx in 0..m {
            let start = 3 + idx * block_bytes;
            serial
                .read_at(start as u64, &mut expected[start..start + block_bytes])
                .unwrap();
        }
        assert_eq!(
            expected[3 + 2 * block_bytes..3 + 3 * block_bytes],
            expected[3..3 + block_bytes],
            "record 2 repeats its anchor, so the fixture decodes through LD anchors"
        );

        // Reads within the header and across its end, then reads anywhere.
        let requests: Vec<(usize, usize)> = [(0, 1), (1, 1), (2, 1), (1, 2), (2, block_bytes + 1)]
            .into_iter()
            .chain((0..400usize).map(|i| {
                let start = (i * 7919) % expected.len();
                let len = match i % 4 {
                    0 => block_bytes,
                    1 => i % block_bytes + 1,
                    2 => (2 + i % 19) * block_bytes,
                    _ => 2 * block_bytes + 3,
                };
                (start, len.min(expected.len() - start))
            }))
            .collect();
        let check = |bed: &VirtualBed, &(start, len): &(usize, usize)| {
            let mut got = vec![0u8; len];
            bed.read_at(start as u64, &mut got).unwrap();
            assert_eq!(
                got,
                expected[start..start + len],
                "read of {len} bytes at {start}"
            );
        };
        let check_together = |bed: &VirtualBed, requests: &[(usize, usize)]| {
            let offsets: Vec<u64> = requests.iter().map(|&(start, _)| start as u64).collect();
            let mut got: Vec<Vec<u8>> = requests.iter().map(|&(_, len)| vec![0u8; len]).collect();
            let mut dsts: Vec<&mut [u8]> = got.iter_mut().map(Vec::as_mut_slice).collect();
            bed.read_ranges(&offsets, &mut dsts).unwrap();
            for (&(start, len), got) in requests.iter().zip(&got) {
                assert_eq!(
                    got[..],
                    expected[start..start + len],
                    "range of {len} bytes at {start}, read together"
                );
            }
        };

        let narrow = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let wide = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        for pool in [&narrow, &wide] {
            let bed = virtual_bed();
            pool.install(|| requests.par_iter().for_each(|request| check(&bed, request)));
            pool.install(|| {
                requests
                    .par_chunks(37)
                    .for_each(|chunk| check_together(&bed, chunk))
            });
            check_together(&bed, &requests);
            let (bed, requests, check) = (&bed, &requests, &check);
            std::thread::scope(|scope| {
                for reader in 0..4 {
                    scope.spawn(move || {
                        for request in requests.iter().skip(reader).step_by(3) {
                            check(bed, request);
                        }
                    });
                }
            });
        }
    }

    /// Ranges read together fail as reading them one at a time in order does:
    /// every range before the first that fails is filled, and that range's
    /// position and error come back, whichever decode finishes first.
    #[test]
    fn ranges_read_together_stop_at_the_first_failing_range() {
        const N: usize = 9;
        let m = 40usize;
        let cats: Vec<u8> = (0..N).map(|i| (i % 3) as u8).collect();
        let record = pack_twobit_values(&cats);
        let mut records: Vec<(u8, Vec<u8>)> = vec![(0, record.clone()); m];
        // A record whose main track is cut short cannot be decoded.
        records[23] = (0, record[..1].to_vec());
        let rec_types: Vec<u8> = records.iter().map(|(ty, _)| *ty).collect();
        let rec_lens: Vec<u32> = records.iter().map(|(_, rec)| rec.len() as u32).collect();
        let data: Vec<u8> = records.iter().flat_map(|(_, rec)| rec.clone()).collect();
        let src: Arc<dyn ByteRangeSource> = Arc::new(VecSource::new(data));
        let hdr = PgenHeader {
            mode: PgenMode::Var,
            m_variants: m as u32,
            n_samples: N as u32,
            fmt_byte: 0,
            block_offsets: vec![0],
            rec_types,
            rec_lens,
        };
        let decoder = PgenDecoder::new(Arc::clone(&src), hdr, N, m, vec![1; m]).unwrap();
        let plan = VariantPlan {
            in_variants: m,
            out_variants: m,
            out_to_in: (0..m as u32).map(|idx| (idx, 1)).collect(),
            alts_per_in: vec![1; m],
        };
        let bed = VirtualBed::new(decoder, plan, N);
        let block_bytes = N.div_ceil(4);
        let row = |idx: usize| 3 + (idx * block_bytes) as u64;
        let mut one = vec![0u8; block_bytes];
        bed.read_at(row(0), &mut one).unwrap();
        let single_error = bed.read_at(row(23), &mut one).unwrap_err().to_string();

        let cases: [(Vec<u64>, Option<usize>); 3] = [
            // A row that cannot be decoded, among rows that can on both sides.
            (
                (0..m)
                    .step_by(3)
                    .map(row)
                    .chain([row(23)])
                    .chain((24..m).map(row))
                    .collect(),
                Some(14),
            ),
            // A range beyond the end of the virtual `.bed`, after a row that cannot be read.
            (vec![row(1), row(23), row(m), row(2)], Some(1)),
            // A range beyond the end, ahead of any row that fails.
            (vec![row(4), row(m), row(23)], Some(1)),
        ];
        for (offsets, failing) in cases {
            let mut got: Vec<Vec<u8>> = offsets.iter().map(|_| vec![0xAA; block_bytes]).collect();
            let mut dsts: Vec<&mut [u8]> = got.iter_mut().map(Vec::as_mut_slice).collect();
            let result = bed.read_ranges(&offsets, &mut dsts);
            let (position, error) = result.expect_err("a range fails");
            assert_eq!(Some(position), failing, "ranges {offsets:?}");
            if offsets[position] == row(23) {
                assert_eq!(error.to_string(), single_error);
            }
            for (range, (offset, got)) in offsets.iter().zip(&got).enumerate().take(position) {
                let mut want = vec![0u8; block_bytes];
                bed.read_at(*offset, &mut want).unwrap();
                assert_eq!(got, &want, "range {range} before the failing one");
            }
        }
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
    fn coerce_pheno_token_handles_missing_values() {
        assert_eq!(coerce_pheno_token("   "), "-9");
        assert_eq!(coerce_pheno_token("NaN"), "-9");
        assert_eq!(coerce_pheno_token("1.5"), "1.5");
        assert_eq!(coerce_pheno_token("nonsense"), "-9");
    }
}
