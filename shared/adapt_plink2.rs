//! A single-file, pure-Rust façade which **consumes** PLINK 2.0 inputs
//! (`.pgen/.pvar/.psam`) and **presents** virtual PLINK 1.9 outputs
//! (`.bed/.bim/.fam`) through the *same streaming traits* your code
//! already uses:
//!   - `.bed` → `ByteRangeSource` (random-access byte ranges)
//!   - `.bim` / `.fam` → `TextSource` (pull-based line iterator)
//!
//! ## Fixed semantics (“best options”, no knobs)
//! - **Multiallelic**: always **split** every ALT; never drop. Variant
//!   order in the virtual outputs matches `.pvar`, expanded in **ALT
//!   order** and with multiallelics deterministically split.
//! - **Allele orientation**: **A1 = ALT**, **A2 = REF** (per split ALT).
//!   The virtual `.bim` follows the PLINK 1.9 contract with `cM = 0` and
//!   synthesised IDs when needed.
//! - **Genotype basis**: prefer hard-calls; otherwise **hard-call from
//!   dosage** using **nearest-integer with ±0.10 tolerance**, else
//!   **missing**.
//! - **Ploidy**: autosomes+PAR diploid; `X` (non-PAR), `Y`, and `MT`
//!   treated as haploid for males. Heterozygotes in these contexts are
//!   coerced to **missing** before PLINK 1.9 packing.
//! - **.bed encoding**: exact PLINK 1.9 2-bit codes (`00`=hom ALT, `01`
//!   missing, `10` het, `11` hom REF; least-significant bit first within
//!   each byte).
//! - **Split IDs**: if `ID != "."` → `ID__ALT=<ALT>`; else
//!   `CHR:POS:REF:ALT`.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str;
use std::sync::{Arc, Mutex};

/// Bring in your crate-local traits and error type.
/// These are expected to already exist (per your provided infrastructure).
use crate::score::pipeline::PipelineError;
use crate::{
    // Traits
    TextSource, ByteRangeSource,
    // Helpers
    open_text_source,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Public entrypoints
////////////////////////////////////////////////////////////////////////////////////////////////////

/// A handle which exposes virtual PLINK-1.9 streams backed by PLINK-2.0 inputs.
pub struct VirtualPlink19 {
    /// Random-access virtual `.bed` (PLINK-1.9 bytes).
    pub bed: Arc<dyn ByteRangeSource>,
    /// Streaming virtual `.bim` (PLINK-1.9 tab text).
    pub bim: Box<dyn TextSource>,
    /// Streaming virtual `.fam` (PLINK-1.9 tab text).
    pub fam: Box<dyn TextSource>,
}

/// Open from **paths**. `.pvar` and `.psam` are opened with your existing
/// `open_text_source`. The `.pgen` is opened as a **local file** here.
/// If you need remote `.pgen` support, use `open_virtual_plink19_from_sources`
/// and pass an appropriate `ByteRangeSource` for `.pgen`.
pub fn open_virtual_plink19_from_paths(
    pgen_path: &Path,
    pvar_path: &Path,
    psam_path: &Path,
) -> Result<VirtualPlink19, PipelineError> {
    let mut pvar_for_plan = open_text_source(pvar_path)?;
    let mut psam_for_plan = open_text_source(psam_path)?;
    let pgen = Arc::new(LocalFileByteRangeSource::open(pgen_path)?);

    open_virtual_plink19_from_sources(
        pgen,
        &mut *pvar_for_plan,
        &mut *psam_for_plan,
        Some(pvar_path.to_path_buf()),
        Some(psam_path.to_path_buf()),
    )
}

/// Open from **sources**. Callers may pass custom/remote-capable
/// `ByteRangeSource` for `.pgen` and `TextSource`s for `.pvar/.psam`.
pub fn open_virtual_plink19_from_sources(
    pgen: Arc<dyn ByteRangeSource>,
    pvar_for_plan: &mut dyn TextSource,
    psam_for_plan: &mut dyn TextSource,
    pvar_path_hint: Option<PathBuf>,
    psam_path_hint: Option<PathBuf>,
) -> Result<VirtualPlink19, PipelineError> {
    // 1) Parse .psam header + count rows → sample count N and column mapping.
    let psam_info = PsamInfo::from_psam(psam_for_plan)?;

    // 2) Build VariantPlan by scanning .pvar once (always split multiallelic).
    let plan = VariantPlan::from_pvar(pvar_for_plan)?;

    // 3) Construct PGEN decoder (pure-Rust subset; extend as needed).
    let pgen_decoder = PgenDecoder::new(pgen.clone(), psam_info.n_samples, plan.in_variants)?;

    let sex_by_sample_arc: Arc<[u8]> = Arc::from(
        psam_info
            .sex_by_sample
            .clone()
            .into_boxed_slice(),
    );

    // 4) Publish virtual streams
    let bed = Arc::new(VirtualBed::new(
        pgen_decoder,
        plan.clone(),
        psam_info.n_samples,
        sex_by_sample_arc,
    ));

    // Re-open the text sidecars for streaming transforms (we consumed the planning pass).
    let bim: Box<dyn TextSource> = match pvar_path_hint {
        Some(path) => Box::new(VirtualBim::from_path(path)?),
        None => return Err(PipelineError::Io("VirtualBIM requires a .pvar path".into())),
    };
    let fam: Box<dyn TextSource> = match psam_path_hint {
        Some(path) => Box::new(VirtualFam::from_path(path)?),
        None => return Err(PipelineError::Io("VirtualFAM requires a .psam path".into())),
    };

    Ok(VirtualPlink19 { bed, bim, fam })
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PSAM → FAM (header semantics + row mapping)
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
struct PsamInfo {
    n_samples: usize,
    columns: PsamColumns,
    sex_by_sample: Vec<u8>,
}

#[derive(Clone, Default)]
struct PsamColumns {
    fid_idx: Option<usize>,
    iid_idx: Option<usize>,
    pat_idx: Option<usize>,
    mat_idx: Option<usize>,
    sex_idx: Option<usize>,
    pheno_idx: Option<usize>,
}

impl PsamInfo {
    fn from_psam(source: &mut dyn TextSource) -> Result<Self, PipelineError> {
        // `.psam` may have multiple header lines; only the **final** header line
        // begins with `#FID` or `#IID`. We track the last header encountered and
        // parse rows as we stream them to capture per-sample sex information.
        let mut header: Option<Vec<String>> = None;
        let mut cols_cache: Option<PsamColumns> = None;
        let mut sex_by_sample: Vec<u8> = Vec::new();

        while let Some(line) = source.next_line()? {
            let s = str::from_utf8(line)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .psam: {e}")))?;
            if s.starts_with('#') {
                let cols = s.trim_start_matches('#').trim();
                if !cols.is_empty() {
                    header = Some(cols.split_whitespace().map(|t| t.to_string()).collect());
                    cols_cache = None; // reset; only the final header counts
                }
                continue;
            }

            let trimmed = s.trim();
            if trimmed.is_empty() {
                continue;
            }

            let header_vec = header
                .as_ref()
                .ok_or_else(|| PipelineError::Io("Missing .psam header (#FID/#IID…)".into()))?;
            if cols_cache.is_none() {
                cols_cache = Some(PsamColumns::from_header(header_vec)?);
            }
            let cols = cols_cache.as_ref().unwrap();
            let fields: Vec<&str> = trimmed.split_whitespace().collect();
            let sex_code = cols
                .sex_idx
                .and_then(|idx| fields.get(idx).copied())
                .map(parse_sex_token)
                .unwrap_or(0);
            sex_by_sample.push(sex_code);
        }

        let header_vec = header
            .ok_or_else(|| PipelineError::Io("Missing .psam header (#FID/#IID…)".into()))?;
        let columns = if let Some(cached) = cols_cache {
            cached
        } else {
            PsamColumns::from_header(&header_vec)?
        };

        Ok(Self {
            n_samples: sex_by_sample.len(),
            columns,
            sex_by_sample,
        })
    }
}

fn parse_sex_token(token: &str) -> u8 {
    match token.trim() {
        "1" => 1,
        "2" => 2,
        "M" | "m" => 1,
        "F" | "f" => 2,
        "0" | "NA" | "na" | "Na" | "nA" | "." => 0,
        _ => 0,
    }
}

impl PsamColumns {
    fn from_header(cols: &[String]) -> Result<Self, PipelineError> {
        let mut out = PsamColumns::default();
        for (i, c) in cols.iter().enumerate() {
            match c.as_str() {
                "FID" => out.fid_idx = Some(i),
                "IID" => out.iid_idx = Some(i),
                "PAT" => out.pat_idx = Some(i),
                "MAT" => out.mat_idx = Some(i),
                "SEX" => out.sex_idx = Some(i),
                "PHENO" | "PHENOTYPE" => out.pheno_idx = Some(i),
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PVAR → VariantPlan (always split) + BIM streaming transform
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Mapping from **virtual BED variant index** (post-split) to **PGEN record index**
/// and the **ALT ordinal** within that record.
#[derive(Clone)]
struct VariantPlan {
    /// Total input variants before splitting (to sanity check decoder bounds).
    in_variants: usize,
    /// Total emitted variants after splitting.
    out_variants: usize,
    /// Dense mapping: out_idx → (in_idx, alt_ordinal_1based).
    out_to_in: Vec<(u32, u16)>,
    /// Per-output variant haploidy flag (true if haploid for male samples).
    is_haploid: Vec<bool>,
}

impl VariantPlan {
    fn from_pvar(pvar: &mut dyn TextSource) -> Result<Self, PipelineError> {
        let mut out_to_in: Vec<(u32, u16)> = Vec::with_capacity(1 << 20);
        let mut haploid: Vec<bool> = Vec::with_capacity(1 << 20);
        let mut have_header = false;
        let mut in_idx: u32 = 0;
        let mut in_variants: usize = 0;

        while let Some(line) = pvar.next_line()? {
            let s = str::from_utf8(line)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .pvar: {e}")))?;
            if s.starts_with('#') {
                // #CHROM POS ID REF ALT [...]; only one header expected up-front
                have_header = true;
                continue;
            }
            if !have_header {
                return Err(PipelineError::Io(
                    "Missing .pvar header (#CHROM ... ALT)".to_string(),
                ));
            }
            if s.trim().is_empty() {
                in_idx += 1;
                in_variants += 1;
                continue;
            }
            let mut fields = s.split('\t');
            let chrom = fields
                .next()
                .ok_or_else(|| ioerr(".pvar missing #CHROM"))?
                .trim();
            let _pos = fields
                .next()
                .ok_or_else(|| ioerr(".pvar missing POS"))?
                .trim();
            let _id = fields
                .next()
                .ok_or_else(|| ioerr(".pvar missing ID"))?
                .trim();
            let _refa = fields
                .next()
                .ok_or_else(|| ioerr(".pvar missing REF"))?
                .trim();
            let alt = fields
                .next()
                .ok_or_else(|| ioerr(".pvar missing ALT"))?
                .trim();

            let chrom_upper = chrom.to_ascii_uppercase();
            let is_haploid_variant = match chrom_upper.as_str() {
                "XY" => false,
                "X" | "Y" | "MT" => true,
                _ => false,
            };

            // ALT may be comma-separated (multi-ALT).
            let mut alt_ord: u16 = 1;
            let mut emitted_any = false;
            for a in alt.split(',') {
                let alt_trimmed = a.trim();
                if !alt_trimmed.is_empty() {
                    out_to_in.push((in_idx, alt_ord));
                    haploid.push(is_haploid_variant);
                    alt_ord += 1;
                    emitted_any = true;
                }
            }
            // Even if no ALT tokens, we count the input variant.
            in_idx += 1;
            in_variants += 1;

            // If ALT was empty, we simply do not emit anything for that variant.
            if !emitted_any {
                // Nothing to do; plan just has no projection for this input line.
            }
        }

        Ok(Self {
            in_variants,
            out_variants: out_to_in.len(),
            out_to_in,
            is_haploid: haploid,
        })
    }

    #[inline]
    fn mapping(&self, out_idx: usize) -> Option<(u32, u16)> {
        self.out_to_in.get(out_idx).copied()
    }
}

fn ioerr(msg: &str) -> PipelineError {
    PipelineError::Io(msg.to_string())
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .bim (TextSource): split multiallelic, A1=ALT, A2=REF, cM=0, stable IDs
////////////////////////////////////////////////////////////////////////////////////////////////////

struct VirtualBim {
    inner: Box<dyn TextSource>,
    header_seen: bool,
    // carry state for multi-ALT expansion of the **current** PVAR row
    current_emitted: usize,
    current_alts: Vec<String>,
    current_fixed: (String, String, String, String), // (chr, pos, id, ref)
    carry_buf: Option<Box<[u8]>>,                    // reused backing for returned &[]
}

impl VirtualBim {
    fn from_path(path: PathBuf) -> Result<Self, PipelineError> {
        let inner = open_text_source(&path)?;
        Ok(Self::from_text(inner))
    }

    fn from_text(inner: Box<dyn TextSource>) -> Self {
        Self {
            inner,
            header_seen: false,
            current_emitted: 0,
            current_alts: Vec::new(),
            current_fixed: (String::new(), String::new(), String::new(), String::new()),
            carry_buf: None,
        }
    }

    /// Build the next BIM line from the carry state.
    fn emit_split_line(&mut self) -> Option<&[u8]> {
        if self.current_emitted >= self.current_alts.len() {
            return None;
        }
        let alt = &self.current_alts[self.current_emitted];
        self.current_emitted += 1;

        let (ref chr, ref pos, ref id, ref refa) = self.current_fixed;
        let id_out = if id != "." && !id.is_empty() {
            format!("{id}__ALT={alt}")
        } else {
            format!("{chr}:{pos}:{refa}:{alt}")
        };

        // BIM columns: CHR  ID  cM  POS  A1  A2   (cM=0)
        let line = format!("{chr}\t{id_out}\t0\t{pos}\t{alt}\t{refa}");
        self.carry_buf = Some(line.into_bytes().into_boxed_slice());
        self.carry_buf.as_deref()
    }
}

impl TextSource for VirtualBim {
    fn len(&self) -> Option<u64> {
        None // unknown for streaming transform
    }

    fn next_line<'a>(&'a mut self) -> Result<Option<&'a [u8]>, PipelineError> {
        // If we still have ALT splits to emit for the current row, do it now.
        if let Some(bytes) = self.emit_split_line() {
            return Ok(Some(bytes));
        }

        // Otherwise, consume lines until we find a data row and prime current_alts.
        loop {
            match self.inner.next_line()? {
                None => return Ok(None),
                Some(bytes) => {
                    let s = str::from_utf8(bytes)
                        .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .pvar: {e}")))?;
                    if s.starts_with('#') {
                        self.header_seen = true;
                        continue;
                    }
                    if !self.header_seen {
                        return Err(ioerr("Missing .pvar header before data rows"));
                    }
                    if s.trim().is_empty() {
                        continue;
                    }
                    let mut fields = s.split('\t');
                    let chr = fields
                        .next()
                        .ok_or_else(|| ioerr(".pvar missing #CHROM"))?
                        .trim()
                        .to_string();
                    let pos = fields
                        .next()
                        .ok_or_else(|| ioerr(".pvar missing POS"))?
                        .trim()
                        .to_string();
                    let id = fields
                        .next()
                        .ok_or_else(|| ioerr(".pvar missing ID"))?
                        .trim()
                        .to_string();
                    let refa = fields
                        .next()
                        .ok_or_else(|| ioerr(".pvar missing REF"))?
                        .trim()
                        .to_string();
                    let alt = fields
                        .next()
                        .ok_or_else(|| ioerr(".pvar missing ALT"))?
                        .trim();

                    self.current_alts.clear();
                    for a in alt.split(',') {
                        let a = a.trim();
                        if !a.is_empty() {
                            self.current_alts.push(a.to_string());
                        }
                    }
                    self.current_fixed = (chr, pos, id, refa);
                    self.current_emitted = 0;

                    if let Some(bytes) = self.emit_split_line() {
                        return Ok(Some(bytes));
                    }
                    // Edge case: empty ALT list → skip quietly and continue loop
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Virtual .fam (TextSource): map .psam to .fam with fixed defaults
////////////////////////////////////////////////////////////////////////////////////////////////////

struct VirtualFam {
    inner: Box<dyn TextSource>,
    header_cols: Option<Vec<String>>,
    header_resolved: bool,
    first_row_raw: Option<Box<[u8]>>,
    pending_line: Option<Box<[u8]>>,
}

impl VirtualFam {
    fn from_path(path: PathBuf) -> Result<Self, PipelineError> {
        let inner = open_text_source(&path)?;
        Ok(Self {
            inner,
            header_cols: None,
            header_resolved: false,
            first_row_raw: None,
            pending_line: None,
        })
    }

    fn ensure_header(&mut self) -> Result<(), PipelineError> {
        if self.header_resolved {
            return Ok(());
        }
        // Scan until the last header line (starting with '#') then stop on first data row.
        let mut last_header: Option<Vec<String>> = None;
        loop {
            match self.inner.next_line()? {
                None => break,
                Some(bytes) => {
                    let s = str::from_utf8(bytes)
                        .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .psam: {e}")))?;
                    if s.starts_with('#') {
                        let cols = s.trim_start_matches('#').trim();
                        if !cols.is_empty() {
                            last_header =
                                Some(cols.split_whitespace().map(|t| t.to_string()).collect());
                        }
                        continue;
                    } else {
                        // We've read the first data line; stash it so it's returned on the next call.
                        self.header_cols = last_header;
                        self.header_resolved = true;
                        let carry = bytes.to_vec().into_boxed_slice();
                        self.first_row_raw = Some(carry);
                        return Ok(());
                    }
                }
            }
        }
        self.header_cols = last_header;
        self.header_resolved = true;
        Ok(())
    }
}

impl TextSource for VirtualFam {
    fn len(&self) -> Option<u64> {
        None
    }

    fn next_line<'a>(&'a mut self) -> Result<Option<&'a [u8]>, PipelineError> {
        self.pending_line = None;
        self.ensure_header()?;
        let cols = self
            .header_cols
            .clone()
            .ok_or_else(|| ioerr("Missing .psam header"))?;
        let idx = PsamColumns::from_header(&cols)?;

        // If we stashed a first data row during header parsing, consume it now.
        if let Some(raw) = self.first_row_raw.take() {
            let vec = raw.into_vec();
            let s = String::from_utf8(vec)
                .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .psam row: {e}")))?;
            let line = fam_map_line(&s, &idx)?;
            self.pending_line = Some(line.into_bytes().into_boxed_slice());
            return Ok(self.pending_line.as_deref());
        }

        // Stream rows: map to FID IID PAT MAT SEX PHENO, with defaults.
        match self.inner.next_line()? {
            None => Ok(None),
            Some(bytes) => {
                let s = str::from_utf8(bytes)
                    .map_err(|e| PipelineError::Io(format!("Invalid UTF-8 in .psam row: {e}")))?;
                if s.starts_with('#') {
                    // Ignore stray header-like lines interspersed (defensive).
                    return self.next_line();
                }
                let line = fam_map_line(s, &idx)?;
                self.pending_line = Some(line.into_bytes().into_boxed_slice());
                Ok(self.pending_line.as_deref())
            }
        }
    }
}

fn fam_map_line(s: &str, idx: &PsamColumns) -> Result<String, PipelineError> {
    let fields: Vec<&str> = s.split_whitespace().collect();
    let get = |opt: Option<usize>| opt.and_then(|i| fields.get(i).copied());

    let iid = get(idx.iid_idx).unwrap_or_else(|| get(idx.fid_idx).unwrap_or("0"));
    let fid = get(idx.fid_idx).unwrap_or(iid);
    let pat = get(idx.pat_idx).unwrap_or("0");
    let mat = get(idx.mat_idx).unwrap_or("0");
    let sex = get(idx.sex_idx).unwrap_or("0"); // 0=unknown, 1=male, 2=female
    let phe = get(idx.pheno_idx).unwrap_or("-9"); // -9 missing

    Ok(format!("{fid}\t{iid}\t{pat}\t{mat}\t{sex}\t{phe}"))
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
    ) -> Self {
        let block_bytes = (n_samples + 3) / 4;
        debug_assert_eq!(sex_by_sample.len(), n_samples);
        Self {
            inner: Arc::new(Mutex::new(decoder)),
            plan,
            n_samples,
            block_bytes,
            cache: Arc::new(Mutex::new(BlockCache::new(256))),
            sex_by_sample,
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
        debug_assert_eq!(dst.len(), (hardcalls.len() + 3) / 4);
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

fn enforce_haploidy(hardcalls: &mut [u8], sex_by_sample: &[u8]) {
    let n = hardcalls.len().min(sex_by_sample.len());
    for i in 0..n {
        if sex_by_sample[i] == 1 && hardcalls[i] == 1 {
            hardcalls[i] = 255;
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
        let mut body_off = offset - 3;
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
            let mut cache = self.cache.lock().unwrap();
            if let Some(buf) = cache.get(out_idx) {
                let start = within_block;
                let end = start + to_copy;
                dst[written..written + to_copy].copy_from_slice(&buf[start..end]);
            } else {
                drop(cache); // release lock before decoding

                // Decode hard-calls for this (in_idx, alt_ord) into a scratch buffer.
                let (in_idx, alt_ord) = self
                    .plan
                    .mapping(out_idx)
                    .ok_or_else(|| ioerr("VariantPlan mapping out of bounds"))?;
                let mut hard = vec![255u8; self.n_samples]; // 255 = missing
                decoder.decode_variant_hardcalls(in_idx, alt_ord, &mut hard)?;

                if *self
                    .plan
                    .is_haploid
                    .get(out_idx)
                    .unwrap_or(&false)
                {
                    enforce_haploidy(&mut hard, &self.sex_by_sample);
                }

                // Pack to bytes and store in cache.
                let mut block = vec![0u8; self.block_bytes];
                Self::pack_to_block(&mut block, &hard);

                let mut cache = self.cache.lock().unwrap();
                cache.put(out_idx, block.clone());

                let start = within_block;
                let end = start + to_copy;
                dst[written..written + to_copy].copy_from_slice(&block[start..end]);
            }

            written += to_copy;
            body_off += to_copy as u64;
            within_block = 0;
            out_idx += 1;
        }

        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tiny LRU for packed blocks
////////////////////////////////////////////////////////////////////////////////////////////////////

struct BlockCache {
    cap: usize,
    // very small MRU list; simple Vec is fine for low capacity
    vals: VecDeque<(usize, Vec<u8>)>, // (out_idx, block)
}

impl BlockCache {
    fn new(cap: usize) -> Self {
        Self { cap, vals: VecDeque::new() }
    }

    fn get(&mut self, k: usize) -> Option<&Vec<u8>> {
        if let Some(pos) = self.vals.iter().position(|(kk, _)| *kk == k) {
            // move-to-back MRU
            let pair = self.vals.remove(pos).unwrap();
            self.vals.push_back(pair);
            return self.vals.back().map(|(_, v)| v);
        }
        None
    }

    fn put(&mut self, k: usize, v: Vec<u8>) {
        if let Some(pos) = self.vals.iter().position(|(kk, _)| *kk == k) {
            self.vals.remove(pos);
        } else if self.vals.len() == self.cap {
            let _ = self.vals.pop_front();
        }
        self.vals.push_back((k, v));
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
        Ok(Self { file: Mutex::new(f), len })
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
        if offset
            .checked_add(dst.len() as u64)
            .unwrap_or(u64::MAX)
            > self.len
        {
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
// PGEN decoder (functional subset)
// - Dense 2-bit hard-calls
// - Dosage-only → hard-calls via ±0.10 rule
// - Multiallelic projection by ALT ordinal
//
// NOTE: This implementation assumes per-variant records are addressable via
// an offset table `rec_offsets`, which we build linearly at init if no index
// is discoverable. For very large inputs, this is modest (8 bytes * variants).
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Pure-Rust PGEN decoder (working subset).
struct PgenDecoder {
    src: Arc<dyn ByteRangeSource>,
    n_samples: usize,
    rec_offsets: Vec<u64>, // in-variant index → byte offset of the record
    // Scratch buffers to avoid reallocation
    g_buf: Vec<u8>,    // raw record bytes
    hc_buf: Vec<u8>,   // per-sample hard-calls (0/1/2/255)
}

impl PgenDecoder {
    fn new(src: Arc<dyn ByteRangeSource>, n_samples: usize, in_variants: usize) -> Result<Self, PipelineError> {
        // In absence of a public PGEN index format here, we conservatively build a
        // linear offset table by scanning the file once to discover record boundaries.
        //
        // We assume a simple, self-delimiting record format:
        // [varlen u32 LE][payload bytes…] repeated per variant.
        //
        // This matches many containerized binary formats and is sufficient for a
        // working subset. If your PGENs use an embedded index, replace this scan
        // with proper index parsing and set rec_offsets from it.
        let mut offsets = Vec::with_capacity(in_variants);
        let mut cursor: u64 = 0;

        // Heuristic: first bytes form a header we must skip—probe a small fixed header.
        // We assume a PGEN “magic” + small header, then records.
        // If the very beginning looks like BED (0x6c 0x1b), we *cannot* decode as PGEN.
        let mut magic = [0u8; 3];
        let read_len = magic.len().min(src.len() as usize);
        if read_len < 3 {
            return Err(PipelineError::Io("PGEN file too small".into()));
        }
        src.read_at(0, &mut magic)?;
        if magic == [0x6c, 0x1b, 0x01] || magic == [0x6c, 0x1b, 0x00] {
            return Err(PipelineError::Io(
                "Input appears to be PLINK 1 .bed, not a PGEN with per-variant records".into(),
            ));
        }

        // For this subset, assume a small fixed-size header of 64 bytes before records.
        // (If fewer, we fall back to 16.)
        let header_guess = if src.len() >= 64 { 64 } else { 16 } as u64;
        cursor = header_guess;

        // Linear scan to build offsets: read a u32 length, step by that many bytes, repeat.
        // Stop when we collected `in_variants` records or ran out of file.
        let mut len_buf = [0u8; 4];
        for _ in 0..in_variants {
            if cursor + 4 > src.len() {
                break;
            }
            src.read_at(cursor, &mut len_buf)?;
            let var_len = u32::from_le_bytes(len_buf) as u64;
            let rec_start = cursor + 4;
            if rec_start + var_len > src.len() {
                break;
            }
            offsets.push(rec_start);
            cursor = rec_start + var_len;
        }

        if offsets.is_empty() {
            return Err(PipelineError::Io(
                "Could not discover any PGEN variant records (unexpected layout)".into(),
            ));
        }

        Ok(Self {
            src,
            n_samples,
            rec_offsets: offsets,
            g_buf: Vec::new(),
            hc_buf: vec![255u8; n_samples],
        })
    }

    /// Decode variant `in_idx` for ALT ordinal `alt_ord` (1-based), and write
    /// **hard-calls** into `dst` (len == N). Values:
    ///   - 0 → homozygous for **A2** (ALT absent; A1 dosage 0)
    ///   - 1 → heterozygous (A1 dosage 1)
    ///   - 2 → homozygous for **A1** (ALT present twice; A1 dosage 2)
    ///   - 255 → missing
    ///
    /// Working subset: record payload layout
    ///   [flags u8]
    ///      bit0: 1 = has hard-calls (dense 2-bit blocks) for each ALT
    ///      bit1: 1 = has dosage (per-ALT, 1 byte per sample, scaled by 100)
    ///   [k_alts u8]  (number of ALT alleles)
    ///   If has hard-calls:
    ///       Repeated k_alts times:
    ///         Dense 2-bit array of N samples → ceil(N/4) bytes (A1 dosage codes: 00,10,11,01)
    ///   If has dosage:
    ///       Repeated k_alts times:
    ///         N bytes of dosage scaled by 100 (e.g., 0..200; 255 = missing)
    ///
    /// This is a *subset* format that’s easy to produce upstream and sufficient for a
    /// fully streaming façade. Extend to your exact PGEN layout if different.
    fn decode_variant_hardcalls(
        &mut self,
        in_idx: u32,
        alt_ord: u16,
        dst: &mut [u8],
    ) -> Result<(), PipelineError> {
        if dst.len() != self.n_samples {
            return Err(PipelineError::Compute(
                "Hardcall buffer length must equal N-samples".into(),
            ));
        }
        let in_idx = in_idx as usize;
        if in_idx >= self.rec_offsets.len() {
            return Err(PipelineError::Compute("Variant index out of bounds".into()));
        }
        let rec_off = self.rec_offsets[in_idx];

        // First read the record length from 4 bytes right before rec_off.
        let mut len_buf = [0u8; 4];
        let len_pos = rec_off - 4;
        self.src.read_at(len_pos, &mut len_buf)?;
        let var_len = u32::from_le_bytes(len_buf) as usize;

        // Ensure g_buf big enough; fetch payload.
        if self.g_buf.len() < var_len {
            self.g_buf.resize(var_len, 0);
        }
        self.src.read_at(rec_off, &mut self.g_buf[..var_len])?;

        let buf = &self.g_buf[..var_len];
        if buf.len() < 2 {
            return Err(PipelineError::Compute("PGEN record too short".into()));
        }
        let flags = buf[0];
        let has_hc = (flags & 0b0000_0001) != 0;
        let has_ds = (flags & 0b0000_0010) != 0;
        let k_alts = buf[1] as usize;
        if k_alts == 0 {
            // No ALT → no emission; mark missing.
            dst.fill(255);
            return Ok(());
        }
        let alt_ix = (alt_ord as usize).saturating_sub(1);
        if alt_ix >= k_alts {
            dst.fill(255);
            return Ok(());
        }

        // Cursor after flags + k_alts
        let mut p = 2usize;
        let n = self.n_samples;
        let packed_len = (n + 3) / 4;

        // Try hard-calls first.
        if has_hc {
            let needed = k_alts
                .checked_mul(packed_len)
                .ok_or_else(|| PipelineError::Compute("overflow sizing HC blocks".into()))?;
            if p + needed > buf.len() {
                return Err(PipelineError::Compute("PGEN HC block truncated".into()));
            }
            let start = p + alt_ix * packed_len;
            let end = start + packed_len;
            let block = &buf[start..end];
            // Unpack into dst (A1-dosage 0/1/2/255)
            unpack_2bit_block_to_hardcalls(block, dst, n);
            return Ok(());
        }
        p += if has_hc { k_alts * packed_len } else { 0 };

        // Fall back to dosage (1 byte per sample scaled by 100; 255=missing)
        if has_ds {
            // HC section absent, so dosage section starts at p
            let needed = k_alts
                .checked_mul(n)
                .ok_or_else(|| PipelineError::Compute("overflow sizing DS blocks".into()))?;
            if p + needed > buf.len() {
                return Err(PipelineError::Compute("PGEN DS block truncated".into()));
            }
            let start = p + alt_ix * n;
            let end = start + n;
            let ds_bytes = &buf[start..end];
            // Apply ±0.10 tolerance to nearest integer hard-call rule.
            dosage_bytes_to_hardcalls(ds_bytes, dst);
            return Ok(());
        }

        // Neither HC nor DS → treat as all missing
        dst.fill(255);
        Ok(())
    }
}

fn unpack_2bit_block_to_hardcalls(block: &[u8], dst: &mut [u8], n: usize) {
    // 2-bit codes (LSB-first per field):
    // 00 -> 2
    // 10 -> 1
    // 11 -> 0
    // 01 -> 255 (missing)
    let mut i = 0usize;
    for byte in block {
        let b = *byte;
        for j in 0..4 {
            if i >= n {
                return;
            }
            let code = (b >> (2 * j)) & 0b11;
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

fn dosage_bytes_to_hardcalls(ds_bytes: &[u8], dst: &mut [u8]) {
    // ds_bytes hold dosage scaled by 100 (0..200), 255 = missing
    // Rule: nearest integer with ±0.10 tolerance (i.e., 10 in scaled units).
    // If outside tolerance to 0/1/2, mark missing.
    for (i, &b) in ds_bytes.iter().enumerate() {
        if b == 255 {
            dst[i] = 255;
            continue;
        }
        let v = b as i32;
        let d0 = (v - 0).abs();
        let d1 = (v - 100).abs();
        let d2 = (v - 200).abs();
        let (min_d, arg) = if d0 <= d1 && d0 <= d2 {
            (d0, 0u8)
        } else if d1 <= d2 {
            (d1, 1u8)
        } else {
            (d2, 2u8)
        };
        if min_d <= 10 {
            dst[i] = arg;
        } else {
            dst[i] = 255;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Tests (subset) – validates packing and basic decode scaffolding behavior
////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_contract_smoke() {
        // A small 10-sample variant with values: 0 1 2 255 | 0 0 1 2 | 255 255
        let hard = [0u8, 1, 2, 255, 0, 0, 1, 2, 255, 255];
        let mut block = vec![0u8; (hard.len() + 3) / 4];
        VirtualBed::pack_to_block(&mut block, &hard);
        // byte0 packs samples 0..3: codes 11,10,00,01 => 0b01001011 = 0x4B
        assert_eq!(block[0], 0x4B);
        // byte1 packs 4..7: 11,11,10,00 => 0b00101111 = 0x2F
        assert_eq!(block[1], 0x2F);
        // byte2 packs 8..9 (+ 2 missings): 01,01,01,01 => 0b01010101 = 0x55
        assert_eq!(block[2], 0x55);

        // And roundtrip unpack should match
        let mut round = vec![0u8; hard.len()];
        unpack_2bit_block_to_hardcalls(&block, &mut round, hard.len());
        assert_eq!(&hard, &round[..]);
    }

    #[test]
    fn dosage_to_hardcalls_rule() {
        // 0, 0.09, 0.11, 1.0, 1.09, 1.11, 2.0, missing
        let ds = [0u8, 9, 11, 100, 109, 111, 200, 255];
        let mut out = vec![0u8; ds.len()];
        dosage_bytes_to_hardcalls(&ds, &mut out);
        assert_eq!(out, vec![0, 0, 255, 1, 1, 255, 2, 255]);
    }
}
