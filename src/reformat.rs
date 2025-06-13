// ========================================================================================
//
//               PGS CATALOG SCORE FILE DIAGNOSTICS & REFORMATTING
//
// ========================================================================================

use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// Checks if a file appears to be in the gnomon-native format by inspecting its header.
/// This is a fast, best-effort check to avoid unnecessary reformatting.
pub fn is_gnomon_native_format(path: &Path) -> io::Result<bool> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    // Scan past any comment lines at the beginning of the file.
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            // File is empty or only contains comments, not a valid native file.
            return Ok(false);
        }
        if !line.starts_with('#') {
            // Found the first non-comment line, which should be the header.
            break;
        }
    }

    // Check if the header matches the expected native format.
    let header = line.trim();
    if header.starts_with("snp_id	effect_allele	other_allele	") {
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Errors that can occur during PGS-to-gnomon reformatting.
#[derive(Debug)]
pub enum ReformatError {
    /// File I/O error.
    Io(io::Error),
    /// Input does not contain the PGS Catalog signature.
    NotPgsFormat,
    /// Header is missing required columns.
    MissingColumns(&'static str),
}

impl Display for ReformatError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ReformatError::Io(e) => write!(f, "I/O error: {}", e),
            ReformatError::NotPgsFormat => write!(f, "Not a PGS Catalog scoring file."),
            ReformatError::MissingColumns(s) => write!(f, "Missing column(s): {}", s),
        }
    }
}

impl Error for ReformatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReformatError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for ReformatError {
    fn from(err: io::Error) -> Self {
        ReformatError::Io(err)
    }
}

/// Reformat a PGS Catalog scoring file into gnomon-native TSV (chr:pos, effect_allele, other_allele, weight).
/// Returns the new file path on success.
pub fn reformat_pgs_file(input_path: &Path) -> Result<PathBuf, ReformatError> {
    let file = File::open(input_path)?;
    let mut reader = BufReader::with_capacity(1 << 20, file);
    let mut line = String::new();
    let mut saw_signature = false;
    let mut score_name: Option<String> = None;

    // Phase 1: Scan metadata comments until header line
    while reader.read_line(&mut line)? > 0 {
        if line.starts_with("###PGS CATALOG SCORING FILE") {
            saw_signature = true;
        }
        if let Some(id) = line.strip_prefix("#pgs_id=") {
            score_name = Some(id.trim().to_string());
        }
        if !line.starts_with('#') {
            break; // header reached
        }
        line.clear();
    }
    if !saw_signature {
        return Err(ReformatError::NotPgsFormat);
    }
    let header_line = line.trim_end();

    // Phase 2: Build column mapping
    let cols: Vec<&str> = header_line.split('\t').collect();
    let mapping = HeaderMapping::from_columns(&cols)?;
    let score_label = score_name.unwrap_or_else(|| "PGS_SCORE".into());
    let output_path = input_path.with_file_name(
        format!("{}.gnomon_format.tsv", input_path.file_stem().unwrap().to_string_lossy())
    );

    // Phase 3: Open writer and emit native header
    let out_file = File::create(&output_path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);
    writeln!(writer, "snp_id\teffect_allele\tother_allele\t{}", score_label)?;

    // Phase 4: Process first data line (header_line is not data)
    // Skip header_line itself

    // Phase 5: Process subsequent lines in-place without reallocating strings
    let mut buf = String::new();
    while reader.read_line(&mut buf)? > 0 {
        if !buf.is_empty() && !buf.starts_with('#') {
            mapping.write_row(buf.trim_end(), &mut writer)?;
        }
        buf.clear();
    }

    writer.flush()?;
    Ok(output_path)
}

// ========================================================================================
//                        PRIVATE TYPES AND HELPERS
// ========================================================================================

/// Defines which chromosome/position columns to use.
#[derive(Debug, Copy, Clone)]
enum PosColumns {
    Harmonized { chr: usize, pos: usize },
    Author    { chr: usize, pos: usize },
}

/// Holds zero-cost indices into each record's tab-separated fields.
#[derive(Debug, Copy, Clone)]
struct HeaderMapping {
    effect_allele: usize,
    other_allele: usize,
    effect_weight: usize,
    pos_cols: PosColumns,
}

impl HeaderMapping {
    /// Build the mapping from header row, ensuring required columns exist.
    fn from_columns(cols: &[&str]) -> Result<Self, ReformatError> {
        let mut ea = None;
        let mut oa = None;
        let mut ew = None;
        let mut hm_chr = None;
        let mut hm_pos = None;
        let mut cn = None;
        let mut cp = None;

        for (i, &name) in cols.iter().enumerate() {
            match name {
                "effect_allele"       => ea = Some(i),
                "hm_inferOtherAllele"  if oa.is_none() => oa = Some(i),
                "other_allele"         if oa.is_none() => oa = Some(i),
                "effect_weight"        => ew = Some(i),
                "hm_chr"               => hm_chr = Some(i),
                "hm_pos"               => hm_pos = Some(i),
                "chr_name"             => cn = Some(i),
                "chr_position"         => cp = Some(i),
                _ => {}
            }
        }

        let effect_allele   = ea.ok_or(ReformatError::MissingColumns("effect_allele"))?;
        let other_allele    = oa.ok_or(ReformatError::MissingColumns("other_allele or hm_inferOtherAllele"))?;
        let effect_weight   = ew.ok_or(ReformatError::MissingColumns("effect_weight"))?;

        // choose pos strategy
        let pos_cols = if let (Some(chr), Some(pos)) = (hm_chr, hm_pos) {
            PosColumns::Harmonized { chr, pos }
        } else if let (Some(chr), Some(pos)) = (cn, cp) {
            PosColumns::Author { chr, pos }
        } else {
            return Err(ReformatError::MissingColumns("(chr_name & chr_position) or (hm_chr & hm_pos)"));
        };

        Ok(HeaderMapping { effect_allele, other_allele, effect_weight, pos_cols })
    }

    /// Parse a data row and write the gnomon-native line if valid.
    #[inline]
    fn write_row(&self, row: &str, w: &mut BufWriter<File>) -> Result<(), ReformatError> {
        let iter = row.split('\t');
        let mut chr = None;
        let mut pos = None;
        let mut ea  = None;
        let mut oa  = None;
        let mut ew  = None;

        for (i, field) in iter.enumerate() {
            match self.pos_cols {
                PosColumns::Harmonized { chr: ci, pos: pi } => {
                    if i == ci { chr = Some(field) }
                    else if i == pi { pos = Some(field) }
                }
                PosColumns::Author { chr: ci, pos: pi } => {
                    if i == ci { chr = Some(field) }
                    else if i == pi { pos = Some(field) }
                }
            }
            if i == self.effect_allele   { ea = Some(field) }
            if i == self.other_allele    { oa = Some(field) }
            if i == self.effect_weight   { ew = Some(field) }

            if chr.is_some() && pos.is_some() && ea.is_some() && oa.is_some() && ew.is_some() {
                break;
            }
        }

        if let (Some(chr), Some(pos), Some(ea), Some(oa), Some(ew)) = (chr, pos, ea, oa, ew) {
            if !chr.is_empty() && !pos.is_empty() && !ea.is_empty() && !oa.is_empty() && !ew.is_empty() {
                // write with a single syscall
                writeln!(w, "{}:{}\t{}\t{}\t{}", chr, pos, ea, oa, ew)?;
            }
        }
        Ok(())
    }
}
