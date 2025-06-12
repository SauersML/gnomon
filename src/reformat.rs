// ========================================================================================
//
//               PGS CATALOG SCORE FILE DIAGNOSTICS & REFORMATTING
//
// ========================================================================================

use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// An error type for PGS-to-gnomon reformatting.
#[derive(Debug)]
pub enum ReformatError {
    /// I/O problem reading or writing files.
    Io(io::Error),
    /// File does not contain the PGS-Catalog signature.
    NotPgsFormat,
    /// Missing one or more required columns.
    MissingColumns(&'static str),
}

impl Display for ReformatError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ReformatError::Io(e) => write!(f, "I/O error during reformatting: {}", e),
            ReformatError::NotPgsFormat => write!(f, "File is not in PGS Catalog format."),
            ReformatError::MissingColumns(s) => write!(f, "Missing required column(s): {}", s),
        }
    }
}

impl Error for ReformatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        if let ReformatError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<io::Error> for ReformatError {
    fn from(err: io::Error) -> Self {
        ReformatError::Io(err)
    }
}

/// Reads a PGS Catalog scoring file (possibly harmonized) and writes a gnomon-native TSV.
/// Returns the path to the new file on success.
pub fn reformat_pgs_file(input_path: &Path) -> Result<PathBuf, ReformatError> {
    let file = File::open(input_path).map_err(ReformatError::Io)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut saw_pgs_signature = false;
    let mut score_name: Option<String> = None;

    // --- Phase 1: Parse metadata comments and locate header ---
    while reader.read_line(&mut line).map_err(ReformatError::Io)? > 0 {
        if line.starts_with("###PGS CATALOG SCORING FILE") {
            saw_pgs_signature = true;
        }
        if let Some(id) = line.strip_prefix("#pgs_id=") {
            score_name = Some(id.trim().to_owned());
        }
        if !line.starts_with('#') {
            break; // this is the header line
        }
        line.clear();
    }
    if !saw_pgs_signature {
        return Err(ReformatError::NotPgsFormat);
    }
    let header_line = line.trim_end();

    // Split header into column names
    let raw_cols: Vec<&str> = header_line.split('\t').collect();
    let mapping = HeaderMapping::from_columns(&raw_cols)?;

    let score_name = score_name.unwrap_or_else(|| "PGS_SCORE".into());
    let output_path = generate_output_path(input_path);

    // --- Phase 2: Open writer and emit native header ---
    let out_file = File::create(&output_path).map_err(ReformatError::Io)?;
    let mut writer = BufWriter::new(out_file);
    write!(writer, "snp_id\teffect_allele\tother_allele\t{}", score_name)
        .map_err(ReformatError::Io)?;
    writeln!(writer).map_err(ReformatError::Io)?;

    // --- Phase 3: Process the first (already-read) data line ---
    if !header_line.is_empty() {
        write_data_line(header_line, &mapping, &mut writer)?;
    }

    // --- Phase 4: Process the rest of the file ---
    for result in reader.lines() {
        let row = result.map_err(ReformatError::Io)?;
        if !row.trim().is_empty() {
            write_data_line(&row, &mapping, &mut writer)?;
        }
    }

    writer.flush().map_err(ReformatError::Io)?;
    Ok(output_path)
}

// ========================================================================================
//                           PRIVATE IMPLEMENTATION TYPES & HELPERS
// ========================================================================================

/// A typed mapping from column names to indices for fast lookup.
#[derive(Debug)]
struct HeaderMapping {
    snp_id: usize,
    effect_allele: usize,
    other_allele: usize,
    effect_weight: usize,
    hm_chr: Option<usize>,
    hm_pos: Option<usize>,
    chr_name: Option<usize>,
    chr_position: Option<usize>,
}

impl HeaderMapping {
    /// Builds a HeaderMapping from a slice of column names, validating required columns.
    fn from_columns(cols: &[&str]) -> Result<Self, ReformatError> {
        let mut map = HeaderMapping {
            snp_id: usize::MAX,
            effect_allele: usize::MAX,
            other_allele: usize::MAX,
            effect_weight: usize::MAX,
            hm_chr: None,
            hm_pos: None,
            chr_name: None,
            chr_position: None,
        };

        for (i, &name) in cols.iter().enumerate() {
            match name {
                "snp_id" => map.snp_id = i,
                "effect_allele" => map.effect_allele = i,
                "other_allele" => {
                    if map.other_allele == usize::MAX {
                        map.other_allele = i;
                    }
                }
                "hm_inferOtherAllele" => map.other_allele = i,
                "effect_weight" => map.effect_weight = i,
                "hm_chr" => map.hm_chr = Some(i),
                "hm_pos" => map.hm_pos = Some(i),
                "chr_name" => map.chr_name = Some(i),
                "chr_position" => map.chr_position = Some(i),
                _ => {}
            }
        }

        let mut missing = Vec::new();
        if map.snp_id == usize::MAX { missing.push("snp_id"); }
        if map.effect_allele == usize::MAX { missing.push("effect_allele"); }
        if map.other_allele == usize::MAX { missing.push("other_allele or hm_inferOtherAllele"); }
        if map.effect_weight == usize::MAX { missing.push("effect_weight"); }
        let have_pos = map.hm_chr.is_some() && map.hm_pos.is_some();
        let have_auth = map.chr_name.is_some() && map.chr_position.is_some();
        if !have_pos && !have_auth {
            missing.push("(chr_name & chr_position) or (hm_chr & hm_pos)");
        }
        if !missing.is_empty() {
            return Err(ReformatError::MissingColumns(Box::leak(missing.join(", ").into_boxed_str())));
        }
        Ok(map)
    }
}

/// Given a data row and a HeaderMapping, write the normalized line to the writer.
fn write_data_line(
    line: &str,
    m: &HeaderMapping,
    w: &mut BufWriter<File>,
) -> Result<(), ReformatError> {
    let fields: Vec<&str> = line.split('\t').collect();
    // Quick length check
    if fields.len() <= m.effect_weight || fields.len() <= m.other_allele {
        return Ok(());
    }

    // Determine SNP ID by priority
    let snp_id = if let (Some(c), Some(p)) = (m.hm_chr, m.hm_pos) {
        let chr = fields[c]; let pos = fields[p];
        if !chr.is_empty() && !pos.is_empty() {
            Some(format!("{}:{}", chr, pos))
        } else { None }
    } else if let (Some(c), Some(p)) = (m.chr_name, m.chr_position) {
        let chr = fields[c]; let pos = fields[p];
        if !chr.is_empty() && !pos.is_empty() {
            Some(format!("{}:{}", chr, pos))
        } else { None }
    } else {
        None
    };

    if let Some(id) = snp_id {
        let ea = fields[m.effect_allele];
        let oa = fields[m.other_allele];
        let wv = fields[m.effect_weight];
        if !ea.is_empty() && !oa.is_empty() && !wv.is_empty() {
            writeln!(w, "{}\t{}\t{}\t{}", id, ea, oa, wv).map_err(ReformatError::Io)?;
        }
    }
    Ok(())
}

/// Generates an output path by appending `.gnomon_format.tsv` to the input stem.
fn generate_output_path(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    input.with_file_name(format!("{}.gnomon_format.tsv", stem))
}
