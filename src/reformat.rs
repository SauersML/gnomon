// ========================================================================================
//
//               PGS CATALOG SCORE FILE DIAGNOSTICS & REFORMATTING
//
// ========================================================================================

use flate2::read::GzDecoder;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::num::ParseIntError;
use std::path::Path;

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// Checks if a file appears to be in the gnomon-native format by inspecting its header.
pub fn is_gnomon_native_format(path: &Path) -> io::Result<bool> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Ok(false);
        }
        if !line.starts_with('#') {
            break;
        }
    }

    let header = line.trim();
    Ok(header.starts_with("variant_id\teffect_allele\tother_allele\t"))
}

/// Errors that can occur during PGS-to-gnomon reformatting.
#[derive(Debug)]
pub enum ReformatError {
    Io(io::Error),
    NotPgsFormat,
    MissingColumns(&'static str),
    Parse(String),
}

impl Display for ReformatError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            ReformatError::Io(e) => write!(f, "I/O error: {}", e),
            ReformatError::NotPgsFormat => write!(f, "Not a PGS Catalog scoring file."),
            ReformatError::MissingColumns(s) => write!(f, "Missing column(s): {}", s),
            ReformatError::Parse(s) => write!(f, "Parse error: {}", s),
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

/// Reformat a PGS Catalog scoring file into a gnomon-native, sorted TSV.
pub fn reformat_pgs_file(input_path: &Path, output_path: &Path) -> Result<(), ReformatError> {
    let file = File::open(input_path)?;

    // Create a reader that can dynamically handle both plain text and gzipped files.
    // This is done by using a "trait object" which can hold any type that implements `io::Read`.
    let a_reader: Box<dyn Read + Send> = if input_path.extension().map_or(false, |ext| ext == "gz") {
        Box::new(GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut reader = BufReader::with_capacity(1 << 20, a_reader);
    let mut line = String::new();
    let mut saw_signature = false;
    let mut score_name: Option<String> = None;

    while reader.read_line(&mut line)? > 0 {
        if line.starts_with("##PGS CATALOG SCORING FILE") {
            saw_signature = true;
        }
        if let Some(id) = line.strip_prefix("#pgs_id=") {
            score_name = Some(id.trim().to_string());
        }
        if !line.starts_with("##") {
            break;
        }
        line.clear();
    }
    if !saw_signature {
        return Err(ReformatError::NotPgsFormat);
    }
    let header_line = line.trim_end();

    let cols: Vec<&str> = header_line.split('\t').collect();
    let mapping = HeaderMapping::from_columns(&cols)?;
    let score_label = score_name.unwrap_or_else(|| "PGS_SCORE".into());

    let mut lines_to_sort: Vec<SortableLine> = reader
        .lines()
        .filter_map(Result::ok)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .par_bridge()
        .map(|line| mapping.create_sortable_line(&line))
        .collect::<Result<_, _>>()?;

    lines_to_sort.par_sort_unstable_by_key(|item| item.key);

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);
    writeln!(writer, "variant_id\teffect_allele\tother_allele\t{}", score_label)?;
    for item in lines_to_sort {
        writeln!(writer, "{}", item.line_data)?;
    }
    writer.flush()?;
    Ok(())
}

/// Sorts a gnomon-native file that is not guaranteed to be sorted.
pub fn sort_native_file(input_path: &Path, output_path: &Path) -> Result<(), ReformatError> {
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut header = String::new();
    while let Some(Ok(line)) = lines.next() {
        if !line.starts_with('#') {
            header = line;
            break;
        }
    }
    if header.is_empty() {
        return Ok(());
    }

    let mut lines_to_sort: Vec<SortableLine> = lines
        .filter_map(Result::ok)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut parts = line.splitn(2, '\t');
            let variant_id_part = parts.next().unwrap_or("");
            let mut key_parts = variant_id_part.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");
            let key = parse_key(chr_str, pos_str)?;
            Ok(SortableLine { key, line_data: line })
        })
        .collect::<Result<_, ReformatError>>()?;

    lines_to_sort.par_sort_unstable_by_key(|item| item.key);

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "{}", header)?;
    for item in lines_to_sort {
        writeln!(writer, "{}", item.line_data)?;
    }
    writer.flush()?;
    Ok(())
}

// ========================================================================================
//                        PRIVATE TYPES AND HELPERS
// ========================================================================================

struct SortableLine {
    key: (u8, u32),
    line_data: String,
}

fn parse_key(chr_str: &str, pos_str: &str) -> Result<(u8, u32), ReformatError> {
    // First, check for special, non-numeric chromosome names case-insensitively.
    if chr_str.eq_ignore_ascii_case("X") {
        let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| ReformatError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
        return Ok((23, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("Y") {
        let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| ReformatError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
        return Ok((24, pos_num));
    }
    if chr_str.eq_ignore_ascii_case("MT") {
        let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| ReformatError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
        return Ok((25, pos_num));
    }

    // Next, handle numeric chromosomes, stripping a potential "chr" prefix case-insensitively.
    let number_part = if chr_str.len() >= 3 && chr_str[..3].eq_ignore_ascii_case("chr") {
        &chr_str[3..]
    } else {
        chr_str
    };

    // Now, parse the remaining part.
    let chr_num: u8 = number_part.parse().map_err(|_| {
        ReformatError::Parse(format!(
            "Invalid chromosome format '{}'. Expected a number, 'X', 'Y', 'MT', or 'chr' prefix.",
            chr_str
        ))
    })?;

    let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| ReformatError::Parse(format!("Invalid position '{}': {}", pos_str, e)))?;
    
    Ok((chr_num, pos_num))
}

#[derive(Debug, Copy, Clone)]
enum PosColumns {
    Harmonized { chr: usize, pos: usize },
    Author { chr: usize, pos: usize },
}

#[derive(Debug, Copy, Clone)]
struct HeaderMapping {
    effect_allele: usize,
    other_allele: usize,
    effect_weight: usize,
    pos_cols: PosColumns,
}

impl HeaderMapping {
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
                "effect_allele" => ea = Some(i),
                "hm_inferOtherAllele" if oa.is_none() => oa = Some(i),
                "other_allele" if oa.is_none() => oa = Some(i),
                "effect_weight" => ew = Some(i),
                "hm_chr" => hm_chr = Some(i),
                "hm_pos" => hm_pos = Some(i),
                "chr_name" => cn = Some(i),
                "chr_position" => cp = Some(i),
                _ => {}
            }
        }

        let effect_allele = ea.ok_or(ReformatError::MissingColumns("effect_allele"))?;
        let other_allele = oa.ok_or(ReformatError::MissingColumns("other_allele or hm_inferOtherAllele"))?;
        let effect_weight = ew.ok_or(ReformatError::MissingColumns("effect_weight"))?;

        let pos_cols = if let (Some(chr), Some(pos)) = (hm_chr, hm_pos) {
            PosColumns::Harmonized { chr, pos }
        } else if let (Some(chr), Some(pos)) = (cn, cp) {
            PosColumns::Author { chr, pos }
        } else {
            return Err(ReformatError::MissingColumns("(chr_name & chr_position) or (hm_chr & hm_pos)"));
        };

        Ok(HeaderMapping {
            effect_allele,
            other_allele,
            effect_weight,
            pos_cols,
        })
    }

    #[inline]
    fn create_sortable_line(&self, row: &str) -> Result<SortableLine, ReformatError> {
        // This function is on the hot path for reformatting. It avoids heap allocations
        // by iterating over the split string once, instead of collecting to a Vec.
        let mut chr: Option<&str> = None;
        let mut pos: Option<&str> = None;
        let mut ea: Option<&str> = None;
        let mut oa: Option<&str> = None;
        let mut weight: Option<&str> = None;

        let (chr_idx, pos_idx) = match self.pos_cols {
            PosColumns::Harmonized { chr, pos } => (chr, pos),
            PosColumns::Author { chr, pos } => (chr, pos),
        };

        for (i, field) in row.split('\t').enumerate() {
            if i == chr_idx {
                chr = Some(field);
            } else if i == pos_idx {
                pos = Some(field);
            } else if i == self.effect_allele {
                ea = Some(field);
            } else if i == self.other_allele {
                oa = Some(field);
            } else if i == self.effect_weight {
                weight = Some(field);
            }
        }

        // Check that all required fields were found and are not empty.
        let chr_str = chr.filter(|s| !s.is_empty()).ok_or_else(|| ReformatError::Parse("Missing chromosome field in row".to_string()))?;
        let pos_str = pos.filter(|s| !s.is_empty()).ok_or_else(|| ReformatError::Parse("Missing position field in row".to_string()))?;
        let ea_str = ea.filter(|s| !s.is_empty()).ok_or(ReformatError::MissingColumns("effect_allele"))?;
        let oa_str = oa.filter(|s| !s.is_empty()).ok_or(ReformatError::MissingColumns("other_allele"))?;
        let weight_str = weight.filter(|s| !s.is_empty()).ok_or(ReformatError::MissingColumns("effect_weight"))?;

        // Construct the variant_id and parse the sort key.
        let key = parse_key(chr_str, pos_str)?;

        // Reconstruct the line in the gnomon-native format.
        // This now correctly uses only the single effect weight.
        let line_data = format!("{}:{}\t{}\t{}\t{}", chr_str, pos_str, ea_str, oa_str, weight_str);

        Ok(SortableLine { key, line_data })
    }
}
