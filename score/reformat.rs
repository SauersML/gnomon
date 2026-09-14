// ========================================================================================
//
//               PGS catalog score file diagnostics & reformatting
//
// ========================================================================================

use flate2::read::MultiGzDecoder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::score::types::parse_chromosome_label;

// ========================================================================================
//                                   Public API
// ========================================================================================

#[derive(Clone, Debug)]
pub struct ReformatOutcome {
    pub score_label: Option<String>,
    pub skip_summary: Option<SkipSummary>,
    pub warning: Option<String>,
    pub wrote_output: bool,
}

#[derive(Clone, Debug)]
pub struct SkipSummary {
    pub input_path: PathBuf,
    pub total_variant_lines: usize,
    pub skipped_count: usize,
    pub kept_count: usize,
    pub skipped_pct: f64,
    pub reason_counts: Vec<(String, usize)>,
    pub examples: Vec<SkipExample>,
}

#[derive(Clone, Debug)]
pub struct SkipExample {
    pub line_number: usize,
    pub identifier: String,
    pub reason: String,
}

pub fn emit_overall_skip_summary(summaries: &[SkipSummary]) {
    if summaries.is_empty() {
        return;
    }

    const MAX_FILES: usize = 8;
    const MAX_REASON_BUCKETS: usize = 8;
    const MAX_EXAMPLES: usize = 5;

    let total_variant_lines: usize = summaries.iter().map(|s| s.total_variant_lines).sum();
    let skipped_count: usize = summaries.iter().map(|s| s.skipped_count).sum();
    let kept_count = total_variant_lines.saturating_sub(skipped_count);
    let skipped_pct = if total_variant_lines == 0 {
        0.0
    } else {
        (skipped_count as f64 / total_variant_lines as f64) * 100.0
    };

    let mut reason_counts: HashMap<String, usize> = HashMap::new();
    for summary in summaries {
        for (reason, count) in &summary.reason_counts {
            *reason_counts.entry(reason.clone()).or_insert(0) += *count;
        }
    }
    let mut reason_counts_sorted: Vec<(String, usize)> = reason_counts.into_iter().collect();
    reason_counts_sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut file_counts: Vec<(&Path, usize, usize)> = summaries
        .iter()
        .map(|s| {
            (
                s.input_path.as_path(),
                s.skipped_count,
                s.total_variant_lines,
            )
        })
        .collect();
    file_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));

    eprintln!(
        "> Warning: Skipped {} of {} variant(s) during score normalization ({:.2}% skipped; {} kept) across {} affected score file(s).",
        skipped_count,
        total_variant_lines,
        skipped_pct,
        kept_count,
        summaries.len()
    );
    eprintln!("> Affected score files (top {MAX_FILES} by skipped count):");
    for (path, file_skipped, file_total) in file_counts.iter().take(MAX_FILES) {
        let file_pct = if *file_total == 0 {
            0.0
        } else {
            (*file_skipped as f64 / *file_total as f64) * 100.0
        };
        eprintln!(
            ">   - {}: {} / {} skipped ({:.2}%)",
            path.display(),
            file_skipped,
            file_total,
            file_pct
        );
    }
    if file_counts.len() > MAX_FILES {
        eprintln!(
            ">   - ... {} more affected score file(s) omitted",
            file_counts.len() - MAX_FILES
        );
    }

    eprintln!("> Skip summary (top {MAX_REASON_BUCKETS} reasons):");
    for (reason, count) in reason_counts_sorted.iter().take(MAX_REASON_BUCKETS) {
        let pct_of_total = if total_variant_lines == 0 {
            0.0
        } else {
            (*count as f64 / total_variant_lines as f64) * 100.0
        };
        eprintln!(">   - {} ({:.2}%): {}", count, pct_of_total, reason);
    }
    if reason_counts_sorted.len() > MAX_REASON_BUCKETS {
        eprintln!(
            ">   - ... {} more reason bucket(s) omitted",
            reason_counts_sorted.len() - MAX_REASON_BUCKETS
        );
    }

    let examples: Vec<(&Path, &SkipExample)> = summaries
        .iter()
        .flat_map(|summary| {
            summary
                .examples
                .iter()
                .map(move |example| (summary.input_path.as_path(), example))
        })
        .take(MAX_EXAMPLES)
        .collect();
    eprintln!("> Example skipped variants (first {}):", examples.len());
    for (path, example) in examples {
        eprintln!(
            ">   - {}: line {} [{}]: {}",
            path.display(),
            example.line_number,
            example.identifier,
            example.reason
        );
    }
}

/// Opens a score file as text, inflating a gzip or BGZF body, and says whether it
/// was compressed. The magic bytes decide, not the name: PGS Catalog downloads
/// are gzipped, and renamed copies keep or lose `.gz` independently of content.
fn open_score_text(path: &Path) -> io::Result<(Box<dyn Read + Send>, bool)> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 2];
    let mut filled = 0;
    while filled < magic.len() {
        match file.read(&mut magic[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    // The probed bytes are replayed rather than sought back over, so a pipe works too.
    let body = io::Cursor::new(magic[..filled].to_vec()).chain(file);
    if magic[..filled] == [0x1F, 0x8B] {
        // BGZF is a series of gzip members, which the multi-member decoder reads whole.
        Ok((Box::new(MultiGzDecoder::new(body)), true))
    } else {
        Ok((Box::new(body), false))
    }
}

/// Checks if a file is in the gnomon-native format as it stands, by inspecting its
/// header. A compressed file never is: `reformat_pgs_file` inflates it first.
pub fn is_gnomon_native_format(path: &Path) -> io::Result<bool> {
    let (body, compressed) = open_score_text(path)?;
    if compressed {
        return Ok(false);
    }
    let mut reader = BufReader::new(body);
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

/// A structured error type for the reformatting process, designed for useful diagnostics.
#[derive(Debug)]
pub enum ReformatError {
    /// An underlying I/O error.
    Io(io::Error),
    /// The file does not contain the expected PGS Catalog signature.
    NotPgsFormat { path: PathBuf },
    /// The header or a data row is missing a required column.
    MissingColumns {
        path: PathBuf,
        line_number: usize,
        line_content: String,
        missing_column_name: String,
        column_diagnostics: Option<String>,
    },
    /// A value in the file could not be parsed correctly.
    Parse {
        path: PathBuf,
        line_number: usize,
        line_content: String,
        details: String,
    },
    /// The provided PLINK fileset could not be re-ordered safely.
    InvalidPlinkFileset { path: PathBuf, details: String },
}

impl Display for ReformatError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // This implementation provides a detailed, multi-line diagnostic report to the user.
        writeln!(f, "Failed to reformat PGS Catalog score file.")?;
        writeln!(
            f,
            "\n-- Diagnostic Details ----------------------------------------------------"
        )?;

        match self {
            ReformatError::Io(e) => {
                writeln!(f, "Reason:       An I/O error occurred.")?;
                writeln!(f, "Details:      {e}")?;
            }
            ReformatError::NotPgsFormat { path } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(
                    f,
                    "Reason:       The file is not a valid PGS Catalog scoring file."
                )?;
                writeln!(
                    f,
                    "Details:      The required signature ('###PGS CATALOG SCORING FILE') was not found in the file's metadata header."
                )?;
            }
            ReformatError::MissingColumns {
                path,
                line_number,
                line_content,
                missing_column_name,
                column_diagnostics,
            } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(f, "Line Number:  {line_number}")?;
                writeln!(f, "Line Content: \"{}\"", line_content.trim())?;
                writeln!(f, "Reason:       A required column or its data is missing.")?;
                writeln!(
                    f,
                    "Details:      Expected to find column '{missing_column_name}', but it was not found in the header or data for this line."
                )?;
                if let Some(diag) = column_diagnostics {
                    writeln!(f, "Header Parse: {diag}")?;
                }
            }
            ReformatError::Parse {
                path,
                line_number,
                line_content,
                details,
            } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(f, "Line Number:  {line_number}")?;
                writeln!(f, "Line Content: \"{}\"", line_content.trim())?;
                writeln!(
                    f,
                    "Reason:       A value in the line could not be parsed correctly."
                )?;
                writeln!(f, "Details:      {details}")?;
            }
            ReformatError::InvalidPlinkFileset { path, details } => {
                writeln!(f, "File:         {}", path.display())?;
                writeln!(f, "Reason:       Unable to sort PLINK fileset.")?;
                writeln!(f, "Details:      {details}")?;
            }
        }
        write!(
            f,
            "--------------------------------------------------------------------------"
        )
    }
}

impl Error for ReformatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ReformatError::Io(e) => Some(e),
            ReformatError::NotPgsFormat { .. } => None,
            ReformatError::MissingColumns { .. } => None,
            ReformatError::Parse { .. } => None,
            ReformatError::InvalidPlinkFileset { .. } => None,
        }
    }
}

impl From<io::Error> for ReformatError {
    fn from(err: io::Error) -> Self {
        ReformatError::Io(err)
    }
}

fn build_header_column_diagnostics(header_line: &str) -> String {
    let tab_columns: Vec<&str> = header_line.split('\t').collect();
    let ws_columns: Vec<&str> = header_line.split_whitespace().collect();

    let delimiter_hint = if tab_columns.len() > 1 {
        "tab-delimited header detected"
    } else if ws_columns.len() > 1 {
        "header appears whitespace-delimited (spaces) instead of tab-delimited"
    } else {
        "header appears to be a single token"
    };

    let parsed_columns = if tab_columns.len() > 1 {
        tab_columns
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
    } else {
        ws_columns
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
    };

    format!(
        "{}; parsed {} column(s): [{}]",
        delimiter_hint,
        parsed_columns.len(),
        parsed_columns.join(", ")
    )
}

/// Reformat a PGS Catalog scoring file into a gnomon-native, sorted TSV.
pub fn reformat_pgs_file(
    input_path: &Path,
    output_path: &Path,
) -> Result<ReformatOutcome, ReformatError> {
    // --- Define helper types within the function scope ---
    #[derive(Clone, Copy)]
    enum ParsingStrategy {
        OriginalOnly,   // Use chr_name/chr_position, fail if unmappable.
        HarmonizedOnly, // Use hm_chr/hm_pos, fail if unmappable.
        SafeFallback,   // Try hm_chr/hm_pos, fall back to orig on parse failure.
    }

    struct ColumnIndices {
        chr: Option<usize>,
        pos: Option<usize>,
        hm_chr: Option<usize>,
        hm_pos: Option<usize>,
        ea: usize,         // effect_allele is mandatory
        ew: usize,         // effect_weight is mandatory
        oa: Option<usize>, // other_allele is optional
        variant_description: Option<usize>,
        variant_id: Option<usize>,
        hm_variant_id: Option<usize>,
        rs_id: Option<usize>,
        hm_rs_id: Option<usize>,
    }

    let derive_score_label = |score_id: Option<&String>| -> String {
        score_id
            .filter(|id| !id.eq_ignore_ascii_case("PGS_SCORE"))
            .cloned()
            .unwrap_or_else(|| {
                input_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| {
                        // Smart fallback: include parent directory to reduce stem collisions.
                        if let Some(parent_name) = input_path
                            .parent()
                            .and_then(|p| p.file_name())
                            .and_then(|n| n.to_str())
                            && parent_name != "."
                            && parent_name != "/"
                        {
                            return format!("{}_{}", parent_name, s);
                        }
                        s.to_string()
                    })
                    .unwrap_or_else(|| {
                        use std::time::{SystemTime, UNIX_EPOCH};
                        let nanos = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0);
                        format!("PGS_SCORE_{}", nanos)
                    })
            })
    };

    // --- Open the file, inflating a gzip or BGZF body ---
    let (body, _) = open_score_text(input_path)?;
    let mut reader = BufReader::new(body);
    let mut line_buffer = String::new();
    // Kept only to write a compressed native file back out verbatim.
    let mut comment_lines = String::new();

    // --- Read and analyze all metadata headers ---
    let mut orig_build_norm: Option<u8> = None;
    let mut hm_build_norm: Option<u8> = None;
    let mut score_id: Option<String> = None;

    let normalize_build = |build_str: &str| -> Option<u8> {
        match build_str.trim().to_lowercase().as_str() {
            "grch37" | "hg19" | "37" => Some(37),
            "grch38" | "hg38" | "38" => Some(38),
            _ => None,
        }
    };

    let mut total_lines_read = 0usize;

    loop {
        line_buffer.clear();
        if reader.read_line(&mut line_buffer)? == 0 {
            // Reached EOF before finding the data header
            return Err(ReformatError::NotPgsFormat {
                path: input_path.to_path_buf(),
            });
        }
        total_lines_read += 1;
        if !line_buffer.starts_with('#') {
            break; // Found the data header line
        }
        comment_lines.push_str(&line_buffer);

        let metadata = line_buffer.trim_start_matches('#').trim();
        if let Some(val) = metadata.strip_prefix("genome_build=") {
            orig_build_norm = normalize_build(val);
        } else if let Some(val) = metadata.strip_prefix("HmPOS_build=") {
            hm_build_norm = normalize_build(val);
        } else if let Some(val) = metadata.strip_prefix("pgs_id=") {
            score_id = Some(val.to_string());
        }
    }

    // A native file reaches this function only when it was compressed (see
    // `is_gnomon_native_format`). It needs no conversion, so it is written out
    // inflated byte for byte, and is then used exactly as the plain file would be.
    let native_header = line_buffer.trim();
    if native_header.starts_with("variant_id\teffect_allele\tother_allele\t") {
        let score_label = native_header.split('\t').nth(3).unwrap_or("").to_string();
        crate::output::write_atomically(output_path, |writer| {
            writer.write_all(comment_lines.as_bytes())?;
            writer.write_all(line_buffer.as_bytes())?;
            io::copy(&mut reader, writer)?;
            Ok(())
        })?;
        return Ok(ReformatOutcome {
            score_label: Some(score_label),
            skip_summary: None,
            warning: None,
            wrote_output: true,
        });
    }

    // --- Determine the one, true, safe parsing strategy ---
    let strategy = match (orig_build_norm, hm_build_norm) {
        // Same build (e.g., 38 -> 38) OR unknown builds (can't prove they are different)
        (Some(o), Some(h)) if o == h => ParsingStrategy::SafeFallback,
        (None, None) => ParsingStrategy::SafeFallback, // Can't prove different, assume safe
        (Some(_), None) => ParsingStrategy::OriginalOnly, // Harmonized doesn't exist
        (None, Some(_)) => ParsingStrategy::HarmonizedOnly, // Original doesn't exist

        // Builds are provably different (e.g., 37 -> 38). Liftover occurred.
        (Some(_), Some(_)) => ParsingStrategy::HarmonizedOnly,
    };

    // --- Map header columns to indices for robust, order-independent access ---
    let header_line = line_buffer.trim();
    let header_map: std::collections::HashMap<&str, usize> = header_line
        .split('\t')
        .enumerate()
        .map(|(i, name)| (name, i))
        .collect();

    let has_effect_weight = header_map.contains_key("effect_weight");
    let has_dosage_weights = header_map.contains_key("dosage_0_weight")
        && header_map.contains_key("dosage_1_weight")
        && header_map.contains_key("dosage_2_weight");

    if !has_effect_weight && has_dosage_weights {
        let label = derive_score_label(score_id.as_ref());
        let warning = format!(
            "Unsupported score format in '{}': found genotype-specific weights \
('dosage_0_weight', 'dosage_1_weight', 'dosage_2_weight') but no additive \
'effect_weight'. This score is skipped.",
            input_path.display()
        );
        return Ok(ReformatOutcome {
            score_label: Some(label),
            skip_summary: None,
            warning: Some(warning),
            wrote_output: false,
        });
    }

    let column_indices = ColumnIndices {
        chr: header_map.get("chr_name").copied(),
        pos: header_map.get("chr_position").copied(),
        hm_chr: header_map.get("hm_chr").copied(),
        hm_pos: header_map.get("hm_pos").copied(),
        ea: *header_map
            .get("effect_allele")
            .ok_or_else(|| ReformatError::MissingColumns {
                path: input_path.to_path_buf(),
                line_number: 0,
                line_content: header_line.to_string(),
                missing_column_name: "effect_allele".to_string(),
                column_diagnostics: Some(build_header_column_diagnostics(header_line)),
            })?,
        ew: *header_map
            .get("effect_weight")
            .ok_or_else(|| ReformatError::MissingColumns {
                path: input_path.to_path_buf(),
                line_number: 0,
                line_content: header_line.to_string(),
                missing_column_name: "effect_weight".to_string(),
                column_diagnostics: Some(build_header_column_diagnostics(header_line)),
            })?,
        oa: header_map
            .get("other_allele")
            .or_else(|| header_map.get("hm_inferOtherAllele"))
            .copied(),
        variant_description: header_map.get("variant_description").copied(),
        variant_id: header_map.get("variant_id").copied(),
        hm_variant_id: header_map
            .get("hm_variant_id")
            .or_else(|| header_map.get("hm_variantID"))
            .copied(),
        rs_id: header_map
            .get("rsID")
            .or_else(|| header_map.get("rsid"))
            .copied(),
        hm_rs_id: header_map
            .get("hm_rsID")
            .or_else(|| header_map.get("hm_rsid"))
            .copied(),
    };

    fn derive_identifier(line: &str, fields: &[&str], columns: &ColumnIndices) -> String {
        fn value_from_idx<'a>(fields: &'a [&'a str], idx: Option<usize>) -> Option<&'a str> {
            idx.and_then(|i| fields.get(i))
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        }

        if let Some(val) = value_from_idx(fields, columns.variant_id) {
            return val.to_string();
        }
        if let Some(val) = value_from_idx(fields, columns.hm_variant_id) {
            return val.to_string();
        }
        if let Some(val) = value_from_idx(fields, columns.rs_id) {
            return val.to_string();
        }
        if let Some(val) = value_from_idx(fields, columns.hm_rs_id) {
            return val.to_string();
        }

        if let (Some(chr), Some(pos)) = (
            value_from_idx(fields, columns.hm_chr).or_else(|| value_from_idx(fields, columns.chr)),
            value_from_idx(fields, columns.hm_pos).or_else(|| value_from_idx(fields, columns.pos)),
        ) {
            return format!("{chr}:{pos}");
        }

        let trimmed = line.trim();
        if !trimmed.is_empty() {
            trimmed.to_string()
        } else {
            "<empty line>".to_string()
        }
    }

    // This is the "hot path". We use the pre-selected strategy and indices to
    // process all data lines in parallel with minimal branching or overhead.
    // Derive score label from pgs_id if available, otherwise from filename.
    // Derive score label strategies:
    // 1. If `#pgs_id` is present AND NOT generic "PGS_SCORE", use it.
    // 2. Otherwise, use filename stem.
    // 3. Fallback to unique generation to avoid collisions.
    let score_label = derive_score_label(score_id.as_ref());

    /// Resolves one data line against the pre-selected strategy. A kept line is
    /// appended to `rows` as `chr:pos\tea\toa\tweight\n` and its key returned; a
    /// skipped one comes back as a record whose `line_number` is `line_index`.
    fn resolve_line<'a>(
        strategy: ParsingStrategy,
        column_indices: &ColumnIndices,
        line_index: usize,
        line: &'a str,
        fields: &mut Vec<&'a str>,
        rows: &mut Vec<u8>,
    ) -> Result<(u8, u32), SkipRecord> {
        fields.clear();
        fields.extend(line.split('\t'));
        let fields: &[&str] = fields;

        let make_skip = |reason: String| SkipRecord {
            line_number: line_index,
            identifier: derive_identifier(line, fields, column_indices),
            reason,
        };

        let attempt_coords = |label: &str,
                              c_idx: Option<usize>,
                              p_idx: Option<usize>|
         -> Result<(u8, u32), String> {
            let Some(chr_idx) = c_idx else {
                return Err(format!("{label} chromosome column is missing"));
            };
            let Some(pos_idx) = p_idx else {
                return Err(format!("{label} position column is missing"));
            };
            let chr_raw = fields.get(chr_idx).copied().unwrap_or("").trim();
            if chr_raw.is_empty() {
                return Err(format!("{label} chromosome value is empty"));
            }
            let pos_raw = fields.get(pos_idx).copied().unwrap_or("").trim();
            if pos_raw.is_empty() {
                return Err(format!("{label} position value is empty"));
            }

            parse_key(chr_raw, pos_raw)
                .map_err(|detail| format!("{label} coordinates invalid: {detail}"))
        };

        // This is the core logic, applying the pre-determined strategy.
        let key = match strategy {
            ParsingStrategy::SafeFallback => {
                match attempt_coords("Harmonized", column_indices.hm_chr, column_indices.hm_pos) {
                    Ok(key) => key,
                    Err(h_reason) => {
                        match attempt_coords("Original", column_indices.chr, column_indices.pos) {
                            Ok(key) => key,
                            Err(o_reason) => {
                                return Err(make_skip(format!(
                                    "Harmonized coordinates unavailable: {h_reason}; Original coordinates unavailable: {o_reason}"
                                )));
                            }
                        }
                    }
                }
            }
            ParsingStrategy::HarmonizedOnly => {
                match attempt_coords("Harmonized", column_indices.hm_chr, column_indices.hm_pos) {
                    Ok(key) => key,
                    Err(reason) => {
                        return Err(make_skip(format!(
                            "Harmonized coordinates unavailable: {reason}"
                        )));
                    }
                }
            }
            ParsingStrategy::OriginalOnly => {
                match attempt_coords("Original", column_indices.chr, column_indices.pos) {
                    Ok(key) => key,
                    Err(reason) => {
                        return Err(make_skip(format!(
                            "Original coordinates unavailable: {reason}"
                        )));
                    }
                }
            }
        };

        // Safely extract other mandatory fields.
        let Some(ea_str) = fields
            .get(column_indices.ea)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            return Err(make_skip("Missing effect_allele value".to_string()));
        };
        let Some(weight_str) = fields
            .get(column_indices.ew)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        else {
            return Err(make_skip("Missing effect_weight value".to_string()));
        };
        let recovered_oa;
        let oa_str = match column_indices
            .oa
            .and_then(|i| fields.get(i))
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            Some(value) => value,
            // Some catalog rows leave other_allele/hm_inferOtherAllele blank but still
            // spell the pair out in variant_description (e.g. `1:100:G:A`). Recover it
            // when unambiguous; a row we cannot pin down is skipped, not fatal, so one
            // bad row does not take down a whole score file.
            None => match column_indices
                .variant_description
                .and_then(|i| fields.get(i))
                .and_then(|value| other_allele_from_variant_description(value, ea_str))
            {
                Some(recovered) => {
                    recovered_oa = recovered;
                    recovered_oa.as_str()
                }
                None => {
                    return Err(make_skip(
                        "Missing other_allele, and variant_description did not yield an unambiguous non-effect allele"
                            .to_string(),
                    ));
                }
            },
        };

        push_decimal(rows, u32::from(key.0));
        rows.push(b':');
        push_decimal(rows, key.1);
        for value in [ea_str, oa_str, weight_str] {
            rows.push(b'\t');
            rows.extend_from_slice(value.as_bytes());
        }
        rows.push(b'\n');

        Ok(key)
    }

    // --- Decode the data section and resolve it block by block, concurrently ---
    // This thread reads newline-terminated blocks and hands each to the pool as
    // it arrives, so decompression and resolution overlap instead of running one
    // after the other, and one buffer per block replaces a String per line and
    // two more per kept row. Lines, line numbers, skip records and the pre-sort
    // row order are exactly what the line-by-line reader produced.
    let resolve_block = |chunk_index: usize, block: Vec<u8>| -> ResolvedChunk {
        let mut resolved = ResolvedChunk {
            line_count: 0,
            variant_lines: 0,
            rows: Vec::new(),
            sortable: Vec::new(),
            skipped: Vec::new(),
            invalid_utf8: None,
        };
        let Ok(chunk) = std::str::from_utf8(&block) else {
            resolved.invalid_utf8 = Some(invalid_utf8_line_error(&block));
            return resolved;
        };
        resolved.rows.reserve(chunk.len() / 2);
        let mut fields = Vec::new();
        for raw_line in chunk.split_inclusive('\n') {
            // `BufRead::lines` semantics: strip "\n" or "\r\n", nothing else.
            let line = match raw_line.strip_suffix('\n') {
                Some(line) => line.strip_suffix('\r').unwrap_or(line),
                None => raw_line,
            };
            let line_index = resolved.line_count;
            resolved.line_count += 1;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            resolved.variant_lines += 1;
            let row_start = resolved.rows.len();
            match resolve_line(
                strategy,
                &column_indices,
                line_index,
                line,
                &mut fields,
                &mut resolved.rows,
            ) {
                Ok(key) => resolved.sortable.push(SortableRow {
                    key,
                    chunk: chunk_index,
                    start: row_start,
                    end: resolved.rows.len(),
                }),
                Err(record) => resolved.skipped.push(record),
            }
        }
        resolved
    };

    let (resolved_tx, resolved_rx) = std::sync::mpsc::channel::<(usize, ResolvedChunk)>();
    let read_outcome: io::Result<()> = rayon::in_place_scope(|scope| {
        let dispatch = |chunk_index: usize, block: Vec<u8>| {
            let resolved_tx = resolved_tx.clone();
            let resolve_block = &resolve_block;
            scope.spawn(move |_| {
                // The receiver outlives the scope, so this send cannot fail.
                let _ = resolved_tx.send((chunk_index, resolve_block(chunk_index, block)));
            });
        };
        let mut next_index = 0;
        let mut carry = Vec::new();
        loop {
            let mut block = Vec::with_capacity(carry.len() + DATA_CHUNK_BYTES);
            block.append(&mut carry);
            let filled = (&mut reader)
                .take(DATA_CHUNK_BYTES as u64)
                .read_to_end(&mut block);
            // Everything up to the last newline is complete lines; the rest waits
            // for the next block. On a failed read, the complete lines already
            // decoded are still checked first, as a line reader would.
            let complete = block.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
            match filled {
                Ok(0) => {
                    if !block.is_empty() {
                        dispatch(next_index, block);
                    }
                    return Ok(());
                }
                Ok(_) if complete == 0 => carry = block,
                Ok(_) => {
                    carry.extend_from_slice(&block[complete..]);
                    block.truncate(complete);
                    dispatch(next_index, block);
                    next_index += 1;
                }
                Err(read_error) => {
                    if complete > 0 {
                        block.truncate(complete);
                        dispatch(next_index, block);
                    }
                    return Err(read_error);
                }
            }
        }
    });
    drop(resolved_tx);
    let mut resolved_blocks: Vec<(usize, ResolvedChunk)> = resolved_rx.into_iter().collect();
    resolved_blocks.sort_unstable_by_key(|(chunk_index, _)| *chunk_index);
    let mut resolved_chunks: Vec<ResolvedChunk> =
        resolved_blocks.into_iter().map(|(_, chunk)| chunk).collect();
    // The first invalid line in file order wins over a later failed read.
    if let Some(error) = resolved_chunks.iter_mut().find_map(|c| c.invalid_utf8.take()) {
        return Err(error.into());
    }
    read_outcome?;

    let mut total_variant_lines = 0usize;
    let mut lines_to_sort =
        Vec::with_capacity(resolved_chunks.iter().map(|c| c.sortable.len()).sum());
    let mut skipped_records = Vec::new();
    let mut first_line_number = total_lines_read + 1;
    for chunk in &mut resolved_chunks {
        total_variant_lines += chunk.variant_lines;
        lines_to_sort.append(&mut chunk.sortable);
        for mut record in chunk.skipped.drain(..) {
            record.line_number += first_line_number;
            skipped_records.push(record);
        }
        first_line_number += chunk.line_count;
    }

    // Sort the resolved data and write it to the gnomon-native file. Rows enter
    // the sort in file order, as before, so equal keys land in the same order.
    lines_to_sort.par_sort_unstable_by_key(|item| item.key);

    crate::output::write_atomically(output_path, |writer| {
        writeln!(
            writer,
            "variant_id\teffect_allele\tother_allele\t{score_label}"
        )?;
        for item in &lines_to_sort {
            writer.write_all(&resolved_chunks[item.chunk].rows[item.start..item.end])?;
        }
        Ok(())
    })?;

    // --- Report any non-fatal issues to the user ---
    let skip_summary = if !skipped_records.is_empty() {
        skipped_records.sort_by_key(|record| record.line_number);
        let skipped_count = skipped_records.len();
        let kept_count = total_variant_lines.saturating_sub(skipped_count);
        let skipped_pct = if total_variant_lines == 0 {
            0.0
        } else {
            (skipped_count as f64 / total_variant_lines as f64) * 100.0
        };
        let mut reason_counts: HashMap<&str, usize> = HashMap::new();
        for record in &skipped_records {
            *reason_counts.entry(record.reason.as_str()).or_insert(0) += 1;
        }
        let mut reason_counts_sorted: Vec<(String, usize)> = reason_counts
            .into_iter()
            .map(|(reason, count)| (reason.to_string(), count))
            .collect();
        reason_counts_sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        const EXAMPLES_PER_FILE: usize = 8;
        Some(SkipSummary {
            input_path: input_path.to_path_buf(),
            total_variant_lines,
            skipped_count,
            kept_count,
            skipped_pct,
            reason_counts: reason_counts_sorted,
            examples: skipped_records
                .into_iter()
                .take(EXAMPLES_PER_FILE)
                .map(|record| SkipExample {
                    line_number: record.line_number,
                    identifier: record.identifier,
                    reason: record.reason,
                })
                .collect(),
        })
    } else {
        None
    };

    Ok(ReformatOutcome {
        score_label: Some(score_label),
        skip_summary,
        warning: None,
        wrote_output: true,
    })
}

/// How many skipped-contig rows to name before collapsing the rest into a count.
const MAX_SKIPPED_CONTIG_EXAMPLES: usize = 5;

/// Whether `sort_native_file` would write `data` back unchanged: valid UTF-8, no
/// carriage returns or blank lines, comments only above a native header, and rows
/// whose chromosome and position all parse, already in key order. Sorting such a
/// file is the identity permutation, so its output is the input plus a final
/// newline when the last line lacks one; that flag is returned. Anything else
/// (including every file that warns or errors) returns `None`.
fn native_file_already_sorted(data: &[u8]) -> Option<bool> {
    let text = std::str::from_utf8(data).ok()?;
    if text.contains('\r') {
        return None;
    }

    let mut rows = text;
    loop {
        let (line, rest) = match rows.split_once('\n') {
            Some(split) => split,
            None if rows.is_empty() => return None,
            None => (rows, ""),
        };
        rows = rest;
        if line.trim().is_empty() {
            return None;
        }
        if !line.starts_with('#') {
            if !line.starts_with("variant_id\teffect_allele\tother_allele\t") {
                return None;
            }
            break;
        }
    }

    // First and last key of a chunk, or `None` if the chunk breaks a condition.
    let chunk_bounds = |&(start, end): &(usize, usize)| -> Option<((u8, u32), (u8, u32))> {
        let mut bounds: Option<((u8, u32), (u8, u32))> = None;
        for line in rows[start..end].split_inclusive('\n') {
            let line = line.strip_suffix('\n').unwrap_or(line);
            if line.trim().is_empty() || line.starts_with('#') {
                return None;
            }
            let variant_id = line.split('\t').next().unwrap_or("");
            let mut key_parts = variant_id.splitn(2, ':');
            let chr_num = parse_chromosome_label(key_parts.next().unwrap_or("")).ok()?;
            let pos_num: u32 = key_parts.next().unwrap_or("").trim().parse().ok()?;
            let key = (chr_num, pos_num);
            bounds = match bounds {
                None => Some((key, key)),
                Some((first, last)) if last <= key => Some((first, key)),
                Some(_) => return None,
            };
        }
        bounds
    };

    // An unsorted file almost always shows it in its first rows: check the first
    // chunk before waking the thread pool, whose spin-up alone would cost an
    // unsorted file more than the check saves. The collect stops at a failure.
    let spans = newline_aligned_spans(rows.as_bytes());
    let Some((first_span, other_spans)) = spans.split_first() else {
        return Some(!text.ends_with('\n'));
    };
    let mut previous_last = chunk_bounds(first_span)?.1;
    let other_bounds: Vec<((u8, u32), (u8, u32))> = other_spans
        .par_iter()
        .map(chunk_bounds)
        .collect::<Option<_>>()?;
    for (first, last) in other_bounds {
        if previous_last > first {
            return None;
        }
        previous_last = last;
    }
    Some(!text.ends_with('\n'))
}

/// Sorts a gnomon-native file that is not guaranteed to be sorted.
pub fn sort_native_file(input_path: &Path, output_path: &Path) -> Result<(), ReformatError> {
    let mut file = File::open(input_path)?;

    // Files written by `reformat_pgs_file` are already sorted, and for those the
    // pass below reproduces its input byte for byte; skip straight to the copy.
    // An unsorted or unusual file nearly always shows it within its first rows,
    // so the complete lines of a prefix are checked before reading the rest. A
    // failed read also just takes the general path, which sees the same bytes.
    let mut data = Vec::new();
    if (&mut file)
        .take(DATA_CHUNK_BYTES as u64)
        .read_to_end(&mut data)
        .is_ok()
    {
        let complete = data.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        if native_file_already_sorted(&data[..complete]).is_some()
            && file.read_to_end(&mut data).is_ok()
            && let Some(missing_final_newline) = native_file_already_sorted(&data)
        {
            crate::output::write_atomically(output_path, |writer| {
                writer.write_all(&data)?;
                if missing_final_newline {
                    writer.write_all(b"\n")?;
                }
                Ok(())
            })?;
            return Ok(());
        }
    }

    // Whatever was read continues straight into the unread rest of the file.
    let reader = BufReader::new(io::Cursor::new(data).chain(file));

    // Separate header lines from data lines to correctly track data line numbers.
    let mut header_lines: Vec<String> = vec![];
    let mut data_lines: Vec<String> = vec![];

    for line_result in reader.lines() {
        let line = line_result.map_err(ReformatError::Io)?;
        if line.trim().is_empty() {
            continue;
        }
        if !line.starts_with('#') {
            data_lines.push(line);
        } else {
            header_lines.push(line);
        }
    }

    if data_lines.is_empty() {
        return Ok(());
    }
    // The gnomon-native header is the first data line.
    let header = data_lines.remove(0);
    if !header.starts_with("variant_id\teffect_allele\tother_allele\t") {
        return Err(ReformatError::Parse {
            path: input_path.to_path_buf(),
            line_number: header_lines.len() + 1,
            line_content: header,
            details:
                "Invalid native score header; expected variant_id/effect_allele/other_allele/score."
                    .to_string(),
        });
    }

    let skipped_contigs: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());

    let mut lines_to_sort: Vec<SortableLine> = data_lines
        .into_par_iter()
        .enumerate()
        .map(|(i, line)| {
            // The line number is relative to the start of the data section.
            let line_number = i + 2;
            if line.is_empty() {
                return Ok(None);
            }

            let mut parts = line.splitn(2, '\t');
            let variant_id_part = parts.next().unwrap_or("");
            let mut key_parts = variant_id_part.splitn(2, ':');
            let chr_str = key_parts.next().unwrap_or("");
            let pos_str = key_parts.next().unwrap_or("");

            // Harmonized catalog rows land on alt/random contigs often enough that a
            // whole-file abort is the wrong response: no primary-assembly genotype file
            // can match them anyway. Drop those rows and summarize them below. A
            // malformed position, by contrast, still means a broken file.
            let Ok(chr_num) = parse_chromosome_label(chr_str) else {
                if let Ok(mut guard) = skipped_contigs.lock() {
                    guard.push((line_number, chr_str.to_string()));
                }
                return Ok(None);
            };
            let pos_num: u32 = pos_str.trim().parse().map_err(|e: ParseIntError| {
                ReformatError::Parse {
                    path: input_path.to_path_buf(),
                    line_number,
                    line_content: line.clone(),
                    details: format!("Invalid position '{pos_str}': {e}"),
                }
            })?;

            Ok(Some(SortableLine {
                key: (chr_num, pos_num),
                line_data: line,
            }))
        })
        .filter_map(|result| result.transpose())
        .collect::<Result<_, ReformatError>>()?;

    let mut skipped_contigs = skipped_contigs.into_inner().unwrap_or_default();
    if !skipped_contigs.is_empty() {
        skipped_contigs.sort_unstable();
        eprintln!(
            "> Warning: Skipped {} row(s) in '{}' on unsupported contigs (expected 1-22, X, Y, or MT).",
            skipped_contigs.len(),
            input_path.display()
        );
        for (line_number, chr) in skipped_contigs.iter().take(MAX_SKIPPED_CONTIG_EXAMPLES) {
            eprintln!(">   - line {line_number}: chromosome '{chr}'");
        }
        if skipped_contigs.len() > MAX_SKIPPED_CONTIG_EXAMPLES {
            eprintln!(
                ">   ... and {} more.",
                skipped_contigs.len() - MAX_SKIPPED_CONTIG_EXAMPLES
            );
        }
    }

    lines_to_sort.par_sort_unstable_by_key(|item| item.key);

    crate::output::write_atomically(output_path, |writer| {
        // Write back any metadata lines that were present.
        for meta_line in &header_lines {
            writeln!(writer, "{meta_line}")?;
        }
        writeln!(writer, "{header}")?;
        for item in &lines_to_sort {
            writeln!(writer, "{}", item.line_data)?;
        }
        Ok(())
    })?;
    Ok(())
}

// ========================================================================================
//                        Private types and helpers
// ========================================================================================

struct SortableLine {
    key: (u8, u32),
    line_data: String,
}

struct SkipRecord {
    line_number: usize,
    identifier: String,
    reason: String,
}

/// The rows one chunk of a score file resolved to, in file order.
struct ResolvedChunk {
    line_count: usize,
    variant_lines: usize,
    /// Formatted native rows, each ending in `\n`.
    rows: Vec<u8>,
    sortable: Vec<SortableRow>,
    /// Skips whose `line_number` is still relative to the chunk's first line.
    skipped: Vec<SkipRecord>,
    /// Set, with nothing resolved, when the block is not UTF-8.
    invalid_utf8: Option<io::Error>,
}

/// A kept row: its sort key and where its bytes sit in `ResolvedChunk::rows`.
struct SortableRow {
    key: (u8, u32),
    chunk: usize,
    start: usize,
    end: usize,
}

/// Target size of the newline-aligned chunks a score file is resolved in.
const DATA_CHUNK_BYTES: usize = 1 << 20;

/// Splits `bytes` into consecutive spans of about `DATA_CHUNK_BYTES`, each ending
/// just past a newline (or at the end of input), so no line straddles two spans.
fn newline_aligned_spans(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut spans = Vec::with_capacity(bytes.len() / DATA_CHUNK_BYTES + 1);
    let mut start = 0;
    while start < bytes.len() {
        let mut end = (start + DATA_CHUNK_BYTES).min(bytes.len());
        if end < bytes.len() {
            end = bytes[end..]
                .iter()
                .position(|&b| b == b'\n')
                .map_or(bytes.len(), |offset| end + offset + 1);
        }
        spans.push((start, end));
        start = end;
    }
    spans
}

/// Appends the decimal digits of `value`, exactly as `Display` renders it.
fn push_decimal(out: &mut Vec<u8>, mut value: u32) {
    let mut digits = [0u8; 10];
    let mut first = digits.len();
    loop {
        first -= 1;
        digits[first] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    out.extend_from_slice(&digits[first..]);
}

/// The error `BufRead::lines` reports for the first line of `bytes` that is not
/// UTF-8, so a whole-buffer reader fails with the error the line reader gave.
fn invalid_utf8_line_error(bytes: &[u8]) -> io::Error {
    for line in BufRead::lines(bytes) {
        if let Err(error) = line {
            return error;
        }
    }
    io::Error::new(
        io::ErrorKind::InvalidData,
        "stream did not contain valid UTF-8",
    )
}

/// Recovers the non-effect allele from a PGS Catalog `variant_description` whose final
/// two colon-separated fields are the allele pair (e.g. `1:100:G:A`).
///
/// Only unambiguous cases resolve: both sides must be plain nucleotide sequences, they
/// must differ, and the effect allele must match exactly one of them.
fn other_allele_from_variant_description(description: &str, effect_allele: &str) -> Option<String> {
    let parts: Vec<&str> = description.trim().split(':').collect();
    if parts.len() < 4 {
        return None;
    }

    let is_nucleotide_sequence = |allele: &str| {
        !allele.is_empty()
            && allele
                .bytes()
                .all(|b| matches!(b.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T'))
    };

    let first = parts[parts.len() - 2].trim();
    let second = parts[parts.len() - 1].trim();
    if !is_nucleotide_sequence(first)
        || !is_nucleotide_sequence(second)
        || first.eq_ignore_ascii_case(second)
    {
        return None;
    }

    if effect_allele.eq_ignore_ascii_case(first) {
        Some(second.to_string())
    } else if effect_allele.eq_ignore_ascii_case(second) {
        Some(first.to_string())
    } else {
        None
    }
}

fn parse_key(chr_str: &str, pos_str: &str) -> Result<(u8, u32), String> {
    let chr_num = parse_chromosome_label(chr_str)?;
    let pos_num: u32 = pos_str
        .parse()
        .map_err(|e: ParseIntError| format!("Invalid position '{pos_str}': {e}"))?;

    Ok((chr_num, pos_num))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn detects_dosage_weight_format_and_skips_without_error() {
        let tmp = tempdir().expect("tempdir");
        let input_path = tmp.path().join("PGS_TEST.txt");
        let output_path = tmp.path().join("PGS_TEST.gnomon.tsv");

        let mut input = File::create(&input_path).expect("create input");
        writeln!(
            input,
            "###PGS CATALOG SCORING FILE - test\n\
#format_version=2.0\n\
#pgs_id=PGS_TEST\n\
#genome_build=hg19\n\
#HmPOS_build=GRCh38\n\
chr_name\tchr_position\teffect_allele\tother_allele\tdosage_0_weight\tdosage_1_weight\tdosage_2_weight\thm_chr\thm_pos\n\
1\t100\tA\tG\t0.0\t0.1\t0.2\t1\t100"
        )
        .expect("write input");

        let outcome = reformat_pgs_file(&input_path, &output_path).expect("reformat outcome");

        assert!(
            !outcome.wrote_output,
            "dosage-weight score should be skipped"
        );
        assert!(outcome.warning.is_some(), "expected explicit warning");
        assert!(
            !output_path.exists(),
            "no normalized output should be produced for skipped score"
        );
    }

    #[test]
    fn sort_native_file_ignores_blank_lines_before_header() {
        let tmp = tempdir().expect("tempdir");
        let input_path = tmp.path().join("scores.gnomon.tsv");
        let output_path = tmp.path().join("scores.sorted.gnomon.tsv");

        let mut input = File::create(&input_path).expect("create input");
        writeln!(
            input,
            "##synthetic=test\n\nvariant_id\teffect_allele\tother_allele\tSCORE\n1:200\tA\tG\t0.2\n1:100\tG\tA\t0.1"
        )
        .expect("write input");

        sort_native_file(&input_path, &output_path).expect("sort native file");
        let sorted = fs::read_to_string(&output_path).expect("read sorted output");
        let lines: Vec<_> = sorted.lines().collect();

        assert_eq!(lines[0], "##synthetic=test");
        assert_eq!(lines[1], "variant_id\teffect_allele\tother_allele\tSCORE");
        assert_eq!(lines[2], "1:100\tG\tA\t0.1");
        assert_eq!(lines[3], "1:200\tA\tG\t0.2");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn sort_native_file_skips_unsupported_contigs() {
        let tmp = tempdir().expect("tempdir");
        let input_path = tmp.path().join("scores.gnomon.tsv");
        let output_path = tmp.path().join("scores.sorted.gnomon.tsv");

        let mut input = File::create(&input_path).expect("create input");
        writeln!(
            input,
            "variant_id\teffect_allele\tother_allele\tSCORE\n\
8_KI270821V1_ALT:557595\tA\tG\t0.3\n\
1:200\tA\tG\t0.2\n\
UN_KI270742V1:130471\tC\tT\t0.4\n\
1:100\tG\tA\t0.1"
        )
        .expect("write input");

        sort_native_file(&input_path, &output_path).expect("sort native file");
        let sorted = fs::read_to_string(&output_path).expect("read sorted output");
        let lines: Vec<_> = sorted.lines().collect();

        assert_eq!(lines[0], "variant_id\teffect_allele\tother_allele\tSCORE");
        assert_eq!(lines[1], "1:100\tG\tA\t0.1");
        assert_eq!(lines[2], "1:200\tA\tG\t0.2");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn sort_native_file_still_rejects_malformed_positions() {
        let tmp = tempdir().expect("tempdir");
        let input_path = tmp.path().join("scores.gnomon.tsv");
        let output_path = tmp.path().join("scores.sorted.gnomon.tsv");

        let mut input = File::create(&input_path).expect("create input");
        writeln!(
            input,
            "variant_id\teffect_allele\tother_allele\tSCORE\n1:not_a_position\tA\tG\t0.2"
        )
        .expect("write input");

        assert!(sort_native_file(&input_path, &output_path).is_err());
    }

    #[test]
    fn other_allele_is_recovered_from_variant_description() {
        assert_eq!(
            other_allele_from_variant_description("1:100:G:A", "A").as_deref(),
            Some("G")
        );
        assert_eq!(
            other_allele_from_variant_description("chr1:100:G:A", "G").as_deref(),
            Some("A")
        );
        // Effect allele matches neither side, so the pairing is not recoverable.
        assert_eq!(
            other_allele_from_variant_description("1:100:G:A", "T"),
            None
        );
        // Not enough fields, and non-nucleotide alleles, stay unresolved.
        assert_eq!(other_allele_from_variant_description("1:100", "A"), None);
        assert_eq!(
            other_allele_from_variant_description("1:100:N:A", "A"),
            None
        );
    }

    #[test]
    fn reformat_recovers_other_allele_from_variant_description() {
        let tmp = tempdir().expect("tempdir");
        let input_path = tmp.path().join("PGS_DESC.txt");
        let output_path = tmp.path().join("PGS_DESC.gnomon.tsv");

        let mut input = File::create(&input_path).expect("create input");
        writeln!(
            input,
            "###PGS CATALOG SCORING FILE - test\n\
#format_version=2.0\n\
#pgs_id=PGS_DESC\n\
#genome_build=GRCh38\n\
#HmPOS_build=GRCh38\n\
chr_name\tchr_position\teffect_allele\tother_allele\tvariant_description\teffect_weight\n\
1\t100\tA\t\t1:100:G:A\t0.5\n\
1\t200\tT\t\tno_alleles_here\t0.7"
        )
        .expect("write input");

        let outcome = reformat_pgs_file(&input_path, &output_path).expect("reformat outcome");
        assert!(outcome.wrote_output, "recoverable row should be written");

        let written = fs::read_to_string(&output_path).expect("read output");
        let lines: Vec<_> = written.lines().collect();
        assert_eq!(lines[lines.len() - 1], "1:100\tA\tG\t0.5");

        let summary = outcome.skip_summary.expect("unrecoverable row is skipped");
        assert_eq!(summary.skipped_count, 1);
    }

    /// A catalog file whose rows run in descending position, so conversion must sort.
    fn catalog_text(rows: usize) -> String {
        let mut text = String::from(
            "###PGS CATALOG SCORING FILE - test\n\
#format_version=2.0\n\
#pgs_id=PGS_GZ\n\
#genome_build=GRCh38\n\
#HmPOS_build=GRCh38\n\
chr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\thm_chr\thm_pos\n",
        );
        for i in 0..rows {
            let pos = 10_000 + 7 * (rows - i);
            text.push_str(&format!("1\t{pos}\tA\tG\t0.{i:05}\t1\t{pos}\n"));
        }
        text
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut encoder =
            flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes).expect("gzip");
        encoder.finish().expect("gzip trailer")
    }

    fn bgzf(bytes: &[u8]) -> Vec<u8> {
        let mut compressed = Vec::new();
        {
            let mut writer = noodles_bgzf::io::Writer::new(&mut compressed);
            writer.write_all(bytes).expect("bgzf");
            // Dropping the writer flushes the last block and the EOF marker.
        }
        assert_eq!(&compressed[12..14], b"BC", "BGZF extra subfield");
        compressed
    }

    #[test]
    fn compressed_catalog_files_convert_exactly_like_the_plain_file() {
        let tmp = tempdir().expect("tempdir");
        // Over a megabyte: several BGZF members and more than one decode block.
        let plain = catalog_text(50_000);
        let inputs = [
            ("plain.txt", plain.as_bytes().to_vec()),
            ("gzip.txt.gz", gzip(plain.as_bytes())),
            ("bgzf.txt.gz", bgzf(plain.as_bytes())),
            ("gzip_body.txt", gzip(plain.as_bytes())),
            ("plain_body.txt.gz", plain.as_bytes().to_vec()),
        ];
        let mut outputs = Vec::new();
        for (name, bytes) in &inputs {
            let input = tmp.path().join(name);
            fs::write(&input, bytes).expect("write input");
            assert!(
                !is_gnomon_native_format(&input).expect("header check reads any encoding"),
                "{name}"
            );
            let output = tmp.path().join(format!("{name}.gnomon.tsv"));
            let outcome = reformat_pgs_file(&input, &output).expect("reformat outcome");
            assert!(outcome.wrote_output, "{name}");
            assert_eq!(outcome.score_label.as_deref(), Some("PGS_GZ"), "{name}");
            outputs.push(fs::read(&output).expect("read output"));
        }
        assert_eq!(outputs[0].iter().filter(|&&b| b == b'\n').count(), 50_001);
        for ((name, _), output) in inputs.iter().zip(&outputs).skip(1) {
            assert!(output == &outputs[0], "{name} converted differently");
        }
    }

    #[test]
    fn compressed_native_files_are_inflated_verbatim() {
        let tmp = tempdir().expect("tempdir");
        let native = "##source=test\n\
variant_id\teffect_allele\tother_allele\tSCORE_A\tSCORE_B\n\
1:200\tA\tG\t0.2\t0.1\n\
1:100\tG\tA\t0.1\t0.3\n";
        let plain_input = tmp.path().join("native.tsv");
        fs::write(&plain_input, native).expect("write input");
        assert!(is_gnomon_native_format(&plain_input).expect("header check"));

        for (name, bytes) in [
            ("native.tsv.gz", gzip(native.as_bytes())),
            ("native.tsv.bgz", bgzf(native.as_bytes())),
        ] {
            let input = tmp.path().join(name);
            fs::write(&input, bytes).expect("write input");
            assert!(
                !is_gnomon_native_format(&input).expect("header check"),
                "{name} must be inflated before use"
            );
            let output = tmp.path().join(format!("{name}.gnomon.tsv"));
            let outcome = reformat_pgs_file(&input, &output).expect("inflate native file");
            assert!(outcome.wrote_output, "{name}");
            assert_eq!(outcome.score_label.as_deref(), Some("SCORE_A"), "{name}");
            assert_eq!(fs::read_to_string(&output).expect("read output"), native);
        }
    }

    #[test]
    fn truncated_gzip_score_file_fails_without_writing_output() {
        let tmp = tempdir().expect("tempdir");
        let compressed = gzip(catalog_text(5_000).as_bytes());
        let input = tmp.path().join("truncated.txt.gz");
        fs::write(&input, &compressed[..compressed.len() / 2]).expect("write input");
        let output = tmp.path().join("truncated.gnomon.tsv");
        assert!(reformat_pgs_file(&input, &output).is_err());
        assert!(
            !output.exists(),
            "a partial download must not become a score file"
        );
    }
}
