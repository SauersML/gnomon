// ========================================================================================
//
//               PGS CATALOG SCORE FILE DIAGNOSTICS & REFORMATTING
//
// ========================================================================================

use flate2::read::MultiGzDecoder;
use rayon::prelude::*;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::num::ParseIntError;
use std::path::{Path, PathBuf};

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
	},
	/// A value in the file could not be parsed correctly.
	Parse {
		path: PathBuf,
		line_number: usize,
		line_content: String,
		details: String,
	},
}

impl Display for ReformatError {
	fn fmt(&self, f: &mut Formatter) -> fmt::Result {
		// This implementation provides a detailed, multi-line diagnostic report to the user.
		writeln!(f, "Failed to reformat PGS Catalog score file.")?;
		writeln!(f, "\n-- Diagnostic Details ----------------------------------------------------")?;

		match self {
			ReformatError::Io(e) => {
				writeln!(f, "Reason:       An I/O error occurred.")?;
				writeln!(f, "Details:      {}", e)?;
			}
			ReformatError::NotPgsFormat { path } => {
				writeln!(f, "File:         {}", path.display())?;
				writeln!(f, "Reason:       The file is not a valid PGS Catalog scoring file.")?;
				writeln!(f, "Details:      The required signature ('###PGS CATALOG SCORING FILE') was not found in the file's metadata header.")?;
			}
			ReformatError::MissingColumns { path, line_number, line_content, missing_column_name } => {
				writeln!(f, "File:         {}", path.display())?;
				writeln!(f, "Line Number:  {}", line_number)?;
				writeln!(f, "Line Content: \"{}\"", line_content.trim())?;
				writeln!(f, "Reason:       A required column or its data is missing.")?;
				writeln!(f, "Details:      Expected to find column '{}', but it was not found in the header or data for this line.", missing_column_name)?;
			}
			ReformatError::Parse { path, line_number, line_content, details } => {
				writeln!(f, "File:         {}", path.display())?;
				writeln!(f, "Line Number:  {}", line_number)?;
				writeln!(f, "Line Content: \"{}\"", line_content.trim())?;
				writeln!(f, "Reason:       A value in the line could not be parsed correctly.")?;
				writeln!(f, "Details:      {}", details)?;
			}
		}
		write!(f, "--------------------------------------------------------------------------")
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
    // --- Define helper types within the function scope ---
    #[derive(Clone, Copy)]
    enum ParsingStrategy {
        OriginalOnly,    // Use chr_name/chr_position, fail if unmappable.
        HarmonizedOnly,  // Use hm_chr/hm_pos, fail if unmappable.
        SafeFallback,    // Try hm_chr/hm_pos, fall back to orig on parse failure.
    }

    struct ColumnIndices {
        chr: Option<usize>,
        pos: Option<usize>,
        hm_chr: Option<usize>,
        hm_pos: Option<usize>,
        ea: usize, // effect_allele is mandatory
        ew: usize, // effect_weight is mandatory
        oa: Option<usize>, // other_allele is optional
    }

    // --- Open file and handle potential GZIP compression ---
    let file = File::open(input_path)?;
    let a_reader: Box<dyn Read + Send> = if input_path.extension().map_or(false, |ext| ext == "gz") {
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut reader = BufReader::new(a_reader);
    let mut line_buffer = String::new();

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

    loop {
        line_buffer.clear();
        if reader.read_line(&mut line_buffer)? == 0 {
            // Reached EOF before finding the data header
            return Err(ReformatError::NotPgsFormat { path: input_path.to_path_buf() });
        }
        if !line_buffer.starts_with('#') {
            break; // Found the data header line
        }

        let metadata = line_buffer.trim_start_matches('#').trim();
        if let Some(val) = metadata.strip_prefix("genome_build=") {
            orig_build_norm = normalize_build(val);
        } else if let Some(val) = metadata.strip_prefix("HmPOS_build=") {
            hm_build_norm = normalize_build(val);
        } else if let Some(val) = metadata.strip_prefix("pgs_id=") {
            score_id = Some(val.to_string());
        }
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

    let column_indices = ColumnIndices {
        chr: header_map.get("chr_name").copied(),
        pos: header_map.get("chr_position").copied(),
        hm_chr: header_map.get("hm_chr").copied(),
        hm_pos: header_map.get("hm_pos").copied(),
        ea: *header_map.get("effect_allele").ok_or_else(|| ReformatError::MissingColumns {
            path: input_path.to_path_buf(), line_number: 0, line_content: header_line.to_string(), missing_column_name: "effect_allele".to_string()
        })?,
        ew: *header_map.get("effect_weight").ok_or_else(|| ReformatError::MissingColumns {
            path: input_path.to_path_buf(), line_number: 0, line_content: header_line.to_string(), missing_column_name: "effect_weight".to_string()
        })?,
        oa: header_map.get("other_allele").or_else(|| header_map.get("hm_inferOtherAllele")).copied(),
    };

    // This is the "hot path". We use the pre-selected strategy and indices to
    // process all data lines in parallel with minimal branching or overhead.
    let score_label = score_id.unwrap_or_else(|| "PGS_SCORE".to_string());

    // --- Define the resolver closure based on the chosen strategy ---
    // The `move` keyword captures the `column_indices` struct by value.
    let resolver = move |line: &str| -> Result<Option<SortableLine>, ReformatError> {
        let fields: Vec<&str> = line.split('\t').collect();

        let get_key = |c_idx, p_idx| {
            if let (Some(chr_idx), Some(pos_idx)) = (c_idx, p_idx) {
                if let (Some(chr_str), Some(pos_str)) = (fields.get(chr_idx), fields.get(pos_idx)) {
                    if !chr_str.is_empty() && !pos_str.is_empty() {
                        return parse_key(chr_str, pos_str).ok();
                    }
                }
            }
            None
        };

        // This is the core logic, applying the pre-determined strategy.
        let key = match strategy {
            ParsingStrategy::SafeFallback => {
                get_key(column_indices.hm_chr, column_indices.hm_pos)
                    .or_else(|| get_key(column_indices.chr, column_indices.pos))
            }
            ParsingStrategy::HarmonizedOnly => get_key(column_indices.hm_chr, column_indices.hm_pos),
            ParsingStrategy::OriginalOnly => get_key(column_indices.chr, column_indices.pos),
        };

        // If no valid key could be derived, skip this line by returning Ok(None).
        let Some(key) = key else { return Ok(None) };

        // Safely extract other mandatory fields.
        let Some(ea_str) = fields.get(column_indices.ea).map(|s| *s) else { return Ok(None) };
        let Some(weight_str) = fields.get(column_indices.ew).map(|s| *s) else { return Ok(None) };
        let oa_str = column_indices.oa.and_then(|i| fields.get(i)).map_or("N", |s| if s.is_empty() { "N" } else { *s });

        let variant_id = format!("{}:{}", key.0, key.1);
        let line_data = format!("{}\t{}\t{}\t{}", variant_id, ea_str, oa_str, weight_str);

        Ok(Some(SortableLine { key, line_data }))
    };

    // --- Read all data lines and process them in parallel ---
    let data_lines: Vec<String> = reader.lines().filter_map(Result::ok).collect();
    let skipped_count = std::sync::atomic::AtomicUsize::new(0);

    let mut lines_to_sort = data_lines
        .into_par_iter()
        .filter_map(|line| {
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            match resolver(&line) {
                Ok(Some(sortable_line)) => Some(Ok(sortable_line)),
                Ok(None) => {
                    // This is a safe skip for an unmappable variant. Count it.
                    skipped_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    None
                }
                Err(e) => Some(Err(e)), // This is a fatal error.
            }
        })
        .collect::<Result<Vec<_>, ReformatError>>()?;



    // Sort the resolved data and write it to the gnomon-native file.
    lines_to_sort.par_sort_unstable_by_key(|item| item.key);

    let out_file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, out_file);
    writeln!(writer, "variant_id\teffect_allele\tother_allele\t{}", score_label)?;
    for item in lines_to_sort {
        writeln!(writer, "{}", item.line_data)?;
    }
    writer.flush()?;

    // --- Report any non-fatal issues to the user ---
    let final_skipped_count = skipped_count.into_inner();
    if final_skipped_count > 0 {
        eprintln!(
            "> Warning: Skipped {} variant(s) from '{}' due to unmappable or incomplete coordinates.",
            final_skipped_count,
            input_path.display()
        );
    }

    Ok(())
}

/// Sorts a gnomon-native file that is not guaranteed to be sorted.
pub fn sort_native_file(input_path: &Path, output_path: &Path) -> Result<(), ReformatError> {
	let file = File::open(input_path)?;
	let reader = BufReader::new(file);
	
	// Separate header lines from data lines to correctly track data line numbers.
	let mut header_lines: Vec<String> = vec![];
	let mut data_lines: Vec<String> = vec![];

	for line_result in reader.lines() {
		let line = line_result.map_err(ReformatError::Io)?;
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
			
			// Call the decoupled parse_key function and map its simple error to our rich error type.
			let key = parse_key(chr_str, pos_str).map_err(|details| ReformatError::Parse {
				path: input_path.to_path_buf(),
				line_number,
				line_content: line.clone(),
				details,
			})?;
			
			Ok(Some(SortableLine { key, line_data: line }))
		})
		.filter_map(|result| result.transpose())
		.collect::<Result<_, ReformatError>>()?;

	lines_to_sort.par_sort_unstable_by_key(|item| item.key);

	let out_file = File::create(output_path)?;
	let mut writer = BufWriter::new(out_file);
	// Write back any metadata lines that were present.
	for meta_line in header_lines {
		writeln!(writer, "{}", meta_line)?;
	}
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

fn parse_key(chr_str: &str, pos_str: &str) -> Result<(u8, u32), String> {
	// First, check for special, non-numeric chromosome names case-insensitively.
	if chr_str.eq_ignore_ascii_case("X") {
		let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| format!("Invalid position '{}': {}", pos_str, e))?;
		return Ok((23, pos_num));
	}
	if chr_str.eq_ignore_ascii_case("Y") {
		let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| format!("Invalid position '{}': {}", pos_str, e))?;
		return Ok((24, pos_num));
	}
	if chr_str.eq_ignore_ascii_case("MT") {
		let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| format!("Invalid position '{}': {}", pos_str, e))?;
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
		format!(
			"Invalid chromosome format '{}'. Expected a number, 'X', 'Y', 'MT', or 'chr' prefix.",
			chr_str
		)
	})?;
	
	let pos_num: u32 = pos_str.parse().map_err(|e: ParseIntError| format!("Invalid position '{}': {}", pos_str, e))?;
	
	Ok((chr_num, pos_num))
}
