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
    let file = File::open(input_path)?;

	// Create a reader that can dynamically handle both plain text and gzipped files.
    // This is done by using a "trait object" which can hold any type that implements `io::Read`.
    let a_reader: Box<dyn Read + Send> = if input_path.extension().map_or(false, |ext| ext == "gz") {
        // Use MultiGzDecoder to correctly read gzip files that may be multi-member
        Box::new(MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut reader = BufReader::with_capacity(1 << 20, a_reader);
    let mut line = String::new();
    let mut saw_signature = false;
    let mut score_name: Option<String> = None;

	// Loop through all metadata lines (those starting with '#'), robustly handling blank lines.
	while reader.read_line(&mut line)? > 0 {
		// Trim both leading/trailing whitespace and newlines to get the actual content.
		let trimmed_line = line.trim();

		// If the line is completely blank after trimming, we clear the buffer and
		// continue to the next line, skipping any further processing for this line.
		if trimmed_line.is_empty() {
			line.clear();
			continue;
		}

		// If the line's content does not start with a '#', we have found the header.
		// We break the loop, leaving the original `line` buffer intact for parsing.
		if !trimmed_line.starts_with('#') {
			break;
		}

		// --- Process Metadata Line ---
		// The line is a comment. Process its content by removing the leading '#' characters.
		let metadata_content = trimmed_line.trim_start_matches('#').trim();

		// Check for the mandatory PGS Catalog signature.
		if metadata_content.starts_with("PGS CATALOG SCORING FILE") {
			saw_signature = true;
		}

		// Extract the PGS ID to use as the score name.
		if let Some(id) = metadata_content.strip_prefix("pgs_id=") {
			score_name = Some(id.trim().to_string());
		}

		// Clear the main buffer to prepare for reading the next line.
		line.clear();
	}

// After processing all metadata, verify that the signature was found.
	if !saw_signature {
		return Err(ReformatError::NotPgsFormat { path: input_path.to_path_buf() });
	}
	let header_line = line.trim_end().to_string();
	let score_label = score_name.unwrap_or_else(|| "PGS_SCORE".into());

	// --- Robust Header Parsing ---
	// Create a map from column names to their index for robust, order-independent access.
	let header_map: std::collections::HashMap<&str, usize> = header_line
		 .split('\t')
		.enumerate()
		.map(|(i, name)| (name, i))
		.collect();

	// Get indices for all potentially required columns, prioritizing harmonized `hm_` columns.
	let chr_idx = *header_map.get("hm_chr").or_else(|| header_map.get("chr_name"))
		.ok_or_else(|| ReformatError::MissingColumns {
			path: input_path.to_path_buf(),
			line_number: 1, // Header is effectively line 1 of the data.
			line_content: header_line.clone(),
			missing_column_name: "(hm_chr or chr_name)".to_string(),
		})?;
	let pos_idx = *header_map.get("hm_pos").or_else(|| header_map.get("chr_position"))
		.ok_or_else(|| ReformatError::MissingColumns {
			path: input_path.to_path_buf(),
			line_number: 1,
			line_content: header_line.clone(),
			missing_column_name: "(hm_pos or chr_position)".to_string(),
		})?;
	let ea_idx = *header_map.get("effect_allele")
		.ok_or_else(|| ReformatError::MissingColumns {
			path: input_path.to_path_buf(),
			line_number: 1,
			line_content: header_line.clone(),
			missing_column_name: "effect_allele".to_string(),
		})?;
	let ew_idx = *header_map.get("effect_weight")
		.ok_or_else(|| ReformatError::MissingColumns {
			path: input_path.to_path_buf(),
			line_number: 1,
			line_content: header_line.clone(),
			missing_column_name: "effect_weight".to_string(),
		})?;
	// The "other allele" is optional; we check for both names and store the index if found.
	let oa_idx = header_map.get("other_allele").or_else(|| header_map.get("hm_inferOtherAllele")).copied();

	// --- Robust Data Row Parsing ---
	// The header has been skipped; subsequent lines are data.
	let mut lines_to_sort: Vec<SortableLine> = reader
		.lines()
		.enumerate()
		.par_bridge()
		.map(|(i, line_result)| {
			// The current line number in the file (add 2 to account for 0-based index and header).
			let line_number = i + 2;
			let line = line_result.map_err(ReformatError::Io)?;

			if line.is_empty() || line.starts_with('#') {
				return Ok(None);
			}

			let fields: Vec<&str> = line.split('\t').collect();

			// Safely get data from fields by index. If a field is missing, `get` returns `None`.
			let chr_str = fields.get(chr_idx).filter(|s| !s.is_empty()).ok_or_else(|| ReformatError::MissingColumns {
				path: input_path.to_path_buf(), line_number, line_content: line.clone(), missing_column_name: "chromosome".to_string()
			})?;
			let pos_str = fields.get(pos_idx).filter(|s| !s.is_empty()).ok_or_else(|| ReformatError::MissingColumns {
				path: input_path.to_path_buf(), line_number, line_content: line.clone(), missing_column_name: "position".to_string()
			})?;
			let ea_str = fields.get(ea_idx).filter(|s| !s.is_empty()).ok_or_else(|| ReformatError::MissingColumns {
				path: input_path.to_path_buf(), line_number, line_content: line.clone(), missing_column_name: "effect_allele".to_string()
			})?;
			let weight_str = fields.get(ew_idx).filter(|s| !s.is_empty()).ok_or_else(|| ReformatError::MissingColumns {
				path: input_path.to_path_buf(), line_number, line_content: line.clone(), missing_column_name: "effect_weight".to_string()
			})?;

			// The "other allele" is not strictly required; use a placeholder 'N' if absent.
			let oa_str = oa_idx.and_then(|i| fields.get(i)).map_or("N", |s| if s.is_empty() { "N" } else { s });

			let key = parse_key(chr_str, pos_str).map_err(|details| ReformatError::Parse {
				path: input_path.to_path_buf(), line_number, line_content: line.clone(), details
			})?;

			let line_data = format!("{}:{}\t{}\t{}\t{}", chr_str, pos_str, ea_str, oa_str, weight_str);
			Ok(Some(SortableLine { key, line_data }))
		})
		.filter_map(|result: Result<Option<SortableLine>, ReformatError>| result.transpose())
		.collect::<Result<Vec<_>, ReformatError>>()?;

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
