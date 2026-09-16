//! VCF and BCF headers, read as bcftools and plink2 read them.
//!
//! VCF 4.3 reserves a definition for some FORMAT and INFO keys (`GP` is
//! `Number=G,Type=Float`), and noodles refuses a header that declares one of
//! them differently, with every record then failing as `invalid record`. Files
//! like that are common: imputation servers declare `GP` with `Number=.`, and
//! nothing downstream reads the declared count, since every value list is
//! split as written. The readers here parse such a header with the check
//! relaxed and say so once; every other header error stands.

use std::error::Error;
use std::io::{self, BufRead, Read};
use std::path::Path;

use noodles_bcf as bcf;
use noodles_vcf::{
    self as vcf,
    header::{FileFormat, StringMaps, parser::FileFormatOption},
};

/// A parsed header, and the definition mismatch it was read past, if any.
pub struct HeaderRead {
    pub header: vcf::Header,
    /// The reserved-definition mismatch the strict parse refused, as noodles
    /// states it, when the header was accepted with that check relaxed.
    pub relaxed: Option<String>,
}

impl HeaderRead {
    /// The header, after printing the relaxation, if any, against `path`.
    pub fn warned(self, path: &Path) -> vcf::Header {
        if let Some(note) = &self.relaxed {
            eprintln!(
                "Warning: {}: the header declares a reserved key against its VCF {}.{} definition \
                 ({note}); values are read as written",
                path.display(),
                self.header.file_format().major(),
                self.header.file_format().minor(),
            );
        }
        self.header
    }
}

/// Reads a VCF header from the start of `reader`'s stream.
pub fn read_vcf_header<R: BufRead>(reader: &mut vcf::io::Reader<R>) -> io::Result<HeaderRead> {
    let mut text = String::new();
    reader.header_reader().read_to_string(&mut text)?;
    parse_header_text(&text, false)
}

/// Reads a BCF header from the start of `reader`'s stream: the magic number,
/// the format version, and the length-prefixed VCF header text.
pub fn read_bcf_header<R: Read>(reader: &mut bcf::io::Reader<R>) -> io::Result<HeaderRead> {
    let inner = reader.get_mut();
    let mut magic = [0u8; 5];
    inner.read_exact(&mut magic)?;
    if &magic[..3] != b"BCF" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid BCF magic number",
        ));
    }
    if magic[3] != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported BCF format version {}.{}", magic[3], magic[4]),
        ));
    }
    let mut len = [0u8; 4];
    inner.read_exact(&mut len)?;
    let len = usize::try_from(u32::from_le_bytes(len)).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "BCF header length overflows")
    })?;
    let mut raw = vec![0u8; len];
    inner.read_exact(&mut raw)?;
    // The text is NUL-padded to its declared length.
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    let text = std::str::from_utf8(&raw[..end])
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    parse_header_text(text, true)
}

/// Parses header text strictly, and again with the reserved-definition check
/// relaxed when that is all the strict parse refused.
pub fn parse_header_text(text: &str, with_string_maps: bool) -> io::Result<HeaderRead> {
    match parse_with(text, vcf::header::Parser::default(), with_string_maps) {
        Ok(header) => Ok(HeaderRead {
            header,
            relaxed: None,
        }),
        Err(err) => {
            let Some(mismatch) = definition_mismatch(&err) else {
                return Err(err);
            };
            // Parsing as VCF 4.2 is the one way noodles offers to skip the
            // check; nothing else in its header parser depends on the version.
            // The header keeps the version the file declares.
            let declared = declared_file_format(text)?;
            let parser = vcf::header::Parser::builder()
                .set_file_format_option(FileFormatOption::FileFormat(FileFormat::new(4, 2)))
                .build();
            let mut header = parse_with(text, parser, with_string_maps)?;
            *header.file_format_mut() = declared;
            Ok(HeaderRead {
                header,
                relaxed: Some(mismatch),
            })
        }
    }
}

fn parse_with(
    text: &str,
    mut parser: vcf::header::Parser,
    with_string_maps: bool,
) -> io::Result<vcf::Header> {
    let invalid = |err: Box<dyn Error + Send + Sync>| io::Error::new(io::ErrorKind::InvalidData, err);
    let mut string_maps = StringMaps::default();
    for line in text.lines() {
        let entry = parser
            .parse_partial(line.as_bytes())
            .map_err(|err| invalid(Box::new(err)))?;
        if with_string_maps {
            string_maps
                .insert_entry(&entry)
                .map_err(|err| invalid(Box::new(err)))?;
        }
    }
    let mut header = parser.finish().map_err(|err| invalid(Box::new(err)))?;
    if with_string_maps {
        *header.string_maps_mut() = string_maps;
    }
    Ok(header)
}

/// The reserved-definition mismatch inside a header parse error, if that is
/// what it is: the deepest cause names the key and both definitions.
fn definition_mismatch(err: &io::Error) -> Option<String> {
    let mut deepest: &dyn Error = err;
    while let Some(cause) = deepest.source() {
        deepest = cause;
    }
    let message = deepest.to_string();
    message.contains("definition mismatch").then_some(message)
}

fn declared_file_format(text: &str) -> io::Result<FileFormat> {
    let first = text.lines().next().unwrap_or_default();
    let version = first
        .strip_prefix("##fileformat=VCFv")
        .and_then(|v| v.trim().split_once('.'))
        .and_then(|(major, minor)| Some((major.parse().ok()?, minor.parse().ok()?)));
    match version {
        Some((major, minor)) => Ok(FileFormat::new(major, minor)),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("header does not open with a file format line: {first:?}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GP_UNKNOWN: &str = "##fileformat=VCFv4.3\n##contig=<ID=1>\n\
##FORMAT=<ID=GT,Number=1,Type=String,Description=\"x\">\n\
##FORMAT=<ID=GP,Number=.,Type=Float,Description=\"x\">\n\
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n";

    #[test]
    fn a_reserved_definition_mismatch_is_read_past_and_named() {
        let read = parse_header_text(GP_UNKNOWN, true).unwrap();
        let note = read.relaxed.expect("relaxed");
        assert!(note.contains("ID=GP"), "{note}");
        assert_eq!(read.header.file_format(), FileFormat::new(4, 3));
        assert!(read.header.formats().contains_key("GP"));
        assert_eq!(read.header.sample_names().len(), 2);
        assert!(read.header.string_maps().strings().get_index_of("GT").is_some());
    }

    #[test]
    fn a_conforming_header_is_not_relaxed() {
        let text = GP_UNKNOWN.replace("Number=.,Type=Float", "Number=G,Type=Float");
        let read = parse_header_text(&text, false).unwrap();
        assert!(read.relaxed.is_none());
        let text = GP_UNKNOWN.replace("VCFv4.3", "VCFv4.2");
        assert!(parse_header_text(&text, false).unwrap().relaxed.is_none());
    }

    #[test]
    fn other_header_errors_still_fail() {
        let text = GP_UNKNOWN.replace("Number=.,Type=Float", "Type=Float");
        assert!(parse_header_text(&text, false).is_err());
    }

    #[test]
    fn vcf_and_bcf_readers_hand_back_the_same_header() {
        let mut vcf_reader = vcf::io::Reader::new(GP_UNKNOWN.as_bytes());
        let from_vcf = read_vcf_header(&mut vcf_reader).unwrap();
        assert!(from_vcf.relaxed.is_some());
        assert_eq!(from_vcf.header.sample_names().len(), 2);

        let mut bytes = b"BCF\x02\x02".to_vec();
        let text = format!("{GP_UNKNOWN}\0");
        bytes.extend((text.len() as u32).to_le_bytes());
        bytes.extend(text.as_bytes());
        let mut bcf_reader = bcf::io::Reader::from(bytes.as_slice());
        let from_bcf = read_bcf_header(&mut bcf_reader).unwrap();
        assert!(from_bcf.relaxed.is_some());
        assert_eq!(from_bcf.header.sample_names(), from_vcf.header.sample_names());
        assert!(from_bcf.header.string_maps().strings().get_index_of("GP").is_some());
    }
}
