//! The GT series of a BCF record, read straight from the record's bytes.
//!
//! BCF stores a genotype as a vector of integer allele codes, one series entry
//! per sample, and the writer picks the narrowest integer width that holds
//! every code in the record: 8-bit for the usual few alleles, 16- or 32-bit
//! once an allele index passes 62, or whenever a writer uses one width for
//! every record. All three are valid BCF 2.2. noodles-bcf (through 0.91)
//! decodes only the 8-bit form and panics on the others with
//! `unhandled type`, so the readers here parse the series themselves.
//!
//! Each code is `(allele + 1) << 1 | phased`; a missing allele (`.`) is `0`, a
//! sample with fewer alleles than the record's widest is padded with the
//! width's end-of-vector sentinel, and the width's missing sentinel may stand
//! in for a missing allele as well.

use std::fmt;

/// A typed-value width the BCF encoding allows for a genotype vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    Int8,
    Int16,
    Int32,
}

impl Width {
    fn bytes(self) -> usize {
        match self {
            Width::Int8 => 1,
            Width::Int16 => 2,
            Width::Int32 => 4,
        }
    }

    /// Reads one code, mapping the width's sentinels to their meaning.
    fn code(self, bytes: &[u8]) -> Code {
        let (value, missing, end) = match self {
            Width::Int8 => (i64::from(bytes[0] as i8), i64::from(i8::MIN), i64::from(i8::MIN + 1)),
            Width::Int16 => (
                i64::from(i16::from_le_bytes([bytes[0], bytes[1]])),
                i64::from(i16::MIN),
                i64::from(i16::MIN + 1),
            ),
            Width::Int32 => (
                i64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
                i64::from(i32::MIN),
                i64::from(i32::MIN + 1),
            ),
        };
        if value == end {
            Code::EndOfVector
        } else if value == missing || value == 0 {
            Code::MissingAllele
        } else if value < 0 {
            Code::Invalid(value)
        } else {
            // `(allele + 1) << 1 | phased`, so the allele index is one below
            // the value's upper bits.
            Code::Allele((value >> 1) as usize - 1)
        }
    }
}

enum Code {
    Allele(usize),
    MissingAllele,
    EndOfVector,
    Invalid(i64),
}

/// The GT series of one BCF record.
#[derive(Clone, Copy, Debug)]
pub struct GenotypeSeries<'a> {
    width: Width,
    /// Codes per sample, the record's widest ploidy.
    len: usize,
    sample_count: usize,
    values: &'a [u8],
}

/// Why a record's sample block could not be read as a series list.
#[derive(Debug, PartialEq, Eq)]
pub struct MalformedSamples(String);

impl fmt::Display for MalformedSamples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed BCF sample block: {}", self.0)
    }
}

impl std::error::Error for MalformedSamples {}

/// One allele of a sample's genotype: its index (REF is 0), or `None` for `.`.
pub type Allele = Option<usize>;

impl<'a> GenotypeSeries<'a> {
    /// Finds the GT series in a record's sample block.
    ///
    /// `block` is the record's whole individual (sample) block, `format_count`
    /// its series count and `sample_count` the record's sample count, all as
    /// the record states them. `gt_key` is the header string-map index of `GT`.
    /// Returns `None` when the record carries no GT series.
    pub fn find(
        block: &'a [u8],
        format_count: usize,
        sample_count: usize,
        gt_key: usize,
    ) -> Result<Option<Self>, MalformedSamples> {
        let mut cursor = Cursor { src: block, pos: 0 };
        for _ in 0..format_count {
            let key = cursor.typed_int()?;
            let (ty, len) = cursor.descriptor()?;
            let unit = match ty {
                1 | 7 => 1,
                2 => 2,
                3 | 5 => 4,
                0 if len == 0 => 0,
                other => {
                    return Err(MalformedSamples(format!(
                        "FORMAT series has unknown value type {other}"
                    )));
                }
            };
            let size = sample_count
                .checked_mul(len)
                .and_then(|n| n.checked_mul(unit))
                .ok_or_else(|| MalformedSamples("FORMAT series size overflows".to_string()))?;
            let values = cursor.take(size)?;
            if key == gt_key {
                let width = match ty {
                    1 => Width::Int8,
                    2 => Width::Int16,
                    3 => Width::Int32,
                    other => {
                        return Err(MalformedSamples(format!(
                            "GT series has value type {other}; a genotype is an integer vector"
                        )));
                    }
                };
                return Ok(Some(GenotypeSeries {
                    width,
                    len,
                    sample_count,
                    values,
                }));
            }
        }
        Ok(None)
    }

    /// The record's sample count, as it states it.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// The alleles of sample `i`, in order, ending at the sample's own ploidy.
    ///
    /// Returns `None` when `i` is past the record's samples. A negative code
    /// other than the width's sentinels is not a genotype and is an error.
    pub fn alleles(&self, i: usize) -> Option<impl Iterator<Item = Result<Allele, MalformedSamples>> + '_> {
        if i >= self.sample_count {
            return None;
        }
        let unit = self.width.bytes();
        let start = i * self.len * unit;
        let end = start + self.len * unit;
        let sample = self.values.get(start..end)?;
        let width = self.width;
        let mut ended = false;
        Some(sample.chunks_exact(unit).filter_map(move |bytes| {
            if ended {
                return None;
            }
            match width.code(bytes) {
                Code::Allele(index) => Some(Ok(Some(index))),
                Code::MissingAllele => Some(Ok(None)),
                Code::EndOfVector => {
                    ended = true;
                    None
                }
                Code::Invalid(value) => Some(Err(MalformedSamples(format!(
                    "GT code {value} is not an allele"
                )))),
            }
        }))
    }
}

struct Cursor<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], MalformedSamples> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.src.len())
            .ok_or_else(|| {
                MalformedSamples(format!(
                    "series needs {n} bytes at offset {} of a {}-byte block",
                    self.pos,
                    self.src.len()
                ))
            })?;
        let out = &self.src[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// A typed-value descriptor: the low nibble is the type, the high nibble
    /// the count, and a count of 15 means the true count follows as an int.
    fn descriptor(&mut self) -> Result<(u8, usize), MalformedSamples> {
        let byte = self.take(1)?[0];
        let ty = byte & 0x0f;
        let count = usize::from(byte >> 4);
        if count == 15 {
            let count = self.typed_int()?;
            Ok((ty, count))
        } else {
            Ok((ty, count))
        }
    }

    /// A single typed integer, as the series key and long counts are stored.
    fn typed_int(&mut self) -> Result<usize, MalformedSamples> {
        let (ty, count) = self.descriptor()?;
        if count != 1 {
            return Err(MalformedSamples(format!(
                "typed integer has count {count}"
            )));
        }
        let value = match ty {
            1 => i64::from(self.take(1)?[0] as i8),
            2 => {
                let b = self.take(2)?;
                i64::from(i16::from_le_bytes([b[0], b[1]]))
            }
            3 => {
                let b = self.take(4)?;
                i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            }
            other => {
                return Err(MalformedSamples(format!(
                    "typed integer has value type {other}"
                )));
            }
        };
        usize::try_from(value)
            .map_err(|_| MalformedSamples(format!("typed integer {value} is negative")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sample block with one GT series (key 1) at the given width, plus a
    /// trailing int8 series (key 2) so the scan has something to skip past.
    fn block(width: u8, calls: &[&[i64]], ploidy: usize) -> Vec<u8> {
        let (ty, eov): (u8, i64) = match width {
            8 => (1, i64::from(i8::MIN + 1)),
            16 => (2, i64::from(i16::MIN + 1)),
            32 => (3, i64::from(i32::MIN + 1)),
            _ => unreachable!(),
        };
        let mut out = vec![0x11, 1, (ploidy as u8) << 4 | ty];
        for call in calls {
            let mut codes: Vec<i64> = call.iter().map(|&a| if a < 0 { 0 } else { (a + 1) << 1 }).collect();
            codes.resize(ploidy, eov);
            for code in codes {
                match width {
                    8 => out.push(code as i8 as u8),
                    16 => out.extend((code as i16).to_le_bytes()),
                    _ => out.extend((code as i32).to_le_bytes()),
                }
            }
        }
        out.extend([0x11, 2, 0x11]);
        out.extend(std::iter::repeat_n(7u8, calls.len()));
        out
    }

    fn decoded(series: GenotypeSeries<'_>, n: usize) -> Vec<Vec<Allele>> {
        (0..n)
            .map(|i| series.alleles(i).unwrap().map(Result::unwrap).collect())
            .collect()
    }

    #[test]
    fn every_width_decodes_to_the_same_alleles() {
        // Allele 60 is the widest code int8 holds ((60 + 1) << 1 = 122).
        let calls: &[&[i64]] = &[&[0, 1], &[1, 1], &[-1, -1], &[1], &[0, 60]];
        let expect = vec![
            vec![Some(0), Some(1)],
            vec![Some(1), Some(1)],
            vec![None, None],
            vec![Some(1)],
            vec![Some(0), Some(60)],
        ];
        for width in [8, 16, 32] {
            let bytes = block(width, calls, 2);
            let series = GenotypeSeries::find(&bytes, 2, calls.len(), 1)
                .unwrap()
                .expect("GT present");
            assert_eq!(decoded(series, calls.len()), expect, "width {width}");
            assert!(series.alleles(calls.len()).is_none());
        }
    }

    #[test]
    fn alleles_past_the_int8_range_need_the_wider_codes() {
        // Allele 70's code is 142: what makes a writer pick int16 in the first place.
        let calls: &[&[i64]] = &[&[0, 70], &[70, 70]];
        let expect = vec![vec![Some(0), Some(70)], vec![Some(70), Some(70)]];
        for width in [16, 32] {
            let bytes = block(width, calls, 2);
            let series = GenotypeSeries::find(&bytes, 2, calls.len(), 1)
                .unwrap()
                .expect("GT present");
            assert_eq!(decoded(series, calls.len()), expect, "width {width}");
        }
    }

    #[test]
    fn absent_gt_and_wrong_key_are_none() {
        let bytes = block(8, &[&[0, 0]], 2);
        assert!(GenotypeSeries::find(&bytes, 2, 1, 9).unwrap().is_none());
        assert!(GenotypeSeries::find(&bytes, 0, 1, 1).unwrap().is_none());
    }

    #[test]
    fn a_truncated_block_is_an_error() {
        let bytes = block(16, &[&[0, 1], &[1, 1]], 2);
        assert!(GenotypeSeries::find(&bytes[..bytes.len() - 6], 2, 2, 1).is_err());
        assert!(GenotypeSeries::find(&bytes[..5], 2, 2, 1).is_err());
    }

    #[test]
    fn long_counts_and_wide_keys_are_read() {
        // Key stored as int16 (0x12), then a ploidy of 15 stored through the
        // long-count form: descriptor 0xF1, count as int8 15.
        let mut out = vec![0x12, 1, 0, 0xF1, 0x11, 15];
        let mut codes = vec![2i8, 4];
        codes.resize(15, i8::MIN + 1);
        out.extend(codes.iter().map(|&c| c as u8));
        let series = GenotypeSeries::find(&out, 1, 1, 1).unwrap().unwrap();
        assert_eq!(decoded(series, 1), vec![vec![Some(0), Some(1)]]);
    }
}
