//! Dosages as exact decimals. A VCF DS or GP value is its written digits when it is a plain
//! decimal of at most fifteen digits, and otherwise the shortest decimal that reads back as the
//! same double, the rule weights follow too; a BCF float is the shortest decimal that reads back
//! as the same f32. The DS and GP rules are evaluated on integers, so a dosage enters a score as
//! the number it is.

use super::DecodedAltDosage;
use crate::score::exact::{shortest_decimal, shortest_decimal_f32};
use std::error::Error;

type DecodeError = Box<dyn Error + Send + Sync>;

/// `digits × 10^-places`, or the missing dosage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Dose {
    pub(super) digits: i64,
    pub(super) places: u8,
}

impl Dose {
    pub(super) const MISSING: Self = Self {
        digits: 0,
        places: u8::MAX,
    };

    /// Whole allele copies.
    pub(super) const fn copies(count: u8) -> Self {
        Self {
            digits: count as i64,
            places: 0,
        }
    }

    /// `digits × 10^exponent` with trailing zeros folded in, as a dose.
    fn from_parts(digits: i64, exponent: i32) -> Result<Self, DecodeError> {
        if exponent >= 0 {
            let digits = 10i64
                .checked_pow(exponent as u32)
                .and_then(|power| digits.checked_mul(power))
                .ok_or("Dosage is too large")?;
            return Ok(Self { digits, places: 0 });
        }
        let places = u8::try_from(-exponent)
            .ok()
            .filter(|&places| places < u8::MAX)
            .ok_or("Dosage has too many decimal places")?;
        Ok(Self { digits, places })
    }

    /// A finite double by its shortest round-trip decimal.
    pub(super) fn from_f64(value: f64) -> Result<Self, DecodeError> {
        if !value.is_finite() {
            return Err("Dosage must be finite".into());
        }
        let (digits, exponent) = shortest_decimal(value);
        Self::from_parts(digits, exponent)
    }

    /// A BCF float by the shortest decimal that reads back as the same f32. `value` is that f32
    /// widened to f64, which is exact.
    pub(super) fn from_f32(value: f64) -> Result<Self, DecodeError> {
        if !value.is_finite() {
            return Err("Dosage must be finite".into());
        }
        let (digits, exponent) = shortest_decimal_f32(value as f32);
        Self::from_parts(digits, exponent)
    }

    /// The dose at `places` decimal places (at least its own), as an integer.
    #[inline(always)]
    pub(super) fn at(self, places: u8) -> Option<i128> {
        10i128
            .checked_pow(u32::from(places.checked_sub(self.places)?))
            .and_then(|power| i128::from(self.digits).checked_mul(power))
    }

    fn is_negative(self) -> bool {
        self.digits < 0
    }
}

/// A plain decimal of at most fifteen digits, without sign, exponent or space, as written.
#[inline]
pub(super) fn plain_decimal(bytes: &[u8]) -> Option<Dose> {
    let mut mantissa = 0i64;
    let mut digits = 0usize;
    let mut fraction_digits = None;
    for &byte in bytes {
        if byte.is_ascii_digit() {
            digits += 1;
            if digits > 15 {
                return None;
            }
            mantissa = mantissa * 10 + i64::from(byte - b'0');
            if let Some(count) = &mut fraction_digits {
                *count += 1;
            }
        } else if byte == b'.' && fraction_digits.is_none() {
            fraction_digits = Some(0u8);
        } else {
            return None;
        }
    }
    (digits > 0).then_some(Dose {
        digits: mantissa,
        places: fraction_digits.unwrap_or(0),
    })
}

/// A DS or GP field value: `None` for an empty or `.` field, an error for text that is not a
/// finite number.
pub(super) fn parse_dose(text: &str) -> Result<Option<Dose>, DecodeError> {
    if let Some(dose) = plain_decimal(text.as_bytes()) {
        return Ok(Some(dose));
    }
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "." {
        return Ok(None);
    }
    if let Some(dose) = plain_decimal(trimmed.as_bytes()) {
        return Ok(Some(dose));
    }
    Dose::from_f64(trimmed.parse::<f64>()?).map(Some)
}

/// 10^0 through 10^18.
const POW10: [i64; 19] = {
    let mut powers = [1i64; 19];
    let mut index = 1;
    while index < powers.len() {
        powers[index] = powers[index - 1] * 10;
        index += 1;
    }
    powers
};

/// The REF dosage a diploid call with ALT dosage `alt` leaves, when `alt` is at most 2 + 1e-6:
/// `max(2 - alt, 0)`. The same tolerance as [`dosage_from_values`], for a plain decimal of at
/// most fifteen digits.
#[inline(always)]
pub(super) fn diploid_reference(alt: Dose) -> Option<Dose> {
    let places = usize::from(alt.places);
    if places > 15 {
        return None;
    }
    // Both fit i64: 2 × 10^15 and a fifteen-digit ALT.
    let remainder = 2 * POW10[places] - alt.digits;
    let tolerance = if places < 6 { 0 } else { POW10[places - 6] };
    (remainder >= -tolerance).then_some(Dose {
        digits: remainder.max(0),
        places: alt.places,
    })
}

/// `remainder × 10^-places` clamped at zero, at the same places, when it is at least -1e-6.
#[inline(always)]
fn reference_within_ploidy(remainder: i128, places: u8) -> Option<Dose> {
    // remainder × 10^6 ≥ -10^places: below six places that is remainder ≥ 0, and from six on it
    // is remainder ≥ -10^(places - 6).
    let tolerance = match places.checked_sub(6) {
        None => 0,
        Some(extra) => match POW10.get(usize::from(extra)) {
            Some(&power) => i128::from(power),
            None => 10i128.checked_pow(u32::from(extra))?,
        },
    };
    if remainder < -tolerance {
        return None;
    }
    Some(Dose {
        digits: i64::try_from(remainder.max(0)).ok()?,
        places,
    })
}

/// The DS rules for one sample's values, in field order, each already read as an exact dose or
/// `None` for a missing value.
pub(super) fn dosage_from_values<I>(
    values: I,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, DecodeError>
where
    I: Iterator<Item = Result<Option<Dose>, DecodeError>>,
{
    if alt_index == 0 || alt_index > alt_count {
        return Err("ALT allele index is out of range".into());
    }
    let mut alt_dosage = None;
    let mut all = Some(Vec::with_capacity(alt_count));
    let mut count = 0;
    for (offset, value) in values.enumerate() {
        let dosage = value?;
        if dosage.is_some_and(Dose::is_negative) {
            return Err("DS dosage must be nonnegative".into());
        }
        if offset + 1 == alt_index {
            alt_dosage = dosage;
        }
        all = all.zip(dosage).map(|(mut values, value)| {
            values.push(value);
            values
        });
        count += 1;
    }
    if count != alt_count {
        return Err(format!(
            "DS field has {count} values, expected {alt_count} alternate allele dosages"
        )
        .into());
    }
    let ref_dosage = match (ploidy, all) {
        (Some(ploidy), Some(values)) => {
            let places = values.iter().map(|value| value.places).max().unwrap_or(0);
            let total = values
                .iter()
                .try_fold(0i128, |sum, value| sum.checked_add(value.at(places)?))
                .ok_or("DS dosages are too long to add exactly")?;
            let whole = Dose::copies(ploidy)
                .at(places)
                .ok_or("DS dosages are too long to add exactly")?;
            // Permit decimal rounding at the dosage boundary without a negative count.
            Some(
                reference_within_ploidy(whole - total, places)
                    .ok_or("DS alternate dosages exceed genotype ploidy")?,
            )
        }
        _ => None,
    };
    Ok(alt_dosage.map(|alt_dosage| DecodedAltDosage {
        alt_dosage,
        ref_dosage,
    }))
}

/// The GP rules for one sample's `actual_len` probabilities, in field order, each already read
/// as an exact dose or `None` for a missing value.
pub(super) fn gp_from_values<I>(
    actual_len: usize,
    mut parts: I,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, DecodeError>
where
    I: Iterator<Item = Result<Option<Dose>, DecodeError>>,
{
    if alt_index == 0 || alt_index > alt_count {
        return Err(format!(
            "ALT allele index {alt_index} is out of range for {alt_count} alternate alleles"
        )
        .into());
    }
    let allele_count = alt_count.checked_add(1).ok_or("GP allele count overflow")?;
    let diploid_len = allele_count
        .checked_add(1)
        .and_then(|next| allele_count.checked_mul(next))
        .map(|n| n / 2)
        .ok_or("GP allele count overflow")?;
    let ploidy = match ploidy {
        Some(ploidy) => ploidy,
        None if actual_len == allele_count => 1,
        None if actual_len == diploid_len => 2,
        None => return Err(format!("GP field has {actual_len} values; cannot determine haploid or diploid ploidy for {alt_count} alternate alleles").into()),
    };
    let expected_len = match ploidy {
        1 => allele_count,
        2 => diploid_len,
        _ => {
            return Err(format!(
                "GP dosage decoding requires haploid or diploid genotypes, got ploidy {ploidy}"
            )
            .into());
        }
    };
    if actual_len != expected_len {
        return Err(format!("GP field has {actual_len} values, expected {expected_len} for ploidy {ploidy} and {alt_count} alternate alleles").into());
    }
    let mut terms = Vec::with_capacity(expected_len);
    for second in 0..allele_count {
        let first_count = if ploidy == 1 { 1 } else { second + 1 };
        for first in 0..first_count {
            let Some(probability) = parts.next().expect("GP cardinality was validated")? else {
                return Ok(None);
            };
            let one = Dose::copies(1).at(probability.places);
            if probability.is_negative() || one.is_none_or(|one| i128::from(probability.digits) > one)
            {
                return Err("GP probabilities must be finite and between zero and one".into());
            }
            let copies =
                i128::from(u8::from(ploidy == 2 && first == alt_index) + u8::from(second == alt_index));
            let ref_copies = i128::from(u8::from(ploidy == 2 && first == 0) + u8::from(second == 0));
            terms.push((probability, copies, ref_copies));
        }
    }
    let places = terms.iter().map(|(p, _, _)| p.places).max().unwrap_or(0);
    let expected = |select: fn(&(Dose, i128, i128)) -> i128| -> Result<Dose, DecodeError> {
        let digits = terms
            .iter()
            .try_fold(0i128, |sum, term| sum.checked_add(term.0.at(places)?.checked_mul(select(term))?))
            .and_then(|digits| i64::try_from(digits).ok())
            .ok_or("GP probabilities are too long to add exactly")?;
        Ok(Dose { digits, places })
    };
    Ok(Some(DecodedAltDosage {
        alt_dosage: expected(|term| term.1)?,
        ref_dosage: Some(expected(|term| term.2)?),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(digits: i64, places: u8) -> Dose {
        Dose { digits, places }
    }

    #[test]
    fn doses_are_read_as_written_or_by_their_shortest_form() {
        for (text, want) in [
            ("0.123", Some(d(123, 3))),
            ("1", Some(d(1, 0))),
            ("2.000", Some(d(2000, 3))),
            ("+0.5", Some(d(5, 1))),
            ("1e-3", Some(d(1, 3))),
            (" 0.25 ", Some(d(25, 2))),
            (".", None),
            ("", None),
        ] {
            assert_eq!(parse_dose(text).unwrap(), want, "{text}");
        }
        assert!(parse_dose("nan").is_err());
        assert!(parse_dose("inf").is_err());
        assert!(parse_dose("x1").is_err());
        assert_eq!(Dose::from_f32(f64::from(0.123f32)).unwrap(), d(123, 3));
        assert_eq!(Dose::from_f32(2.0).unwrap(), d(2, 0));
    }

    #[test]
    fn ds_keeps_the_tolerance_and_clamp_on_exact_values() {
        let values = |texts: &[&str]| -> Vec<Result<Option<Dose>, DecodeError>> {
            texts.iter().map(|t| parse_dose(t)).collect()
        };
        let got = dosage_from_values(values(&["0.3"]).into_iter(), 1, 1, Some(2)).unwrap().unwrap();
        assert_eq!((got.alt_dosage, got.ref_dosage), (d(3, 1), Some(d(17, 1))));
        // 2 - 2.000001 = -1e-6 is accepted and clamps to zero; -1.1e-6 is refused.
        let got = dosage_from_values(values(&["2.000001"]).into_iter(), 1, 1, Some(2)).unwrap().unwrap();
        assert_eq!(got.ref_dosage, Some(d(0, 6)));
        assert!(dosage_from_values(values(&["2.0000011"]).into_iter(), 1, 1, Some(2)).is_err());
        // The boundary is the decimal 1e-6: 1 - 1.000001 is on it, though as doubles it is
        // -1.00000000000008e-6.
        let got = dosage_from_values(values(&["1.000001"]).into_iter(), 1, 1, Some(1)).unwrap().unwrap();
        assert_eq!(got.ref_dosage, Some(d(0, 6)));
        let got = dosage_from_values(values(&["0.7", "0.7", "0.600001"]).into_iter(), 1, 3, Some(2)).unwrap().unwrap();
        assert_eq!(got.ref_dosage, Some(d(0, 6)));
        assert!(dosage_from_values(values(&["-0.1"]).into_iter(), 1, 1, Some(2)).is_err());
        let got = dosage_from_values(values(&["0.25", "0.5"]).into_iter(), 2, 2, Some(2)).unwrap().unwrap();
        assert_eq!((got.alt_dosage, got.ref_dosage), (d(5, 1), Some(d(125, 2))));
        assert!(dosage_from_values(values(&["."]).into_iter(), 1, 1, Some(2)).unwrap().is_none());
        let got = dosage_from_values(values(&["0.3"]).into_iter(), 1, 1, None).unwrap().unwrap();
        assert_eq!(got.ref_dosage, None);
        assert_eq!(diploid_reference(d(2000001, 6)), Some(d(0, 6)));
        assert_eq!(diploid_reference(d(20000011, 7)), None);
        assert_eq!(diploid_reference(d(5, 1)), Some(d(15, 1)));
    }

    #[test]
    fn gp_expected_dosages_are_exact() {
        let parts = ["0.1", "0.2", "0.7"].map(parse_dose);
        let got = gp_from_values(3, parts.into_iter(), 1, 1, Some(2)).unwrap().unwrap();
        // ALT: 0.2 + 2 x 0.7 = 1.6; REF: 2 x 0.1 + 0.2 = 0.4.
        assert_eq!((got.alt_dosage, got.ref_dosage), (d(16, 1), Some(d(4, 1))));
        let parts = ["0.1", "1.2", "0.0"].map(parse_dose);
        assert!(gp_from_values(3, parts.into_iter(), 1, 1, Some(2)).is_err());
        let parts = ["0.1", ".", "0.9"].map(parse_dose);
        assert!(gp_from_values(3, parts.into_iter(), 1, 1, Some(2)).unwrap().is_none());
    }
}
