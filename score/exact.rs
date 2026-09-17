//! Exact score arithmetic.
//!
//! Every term a score receives is a weight times a small integer dosage (or, for the
//! complex-variant averaging fallback, a small rational). A weight is held at its shortest
//! round-trip decimal form, so per score one power of ten, times the least common multiple
//! of its averaging denominators, turns every term into an integer. Sums of those integers
//! cannot depend on the order, grouping, partition or thread count of the accumulation.
//! Floating point appears once, when a finished sum is rounded to f64 for output.

/// One score's fixed point: a cell holding `v` means `v * 2^exp / scale`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedPoint {
    pub(crate) exp: i32,
    pub(crate) scale: u128,
}

impl FixedPoint {
    /// `v * 2^exp / (scale * divisor)`, correctly rounded; 0 when `divisor` is 0. A planned
    /// scale leaves room for any u32 divisor, so the product fits u128.
    #[inline]
    pub(crate) fn quotient(&self, v: i128, divisor: u32) -> f64 {
        if divisor == 0 {
            return 0.0;
        }
        let q = self.scale * u128::from(divisor);
        if self.exp <= 0 && self.exp > -53 && v.unsigned_abs() < 1u128 << 53 {
            let shift = self.exp.unsigned_abs();
            if q < 1u128 << (53 - shift) {
                // Both operands are exact doubles, so the one division rounds correctly.
                return v as f64 / (q << shift) as f64;
            }
        }
        round_quotient(v, self.exp, q)
    }
}

/// `value = digits * 10^exponent` from the shortest round-trip form of a finite `value`,
/// with trailing zeros folded into the exponent. Zero is `(0, 0)`.
pub(crate) fn shortest_decimal(value: f64) -> (i64, i32) {
    if value == 0.0 {
        return (0, 0);
    }
    let mut buffer = ryu::Buffer::new();
    let text = buffer.format_finite(value.abs());
    let (mantissa, mut exponent) = match text.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i32>().unwrap_or(0)),
        None => (text, 0),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    exponent -= fraction.len() as i32;
    // A shortest form has at most 17 significant digits, plus leading zeros, so i64 holds them.
    let mut digits = integer
        .bytes()
        .chain(fraction.bytes())
        .fold(0i64, |digits, byte| digits * 10 + i64::from(byte - b'0'));
    while digits % 10 == 0 {
        digits /= 10;
        exponent += 1;
    }
    (if value < 0.0 { -digits } else { digits }, exponent)
}

/// Two carry-free i64 limbs for one score: `v = hi * 2^bits + lo`, with `lo` in `[0, 2^bits)`.
/// A cell receives at most one term per entry, so for `terms` entries whose integers need
/// `term_bits` bits (sign included) neither limb's running sum can overflow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Split {
    pub(crate) bits: u32,
}

impl Split {
    /// The middle of the feasible window, or `None` when two i64 limbs cannot hold the sums.
    pub(crate) fn plan(term_bits: u32, terms: u64) -> Option<Self> {
        if term_bits > 128 {
            return None;
        }
        let count_bits = 64 - terms.checked_add(1)?.leading_zeros() as i32;
        let low = (term_bits as i32 + count_bits - 62).max(1);
        let high = 62 - count_bits;
        (low <= high).then(|| Self {
            bits: ((low + high) / 2) as u32,
        })
    }

    #[inline(always)]
    pub(crate) fn parts(self, v: i128) -> (i64, i64) {
        (
            (v & ((1i128 << self.bits) - 1)) as i64,
            (v >> self.bits) as i64,
        )
    }

    #[inline(always)]
    pub(crate) fn join(self, lo: i64, hi: i64) -> i128 {
        (i128::from(hi) << self.bits) + i128::from(lo)
    }
}

/// `v * 2^exp / q` rounded to the nearest f64, ties to even, for any nonzero q.
fn round_quotient(v: i128, exp: i32, q: u128) -> f64 {
    assert!(q >= 1);
    if v == 0 {
        return 0.0;
    }
    let negative = v < 0;
    // Normalise the dividend to 128 bits. A 64-bit denominator leaves at least 64
    // significant quotient bits; a wider denominator needs more division bits.
    let shift = v.unsigned_abs().leading_zeros();
    let a = v.unsigned_abs() << shift;
    let mut e = exp - shift as i32;
    let (mut quotient, mut remainder) = if q <= u128::from(u64::MAX) {
        let q64 = q as u64;
        let (hi, lo) = ((a >> 64) as u64, a as u64);
        let rest = (u128::from(hi % q64) << 64) | u128::from(lo);
        ((u128::from(hi / q64) << 64) | (rest / q), rest % q)
    } else {
        (a / q, a % q)
    };
    while quotient < 1u128 << 63 {
        // Subtract before doubling when the next bit is 1: even q near u128::MAX
        // cannot overflow. The quotient never exceeds 64 bits in this loop.
        let next = remainder >= q - remainder;
        remainder = if next {
            remainder - (q - remainder)
        } else {
            remainder * 2
        };
        quotient = (quotient << 1) | u128::from(next);
        e -= 1;
    }
    let mut sticky = remainder != 0;
    // Binary exponent of the last kept bit: 52 below the leading bit, but not below the
    // subnormal floor. Everything under it is folded into guard and sticky bits.
    let top = e + 127 - quotient.leading_zeros() as i32;
    let last = (top - 52).max(-1074);
    let drop = (last - e) as u32;
    let mantissa = if drop > 128 {
        // Below half the smallest subnormal.
        0
    } else if drop == 128 {
        // Only a guard bit survives: round up past the midpoint, and on a tie to even (zero).
        let guard = quotient >> 127 != 0;
        sticky |= quotient & ((1u128 << 127) - 1) != 0;
        u128::from(guard && sticky)
    } else {
        let below = quotient & ((1u128 << drop) - 1);
        let kept = quotient >> drop;
        let half = 1u128 << (drop - 1);
        sticky |= below & (half - 1) != 0;
        let round_up = below & half != 0 && (sticky || kept & 1 == 1);
        quotient = kept + u128::from(round_up);
        quotient
    };
    e = last;
    let magnitude = scale_by_power_of_two(mantissa as u64, e);
    if negative { -magnitude } else { magnitude }
}

/// `m * 2^e` for `m <= 2^53`, where `e >= -1074` and the value needs no rounding.
fn scale_by_power_of_two(mut m: u64, mut e: i32) -> f64 {
    if m == 0 {
        return 0.0;
    }
    if m >> 53 != 0 {
        // Rounding carried into a 54th bit; the low bit is zero.
        m >>= 1;
        e += 1;
    }
    let bits = 64 - m.leading_zeros() as i32;
    let top = e + bits - 1;
    if top > 1023 {
        return f64::INFINITY;
    }
    if top >= -1022 {
        let fraction = (m << (53 - bits)) & ((1u64 << 52) - 1);
        f64::from_bits(((top + 1023) as u64) << 52 | fraction)
    } else {
        // Subnormal: e == -1074, so m is the raw fraction field.
        f64::from_bits(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Sign of `x * 2^shift * q - v` for a nonnegative dyadic `x = m * 2^e`, exactly,
    /// within the ranges these tests use (everything fits in i128 after alignment).
    fn exact_cmp(m: u64, e: i32, v: i128, exp: i32, q: u128) -> std::cmp::Ordering {
        let d = e - exp;
        let left = i128::from(m) * q as i128;
        if d >= 0 {
            (left << d).cmp(&v)
        } else {
            left.cmp(&(v << -d))
        }
    }

    #[test]
    fn rounding_is_correct_against_exact_midpoints() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for case in 0..200_000 {
            let bits = 1 + rng.next() % 100;
            let v = (rng.next() as i128 | (i128::from(rng.next()) << 64)) & ((1i128 << bits) - 1);
            let v = if v == 0 { 1 } else { v };
            let q = [1u128, 2, 3, 6, 7, 10, 60, 3200, 3201, 12480, 65535][case % 11];
            let exp = -80 + (rng.next() % 60) as i32;
            let r = round_quotient(v, exp, q);
            assert!(r > 0.0 && r.is_finite());
            // Full 53-bit mantissa and ulp exponent of a normal r.
            let bits = r.to_bits();
            let (m, e) = (
                (bits & ((1u64 << 52) - 1)) | (1u64 << 52),
                ((bits >> 52) as i32) - 1075,
            );
            // The exact value lies between the midpoints to both neighbours. Below a power
            // of two the neighbour is half an ulp away, so that midpoint is a quarter ulp.
            let (lo_m, lo_e) = if m == 1 << 52 {
                (4 * m - 1, e - 2)
            } else {
                (2 * m - 1, e - 1)
            };
            use std::cmp::Ordering::*;
            let below = exact_cmp(lo_m, lo_e, v, exp, q);
            let above = exact_cmp(2 * m + 1, e - 1, v, exp, q);
            assert!(
                below != Greater && above != Less,
                "v={v} exp={exp} q={q} r={r:e}"
            );
            if below == Equal || above == Equal {
                assert_eq!(m & 1, 0, "tie not to even: v={v} exp={exp} q={q} r={r:e}");
            }
        }
    }

    #[test]
    fn the_one_division_path_agrees_with_long_division() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for case in 0..200_000 {
            let magnitude = (rng.next() >> (11 + case % 40)) as i128;
            let v = if rng.next() & 1 == 1 { -magnitude } else { magnitude };
            let places = (rng.next() % 12) as u32;
            let multiple = [1u128, 2, 6, 12, 60][case % 5];
            let fixed = FixedPoint {
                exp: -(places as i32),
                scale: 5u128.pow(places) * multiple,
            };
            let divisor = [1u32, 2, 3, 7, 20_000, 1_000_003][case % 6];
            assert_eq!(
                fixed.quotient(v, divisor).to_bits(),
                round_quotient(v, fixed.exp, fixed.scale * u128::from(divisor)).to_bits(),
                "v={v} places={places} multiple={multiple} divisor={divisor}"
            );
        }
    }

    #[test]
    fn shortest_forms_recover_written_decimals() {
        for (value, want) in [
            (0.123456, (123456, -6)),
            (-0.5, (-5, -1)),
            (1.0, (1, 0)),
            (1500.0, (15, 2)),
            (1e-7, (1, -7)),
            (-3.25e-12, (-325, -14)),
            (0.1 + 0.2, (30000000000000004, -17)),
            (5e-324, (5, -324)),
            (0.0, (0, 0)),
        ] {
            assert_eq!(shortest_decimal(value), want, "{value:e}");
        }
        // Every decimal written with up to 15 significant digits comes back as written; with more
        // digits the double may not hold the written value.
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..100_000 {
            let digits = (rng.next() % 999_999_999_999_999) as i64 + 1;
            let exponent = (rng.next() % 40) as i32 - 30;
            let text = format!("{digits}e{exponent}");
            let (got_digits, got_exponent) = shortest_decimal(text.parse().unwrap());
            let mut want = (digits, exponent);
            while want.0 % 10 == 0 {
                want = (want.0 / 10, want.1 + 1);
            }
            assert_eq!((got_digits, got_exponent), want, "{text}");
        }
    }

    #[test]
    fn scaled_averages_keep_the_entire_denominator() {
        let plan = FixedPoint {
            exp: -70,
            scale: 1 << 63,
        };
        // The denominator is exactly 2^64, which previously truncated to zero.
        assert_eq!(plan.quotient(1 << 100, 2), 2f64.powi(-34));
        let plan = FixedPoint {
            exp: 0,
            scale: u128::from(u64::MAX),
        };
        // Seven whole denominators come back exactly, and (2^127 - 1) / ((2^64 - 1)(2^32 - 1)) =
        // 2^31 + 1/2 + about 2^-32, which rounds to 2^31 + 1/2.
        let whole = i128::try_from(u128::from(u64::MAX) * u128::from(u32::MAX) * 7).unwrap();
        assert_eq!(plan.quotient(whole, u32::MAX), 7.0);
        assert_eq!(plan.quotient(i128::MAX, u32::MAX), 2147483648.5);
        assert_eq!(plan.quotient(0, u32::MAX), 0.0);
        assert_eq!(plan.quotient(12345, 0), 0.0);
    }

    #[test]
    fn impossible_splits_are_rejected_without_wrapping() {
        assert!(Split::plan(1, u64::MAX).is_none());
        assert!(Split::plan(u32::MAX, 1).is_none());
    }

    #[test]
    fn limb_splits_join_exactly_and_refuse_ranges_they_cannot_hold() {
        // PGS004525: 79-bit terms over 1.07 M variants fit; a 100-bit range over 2^21 does not.
        let split = Split::plan(79, 1_068_120).expect("fits two limbs");
        assert!((38..=41).contains(&split.bits));
        assert!(Split::plan(100, 1 << 21).is_none());
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let (mut lo, mut hi, mut exact) = (0i64, 0i64, 0i128);
        for _ in 0..1_068_120 {
            let magnitude =
                (i128::from(rng.next()) << 16 | i128::from(rng.next() >> 48)) & ((1i128 << 78) - 1);
            let term = if rng.next() & 1 == 1 {
                -magnitude
            } else {
                magnitude
            };
            let (l, h) = split.parts(term);
            assert!(l >= 0 && l < 1 << split.bits);
            assert_eq!(split.join(l, h), term);
            lo = lo.checked_add(l).expect("lo limb overflow");
            hi = hi.checked_add(h).expect("hi limb overflow");
            exact += term;
        }
        assert_eq!(split.join(lo, hi), exact);
    }

    #[test]
    fn negative_values_mirror_positive_ones() {
        for (v, exp, q) in [
            (12345i128, -20, 7u128),
            (1 << 90, -120, 3201),
            (3, -1076, 1),
        ] {
            assert_eq!(round_quotient(-v, exp, q), -round_quotient(v, exp, q));
        }
        // Subnormal results keep the correctly rounded fraction field.
        assert_eq!(round_quotient(3, -1076, 1), f64::from_bits(1));
    }
}
