//! Exact score arithmetic.
//!
//! Every term a score receives is a weight times a small integer dosage (or, for
//! the complex-variant averaging fallback, a small rational). A weight is a finite
//! f64, so it is an integer multiple of a power of two. Per score, one binary
//! exponent turns every term into an integer; sums of those integers are exact in
//! i128, so the order, grouping, partition and thread count of the accumulation
//! cannot change a single bit of the result. Floating point appears once, when a
//! finished sum is rounded to f64 for output.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed by the exact score accumulators once they land"
    )
)]

/// One score's fixed point: a cell holding `v` means `v * 2^exp / scale`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedPoint {
    pub(crate) exp: i32,
    /// Least common multiple of the averaging denominators; 1 without complex averaging.
    pub(crate) scale: u64,
}

/// A finite nonzero f64 as (odd signed mantissa, exponent).
fn f64_parts(value: f64) -> Option<(i64, i32)> {
    let bits = value.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = (bits & ((1u64 << 52) - 1)) as i64;
    let (mantissa, exp) = if field == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1i64 << 52), field - 1075)
    };
    if mantissa == 0 {
        return None;
    }
    let tz = mantissa.trailing_zeros() as i32;
    let odd = mantissa >> tz;
    Some((if bits >> 63 == 1 { -odd } else { odd }, exp + tz))
}

impl FixedPoint {
    /// The exponent that makes every coefficient an integer, if a cell can hold
    /// `terms` of them at up to `max_multiplier` each without overflow. `None` for
    /// non-finite coefficients or a range wider than i128.
    pub(crate) fn plan(
        coefficients: impl IntoIterator<Item = f64>,
        terms: u64,
        max_multiplier: u64,
        scale: u64,
    ) -> Option<Self> {
        if scale == 0 {
            return None;
        }
        let (mut low, mut high) = (i32::MAX, i32::MIN);
        for value in coefficients {
            if !value.is_finite() {
                return None;
            }
            if let Some((mantissa, exp)) = f64_parts(value) {
                low = low.min(exp);
                high = high.max(exp + (64 - mantissa.unsigned_abs().leading_zeros()) as i32);
            }
        }
        if low == i32::MAX {
            return Some(Self { exp: 0, scale });
        }
        let bound = u128::from(terms.max(1))
            .checked_mul(u128::from(max_multiplier.max(1)))?
            .checked_mul(u128::from(scale))?;
        let headroom = 128 - bound.leading_zeros() as i32;
        (high - low + headroom < 127).then_some(Self { exp: low, scale })
    }

    /// `value * 2^-exp * scale` exactly. The value must have been seen by `plan`.
    #[inline]
    pub(crate) fn to_fixed(&self, value: f64) -> i128 {
        f64_parts(value).map_or(0, |(mantissa, exp)| {
            (i128::from(mantissa) << (exp - self.exp)) * i128::from(self.scale)
        })
    }

    /// `v * 2^exp / scale`, correctly rounded (ties to even).
    pub(crate) fn to_f64(&self, v: i128) -> f64 {
        round_quotient(v, self.exp, u128::from(self.scale))
    }

    /// `v * 2^exp / (scale * divisor)`, correctly rounded; 0 when `divisor` is 0.
    pub(crate) fn quotient(&self, v: i128, divisor: u64) -> f64 {
        if divisor == 0 {
            return 0.0;
        }
        round_quotient(v, self.exp, u128::from(self.scale) * u128::from(divisor))
    }
}

/// Two carry-free i64 limbs for one score: `v = hi * 2^bits + lo`, with `lo` in `[0, 2^bits)`.
/// A cell receives at most one term per variant, so for `terms` variants whose integers need
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
    // significant quotient bits; a wider scale * divisor needs more division bits.
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
    fn planned_coefficients_round_trip_and_sums_ignore_order() {
        let weights: Vec<f64> = (0..4096)
            .map(|i| ((i * 7919 % 10007) as f64 - 5003.0) * 1.37e-7)
            .chain([0.123456789, -3.5e-5, 2.0, 7.0])
            .collect();
        let plan = FixedPoint::plan(weights.iter().copied(), 4100, 2, 1).expect("fits");
        for &w in &weights {
            assert_eq!(plan.to_f64(plan.to_fixed(w)), w);
        }
        let forward: i128 = weights.iter().map(|&w| plan.to_fixed(w)).sum();
        let backward: i128 = weights.iter().rev().map(|&w| plan.to_fixed(w)).sum();
        assert_eq!(forward, backward);
        assert!(FixedPoint::plan([1e300, 1e-300], 4, 2, 1).is_none());
        assert!(FixedPoint::plan([f64::NAN], 4, 2, 1).is_none());
        assert_eq!(plan.quotient(forward, 0), 0.0);
    }

    #[test]
    fn impossible_bounds_are_rejected_without_wrapping() {
        assert!(FixedPoint::plan([1.0], u64::MAX, u64::MAX, u64::MAX).is_none());
        assert!(FixedPoint::plan([1.0], 1, 1, 0).is_none());
        assert!(Split::plan(1, u64::MAX).is_none());
        assert!(Split::plan(u32::MAX, 1).is_none());
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
            scale: u64::MAX,
        };
        assert_eq!(plan.quotient(i128::MAX, u64::MAX), 0.5);
        assert_eq!(plan.quotient(i128::MIN, u64::MAX), -0.5);
        assert_eq!(plan.quotient(0, u64::MAX), 0.0);
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
