//! The CRC-32 of inflated BGZF data.
//!
//! Which routine runs, and the CPU features it needs:
//! - x86-64 with VPCLMULQDQ, PCLMULQDQ and AVX2, detected at run time, for inputs of
//!   at least [`x86::MIN_LEN`] bytes: four 256-bit vectors folded here at a time, so
//!   the speed does not depend on the C compiler that built libdeflate (gcc before
//!   10.1 leaves libdeflate's own VPCLMULQDQ routine out and folds 128 bits at a time).
//! - Everything else, including a CPU missing any of those features: libdeflate's
//!   routine, which selects its own among PCLMULQDQ folds, a VPCLMULQDQ fold where
//!   its compiler built one, and a table-driven loop.

/// The CRC-32 of `data`, as gzip and BGZF record it.
pub(super) fn crc32(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    if data.len() >= x86::MIN_LEN && x86::available() {
        // SAFETY: `x86::available()` has just confirmed VPCLMULQDQ, PCLMULQDQ and AVX2
        // at run time, every feature `x86::crc32` is compiled with.
        return unsafe { x86::crc32(data) };
    }
    libdeflater::crc32(data)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::{
        __m128i, __m256i, _mm_clmulepi64_si128, _mm_loadu_si128, _mm_set_epi64x, _mm_storeu_si128,
        _mm_xor_si128, _mm256_castsi256_si128, _mm256_clmulepi64_epi128, _mm256_extracti128_si256,
        _mm256_loadu_si256, _mm256_set_epi64x, _mm256_xor_si256,
    };

    /// The shortest input folded here: the four vectors the fold starts from.
    pub(super) const MIN_LEN: usize = 4 * 32;

    /// The gzip polynomial without its x^32 term, bit i the coefficient of x^i.
    const POLYNOMIAL: u32 = 0x04c1_1db7;

    /// Multipliers that move a lane 1024 bits forward: four 256-bit vectors.
    const FOLD_1024: (u64, u64) = fold_multipliers(1024);
    /// Multipliers that move a lane 128 bits forward: one lane.
    const FOLD_128: (u64, u64) = fold_multipliers(128);

    /// The register update table of the reflected polynomial.
    const TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut index = 0usize;
        while index < 256 {
            let mut value = index as u32;
            let mut bit = 0;
            while bit < 8 {
                value = if value & 1 == 1 {
                    (value >> 1) ^ POLYNOMIAL.reverse_bits()
                } else {
                    value >> 1
                };
                bit += 1;
            }
            table[index] = value;
            index += 1;
        }
        table
    };

    /// Whether this CPU has VPCLMULQDQ, PCLMULQDQ and AVX2, the features [`crc32`]
    /// is compiled with.
    pub(super) fn available() -> bool {
        is_x86_feature_detected!("vpclmulqdq")
            && is_x86_feature_detected!("pclmulqdq")
            && is_x86_feature_detected!("avx2")
    }

    /// x^n mod the gzip polynomial, bit i the coefficient of x^i.
    const fn x_pow_mod(n: u32) -> u32 {
        let mut value = 1u32;
        let mut step = 0;
        while step < n {
            let carry = value & 0x8000_0000 != 0;
            value <<= 1;
            if carry {
                value ^= POLYNOMIAL;
            }
            step += 1;
        }
        value
    }

    /// The multipliers that move a 128-bit lane of reflected message bits
    /// `distance` bits forward, for its low and its high 64 bits.
    ///
    /// In a lane, bit i is the coefficient of x^(127 - i), so its low half stands
    /// for x^64 times a 64-bit polynomial and its high half for x^0 times one.
    /// Moving it forward multiplies the halves by x^(distance + 64) and x^distance.
    /// Each multiplier is (x^(distance ± 32) mod G) · x^32, congruent to those
    /// powers, written with bit j the coefficient of x^(64 - j): a carry-less product
    /// of a half and its multiplier then has bit t the coefficient of x^(127 - t),
    /// which is the lane `distance` bits later.
    pub(super) const fn fold_multipliers(distance: u32) -> (u64, u64) {
        (
            (x_pow_mod(distance + 32).reverse_bits() as u64) << 1,
            (x_pow_mod(distance - 32).reverse_bits() as u64) << 1,
        )
    }

    /// Runs the CRC register over `bytes`, without the inversions at either end.
    pub(super) fn update(mut register: u32, bytes: &[u8]) -> u32 {
        for &byte in bytes {
            register = TABLE[usize::from(register as u8 ^ byte)] ^ (register >> 8);
        }
        register
    }

    /// The CRC-32 of at least [`MIN_LEN`] bytes.
    ///
    /// Four 256-bit accumulators start as the first four vectors, with the
    /// register's all-ones initial state in the first 32 bits, and each folds
    /// 1024 bits forward onto the vector that far ahead. Their eight lanes then fold
    /// into one, 128 bits at a time, which folds in every 16 bytes after them; the
    /// register runs over that lane and the bytes left.
    #[target_feature(enable = "avx2,pclmulqdq,vpclmulqdq")]
    pub(super) fn crc32(data: &[u8]) -> u32 {
        let (vectors, _) = data.as_chunks::<32>();
        let (first, later) = vectors
            .split_first_chunk::<4>()
            .expect("the input holds four vectors");
        let mut accumulators = [
            load256(&first[0]),
            load256(&first[1]),
            load256(&first[2]),
            load256(&first[3]),
        ];
        accumulators[0] =
            _mm256_xor_si256(accumulators[0], _mm256_set_epi64x(0, 0, 0, 0xffff_ffff));
        let fold_1024 = _mm256_set_epi64x(
            FOLD_1024.1 as i64,
            FOLD_1024.0 as i64,
            FOLD_1024.1 as i64,
            FOLD_1024.0 as i64,
        );
        let (groups, _) = later.as_chunks::<4>();
        for group in groups {
            for (accumulator, vector) in accumulators.iter_mut().zip(group) {
                *accumulator = fold256(*accumulator, load256(vector), fold_1024);
            }
        }

        let fold_128 = _mm_set_epi64x(FOLD_128.1 as i64, FOLD_128.0 as i64);
        let mut lane = _mm256_castsi256_si128(accumulators[0]);
        lane = fold128(lane, _mm256_extracti128_si256(accumulators[0], 1), fold_128);
        for &accumulator in &accumulators[1..] {
            lane = fold128(lane, _mm256_castsi256_si128(accumulator), fold_128);
            lane = fold128(lane, _mm256_extracti128_si256(accumulator, 1), fold_128);
        }
        let folded = 32 * 4 * (1 + groups.len());
        let (lanes, bytes) = data[folded..].as_chunks::<16>();
        for next in lanes {
            lane = fold128(lane, load128(next), fold_128);
        }
        let mut last = [0u8; 16];
        // SAFETY: the pointer covers the 16 bytes of `last`, and SSE2, which the
        // store needs, is part of the x86-64 baseline.
        unsafe { _mm_storeu_si128(last.as_mut_ptr().cast(), lane) };
        !update(update(0, &last), bytes)
    }

    /// `next` XOR both lanes of `accumulator` moved forward by `multipliers`.
    #[inline]
    #[target_feature(enable = "avx2,vpclmulqdq")]
    fn fold256(accumulator: __m256i, next: __m256i, multipliers: __m256i) -> __m256i {
        let low = _mm256_clmulepi64_epi128(accumulator, multipliers, 0x00);
        let high = _mm256_clmulepi64_epi128(accumulator, multipliers, 0x11);
        _mm256_xor_si256(next, _mm256_xor_si256(low, high))
    }

    /// `next` XOR `lane` moved forward by `multipliers`.
    #[inline]
    #[target_feature(enable = "avx2,pclmulqdq")]
    fn fold128(lane: __m128i, next: __m128i, multipliers: __m128i) -> __m128i {
        let low = _mm_clmulepi64_si128(lane, multipliers, 0x00);
        let high = _mm_clmulepi64_si128(lane, multipliers, 0x11);
        _mm_xor_si128(next, _mm_xor_si128(low, high))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load256(vector: &[u8; 32]) -> __m256i {
        // SAFETY: the pointer covers the 32 bytes of `vector`; the unaligned load
        // needs AVX, which this function is compiled with.
        unsafe { _mm256_loadu_si256(vector.as_ptr().cast()) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn load128(lane: &[u8; 16]) -> __m128i {
        // SAFETY: the pointer covers the 16 bytes of `lane`; the unaligned load
        // needs SSE2, part of the x86-64 baseline.
        unsafe { _mm_loadu_si128(lane.as_ptr().cast()) }
    }
}

#[cfg(test)]
mod tests {
    use super::crc32;

    /// SplitMix64 draws.
    fn draw(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// crc32fast's CRC-32, which every path is compared with.
    fn reference(data: &[u8]) -> u32 {
        let mut crc = flate2::Crc::new();
        crc.update(data);
        crc.sum()
    }

    /// Checks every path on `data`: the dispatch, libdeflate's routine (which a CPU
    /// without VPCLMULQDQ, PCLMULQDQ and AVX2 takes), the table-driven register run
    /// over the whole input, and the folding routine on a CPU that has its features.
    fn check(data: &[u8]) {
        let expected = reference(data);
        assert_eq!(crc32(data), expected, "dispatch, {} bytes", data.len());
        assert_eq!(libdeflater::crc32(data), expected, "libdeflate, {} bytes", data.len());
        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(!super::x86::update(!0, data), expected, "table, {} bytes", data.len());
            if data.len() >= super::x86::MIN_LEN && super::x86::available() {
                // SAFETY: `available()` has just confirmed VPCLMULQDQ, PCLMULQDQ and
                // AVX2 at run time, every feature `x86::crc32` is compiled with.
                let folded = unsafe { super::x86::crc32(data) };
                assert_eq!(folded, expected, "fold, {} bytes", data.len());
            }
        }
    }

    /// Every path gives crc32fast's CRC-32 over lengths 0 through 4096 at every
    /// alignment offset 0 through 63, over lengths past 4 MiB, and over random
    /// buffers of random lengths at random offsets.
    #[test]
    fn every_crc32_path_matches_crc32fast() {
        let mut state = 0x2362;
        let bytes: Vec<u8> = (0..(4 << 20) + 4096 + 64)
            .map(|_| draw(&mut state) as u8)
            .collect();
        for offset in 0..64 {
            for len in 0..=4096 {
                check(&bytes[offset..offset + len]);
            }
        }
        for (offset, len) in [
            (0, 4 << 20),
            (1, (4 << 20) + 1),
            (17, (4 << 20) + 127),
            (63, (4 << 20) + 4096),
        ] {
            check(&bytes[offset..offset + len]);
        }
        for _ in 0..300 {
            let len = (draw(&mut state) % 70_000) as usize;
            let offset = (draw(&mut state) % 64) as usize;
            let buffer: Vec<u8> = (0..offset + len).map(|_| draw(&mut state) as u8).collect();
            check(&buffer[offset..]);
        }
    }

    /// Flipping any one bit of a 64 KiB block changes its CRC-32, and every path
    /// still gives crc32fast's value for the changed block.
    #[test]
    fn single_bit_flips_change_the_crc32_on_every_path() {
        let mut state = 0x6b17;
        let mut block: Vec<u8> = (0..65_280).map(|_| draw(&mut state) as u8).collect();
        let original = crc32(&block);
        for _ in 0..300 {
            let bit = (draw(&mut state) % (8 * block.len() as u64)) as usize;
            block[bit / 8] ^= 1 << (bit % 8);
            assert_ne!(crc32(&block), original, "bit {bit} flipped");
            check(&block);
            block[bit / 8] ^= 1 << (bit % 8);
        }
    }

    /// The multipliers are the published constants of the Linux kernel's
    /// crc32-pclmul folding by one lane and by four.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fold_multipliers_are_the_published_constants() {
        assert_eq!(
            super::x86::fold_multipliers(128),
            (0x1_7519_97d0, 0x0_ccaa_009e)
        );
        assert_eq!(
            super::x86::fold_multipliers(512),
            (0x1_5444_2bd4, 0x1_c6e4_1596)
        );
    }
}
