//! Parity tests: a `.pgen` read through the virtual PLINK-1.9 façade must
//! produce byte-identical `.bed` content to the real `.bed` it was converted
//! from — including under the sparse, out-of-order access pattern the score
//! pipeline actually uses.
//!
//! The fixture in `data/testdata` is deliberately built with strong LD so that
//! `plink2 --make-pgen` emits mostly LD-compressed (record type 2/3) records.
//! Those records are the ones that decode incorrectly if the reader assumes a
//! strictly sequential pass, so they are exactly what needs covering.
//! See `data/testdata/README.md` for how to regenerate it.

use gnomon::adapt_plink2::{GenomeBuild, open_virtual_plink19_from_paths};
use std::path::{Path, PathBuf};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("data/testdata")
}

const N_SAMPLES: usize = 200;
const N_VARIANTS: usize = 1500;

fn block_bytes() -> usize {
    N_SAMPLES.div_ceil(4)
}

/// Deterministic shuffle so a failure is always reproducible.
fn shuffled_indices(n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state: u64 = 0x243f_6a88_85a3_08d3;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        idx.swap(i, j);
    }
    idx
}

fn open_fixture() -> gnomon::adapt_plink2::VirtualPlink19 {
    let dir = fixture_dir();
    open_virtual_plink19_from_paths(
        &dir.join("ld_p.pgen"),
        &dir.join("ld_p.pvar"),
        &dir.join("ld_p.psam"),
        GenomeBuild::Grch38,
    )
    .expect("opening the pgen fixture")
}

fn real_bed() -> Vec<u8> {
    std::fs::read(fixture_dir().join("ld.bed")).expect("reading the bed fixture")
}

#[test]
fn virtual_bed_matches_real_bed_sequentially() {
    let vp = open_fixture();
    let bed = real_bed();
    assert_eq!(vp.n_samples(), N_SAMPLES);
    assert_eq!(vp.n_variants(), N_VARIANTS);
    assert_eq!(bed.len(), 3 + N_VARIANTS * block_bytes());

    let src = vp.bed_source();
    let bb = block_bytes();
    let mut got = vec![0u8; bb];
    for v in 0..N_VARIANTS {
        let off = 3 + v * bb;
        src.read_at(off as u64, &mut got).expect("virtual read");
        assert_eq!(
            got,
            &bed[off..off + bb],
            "sequential mismatch at variant {v}"
        );
    }
}

/// The score pipeline reads only the variants a score file matched, so the
/// reader must be correct for an arbitrary subset in arbitrary order — not just
/// for a full sequential sweep.
#[test]
fn virtual_bed_matches_real_bed_in_random_order() {
    let vp = open_fixture();
    let bed = real_bed();
    let src = vp.bed_source();
    let bb = block_bytes();
    let mut got = vec![0u8; bb];

    for &v in shuffled_indices(N_VARIANTS).iter() {
        let off = 3 + v * bb;
        src.read_at(off as u64, &mut got).expect("virtual read");
        assert_eq!(
            got,
            &bed[off..off + bb],
            "random-order mismatch at variant {v}"
        );
    }
}

/// A sparse subset, ascending — the exact shape of a real scoring run where a
/// PGS matches a small fraction of the cohort's variants.
#[test]
fn virtual_bed_matches_real_bed_for_sparse_subset() {
    let vp = open_fixture();
    let bed = real_bed();
    let src = vp.bed_source();
    let bb = block_bytes();
    let mut got = vec![0u8; bb];

    for v in (0..N_VARIANTS).step_by(37) {
        let off = 3 + v * bb;
        src.read_at(off as u64, &mut got).expect("virtual read");
        assert_eq!(got, &bed[off..off + bb], "sparse mismatch at variant {v}");
    }
}

/// Re-reading a variant must return the same bytes; a stateful decoder that
/// mutates shared anchor state can pass once and fail on the second visit.
#[test]
fn virtual_bed_reads_are_idempotent() {
    let vp = open_fixture();
    let bed = real_bed();
    let src = vp.bed_source();
    let bb = block_bytes();
    let mut got = vec![0u8; bb];

    for v in [1499usize, 0, 1499, 750, 1499, 0] {
        let off = 3 + v * bb;
        src.read_at(off as u64, &mut got).expect("virtual read");
        assert_eq!(got, &bed[off..off + bb], "idempotency mismatch at {v}");
    }
}

/// Multi-variant reads spanning several blocks must agree with per-block reads.
#[test]
fn virtual_bed_multi_block_reads_match() {
    let vp = open_fixture();
    let bed = real_bed();
    let src = vp.bed_source();
    let bb = block_bytes();

    let start = 500usize;
    let span = 64usize;
    let mut got = vec![0u8; bb * span];
    let off = 3 + start * bb;
    src.read_at(off as u64, &mut got).expect("virtual read");
    assert_eq!(got, &bed[off..off + bb * span], "multi-block mismatch");
}

/// For a biallelic cohort the virtual `.bim` must be field-for-field identical
/// to the real one — same IDs, same A1/A2, same order.
///
/// This is the guarantee downstream coverage QC depends on: a cohort scored via
/// PGEN and via PLINK 1.9 has to present the same variant identifiers, or the
/// match rate reads as zero and the WGS data looks worse than it is.
#[test]
fn virtual_bim_matches_real_bim_rows() {
    let vp = open_fixture();
    let real = std::fs::read_to_string(fixture_dir().join("ld.bim")).expect("reading bim fixture");
    let mut virt = vp.bim_source().expect("virtual bim");

    let mut n = 0usize;
    for real_line in real.lines() {
        let line = virt
            .next_line()
            .expect("virtual bim read")
            .expect("virtual bim ended early");
        let got = String::from_utf8(line.to_vec()).expect("utf8");
        let rf: Vec<&str> = real_line.split_whitespace().collect();
        let gf: Vec<&str> = got.split_whitespace().collect();
        assert_eq!(gf.len(), 6, "bim row {n} is not 6 columns: {got:?}");
        assert_eq!(gf, rf, "bim row {n} differs from the real .bim");
        n += 1;
    }
    assert_eq!(n, N_VARIANTS);
    assert!(
        virt.next_line().expect("virtual bim read").is_none(),
        "virtual bim has more rows than the real bim"
    );
}

/// A `.pgen` must be read as scattered small ranges, not swept end to end.
///
/// This is the whole reason for reading PGEN instead of the equivalent `.bed`:
/// a variant record is roughly an order of magnitude smaller than the
/// fixed-stride `.bed` record it decodes to, but that only pays off if a run
/// which touches a third of the variants also transfers about a third of the
/// file. Counting bytes at the byte-range layer keeps the guarantee honest
/// regardless of transport (local, `gs://`, http).
#[test]
fn scoring_a_subset_reads_only_part_of_the_pgen() {
    use gnomon::files::ByteRangeSource;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Wraps a byte source and tallies how much is actually requested.
    struct CountingSource {
        inner: Vec<u8>,
        bytes_read: Arc<AtomicU64>,
    }

    impl ByteRangeSource for CountingSource {
        fn len(&self) -> u64 {
            self.inner.len() as u64
        }

        fn read_at(
            &self,
            offset: u64,
            dst: &mut [u8],
        ) -> Result<(), gnomon::pipeline_error::PipelineError> {
            let start = offset as usize;
            let end = start + dst.len();
            if end > self.inner.len() {
                return Err(gnomon::pipeline_error::PipelineError::Io(
                    "read past end".into(),
                ));
            }
            dst.copy_from_slice(&self.inner[start..end]);
            self.bytes_read
                .fetch_add(dst.len() as u64, Ordering::Relaxed);
            Ok(())
        }
    }

    let dir = fixture_dir();
    let pgen_bytes = std::fs::read(dir.join("ld_p.pgen")).expect("reading pgen");
    let pgen_len = pgen_bytes.len() as u64;
    let bytes_read = Arc::new(AtomicU64::new(0));

    let source: Arc<dyn ByteRangeSource> = Arc::new(CountingSource {
        inner: pgen_bytes,
        bytes_read: Arc::clone(&bytes_read),
    });

    let pvar_path = dir.join("ld_p.pvar");
    let pvar: gnomon::adapt_plink2::PvarFactory =
        Arc::new(move || gnomon::files::open_text_source(&pvar_path));
    let mut psam = gnomon::files::open_text_source(&dir.join("ld_p.psam")).expect("psam");

    let vp = gnomon::adapt_plink2::open_virtual_plink19_from_sources(
        source,
        pvar,
        &mut *psam,
        GenomeBuild::Grch38,
    )
    .expect("opening pgen from sources");

    // Opening parses the header; measure only what the variant reads cost.
    let after_open = bytes_read.load(Ordering::Relaxed);

    let bed = vp.bed_source();
    let bb = block_bytes();
    let mut got = vec![0u8; bb];
    let mut touched = 0u64;
    for v in (0..N_VARIANTS).step_by(3) {
        let off = 3 + v * bb;
        bed.read_at(off as u64, &mut got).expect("virtual read");
        touched += 1;
    }

    let variant_bytes = bytes_read.load(Ordering::Relaxed) - after_open;
    let fraction_of_file = variant_bytes as f64 / pgen_len as f64;
    let fraction_of_variants = touched as f64 / N_VARIANTS as f64;

    // Reading a third of the variants must not mean reading the whole file.
    // The bound is deliberately loose: LD-compressed records also pull in their
    // anchor, so the bytes read legitimately exceed the variant fraction.
    assert!(
        fraction_of_file < 0.90,
        "reading {:.0}% of variants transferred {:.0}% of the .pgen \
         ({variant_bytes} of {pgen_len} bytes) -- ranges are not being honoured",
        fraction_of_variants * 100.0,
        fraction_of_file * 100.0,
    );
    eprintln!(
        "read {:.1}% of variants -> {:.1}% of the .pgen ({variant_bytes}/{pgen_len} bytes)",
        fraction_of_variants * 100.0,
        fraction_of_file * 100.0
    );
}
