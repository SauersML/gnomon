// ========================================================================================
//
//                 GNOMON COMPUTE PATH PERFORMANCE BENCHMARK
//
// ========================================================================================
//
// This benchmark empirically measures the performance crossover point between the
// "pivot" (person-major) and "no-pivot" (variant-major) compute paths based on
// allele frequency (a.k.a. "variant density").
//
// ========================================================================================

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use gnomon::batch;
use gnomon::kernel;
use gnomon::types::{
    PersonSubset, PreparationResult, ReconciledVariantIndex, ScoreColumnIndex,
};

use crossbeam_queue::ArrayQueue;
use rand::seq::index;
use std::sync::Arc;
use std::time::Instant;

// --- Benchmark Tuning Parameters ---

/// The number of individuals to simulate in the cohort.
const NUM_PEOPLE: usize = 20_000;
/// The number of score columns to simulate.
const NUM_SCORES: usize = 10;
/// The number of dense variants to process in a single person-major batch,
/// mirroring the real application's pipeline (`pipeline::DENSE_BATCH_SIZE`).
const PIVOT_PATH_BATCH_SIZE: usize = 256;
/// The allele frequencies to test. This array defines the x-axis of the final plot.
const ALLELE_FREQUENCIES: [f32; 15] = [
    0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.2, 0.4, 0.5,
];

/// Creates a realistic, shared "computation blueprint" for the benchmark.
///
/// This function simulates the output of the `prepare` module, providing a valid
/// `PreparationResult` that both compute paths can use. It is created once and
/// shared using an `Arc`. The number of variants is configurable to support
/// both single-variant and batch-based tests.
fn setup_benchmark_context(
    num_people: usize,
    num_scores: usize,
    num_variants: usize,
) -> Arc<PreparationResult> {
    let bytes_per_variant = ((num_people as u64) + 3) / 4;
    let stride = (num_scores + kernel::LANE_COUNT - 1) / kernel::LANE_COUNT * kernel::LANE_COUNT;

    // Simulate matrices for the specified number of variants.
    let matrix_size = num_variants * stride;
    let weights_matrix = vec![1.0f32; matrix_size];
    let flip_mask_matrix = vec![0u8; matrix_size];
    let variant_to_scores_map = vec![vec![ScoreColumnIndex(0)]; num_variants];

    let output_idx_to_fam_idx: Vec<u32> = (0..num_people as u32).collect();

    // The contents of many fields are not critical for this specific benchmark,
    // but they must be valid for the `PreparationResult` to be constructed.
    let prep_result = PreparationResult::new(
        weights_matrix,
        flip_mask_matrix,
        stride,
        vec![], // required_bim_indices (not directly used)
        vec![], // complex_rules
        (0..num_scores).map(|i| format!("score_{}", i)).collect(),
        vec![1; num_scores], // score_variant_counts
        variant_to_scores_map,
        PersonSubset::All,
        (0..num_people).map(|i| format!("IID_{}", i)).collect(),
        num_people,
        num_people,
        num_variants, // total_variants_in_bim
        num_variants, // num_reconciled_variants
        bytes_per_variant,
        (0..num_people).map(|i| Some(i as u32)).collect(),
        output_idx_to_fam_idx,
    );
    Arc::new(prep_result)
}

/// Generates PLINK .bed format data for a single variant.
///
/// This function creates a `Vec<u8>` with a specified number of non-reference
/// genotypes, which serves as a direct proxy for the allele frequency / density.
fn generate_variant_data(num_people: usize, allele_frequency: f32) -> Vec<u8> {
    let bytes_per_variant = (num_people + 3) / 4;
    let mut variant_data = vec![0u8; bytes_per_variant]; // Default: all homozygous reference (0b00)

    let num_people_with_alt = (num_people as f32 * allele_frequency).round() as usize;

    let mut rng = rand::thread_rng();
    let indices_to_set: Vec<usize> =
        index::sample(&mut rng, num_people, num_people_with_alt).into_vec();

    for person_idx in indices_to_set {
        let byte_index = person_idx / 4;
        let bit_offset = (person_idx % 4) * 2;
        // Set genotype to heterozygous (0b10).
        variant_data[byte_index] |= 0b10 << bit_offset;
    }

    variant_data
}

/// The main benchmark function that orchestrates the performance test.
fn benchmark_path_crossover(c: &mut Criterion) {
    // --- Global Setup (done once) ---
    // Create a single, shared PreparationResult sized for the larger Pivot path batch.
    // This is efficient and can be safely used by the No-Pivot path as well.
    let prep_result =
        setup_benchmark_context(NUM_PEOPLE, NUM_SCORES, PIVOT_PATH_BATCH_SIZE);
    let result_size = NUM_PEOPLE * NUM_SCORES;

    let mut scores_out = vec![0.0f64; result_size];
    let mut missing_counts_out = vec![0u32; result_size];

    // --- Benchmark Group Definition ---
    let mut group = c.benchmark_group("Path Crossover: Pivot vs. No-Pivot");
    // Report throughput in terms of total people scored per variant.
    // This gives a consistent y-axis of "time per variant" on the final plot.
    group.throughput(Throughput::Elements(NUM_PEOPLE as u64));

    for &freq in ALLELE_FREQUENCIES.iter() {
        // --- 1. BENCHMARK VARIANT-MAJOR (NO-PIVOT) PATH ---
        // This path is benchmarked correctly: one variant at a time.
        let variant_data = generate_variant_data(NUM_PEOPLE, freq);
        group.bench_with_input(
            BenchmarkId::new("No-Pivot (Variant-Major)", freq),
            &variant_data,
            |b, data| {
                b.iter(|| {
                    batch::run_variant_major_path(
                        black_box(data),
                        black_box(&prep_result),
                        black_box(&mut scores_out),
                        black_box(&mut missing_counts_out),
                        // We test processing the first variant in the context.
                        black_box(ReconciledVariantIndex(0)),
                    )
                    .unwrap();
                });
            },
        );

        // --- 2. BENCHMARK PERSON-MAJOR (PIVOT) PATH (CORRECTED) ---
        // This path is benchmarked under its intended operating conditions: processing a
        // full batch of dense variants.
        let tile_pool = Arc::new(ArrayQueue::new(4));
        let sparse_index_pool = Arc::new(batch::SparseIndexPool::new());

        // Prepare a full batch of variants with the same frequency.
        let mut batch_variant_data = Vec::with_capacity(
            PIVOT_PATH_BATCH_SIZE * prep_result.bytes_per_variant as usize,
        );
        for _ in 0..PIVOT_PATH_BATCH_SIZE {
            batch_variant_data.extend_from_slice(&generate_variant_data(NUM_PEOPLE, freq));
        }

        let reconciled_indices: Vec<_> = (0..PIVOT_PATH_BATCH_SIZE as u32)
            .map(ReconciledVariantIndex)
            .collect();
        // The weights/flips are for the full batch, taken from the shared prep_result.
        let weights_for_batch = prep_result.weights_matrix().to_vec();
        let flips_for_batch = prep_result.flip_mask_matrix().to_vec();

        group.bench_with_input(
            BenchmarkId::new("Pivot (Person-Major)", freq),
            &batch_variant_data, // Note: passing the full batch data
            |b, data| {
                // We use `iter_custom` to measure the time for the entire batch and then
                // divide by the batch size. This calculates the amortized, per-variant
                // cost, which is the correct metric for a fair comparison.
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        batch::run_person_major_path(
                            black_box(data),
                            black_box(&weights_for_batch),
                            black_box(&flips_for_batch),
                            black_box(&reconciled_indices),
                            black_box(&prep_result),
                            black_box(&mut scores_out),
                            black_box(&mut missing_counts_out),
                            black_box(&tile_pool),
                            black_box(&sparse_index_pool),
                        )
                        .unwrap();
                    }
                    // Return the amortized time per variant.
                    start.elapsed() / (PIVOT_PATH_BATCH_SIZE as u32)
                });
            },
        );
    }

    group.finish();
}

// Boilerplate to register the benchmark group with the Criterion runner.
criterion_group!(benches, benchmark_path_crossover);
criterion_main!(benches);
