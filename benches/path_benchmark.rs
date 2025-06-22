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

// --- Benchmark Tuning Parameters ---

/// The number of individuals to simulate in the cohort.
const NUM_PEOPLE: usize = 20_000;
/// The number of score columns to simulate.
const NUM_SCORES: usize = 10;
/// The allele frequencies to test. This array defines the x-axis of the final plot.
const ALLELE_FREQUENCIES: [f32; 15] = [
    0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.2, 0.4, 0.5,
];

/// Creates a realistic, shared "computation blueprint" for the benchmark.
///
/// This function simulates the output of the `prepare` module, providing a valid
/// `PreparationResult` that both compute paths can use. It is created once and
/// shared using an `Arc` to mimic the behavior of the real application pipeline.
fn setup_benchmark_context(num_people: usize, num_scores: usize) -> Arc<PreparationResult> {
    let bytes_per_variant = ((num_people as u64) + 3) / 4;
    let stride = (num_scores + kernel::LANE_COUNT - 1) / kernel::LANE_COUNT * kernel::LANE_COUNT;

    // Simulate matrices for a single variant, as we benchmark one variant at a time.
    let weights_matrix = vec![1.0f32; stride];
    let flip_mask_matrix = vec![0u8; stride];

    let output_idx_to_fam_idx: Vec<u32> = (0..num_people as u32).collect();

    // The contents of many fields are not critical for this specific benchmark,
    // but they must be valid for the `PreparationResult` to be constructed.
    let prep_result = PreparationResult::new(
        weights_matrix,
        flip_mask_matrix,
        stride,
        vec![], // required_bim_indices (not directly used by the tested functions)
        vec![], // complex_rules
        (0..num_scores).map(|i| format!("score_{}", i)).collect(),
        vec![1; num_scores], // score_variant_counts
        vec![vec![ScoreColumnIndex(0)]], // variant_to_scores_map
        PersonSubset::All,
        (0..num_people).map(|i| format!("IID_{}", i)).collect(),
        num_people,
        num_people,
        1, // total_variants_in_bim
        1, // num_reconciled_variants
        bytes_per_variant,
        (0..num_people).map(|i| Some(i as u32)).collect(),
        output_idx_to_fam_idx,
    );
    Arc::new(prep_result)
}

/// Generates PLINK .bed format data for a single variant.
///
/// This function creates a `Vec<u8>` with a specified number of non-reference
/// genotypes, which serves as a direct proxy for the allele frequency / density
/// that the `assess_path` function measures.
///
/// # Arguments
/// * `num_people`: The total number of individuals in the cohort.
/// * `allele_frequency`: The target frequency of the alternate allele.
fn generate_variant_data(num_people: usize, allele_frequency: f32) -> Vec<u8> {
    let bytes_per_variant = (num_people + 3) / 4;
    let mut variant_data = vec![0u8; bytes_per_variant]; // Default: all homozygous reference (0b00)

    let num_people_with_alt = (num_people as f32 * allele_frequency).round() as usize;

    let mut rng = rand::thread_rng();
    // Get a random sample of unique person indices to modify.
    let indices_to_set: Vec<usize> =
        index::sample(&mut rng, num_people, num_people_with_alt).into_vec();

    for person_idx in indices_to_set {
        let byte_index = person_idx / 4;
        let bit_offset = (person_idx % 4) * 2;
        // Set genotype to heterozygous (0b10). This introduces set bits, which
        // is what the density assessment function counts.
        variant_data[byte_index] |= 0b10 << bit_offset;
    }

    variant_data
}

/// The main benchmark function that orchestrates the performance test.
fn benchmark_path_crossover(c: &mut Criterion) {
    // --- Global Setup (done once) ---
    let prep_result = setup_benchmark_context(NUM_PEOPLE, NUM_SCORES);
    let result_size = NUM_PEOPLE * NUM_SCORES;

    // These buffers are reused across iterations to avoid allocation overhead in the timing loop.
    let mut scores_out = vec![0.0f64; result_size];
    let mut missing_counts_out = vec![0u32; result_size];

    // --- Benchmark Group Definition ---
    let mut group = c.benchmark_group("Path Crossover: Pivot vs. No-Pivot");
    // Report throughput in terms of people scored per iteration.
    // The final report will show `time / num_people`.
    group.throughput(Throughput::Elements(NUM_PEOPLE as u64));

    for &freq in ALLELE_FREQUENCIES.iter() {
        // Data for this specific frequency is generated here.
        let variant_data = generate_variant_data(NUM_PEOPLE, freq);

        // --- 1. BENCHMARK VARIANT-MAJOR (NO-PIVOT) PATH ---
        group.bench_with_input(
            BenchmarkId::new("No-Pivot (Variant-Major)", freq),
            &variant_data,
            |b, data| {
                b.iter(|| {
                    // The `black_box` calls are essential to prevent the compiler
                    // from optimizing away the function call or its arguments.
                    batch::run_variant_major_path(
                        black_box(data),
                        black_box(&prep_result),
                        black_box(&mut scores_out),
                        black_box(&mut missing_counts_out),
                        black_box(ReconciledVariantIndex(0)),
                    )
                    .unwrap();
                });
            },
        );

        // --- 2. BENCHMARK PERSON-MAJOR (PIVOT) PATH ---
        // Setup specific to the person-major path.
        let tile_pool = Arc::new(ArrayQueue::new(4));
        let sparse_index_pool = Arc::new(batch::SparseIndexPool::new());
        // Simulate a batch of 1 variant.
        let reconciled_indices = vec![ReconciledVariantIndex(0)];
        let weights_for_batch = prep_result.weights_matrix().to_vec();
        let flips_for_batch = prep_result.flip_mask_matrix().to_vec();

        group.bench_with_input(
            BenchmarkId::new("Pivot (Person-Major)", freq),
            &variant_data,
            |b, data| {
                b.iter(|| {
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
                });
            },
        );
    }

    group.finish();
}

// Boilerplate to register the benchmark group with the Criterion runner.
criterion_group!(benches, benchmark_path_crossover);
criterion_main!(benches);
