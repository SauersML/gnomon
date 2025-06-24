use crate::batch::{self, SparseIndexPool};
use crate::decide::{self, DecisionContext, RunStrategy};
use crate::io;
use crate::types::{
    BimRowIndex, EffectAlleleDosage, PreparationResult, ReconciledVariantIndex, ScoreColumnIndex,
    WorkItem,
};
use ahash::AHashSet;
use crossbeam_channel::{bounded, Receiver};
use crossbeam_queue::ArrayQueue;
use memmap2::Mmap;
use num_cpus;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::thread;

// --- Pipeline Tuning Parameters ---

/// The maximum number of sparse work items that can be buffered in the channel.
/// Provides backpressure against a fast producer.
const SPARSE_CHANNEL_BOUND: usize = 8192;
/// The maximum number of dense work items that can be buffered in the channel.
const DENSE_CHANNEL_BOUND: usize = 4096;
/// The number of dense variants to process in a single person-major batch.
/// Tuned for L3 cache efficiency.
const DENSE_BATCH_SIZE: usize = 256;
/// The number of reusable memory buffers for variant data.
const BUFFER_POOL_SIZE: usize = 16384;

// ========================================================================================
//                          PUBLIC API, CONTEXT & ERROR HANDLING
// ========================================================================================

/// An iterator that pulls items from a channel and groups them into batches.
///
/// This is a `Send`-compatible replacement for `itertools::chunks` on a channel
/// iterator, enabling true streaming processing with `rayon::par_bridge`. It is
/// the key to enabling simultaneous I/O and computation for the dense path.
struct ChannelBatcher<T> {
    rx: Receiver<Result<T, PipelineError>>,
    batch_size: usize,
}

impl<T> ChannelBatcher<T> {
    fn new(rx: Receiver<Result<T, PipelineError>>, batch_size: usize) -> Self {
        Self { rx, batch_size }
    }
}

// The implementation of the `Iterator` trait is what allows this to be used in loops
// and with adapters like `par_bridge`.
impl<T: Send> Iterator for ChannelBatcher<T> {
    // The iterator yields a `Result` containing either a `Vec` of items (a batch)
    // or a `PipelineError` if one was sent by the producer.
    type Item = Result<Vec<T>, PipelineError>;

    fn next(&mut self) -> Option<Self::Item> {
        // First, block waiting for one item. If the channel is empty and has been
        // closed by the producer, `recv()` will return an error, and we'll return `None`,
        // ending the iteration. This is the correct way to terminate the stream.
        match self.rx.recv() {
            // Happy path: We received a valid work item from the producer.
            Ok(Ok(first_item)) => {
                let mut batch = Vec::with_capacity(self.batch_size);
                batch.push(first_item);

                // Greedily pull more items from the channel *without blocking* until
                // the batch is full or the channel is momentarily empty.
                while batch.len() < self.batch_size {
                    match self.rx.try_recv() {
                        Ok(Ok(item)) => batch.push(item),
                        // An error was sent down the channel. Stop and propagate it.
                        Ok(Err(e)) => return Some(Err(e)),
                        // The channel is empty or disconnected. The current batch is finished.
                        Err(_) => break,
                    }
                }
                // Return the completed (or partially-filled) batch.
                Some(Ok(batch))
            }
            // An error was sent down the channel as the first item. Propagate it.
            Ok(Err(e)) => Some(Err(e)),
            // The producer has disconnected the channel. End the iteration.
            Err(_) => None,
        }
    }
}

/// A specialized error type for the pipeline, allowing for robust, clonable error
/// propagation from any concurrent stage.
#[derive(Debug, Clone)]
pub enum PipelineError {
    Compute(String),
    Io(String),
    Producer(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Compute(e) => write!(f, "Computation error: {}", e),
            PipelineError::Io(e) => write!(f, "I/O error: {}", e),
            PipelineError::Producer(e) => write!(f, "Producer thread error: {}", e),
        }
    }
}
impl Error for PipelineError {}

// Enables easy conversion from batch errors into a pipeline error.
impl From<Box<dyn Error + Send + Sync>> for PipelineError {
    fn from(e: Box<dyn Error + Send + Sync>) -> Self {
        PipelineError::Compute(e.to_string())
    }
}

/// Owns shared resource pools and provides a handle to the read-only preparation results.
pub struct PipelineContext {
    pub prep_result: Arc<PreparationResult>,
    pub tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    pub sparse_index_pool: Arc<SparseIndexPool>,
}

impl PipelineContext {
    /// Creates a new `PipelineContext`, allocating all necessary memory pools.
    pub fn new(prep_result: Arc<PreparationResult>) -> Self {
        Self {
            prep_result,
            tile_pool: Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 4)),
            sparse_index_pool: Arc::new(SparseIndexPool::new()),
        }
    }
}

/// Executes the entire concurrent compute pipeline.
///
/// This is the primary public entry point. It is synchronous and returns the
/// final aggregated scores and counts upon successful completion.
pub fn run(
    context: &PipelineContext,
    plink_prefix: &Path,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // --- 1. Setup: Memory-map the file, create channels and a shared buffer pool ---
    let bed_path = plink_prefix.with_extension("bed");
    let bed_file =
        File::open(&bed_path).map_err(|e| PipelineError::Io(format!("Opening {}: {}", bed_path.display(), e)))?;
    let mmap = Arc::new(unsafe { Mmap::map(&bed_file).map_err(|e| PipelineError::Io(e.to_string()))? });
    mmap.advise(memmap2::Advice::Sequential)
        .map_err(|e| PipelineError::Io(e.to_string()))?;

    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(SPARSE_CHANNEL_BOUND);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(DENSE_CHANNEL_BOUND);

    let buffer_pool = Arc::new(ArrayQueue::new(BUFFER_POOL_SIZE));
    for _ in 0..BUFFER_POOL_SIZE {
        buffer_pool
            .push(Vec::with_capacity(
                context.prep_result.bytes_per_variant as usize,
            ))
            .unwrap();
    }

    // --- 2. Pre-computation & STRATEGY SELECTION ---
    // This is the core of the high-performance design. All decisions are made once, up front.
    let prep_result = &context.prep_result;

    // A. Create the single, `f32`-based context for the decision engine.
    // This performs the type casts ONCE per run, not millions of times per variant.
    let run_ctx = DecisionContext {
        n_cohort: prep_result.total_people_in_fam as f32,
        k_scores: prep_result.score_names.len() as f32,
        subset_frac: prep_result.num_people_to_score as f32
            / prep_result.total_people_in_fam as f32,
        freq: 0.0, // Dummy value, not used by the meta-model
    };

    // B. Call the meta-model ONCE to choose the global strategy.
    // let strategy = decide::choose_run_strategy(&run_ctx);
    // Benchmarks indicate UseComplexTree is usually but not always better.
    // Indicates UseComplexTree choices in decide.rs is not optimal
    let strategy = decide::RunStrategy::UseComplexTree;     // --- FORCED for now
    eprintln!("> Decision Engine Strategy: {:?}", strategy);

    // C. Calculate the single, global baseline score.
    let num_scores = prep_result.score_names.len();
    let stride = prep_result.stride();
    let master_baseline: Vec<f64> = (0..prep_result.num_reconciled_variants)
        .into_par_iter()
        .fold(
            || vec![0.0f64; num_scores], // Thread-local accumulator
            |mut local_baseline, i| {
                let flip_row_offset = i * stride;
                let flip_row = &prep_result.flip_mask_matrix()[flip_row_offset..flip_row_offset + stride];
                let weight_row = &prep_result.weights_matrix()[flip_row_offset..flip_row_offset + stride];
                for k in 0..num_scores {
                    if flip_row[k] == 1 {
                        local_baseline[k] += 2.0 * weight_row[k] as f64;
                    }
                }
                local_baseline
            },
        )
        .reduce(
            || vec![0.0f64; num_scores], // Identity for reduction
            |mut a, b| {
                a.par_iter_mut().zip(b).for_each(|(v_a, v_b)| *v_a += v_b);
                a
            },
        );

    // --- 3. Orchestration: Use a scoped thread for safe producer/consumer execution ---
    let final_result: Result<(Vec<f64>, Vec<u32>), PipelineError> = thread::scope(|s| {
        // The producer's logic is constructed here, before the thread is spawned.
        // This closure captures all necessary resources to be moved into the thread.
        let producer_logic = {
            let mmap = Arc::clone(&mmap);
            let prep_result = Arc::clone(&context.prep_result);
            let buffer_pool = Arc::clone(&buffer_pool);

            move || match strategy {
                RunStrategy::UseSimpleTree => {
                    // TIER 2 DECISION: The global path is determined ONCE.
                    let global_path = decide::decide_path_without_freq(&run_ctx);
                    eprintln!("> Global path determined for all variants: {:?}", global_path);
                    // Create a trivial closure that captures the constant path.
                    let path_decider = |_variant_data: &[u8]| global_path;
                    // Call the generic producer with the specialized, trivial closure.
                    io::producer_thread(mmap, prep_result, sparse_tx, dense_tx, buffer_pool, path_decider);
                }
                RunStrategy::UseComplexTree => {
                    eprintln!("> Path will be determined per-variant using frequency.");
                    // Create a closure that performs the real per-variant work.
                    let path_decider = |variant_data: &[u8]| {
                        let current_freq = batch::assess_variant_density(variant_data, run_ctx.n_cohort as usize);
                        let variant_ctx = DecisionContext { freq: current_freq, ..run_ctx };
                        decide::decide_path_with_freq(&variant_ctx)
                    };
                    // Call the generic producer with the specialized, complex closure.
                    io::producer_thread(mmap, prep_result, sparse_tx, dense_tx, buffer_pool, path_decider);
                }
            }
        };

        let producer_handle = s.spawn(producer_logic);

        // Run both consumer streams in parallel on the Rayon thread pool.
        let (sparse_result, dense_result) = rayon::join(
            || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
            || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
        );

        // This join is critical for propagating panics from the producer.
        producer_handle.join().map_err(|_| {
            PipelineError::Producer("Producer thread panicked.".to_string())
        })?;

        // --- 4. Aggregate final results from both consumer streams ---
        let (sparse_adjustments, sparse_counts) = sparse_result?;
        let (dense_adjustments, dense_counts) = dense_result?;

        let num_people = prep_result.num_people_to_score;
        let mut final_scores = Vec::with_capacity(num_people * num_scores);
        for _ in 0..num_people {
            final_scores.extend_from_slice(&master_baseline);
        }

        let mut final_counts = vec![0u32; num_people * num_scores];
        final_counts.par_iter_mut().zip(sparse_counts).for_each(|(m, p)| *m += p);
        final_counts.par_iter_mut().zip(dense_counts).for_each(|(m, p)| *m += p);

        final_scores.par_iter_mut().zip(sparse_adjustments).for_each(|(m, p)| *m += p);
        final_scores.par_iter_mut().zip(dense_adjustments).for_each(|(m, p)| *m += p);
        
        // --- 5. Final Step: Resolve complex variants and apply their contributions ---
        if !prep_result.complex_rules.is_empty() {
            eprintln!(
                "> Resolving {} unique complex variant rule(s)...",
                prep_result.complex_rules.len()
            );
            resolve_complex_variants(prep_result, &mut final_scores, &mut final_counts, &mmap)?;
        }

        Ok((final_scores, final_counts))
    });

    final_result
}

// ========================================================================================
//                        PIPELINE STAGE IMPLEMENTATIONS
// ========================================================================================

/// A RAII guard that ensures a byte buffer is automatically returned to the shared
/// buffer pool when it goes out of scope. This is critical for preventing resource
// leaks in the consumer streams, especially when errors occur.
struct BufferGuard<'a> {
    /// The buffer being managed. Wrapped in an `Option` to allow ownership to be
    /// taken in the `drop` implementation.
    buffer: Option<Vec<u8>>,
    /// A reference to the shared pool where the buffer will be returned.
    pool: &'a ArrayQueue<Vec<u8>>,
}

impl<'a> Drop for BufferGuard<'a> {
    fn drop(&mut self) {
        // When the guard is dropped, it returns its buffer to the pool.
        if let Some(mut buf) = self.buffer.take() {
            buf.clear();
            let _ = self.pool.push(buf);
        }
    }
}

type ConsumerResult = Result<(Vec<f64>, Vec<u32>), PipelineError>;

/// A contention-free consumer for the sparse variant stream, using Rayon's
/// fold/reduce pattern for maximum parallelism with no locks.
fn process_sparse_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = prep_result.num_people_to_score * prep_result.score_names.len();

    // The fold/reduce pattern creates thread-local accumulators for scores and counts.
    // After processing a work item, its data buffer is immediately returned to the
    // shared pool, creating a true, continuous recycling system.
    let final_result = rx
        .into_iter() // Convert the channel to a blocking iterator.
        .par_bridge() // Bridge it to a Rayon parallel iterator.
        .try_fold(
            || (vec![0.0f64; result_size], vec![0u32; result_size]), // Each thread gets its own accumulator.
            |mut acc, work_result| {
                // The work_item and its buffer are processed within this scope.
                // The `_guard` ensures the buffer is returned to the pool when this
                // scope ends, whether by success or by `?` propagating an error.
                {
                    let work_item = work_result?;
                    let _guard = BufferGuard {
                        buffer: Some(work_item.data),
                        pool: &buffer_pool,
                    };

                    batch::run_variant_major_path(
                        // The guard holds the buffer, so we borrow it from there.
                        _guard.buffer.as_ref().unwrap(),
                        prep_result,
                        &mut acc.0,
                        &mut acc.1,
                        work_item.reconciled_variant_index,
                    )?;
                }
                Ok::<_, PipelineError>(acc)
            },
        )
        .try_reduce(
            || (vec![0.0f64; result_size], vec![0u32; result_size]), // Identity for the reduction.
            |mut a, b| {
                // Combine accumulators from two threads in parallel.
                a.0.par_iter_mut().zip(b.0).for_each(|(v_a, v_b)| *v_a += v_b);
                a.1.par_iter_mut().zip(b.1).for_each(|(v_a, v_b)| *v_a += v_b);
                Ok(a)
            },
        )?;

    // `try_reduce` returns `Result<(scores, counts), PipelineError>`.
    // The `?` operator has already unwrapped the Result, leaving just the tuple.
    // With an identity function, try_reduce handles empty streams by returning the identity.
    Ok(final_result)
}

/// A contention-free consumer for the dense variant stream. It uses a custom
/// batching iterator to group items, which are then processed in parallel by Rayon.
/// This implementation allows I/O and computation to run concurrently.
fn process_dense_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = prep_result.num_people_to_score * prep_result.score_names.len();

    // Instantiate our new Send-compatible batching iterator.
    let batch_iterator = ChannelBatcher::new(rx, DENSE_BATCH_SIZE);

    // Use the exact same fold/reduce pattern as the sparse stream, but on batches.
    let final_result = batch_iterator
        .par_bridge() // This is now possible and correct.
        .try_fold(
            || { // Per-thread accumulator initializer
                (
                    vec![0.0f64; result_size],
                    vec![0u32; result_size],
                    Vec::with_capacity(DENSE_BATCH_SIZE * (prep_result.bytes_per_variant as usize)),
                )
            },
            |mut acc, batch_result| {
                // The `?` operator handles propagating errors from the channel.
                let batch = batch_result?;
                if batch.is_empty() { return Ok(acc); }

                let reconciled_indices: Vec<ReconciledVariantIndex> =
                    batch.iter().map(|wi| wi.reconciled_variant_index).collect();

                let concatenated_data = &mut acc.2;
                concatenated_data.clear();
                
                // BufferGuard ensures all variant data buffers are returned to the pool,
                // even if an error occurs during computation.
                let guards: Vec<_> = batch.into_iter().map(|wi| {
                    concatenated_data.extend_from_slice(&wi.data);
                    BufferGuard { buffer: Some(wi.data), pool: &buffer_pool }
                }).collect();

                let stride = prep_result.stride();
                let mut weights_for_batch = Vec::with_capacity(reconciled_indices.len() * stride);
                let mut flips_for_batch = Vec::with_capacity(reconciled_indices.len() * stride);
                for &reconciled_idx in &reconciled_indices {
                    let src_offset = reconciled_idx.0 as usize * stride;
                    weights_for_batch.extend_from_slice(&prep_result.weights_matrix()[src_offset..src_offset + stride]);
                    flips_for_batch.extend_from_slice(&prep_result.flip_mask_matrix()[src_offset..src_offset + stride]);
                }

                batch::run_person_major_path(
                    concatenated_data, &weights_for_batch, &flips_for_batch,
                    &reconciled_indices, prep_result, &mut acc.0, &mut acc.1,
                    &context.tile_pool, &context.sparse_index_pool,
                )?;
                
                drop(guards); // Explicitly drop to return buffers to the pool.

                Ok::<_, PipelineError>(acc)
            },
        )
        .try_reduce(
            || (vec![0.0; result_size], vec![0; result_size], Vec::new()),
            |mut a, b| {
                a.0.par_iter_mut().zip(b.0).for_each(|(v_a, v_b)| *v_a += v_b);
                a.1.par_iter_mut().zip(b.1).for_each(|(v_a, v_b)| *v_a += v_b);
                Ok(a)
            },
        )?;

    // The `?` operator unwrapped the `Result` from the reduction. If the stream was
    // empty, `try_reduce` (on a `TryFold` iterator) returns the identity value, so
    // `final_result` correctly contains the initial empty vectors. We just need to destructure the tuple.
    let (scores, counts, _) = final_result;
    Ok((scores, counts))
}

/// A small, inlineable helper to read a 2-bit packed genotype from the memory-mapped
/// .bed file. This encapsulates the complex byte and bit offset calculations.
///
/// # Arguments
/// * `mmap`: A slice representing the entire memory-mapped .bed file.
/// * `bytes_per_variant`: The number of bytes for each variant's data, pre-calculated.
/// * `bim_row_index`: The 0-based row index of the variant in the original .bim file.
/// * `fam_index`: The 0-based index of the person in the original .fam file.
///
/// # Returns
/// A `u8` containing the 2-bit genotype (`00`, `01`, `10`, or `11`).
#[inline(always)]
fn get_packed_genotype(
    mmap: &[u8],
    bytes_per_variant: u64,
    bim_row_index: BimRowIndex,
    fam_index: u32,
) -> u8 {
    // The +3 skips the PLINK .bed file magic number (0x6c, 0x1b, 0x01).
    let variant_start_offset = (3 + bim_row_index.0 as u64 * bytes_per_variant) as usize;
    let person_byte_offset = (fam_index / 4) as usize;
    let final_byte_offset = variant_start_offset + person_byte_offset;

    // Each person's genotype is 2 bits. We find which 2-bit slot in the byte it is.
    let bit_offset_in_byte = (fam_index % 4) * 2;

    // This indexing is safe because the preparation phase guarantees all indices are valid.
    let packed_byte = unsafe { *mmap.get_unchecked(final_byte_offset) };
    (packed_byte >> bit_offset_in_byte) & 0b11
}

/// The "slow path" resolver for complex, multiallelic variants.
///
/// This function runs *after* the main high-performance pipeline is complete. It
/// iterates through each person and resolves their score contributions for the small
/// set of variants that could not be handled by the fast path.
fn resolve_complex_variants(
    prep_result: &Arc<PreparationResult>,
    final_scores: &mut [f64],
    final_missing_counts: &mut [u32],
    mmap: &Arc<Mmap>,
) -> Result<(), PipelineError> {
    let num_scores = prep_result.score_names.len();

    final_scores
        .par_chunks_mut(num_scores)
        .zip(final_missing_counts.par_chunks_mut(num_scores))
        .enumerate()
        .try_for_each(|(person_output_idx, (person_scores_slice, person_counts_slice))| {
            let original_fam_idx = prep_result.output_idx_to_fam_idx[person_output_idx];

            for group_rule in &prep_result.complex_rules {
                // --- Step 1: Gather Evidence ---
                // For this person, find ALL valid, non-missing genotypes for this variant
                // across all possible contexts.
                let mut valid_interpretations =
                    Vec::with_capacity(group_rule.possible_contexts.len());

                for context in &group_rule.possible_contexts {
                    let (bim_idx, ..) = context;
                    let packed_geno = get_packed_genotype(
                        mmap,
                        prep_result.bytes_per_variant,
                        *bim_idx,
                        original_fam_idx,
                    );

                    // A packed value of `0b01` means the genotype is missing.
                    if packed_geno != 0b01 {
                        valid_interpretations.push((packed_geno, context));
                    }
                }

                // --- Step 2: Apply Decision Policy ---
                match valid_interpretations.len() {
                    0 => { // CASE: Truly Missing
                        // The person's genotype was missing for this variant under all contexts.
                        // BUG FIX: Use a HashSet to increment the missing count only ONCE per
                        // score column, even if multiple rules for this locus affect the same score.
                        let mut counted_cols: AHashSet<ScoreColumnIndex> = AHashSet::new();
                        for score_info in &group_rule.score_applications {
                            counted_cols.insert(score_info.score_column_index);
                        }
                        for score_col in counted_cols {
                            person_counts_slice[score_col.0] += 1;
                        }
                    }
                    1 => { // CASE: Success - One Unambiguous Interpretation
                        let (winning_geno, winning_context) = valid_interpretations[0];
                        let (_bim_idx, winning_a1, winning_a2) = winning_context;

                        for score_info in &group_rule.score_applications {
                            let effect_allele = &score_info.effect_allele;
                            let score_col = score_info.score_column_index.0;
                            let weight = score_info.weight as f64;

                            // BUG FIX: If the locus is present, it is never counted as missing.
                            // If the effect allele for this specific rule does not exist in the
                            // winning genotype, its dosage is simply zero.
                            if effect_allele != winning_a1 && effect_allele != winning_a2 {
                                continue;
                            }

                            // The PLINK .bed file encodes genotypes relative to (a1, a2).
                            // 0b00=hom_a1, 0b10=het, 0b11=hom_a2.
                            let dosage: f64 = if effect_allele == winning_a1 {
                                match winning_geno {
                                    0b00 => 2.0, 0b10 => 1.0, 0b11 => 0.0, _ => unreachable!(),
                                }
                            } else { // effect_allele must be winning_a2
                                match winning_geno {
                                    0b00 => 0.0, 0b10 => 1.0, 0b11 => 2.0, _ => unreachable!(),
                                }
                            };
                            
                            // The slow path should not interact with the baseline.
                            // Variants processed here were never part of the baseline calculation,
                            // so we simply add their direct `dosage * weight` contribution.
                            let contribution = dosage * weight;
                            person_scores_slice[score_col] += contribution;
                        }
                    }
                    _ => { // CASE: Fatal Error - Contradictory Data
                        let iid = &prep_result.final_person_iids[person_output_idx];
                        let first_context_id = group_rule.possible_contexts[0].0 .0;
                        return Err(PipelineError::Compute(format!(
                            "Fatal data inconsistency for individual '{}'. Variant at location corresponding to BIM row index '{}' has conflicting, non-missing genotypes in the input .bed file.",
                            iid, first_context_id
                        )));
                    }
                }
            }
            Ok(())
        })?;

    Ok(())
}
