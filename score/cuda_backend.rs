use crate::score::complex::{ComplexVariantResolver, resolve_complex_variants};
use crate::score::decide::ComputePath;
use crate::score::io;
use crate::score::pipeline::{PipelineContext, PipelineError};
use crate::score::types::{
    BimRowIndex, FilesetBoundary, PipelineKind, PreparationResult, WorkItem,
};
use ahash::AHashMap;
use crossbeam_channel::{Receiver, bounded};
use crossbeam_queue::ArrayQueue;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DriverError,
    LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const DENSE_CHANNEL_BOUND: usize = 4096;
const BUFFER_POOL_SIZE: usize = 16384;
const SPOOL_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const MIN_GPU_WORK: usize = 100_000;
const MIN_MEGA_BATCH_VARIANTS: usize = 256;
const MAX_MEGA_BATCH_VARIANTS: usize = 4096;

fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
    let draw_target = if std::io::stderr().is_terminal() {
        ProgressDrawTarget::stderr_with_hz(20)
    } else {
        ProgressDrawTarget::hidden()
    };
    let pb = ProgressBar::with_draw_target(Some(len), draw_target);
    pb.set_style(
        ProgressStyle::with_template(
            "\n> [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message(message.to_string());
    pb
}

const CUDA_KERNELS: &str = r#"
extern "C" __global__ void unpack_plink(
    const unsigned char* packed,
    const unsigned int* out_to_fam,
    int num_people,
    int batch_variants,
    int bytes_per_variant,
    float* dosage,
    float* missing
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_people * batch_variants;
    if (idx >= total) return;

    int person = idx / batch_variants;
    int variant = idx % batch_variants;

    unsigned int fam_idx = out_to_fam[person];
    int byte_idx = (int)(fam_idx >> 2);
    int bit_shift = (int)((fam_idx & 3u) << 1);

    unsigned char b = packed[(size_t)variant * (size_t)bytes_per_variant + (size_t)byte_idx];
    unsigned char gt = (b >> bit_shift) & 0x3u;

    float d = 0.0f;
    float m = 0.0f;
    if (gt == 1u) {
        m = 1.0f;
    } else if (gt == 2u) {
        d = 1.0f;
    } else if (gt == 3u) {
        d = 2.0f;
    }

    dosage[idx] = d;
    missing[idx] = m;
}

extern "C" __global__ void build_batch_mats(
    const float* weights,
    const unsigned char* flips,
    const float* count_weights,
    const unsigned int* reconciled_indices,
    int batch_variants,
    int stride,
    int num_scores,
    float* out_effective,
    float* out_missing_corr,
    float* out_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_variants * num_scores;
    if (idx >= total) return;

    int v = idx / num_scores;
    int s = idx % num_scores;
    unsigned int reconciled = reconciled_indices[v];
    int src = (int)reconciled * stride + s;

    float w = weights[src];
    unsigned char f = flips[src];
    out_count[idx] = count_weights[src];
    if (f == 1u) {
        out_effective[idx] = -w;
        out_missing_corr[idx] = -2.0f * w;
    } else {
        out_effective[idx] = w;
        out_missing_corr[idx] = 0.0f;
    }
}
"#;

struct SpoolState {
    writer: BufWriter<File>,
    offsets: AHashMap<BimRowIndex, u64>,
    cursor: u64,
}

struct BufferGuard<'a> {
    buffer: Option<Vec<u8>>,
    pool: &'a ArrayQueue<Vec<u8>>,
}

impl<'a> Drop for BufferGuard<'a> {
    fn drop(&mut self) {
        if let Some(mut buf) = self.buffer.take() {
            buf.clear();
            let _ = self.pool.push(buf);
        }
    }
}

struct CudaRuntime {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    unpack_kernel: CudaFunction,
    build_batch_mats_kernel: CudaFunction,
    full_weights: CudaSlice<f32>,
    full_flips: CudaSlice<u8>,
    full_count_weights: CudaSlice<f32>,
    output_map: CudaSlice<u32>,
    mega_batch_variants: usize,
    pinned_staging: PinnedHostSlice<u8>,
}

pub fn try_run_cuda(
    context: &PipelineContext,
) -> Result<Option<(Vec<f64>, Vec<u32>)>, PipelineError> {
    let prep = &context.prep_result;
    if prep
        .num_people_to_score
        .saturating_mul(prep.score_names.len())
        < MIN_GPU_WORK
    {
        eprintln!("> Backend: CPU fallback (problem size below CUDA threshold)");
        return Ok(None);
    }

    let runtime = match CudaRuntime::new(prep) {
        Ok(runtime) => {
            eprintln!("> Backend: CUDA");
            runtime
        }
        Err(reason) => {
            eprintln!("> Backend: CPU fallback ({reason})");
            return Ok(None);
        }
    };

    let result = match &prep.pipeline_kind {
        PipelineKind::SingleFile(path) => run_single_file_cuda(context, path, runtime)?,
        PipelineKind::MultiFile(boundaries) => run_multi_file_cuda(context, boundaries, runtime)?,
    };

    Ok(Some(result))
}

impl CudaRuntime {
    fn new(prep: &PreparationResult) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        ctx.bind_to_thread()
            .map_err(|e| format!("Failed to bind CUDA context: {e:?}"))?;
        let stream = ctx.default_stream();

        let (free_mem, _) = cudarc::driver::result::mem_get_info()
            .map_err(|e| format!("Failed to query device memory: {e:?}"))?;

        let count_weights_len = prep.num_reconciled_variants * prep.stride();
        let static_bytes = prep.weights_matrix().len() * std::mem::size_of::<f32>()
            + prep.flip_mask_matrix().len() * std::mem::size_of::<u8>()
            + count_weights_len * std::mem::size_of::<f32>()
            + prep.output_idx_to_fam_idx.len() * std::mem::size_of::<u32>();

        let result_size = prep.num_people_to_score * prep.score_names.len();
        let constant_batch_bytes = result_size * std::mem::size_of::<f32>() * 3;

        let mut mega = MAX_MEGA_BATCH_VARIANTS;
        let budget = (free_mem as f64 * 0.8) as usize;
        while mega > MIN_MEGA_BATCH_VARIANTS {
            let per_variant_bytes = prep.bytes_per_variant as usize
                + prep.num_people_to_score * std::mem::size_of::<f32>() * 2
                + prep.score_names.len() * std::mem::size_of::<f32>() * 3;
            let required = static_bytes
                + constant_batch_bytes
                + per_variant_bytes * mega
                + mega * prep.bytes_per_variant as usize;
            if required <= budget {
                break;
            }
            mega /= 2;
        }

        if mega < MIN_MEGA_BATCH_VARIANTS {
            return Err("Insufficient GPU memory for minimum CUDA mega-batch".to_string());
        }

        let full_weights = stream
            .memcpy_stod(prep.weights_matrix())
            .map_err(|e| format!("Failed to upload full weights matrix: {e:?}"))?;
        let full_flips = stream
            .memcpy_stod(prep.flip_mask_matrix())
            .map_err(|e| format!("Failed to upload full flip mask matrix: {e:?}"))?;
        let output_map = stream
            .memcpy_stod(&prep.output_idx_to_fam_idx)
            .map_err(|e| format!("Failed to upload output index map: {e:?}"))?;

        let mut host_count_weights = vec![0.0f32; count_weights_len];
        for (variant_idx, score_cols) in prep.variant_to_scores_map.iter().enumerate() {
            let row_off = variant_idx * prep.stride();
            for score_idx in score_cols {
                host_count_weights[row_off + score_idx.0] = 1.0;
            }
        }
        let full_count_weights = stream
            .memcpy_stod(&host_count_weights)
            .map_err(|e| format!("Failed to upload full count-weight matrix: {e:?}"))?;

        let ptx = compile_ptx(CUDA_KERNELS).map_err(|e| format!("NVRTC compile failed: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("Failed to load CUDA module: {e:?}"))?;

        let unpack_kernel = module
            .load_function("unpack_plink")
            .map_err(|e| format!("Failed to load unpack_plink kernel: {e:?}"))?;
        let build_batch_mats_kernel = module
            .load_function("build_batch_mats")
            .map_err(|e| format!("Failed to load build_batch_mats kernel: {e:?}"))?;

        let max_packed = mega * prep.bytes_per_variant as usize;
        let pinned_staging = unsafe { ctx.alloc_pinned::<u8>(max_packed) }
            .map_err(|e| format!("Failed to allocate pinned host buffer: {e:?}"))?;

        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| format!("Failed to initialize cuBLAS: {e:?}"))?;

        Ok(Self {
            _ctx: ctx,
            stream,
            blas,
            unpack_kernel,
            build_batch_mats_kernel,
            full_weights,
            full_flips,
            full_count_weights,
            output_map,
            mega_batch_variants: mega,
            pinned_staging,
        })
    }
}

fn run_single_file_cuda(
    context: &PipelineContext,
    bed_path: &Path,
    mut runtime: CudaRuntime,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let prep_result = &context.prep_result;
    let bed_source = io::open_bed_source(bed_path)?;
    let shared_source = bed_source.byte_source();

    let (sparse_rx, sparse_tx, dense_rx, dense_tx, buffer_pool, variants_processed_count) =
        create_dense_only_pipeline_channels(context);
    let total_variants = prep_result.num_reconciled_variants as u64;
    let pb = create_progress_bar(total_variants, "Computing scores (CUDA)...");

    let sparse_drain_handle = {
        let pool = Arc::clone(&buffer_pool);
        thread::spawn(move || {
            while let Ok(message) = sparse_rx.recv() {
                match message {
                    Ok(mut work_item) => {
                        work_item.data.clear();
                        let _ = pool.push(work_item.data);
                    }
                    Err(_) => break,
                }
            }
        })
    };
    let (progress_done, progress_handle) = {
        let counter = Arc::clone(&variants_processed_count);
        let pb_clone = pb.clone();
        let done = Arc::new(AtomicBool::new(false));
        let done_for_thread = Arc::clone(&done);
        let join = thread::spawn(move || {
            while !done_for_thread.load(Ordering::Relaxed)
                && counter.load(Ordering::Relaxed) < total_variants
            {
                pb_clone.set_position(counter.load(Ordering::Relaxed));
                thread::sleep(Duration::from_millis(200));
            }
            pb_clone.set_position(counter.load(Ordering::Relaxed));
        });
        (done, join)
    };

    let has_complex = !prep_result.complex_rules.is_empty();
    let is_remote = bed_source.mmap().is_none();
    let should_spool = has_complex && is_remote;
    let mut spool_state: Option<SpoolState> = None;
    let mut spool_path: Option<PathBuf> = None;
    if should_spool {
        let (spool_dir, spool_stem) = derive_spool_destination(bed_path);
        fs::create_dir_all(&spool_dir).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool directory {}: {e}",
                spool_dir.display()
            ))
        })?;
        let filename = unique_spool_filename(&spool_stem);
        let path = spool_dir.join(&filename);
        let file = File::create(&path).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool file {}: {e}",
                path.display()
            ))
        })?;
        let complex_variant_count = prep_result
            .required_is_complex()
            .iter()
            .filter(|&&flag| flag != 0)
            .count() as u64;
        let offsets_capacity = usize::try_from(complex_variant_count)
            .unwrap_or(usize::MAX / 2)
            .max(1);
        spool_state = Some(SpoolState {
            writer: BufWriter::with_capacity(SPOOL_BUFFER_SIZE, file),
            offsets: AHashMap::with_capacity(offsets_capacity),
            cursor: 0,
        });
        spool_path = Some(path);
    }

    let mut local_spool = spool_state.take();
    let (mut final_scores, mut final_counts, mut spool_state) = thread::scope(|s| {
        let producer_handle = s.spawn({
            let source = Arc::clone(&shared_source);
            let prep = Arc::clone(&context.prep_result);
            let pool = Arc::clone(&buffer_pool);
            let counter = Arc::clone(&variants_processed_count);
            move || -> Result<Option<SpoolState>, PipelineError> {
                let spool_plan = if should_spool {
                    let state = local_spool
                        .as_mut()
                        .expect("spool state missing despite spooling enabled");
                    Some(create_spool_plan(prep.as_ref(), state)?)
                } else {
                    None
                };
                io::producer_thread(
                    Arc::clone(&source),
                    Arc::clone(&prep),
                    sparse_tx,
                    dense_tx,
                    pool,
                    counter,
                    |_| ComputePath::Pivot,
                    spool_plan,
                );
                Ok(local_spool)
            }
        });

        let compute_result = process_dense_stream_cuda(
            dense_rx,
            prep_result,
            &mut runtime,
            Arc::clone(&buffer_pool),
        );
        let spool_result = producer_handle
            .join()
            .map_err(|_| PipelineError::Producer("CUDA producer thread panicked".to_string()))?;
        let (scores, counts) = compute_result?;
        let spool = spool_result?;
        Ok::<_, PipelineError>((scores, counts, spool))
    })?;
    progress_done.store(true, Ordering::Relaxed);
    let _ = sparse_drain_handle.join();
    let _ = progress_handle.join();
    pb.finish_with_message("Computation complete.");

    if !prep_result.complex_rules.is_empty() {
        if should_spool {
            let spool_file_path = spool_path
                .clone()
                .expect("spool path missing despite spooling enabled");
            let (offsets, spool_bytes_per_variant) = {
                let mut state = spool_state
                    .take()
                    .expect("spool state missing despite spooling enabled");
                state.writer.flush().map_err(|e| {
                    PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
                })?;
                (state.offsets, prep_result.spool_bytes_per_variant())
            };
            let mmap = if spool_bytes_per_variant == 0 {
                let mut anon = MmapOptions::new().len(1).map_anon().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to allocate anonymous mapping for empty complex spool: {e}"
                    ))
                })?;
                anon.copy_from_slice(&[0u8]);
                anon.make_read_only().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to convert anonymous mapping to read-only: {e}"
                    ))
                })?
            } else {
                let spool_file = File::open(&spool_file_path).map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to open complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?;
                unsafe { Mmap::map(&spool_file) }.map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to memory-map complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?
            };
            let dense_map = Arc::new(prep_result.spool_dense_map().to_vec());
            let resolver = ComplexVariantResolver::from_spool(
                Arc::new(mmap),
                offsets,
                spool_bytes_per_variant,
                dense_map,
            );
            if let Err(e) = resolve_complex_variants(
                &resolver,
                prep_result,
                &mut final_scores,
                &mut final_counts,
            ) {
                let _ = fs::remove_file(&spool_file_path);
                return Err(e);
            }
            let keep_spool = env::var("GNOMON_KEEP_SPOOL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !keep_spool {
                let _ = fs::remove_file(&spool_file_path);
            }
        } else {
            let resolver = ComplexVariantResolver::from_single_source(bed_source.clone());
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

fn run_multi_file_cuda(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
    mut runtime: CudaRuntime,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let prep_result = &context.prep_result;
    let bed_sources: Vec<io::BedSource> = boundaries
        .iter()
        .map(|b| io::open_bed_source(&b.bed_path))
        .collect::<Result<_, _>>()?;
    let any_remote = bed_sources.iter().any(|s| s.mmap().is_none());
    let shared_sources = Arc::new(bed_sources);

    let (sparse_rx, sparse_tx, dense_rx, dense_tx, buffer_pool, variants_processed_count) =
        create_dense_only_pipeline_channels(context);
    let total_variants = prep_result.num_reconciled_variants as u64;
    let pb = create_progress_bar(total_variants, "Computing scores (CUDA)...");

    let sparse_drain_handle = {
        let pool = Arc::clone(&buffer_pool);
        thread::spawn(move || {
            while let Ok(message) = sparse_rx.recv() {
                match message {
                    Ok(mut work_item) => {
                        work_item.data.clear();
                        let _ = pool.push(work_item.data);
                    }
                    Err(_) => break,
                }
            }
        })
    };
    let (progress_done, progress_handle) = {
        let counter = Arc::clone(&variants_processed_count);
        let pb_clone = pb.clone();
        let done = Arc::new(AtomicBool::new(false));
        let done_for_thread = Arc::clone(&done);
        let join = thread::spawn(move || {
            while !done_for_thread.load(Ordering::Relaxed)
                && counter.load(Ordering::Relaxed) < total_variants
            {
                pb_clone.set_position(counter.load(Ordering::Relaxed));
                thread::sleep(Duration::from_millis(200));
            }
            pb_clone.set_position(counter.load(Ordering::Relaxed));
        });
        (done, join)
    };

    let has_complex = !prep_result.complex_rules.is_empty();
    let should_spool = has_complex && any_remote;
    let mut spool_state: Option<SpoolState> = None;
    let mut spool_path: Option<PathBuf> = None;
    if should_spool {
        let (spool_dir, spool_stem) = derive_spool_destination(&boundaries[0].bed_path);
        fs::create_dir_all(&spool_dir).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool directory {}: {e}",
                spool_dir.display()
            ))
        })?;
        let filename = unique_spool_filename(&spool_stem);
        let path = spool_dir.join(&filename);
        let file = File::create(&path).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool file {}: {e}",
                path.display()
            ))
        })?;
        let complex_variant_count = prep_result
            .required_is_complex()
            .iter()
            .filter(|&&flag| flag != 0)
            .count() as u64;
        let offsets_capacity = usize::try_from(complex_variant_count)
            .unwrap_or(usize::MAX / 2)
            .max(1);
        spool_state = Some(SpoolState {
            writer: BufWriter::with_capacity(SPOOL_BUFFER_SIZE, file),
            offsets: AHashMap::with_capacity(offsets_capacity),
            cursor: 0,
        });
        spool_path = Some(path);
    }

    let mut local_spool = spool_state.take();
    let (mut final_scores, mut final_counts, mut spool_state) = thread::scope(|s| {
        let producer_handle = s.spawn({
            let prep = Arc::clone(&context.prep_result);
            let pool = Arc::clone(&buffer_pool);
            let counter = Arc::clone(&variants_processed_count);
            let sources = Arc::clone(&shared_sources);
            let boundaries_owned = boundaries.to_vec();
            move || -> Result<Option<SpoolState>, PipelineError> {
                let spool_plan = if should_spool {
                    let state = local_spool
                        .as_mut()
                        .expect("spool state missing despite spooling enabled");
                    Some(create_spool_plan(prep.as_ref(), state)?)
                } else {
                    None
                };
                io::multi_file_producer_thread(
                    Arc::clone(&prep),
                    &boundaries_owned,
                    sources.as_ref(),
                    sparse_tx,
                    dense_tx,
                    pool,
                    counter,
                    |_| ComputePath::Pivot,
                    spool_plan,
                );
                Ok(local_spool)
            }
        });

        let compute_result = process_dense_stream_cuda(
            dense_rx,
            prep_result,
            &mut runtime,
            Arc::clone(&buffer_pool),
        );
        let spool_result = producer_handle
            .join()
            .map_err(|_| PipelineError::Producer("CUDA producer thread panicked".to_string()))?;
        let (scores, counts) = compute_result?;
        let spool = spool_result?;
        Ok::<_, PipelineError>((scores, counts, spool))
    })?;
    progress_done.store(true, Ordering::Relaxed);
    let _ = sparse_drain_handle.join();
    let _ = progress_handle.join();
    pb.finish_with_message("Computation complete.");

    if !prep_result.complex_rules.is_empty() {
        if should_spool {
            let spool_file_path = spool_path
                .clone()
                .expect("spool path missing despite spooling enabled");
            let (offsets, spool_bytes_per_variant) = {
                let mut state = spool_state
                    .take()
                    .expect("spool state missing despite spooling enabled");
                state.writer.flush().map_err(|e| {
                    PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
                })?;
                (state.offsets, prep_result.spool_bytes_per_variant())
            };
            let mmap = if spool_bytes_per_variant == 0 {
                let mut anon = MmapOptions::new().len(1).map_anon().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to allocate anonymous mapping for empty complex spool: {e}"
                    ))
                })?;
                anon.copy_from_slice(&[0u8]);
                anon.make_read_only().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to convert anonymous mapping to read-only: {e}"
                    ))
                })?
            } else {
                let spool_file = File::open(&spool_file_path).map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to open complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?;
                unsafe { Mmap::map(&spool_file) }.map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to memory-map complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?
            };
            let dense_map = Arc::new(prep_result.spool_dense_map().to_vec());
            let resolver = ComplexVariantResolver::from_spool(
                Arc::new(mmap),
                offsets,
                spool_bytes_per_variant,
                dense_map,
            );
            if let Err(e) = resolve_complex_variants(
                &resolver,
                prep_result,
                &mut final_scores,
                &mut final_counts,
            ) {
                let _ = fs::remove_file(&spool_file_path);
                return Err(e);
            }
            let keep_spool = env::var("GNOMON_KEEP_SPOOL")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            if !keep_spool {
                let _ = fs::remove_file(&spool_file_path);
            }
        } else {
            let resolver = ComplexVariantResolver::from_multi_sources(
                shared_sources.as_ref().clone(),
                boundaries.to_vec(),
            )?;
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

fn create_dense_only_pipeline_channels(
    context: &PipelineContext,
) -> (
    Receiver<Result<WorkItem, PipelineError>>,
    crossbeam_channel::Sender<Result<WorkItem, PipelineError>>,
    Receiver<Result<WorkItem, PipelineError>>,
    crossbeam_channel::Sender<Result<WorkItem, PipelineError>>,
    Arc<ArrayQueue<Vec<u8>>>,
    Arc<AtomicU64>,
) {
    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(8192);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(DENSE_CHANNEL_BOUND);

    let buffer_pool = Arc::new(ArrayQueue::new(BUFFER_POOL_SIZE));
    for _ in 0..BUFFER_POOL_SIZE {
        buffer_pool
            .push(Vec::with_capacity(
                context.prep_result.bytes_per_variant as usize,
            ))
            .unwrap();
    }

    let variants_processed_count = Arc::new(AtomicU64::new(0));

    (
        sparse_rx,
        sparse_tx,
        dense_rx,
        dense_tx,
        buffer_pool,
        variants_processed_count,
    )
}

fn process_dense_stream_cuda(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    prep_result: &PreparationResult,
    runtime: &mut CudaRuntime,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let num_people = prep_result.num_people_to_score;
    let num_scores = prep_result.score_names.len();
    let result_size = num_people * num_scores;
    let bytes_per_variant = prep_result.bytes_per_variant as usize;

    let baseline = compute_cpu_precise_baseline(prep_result);
    let mut final_scores = Vec::with_capacity(result_size);
    for _ in 0..num_people {
        final_scores.extend_from_slice(&baseline);
    }
    let mut final_counts = vec![0u32; result_size];

    let mega = runtime.mega_batch_variants;
    let max_packed = mega * bytes_per_variant;
    let mut d_packed = runtime
        .stream
        .alloc_zeros::<u8>(max_packed)
        .map_err(map_driver_err("Failed to allocate packed device buffer"))?;
    let mut d_dosage = runtime
        .stream
        .alloc_zeros::<f32>(num_people * mega)
        .map_err(map_driver_err("Failed to allocate dosage device buffer"))?;
    let mut d_missing = runtime
        .stream
        .alloc_zeros::<f32>(num_people * mega)
        .map_err(map_driver_err("Failed to allocate missing device buffer"))?;
    let mut d_w_eff = runtime
        .stream
        .alloc_zeros::<f32>(mega * num_scores)
        .map_err(map_driver_err(
            "Failed to allocate effective-weight device buffer",
        ))?;
    let mut d_w_corr = runtime
        .stream
        .alloc_zeros::<f32>(mega * num_scores)
        .map_err(map_driver_err(
            "Failed to allocate missing-correction weight buffer",
        ))?;
    let mut d_count_w = runtime
        .stream
        .alloc_zeros::<f32>(mega * num_scores)
        .map_err(map_driver_err("Failed to allocate count-weight buffer"))?;
    let mut d_reconciled_indices = runtime
        .stream
        .alloc_zeros::<u32>(mega)
        .map_err(map_driver_err("Failed to allocate reconciled-index buffer"))?;
    let mut d_out_scores = runtime
        .stream
        .alloc_zeros::<f32>(result_size)
        .map_err(map_driver_err("Failed to allocate score output buffer"))?;
    let mut d_out_corr = runtime
        .stream
        .alloc_zeros::<f32>(result_size)
        .map_err(map_driver_err(
            "Failed to allocate correction output buffer",
        ))?;
    let mut d_out_counts = runtime
        .stream
        .alloc_zeros::<f32>(result_size)
        .map_err(map_driver_err("Failed to allocate output count buffer"))?;

    let mut host_out_scores = vec![0f32; result_size];
    let mut host_out_corr = vec![0f32; result_size];
    let mut host_out_counts = vec![0f32; result_size];

    let mut batch: Vec<WorkItem> = Vec::with_capacity(mega);

    loop {
        batch.clear();
        match rx.recv() {
            Ok(Ok(first_item)) => batch.push(first_item),
            Ok(Err(e)) => return Err(e),
            Err(_) => break,
        }
        while batch.len() < mega {
            match rx.try_recv() {
                Ok(Ok(item)) => batch.push(item),
                Ok(Err(e)) => return Err(e),
                Err(_) => break,
            }
        }

        let batch_len = batch.len();
        if batch_len == 0 {
            continue;
        }

        let mut reconciled_indices = Vec::with_capacity(batch_len);
        let packed_len = batch_len * bytes_per_variant;
        let pinned_slice = runtime
            .pinned_staging
            .as_mut_slice()
            .map_err(map_driver_err("Failed to map pinned host slice"))?;
        let mut guards: Vec<BufferGuard<'_>> = Vec::with_capacity(batch_len);

        for (i, wi) in batch.drain(..).enumerate() {
            let start = i * bytes_per_variant;
            let end = start + bytes_per_variant;
            pinned_slice[start..end].copy_from_slice(&wi.data);
            reconciled_indices.push(wi.reconciled_variant_index);
            guards.push(BufferGuard {
                buffer: Some(wi.data),
                pool: &buffer_pool,
            });
        }

        {
            let mut d_packed_view: CudaViewMut<'_, u8> = d_packed.slice_mut(0..packed_len);
            runtime
                .stream
                .memcpy_htod(&pinned_slice[..packed_len], &mut d_packed_view)
                .map_err(map_driver_err("Failed to copy packed batch to device"))?;
        }

        let host_reconciled_indices: Vec<u32> = reconciled_indices.iter().map(|v| v.0).collect();
        {
            let mut d_reconciled_view = d_reconciled_indices.slice_mut(0..batch_len);
            runtime
                .stream
                .memcpy_htod(&host_reconciled_indices, &mut d_reconciled_view)
                .map_err(map_driver_err(
                    "Failed to upload reconciled variant indices",
                ))?;
        }

        let unpack_elems = num_people * batch_len;
        unsafe {
            runtime
                .stream
                .launch_builder(&runtime.unpack_kernel)
                .arg(&d_packed.slice(0..packed_len))
                .arg(&runtime.output_map)
                .arg(&(num_people as i32))
                .arg(&(batch_len as i32))
                .arg(&(bytes_per_variant as i32))
                .arg(&mut d_dosage.slice_mut(0..unpack_elems))
                .arg(&mut d_missing.slice_mut(0..unpack_elems))
                .launch(LaunchConfig::for_num_elems(unpack_elems as u32))
                .map_err(map_driver_err("Failed to launch unpack_plink kernel"))?;
        }

        let weights_elems = batch_len * num_scores;
        unsafe {
            runtime
                .stream
                .launch_builder(&runtime.build_batch_mats_kernel)
                .arg(&runtime.full_weights)
                .arg(&runtime.full_flips)
                .arg(&runtime.full_count_weights)
                .arg(&d_reconciled_indices.slice(0..batch_len))
                .arg(&(batch_len as i32))
                .arg(&(prep_result.stride() as i32))
                .arg(&(num_scores as i32))
                .arg(&mut d_w_eff.slice_mut(0..weights_elems))
                .arg(&mut d_w_corr.slice_mut(0..weights_elems))
                .arg(&mut d_count_w.slice_mut(0..weights_elems))
                .launch(LaunchConfig::for_num_elems(weights_elems as u32))
                .map_err(map_driver_err("Failed to launch build_batch_mats kernel"))?;
        }

        run_row_major_gemm(
            &runtime.blas,
            num_people,
            batch_len,
            num_scores,
            &d_dosage.slice(0..unpack_elems),
            &d_w_eff.slice(0..weights_elems),
            &mut d_out_scores,
            0.0f32,
        )?;

        run_row_major_gemm(
            &runtime.blas,
            num_people,
            batch_len,
            num_scores,
            &d_missing.slice(0..unpack_elems),
            &d_w_corr.slice(0..weights_elems),
            &mut d_out_corr,
            0.0f32,
        )?;

        run_row_major_gemm(
            &runtime.blas,
            num_people,
            batch_len,
            num_scores,
            &d_missing.slice(0..unpack_elems),
            &d_count_w.slice(0..weights_elems),
            &mut d_out_counts,
            0.0f32,
        )?;
        runtime
            .stream
            .memcpy_dtoh(&d_out_scores, &mut host_out_scores)
            .map_err(map_driver_err("Failed to copy score output from device"))?;
        runtime
            .stream
            .memcpy_dtoh(&d_out_corr, &mut host_out_corr)
            .map_err(map_driver_err(
                "Failed to copy correction output from device",
            ))?;
        runtime
            .stream
            .memcpy_dtoh(&d_out_counts, &mut host_out_counts)
            .map_err(map_driver_err("Failed to copy count output from device"))?;

        for i in 0..result_size {
            final_scores[i] += host_out_scores[i] as f64 + host_out_corr[i] as f64;
            final_counts[i] += host_out_counts[i].round() as u32;
        }

        drop(guards);
    }

    runtime
        .stream
        .synchronize()
        .map_err(map_driver_err("Failed to synchronize CUDA stream"))?;

    Ok((final_scores, final_counts))
}

fn run_row_major_gemm(
    blas: &CudaBlas,
    m_rows: usize,
    k_shared: usize,
    n_cols: usize,
    a: &CudaView<'_, f32>,
    b: &CudaView<'_, f32>,
    c: &mut CudaSlice<f32>,
    beta: f32,
) -> Result<(), PipelineError> {
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n_cols as i32,
        n: m_rows as i32,
        k: k_shared as i32,
        alpha: 1.0f32,
        lda: n_cols as i32,
        ldb: k_shared as i32,
        beta,
        ldc: n_cols as i32,
    };

    unsafe {
        blas.gemm(cfg, b, a, c)
            .map_err(|e| PipelineError::Compute(format!("cuBLAS GEMM failed: {e:?}")))
    }
}

fn compute_cpu_precise_baseline(prep_result: &PreparationResult) -> Vec<f64> {
    let num_scores = prep_result.score_names.len();
    let stride = prep_result.stride();
    (0..prep_result.num_reconciled_variants)
        .into_par_iter()
        .fold(
            || vec![0.0f64; num_scores],
            |mut local_baseline, i| {
                let flip_row_offset = i * stride;
                let flip_row =
                    &prep_result.flip_mask_matrix()[flip_row_offset..flip_row_offset + stride];
                let weight_row =
                    &prep_result.weights_matrix()[flip_row_offset..flip_row_offset + stride];
                for k in 0..num_scores {
                    if flip_row[k] == 1 {
                        local_baseline[k] += 2.0 * weight_row[k] as f64;
                    }
                }
                local_baseline
            },
        )
        .reduce(
            || vec![0.0f64; num_scores],
            |mut a, b| {
                for (v_a, v_b) in a.iter_mut().zip(b) {
                    *v_a += v_b;
                }
                a
            },
        )
}

fn derive_spool_destination(base_path: &Path) -> (PathBuf, String) {
    let stem = base_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "gnomon_results".to_string());
    let path_str = base_path.to_string_lossy();
    if path_str.starts_with("gs://")
        || path_str.starts_with("http://")
        || path_str.starts_with("https://")
    {
        (Path::new(".").to_path_buf(), stem)
    } else {
        let dir = match base_path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => Path::new(".").to_path_buf(),
        };
        (dir, stem)
    }
}

fn unique_spool_filename(stem: &str) -> String {
    use rand::{RngCore, thread_rng};
    let pid = process::id();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    let timestamp = now.as_nanos();
    let mut rng = thread_rng();
    let random_component: u32 = rng.next_u32();
    format!(
        "{}.{}.{}.{}.complex_spool.bin",
        stem, pid, timestamp, random_component
    )
}

fn create_spool_plan<'a>(
    prep_result: &'a PreparationResult,
    state: &'a mut SpoolState,
) -> Result<io::SpoolPlan<'a>, PipelineError> {
    let stride = prep_result.spool_bytes_per_variant();
    let stride_usize = usize::try_from(stride).map_err(|_| {
        PipelineError::Compute(format!(
            "spool stride of {} bytes does not fit on this platform",
            stride
        ))
    })?;
    Ok(io::SpoolPlan {
        is_complex_for_required: prep_result.required_is_complex(),
        compact_byte_index: prep_result.spool_compact_byte_index(),
        bytes_per_spooled_variant: stride,
        bytes_per_spooled_variant_usize: stride_usize,
        scratch: vec![0u8; stride_usize],
        file: &mut state.writer,
        offsets: &mut state.offsets,
        cursor: &mut state.cursor,
    })
}

fn map_driver_err(context: &'static str) -> impl FnOnce(DriverError) -> PipelineError {
    move |e| PipelineError::Compute(format!("{context}: {e:?}"))
}
