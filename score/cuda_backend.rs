use crate::score::complex::{ComplexVariantResolver, resolve_complex_variants};
use crate::score::decide::ComputePath;
use crate::score::io;
use crate::score::pipeline::{PipelineContext, PipelineError};
use crate::score::types::{
    BimRowIndex, FilesetBoundary, PipelineKind, PreparationResult, WorkItem,
};
use ahash::AHashMap;
use crossbeam_channel::{Receiver, Sender, bounded};
use crossbeam_queue::ArrayQueue;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut,
    DriverError, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, IsTerminal, Write};
use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
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
const PIPELINE_SLOTS: usize = 2;

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
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long total =
        (unsigned long long)num_people * (unsigned long long)batch_variants;
    if (idx >= total) return;

    unsigned long long person = idx / (unsigned long long)batch_variants;
    unsigned long long variant = idx % (unsigned long long)batch_variants;

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
    const unsigned char* count_mask,
    const unsigned int* reconciled_indices,
    int batch_variants,
    int stride,
    int num_scores,
    float* out_effective,
    float* out_missing_corr,
    float* out_count
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    unsigned long long total =
        (unsigned long long)batch_variants * (unsigned long long)num_scores;
    if (idx >= total) return;

    unsigned long long v = idx / (unsigned long long)num_scores;
    unsigned long long s = idx % (unsigned long long)num_scores;
    unsigned int reconciled = reconciled_indices[v];
    size_t src = (size_t)reconciled * (size_t)stride + (size_t)s;

    float w = weights[src];
    unsigned char f = flips[src];
    out_count[idx] = count_mask[src] ? 1.0f : 0.0f;
    if (f == 1u) {
        out_effective[idx] = -w;
        out_missing_corr[idx] = -2.0f * w;
    } else {
        out_effective[idx] = w;
        out_missing_corr[idx] = 0.0f;
    }
}

extern "C" __global__ void accumulate_outputs_f64(
    const float* out_scores,
    const float* out_corr,
    const float* out_counts,
    double* final_scores,
    double* final_counts,
    unsigned long long len
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= len) return;

    final_scores[idx] += (double)out_scores[idx] + (double)out_corr[idx];
    final_counts[idx] += (double)out_counts[idx];
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
    compute_stream: Arc<CudaStream>,
    copy_stream: Arc<CudaStream>,
    blas: CudaBlas,
    unpack_kernel: CudaFunction,
    build_batch_mats_kernel: CudaFunction,
    accumulate_outputs_kernel: CudaFunction,
    full_weights: CudaSlice<f32>,
    full_flips: CudaSlice<u8>,
    full_count_mask: CudaSlice<u8>,
    output_map: CudaSlice<u32>,
    mega_batch_variants: usize,
    pinned_staging: Vec<PinnedHostSlice<u8>>,
    pinned_reconciled: Vec<PinnedHostSlice<u32>>,
}

struct GpuChannels {
    dense_rx: Receiver<Result<WorkItem, PipelineError>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    variants_processed_count: Arc<AtomicU64>,
}

struct PipelineSupport {
    progress_done: Arc<AtomicBool>,
    progress_handle: Option<thread::JoinHandle<()>>,
    fallback_log_handle: Option<thread::JoinHandle<()>>,
    pb: ProgressBar,
}

struct PipelineSupportGuard {
    support: Option<PipelineSupport>,
    completed: bool,
}

impl PipelineSupportGuard {
    fn new(support: PipelineSupport) -> Self {
        Self {
            support: Some(support),
            completed: false,
        }
    }

    fn mark_completed(&mut self) {
        self.completed = true;
    }
}

impl Drop for PipelineSupportGuard {
    fn drop(&mut self) {
        if let Some(support) = self.support.take() {
            finish_pipeline_support(support, self.completed);
        }
    }
}

#[derive(Copy, Clone)]
struct CudaDims {
    num_people: usize,
    num_scores: usize,
    bytes_per_variant: usize,
    result_size: usize,
    stride: usize,
}

impl CudaDims {
    fn from_prep(prep_result: &PreparationResult) -> Result<Self, PipelineError> {
        let num_people = prep_result.num_people_to_score;
        let num_scores = prep_result.score_names.len();
        let result_size = checked_mul_usize("result_size", num_people, num_scores)?;
        Ok(Self {
            num_people,
            num_scores,
            bytes_per_variant: prep_result.bytes_per_variant as usize,
            result_size,
            stride: prep_result.stride(),
        })
    }

    #[inline]
    fn num_people_i32(self) -> Result<i32, PipelineError> {
        checked_i32("num_people", self.num_people)
    }

    #[inline]
    fn num_scores_i32(self) -> Result<i32, PipelineError> {
        checked_i32("num_scores", self.num_scores)
    }

    #[inline]
    fn bytes_per_variant_i32(self) -> Result<i32, PipelineError> {
        checked_i32("bytes_per_variant", self.bytes_per_variant)
    }

    #[inline]
    fn stride_i32(self) -> Result<i32, PipelineError> {
        checked_i32("stride", self.stride)
    }
}

#[derive(Copy, Clone)]
struct BatchShape {
    dims: CudaDims,
    batch_variants: NonZeroUsize,
}

struct PendingBatch {
    slot: usize,
    shape: BatchShape,
    packed_len: usize,
    copy_done_event: CudaEvent,
}

impl BatchShape {
    #[inline]
    fn from_counts(dims: CudaDims, batch_len: usize) -> Result<Self, PipelineError> {
        let batch_variants = NonZeroUsize::new(batch_len).ok_or_else(|| {
            PipelineError::Compute("Encountered empty CUDA batch during compute".to_string())
        })?;
        Ok(Self {
            dims,
            batch_variants,
        })
    }

    #[inline]
    fn batch_len(self) -> usize {
        self.batch_variants.get()
    }

    #[inline]
    fn unpack_elems(self) -> Result<usize, PipelineError> {
        checked_mul_usize("unpack_elems", self.dims.num_people, self.batch_len())
    }

    #[inline]
    fn weight_elems(self) -> Result<usize, PipelineError> {
        checked_mul_usize("weight_elems", self.batch_len(), self.dims.num_scores)
    }
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

    let runtime = match init_cuda_runtime_safely(prep) {
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

fn init_cuda_runtime_safely(prep: &PreparationResult) -> Result<CudaRuntime, String> {
    match catch_unwind(AssertUnwindSafe(|| CudaRuntime::new(prep))) {
        Ok(runtime) => runtime,
        Err(payload) => Err(format!(
            "CUDA runtime initialization panicked ({})",
            panic_payload_to_string(payload)
        )),
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

impl CudaRuntime {
    fn new(prep: &PreparationResult) -> Result<Self, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("CUDA init failed: {e:?}"))?;
        ctx.bind_to_thread()
            .map_err(|e| format!("Failed to bind CUDA context: {e:?}"))?;
        let copy_stream = ctx
            .new_stream()
            .map_err(|e| format!("Failed to create CUDA copy stream: {e:?}"))?;
        let compute_stream = ctx
            .new_stream()
            .map_err(|e| format!("Failed to create CUDA compute stream: {e:?}"))?;

        let (free_mem, _) = cudarc::driver::result::mem_get_info()
            .map_err(|e| format!("Failed to query device memory: {e:?}"))?;

        let count_weights_len = prep
            .num_reconciled_variants
            .checked_mul(prep.stride())
            .ok_or_else(|| "CUDA memory estimate overflow: count_weights_len".to_string())?;
        let static_bytes = prep
            .weights_matrix()
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|v| {
                v.checked_add(
                    prep.flip_mask_matrix()
                        .len()
                        .checked_mul(std::mem::size_of::<u8>())?,
                )
            })
            .and_then(|v| v.checked_add(count_weights_len.checked_mul(std::mem::size_of::<u8>())?))
            .and_then(|v| {
                v.checked_add(
                    prep.output_idx_to_fam_idx
                        .len()
                        .checked_mul(std::mem::size_of::<u32>())?,
                )
            })
            .ok_or_else(|| "CUDA memory estimate overflow: static device allocations".to_string())?;

        let num_people = prep.num_people_to_score;
        let num_scores = prep.score_names.len();
        let result_size = num_people
            .checked_mul(num_scores)
            .ok_or_else(|| "CUDA memory estimate overflow: result_size".to_string())?;
        let bytes_per_variant = prep.bytes_per_variant as usize;

        let estimate_required_bytes = |mega: usize| -> Option<usize> {
            let slot_packed = PIPELINE_SLOTS.checked_mul(mega.checked_mul(bytes_per_variant)?)?;
            let slot_reconciled =
                PIPELINE_SLOTS.checked_mul(mega.checked_mul(std::mem::size_of::<u32>())?)?;
            let slot_outputs = PIPELINE_SLOTS
                .checked_mul(3)?
                .checked_mul(result_size.checked_mul(std::mem::size_of::<f32>())?)?;

            let d_dosage = num_people
                .checked_mul(mega)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_missing = num_people
                .checked_mul(mega)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_w_eff = mega
                .checked_mul(num_scores)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_w_corr = mega
                .checked_mul(num_scores)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_count_w = mega
                .checked_mul(num_scores)?
                .checked_mul(std::mem::size_of::<f32>())?;

            let d_final_scores = result_size.checked_mul(std::mem::size_of::<f64>())?;
            let d_final_counts = result_size.checked_mul(std::mem::size_of::<f64>())?;

            static_bytes
                .checked_add(slot_packed)?
                .checked_add(slot_reconciled)?
                .checked_add(slot_outputs)?
                .checked_add(d_dosage)?
                .checked_add(d_missing)?
                .checked_add(d_w_eff)?
                .checked_add(d_w_corr)?
                .checked_add(d_count_w)?
                .checked_add(d_final_scores)?
                .checked_add(d_final_counts)
        };

        let mut mega = MAX_MEGA_BATCH_VARIANTS;
        let budget = (free_mem as f64 * 0.8) as usize;
        while mega > MIN_MEGA_BATCH_VARIANTS {
            let required = estimate_required_bytes(mega).ok_or_else(|| {
                format!("CUDA memory estimate overflow while evaluating mega-batch={mega}")
            })?;
            if required <= budget {
                break;
            }
            mega /= 2;
        }

        if mega < MIN_MEGA_BATCH_VARIANTS {
            return Err("Insufficient GPU memory for minimum CUDA mega-batch".to_string());
        }

        let full_weights = compute_stream
            .clone_htod(prep.weights_matrix())
            .map_err(|e| format!("Failed to upload full weights matrix: {e:?}"))?;
        let full_flips = compute_stream
            .clone_htod(prep.flip_mask_matrix())
            .map_err(|e| format!("Failed to upload full flip mask matrix: {e:?}"))?;
        let output_map = compute_stream
            .clone_htod(&prep.output_idx_to_fam_idx)
            .map_err(|e| format!("Failed to upload output index map: {e:?}"))?;

        let mut host_count_mask = vec![0u8; count_weights_len];
        for (variant_idx, score_cols) in prep.variant_to_scores_map.iter().enumerate() {
            let row_off = variant_idx * prep.stride();
            for score_idx in score_cols {
                host_count_mask[row_off + score_idx.0] = 1;
            }
        }
        let full_count_mask = compute_stream
            .clone_htod(&host_count_mask)
            .map_err(|e| format!("Failed to upload full count-mask matrix: {e:?}"))?;

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
        let accumulate_outputs_kernel = module
            .load_function("accumulate_outputs_f64")
            .map_err(|e| format!("Failed to load accumulate_outputs_f64 kernel: {e:?}"))?;

        let max_packed = mega * prep.bytes_per_variant as usize;
        let mut pinned_staging = Vec::with_capacity(PIPELINE_SLOTS);
        let mut pinned_reconciled = Vec::with_capacity(PIPELINE_SLOTS);
        for _ in 0..PIPELINE_SLOTS {
            let pinned = unsafe { ctx.alloc_pinned::<u8>(max_packed) }
                .map_err(|e| format!("Failed to allocate pinned host buffer: {e:?}"))?;
            pinned_staging.push(pinned);
            let pinned_indices = unsafe { ctx.alloc_pinned::<u32>(mega) }
                .map_err(|e| format!("Failed to allocate pinned reconciled-index buffer: {e:?}"))?;
            pinned_reconciled.push(pinned_indices);
        }

        let blas = CudaBlas::new(compute_stream.clone())
            .map_err(|e| format!("Failed to initialize cuBLAS: {e:?}"))?;

        Ok(Self {
            _ctx: ctx,
            compute_stream,
            copy_stream,
            blas,
            unpack_kernel,
            build_batch_mats_kernel,
            accumulate_outputs_kernel,
            full_weights,
            full_flips,
            full_count_mask,
            output_map,
            mega_batch_variants: mega,
            pinned_staging,
            pinned_reconciled,
        })
    }
}

fn run_single_file_cuda(
    context: &PipelineContext,
    bed_path: &Path,
    runtime: CudaRuntime,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let bed_source = io::open_bed_source(bed_path)?;
    let input = CudaInput::Single {
        bed_path: bed_path.to_path_buf(),
        bed_source,
    };
    run_cuda_pipeline(context, runtime, input)
}

fn run_multi_file_cuda(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
    runtime: CudaRuntime,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let bed_sources: Vec<io::BedSource> = boundaries
        .iter()
        .map(|b| io::open_bed_source(&b.bed_path))
        .collect::<Result<_, _>>()?;
    let input = CudaInput::Multi {
        boundaries: boundaries.to_vec(),
        bed_sources: Arc::new(bed_sources),
    };
    run_cuda_pipeline(context, runtime, input)
}

enum CudaInput {
    Single {
        bed_path: PathBuf,
        bed_source: io::BedSource,
    },
    Multi {
        boundaries: Vec<FilesetBoundary>,
        bed_sources: Arc<Vec<io::BedSource>>,
    },
}

fn run_cuda_pipeline(
    context: &PipelineContext,
    mut runtime: CudaRuntime,
    input: CudaInput,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let prep_result = &context.prep_result;
    let channels = create_gpu_channels(context);
    let mut support_guard = PipelineSupportGuard::new(start_pipeline_support(
        prep_result.num_reconciled_variants as u64,
        Arc::clone(&channels.variants_processed_count),
        Arc::clone(&channels.buffer_pool),
    ));

    let has_complex = !prep_result.complex_rules.is_empty();
    let should_spool = has_complex
        && match &input {
            CudaInput::Single { bed_source, .. } => bed_source.mmap().is_none(),
            CudaInput::Multi { bed_sources, .. } => bed_sources.iter().any(|s| s.mmap().is_none()),
        };

    let mut spool_state: Option<SpoolState> = None;
    let mut spool_path: Option<PathBuf> = None;
    if should_spool {
        let base_path = match &input {
            CudaInput::Single { bed_path, .. } => bed_path.as_path(),
            CudaInput::Multi { boundaries, .. } => boundaries[0].bed_path.as_path(),
        };
        let (spool_dir, spool_stem) = derive_spool_destination(base_path);
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
    let producer_input = match &input {
        CudaInput::Single { bed_source, .. } => CudaProducerInput::Single {
            source: bed_source.byte_source(),
        },
        CudaInput::Multi {
            boundaries,
            bed_sources,
        } => CudaProducerInput::Multi {
            boundaries: boundaries.clone(),
            bed_sources: Arc::clone(bed_sources),
        },
    };

    let (mut final_scores, mut final_counts) = thread::scope(|s| {
        let local_spool_ref = &mut local_spool;
        let producer_handle = s.spawn({
            let prep = Arc::clone(&context.prep_result);
            let pool = Arc::clone(&channels.buffer_pool);
            let counter = Arc::clone(&channels.variants_processed_count);
            let dense_tx = channels.dense_tx;
            move || -> Result<(), PipelineError> {
                let spool_plan = if should_spool {
                    let state = local_spool_ref
                        .as_mut()
                        .expect("spool state missing despite spooling enabled");
                    Some(create_spool_plan(prep.as_ref(), state)?)
                } else {
                    None
                };
                match producer_input {
                    CudaProducerInput::Single { source } => {
                        io::producer_thread(
                            source,
                            Arc::clone(&prep),
                            None,
                            dense_tx,
                            pool,
                            counter,
                            |_| ComputePath::Pivot,
                            spool_plan,
                        );
                    }
                    CudaProducerInput::Multi {
                        boundaries,
                        bed_sources,
                    } => {
                        io::multi_file_producer_thread(
                            Arc::clone(&prep),
                            &boundaries,
                            bed_sources.as_ref(),
                            None,
                            dense_tx,
                            pool,
                            counter,
                            |_| ComputePath::Pivot,
                            spool_plan,
                        );
                    }
                }
                Ok(())
            }
        });

        let compute_result = process_dense_stream_cuda(
            channels.dense_rx,
            prep_result,
            &mut runtime,
            Arc::clone(&channels.buffer_pool),
        );
        let spool_result = producer_handle
            .join()
            .map_err(|_| PipelineError::Producer("CUDA producer thread panicked".to_string()))?;
        let (scores, counts) = compute_result?;
        spool_result?;
        Ok::<_, PipelineError>((scores, counts))
    })?;
    let mut spool_state = local_spool;
    support_guard.mark_completed();

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
            let resolver = match &input {
                CudaInput::Single { bed_source, .. } => {
                    ComplexVariantResolver::from_single_source(bed_source.clone())
                }
                CudaInput::Multi {
                    boundaries,
                    bed_sources,
                } => ComplexVariantResolver::from_multi_sources(
                    bed_sources.as_ref().clone(),
                    boundaries.clone(),
                )?,
            };
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

enum CudaProducerInput {
    Single {
        source: Arc<dyn io::ByteRangeSource>,
    },
    Multi {
        boundaries: Vec<FilesetBoundary>,
        bed_sources: Arc<Vec<io::BedSource>>,
    },
}

fn create_gpu_channels(context: &PipelineContext) -> GpuChannels {
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

    GpuChannels {
        dense_rx,
        dense_tx,
        buffer_pool,
        variants_processed_count,
    }
}

fn start_pipeline_support(
    total_variants: u64,
    variants_processed_count: Arc<AtomicU64>,
    _buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> PipelineSupport {
    let pb = create_progress_bar(total_variants, "Computing scores (CUDA)...");
    let use_fallback_logs = pb.is_hidden();
    let progress_done = Arc::new(AtomicBool::new(false));
    let progress_handle = if use_fallback_logs {
        None
    } else {
        let done_for_thread = Arc::clone(&progress_done);
        let pb_clone = pb.clone();
        let counter_for_pb = Arc::clone(&variants_processed_count);
        Some(thread::spawn(move || {
            while !done_for_thread.load(Ordering::Relaxed)
                && counter_for_pb.load(Ordering::Relaxed) < total_variants
            {
                pb_clone.set_position(counter_for_pb.load(Ordering::Relaxed));
                thread::sleep(Duration::from_millis(200));
            }
            pb_clone.set_position(counter_for_pb.load(Ordering::Relaxed));
        }))
    };

    let fallback_log_handle = if use_fallback_logs {
        let done_for_logs = Arc::clone(&progress_done);
        let counter_for_logs = Arc::clone(&variants_processed_count);
        Some(thread::spawn(move || {
            let mut last_reported: Option<u64> = None;
            while !done_for_logs.load(Ordering::Relaxed) {
                let processed = counter_for_logs.load(Ordering::Relaxed);
                if last_reported != Some(processed) {
                    let pct = if total_variants == 0 {
                        100.0
                    } else {
                        (processed as f64 * 100.0) / (total_variants as f64)
                    };
                    eprintln!(
                        "> CUDA progress: {}/{} ({pct:.1}%)",
                        processed, total_variants
                    );
                    last_reported = Some(processed);
                }
                thread::sleep(Duration::from_secs(2));
            }
            let processed = counter_for_logs.load(Ordering::Relaxed);
            let pct = if total_variants == 0 {
                100.0
            } else {
                (processed as f64 * 100.0) / (total_variants as f64)
            };
            eprintln!(
                "> CUDA progress: {}/{} ({pct:.1}%)",
                processed, total_variants
            );
        }))
    } else {
        None
    };

    PipelineSupport {
        progress_done,
        progress_handle,
        fallback_log_handle,
        pb,
    }
}

fn finish_pipeline_support(support: PipelineSupport, completed: bool) {
    support.progress_done.store(true, Ordering::Relaxed);
    if let Some(handle) = support.fallback_log_handle {
        let _ = handle.join();
    }
    if let Some(handle) = support.progress_handle {
        let _ = handle.join();
    }
    let message = if completed {
        "Computation complete."
    } else {
        "Computation aborted."
    };
    support.pb.finish_with_message(message);
}

fn process_dense_stream_cuda(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    prep_result: &PreparationResult,
    runtime: &mut CudaRuntime,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let dims = CudaDims::from_prep(prep_result)?;

    let baseline = compute_cpu_precise_baseline(prep_result);
    let mut final_scores = Vec::with_capacity(dims.result_size);
    for _ in 0..dims.num_people {
        final_scores.extend_from_slice(&baseline);
    }
    let mut final_counts = vec![0u32; dims.result_size];

    let mega = runtime.mega_batch_variants;
    let max_packed = checked_mul_usize("max_packed", mega, dims.bytes_per_variant)?;
    let mut d_packed_slots: Vec<CudaSlice<u8>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .copy_stream
                .alloc_zeros::<u8>(max_packed)
                .map_err(map_driver_err("Failed to allocate packed device buffer"))
        })
        .collect::<Result<_, _>>()?;
    let mut d_reconciled_slots: Vec<CudaSlice<u32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .copy_stream
                .alloc_zeros::<u32>(mega)
                .map_err(map_driver_err("Failed to allocate reconciled-index buffer"))
        })
        .collect::<Result<_, _>>()?;
    let mut d_dosage = runtime
        .compute_stream
        .alloc_zeros::<f32>(checked_mul_usize(
            "dosage buffer len",
            dims.num_people,
            mega,
        )?)
        .map_err(map_driver_err("Failed to allocate dosage device buffer"))?;
    let mut d_missing = runtime
        .compute_stream
        .alloc_zeros::<f32>(checked_mul_usize(
            "missing buffer len",
            dims.num_people,
            mega,
        )?)
        .map_err(map_driver_err("Failed to allocate missing device buffer"))?;
    let mut d_w_eff = runtime
        .compute_stream
        .alloc_zeros::<f32>(checked_mul_usize(
            "effective-weight buffer len",
            mega,
            dims.num_scores,
        )?)
        .map_err(map_driver_err(
            "Failed to allocate effective-weight device buffer",
        ))?;
    let mut d_w_corr = runtime
        .compute_stream
        .alloc_zeros::<f32>(checked_mul_usize(
            "missing-correction buffer len",
            mega,
            dims.num_scores,
        )?)
        .map_err(map_driver_err(
            "Failed to allocate missing-correction weight buffer",
        ))?;
    let mut d_count_w = runtime
        .compute_stream
        .alloc_zeros::<f32>(checked_mul_usize(
            "count-weight buffer len",
            mega,
            dims.num_scores,
        )?)
        .map_err(map_driver_err("Failed to allocate count-weight buffer"))?;
    let mut d_out_scores_slots: Vec<CudaSlice<f32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<f32>(dims.result_size)
                .map_err(map_driver_err("Failed to allocate score output buffer"))
        })
        .collect::<Result<_, _>>()?;
    let mut d_out_corr_slots: Vec<CudaSlice<f32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<f32>(dims.result_size)
                .map_err(map_driver_err(
                    "Failed to allocate correction output buffer",
                ))
        })
        .collect::<Result<_, _>>()?;
    let mut d_out_counts_slots: Vec<CudaSlice<f32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<f32>(dims.result_size)
                .map_err(map_driver_err("Failed to allocate output count buffer"))
        })
        .collect::<Result<_, _>>()?;

    let mut d_final_scores = runtime
        .compute_stream
        .clone_htod(&final_scores)
        .map_err(map_driver_err("Failed to upload baseline scores to device"))?;
    let mut d_final_counts = runtime
        .compute_stream
        .alloc_zeros::<f64>(dims.result_size)
        .map_err(map_driver_err(
            "Failed to allocate final count accumulator on device",
        ))?;

    let mut batch: Vec<WorkItem> = Vec::with_capacity(mega);
    let mut pending: Option<PendingBatch> = None;
    let mut slot_last_compute_done: [Option<CudaEvent>; PIPELINE_SLOTS] = [None, None];
    let mut slot_last_copy_done: [Option<CudaEvent>; PIPELINE_SLOTS] = [None, None];
    let mut batch_counter: usize = 0;

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
        let shape = BatchShape::from_counts(dims, batch_len)?;

        let slot = batch_counter % PIPELINE_SLOTS;
        batch_counter += 1;

        if let Some(event) = &slot_last_copy_done[slot] {
            event
                .synchronize()
                .map_err(map_driver_err("Failed waiting for prior copy event"))?;
        }
        if let Some(event) = &slot_last_compute_done[slot] {
            runtime.copy_stream.wait(event).map_err(map_driver_err(
                "Copy stream failed waiting for compute event",
            ))?;
        }

        let packed_len =
            checked_mul_usize("packed_len", shape.batch_len(), dims.bytes_per_variant)?;
        let pinned_slice = runtime.pinned_staging[slot]
            .as_mut_slice()
            .map_err(map_driver_err("Failed to map pinned host slice"))?;
        let pinned_reconciled =
            runtime.pinned_reconciled[slot]
                .as_mut_slice()
                .map_err(map_driver_err(
                    "Failed to map pinned reconciled-index host slice",
                ))?;
        let mut guards: Vec<BufferGuard<'_>> = Vec::with_capacity(shape.batch_len());
        for (i, wi) in batch.drain(..).enumerate() {
            let start = checked_mul_usize("packed copy start", i, dims.bytes_per_variant)?;
            let end = start + dims.bytes_per_variant;
            pinned_slice[start..end].copy_from_slice(&wi.data);
            pinned_reconciled[i] = wi.reconciled_variant_index.0;
            guards.push(BufferGuard {
                buffer: Some(wi.data),
                pool: &buffer_pool,
            });
        }

        {
            let mut d_packed_view: CudaViewMut<'_, u8> =
                d_packed_slots[slot].slice_mut(0..packed_len);
            runtime
                .copy_stream
                .memcpy_htod(&pinned_slice[..packed_len], &mut d_packed_view)
                .map_err(map_driver_err("Failed to copy packed batch to device"))?;
        }
        {
            let mut d_reconciled_view = d_reconciled_slots[slot].slice_mut(0..shape.batch_len());
            runtime
                .copy_stream
                .memcpy_htod(
                    &pinned_reconciled[..shape.batch_len()],
                    &mut d_reconciled_view,
                )
                .map_err(map_driver_err(
                    "Failed to upload reconciled variant indices",
                ))?;
        }
        let copy_done_event = runtime
            .copy_stream
            .record_event(None)
            .map_err(map_driver_err("Failed to record copy completion event"))?;
        slot_last_copy_done[slot] = Some(runtime.copy_stream.record_event(None).map_err(
            map_driver_err("Failed to record slot copy completion event"),
        )?);
        drop(guards);

        let current = PendingBatch {
            slot,
            shape,
            packed_len,
            copy_done_event,
        };

        if let Some(prev) = pending.take() {
            let slot = prev.slot;
            slot_last_compute_done[slot] = Some(run_pending_compute_cuda(
                runtime,
                dims,
                prev,
                &d_packed_slots,
                &d_reconciled_slots,
                &mut d_dosage,
                &mut d_missing,
                &mut d_w_eff,
                &mut d_w_corr,
                &mut d_count_w,
                &mut d_out_scores_slots,
                &mut d_out_corr_slots,
                &mut d_out_counts_slots,
                &mut d_final_scores,
                &mut d_final_counts,
            )?);
        }
        pending = Some(current);
    }

    if let Some(last) = pending.take() {
        let slot = last.slot;
        slot_last_compute_done[slot] = Some(run_pending_compute_cuda(
            runtime,
            dims,
            last,
            &d_packed_slots,
            &d_reconciled_slots,
            &mut d_dosage,
            &mut d_missing,
            &mut d_w_eff,
            &mut d_w_corr,
            &mut d_count_w,
            &mut d_out_scores_slots,
            &mut d_out_corr_slots,
            &mut d_out_counts_slots,
            &mut d_final_scores,
            &mut d_final_counts,
        )?);
    }

    runtime
        .copy_stream
        .synchronize()
        .map_err(map_driver_err("Failed to synchronize CUDA copy stream"))?;
    runtime
        .compute_stream
        .synchronize()
        .map_err(map_driver_err("Failed to synchronize CUDA compute stream"))?;

    let mut final_counts_f64 = vec![0f64; dims.result_size];
    runtime
        .compute_stream
        .memcpy_dtoh(&d_final_scores, &mut final_scores)
        .map_err(map_driver_err(
            "Failed to copy final score accumulator from device",
        ))?;
    runtime
        .compute_stream
        .memcpy_dtoh(&d_final_counts, &mut final_counts_f64)
        .map_err(map_driver_err(
            "Failed to copy final count accumulator from device",
        ))?;
    for (dst, src) in final_counts.iter_mut().zip(final_counts_f64.into_iter()) {
        *dst = src.round() as u32;
    }

    Ok((final_scores, final_counts))
}

#[allow(clippy::too_many_arguments)]
fn run_pending_compute_cuda(
    runtime: &mut CudaRuntime,
    dims: CudaDims,
    work: PendingBatch,
    d_packed_slots: &[CudaSlice<u8>],
    d_reconciled_slots: &[CudaSlice<u32>],
    d_dosage: &mut CudaSlice<f32>,
    d_missing: &mut CudaSlice<f32>,
    d_w_eff: &mut CudaSlice<f32>,
    d_w_corr: &mut CudaSlice<f32>,
    d_count_w: &mut CudaSlice<f32>,
    d_out_scores_slots: &mut [CudaSlice<f32>],
    d_out_corr_slots: &mut [CudaSlice<f32>],
    d_out_counts_slots: &mut [CudaSlice<f32>],
    d_final_scores: &mut CudaSlice<f64>,
    d_final_counts: &mut CudaSlice<f64>,
) -> Result<CudaEvent, PipelineError> {
    runtime
        .compute_stream
        .wait(&work.copy_done_event)
        .map_err(map_driver_err(
            "Compute stream failed waiting for copy event",
        ))?;

    let unpack_elems = work.shape.unpack_elems()?;
    let num_people_i32 = work.shape.dims.num_people_i32()?;
    let batch_len_i32 = checked_i32("batch_len", work.shape.batch_len())?;
    let bytes_per_variant_i32 = work.shape.dims.bytes_per_variant_i32()?;
    let unpack_launch_elems = checked_u32("unpack kernel launch elements", unpack_elems)?;
    unsafe {
        runtime
            .compute_stream
            .launch_builder(&runtime.unpack_kernel)
            .arg(&d_packed_slots[work.slot].slice(0..work.packed_len))
            .arg(&runtime.output_map)
            .arg(&num_people_i32)
            .arg(&batch_len_i32)
            .arg(&bytes_per_variant_i32)
            .arg(&mut d_dosage.slice_mut(0..unpack_elems))
            .arg(&mut d_missing.slice_mut(0..unpack_elems))
            .launch(LaunchConfig::for_num_elems(unpack_launch_elems))
            .map_err(map_driver_err("Failed to launch unpack_plink kernel"))?;
    }

    let weights_elems = work.shape.weight_elems()?;
    let stride_i32 = work.shape.dims.stride_i32()?;
    let num_scores_i32 = work.shape.dims.num_scores_i32()?;
    let weights_launch_elems =
        checked_u32("build_batch_mats kernel launch elements", weights_elems)?;
    unsafe {
        runtime
            .compute_stream
            .launch_builder(&runtime.build_batch_mats_kernel)
            .arg(&runtime.full_weights)
            .arg(&runtime.full_flips)
            .arg(&runtime.full_count_mask)
            .arg(&d_reconciled_slots[work.slot].slice(0..work.shape.batch_len()))
            .arg(&batch_len_i32)
            .arg(&stride_i32)
            .arg(&num_scores_i32)
            .arg(&mut d_w_eff.slice_mut(0..weights_elems))
            .arg(&mut d_w_corr.slice_mut(0..weights_elems))
            .arg(&mut d_count_w.slice_mut(0..weights_elems))
            .launch(LaunchConfig::for_num_elems(weights_launch_elems))
            .map_err(map_driver_err("Failed to launch build_batch_mats kernel"))?;
    }

    run_row_major_gemm(
        &runtime.blas,
        dims.num_people,
        work.shape.batch_len(),
        dims.num_scores,
        &d_dosage.slice(0..unpack_elems),
        &d_w_eff.slice(0..weights_elems),
        &mut d_out_scores_slots[work.slot],
        0.0f32,
    )?;
    run_row_major_gemm(
        &runtime.blas,
        dims.num_people,
        work.shape.batch_len(),
        dims.num_scores,
        &d_missing.slice(0..unpack_elems),
        &d_w_corr.slice(0..weights_elems),
        &mut d_out_corr_slots[work.slot],
        0.0f32,
    )?;
    run_row_major_gemm(
        &runtime.blas,
        dims.num_people,
        work.shape.batch_len(),
        dims.num_scores,
        &d_missing.slice(0..unpack_elems),
        &d_count_w.slice(0..weights_elems),
        &mut d_out_counts_slots[work.slot],
        0.0f32,
    )?;

    let result_len_u64 = u64::try_from(dims.result_size).map_err(|_| {
        PipelineError::Compute(format!(
            "result_size={} exceeds u64::MAX for accumulation kernel",
            dims.result_size
        ))
    })?;
    let result_len_u32 = checked_u32(
        "accumulate_outputs_f64 kernel launch elements",
        dims.result_size,
    )?;
    unsafe {
        runtime
            .compute_stream
            .launch_builder(&runtime.accumulate_outputs_kernel)
            .arg(&d_out_scores_slots[work.slot])
            .arg(&d_out_corr_slots[work.slot])
            .arg(&d_out_counts_slots[work.slot])
            .arg(d_final_scores)
            .arg(d_final_counts)
            .arg(&result_len_u64)
            .launch(LaunchConfig::for_num_elems(result_len_u32))
            .map_err(map_driver_err(
                "Failed to launch accumulate_outputs_f64 kernel",
            ))?;
    }

    runtime
        .compute_stream
        .record_event(None)
        .map_err(map_driver_err("Failed to record compute completion event"))
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
    let m_i32 = checked_i32("gemm m (n_cols)", n_cols)?;
    let n_i32 = checked_i32("gemm n (m_rows)", m_rows)?;
    let k_i32 = checked_i32("gemm k (k_shared)", k_shared)?;
    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: m_i32,
        n: n_i32,
        k: k_i32,
        alpha: 1.0f32,
        lda: m_i32,
        ldb: k_i32,
        beta,
        ldc: m_i32,
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
    let pid = process::id();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    let timestamp = now.as_nanos();
    let random_component: u32 = rand::random();
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

fn checked_i32(label: &str, value: usize) -> Result<i32, PipelineError> {
    i32::try_from(value).map_err(|_| {
        PipelineError::Compute(format!(
            "{label}={value} exceeds i32::MAX required by CUDA/cuBLAS APIs"
        ))
    })
}

fn checked_u32(label: &str, value: usize) -> Result<u32, PipelineError> {
    u32::try_from(value).map_err(|_| {
        PipelineError::Compute(format!(
            "{label}={value} exceeds u32::MAX required by CUDA launch config"
        ))
    })
}

fn checked_mul_usize(label: &str, a: usize, b: usize) -> Result<usize, PipelineError> {
    a.checked_mul(b).ok_or_else(|| {
        PipelineError::Compute(format!("{label} overflow: {a} * {b} exceeds usize::MAX"))
    })
}
