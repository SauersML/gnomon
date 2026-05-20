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
use cudarc::nvrtc::{Ptx, compile_ptx, result as nvrtc_result, sys as nvrtc_sys};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use memmap2::{Mmap, MmapOptions};
use std::env;
use std::ffi::CString;
use std::fs::{self, File};
use std::io::{BufWriter, IsTerminal, Write};
use std::num::NonZeroUsize;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DENSE_CHANNEL_BOUND: usize = 4096;
const BUFFER_POOL_SIZE: usize = 16384;
const SPOOL_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const MIN_GPU_WORK: usize = 100_000;
const MIN_MEGA_BATCH_VARIANTS: usize = 256;
const MAX_MEGA_BATCH_VARIANTS: usize = 16384;
const MIN_SCORE_TILE_SIZE: usize = 8;
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
    const float* sparse_weights,
    const float* sparse_missing_corrections,
    const unsigned int* sparse_columns,
    const unsigned long long* sparse_row_offsets,
    const unsigned int* reconciled_indices,
    int batch_variants,
    int num_scores_tile,
    int score_offset,
    float* out_effective,
    float* out_missing_corr,
    float* out_count
) {
    unsigned long long v =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (v >= (unsigned long long)batch_variants) return;

    unsigned int reconciled = reconciled_indices[v];
    size_t row_base = (size_t)v * (size_t)num_scores_tile;

    // Dense row init for GEMM inputs.
    for (int s = 0; s < num_scores_tile; ++s) {
        size_t dst = row_base + (size_t)s;
        out_effective[dst] = 0.0f;
        out_missing_corr[dst] = 0.0f;
        out_count[dst] = 0.0f;
    }

    unsigned long long start = sparse_row_offsets[reconciled];
    unsigned long long end = sparse_row_offsets[reconciled + 1u];
    for (unsigned long long p = start; p < end; ++p) {
        unsigned int col = sparse_columns[p];
        if (col < (unsigned int)score_offset) continue;
        unsigned int tile_col = col - (unsigned int)score_offset;
        if (tile_col >= (unsigned int)num_scores_tile) continue;
        size_t dst = row_base + (size_t)tile_col;
        out_effective[dst] = sparse_weights[p];
        out_missing_corr[dst] = -sparse_missing_corrections[p];
        out_count[dst] = 1.0f;
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

// Field order is part of CUDA correctness here. Rust drops fields
// top-to-bottom. cudarc device slices enqueue `cuMemFreeAsync` on their
// owning streams when dropped, while `CudaBlas` calls `cublasDestroy`.
// On the AoU T4 image, destroying cuBLAS after the stream has received a
// backlog of async frees aborts in glibc with "double free or corruption
// (!prev)" immediately after CUDA compute reaches 100%.
//
// Keep context, streams, and cuBLAS first so their runtime-owned Arcs are
// released before the leaf allocations. The raw stream/context handles
// still stay alive through the Arcs held by later CudaSlices, and those
// slices can then queue their async frees onto valid streams.
struct CudaRuntime {
    // Keeps the context alive until every resource below is gone.
    ctx: Arc<CudaContext>,
    compute_stream: Arc<CudaStream>,
    copy_stream: Arc<CudaStream>,
    blas: CudaBlas,
    unpack_kernel: CudaFunction,
    build_batch_mats_kernel: CudaFunction,
    sparse_weights: CudaSlice<f32>,
    sparse_missing_corrections: CudaSlice<f32>,
    sparse_columns: CudaSlice<u32>,
    sparse_row_offsets: CudaSlice<u64>,
    output_map: CudaSlice<u32>,
    mega_batch_variants: usize,
    gpu_score_chunk_size: usize,
    pinned_staging: Vec<PinnedHostSlice<u8>>,
    pinned_reconciled: Vec<PinnedHostSlice<u32>>,
}

impl Drop for CudaRuntime {
    fn drop(&mut self) {
        // Drop runs before field teardown. Synchronise both streams while
        // the context and cuBLAS handle are intact; then field order makes
        // cublasDestroy happen before CudaSlice drops enqueue async frees.
        let _ = self.ctx.bind_to_thread();
        let _ = self.compute_stream.synchronize();
        let _ = self.copy_stream.synchronize();
    }
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
}

impl CudaDims {
    fn from_prep(prep_result: &PreparationResult) -> Result<Self, PipelineError> {
        let num_people = prep_result.num_people_to_score;
        let num_scores = prep_result.score_names.len();
        Ok(Self {
            num_people,
            num_scores,
            bytes_per_variant: prep_result.bytes_per_variant as usize,
        })
    }

    #[inline]
    fn num_people_i32(self) -> Result<i32, PipelineError> {
        checked_i32("num_people", self.num_people)
    }

    #[inline]
    fn bytes_per_variant_i32(self) -> Result<i32, PipelineError> {
        checked_i32("bytes_per_variant", self.bytes_per_variant)
    }

    #[inline]
    fn result_size(self) -> Result<usize, PipelineError> {
        checked_mul_usize("result_size", self.num_people, self.num_scores)
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

    if !cuda_driver_likely_available() {
        eprintln!("> Backend: CPU fallback (no CUDA driver/device detected)");
        return Ok(None);
    }

    let runtime = match init_cuda_runtime_safely(prep) {
        Ok(runtime) => {
            eprintln!("> Backend: CUDA");
            report_loaded_cuda_libraries();
            runtime
        }
        Err(reason) => {
            eprintln!("> Backend: CPU fallback ({reason})");
            return Ok(None);
        }
    };

    // Wrap execution in catch_unwind so a panic from any cudarc layer — most
    // commonly cudarc::nvrtc dlopen failing on hosts that have a CUDA driver
    // and device files but no libnvrtc on the loader path — falls back to the
    // CPU pipeline instead of aborting the process.
    let cuda_result = catch_unwind(AssertUnwindSafe(|| match &prep.pipeline_kind {
        PipelineKind::SingleFile(path) => run_single_file_cuda(context, path, runtime),
        PipelineKind::MultiFile(boundaries) => run_multi_file_cuda(context, boundaries, runtime),
    }));

    match cuda_result {
        Ok(Ok(scores)) => Ok(Some(scores)),
        Ok(Err(e)) => Err(e),
        Err(payload) => {
            eprintln!(
                "> Backend: CPU fallback (CUDA execution panicked: {})",
                panic_payload_to_string(payload)
            );
            Ok(None)
        }
    }
}

/// Print which CUDA shared libraries the process actually dlopened.
///
/// gnomon's release binary uses cudarc with `dynamic-loading`, which
/// dlopens libcuda, libcudart, libcublas, libnvrtc at runtime based on
/// the runtime linker's search order (LD_LIBRARY_PATH, system paths,
/// ldconfig cache, …). Different CUDA library *versions* between the
/// driver (system) and the runtime libs (e.g. pip-installed
/// `nvidia-cublas-cu12`) are ABI-compatible enough to load but not
/// always ABI-compatible enough to *run* — the symptom we've seen on
/// AoU Turing T4 is "double free or corruption (!prev)" at the end of
/// the CUDA pipeline when pip-wheel CUDA libs precede the system libs
/// on LD_LIBRARY_PATH.
///
/// Printing the resolved library paths up front turns this from "spend
/// a day in gdb" into "spot the wrong path in the run log".
fn report_loaded_cuda_libraries() {
    let proc_self_maps = match fs::read_to_string("/proc/self/maps") {
        Ok(content) => content,
        Err(_) => return, // not on a Linux that exposes /proc/self/maps
    };
    let mut cuda_libs: Vec<String> = Vec::new();
    let mut seen: ahash::AHashSet<String> = ahash::AHashSet::new();
    for line in proc_self_maps.lines() {
        // Each /proc/self/maps line ends with the mapped path after the
        // last whitespace run.
        let path = match line.split_whitespace().last() {
            Some(p) if p.starts_with('/') => p,
            _ => continue,
        };
        let file_name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        let is_cuda = file_name.starts_with("libcuda")
            || file_name.starts_with("libcudart")
            || file_name.starts_with("libcublas")
            || file_name.starts_with("libnvrtc");
        if !is_cuda {
            continue;
        }
        if seen.insert(path.to_string()) {
            cuda_libs.push(path.to_string());
        }
    }
    if cuda_libs.is_empty() {
        return;
    }
    cuda_libs.sort();
    eprintln!("> CUDA libs loaded:");
    let mut wheel_paths: Vec<&str> = Vec::new();
    for path in &cuda_libs {
        eprintln!(">   {path}");
        if path.contains("/site-packages/nvidia/") {
            wheel_paths.push(path);
        }
    }
    if !wheel_paths.is_empty() {
        eprintln!(
            "> WARNING: CUDA libs above include pip-wheel paths under \
             site-packages/nvidia/. If those shadow the system CUDA \
             stack (e.g. /usr/local/cuda/lib64), an ABI/version mismatch \
             can corrupt the heap at the end of the CUDA pipeline."
        );
        eprintln!(
            "> Fix: put your system CUDA paths first on LD_LIBRARY_PATH \
             (or unset LD_LIBRARY_PATH and let the loader use its default)."
        );
        for path in wheel_paths {
            eprintln!(">   pip-wheel: {path}");
        }
    }
}

fn cuda_driver_likely_available() -> bool {
    // Respect explicit disabling commonly used in CI/container environments.
    if let Ok(devices) = env::var("CUDA_VISIBLE_DEVICES") {
        let v = devices.trim();
        if v.is_empty() || v == "-1" || v.eq_ignore_ascii_case("none") {
            return false;
        }
    }

    // Fast path on Linux GPU hosts/containers.
    if cfg!(target_os = "linux") && Path::new("/dev/nvidiactl").exists() {
        return true;
    }

    // Portable probe fallback.
    std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
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

        let (free_mem, total_mem) = cudarc::driver::result::mem_get_info()
            .map_err(|e| format!("Failed to query device memory: {e:?}"))?;
        eprintln!(
            "> CUDA device memory: free={:.2} GiB / total={:.2} GiB ({:.1}% available)",
            free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            if total_mem == 0 { 0.0 } else { 100.0 * free_mem as f64 / total_mem as f64 },
        );

        let static_bytes = prep
            .sparse_weights()
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|v| {
                v.checked_add(
                    prep.sparse_missing_corrections()
                        .len()
                        .checked_mul(std::mem::size_of::<f32>())?,
                )
            })
            .and_then(|v| {
                v.checked_add(
                    prep.sparse_score_columns()
                        .len()
                        .checked_mul(std::mem::size_of::<u32>())?,
                )
            })
            .and_then(|v| {
                v.checked_add(
                    prep.sparse_row_offsets()
                        .len()
                        .checked_mul(std::mem::size_of::<u64>())?,
                )
            })
            .and_then(|v| {
                v.checked_add(
                    prep.output_idx_to_fam_idx
                        .len()
                        .checked_mul(std::mem::size_of::<u32>())?,
                )
            })
            .ok_or_else(|| {
                "CUDA memory estimate overflow: static device allocations".to_string()
            })?;

        let num_people = prep.num_people_to_score;
        let num_scores = prep.score_names.len();
        let bytes_per_variant = prep.bytes_per_variant as usize;

        let estimate_required_bytes = |mega: usize, tile_scores: usize| -> Option<usize> {
            let tile_result = num_people.checked_mul(tile_scores)?;
            let slot_packed = PIPELINE_SLOTS.checked_mul(mega.checked_mul(bytes_per_variant)?)?;
            let slot_reconciled =
                PIPELINE_SLOTS.checked_mul(mega.checked_mul(std::mem::size_of::<u32>())?)?;
            let slot_outputs = PIPELINE_SLOTS
                .checked_mul(3)?
                .checked_mul(tile_result.checked_mul(std::mem::size_of::<f32>())?)?;

            let d_dosage = num_people
                .checked_mul(mega)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_missing = num_people
                .checked_mul(mega)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_w_eff = mega
                .checked_mul(tile_scores)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_w_corr = mega
                .checked_mul(tile_scores)?
                .checked_mul(std::mem::size_of::<f32>())?;
            let d_count_w = mega
                .checked_mul(tile_scores)?
                .checked_mul(std::mem::size_of::<f32>())?;

            static_bytes
                .checked_add(slot_packed)?
                .checked_add(slot_reconciled)?
                .checked_add(slot_outputs)?
                .checked_add(d_dosage)?
                .checked_add(d_missing)?
                .checked_add(d_w_eff)?
                .checked_add(d_w_corr)?
                .checked_add(d_count_w)
        };

        let budget = (free_mem as f64 * 0.8) as usize;
        eprintln!(
            "> CUDA budget (80% of free): {:.2} GiB; static device buffers: {:.2} MiB; \
             per-variant packed bytes: {bytes_per_variant}; people={num_people}, scores={num_scores}",
            budget as f64 / (1024.0 * 1024.0 * 1024.0),
            static_bytes as f64 / (1024.0 * 1024.0),
        );
        let lane_groups = num_scores.div_ceil(MIN_SCORE_TILE_SIZE).max(1);
        let mut tile_candidates_lane_groups = Vec::with_capacity((usize::BITS as usize) + 1);
        tile_candidates_lane_groups.push(lane_groups);
        let mut p2 = highest_power_of_two_le(lane_groups);
        while p2 >= 1 {
            if !tile_candidates_lane_groups.contains(&p2) {
                tile_candidates_lane_groups.push(p2);
            }
            p2 /= 2;
        }
        let mut selected: Option<(usize, usize)> = None;
        let mut smallest_required: Option<(usize, usize, usize)> = None;

        for tile_lane_groups in tile_candidates_lane_groups {
            let score_tile = tile_lane_groups
                .checked_mul(MIN_SCORE_TILE_SIZE)
                .ok_or_else(|| "CUDA score tile computation overflow".to_string())?
                .min(num_scores.max(MIN_SCORE_TILE_SIZE));
            let mut mega = MAX_MEGA_BATCH_VARIANTS;
            while mega >= MIN_MEGA_BATCH_VARIANTS {
                let required = estimate_required_bytes(mega, score_tile).ok_or_else(|| {
                    format!(
                        "CUDA memory estimate overflow while evaluating mega-batch={mega}, score_tile={score_tile}"
                    )
                })?;
                if required <= budget {
                    match selected {
                        None => selected = Some((mega, score_tile)),
                        Some((best_mega, best_tile)) => {
                            // Prefer larger score tiles first (better GEMM shape), then larger mega-batches.
                            if score_tile > best_tile
                                || (score_tile == best_tile && mega > best_mega)
                            {
                                selected = Some((mega, score_tile));
                            }
                        }
                    }
                    break;
                }
                // Track the cheapest (mega, score_tile, required) we considered, so
                // we can tell the user how much VRAM the minimal viable layout would
                // have needed when we end up falling back.
                match smallest_required {
                    None => smallest_required = Some((mega, score_tile, required)),
                    Some((_, _, best)) if required < best => {
                        smallest_required = Some((mega, score_tile, required));
                    }
                    _ => {}
                }
                mega /= 2;
            }
        }

        let (mega, gpu_score_chunk_size) = selected.ok_or_else(|| {
            let mut msg = format!(
                "Insufficient GPU memory for minimum CUDA workload (budget {:.2} GiB = 80% of {:.2} GiB free)",
                budget as f64 / (1024.0 * 1024.0 * 1024.0),
                free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            if let Some((m, t, req)) = smallest_required {
                msg.push_str(&format!(
                    "; smallest tile considered (mega={m}, score_tile={t}) needed {:.2} GiB, \
                     short by {:.2} GiB",
                    req as f64 / (1024.0 * 1024.0 * 1024.0),
                    (req.saturating_sub(budget)) as f64 / (1024.0 * 1024.0 * 1024.0),
                ));
            }
            if free_mem < total_mem / 2 {
                msg.push_str(&format!(
                    "; note: only {:.0}% of device memory is free — another process likely \
                     holds an allocation",
                    100.0 * free_mem as f64 / total_mem.max(1) as f64,
                ));
            }
            msg
        })?;
        eprintln!(
            "> CUDA tiling: mega_batch_variants={mega}, gpu_score_chunk_size={gpu_score_chunk_size}, vram_budget_bytes={budget}"
        );

        let sparse_weights = compute_stream
            .clone_htod(prep.sparse_weights())
            .map_err(|e| format!("Failed to upload sparse weights: {e:?}"))?;
        let sparse_missing_corrections = compute_stream
            .clone_htod(prep.sparse_missing_corrections())
            .map_err(|e| format!("Failed to upload sparse missing-corrections: {e:?}"))?;
        let sparse_columns = compute_stream
            .clone_htod(prep.sparse_score_columns())
            .map_err(|e| format!("Failed to upload sparse score columns: {e:?}"))?;
        let sparse_row_offsets = compute_stream
            .clone_htod(prep.sparse_row_offsets())
            .map_err(|e| format!("Failed to upload sparse row offsets: {e:?}"))?;
        let host_output_map: Vec<u32> =
            prep.output_idx_to_fam_idx.iter().map(|idx| idx.0).collect();
        let output_map = compute_stream
            .clone_htod(&host_output_map)
            .map_err(|e| format!("Failed to upload output index map: {e:?}"))?;

        let ptx = compile_ptx(CUDA_KERNELS).map_err(|e| format!("NVRTC compile failed: {e:?}"))?;
        let module = match ctx.load_module(ptx) {
            Ok(module) => module,
            Err(load_err) if should_retry_with_cubin(load_err) => {
                let (cc_major, cc_minor) = ctx
                    .compute_capability()
                    .map_err(|e| format!("Failed to query device compute capability: {e:?}"))?;
                eprintln!(
                    "> CUDA module load rejected PTX ({:?}); retrying with CUBIN for sm_{}{}.",
                    load_err.0, cc_major, cc_minor
                );
                let cubin = compile_cubin_for_device(cc_major, cc_minor)?;
                ctx.load_module(Ptx::from_binary(cubin)).map_err(|e| {
                    format!(
                        "Failed to load CUDA module after CUBIN fallback (original PTX load error: {load_err:?}): {e:?}"
                    )
                })?
            }
            Err(e) => return Err(format!("Failed to load CUDA module: {e:?}")),
        };

        let unpack_kernel = module
            .load_function("unpack_plink")
            .map_err(|e| format!("Failed to load unpack_plink kernel: {e:?}"))?;
        let build_batch_mats_kernel = module
            .load_function("build_batch_mats")
            .map_err(|e| format!("Failed to load build_batch_mats kernel: {e:?}"))?;
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
            ctx,
            compute_stream,
            copy_stream,
            blas,
            unpack_kernel,
            build_batch_mats_kernel,
            sparse_weights,
            sparse_missing_corrections,
            sparse_columns,
            sparse_row_offsets,
            output_map,
            mega_batch_variants: mega,
            gpu_score_chunk_size,
            pinned_staging,
            pinned_reconciled,
        })
    }
}

fn should_retry_with_cubin(load_err: DriverError) -> bool {
    matches!(
        load_err.0,
        cudarc::driver::sys::CUresult::CUDA_ERROR_UNSUPPORTED_PTX_VERSION
            | cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_PTX
    )
}

fn compile_cubin_for_device(cc_major: i32, cc_minor: i32) -> Result<Vec<u8>, String> {
    let arch_flag = format!("--gpu-architecture=sm_{cc_major}{cc_minor}");

    let src = CString::new(CUDA_KERNELS)
        .map_err(|_| "CUDA kernel source contained interior NUL".to_string())?;
    let prog = nvrtc_result::create_program(src.as_c_str(), None)
        .map_err(|e| format!("NVRTC create_program failed for CUBIN fallback: {e:?}"))?;

    let compile_res = unsafe { nvrtc_result::compile_program(prog, &[arch_flag.as_str()]) };
    if let Err(err) = compile_res {
        let log = nvrtc_program_log(prog).unwrap_or_else(|| "<no NVRTC log available>".to_string());
        let _ = unsafe { nvrtc_result::destroy_program(prog) };
        return Err(format!(
            "NVRTC CUBIN fallback compile failed ({err:?}) with {arch_flag}: {log}"
        ));
    }

    let mut cubin_size: usize = 0;
    let size_res = unsafe { nvrtc_sys::nvrtcGetCUBINSize(prog, &mut cubin_size as *mut _) };
    if let Err(e) = size_res.result() {
        let _ = unsafe { nvrtc_result::destroy_program(prog) };
        return Err(format!("NVRTC CUBIN fallback get size failed: {e:?}"));
    }

    let mut cubin_raw: Vec<std::ffi::c_char> = vec![0; cubin_size];
    let cubin_res = unsafe { nvrtc_sys::nvrtcGetCUBIN(prog, cubin_raw.as_mut_ptr()) };
    if let Err(e) = cubin_res.result() {
        let _ = unsafe { nvrtc_result::destroy_program(prog) };
        return Err(format!("NVRTC CUBIN fallback get data failed: {e:?}"));
    }

    let destroy_res = unsafe { nvrtc_result::destroy_program(prog) };
    if let Err(e) = destroy_res {
        return Err(format!(
            "NVRTC CUBIN fallback destroy_program failed: {e:?}"
        ));
    }

    Ok(cubin_raw.into_iter().map(|b| b as u8).collect())
}

fn nvrtc_program_log(prog: nvrtc_sys::nvrtcProgram) -> Option<String> {
    let raw = unsafe { nvrtc_result::get_program_log(prog) }.ok()?;
    let mut bytes: Vec<u8> = raw.into_iter().map(|b| b as u8).collect();
    if let Some(pos) = bytes.iter().position(|&b| b == 0) {
        bytes.truncate(pos);
    }
    Some(String::from_utf8_lossy(&bytes).into_owned())
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
    let progress_counter = Arc::clone(&channels.variants_processed_count);
    let mut support_guard = PipelineSupportGuard::new(start_pipeline_support(
        prep_result.num_reconciled_variants as u64,
        Arc::clone(&channels.variants_processed_count),
        Arc::clone(&channels.buffer_pool),
    ));
    let log_progress_to_stderr = support_guard
        .support
        .as_ref()
        .map(|s| s.pb.is_hidden())
        .unwrap_or(false);

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
            Arc::clone(&progress_counter),
            log_progress_to_stderr,
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

    // Tear CUDA down explicitly here, between the GPU compute phase and
    // the CPU-only complex-variant resolver below, so:
    //
    //   * The log thread is joined and the progress bar finalised
    //     (`drop(support_guard)`) BEFORE cudarc teardown begins. The
    //     fallback log thread sleeps in 2-second intervals calling
    //     `eprintln!` -- which takes the glibc stdio lock and
    //     allocates -- and the runtime's internal cuBLAS / cudart
    //     teardown also goes through glibc malloc/free. Joining first
    //     quiesces the heap so cudarc teardown runs without a
    //     concurrent writer.
    //
    //   * The runtime drops with its fields in the order declared at
    //     the top of this file: ctx, streams, blas, kernels, slices,
    //     pinned. That means `cublasDestroy` runs while the compute
    //     stream is still synchronised and free of queued
    //     `cuMemFreeAsync` calls. The streams' raw destruction
    //     (`cuStreamDestroy`) only happens once the last `Arc<CudaStream>`
    //     held by a slice drops, naturally serialising the GPU
    //     teardown.
    //
    // Inverting either piece -- explicit drops gone, or leaf resources
    // at the top of the struct -- has empirically aborted scoring with
    // "double free or corruption (!prev)" right after CUDA compute
    // completes on the AoU bare-metal Turing T4 image.
    drop(support_guard);
    drop(runtime);

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
    // Only the indicatif progress bar gets a background tick thread, and
    // only when stderr is a terminal (`pb.is_hidden() == false`). When
    // stderr is a pipe -- the typical AoU bare-metal case where the
    // biobank driver tees gnomon's output -- progress is printed from
    // the main compute loop in `process_dense_stream_cuda` instead. The
    // previous design spawned a separate "fallback log" thread that
    // periodically called `eprintln!` from its own thread, and on the
    // AoU image that pattern reliably tripped glibc's tcache
    // consistency check during cudarc/cuBLAS teardown -- "double free
    // or corruption (!prev)" between CUDA progress 100% and the
    // complex-variant resolver banner, even on single-PGS / 4 000-
    // variant inputs. With no background log thread there is no
    // cross-thread tcache migration to race against teardown.
    let progress_done = Arc::new(AtomicBool::new(false));
    let progress_handle = if pb.is_hidden() {
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

    PipelineSupport {
        progress_done,
        progress_handle,
        pb,
    }
}

fn finish_pipeline_support(support: PipelineSupport, completed: bool) {
    support.progress_done.store(true, Ordering::Relaxed);
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
    progress_counter: Arc<AtomicU64>,
    log_progress_to_stderr: bool,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let dims = CudaDims::from_prep(prep_result)?;
    let result_size = dims.result_size()?;
    let total_variants = prep_result.num_reconciled_variants as u64;
    let mut last_logged_at = Instant::now();
    let mut last_logged_value: Option<u64> = None;
    let mut emit_progress_log = |force_final: bool| {
        if !log_progress_to_stderr {
            return;
        }
        let processed = progress_counter.load(Ordering::Relaxed);
        let now = Instant::now();
        let due = force_final
            || (last_logged_value != Some(processed)
                && now.duration_since(last_logged_at) >= Duration::from_secs(2));
        if !due {
            return;
        }
        let pct = if total_variants == 0 {
            100.0
        } else {
            (processed as f64 * 100.0) / (total_variants as f64)
        };
        eprintln!("> CUDA progress: {processed}/{total_variants} ({pct:.1}%)");
        last_logged_at = now;
        last_logged_value = Some(processed);
    };
    emit_progress_log(true);

    let baseline = compute_cpu_precise_baseline(prep_result);
    let mut final_scores = Vec::with_capacity(result_size);
    for _ in 0..dims.num_people {
        final_scores.extend_from_slice(&baseline);
    }
    let mut final_counts = vec![0u32; result_size];

    let mega = runtime.mega_batch_variants;
    let gpu_score_chunk_size = runtime.gpu_score_chunk_size.max(MIN_SCORE_TILE_SIZE);
    let max_packed = checked_mul_usize("max_packed", mega, dims.bytes_per_variant)?;
    let max_weights_tile_elems =
        checked_mul_usize("max_weights_tile_elems", mega, gpu_score_chunk_size)?;
    let max_tile_result_elems = checked_mul_usize(
        "max_tile_result_elems",
        dims.num_people,
        gpu_score_chunk_size,
    )?;
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
        .alloc_zeros::<f32>(max_weights_tile_elems)
        .map_err(map_driver_err(
            "Failed to allocate effective-weight device buffer",
        ))?;
    let mut d_w_corr = runtime
        .compute_stream
        .alloc_zeros::<f32>(max_weights_tile_elems)
        .map_err(map_driver_err(
            "Failed to allocate missing-correction weight buffer",
        ))?;
    let mut d_count_w = runtime
        .compute_stream
        .alloc_zeros::<f32>(max_weights_tile_elems)
        .map_err(map_driver_err("Failed to allocate count-weight buffer"))?;
    let mut d_out_scores_slots: Vec<CudaSlice<f32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<f32>(max_tile_result_elems)
                .map_err(map_driver_err("Failed to allocate score output buffer"))
        })
        .collect::<Result<_, _>>()?;
    let mut d_out_corr_slots: Vec<CudaSlice<f32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<f32>(max_tile_result_elems)
                .map_err(map_driver_err(
                    "Failed to allocate correction output buffer",
                ))
        })
        .collect::<Result<_, _>>()?;
    let mut d_out_counts_slots: Vec<CudaSlice<f32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<f32>(max_tile_result_elems)
                .map_err(map_driver_err("Failed to allocate output count buffer"))
        })
        .collect::<Result<_, _>>()?;
    let mut host_tile_scores_slots = vec![vec![0.0f32; max_tile_result_elems]; PIPELINE_SLOTS];
    let mut host_tile_corr_slots = vec![vec![0.0f32; max_tile_result_elems]; PIPELINE_SLOTS];
    let mut host_tile_counts_slots = vec![vec![0.0f32; max_tile_result_elems]; PIPELINE_SLOTS];

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

        emit_progress_log(false);

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
                &mut host_tile_scores_slots,
                &mut host_tile_corr_slots,
                &mut host_tile_counts_slots,
                &mut final_scores,
                &mut final_counts,
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
            &mut host_tile_scores_slots,
            &mut host_tile_corr_slots,
            &mut host_tile_counts_slots,
            &mut final_scores,
            &mut final_counts,
        )?);
    }

    // Drain the streams *and then* explicitly free every per-batch
    // device allocation that this function owns BEFORE returning, while
    // we still have host control. cudarc's `CudaSlice<T>::drop` queues
    // `cuMemFreeAsync` onto `compute_stream`; if we let these locals
    // drop implicitly at function return, the queued frees outlive this
    // function and land on `compute_stream` concurrently with cudarc's
    // CudaRuntime teardown (cublasDestroy, then the runtime's own slice
    // drops). Empirically that race trips glibc with "double free or
    // corruption (!prev)" on the AoU Turing T4 image even for a single
    // 4 289-variant score, where the pipeline finishes in 2 seconds —
    // d29d7df2 reordered CudaRuntime fields to ameliorate it but did
    // not eliminate the function-local source of the backlog.
    //
    // The correct fix is to drop the locals, then synchronise — so the
    // sync waits for the cuMemFreeAsync queue this function generated,
    // and nothing in the runtime's teardown path can race against it.
    drop(host_tile_counts_slots);
    drop(host_tile_corr_slots);
    drop(host_tile_scores_slots);
    drop(d_out_counts_slots);
    drop(d_out_corr_slots);
    drop(d_out_scores_slots);
    drop(d_count_w);
    drop(d_w_corr);
    drop(d_w_eff);
    drop(d_missing);
    drop(d_dosage);
    drop(d_reconciled_slots);
    drop(d_packed_slots);
    drop(slot_last_compute_done);
    drop(slot_last_copy_done);
    drop(pending);
    drop(batch);

    runtime
        .copy_stream
        .synchronize()
        .map_err(map_driver_err("Failed to synchronize CUDA copy stream"))?;
    runtime
        .compute_stream
        .synchronize()
        .map_err(map_driver_err("Failed to synchronize CUDA compute stream"))?;

    emit_progress_log(true);

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
    host_tile_scores_slots: &mut [Vec<f32>],
    host_tile_corr_slots: &mut [Vec<f32>],
    host_tile_counts_slots: &mut [Vec<f32>],
    final_scores: &mut [f64],
    final_counts: &mut [u32],
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

    let build_launch_elems = checked_u32(
        "build_batch_mats kernel launch elements",
        work.shape.batch_len(),
    )?;
    for score_offset in (0..dims.num_scores).step_by(runtime.gpu_score_chunk_size) {
        let tile_scores = (dims.num_scores - score_offset).min(runtime.gpu_score_chunk_size);
        let tile_scores_i32 = checked_i32("tile_scores", tile_scores)?;
        let score_offset_i32 = checked_i32("score_offset", score_offset)?;
        let weights_elems =
            checked_mul_usize("weights_elems", work.shape.batch_len(), tile_scores)?;
        let tile_result_elems =
            checked_mul_usize("tile_result_elems", dims.num_people, tile_scores)?;

        unsafe {
            runtime
                .compute_stream
                .launch_builder(&runtime.build_batch_mats_kernel)
                .arg(&runtime.sparse_weights)
                .arg(&runtime.sparse_missing_corrections)
                .arg(&runtime.sparse_columns)
                .arg(&runtime.sparse_row_offsets)
                .arg(&d_reconciled_slots[work.slot].slice(0..work.shape.batch_len()))
                .arg(&batch_len_i32)
                .arg(&tile_scores_i32)
                .arg(&score_offset_i32)
                .arg(&mut d_w_eff.slice_mut(0..weights_elems))
                .arg(&mut d_w_corr.slice_mut(0..weights_elems))
                .arg(&mut d_count_w.slice_mut(0..weights_elems))
                .launch(LaunchConfig::for_num_elems(build_launch_elems))
                .map_err(map_driver_err("Failed to launch build_batch_mats kernel"))?;
        }

        run_row_major_gemm(
            &runtime.blas,
            dims.num_people,
            work.shape.batch_len(),
            tile_scores,
            &d_dosage.slice(0..unpack_elems),
            &d_w_eff.slice(0..weights_elems),
            &mut d_out_scores_slots[work.slot],
            0.0f32,
        )?;
        run_row_major_gemm(
            &runtime.blas,
            dims.num_people,
            work.shape.batch_len(),
            tile_scores,
            &d_missing.slice(0..unpack_elems),
            &d_w_corr.slice(0..weights_elems),
            &mut d_out_corr_slots[work.slot],
            0.0f32,
        )?;
        run_row_major_gemm(
            &runtime.blas,
            dims.num_people,
            work.shape.batch_len(),
            tile_scores,
            &d_missing.slice(0..unpack_elems),
            &d_count_w.slice(0..weights_elems),
            &mut d_out_counts_slots[work.slot],
            0.0f32,
        )?;

        runtime
            .compute_stream
            .memcpy_dtoh(
                &d_out_scores_slots[work.slot].slice(0..tile_result_elems),
                &mut host_tile_scores_slots[work.slot][..tile_result_elems],
            )
            .map_err(map_driver_err("Failed to copy score tile output to host"))?;
        runtime
            .compute_stream
            .memcpy_dtoh(
                &d_out_corr_slots[work.slot].slice(0..tile_result_elems),
                &mut host_tile_corr_slots[work.slot][..tile_result_elems],
            )
            .map_err(map_driver_err(
                "Failed to copy correction tile output to host",
            ))?;
        runtime
            .compute_stream
            .memcpy_dtoh(
                &d_out_counts_slots[work.slot].slice(0..tile_result_elems),
                &mut host_tile_counts_slots[work.slot][..tile_result_elems],
            )
            .map_err(map_driver_err("Failed to copy count tile output to host"))?;

        // The three `memcpy_dtoh` calls above issue `cuMemcpyDtoHAsync`.
        // cudarc's host-slice implementation for `Vec<T>` / `[T]`
        // returns `SyncOnDrop::Sync(None)`, so there is no implicit
        // stream sync when the temporary host borrow is dropped. Without
        // this explicit barrier, the CPU accumulation loop can read
        // `host_tile_*_slots[work.slot]` while CUDA is still writing the
        // same pages. That host read / DMA write race is timing-sensitive:
        // compute-sanitizer slows the path enough that PGS001320 reaches
        // output, while the normal run can abort immediately after the
        // 100% progress line.
        //
        // Sync once after all three copies and proceed with complete
        // host buffers.
        runtime
            .compute_stream
            .synchronize()
            .map_err(map_driver_err(
                "Failed to synchronize compute stream after DtoH copies",
            ))?;

        for person_idx in 0..dims.num_people {
            let src_base = person_idx * tile_scores;
            let dst_base = person_idx * dims.num_scores + score_offset;
            for j in 0..tile_scores {
                let src_idx = src_base + j;
                let dst_idx = dst_base + j;
                final_scores[dst_idx] += host_tile_scores_slots[work.slot][src_idx] as f64
                    + host_tile_corr_slots[work.slot][src_idx] as f64;
                final_counts[dst_idx] += host_tile_counts_slots[work.slot][src_idx].round() as u32;
            }
        }
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

#[inline]
fn highest_power_of_two_le(v: usize) -> usize {
    debug_assert!(v > 0);
    let shift = usize::BITS - 1 - v.leading_zeros();
    1usize << shift
}

fn compute_cpu_precise_baseline(prep_result: &PreparationResult) -> Vec<f64> {
    let mut baseline = vec![0.0f64; prep_result.score_names.len()];
    for (&col, &corr) in prep_result
        .sparse_score_columns()
        .iter()
        .zip(prep_result.sparse_missing_corrections())
    {
        baseline[col as usize] += corr as f64;
    }
    baseline
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

#[cfg(test)]
mod tests {
    #[test]
    fn cuda_runtime_field_order_keeps_cublas_teardown_before_async_frees() {
        let source = include_str!("cuda_backend.rs");
        let struct_start = source
            .find("struct CudaRuntime {")
            .expect("CudaRuntime struct must exist");
        let after_start = &source[struct_start..];
        let struct_end = after_start
            .find("\n}\n\nimpl Drop for CudaRuntime")
            .expect("CudaRuntime struct end must precede its Drop impl");
        let body = &after_start[..struct_end];

        let context_pos = body.find("ctx: Arc<CudaContext>").expect("ctx field");
        let compute_stream_pos = body
            .find("compute_stream: Arc<CudaStream>")
            .expect("compute_stream field");
        let copy_stream_pos = body
            .find("copy_stream: Arc<CudaStream>")
            .expect("copy_stream field");
        let blas_pos = body.find("blas: CudaBlas").expect("blas field");
        let kernel_pos = body
            .find("unpack_kernel: CudaFunction")
            .expect("kernel field");
        let slice_pos = body
            .find("sparse_weights: CudaSlice<f32>")
            .expect("first CudaSlice field");
        let pinned_pos = body
            .find("pinned_staging: Vec<PinnedHostSlice<u8>>")
            .expect("pinned field");

        assert!(context_pos < compute_stream_pos);
        assert!(compute_stream_pos < copy_stream_pos);
        assert!(copy_stream_pos < blas_pos);
        assert!(blas_pos < kernel_pos);
        assert!(kernel_pos < slice_pos);
        assert!(slice_pos < pinned_pos);
    }

    #[test]
    fn cuda_runtime_is_dropped_after_progress_threads_are_joined() {
        let source = include_str!("cuda_backend.rs");
        let support_drop_pos = source
            .find("drop(support_guard);\n    drop(runtime);")
            .expect("CUDA support guard must be dropped immediately before runtime");
        let resolver_pos = source
            .find("if !prep_result.complex_rules.is_empty()")
            .expect("complex resolver branch must exist");

        assert!(support_drop_pos < resolver_pos);
    }
}
