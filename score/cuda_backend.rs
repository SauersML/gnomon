use crate::score::checkpoint::ScoreCheckpointWriter;
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
    DriverError, LaunchConfig, PinnedHostSlice, PushKernelArg, sys as cuda_sys,
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
const MIN_MEGA_BATCH_VARIANTS: usize = 1;
const MIN_SCORE_TILE_SIZE: usize = 1;
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
__device__ __forceinline__ unsigned long long linear_thread_id() {
    return (((unsigned long long)blockIdx.y * (unsigned long long)gridDim.x) +
            (unsigned long long)blockIdx.x) *
               (unsigned long long)blockDim.x +
           (unsigned long long)threadIdx.x;
}

extern "C" __global__ void unpack_plink(
    const unsigned char* packed,
    const unsigned int* out_to_fam,
    int num_people,
    int batch_variants,
    int bytes_per_variant,
    float* dosage,
    float* missing
) {
    unsigned long long idx = linear_thread_id();
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

extern "C" __global__ void zero_batch_mats(
    unsigned long long total_elements,
    float* out_effective,
    float* out_missing_corr,
    float* out_count
) {
    unsigned long long idx = linear_thread_id();
    if (idx >= total_elements) return;

    out_effective[idx] = 0.0f;
    out_missing_corr[idx] = 0.0f;
    out_count[idx] = 0.0f;
}

extern "C" __global__ void scatter_batch_mats(
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
    unsigned long long v = linear_thread_id();
    if (v >= (unsigned long long)batch_variants) return;

    unsigned int reconciled = reconciled_indices[v];
    size_t row_base = (size_t)v * (size_t)num_scores_tile;

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

extern "C" __global__ void combine_score_outputs(
    const float* scores,
    const float* missing_corr,
    const float* missing_counts,
    unsigned long long total_elements,
    float* combined_scores,
    unsigned int* rounded_counts
) {
    unsigned long long idx = linear_thread_id();
    if (idx >= total_elements) return;

    float count = missing_counts[idx];
    combined_scores[idx] = scores[idx] + missing_corr[idx];
    rounded_counts[idx] = (unsigned int)floorf(count + 0.5f);
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
    zero_batch_mats_kernel: CudaFunction,
    scatter_batch_mats_kernel: CudaFunction,
    combine_score_outputs_kernel: CudaFunction,
    sparse_weights: CudaSlice<f32>,
    sparse_missing_corrections: CudaSlice<f32>,
    sparse_columns: CudaSlice<u32>,
    sparse_row_offsets: CudaSlice<u64>,
    output_map: CudaSlice<u32>,
    mega_batch_variants: usize,
    gpu_score_chunk_size: usize,
    unpack_block_size: u32,
    zero_block_size: u32,
    scatter_block_size: u32,
    combine_block_size: u32,
    device_info: CudaDeviceInfo,
    pinned_staging: Vec<PinnedHostSlice<u8>>,
    pinned_reconciled: Vec<PinnedHostSlice<u32>>,
}

struct CudaDeviceInfo {
    ordinal: usize,
    name: String,
    compute_capability: (i32, i32),
    multiprocessors: i32,
    max_threads_per_multiprocessor: i32,
    max_threads_per_block: i32,
    warp_size: i32,
    concurrent_kernels: bool,
    memory_clock_rate_khz: i32,
    global_memory_bus_width_bits: i32,
    max_grid_dim_x: i32,
    max_grid_dim_y: i32,
}

impl CudaDeviceInfo {
    fn query(ctx: &CudaContext) -> Result<Self, String> {
        let name = ctx
            .name()
            .map_err(|e| format!("Failed to query CUDA device name: {e:?}"))?;
        let compute_capability = ctx
            .compute_capability()
            .map_err(|e| format!("Failed to query CUDA compute capability: {e:?}"))?;
        let attr = |attribute, label: &str| {
            ctx.attribute(attribute)
                .map_err(|e| format!("Failed to query CUDA device attribute {label}: {e:?}"))
        };
        Ok(Self {
            ordinal: ctx.ordinal(),
            name,
            compute_capability,
            multiprocessors: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                "multiprocessor_count",
            )?,
            max_threads_per_multiprocessor: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                "max_threads_per_multiprocessor",
            )?,
            max_threads_per_block: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                "max_threads_per_block",
            )?,
            warp_size: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                "warp_size",
            )?,
            concurrent_kernels: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
                "concurrent_kernels",
            )? != 0,
            memory_clock_rate_khz: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                "memory_clock_rate",
            )?,
            global_memory_bus_width_bits: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
                "global_memory_bus_width",
            )?,
            max_grid_dim_x: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                "max_grid_dim_x",
            )?,
            max_grid_dim_y: attr(
                cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                "max_grid_dim_y",
            )?,
        })
    }

    fn theoretical_memory_bandwidth_gbps(&self) -> f64 {
        if self.memory_clock_rate_khz <= 0 || self.global_memory_bus_width_bits <= 0 {
            return 0.0;
        }
        (self.memory_clock_rate_khz as f64 * self.global_memory_bus_width_bits as f64) / 4_000_000.0
    }

    fn print(&self) {
        eprintln!(
            "> CUDA device: ordinal={} {} (sm_{}{}, SMs={}, warp={}, max_threads/block={}, max_threads/SM={}, max_grid={}x{}, concurrent_kernels={}, theoretical_mem_bw={:.1} GB/s)",
            self.ordinal,
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.multiprocessors,
            self.warp_size,
            self.max_threads_per_block,
            self.max_threads_per_multiprocessor,
            self.max_grid_dim_x,
            self.max_grid_dim_y,
            self.concurrent_kernels,
            self.theoretical_memory_bandwidth_gbps(),
        );
    }
}

#[derive(Copy, Clone)]
struct CudaTiling {
    mega_batch_variants: usize,
    score_tile: usize,
    required_bytes: usize,
}

struct CudaDevicePlan {
    ctx: Arc<CudaContext>,
    device_info: CudaDeviceInfo,
    free_mem: usize,
    total_mem: usize,
    budget: usize,
    tiling: CudaTiling,
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
    end_reconciled_exclusive: usize,
    copy_done_event: CudaEvent,
}

#[derive(Default)]
struct CudaStats {
    batches: u64,
    score_tiles: u64,
    variants: u64,
    h2d_bytes: u64,
    dtoh_bytes: u64,
    gemm_flops: u128,
    input_wait: Duration,
    slot_wait: Duration,
    host_pack: Duration,
    h2d_enqueue: Duration,
    gpu_wait: Duration,
    gpu_stream_ms: f64,
    cpu_accum: Duration,
    teardown_sync: Duration,
    total_wall: Duration,
    max_batch_variants: usize,
}

impl CudaStats {
    fn print(&self, dims: CudaDims, runtime: &CudaRuntime) {
        let wall = self.total_wall.as_secs_f64();
        let genotypes = self.variants as f64 * dims.num_people as f64;
        let genotypes_per_sec = if wall > 0.0 { genotypes / wall } else { 0.0 };
        let variants_per_sec = if wall > 0.0 {
            self.variants as f64 / wall
        } else {
            0.0
        };
        let tflops = if self.gpu_stream_ms > 0.0 {
            self.gemm_flops as f64 / (self.gpu_stream_ms / 1000.0) / 1.0e12
        } else {
            0.0
        };
        let h2d_gib = self.h2d_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let dtoh_gib = self.dtoh_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let bottlenecks = [
            ("waiting for producer/input", self.input_wait),
            ("waiting for slot reuse", self.slot_wait),
            ("packing pinned host batches", self.host_pack),
            ("enqueueing HtoD copies", self.h2d_enqueue),
            ("waiting for GPU kernels/DtoH", self.gpu_wait),
            ("CPU accumulation", self.cpu_accum),
            ("CUDA teardown synchronization", self.teardown_sync),
        ];
        let (bottleneck, bottleneck_time) = bottlenecks
            .into_iter()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .unwrap_or(("none", Duration::ZERO));

        eprintln!(
            "> CUDA scoring summary: wall={:.2}s, batches={}, score_tiles={}, max_batch_variants={}, variants/s={:.1}, genotypes/s={:.2}G",
            wall,
            self.batches,
            self.score_tiles,
            self.max_batch_variants,
            variants_per_sec,
            genotypes_per_sec / 1.0e9,
        );
        eprintln!(
            "> CUDA data movement: HtoD={:.2} GiB, DtoH={:.2} GiB; GPU stream time={:.2}s, estimated SGEMM={:.2} TFLOP/s",
            h2d_gib,
            dtoh_gib,
            self.gpu_stream_ms / 1000.0,
            tflops,
        );
        eprintln!(
            "> CUDA host timing: input_wait={:.2}s, slot_wait={:.2}s, pack={:.2}s, h2d_enqueue={:.2}s, gpu_wait={:.2}s, cpu_accum={:.2}s, teardown_sync={:.2}s",
            self.input_wait.as_secs_f64(),
            self.slot_wait.as_secs_f64(),
            self.host_pack.as_secs_f64(),
            self.h2d_enqueue.as_secs_f64(),
            self.gpu_wait.as_secs_f64(),
            self.cpu_accum.as_secs_f64(),
            self.teardown_sync.as_secs_f64(),
        );
        eprintln!(
            "> CUDA bottleneck: {bottleneck} ({:.2}s measured host wall); device='{}', sm_{}{}, SMs={}, mega_batch_variants={}, score_tile={}",
            bottleneck_time.as_secs_f64(),
            runtime.device_info.name,
            runtime.device_info.compute_capability.0,
            runtime.device_info.compute_capability.1,
            runtime.device_info.multiprocessors,
            runtime.mega_batch_variants,
            runtime.gpu_score_chunk_size,
        );
    }
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

/// Return the canonical CUDA library family name for a file basename,
/// or `None` if the basename isn't a CUDA library we care about.
///
/// Families are matched on the basename's stem (everything before the
/// first `.so`), so `libcublas.so.12.3.4.1` and `libcublas.so.12` both
/// resolve to `libcublas`, while `libcublasLt.so.12` resolves to
/// `libcublasLt` — a *separate* family despite the shared prefix.
fn cuda_library_family(basename: &str) -> Option<&'static str> {
    let stem = basename.split(".so").next()?;
    match stem {
        "libcuda" => Some("libcuda"),
        "libcudart" => Some("libcudart"),
        "libcublas" => Some("libcublas"),
        "libcublasLt" => Some("libcublasLt"),
        "libcusparse" => Some("libcusparse"),
        "libcusolver" => Some("libcusolver"),
        "libnvJitLink" => Some("libnvJitLink"),
        "libnvrtc" => Some("libnvrtc"),
        _ => None,
    }
}

/// Walk `/proc/self/maps` and return CUDA libraries grouped by family.
/// `BTreeMap`/`BTreeSet` give stable ordering in diagnostic output.
/// Returns an empty map on non-Linux hosts (where `/proc/self/maps`
/// doesn't exist) — callers treat that as "no conflicts to detect".
fn cuda_library_mappings()
-> std::collections::BTreeMap<&'static str, std::collections::BTreeSet<String>> {
    let mut by_family: std::collections::BTreeMap<
        &'static str,
        std::collections::BTreeSet<String>,
    > = std::collections::BTreeMap::new();
    let Ok(maps) = fs::read_to_string("/proc/self/maps") else {
        return by_family;
    };
    for line in maps.lines() {
        let Some(path) = line.split_whitespace().last() else {
            continue;
        };
        if !path.starts_with('/') {
            continue;
        }
        let name = Path::new(path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if let Some(family) = cuda_library_family(name) {
            by_family
                .entry(family)
                .or_default()
                .insert(path.to_string());
        }
    }
    by_family
}

/// Detect whether more than one distinct file is mapped into the
/// process for the same CUDA SONAME family.
///
/// This is the canonical failure mode behind the AoU "double free or
/// corruption (!prev)" abort. glibc's dlopen deduplicates by
/// `(device, inode)`, not by SONAME. When pip's `nvidia-*-cu12`
/// wheels and the system CUDA toolkit are both reachable to the
/// loader, an absolute-path `dlopen` of the wheel and a bare-name
/// `dlopen("libcublas.so")` that walks `ld.so.cache` can succeed
/// independently and leave *two distinct files* with the same SONAME
/// permanently mapped. cuBLAS state then splits across two cuBLAS
/// implementations whose internal allocators, struct layouts, and
/// static initializers disagree — and the next `cublasDestroy`
/// frees a chunk one allocator never owned, tripping glibc's malloc
/// consistency check.
///
/// On `Ok(())`, mappings are consistent. On `Err`, the string is a
/// multi-line, actionable report naming every conflicting path.
fn detect_cuda_library_conflicts() -> Result<(), String> {
    let by_family = cuda_library_mappings();
    let conflicts: Vec<(&'static str, Vec<String>)> = by_family
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .map(|(family, paths)| (family, paths.into_iter().collect()))
        .collect();
    if conflicts.is_empty() {
        return Ok(());
    }
    let mut msg = String::from(
        "CUDA library conflict: multiple distinct files share a SONAME and \
         coexist in this process. glibc dlopen deduplicates by (device, \
         inode) not by SONAME, so all of the following are simultaneously \
         mapped. cuBLAS handle state would split across them and the next \
         cublasDestroy_v2 would abort with 'double free or corruption \
         (!prev)'.",
    );
    for (family, paths) in &conflicts {
        msg.push_str(&format!("\n  {family}:"));
        for path in paths {
            msg.push_str(&format!("\n    {path}"));
        }
    }
    msg.push_str(
        "\nKeep exactly one CUDA toolkit reachable to the loader: either the \
         system toolkit (usually /usr/local/cuda*) or the pip nvidia-*-cu12 \
         wheels, not both. Refusing to create cuBLAS handles to avoid \
         crashing later.",
    );
    Err(msg)
}

/// Print every CUDA library currently mapped into the process, for
/// post-init diagnostics. Pairs with `detect_cuda_library_conflicts`
/// — by the time we call this, any conflict has already been
/// rejected as a hard error.
fn report_loaded_cuda_libraries() {
    let by_family = cuda_library_mappings();
    if by_family.is_empty() {
        return;
    }
    let mut all: Vec<String> = by_family.into_values().flatten().collect();
    all.sort();
    eprintln!("> CUDA libs loaded:");
    for path in &all {
        eprintln!(">   {path}");
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

fn preflight_cuda_dynamic_libraries() -> Result<(), String> {
    if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
        return Err(format!(
            "NVRTC library not loadable; CUDA scoring requires libnvrtc for kernel compilation. \
             Searched library names: {}",
            cudarc::get_lib_name_candidates("nvrtc").join(", ")
        ));
    }
    if !unsafe { cudarc::cublas::sys::is_culib_present() } {
        return Err(format!(
            "cuBLAS library not loadable; CUDA scoring requires cuBLAS for score matrix products. \
             Searched library names: {}",
            cudarc::get_lib_name_candidates("cublas").join(", ")
        ));
    }
    Ok(())
}

fn estimate_static_cuda_bytes(prep: &PreparationResult) -> Result<usize, String> {
    prep.sparse_weights()
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
        .ok_or_else(|| "CUDA memory estimate overflow: static device allocations".to_string())
}

fn estimate_cuda_required_bytes(
    prep: &PreparationResult,
    static_bytes: usize,
    mega: usize,
    tile_scores: usize,
) -> Option<usize> {
    let num_people = prep.num_people_to_score;
    let bytes_per_variant = prep.bytes_per_variant as usize;
    let tile_result = num_people.checked_mul(tile_scores)?;
    let slot_packed = PIPELINE_SLOTS.checked_mul(mega.checked_mul(bytes_per_variant)?)?;
    let slot_reconciled =
        PIPELINE_SLOTS.checked_mul(mega.checked_mul(std::mem::size_of::<u32>())?)?;
    let slot_outputs = PIPELINE_SLOTS
        .checked_mul(5)?
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
}

fn select_cuda_tiling(
    prep: &PreparationResult,
    static_bytes: usize,
    free_mem: usize,
    total_mem: usize,
) -> Result<(usize, CudaTiling), String> {
    let budget = (free_mem as f64 * 0.8) as usize;
    let num_scores = prep.score_names.len();
    let largest_candidate_mega =
        highest_power_of_two_le(prep.num_reconciled_variants.max(1)).max(MIN_MEGA_BATCH_VARIANTS);
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

    let mut selected: Option<CudaTiling> = None;
    let mut smallest_required: Option<CudaTiling> = None;
    for tile_lane_groups in tile_candidates_lane_groups {
        let score_tile = tile_lane_groups
            .checked_mul(MIN_SCORE_TILE_SIZE)
            .ok_or_else(|| "CUDA score tile computation overflow".to_string())?
            .min(num_scores.max(MIN_SCORE_TILE_SIZE));
        let mut mega = largest_candidate_mega;
        while mega >= MIN_MEGA_BATCH_VARIANTS {
            let required =
                estimate_cuda_required_bytes(prep, static_bytes, mega, score_tile).ok_or_else(
                    || {
                        format!(
                            "CUDA memory estimate overflow while evaluating mega-batch={mega}, score_tile={score_tile}"
                        )
                    },
                )?;
            let tiling = CudaTiling {
                mega_batch_variants: mega,
                score_tile,
                required_bytes: required,
            };
            if required <= budget {
                match selected {
                    None => selected = Some(tiling),
                    Some(best) => {
                        if score_tile > best.score_tile
                            || (score_tile == best.score_tile && mega > best.mega_batch_variants)
                        {
                            selected = Some(tiling);
                        }
                    }
                }
                break;
            }
            match smallest_required {
                None => smallest_required = Some(tiling),
                Some(best) if required < best.required_bytes => smallest_required = Some(tiling),
                _ => {}
            }
            mega /= 2;
        }
    }

    selected.map(|tiling| (budget, tiling)).ok_or_else(|| {
        let mut msg = format!(
            "insufficient memory for minimum CUDA workload (budget {:.2} GiB = 80% of {:.2} GiB free)",
            budget as f64 / (1024.0 * 1024.0 * 1024.0),
            free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        if let Some(smallest) = smallest_required {
            msg.push_str(&format!(
                "; smallest tile considered (mega={}, score_tile={}) needed {:.2} GiB, short by {:.2} GiB",
                smallest.mega_batch_variants,
                smallest.score_tile,
                smallest.required_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                smallest.required_bytes.saturating_sub(budget) as f64
                    / (1024.0 * 1024.0 * 1024.0),
            ));
        }
        if free_mem < total_mem / 2 {
            msg.push_str(&format!(
                "; only {:.0}% of device memory is free",
                100.0 * free_mem as f64 / total_mem.max(1) as f64,
            ));
        }
        msg
    })
}

fn select_cuda_device_plan(
    prep: &PreparationResult,
    static_bytes: usize,
) -> Result<CudaDevicePlan, String> {
    let device_count = CudaContext::device_count()
        .map_err(|e| format!("Failed to query CUDA device count: {e:?}"))?;
    if device_count <= 0 {
        return Err("CUDA reported zero visible devices".to_string());
    }

    let mut best: Option<CudaDevicePlan> = None;
    let mut rejections = Vec::new();
    for ordinal in 0..device_count as usize {
        let ctx = match CudaContext::new(ordinal) {
            Ok(ctx) => ctx,
            Err(e) => {
                rejections.push(format!("ordinal {ordinal}: init failed: {e:?}"));
                continue;
            }
        };
        if let Err(e) = ctx.bind_to_thread() {
            rejections.push(format!("ordinal {ordinal}: bind failed: {e:?}"));
            continue;
        }
        let device_info = match CudaDeviceInfo::query(&ctx) {
            Ok(info) => info,
            Err(e) => {
                rejections.push(format!("ordinal {ordinal}: {e}"));
                continue;
            }
        };
        let (free_mem, total_mem) = match ctx.mem_get_info() {
            Ok(info) => info,
            Err(e) => {
                rejections.push(format!("ordinal {ordinal}: memory query failed: {e:?}"));
                continue;
            }
        };
        let (budget, tiling) = match select_cuda_tiling(prep, static_bytes, free_mem, total_mem) {
            Ok(plan) => plan,
            Err(e) => {
                rejections.push(format!("ordinal {ordinal} {}: {e}", device_info.name));
                continue;
            }
        };
        let candidate = CudaDevicePlan {
            ctx,
            device_info,
            free_mem,
            total_mem,
            budget,
            tiling,
        };
        if best
            .as_ref()
            .map(|current| cuda_device_plan_is_better(&candidate, current))
            .unwrap_or(true)
        {
            best = Some(candidate);
        }
    }

    let selected = best.ok_or_else(|| {
        if rejections.is_empty() {
            "No CUDA device could be initialized".to_string()
        } else {
            format!(
                "No visible CUDA device can run this workload:\n  {}",
                rejections.join("\n  ")
            )
        }
    })?;
    selected
        .ctx
        .bind_to_thread()
        .map_err(|e| format!("Failed to bind selected CUDA device: {e:?}"))?;
    eprintln!(
        "> CUDA device selection: chose ordinal {} from {} visible device(s)",
        selected.device_info.ordinal, device_count
    );
    Ok(selected)
}

fn cuda_device_plan_is_better(candidate: &CudaDevicePlan, current: &CudaDevicePlan) -> bool {
    let cand = (
        candidate.tiling.score_tile,
        candidate.tiling.mega_batch_variants,
        candidate.free_mem,
        candidate.device_info.multiprocessors,
    );
    let curr = (
        current.tiling.score_tile,
        current.tiling.mega_batch_variants,
        current.free_mem,
        current.device_info.multiprocessors,
    );
    cand > curr
}

impl CudaRuntime {
    fn new(prep: &PreparationResult) -> Result<Self, String> {
        preflight_cuda_dynamic_libraries()?;

        let static_bytes = estimate_static_cuda_bytes(prep)?;
        let plan = select_cuda_device_plan(prep, static_bytes)?;
        let CudaDevicePlan {
            ctx,
            device_info,
            free_mem,
            total_mem,
            budget,
            tiling,
        } = plan;
        device_info.print();
        let copy_stream = ctx
            .new_stream()
            .map_err(|e| format!("Failed to create CUDA copy stream: {e:?}"))?;
        let compute_stream = ctx
            .new_stream()
            .map_err(|e| format!("Failed to create CUDA compute stream: {e:?}"))?;

        eprintln!(
            "> CUDA device memory: free={:.2} GiB / total={:.2} GiB ({:.1}% available)",
            free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            if total_mem == 0 {
                0.0
            } else {
                100.0 * free_mem as f64 / total_mem as f64
            },
        );

        let num_people = prep.num_people_to_score;
        let num_scores = prep.score_names.len();
        let bytes_per_variant = prep.bytes_per_variant as usize;

        eprintln!(
            "> CUDA budget (80% of free): {:.2} GiB; static device buffers: {:.2} MiB; \
             per-variant packed bytes: {bytes_per_variant}; people={num_people}, scores={num_scores}",
            budget as f64 / (1024.0 * 1024.0 * 1024.0),
            static_bytes as f64 / (1024.0 * 1024.0),
        );
        let mega = tiling.mega_batch_variants;
        let gpu_score_chunk_size = tiling.score_tile;
        eprintln!(
            "> CUDA tiling: mega_batch_variants={mega}, gpu_score_chunk_size={gpu_score_chunk_size}, required_bytes={}, vram_budget_bytes={budget}",
            tiling.required_bytes
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
        let zero_batch_mats_kernel = module
            .load_function("zero_batch_mats")
            .map_err(|e| format!("Failed to load zero_batch_mats kernel: {e:?}"))?;
        let scatter_batch_mats_kernel = module
            .load_function("scatter_batch_mats")
            .map_err(|e| format!("Failed to load scatter_batch_mats kernel: {e:?}"))?;
        let combine_score_outputs_kernel = module
            .load_function("combine_score_outputs")
            .map_err(|e| format!("Failed to load combine_score_outputs kernel: {e:?}"))?;
        let unpack_block_size =
            choose_kernel_block_size(&unpack_kernel, &device_info, "unpack_plink")?;
        let zero_block_size =
            choose_kernel_block_size(&zero_batch_mats_kernel, &device_info, "zero_batch_mats")?;
        let scatter_block_size = choose_kernel_block_size(
            &scatter_batch_mats_kernel,
            &device_info,
            "scatter_batch_mats",
        )?;
        let combine_block_size = choose_kernel_block_size(
            &combine_score_outputs_kernel,
            &device_info,
            "combine_score_outputs",
        )?;
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

        // Force-init cudarc's persistent libcublas handle BEFORE the
        // conflict scan, so `/proc/self/maps` reflects every CUDA
        // library that `CudaBlas::new` would touch. `culib()` caches
        // the loaded `Library` in a `OnceLock` and never drops it, so
        // unlike `is_culib_present()` (which dlopen→dlclose's its
        // probe handles) the mapping is guaranteed to stick. The call
        // panics with `panic_no_lib_found` when no candidate is
        // dlopen'able — wrapped in catch_unwind so we degrade to CPU
        // fallback instead of aborting the process.
        match catch_unwind(AssertUnwindSafe(|| unsafe {
            let _ = cudarc::cublas::sys::culib();
        })) {
            Ok(()) => {}
            Err(payload) => {
                return Err(format!(
                    "cuBLAS library not loadable on this host ({})",
                    panic_payload_to_string(payload)
                ));
            }
        }
        detect_cuda_library_conflicts()?;

        let blas = CudaBlas::new(compute_stream.clone())
            .map_err(|e| format!("Failed to initialize cuBLAS: {e:?}"))?;

        Ok(Self {
            ctx,
            compute_stream,
            copy_stream,
            blas,
            unpack_kernel,
            zero_batch_mats_kernel,
            scatter_batch_mats_kernel,
            combine_score_outputs_kernel,
            sparse_weights,
            sparse_missing_corrections,
            sparse_columns,
            sparse_row_offsets,
            output_map,
            mega_batch_variants: mega,
            gpu_score_chunk_size,
            unpack_block_size,
            zero_block_size,
            scatter_block_size,
            combine_block_size,
            device_info,
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
    let resume_from = context.checkpoint_completed_variants();
    if resume_from > 0 {
        eprintln!(
            "> Resuming CUDA score computation from checkpoint at {}/{} variants.",
            resume_from, prep_result.num_reconciled_variants
        );
        channels
            .variants_processed_count
            .store(resume_from as u64, Ordering::Relaxed);
    }
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
                            resume_from,
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
                            resume_from,
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
            context,
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
    context: &PipelineContext,
    runtime: &mut CudaRuntime,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    progress_counter: Arc<AtomicU64>,
    log_progress_to_stderr: bool,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let dims = CudaDims::from_prep(prep_result)?;
    let result_size = dims.result_size()?;
    let total_variants = prep_result.num_reconciled_variants as u64;
    let total_wall_start = Instant::now();
    let mut stats = CudaStats::default();
    let mut checkpoint_writer = match (
        context.checkpoint_path.as_ref(),
        context.checkpoint_fingerprint,
    ) {
        (Some(path), Some(fingerprint)) => Some(ScoreCheckpointWriter::new(
            path.clone(),
            fingerprint,
            prep_result.num_reconciled_variants,
        )),
        _ => None,
    };
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
    if let Some(checkpoint) = context.checkpoint.as_ref() {
        if checkpoint.sum_scores.len() != final_scores.len()
            || checkpoint.missing_counts.len() != final_counts.len()
        {
            return Err(PipelineError::Compute(format!(
                "Checkpoint accumulator shape mismatch: scores {} vs {}, counts {} vs {}.",
                checkpoint.sum_scores.len(),
                final_scores.len(),
                checkpoint.missing_counts.len(),
                final_counts.len()
            )));
        }
        final_scores.copy_from_slice(&checkpoint.sum_scores);
        final_counts.copy_from_slice(&checkpoint.missing_counts);
    }

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
    eprintln!(
        "> CUDA scoring buffers: pipeline_slots={PIPELINE_SLOTS}, max_packed={:.2} MiB/slot, \
         dosage_or_missing={:.2} GiB, weights_tile={:.2} MiB, output_tile={:.2} MiB/slot, \
         unpack_block={}, zero_block={}, scatter_block={}, combine_block={}",
        max_packed as f64 / (1024.0 * 1024.0),
        checked_mul_usize("dosage buffer bytes", dims.num_people, mega)?
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| PipelineError::Compute(
                "dosage buffer byte estimate overflow".to_string()
            ))? as f64
            / (1024.0 * 1024.0 * 1024.0),
        max_weights_tile_elems as f64 * std::mem::size_of::<f32>() as f64 / (1024.0 * 1024.0),
        max_tile_result_elems as f64 * std::mem::size_of::<f32>() as f64 / (1024.0 * 1024.0),
        runtime.unpack_block_size,
        runtime.zero_block_size,
        runtime.scatter_block_size,
        runtime.combine_block_size,
    );
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
    let mut d_out_combined_slots: Vec<CudaSlice<f32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<f32>(max_tile_result_elems)
                .map_err(map_driver_err(
                    "Failed to allocate combined score output buffer",
                ))
        })
        .collect::<Result<_, _>>()?;
    let mut d_out_counts_u32_slots: Vec<CudaSlice<u32>> = (0..PIPELINE_SLOTS)
        .map(|_| {
            runtime
                .compute_stream
                .alloc_zeros::<u32>(max_tile_result_elems)
                .map_err(map_driver_err(
                    "Failed to allocate rounded output count buffer",
                ))
        })
        .collect::<Result<_, _>>()?;
    let mut host_tile_scores_slots = vec![vec![0.0f32; max_tile_result_elems]; PIPELINE_SLOTS];
    let mut host_tile_counts_slots = vec![vec![0u32; max_tile_result_elems]; PIPELINE_SLOTS];

    let mut batch: Vec<WorkItem> = Vec::with_capacity(mega);
    let mut pending: Option<PendingBatch> = None;
    let mut slot_last_compute_done: [Option<CudaEvent>; PIPELINE_SLOTS] = [None, None];
    let mut slot_last_copy_done: [Option<CudaEvent>; PIPELINE_SLOTS] = [None, None];
    let mut batch_counter: usize = 0;

    loop {
        batch.clear();
        let input_wait_start = Instant::now();
        match rx.recv() {
            Ok(Ok(first_item)) => batch.push(first_item),
            Ok(Err(e)) => return Err(e),
            Err(_) => break,
        }
        stats.input_wait += input_wait_start.elapsed();
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
        let end_reconciled_exclusive = batch
            .last()
            .map(|item| item.reconciled_variant_index.0 as usize + 1)
            .unwrap_or(0);
        let shape = BatchShape::from_counts(dims, batch_len)?;

        let slot = batch_counter % PIPELINE_SLOTS;
        batch_counter += 1;

        emit_progress_log(false);

        if let Some(event) = &slot_last_copy_done[slot] {
            let wait_start = Instant::now();
            event
                .synchronize()
                .map_err(map_driver_err("Failed waiting for prior copy event"))?;
            stats.slot_wait += wait_start.elapsed();
        }
        if let Some(event) = &slot_last_compute_done[slot] {
            let wait_start = Instant::now();
            runtime.copy_stream.wait(event).map_err(map_driver_err(
                "Copy stream failed waiting for compute event",
            ))?;
            stats.slot_wait += wait_start.elapsed();
        }

        let pack_start = Instant::now();
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
        stats.host_pack += pack_start.elapsed();

        let h2d_start = Instant::now();
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
        stats.h2d_enqueue += h2d_start.elapsed();
        stats.h2d_bytes +=
            packed_len as u64 + (shape.batch_len() as u64 * std::mem::size_of::<u32>() as u64);
        stats.variants += shape.batch_len() as u64;
        stats.batches += 1;
        stats.max_batch_variants = stats.max_batch_variants.max(shape.batch_len());
        let copy_done_event = record_cuda_sync_event(
            &runtime.copy_stream,
            "Failed to record copy completion event",
        )?;
        slot_last_copy_done[slot] = Some(record_cuda_sync_event(
            &runtime.copy_stream,
            "Failed to record slot copy completion event",
        )?);
        drop(guards);

        let current = PendingBatch {
            slot,
            shape,
            packed_len,
            end_reconciled_exclusive,
            copy_done_event,
        };

        if let Some(prev) = pending.take() {
            let slot = prev.slot;
            let completed = prev.end_reconciled_exclusive;
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
                &mut d_out_combined_slots,
                &mut d_out_counts_u32_slots,
                &mut host_tile_scores_slots,
                &mut host_tile_counts_slots,
                &mut final_scores,
                &mut final_counts,
                &mut stats,
            )?);
            maybe_save_cuda_checkpoint(
                checkpoint_writer.as_mut(),
                completed,
                &final_scores,
                &final_counts,
                false,
            )?;
        }
        pending = Some(current);
    }

    if let Some(last) = pending.take() {
        let slot = last.slot;
        let completed = last.end_reconciled_exclusive;
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
            &mut d_out_combined_slots,
            &mut d_out_counts_u32_slots,
            &mut host_tile_scores_slots,
            &mut host_tile_counts_slots,
            &mut final_scores,
            &mut final_counts,
            &mut stats,
        )?);
        maybe_save_cuda_checkpoint(
            checkpoint_writer.as_mut(),
            completed,
            &final_scores,
            &final_counts,
            true,
        )?;
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
    drop(host_tile_scores_slots);
    drop(d_out_counts_u32_slots);
    drop(d_out_combined_slots);
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

    let teardown_sync_start = Instant::now();
    runtime
        .copy_stream
        .synchronize()
        .map_err(map_driver_err("Failed to synchronize CUDA copy stream"))?;
    runtime
        .compute_stream
        .synchronize()
        .map_err(map_driver_err("Failed to synchronize CUDA compute stream"))?;
    stats.teardown_sync += teardown_sync_start.elapsed();

    emit_progress_log(true);
    stats.total_wall = total_wall_start.elapsed();
    stats.print(dims, runtime);

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
    d_out_combined_slots: &mut [CudaSlice<f32>],
    d_out_counts_u32_slots: &mut [CudaSlice<u32>],
    host_tile_scores_slots: &mut [Vec<f32>],
    host_tile_counts_slots: &mut [Vec<u32>],
    final_scores: &mut [f64],
    final_counts: &mut [u32],
    stats: &mut CudaStats,
) -> Result<CudaEvent, PipelineError> {
    runtime
        .compute_stream
        .wait(&work.copy_done_event)
        .map_err(map_driver_err(
            "Compute stream failed waiting for copy event",
        ))?;
    let gpu_start_event = record_cuda_timing_event(
        &runtime.compute_stream,
        "Failed to record CUDA work start event",
    )?;
    let mut next_tile_start_event = Some(gpu_start_event);
    let mut final_compute_done_event = None;

    let unpack_elems = work.shape.unpack_elems()?;
    let num_people_i32 = work.shape.dims.num_people_i32()?;
    let batch_len_i32 = checked_i32("batch_len", work.shape.batch_len())?;
    let bytes_per_variant_i32 = work.shape.dims.bytes_per_variant_i32()?;
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
            .launch(launch_config_for_num_elems(
                unpack_elems,
                runtime.unpack_block_size,
                &runtime.device_info,
            )?)
            .map_err(map_driver_err("Failed to launch unpack_plink kernel"))?;
    }

    for score_offset in (0..dims.num_scores).step_by(runtime.gpu_score_chunk_size) {
        let tile_scores = (dims.num_scores - score_offset).min(runtime.gpu_score_chunk_size);
        let tile_start_event = match next_tile_start_event.take() {
            Some(event) => event,
            None => record_cuda_timing_event(
                &runtime.compute_stream,
                "Failed to record CUDA tile start event",
            )?,
        };
        let tile_scores_i32 = checked_i32("tile_scores", tile_scores)?;
        let score_offset_i32 = checked_i32("score_offset", score_offset)?;
        let weights_elems =
            checked_mul_usize("weights_elems", work.shape.batch_len(), tile_scores)?;
        let tile_result_elems =
            checked_mul_usize("tile_result_elems", dims.num_people, tile_scores)?;
        stats.score_tiles += 1;
        stats.gemm_flops +=
            6u128 * dims.num_people as u128 * work.shape.batch_len() as u128 * tile_scores as u128;

        unsafe {
            runtime
                .compute_stream
                .launch_builder(&runtime.zero_batch_mats_kernel)
                .arg(&checked_u64("weights_elems", weights_elems)?)
                .arg(&mut d_w_eff.slice_mut(0..weights_elems))
                .arg(&mut d_w_corr.slice_mut(0..weights_elems))
                .arg(&mut d_count_w.slice_mut(0..weights_elems))
                .launch(launch_config_for_num_elems(
                    weights_elems,
                    runtime.zero_block_size,
                    &runtime.device_info,
                )?)
                .map_err(map_driver_err("Failed to launch zero_batch_mats kernel"))?;
        }

        unsafe {
            runtime
                .compute_stream
                .launch_builder(&runtime.scatter_batch_mats_kernel)
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
                .launch(launch_config_for_num_elems(
                    work.shape.batch_len(),
                    runtime.scatter_block_size,
                    &runtime.device_info,
                )?)
                .map_err(map_driver_err("Failed to launch scatter_batch_mats kernel"))?;
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

        unsafe {
            runtime
                .compute_stream
                .launch_builder(&runtime.combine_score_outputs_kernel)
                .arg(&d_out_scores_slots[work.slot].slice(0..tile_result_elems))
                .arg(&d_out_corr_slots[work.slot].slice(0..tile_result_elems))
                .arg(&d_out_counts_slots[work.slot].slice(0..tile_result_elems))
                .arg(&checked_u64("tile_result_elems", tile_result_elems)?)
                .arg(&mut d_out_combined_slots[work.slot].slice_mut(0..tile_result_elems))
                .arg(&mut d_out_counts_u32_slots[work.slot].slice_mut(0..tile_result_elems))
                .launch(launch_config_for_num_elems(
                    tile_result_elems,
                    runtime.combine_block_size,
                    &runtime.device_info,
                )?)
                .map_err(map_driver_err(
                    "Failed to launch combine_score_outputs kernel",
                ))?;
        }

        runtime
            .compute_stream
            .memcpy_dtoh(
                &d_out_combined_slots[work.slot].slice(0..tile_result_elems),
                &mut host_tile_scores_slots[work.slot][..tile_result_elems],
            )
            .map_err(map_driver_err("Failed to copy score tile output to host"))?;
        runtime
            .compute_stream
            .memcpy_dtoh(
                &d_out_counts_u32_slots[work.slot].slice(0..tile_result_elems),
                &mut host_tile_counts_slots[work.slot][..tile_result_elems],
            )
            .map_err(map_driver_err("Failed to copy count tile output to host"))?;
        stats.dtoh_bytes +=
            (tile_result_elems * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>())) as u64;
        let tile_done_event = record_cuda_timing_event(
            &runtime.compute_stream,
            "Failed to record CUDA tile completion event",
        )?;

        // The `memcpy_dtoh` calls above issue `cuMemcpyDtoHAsync`.
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
        let gpu_wait_start = Instant::now();
        runtime
            .compute_stream
            .synchronize()
            .map_err(map_driver_err(
                "Failed to synchronize compute stream after DtoH copies",
            ))?;
        stats.gpu_wait += gpu_wait_start.elapsed();
        stats.gpu_stream_ms += tile_start_event
            .elapsed_ms(&tile_done_event)
            .map_err(map_driver_err("Failed to measure CUDA tile elapsed time"))?
            as f64;
        final_compute_done_event = Some(tile_done_event);

        let accum_start = Instant::now();
        for person_idx in 0..dims.num_people {
            let src_base = person_idx * tile_scores;
            let dst_base = person_idx * dims.num_scores + score_offset;
            for j in 0..tile_scores {
                let src_idx = src_base + j;
                let dst_idx = dst_base + j;
                final_scores[dst_idx] += host_tile_scores_slots[work.slot][src_idx] as f64;
                final_counts[dst_idx] += host_tile_counts_slots[work.slot][src_idx];
            }
        }
        stats.cpu_accum += accum_start.elapsed();
    }

    final_compute_done_event
        .ok_or_else(|| PipelineError::Compute("CUDA compute produced no score tiles".to_string()))
}

fn maybe_save_cuda_checkpoint(
    writer: Option<&mut ScoreCheckpointWriter>,
    completed_variants: usize,
    final_scores: &[f64],
    final_counts: &[u32],
    force: bool,
) -> Result<(), PipelineError> {
    let Some(writer) = writer else {
        return Ok(());
    };
    let wrote = writer
        .maybe_save(completed_variants, final_scores, final_counts, force)
        .map_err(|e| PipelineError::Io(format!("Failed to write score checkpoint: {e}")))?;
    if wrote {
        eprintln!("> Score checkpoint saved after {completed_variants} completed variants.");
    }
    Ok(())
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

fn choose_kernel_block_size(
    kernel: &CudaFunction,
    device: &CudaDeviceInfo,
    label: &str,
) -> Result<u32, String> {
    let warp = u32::try_from(device.warp_size.max(1))
        .map_err(|_| format!("Invalid CUDA warp size {}", device.warp_size))?;
    let function_limit = kernel
        .max_threads_per_block()
        .map_err(|e| format!("Failed to query {label} max threads/block: {e:?}"))?;
    let function_limit = u32::try_from(function_limit.max(1))
        .map_err(|_| format!("Invalid {label} max threads/block {function_limit}"))?;
    let device_limit = u32::try_from(device.max_threads_per_block.max(1)).map_err(|_| {
        format!(
            "Invalid CUDA max_threads_per_block {}",
            device.max_threads_per_block
        )
    })?;
    let max_block = function_limit.min(device_limit);
    if max_block < warp {
        return Err(format!(
            "{label} allows only {max_block} threads/block, below CUDA warp size {warp}"
        ));
    }

    let mut candidates = Vec::new();
    let mut block = (warp * 4).min(max_block).max(warp);
    while block <= max_block {
        if block % warp == 0 {
            candidates.push(block);
        }
        match block.checked_mul(2) {
            Some(next) if next > block => block = next,
            _ => break,
        }
    }
    if candidates.last().copied() != Some(max_block) && max_block % warp == 0 {
        candidates.push(max_block);
    }

    let mut best: Option<(u32, u32)> = None;
    for block in candidates {
        let active_blocks = kernel
            .occupancy_max_active_blocks_per_multiprocessor(block, 0, None)
            .map_err(|e| format!("Failed to query {label} occupancy for block={block}: {e:?}"))?;
        let active_threads = active_blocks.saturating_mul(block);
        match best {
            None => best = Some((block, active_threads)),
            Some((best_block, best_threads)) => {
                if active_threads > best_threads
                    || (active_threads == best_threads && block < best_block)
                {
                    best = Some((block, active_threads));
                }
            }
        }
    }

    let (block, active_threads) =
        best.ok_or_else(|| format!("No legal CUDA block size candidates for {label}"))?;
    let active_blocks = if block == 0 {
        0
    } else {
        active_threads / block
    };
    let occupancy_pct = if device.max_threads_per_multiprocessor <= 0 {
        0.0
    } else {
        100.0 * active_threads as f64 / device.max_threads_per_multiprocessor as f64
    };
    eprintln!(
        "> CUDA launch {label}: block_threads={block}, active_blocks/SM={active_blocks}, active_threads/SM={active_threads} ({occupancy_pct:.1}% occupancy cap)"
    );
    Ok(block)
}

fn launch_config_for_num_elems(
    n: usize,
    block_size: u32,
    device: &CudaDeviceInfo,
) -> Result<LaunchConfig, PipelineError> {
    let block_size = block_size.max(1);
    let grid_blocks = (n as u128).div_ceil(block_size as u128);
    let max_grid_x = u32::try_from(device.max_grid_dim_x.max(1)).map_err(|_| {
        PipelineError::Compute(format!(
            "CUDA device reported invalid max_grid_dim_x={}",
            device.max_grid_dim_x
        ))
    })?;
    let max_grid_y = u32::try_from(device.max_grid_dim_y.max(1)).map_err(|_| {
        PipelineError::Compute(format!(
            "CUDA device reported invalid max_grid_dim_y={}",
            device.max_grid_dim_y
        ))
    })?;
    let grid_x_u128 = grid_blocks.min(max_grid_x as u128).max(1);
    let grid_y_u128 = grid_blocks.div_ceil(grid_x_u128).max(1);
    let grid_x = u32::try_from(grid_x_u128).map_err(|_| {
        PipelineError::Compute(format!(
            "CUDA launch x-grid overflow for {n} elements at block size {block_size}"
        ))
    })?;
    let grid_y = u32::try_from(grid_y_u128).map_err(|_| {
        PipelineError::Compute(format!(
            "CUDA launch requires {grid_blocks} grid blocks for {n} elements at block size {block_size}, exceeding CUDA's 2D grid limit"
        ))
    })?;
    if grid_y > max_grid_y {
        return Err(PipelineError::Compute(format!(
            "CUDA launch requires grid {}x{} for {n} elements at block size {block_size}, exceeding device limit {}x{}",
            grid_x, grid_y, max_grid_x, max_grid_y
        )));
    }
    Ok(LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    })
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

fn record_cuda_sync_event(
    stream: &CudaStream,
    context: &'static str,
) -> Result<CudaEvent, PipelineError> {
    stream.record_event(None).map_err(map_driver_err(context))
}

fn record_cuda_timing_event(
    stream: &CudaStream,
    context: &'static str,
) -> Result<CudaEvent, PipelineError> {
    stream
        .record_event(Some(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(map_driver_err(context))
}

fn checked_i32(label: &str, value: usize) -> Result<i32, PipelineError> {
    i32::try_from(value).map_err(|_| {
        PipelineError::Compute(format!(
            "{label}={value} exceeds i32::MAX required by CUDA/cuBLAS APIs"
        ))
    })
}

fn checked_u64(label: &str, value: usize) -> Result<u64, PipelineError> {
    u64::try_from(value).map_err(|_| {
        PipelineError::Compute(format!(
            "{label}={value} exceeds u64::MAX required by CUDA kernel APIs"
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

    #[test]
    fn cuda_dynamic_libraries_are_preflighted_before_nvrtc_compile() {
        let source = include_str!("cuda_backend.rs");
        let runtime_new_pos = source
            .find("fn new(prep: &PreparationResult) -> Result<Self, String> {")
            .expect("CudaRuntime::new must exist");
        let after_runtime_new = &source[runtime_new_pos..];
        let preflight_pos = after_runtime_new
            .find("preflight_cuda_dynamic_libraries()?;")
            .expect("CudaRuntime::new must preflight dynamic CUDA libraries");
        let compile_pos = after_runtime_new
            .find("compile_ptx(CUDA_KERNELS)")
            .expect("CudaRuntime::new must compile CUDA kernels");

        assert!(preflight_pos < compile_pos);
    }

    #[test]
    fn cuda_preflight_uses_non_panicking_cudarc_probes() {
        let source = include_str!("cuda_backend.rs");
        let preflight_start = source
            .find("fn preflight_cuda_dynamic_libraries()")
            .expect("CUDA dynamic library preflight must exist");
        let after_preflight_start = &source[preflight_start..];
        let preflight_end = after_preflight_start
            .find("\n}\n\nimpl CudaRuntime")
            .expect("preflight function must precede CudaRuntime impl");
        let body = &after_preflight_start[..preflight_end];

        assert!(body.contains("cudarc::nvrtc::sys::is_culib_present()"));
        assert!(body.contains("cudarc::cublas::sys::is_culib_present()"));
        assert!(!body.contains("culib()"));
    }

    #[test]
    fn cuda_elapsed_time_events_are_created_with_timing_enabled() {
        let source = include_str!("cuda_backend.rs");
        let timing_helper_start = source
            .find("fn record_cuda_timing_event(")
            .expect("CUDA timing event helper must exist");
        let after_timing_helper = &source[timing_helper_start..];
        let timing_helper_end = after_timing_helper
            .find("\n}\n\nfn checked_i32")
            .expect("timing event helper must precede checked_i32");
        let timing_helper_body = &after_timing_helper[..timing_helper_end];

        assert!(
            timing_helper_body
                .contains(".record_event(Some(cuda_sys::CUevent_flags::CU_EVENT_DEFAULT))")
        );

        let elapsed_pos = source
            .find(".elapsed_ms(&tile_done_event)")
            .expect("CUDA tile elapsed timing must be recorded");
        let before_elapsed = &source[..elapsed_pos];
        assert!(before_elapsed.contains("let gpu_start_event = record_cuda_timing_event("));
        assert!(before_elapsed.contains("let tile_done_event = record_cuda_timing_event("));
    }
}
