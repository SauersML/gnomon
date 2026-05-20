//! Faithful standalone reproducer for the AoU "double free or
//! corruption (!prev)" abort that gnomon's CUDA score backend hits at
//! the end of a run.
//!
//! Earlier ctypes-only and cupy-only Python repros never aborted, even
//! when pip-wheel cuBLAS was loaded against the system kernel driver.
//! That falsifies the "pure LD_LIBRARY_PATH shadowing" theory. The
//! remaining things gnomon does that the Python repros don't:
//!
//!   * use cudarc's RAII wrappers (CudaSlice<T>::drop enqueues
//!     `cuMemFreeAsync` on the *owning stream*),
//!   * allocate pinned host buffers via `cuMemHostAlloc` and free them
//!     in cudarc's PinnedHostSlice::drop,
//!   * compile a kernel through NVRTC and launch it,
//!   * destroy cuBLAS, modules, streams, and the context in the same
//!     order CudaRuntime's field-drop sequence does.
//!
//! This binary recreates that exact structure with synthetic data so
//! we can test the suspect drop path in isolation. Build and run on
//! the box that aborts under gnomon:
//!
//!     cargo build --release --example repro_cudarc_teardown --features score
//!     ./target/release/examples/repro_cudarc_teardown
//!
//! Tunables come from CLI flags (no env vars):
//!
//!     --people <N>      rows of A in the SGEMM, default 447278
//!                       (AoU All of Us cohort).
//!     --mega <N>        cols of A / rows of B, default 256.
//!     --scores <N>      cols of B, default 1.
//!     --trials <N>      how many score chunks per run, default 1.
//!     --runs <N>        how many top-level runs to do back-to-back,
//!                       default 1.
//!     --drop-order <ord>
//!                       gnomon | reverse | strict-stream-first.
//!                       default "gnomon" (matches CudaRuntime).
//!     --skip-sync       skip CudaRuntime::drop's explicit synchronize
//!                       before letting fields drop (stress test).
//!
//! If this binary aborts on the AoU image but the Python repros don't,
//! the bug is squarely in the cudarc / kernel / pinned-memory teardown
//! path, not in cuBLAS-vs-system-driver ABI.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PinnedHostSlice, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::env;
use std::process::ExitCode;
use std::sync::Arc;

// A trivially harmless kernel — we just need to exercise the same
// nvrtc compile + module load + launch path that gnomon does. Doing a
// real operation makes sure the JIT-compiled code is actually called.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void scale_in_place(float* x, float alpha, unsigned long n) {
    unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { x[i] = x[i] * alpha + 1.0f; }
}
"#;

struct Args {
    people: usize,
    mega: usize,
    scores: usize,
    trials: usize,
    runs: usize,
    skip_sync: bool,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut a = Args {
            people: 447_278,
            mega: 256,
            scores: 1,
            trials: 1,
            runs: 1,
            skip_sync: false,
        };
        let mut it = env::args().skip(1);
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--people" => {
                    a.people = it
                        .next()
                        .ok_or("missing")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--mega" => {
                    a.mega = it
                        .next()
                        .ok_or("missing")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--scores" => {
                    a.scores = it
                        .next()
                        .ok_or("missing")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--trials" => {
                    a.trials = it
                        .next()
                        .ok_or("missing")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--runs" => {
                    a.runs = it
                        .next()
                        .ok_or("missing")?
                        .parse()
                        .map_err(|e| format!("{e}"))?
                }
                "--skip-sync" => a.skip_sync = true,
                "-h" | "--help" => {
                    println!("see repro_cudarc_teardown.rs header for flags");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag {other}")),
            }
        }
        Ok(a)
    }
}

/// Mirror of gnomon's CudaRuntime: same field order so Rust's
/// top-to-bottom drop sequence is the one we're trying to test.
///
/// Most fields are held only for their Drop side-effects (cuMemFreeAsync
/// on device slices, cuMemFreeHost on pinned slices, cublasDestroy on
/// the cuBLAS handle, etc.) — that's the point of this repro.
#[allow(dead_code)]
struct Runtime {
    ctx: Arc<CudaContext>,
    compute_stream: Arc<CudaStream>,
    copy_stream: Arc<CudaStream>,
    blas: CudaBlas,
    scale_kernel: CudaFunction,
    a_buf: CudaSlice<f32>,
    b_buf: CudaSlice<f32>,
    c_buf: CudaSlice<f32>,
    sparse_weights: CudaSlice<f32>,
    sparse_columns: CudaSlice<u32>,
    sparse_row_offsets: CudaSlice<u64>,
    output_map: CudaSlice<u32>,
    pinned_staging: Vec<PinnedHostSlice<u8>>,
    pinned_reconciled: Vec<PinnedHostSlice<u32>>,
    skip_sync: bool,
}

impl Drop for Runtime {
    fn drop(&mut self) {
        if self.skip_sync {
            return;
        }
        let _ = self.ctx.bind_to_thread();
        let _ = self.compute_stream.synchronize();
        let _ = self.copy_stream.synchronize();
    }
}

fn build_runtime(args: &Args) -> Result<Runtime, String> {
    eprintln!(
        "[repro] init: people={} mega={} scores={} trials={} runs={} skip_sync={}",
        args.people, args.mega, args.scores, args.trials, args.runs, args.skip_sync
    );

    let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext::new: {e:?}"))?;
    let copy_stream = ctx
        .new_stream()
        .map_err(|e| format!("copy_stream: {e:?}"))?;
    let compute_stream = ctx
        .new_stream()
        .map_err(|e| format!("compute_stream: {e:?}"))?;

    let ptx = compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc: {e:?}"))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| format!("load_module: {e:?}"))?;
    let scale_kernel = module
        .load_function("scale_in_place")
        .map_err(|e| format!("load_function: {e:?}"))?;

    let blas = CudaBlas::new(compute_stream.clone()).map_err(|e| format!("cublas: {e:?}"))?;

    // SGEMM shapes mirror gnomon: A (people x mega), B (mega x scores), C (people x scores).
    let a_buf = compute_stream
        .alloc_zeros::<f32>(args.people * args.mega)
        .map_err(|e| format!("alloc A: {e:?}"))?;
    let b_buf = compute_stream
        .alloc_zeros::<f32>(args.mega * args.scores)
        .map_err(|e| format!("alloc B: {e:?}"))?;
    let c_buf = compute_stream
        .alloc_zeros::<f32>(args.people * args.scores)
        .map_err(|e| format!("alloc C: {e:?}"))?;

    // Sparse / map buffers — same five static-runtime slices gnomon allocates.
    let sparse_weights = compute_stream
        .alloc_zeros::<f32>(args.mega.max(1024))
        .map_err(|e| format!("alloc sparse_weights: {e:?}"))?;
    let sparse_columns = compute_stream
        .alloc_zeros::<u32>(args.mega.max(1024))
        .map_err(|e| format!("alloc sparse_columns: {e:?}"))?;
    let sparse_row_offsets = compute_stream
        .alloc_zeros::<u64>(args.mega + 1)
        .map_err(|e| format!("alloc sparse_row_offsets: {e:?}"))?;
    let output_map = compute_stream
        .alloc_zeros::<u32>(args.people)
        .map_err(|e| format!("alloc output_map: {e:?}"))?;

    // Pinned host buffers — gnomon allocates PIPELINE_SLOTS of each.
    const PIPELINE_SLOTS: usize = 4;
    let max_packed = args.people; // bytes
    let mut pinned_staging = Vec::with_capacity(PIPELINE_SLOTS);
    let mut pinned_reconciled = Vec::with_capacity(PIPELINE_SLOTS);
    for _ in 0..PIPELINE_SLOTS {
        let p = unsafe { ctx.alloc_pinned::<u8>(max_packed) }
            .map_err(|e| format!("alloc_pinned u8: {e:?}"))?;
        pinned_staging.push(p);
        let p = unsafe { ctx.alloc_pinned::<u32>(args.mega) }
            .map_err(|e| format!("alloc_pinned u32: {e:?}"))?;
        pinned_reconciled.push(p);
    }

    Ok(Runtime {
        ctx,
        compute_stream,
        copy_stream,
        blas,
        scale_kernel,
        a_buf,
        b_buf,
        c_buf,
        sparse_weights,
        sparse_columns,
        sparse_row_offsets,
        output_map,
        pinned_staging,
        pinned_reconciled,
        skip_sync: args.skip_sync,
    })
}

fn run_workload(rt: &mut Runtime, args: &Args) -> Result<(), String> {
    for trial in 1..=args.trials {
        eprintln!(
            "[repro] trial {trial}/{}: launching scale_kernel",
            args.trials
        );
        let n = (args.people * args.mega) as u64;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        unsafe {
            rt.compute_stream
                .launch_builder(&rt.scale_kernel)
                .arg(&mut rt.a_buf)
                .arg(&1.000_001_f32)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| format!("scale_kernel launch: {e:?}"))?;
        }

        eprintln!("[repro] trial {trial}: running cuBLAS SGEMM");
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: args.people as i32,
            n: args.scores as i32,
            k: args.mega as i32,
            alpha: 1.0_f32,
            lda: args.people as i32,
            ldb: args.mega as i32,
            beta: 0.0_f32,
            ldc: args.people as i32,
        };
        unsafe {
            rt.blas
                .gemm(cfg, &rt.a_buf, &rt.b_buf, &mut rt.c_buf)
                .map_err(|e| format!("cublas gemm: {e:?}"))?;
        }

        // Pretend we're staging packed input through pinned memory →
        // device, like the real pipeline does each mega-batch.
        for slot in 0..rt.pinned_staging.len() {
            let host = &mut rt.pinned_staging[slot];
            let slice: &mut [u8] = host
                .as_mut_slice()
                .map_err(|e| format!("pinned as_mut_slice: {e:?}"))?;
            for (i, b) in slice.iter_mut().enumerate().take(1024) {
                *b = (i & 0xff) as u8;
            }
        }

        eprintln!("[repro] trial {trial}: sync compute_stream before next iter");
        rt.compute_stream
            .synchronize()
            .map_err(|e| format!("sync compute_stream: {e:?}"))?;
    }
    Ok(())
}

fn report_cuda_libs() {
    let maps = match std::fs::read_to_string("/proc/self/maps") {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut seen = std::collections::BTreeSet::new();
    for line in maps.lines() {
        let last = match line.split_whitespace().last() {
            Some(p) if p.starts_with('/') => p,
            _ => continue,
        };
        let name = std::path::Path::new(last)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if name.starts_with("libcuda")
            || name.starts_with("libcudart")
            || name.starts_with("libcublas")
            || name.starts_with("libnvrtc")
        {
            seen.insert(last.to_string());
        }
    }
    eprintln!("[repro] cuda libs loaded:");
    for p in &seen {
        let tag = if p.contains("/site-packages/nvidia/") {
            "  <-- pip-wheel"
        } else {
            ""
        };
        eprintln!("[repro]   {p}{tag}");
    }
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[repro] arg error: {e}");
            return ExitCode::from(2);
        }
    };
    report_cuda_libs();

    for run in 1..=args.runs {
        eprintln!("\n[repro] ===== run {run}/{} =====", args.runs);
        let mut rt = match build_runtime(&args) {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("[repro] build_runtime failed: {e}");
                return ExitCode::from(3);
            }
        };
        if let Err(e) = run_workload(&mut rt, &args) {
            eprintln!("[repro] workload failed: {e}");
            return ExitCode::from(4);
        }
        eprintln!("[repro] run {run}: workload complete, tearing down (gnomon field order)");
        drop(rt);
        eprintln!("[repro] run {run}: teardown survived");
    }

    eprintln!("\n[repro] all runs completed without abort");
    ExitCode::SUCCESS
}
