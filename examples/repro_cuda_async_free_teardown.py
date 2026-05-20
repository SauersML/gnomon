#!/usr/bin/env python3
"""Minimal CUDA/cuBLAS teardown reproducer without using gnomon.

This models the failing mechanism from `gnomon score PGS001320` directly:

1. Create a CUDA context, one stream, and one cuBLAS handle bound to it.
2. Allocate the same class of large per-pipeline buffers with cuMemAllocAsync.
3. Run three cuBLAS SGEMMs using the single-score PGS001320 shape.
4. Synchronize the stream and print a 100% progress marker.
5. Run two teardown orders in one process:
   - fixed: enqueue cuMemFreeAsync for local buffers, then synchronize those frees.
   - bad: synchronize first, enqueue cuMemFreeAsync, then immediately destroy cuBLAS.

If the issue is the CUDA/cuBLAS/cudarc async-free teardown path, the fixed pass
should complete and the bad pass is the small mechanism-level repro.

Run on the AoU CUDA host:

    python3 examples/repro_cuda_async_free_teardown.py
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path


CUDA_SUCCESS = 0
CUBLAS_STATUS_SUCCESS = 0
CU_STREAM_NON_BLOCKING = 1
CUBLAS_OP_N = 0
PEOPLE = 447_278
MEGA = 256
TILE_SCORES = 1
TRIALS = 1


def _nvidia_lib_path(component: str, soname: str) -> str | None:
    try:
        import nvidia  # type: ignore[import-not-found]
    except Exception:
        return None

    for parent in nvidia.__path__:
        path = Path(parent) / component / "lib" / soname
        if path.exists():
            return str(path)
    return None


def load_cuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)


def load_cublas() -> ctypes.CDLL:
    candidates = [
        ctypes.util.find_library("cublas"),
        _nvidia_lib_path("cublas", "libcublas.so.12"),
        "libcublas.so.12",
        "libcublas.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
    raise OSError("could not load libcublas")


def configure_symbols(cuda: ctypes.CDLL, cublas: ctypes.CDLL) -> None:
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int

    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int

    cuda.cuCtxCreate_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    cuda.cuCtxCreate_v2.restype = ctypes.c_int

    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxSetCurrent.restype = ctypes.c_int

    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.restype = ctypes.c_int

    cuda.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    cuda.cuStreamCreate.restype = ctypes.c_int

    cuda.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
    cuda.cuStreamSynchronize.restype = ctypes.c_int

    cuda.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
    cuda.cuStreamDestroy_v2.restype = ctypes.c_int

    cuda.cuMemAllocAsync.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    cuda.cuMemAllocAsync.restype = ctypes.c_int

    cuda.cuMemFreeAsync.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p]
    cuda.cuMemFreeAsync.restype = ctypes.c_int

    cuda.cuMemsetD8Async.argtypes = [
        ctypes.c_ulonglong,
        ctypes.c_ubyte,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    cuda.cuMemsetD8Async.restype = ctypes.c_int

    cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublas.cublasCreate_v2.restype = ctypes.c_int

    cublas.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cublas.cublasSetStream_v2.restype = ctypes.c_int

    cublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublas.cublasDestroy_v2.restype = ctypes.c_int

    cublas.cublasSgemm_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    cublas.cublasSgemm_v2.restype = ctypes.c_int


def check_cuda(code: int, what: str) -> None:
    if code != CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed with CUresult={code}")


def check_cublas(code: int, what: str) -> None:
    if code != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"{what} failed with cublasStatus={code}")


class DeviceAllocs:
    def __init__(self, cuda: ctypes.CDLL, stream: ctypes.c_void_p) -> None:
        self.cuda = cuda
        self.stream = stream
        self.ptrs: list[tuple[str, ctypes.c_ulonglong, int]] = []

    def alloc_zeroed(self, name: str, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_ulonglong(0)
        check_cuda(
            self.cuda.cuMemAllocAsync(ctypes.byref(ptr), nbytes, self.stream),
            f"cuMemAllocAsync({name}, {nbytes})",
        )
        check_cuda(
            self.cuda.cuMemsetD8Async(ptr, 0, nbytes, self.stream),
            f"cuMemsetD8Async({name})",
        )
        self.ptrs.append((name, ptr, nbytes))
        return ctypes.c_void_p(ptr.value)

    def free_async_all(self) -> None:
        while self.ptrs:
            name, ptr, _nbytes = self.ptrs.pop()
            check_cuda(self.cuda.cuMemFreeAsync(ptr, self.stream), f"cuMemFreeAsync({name})")


def run_gemm_triplet(
    cublas: ctypes.CDLL,
    handle: ctypes.c_void_p,
    people: int,
    mega: int,
    tile_scores: int,
    dosage: ctypes.c_void_p,
    missing: ctypes.c_void_p,
    w_eff: ctypes.c_void_p,
    w_corr: ctypes.c_void_p,
    w_count: ctypes.c_void_p,
    out_score: ctypes.c_void_p,
    out_corr: ctypes.c_void_p,
    out_count: ctypes.c_void_p,
) -> None:
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # Same column-major cuBLAS interpretation as gnomon's row-major wrapper:
    # C(tile_scores x people) = B(tile_scores x mega) * A(mega x people).
    for label, left, right, out in [
        ("score", w_eff, dosage, out_score),
        ("correction", w_corr, missing, out_corr),
        ("count", w_count, missing, out_count),
    ]:
        check_cublas(
            cublas.cublasSgemm_v2(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                tile_scores,
                people,
                mega,
                ctypes.byref(alpha),
                left,
                tile_scores,
                right,
                mega,
                ctypes.byref(beta),
                out,
                tile_scores,
            ),
            f"cublasSgemm({label})",
        )


def run_case(mode: str) -> None:
    cuda = load_cuda()
    cublas = load_cublas()
    configure_symbols(cuda, cublas)

    print(
        f"\n=== {mode} teardown ===\n"
        f"shape: people={PEOPLE} mega={MEGA} tile_scores={TILE_SCORES} trials={TRIALS}",
        flush=True,
    )

    check_cuda(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int(0)
    check_cuda(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check_cuda(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, device), "cuCtxCreate_v2")
    check_cuda(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    stream = ctypes.c_void_p()
    check_cuda(
        cuda.cuStreamCreate(ctypes.byref(stream), CU_STREAM_NON_BLOCKING),
        "cuStreamCreate",
    )

    handle = ctypes.c_void_p()
    check_cublas(cublas.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate_v2")
    check_cublas(cublas.cublasSetStream_v2(handle, stream), "cublasSetStream_v2")

    static_allocs = DeviceAllocs(cuda, stream)
    static_allocs.alloc_zeroed("runtime_static_output_map", PEOPLE * 4)
    static_allocs.alloc_zeroed("runtime_static_sparse_offsets", (MEGA + 1) * 8)

    try:
        for trial in range(TRIALS):
            print(f"trial {trial + 1}/{TRIALS}: allocate local pipeline buffers", flush=True)
            local = DeviceAllocs(cuda, stream)
            dosage = local.alloc_zeroed("d_dosage", PEOPLE * MEGA * 4)
            missing = local.alloc_zeroed("d_missing", PEOPLE * MEGA * 4)
            w_eff = local.alloc_zeroed("d_w_eff", MEGA * TILE_SCORES * 4)
            w_corr = local.alloc_zeroed("d_w_corr", MEGA * TILE_SCORES * 4)
            w_count = local.alloc_zeroed("d_count_w", MEGA * TILE_SCORES * 4)

            # gnomon allocates output slots for gpu_score_chunk_size=8 even when
            # tile_scores is 1. Keep that allocation shape because it contributes
            # to the async-free queue pressure.
            output_slot_scores = max(TILE_SCORES, 8)
            out_score = local.alloc_zeroed("d_out_scores", PEOPLE * output_slot_scores * 4)
            out_corr = local.alloc_zeroed("d_out_corr", PEOPLE * output_slot_scores * 4)
            out_count = local.alloc_zeroed("d_out_counts", PEOPLE * output_slot_scores * 4)

            print("trial: run 3 cuBLAS SGEMMs", flush=True)
            run_gemm_triplet(
                cublas,
                handle,
                PEOPLE,
                MEGA,
                TILE_SCORES,
                dosage,
                missing,
                w_eff,
                w_corr,
                w_count,
                out_score,
                out_corr,
                out_count,
            )

            print("trial: synchronize stream before reporting 100%", flush=True)
            check_cuda(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize(before progress)")
            print("CUDA progress: 100.0%", flush=True)

            if mode == "fixed":
                print("fixed mode: enqueue local async frees, then synchronize them", flush=True)
                local.free_async_all()
                check_cuda(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize(after local frees)")
            else:
                print("bad mode: enqueue local async frees after sync, do not drain yet", flush=True)
                local.free_async_all()

        print("teardown: destroy cuBLAS handle", flush=True)
        check_cublas(cublas.cublasDestroy_v2(handle), "cublasDestroy_v2")
        handle = ctypes.c_void_p()

        print("teardown: free static runtime buffers", flush=True)
        static_allocs.free_async_all()
        check_cuda(cuda.cuStreamSynchronize(stream), "cuStreamSynchronize(final)")

        print("teardown: destroy stream/context", flush=True)
        check_cuda(cuda.cuStreamDestroy_v2(stream), "cuStreamDestroy_v2")
        stream = ctypes.c_void_p()
        check_cuda(cuda.cuCtxDestroy_v2(ctx), "cuCtxDestroy_v2")
        ctx = ctypes.c_void_p()
    finally:
        if handle:
            cublas.cublasDestroy_v2(handle)
        if stream:
            cuda.cuStreamDestroy_v2(stream)
        if ctx:
            cuda.cuCtxDestroy_v2(ctx)

    print(f"{mode} teardown completed without abort", flush=True)


def main() -> int:
    run_case("fixed")
    run_case("bad")
    print("\nall teardown modes completed without abort", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        raise
