#!/usr/bin/env python3
"""Minimal CUDA/cuBLAS loader/teardown reproducer without using gnomon.

This models the failing mechanism from `gnomon score PGS001320` directly:

1. Create a CUDA context, one stream, and one cuBLAS handle bound to it.
2. Allocate the same class of large per-pipeline buffers with cuMemAllocAsync.
3. Run three cuBLAS SGEMMs using the single-score PGS001320 shape.
4. Synchronize the stream and print a 100% progress marker.
5. Run isolated child processes for both CUDA userspace-library choices:
   - system: libcublas resolved from the host loader path.
   - wheel: pip-wheel CUDA libs loaded by absolute path from `nvidia-*` wheels.
6. In each child, run two teardown orders:
   - fixed: enqueue cuMemFreeAsync for local buffers, then synchronize those frees.
   - bad: synchronize first, enqueue cuMemFreeAsync, then immediately destroy cuBLAS.

If the issue is the pip-wheel/system CUDA library mix, the wheel/bad child is
the small mechanism-level repro.

Run on the AoU CUDA host:

    python3 examples/repro_cuda_async_free_teardown.py
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
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
INTERNAL_CHILD_ARG = "__gnomon_cuda_teardown_child__"


def _nvidia_roots() -> list[Path]:
    roots: list[Path] = []
    home = Path.home()
    archive_patterns = [
        home / "aou-gpu-baremetal/gnomon_runtime/biobank/uv/archive-v0",
        home / ".cache/uv/archive-v0",
    ]
    for archive_root in archive_patterns:
        if not archive_root.exists():
            continue
        archive_roots = list(archive_root.glob("*/lib/python*/site-packages/nvidia"))
        archive_roots.sort(
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )
        roots.extend(archive_roots)

    try:
        import nvidia  # type: ignore[import-not-found]
    except Exception:
        pass
    else:
        roots.extend(Path(p) for p in nvidia.__path__)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _nvidia_lib_path(root: Path, component: str, soname: str) -> str | None:
    path = root / component / "lib" / soname
    if path.exists():
        return str(path)
    return None


def _nvidia_lib_dirs(root: Path) -> list[str]:
    # Match the biobank wrapper's pip-wheel CUDA directory order.
    components = [
        "cuda_runtime",
        "cusolver",
        "cusparse",
        "nvjitlink",
        "cublas",
        "curand",
        "cuda_nvrtc",
    ]
    dirs: list[str] = []
    for component in components:
        lib_dir = root / component / "lib"
        if lib_dir.is_dir():
            dirs.append(str(lib_dir))
    return dirs


def load_cuda() -> ctypes.CDLL:
    return ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)


def load_cublas_system() -> tuple[ctypes.CDLL, str]:
    candidates = [
        ctypes.util.find_library("cublas"),
        "libcublas.so.12",
        "libcublas.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL), candidate
        except OSError:
            pass
    raise OSError("could not load libcublas")


def load_cublas_wheel(root: Path) -> tuple[ctypes.CDLL, str]:
    libs = [
        ("nvjitlink", "libnvJitLink.so.12"),
        ("cuda_runtime", "libcudart.so.12"),
        ("cublas", "libcublasLt.so.12"),
        ("cublas", "libcublas.so.12"),
    ]
    loaded: list[tuple[str, str]] = []
    for component, soname in libs:
        path = _nvidia_lib_path(root, component, soname)
        if path is None:
            raise OSError(f"could not find pip-wheel CUDA library {component}/{soname} in {root}")
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        loaded.append((soname, path))
    return ctypes.CDLL(loaded[-1][1], mode=ctypes.RTLD_GLOBAL), loaded[-1][1]


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


def run_case(loader: str, mode: str, root_arg: str | None = None) -> None:
    cuda = load_cuda()
    if loader == "wheel":
        if root_arg is None:
            raise RuntimeError("wheel loader requires a root path")
        root = Path(root_arg)
        cublas, cublas_path = load_cublas_wheel(root)
        loader_label = f"wheel:{root}"
    elif loader == "system":
        cublas, cublas_path = load_cublas_system()
        loader_label = "system"
    else:
        raise RuntimeError(f"unknown loader mode: {loader}")
    configure_symbols(cuda, cublas)

    print(
        f"\n=== {loader_label} libs / {mode} teardown ===\n"
        f"cublas: {cublas_path}\n"
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


def child_main(loader: str, mode: str, root_arg: str | None = None) -> int:
    run_case(loader, mode, root_arg)
    return 0


def run_child(loader: str, mode: str, root: Path | None = None) -> int:
    cmd = [sys.executable, __file__, INTERNAL_CHILD_ARG, loader, mode]
    label = loader
    env = os.environ.copy()
    if root is not None:
        cmd.append(str(root))
        label = f"{loader}:{root}"
        lib_dirs = _nvidia_lib_dirs(root)
        prior_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*lib_dirs, prior_ld])
    print(f"\n##### child: {label} libs / {mode} teardown #####", flush=True)
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent, env=env)
    if result.returncode < 0:
        print(
            f"child {label}/{mode} died from signal {-result.returncode}",
            flush=True,
        )
    else:
        print(f"child {label}/{mode} exit={result.returncode}", flush=True)
    return result.returncode


def main() -> int:
    if len(sys.argv) in (4, 5) and sys.argv[1] == INTERNAL_CHILD_ARG:
        root_arg = sys.argv[4] if len(sys.argv) == 5 else None
        return child_main(sys.argv[2], sys.argv[3], root_arg)
    if len(sys.argv) != 1:
        print("this script takes no arguments", file=sys.stderr)
        return 2

    roots = _nvidia_roots()
    print("discovered CUDA wheel roots:", flush=True)
    for root in roots:
        print(f"  {root}", flush=True)

    cases: list[tuple[str, str, Path | None]] = [
        ("system", "fixed", None),
        ("system", "bad", None),
    ]
    for root in roots:
        cases.append(("wheel", "fixed", root))
        cases.append(("wheel", "bad", root))

    failures = 0
    for loader, mode, root in cases:
        rc = run_child(loader, mode, root)
        if rc != 0:
            failures += 1

    if failures:
        print(f"\n{failures} child run(s) failed or aborted", flush=True)
        return 1
    print("\nall child runs completed without abort", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        raise
