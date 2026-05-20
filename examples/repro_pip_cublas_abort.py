"""
Minimal reproducer for the AoU CUDA heap-corruption abort, with no
gnomon involved.

Hypothesis under test:
    Putting the `nvidia-*-cu12` pip-wheel CUDA libraries ahead of the
    system CUDA stack on LD_LIBRARY_PATH causes heap corruption at the
    end of a cuBLAS workflow, because pip's cuBLAS/cudart are a
    different patch version than the system libcuda the kernel driver
    is pinned to. Symptom: "double free or corruption (!prev)" at
    teardown.

How this script tests it:
    Use ctypes.CDLL with the SONAMEs cudarc would dlopen, so the
    runtime linker picks whichever copy LD_LIBRARY_PATH points at first
    — same resolution path as the gnomon release binary.

    Then drive a non-trivial cuBLAS SGEMM, sync, destroy the handle,
    free device memory, and exit. If the hypothesis is right, the
    "wheels-first" invocation aborts here; the "system-first" one runs
    clean.

Run via examples/repro_pip_cublas_abort.sh (which sets LD_LIBRARY_PATH
correctly for each variant). Running this script directly only exercises
whichever order the current environment already has.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

LABEL = sys.argv[1] if len(sys.argv) > 1 else "default"

# -----------------------------------------------------------------------------
# Show which CUDA libs the loader actually picked. Same logic gnomon now
# prints from `report_loaded_cuda_libraries()`.
# -----------------------------------------------------------------------------
def show_loaded_cuda_libs(stage: str) -> None:
    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError:
        return
    seen: set[str] = set()
    print(f"[{LABEL}] cuda libs loaded ({stage}):")
    for line in maps.splitlines():
        parts = line.split()
        if not parts:
            continue
        path = parts[-1]
        if not path.startswith("/"):
            continue
        name = Path(path).name
        if not any(
            name.startswith(prefix)
            for prefix in ("libcuda", "libcudart", "libcublas", "libnvrtc")
        ):
            continue
        if path in seen:
            continue
        seen.add(path)
        flag = " <-- pip-wheel" if "/site-packages/nvidia/" in path else ""
        print(f"[{LABEL}]   {path}{flag}")


# -----------------------------------------------------------------------------
# Load libcuda + libcudart + libcublas using their SONAMEs. The loader
# walks LD_LIBRARY_PATH → /etc/ld.so.cache → default paths in that
# order, so whichever copy is first wins.
# -----------------------------------------------------------------------------
libcuda = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
libcudart = ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
libcublas = ctypes.CDLL("libcublas.so.12", mode=ctypes.RTLD_GLOBAL)

show_loaded_cuda_libs("after dlopen")


def check_cuda(rc: int, where: str) -> None:
    if rc != 0:
        # cudaGetErrorString returns a char* with a human-readable name.
        libcudart.cudaGetErrorString.restype = ctypes.c_char_p
        msg = libcudart.cudaGetErrorString(rc) or b"?"
        raise RuntimeError(f"{where}: cuda rc={rc} ({msg.decode()})")


def check_cublas(rc: int, where: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{where}: cublas status={rc}")


# Bind the handful of entry points we use.
libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
libcudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
libcudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
libcudart.cudaFree.argtypes = [ctypes.c_void_p]
libcudart.cudaDeviceSynchronize.argtypes = []

libcublas.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
libcublas.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
libcublas.cublasSgemm_v2.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # transA
    ctypes.c_int,     # transB
    ctypes.c_int, ctypes.c_int, ctypes.c_int,  # m, n, k
    ctypes.POINTER(ctypes.c_float),  # alpha
    ctypes.c_void_p, ctypes.c_int,   # A, lda
    ctypes.c_void_p, ctypes.c_int,   # B, ldb
    ctypes.POINTER(ctypes.c_float),  # beta
    ctypes.c_void_p, ctypes.c_int,   # C, ldc
]

CUBLAS_OP_N = 0
M = N = K = 4096
ELEM = 4  # float32

check_cuda(libcudart.cudaSetDevice(0), "cudaSetDevice")
print(f"[{LABEL}] device 0 selected")

# Allocate A, B, C.
def alloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    check_cuda(libcudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    check_cuda(libcudart.cudaMemset(ptr, 1, nbytes), "cudaMemset")
    return ptr

bytes_per = M * N * ELEM
a = alloc(bytes_per)
b = alloc(bytes_per)
c = alloc(bytes_per)
print(f"[{LABEL}] allocated 3x{bytes_per/1e6:.1f} MB on device")

handle = ctypes.c_void_p()
check_cublas(libcublas.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate")
print(f"[{LABEL}] cublas handle created")

alpha = ctypes.c_float(1.0)
beta = ctypes.c_float(0.0)

# Run a bunch of GEMMs so cuBLAS picks its largest workspace kernel
# (the same code path that aborts in gnomon).
for i in range(40):
    rc = libcublas.cublasSgemm_v2(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        ctypes.byref(alpha),
        a, M,
        b, K,
        ctypes.byref(beta),
        c, M,
    )
    check_cublas(rc, f"cublasSgemm iter={i}")

check_cuda(libcudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")
print(f"[{LABEL}] 40 SGEMMs complete, syncing")

# Teardown — this is where the abort lands in gnomon.
check_cublas(libcublas.cublasDestroy_v2(handle), "cublasDestroy")
print(f"[{LABEL}] cublas handle destroyed")

check_cuda(libcudart.cudaFree(a), "cudaFree(a)")
check_cuda(libcudart.cudaFree(b), "cudaFree(b)")
check_cuda(libcudart.cudaFree(c), "cudaFree(c)")
print(f"[{LABEL}] device buffers freed")

show_loaded_cuda_libs("at exit")
print(f"[{LABEL}] done — exiting cleanly")
