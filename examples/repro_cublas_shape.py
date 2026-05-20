#!/usr/bin/env python3
"""Pure-Python reproducer for the gnomon `(!prev)` SIGABRT — no gnomon binary.

The bug fires deterministically on `PGS001320` (PRIVÉ 2022 sparse PRS,
4 289 overlapping variants on AoU). Under the cuda backend gnomon
issues this exact GEMM sequence on its final batch:

      M = tile_scores   = 1
      N = num_people    = 447 278
      K = batch_len     = 193

That's `m=1` cuBLAS sgemm with a tiny k — effectively gemv-in-disguise.
cuBLAS dispatches a different kernel for this shape; if its internal
workspace sizing has a corner-case bug, it scribbles past the workspace
into the host glibc arena, and glibc reports it as
"double free or corruption (!prev)" the next time anything frees memory.

This script reproduces the same GEMM shape via cupy (which calls into
the same libcublas the gnomon binary links). It also issues the same
three GEMMs per batch and the same memcpy-dtoh-into-pageable-host that
cuda_backend.rs does, so it mirrors gnomon's heap-layout pressure.

Usage
-----

    pip install 'cupy-cuda12x' numpy
    python examples/repro_cublas_shape.py

Sweep knobs (no env vars — edit the constants up top or pass via
command line if you've extended argparse):

  * K_LAST_BATCH    last-batch column count
  * TILE_SCORES     number of score columns (gnomon: 1..N)
  * NUM_PEOPLE      AoU is 447 278
  * MEGA            full-batch ceiling
  * N_FULL_BATCHES  how many full mega-batches precede the partial one
  * TRIALS          repeat the whole sequence N times to expose flakes

If the script aborts with `(!prev)` or any other glibc message, the bug
is in cuBLAS for this shape (or in cupy's host alloc, which is the same
glibc arena). If it runs clean, the bug is elsewhere in gnomon's
pipeline — try varying `K_LAST_BATCH` and `TILE_SCORES` first.
"""

from __future__ import annotations

import sys
import time

try:
    import cupy as cp
    import numpy as np
except ImportError as exc:
    print(f"cupy and numpy are required for this repro ({exc})", file=sys.stderr)
    print("  pip install 'cupy-cuda12x' numpy", file=sys.stderr)
    sys.exit(2)


# Shape knobs. Defaults match the failing PGS001320 last batch on AoU.
NUM_PEOPLE = 447_278
K_LAST_BATCH = 193
TILE_SCORES = 1
MEGA = 2048
N_FULL_BATCHES = 2
PIPELINE_SLOTS = 2
TRIALS = 1


def _print_shape() -> None:
    print(
        f"shape: NUM_PEOPLE={NUM_PEOPLE:,}  MEGA={MEGA:,}  K_LAST_BATCH={K_LAST_BATCH:,}  "
        f"TILE_SCORES={TILE_SCORES}  FULL_BATCHES={N_FULL_BATCHES}  TRIALS={TRIALS}",
        flush=True,
    )


def _alloc_buffers():
    """Allocate the same device + host buffers that gnomon's CudaRuntime
    and process_dense_stream_cuda allocate."""
    print("alloc: device buffers", flush=True)
    d_dosage = cp.zeros((NUM_PEOPLE, MEGA), dtype=cp.float32)         # ~3.6 GB
    d_w_eff = cp.zeros((MEGA, TILE_SCORES), dtype=cp.float32)
    d_w_corr = cp.zeros((MEGA, TILE_SCORES), dtype=cp.float32)
    d_count_w = cp.zeros((MEGA, TILE_SCORES), dtype=cp.float32)
    d_out_slots = [
        cp.zeros((NUM_PEOPLE, TILE_SCORES), dtype=cp.float32)
        for _ in range(PIPELINE_SLOTS)
    ]

    # Pinned-host staging mirrors `pinned_staging` in cuda_backend.rs.
    print("alloc: pinned-host buffers", flush=True)
    bytes_per_variant = (NUM_PEOPLE + 3) // 4
    pinned_staging = [
        cp.cuda.alloc_pinned_memory(MEGA * bytes_per_variant) for _ in range(PIPELINE_SLOTS)
    ]
    pinned_reconciled = [
        cp.cuda.alloc_pinned_memory(MEGA * 4) for _ in range(PIPELINE_SLOTS)
    ]

    # Pageable host result buffers (same as gnomon's host_tile_*_slots).
    host_tile_slots = [
        np.zeros(NUM_PEOPLE * TILE_SCORES, dtype=np.float32)
        for _ in range(PIPELINE_SLOTS)
    ]

    return {
        "d_dosage": d_dosage,
        "d_w_eff": d_w_eff,
        "d_w_corr": d_w_corr,
        "d_count_w": d_count_w,
        "d_out_slots": d_out_slots,
        "pinned_staging": pinned_staging,
        "pinned_reconciled": pinned_reconciled,
        "host_tile_slots": host_tile_slots,
    }


def _run_batch(state, slot: int, batch_len: int) -> None:
    """Mirror run_pending_compute_cuda for one batch: three GEMMs +
    one memcpy_dtoh into a pageable host buffer."""
    d_dosage = state["d_dosage"][:, :batch_len]                  # [N, K]
    d_w_eff = state["d_w_eff"][:batch_len, :]                    # [K, M]
    d_w_corr = state["d_w_corr"][:batch_len, :]                  # [K, M]
    d_count_w = state["d_count_w"][:batch_len, :]                # [K, M]
    d_out = state["d_out_slots"][slot]                           # [N, M]

    # Three identical-shape GEMMs gnomon issues per batch. Each writes
    # C = A @ B into d_out (beta=0 overwrite). cupy.matmul ends up in
    # cuBLAS sgemm — same kernel selector gnomon hits.
    cp.matmul(d_dosage, d_w_eff, out=d_out)
    cp.matmul(d_dosage, d_w_corr, out=d_out)
    cp.matmul(d_dosage, d_count_w, out=d_out)

    # Copy result to pageable host vec, exactly like cuda_backend.rs.
    host_buf = state["host_tile_slots"][slot]
    d_out_flat = d_out.reshape(-1)
    cp.cuda.runtime.memcpyAsync(
        host_buf.ctypes.data,
        d_out_flat.data.ptr,
        d_out_flat.nbytes,
        cp.cuda.runtime.memcpyDeviceToHost,
        cp.cuda.Stream.null.ptr,
    )


def _run_trial(trial_idx: int) -> None:
    print(f"\n----- trial {trial_idx + 1} / {TRIALS} -----", flush=True)
    start = time.time()

    state = _alloc_buffers()
    cp.cuda.runtime.deviceSynchronize()

    batches = [MEGA] * N_FULL_BATCHES + [K_LAST_BATCH]
    for i, batch_len in enumerate(batches):
        slot = i % PIPELINE_SLOTS
        print(
            f"  batch {i + 1}/{len(batches)} slot={slot} batch_len={batch_len}",
            flush=True,
        )
        _run_batch(state, slot, batch_len)

    print("  syncing streams", flush=True)
    cp.cuda.runtime.deviceSynchronize()

    print(
        f"  host_tile_slots[1][:4]={state['host_tile_slots'][1][:4]}",
        flush=True,
    )

    print("  freeing buffers (mirror CudaRuntime drop)", flush=True)
    del state["host_tile_slots"]
    del state["pinned_reconciled"]
    del state["pinned_staging"]
    del state["d_out_slots"]
    del state["d_count_w"]
    del state["d_w_corr"]
    del state["d_w_eff"]
    del state["d_dosage"]
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    elapsed = time.time() - start
    print(f"  trial {trial_idx + 1} ok ({elapsed:.1f}s)", flush=True)


def main() -> int:
    _print_shape()
    for trial in range(TRIALS):
        _run_trial(trial)
    print(
        "\nNo abort observed. Either the bug is not in cuBLAS for this "
        "shape, or this driver/cuBLAS version is unaffected.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
