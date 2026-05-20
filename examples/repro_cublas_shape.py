#!/usr/bin/env python3
"""Full-pipeline reproducer for the gnomon `(!prev)` SIGABRT — no gnomon binary.

Builds and runs gnomon's two custom CUDA kernels (`unpack_plink`,
`build_batch_mats`) along with the three cuBLAS sgemms gnomon issues per
batch, against the exact shapes that PGS001320 produces on AoU:

      M = tile_scores = 1            (single-PGS case)
      N = num_people  = 447_278
      MEGA            = 2_048        (batch ceiling)
      K_LAST_BATCH    = 193          (gnomon's partial last batch)
      N_FULL_BATCHES  = 2            (PGS001320 has 4_289 = 2*2048 + 193)
      bytes_per_var   = (N + 3) // 4

This is everything the gnomon binary does on the GPU. If the bug is in
cuBLAS or one of the two kernels for this shape, this script crashes.
If this runs clean, the bug is in gnomon's Rust prep / scoring layer,
not the GPU pipeline.

Run:
    pip install 'cupy-cuda12x' numpy
    python examples/repro_cublas_shape.py
"""

from __future__ import annotations

import sys
import time

try:
    import cupy as cp
    import numpy as np
except ImportError as exc:
    print(f"cupy and numpy are required ({exc})", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Shape (matches gnomon's PGS001320 last batch on AoU)
# ---------------------------------------------------------------------------
NUM_PEOPLE = 447_278
TOTAL_PEOPLE_IN_FAM = NUM_PEOPLE   # all kept; bytes_per_variant matches
MEGA = 2_048
K_LAST_BATCH = 193
N_FULL_BATCHES = 2
TILE_SCORES = 1
NUM_SCORES = 1                       # PGS001320 alone
PIPELINE_SLOTS = 2
NUM_RECONCILED = N_FULL_BATCHES * MEGA + K_LAST_BATCH  # 4 289
BYTES_PER_VARIANT = (TOTAL_PEOPLE_IN_FAM + 3) // 4
TRIALS = 1


# ---------------------------------------------------------------------------
# CUDA kernel source — copied verbatim from gnomon's score/cuda_backend.rs
# ---------------------------------------------------------------------------
KERNEL_SRC = r"""
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
"""


def _print_shape() -> None:
    print(
        f"shape: NUM_PEOPLE={NUM_PEOPLE:,}  NUM_SCORES={NUM_SCORES}  "
        f"NUM_RECONCILED={NUM_RECONCILED:,}  MEGA={MEGA:,}  "
        f"K_LAST_BATCH={K_LAST_BATCH:,}  TILE_SCORES={TILE_SCORES}  "
        f"BYTES_PER_VARIANT={BYTES_PER_VARIANT:,}",
        flush=True,
    )


def _build_kernels():
    print("compiling kernels", flush=True)
    module = cp.RawModule(code=KERNEL_SRC, options=("--use_fast_math",))
    return module.get_function("unpack_plink"), module.get_function("build_batch_mats")


def _build_sparse_csr():
    """Build a small CSR mirroring PGS001320's shape: 4289 rows, ~all simple,
    nnz roughly equal to row count (one score column per variant)."""
    nnz = NUM_RECONCILED * NUM_SCORES
    sparse_weights = cp.random.uniform(-1.0, 1.0, size=nnz, dtype=cp.float32)
    sparse_missing_corrections = cp.random.uniform(-0.1, 0.1, size=nnz, dtype=cp.float32)
    sparse_columns = cp.zeros(nnz, dtype=cp.uint32)
    sparse_row_offsets = cp.arange(NUM_RECONCILED + 1, dtype=cp.uint64) * NUM_SCORES
    return sparse_weights, sparse_missing_corrections, sparse_columns, sparse_row_offsets


def _run_trial(trial_idx: int) -> None:
    print(f"\n----- trial {trial_idx + 1} / {TRIALS} -----", flush=True)
    start = time.time()

    unpack_kernel, build_kernel = _build_kernels()

    # --- Static buffers (CudaRuntime equivalents) ---------------------------
    print("alloc: static device buffers", flush=True)
    sparse_w, sparse_mc, sparse_cols, sparse_row_off = _build_sparse_csr()
    out_to_fam = cp.arange(NUM_PEOPLE, dtype=cp.uint32)

    # --- Per-pipeline buffers ----------------------------------------------
    print("alloc: per-pipeline device buffers", flush=True)
    d_packed_slots = [
        cp.zeros(MEGA * BYTES_PER_VARIANT, dtype=cp.uint8) for _ in range(PIPELINE_SLOTS)
    ]
    d_reconciled_slots = [
        cp.zeros(MEGA, dtype=cp.uint32) for _ in range(PIPELINE_SLOTS)
    ]
    d_dosage = cp.zeros(NUM_PEOPLE * MEGA, dtype=cp.float32)
    d_missing = cp.zeros(NUM_PEOPLE * MEGA, dtype=cp.float32)
    d_w_eff = cp.zeros(MEGA * 8, dtype=cp.float32)            # gpu_score_chunk_size=8
    d_w_corr = cp.zeros(MEGA * 8, dtype=cp.float32)
    d_count_w = cp.zeros(MEGA * 8, dtype=cp.float32)
    d_out_scores_slots = [
        cp.zeros(NUM_PEOPLE * 8, dtype=cp.float32) for _ in range(PIPELINE_SLOTS)
    ]
    d_out_corr_slots = [
        cp.zeros(NUM_PEOPLE * 8, dtype=cp.float32) for _ in range(PIPELINE_SLOTS)
    ]
    d_out_counts_slots = [
        cp.zeros(NUM_PEOPLE * 8, dtype=cp.float32) for _ in range(PIPELINE_SLOTS)
    ]

    # --- Host buffers ------------------------------------------------------
    print("alloc: pageable host result buffers", flush=True)
    host_tile_scores = [
        np.zeros(NUM_PEOPLE * 8, dtype=np.float32) for _ in range(PIPELINE_SLOTS)
    ]
    host_tile_corr = [
        np.zeros(NUM_PEOPLE * 8, dtype=np.float32) for _ in range(PIPELINE_SLOTS)
    ]
    host_tile_counts = [
        np.zeros(NUM_PEOPLE * 8, dtype=np.float32) for _ in range(PIPELINE_SLOTS)
    ]
    final_scores = np.zeros(NUM_PEOPLE * NUM_SCORES, dtype=np.float64)
    final_counts = np.zeros(NUM_PEOPLE * NUM_SCORES, dtype=np.uint32)

    # --- Batch loop --------------------------------------------------------
    batch_lens = [MEGA] * N_FULL_BATCHES + [K_LAST_BATCH]
    for i, batch_len in enumerate(batch_lens):
        slot = i % PIPELINE_SLOTS
        print(f"  batch {i + 1}/{len(batch_lens)} slot={slot} batch_len={batch_len}", flush=True)

        # Fake packed data
        cp.random.seed(i)
        d_packed_view = d_packed_slots[slot][: batch_len * BYTES_PER_VARIANT]
        d_packed_view[:] = cp.random.randint(0, 256, size=d_packed_view.size, dtype=cp.uint8)

        # Reconciled indices: variants i*MEGA..i*MEGA+batch_len
        d_reconciled_slots[slot][:batch_len] = cp.arange(
            i * MEGA, i * MEGA + batch_len, dtype=cp.uint32
        )

        # unpack_plink
        unpack_elems = NUM_PEOPLE * batch_len
        threads = 1024
        blocks = (unpack_elems + threads - 1) // threads
        unpack_kernel(
            (blocks,),
            (threads,),
            (
                d_packed_slots[slot],
                out_to_fam,
                np.int32(NUM_PEOPLE),
                np.int32(batch_len),
                np.int32(BYTES_PER_VARIANT),
                d_dosage[:unpack_elems],
                d_missing[:unpack_elems],
            ),
        )

        # build_batch_mats — loop over score-offset tiles (gnomon's
        # gpu_score_chunk_size=8 means one tile here)
        for score_offset in range(0, NUM_SCORES, 8):
            tile_scores = min(NUM_SCORES - score_offset, 8)
            weights_elems = batch_len * tile_scores
            tile_result_elems = NUM_PEOPLE * tile_scores

            build_blocks = (batch_len + threads - 1) // threads
            build_kernel(
                (build_blocks,),
                (threads,),
                (
                    sparse_w,
                    sparse_mc,
                    sparse_cols,
                    sparse_row_off,
                    d_reconciled_slots[slot][:batch_len],
                    np.int32(batch_len),
                    np.int32(tile_scores),
                    np.int32(score_offset),
                    d_w_eff[:weights_elems],
                    d_w_corr[:weights_elems],
                    d_count_w[:weights_elems],
                ),
            )

            # Three GEMMs gnomon issues per tile:
            # C = dosage @ w_eff
            a = d_dosage[:unpack_elems].reshape(NUM_PEOPLE, batch_len)
            b_eff = d_w_eff[:weights_elems].reshape(batch_len, tile_scores)
            c_out = d_out_scores_slots[slot][:tile_result_elems].reshape(NUM_PEOPLE, tile_scores)
            cp.matmul(a, b_eff, out=c_out)

            b_corr = d_w_corr[:weights_elems].reshape(batch_len, tile_scores)
            c_corr = d_out_corr_slots[slot][:tile_result_elems].reshape(NUM_PEOPLE, tile_scores)
            m = d_missing[:unpack_elems].reshape(NUM_PEOPLE, batch_len)
            cp.matmul(m, b_corr, out=c_corr)

            b_count = d_count_w[:weights_elems].reshape(batch_len, tile_scores)
            c_count = d_out_counts_slots[slot][:tile_result_elems].reshape(NUM_PEOPLE, tile_scores)
            cp.matmul(m, b_count, out=c_count)

            # memcpy_dtoh into pageable host
            host_scores_slice = host_tile_scores[slot][:tile_result_elems]
            host_corr_slice = host_tile_corr[slot][:tile_result_elems]
            host_counts_slice = host_tile_counts[slot][:tile_result_elems]
            cp.cuda.runtime.memcpyAsync(
                host_scores_slice.ctypes.data,
                d_out_scores_slots[slot].data.ptr,
                tile_result_elems * 4,
                cp.cuda.runtime.memcpyDeviceToHost,
                cp.cuda.Stream.null.ptr,
            )
            cp.cuda.runtime.memcpyAsync(
                host_corr_slice.ctypes.data,
                d_out_corr_slots[slot].data.ptr,
                tile_result_elems * 4,
                cp.cuda.runtime.memcpyDeviceToHost,
                cp.cuda.Stream.null.ptr,
            )
            cp.cuda.runtime.memcpyAsync(
                host_counts_slice.ctypes.data,
                d_out_counts_slots[slot].data.ptr,
                tile_result_elems * 4,
                cp.cuda.runtime.memcpyDeviceToHost,
                cp.cuda.Stream.null.ptr,
            )
            cp.cuda.runtime.deviceSynchronize()

            # Host-side accumulate — mirror cuda_backend.rs lines 1473-1483
            for person in range(NUM_PEOPLE):
                src_base = person * tile_scores
                dst_base = person * NUM_SCORES + score_offset
                for j in range(tile_scores):
                    final_scores[dst_base + j] += float(host_scores_slice[src_base + j])
                    final_scores[dst_base + j] += float(host_corr_slice[src_base + j])
                    final_counts[dst_base + j] += int(round(host_counts_slice[src_base + j]))
                if person >= 1000:
                    break  # don't iterate 447k people each batch in pure Python

    cp.cuda.runtime.deviceSynchronize()
    print("  syncing complete", flush=True)
    print(f"  final_scores[:4]={final_scores[:4]}", flush=True)

    # --- Teardown — mirror CudaRuntime drop order ---------------------------
    print("  freeing buffers (mirror CudaRuntime drop)", flush=True)
    del host_tile_counts, host_tile_corr, host_tile_scores
    del d_out_counts_slots, d_out_corr_slots, d_out_scores_slots
    del d_count_w, d_w_corr, d_w_eff
    del d_missing, d_dosage
    del d_reconciled_slots, d_packed_slots
    del out_to_fam
    del sparse_row_off, sparse_cols, sparse_mc, sparse_w
    cp.get_default_memory_pool().free_all_blocks()

    elapsed = time.time() - start
    print(f"  trial {trial_idx + 1} ok ({elapsed:.1f}s)", flush=True)


def main() -> int:
    _print_shape()
    for trial in range(TRIALS):
        _run_trial(trial)
    print(
        "\nNo abort observed. The GPU pipeline (kernels + cuBLAS) is "
        "not the source of the heap corruption — bug lives in gnomon's "
        "Rust prep / score path.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
