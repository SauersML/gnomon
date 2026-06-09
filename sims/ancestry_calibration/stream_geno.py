"""Streaming genotype simulation + incremental PLINK1 .bed writing.

Lets the real-P+T pipeline reach ~100 Mb (20x5Mb) WITHOUT ever holding the full
dense dosage matrix in RAM (which is ~600-700 GB for grid2d @100Mb at full n).

Per 5 Mb chunk we:
  - simulate ancestry+mutations,
  - append every variant to the .bed (SNP-major, vectorised 2-bit packing),
  - write its .bim line,
  - reservoir-sample PCA-candidate columns (MAF>=0.05) and causal-candidate
    columns (MAF>=0.01), retaining ONLY those few-thousand dense columns in RAM.

So peak RAM is ~ (nInd x (n_pca_keep + n_causal_keep)) float32 + one chunk's dense
genotypes, not the whole genome.

The .bed encoding matches PLINK1: 2 bits/sample, LSB-first, 4 samples/byte,
code 0b00=hom-A1(2 alt), 0b10=het, 0b11=hom-A2(0 alt), 0b01=missing; A1=alt.
"""
from __future__ import annotations

import gc
import numpy as np
import msprime


# index by alt-dosage 0,1,2 -> PLINK1 2-bit code
_CODE_MAP = np.array([0b11, 0b10, 0b00], dtype=np.uint8)


def _write_bed_block(dos_block: np.ndarray, fh, col_batch: int = 4096) -> None:
    """Stream a (n_ind, n_var) alt-dosage block to an open .bed file handle in
    SNP-major PLINK1 2-bit packing, processing VAR columns in small batches so the
    transient allocation is bounded (avoids the full int64/uint8 copies of the whole
    block, which for grid2d would be ~20 GB and was the memory blow-up)."""
    n_ind, n_var = dos_block.shape
    n_bytes = (n_ind + 3) // 4
    pad = n_bytes * 4 - n_ind
    within = (np.arange(n_bytes * 4) & 3).astype(np.uint8)
    shifts_full = (within * 2)                               # (n_bytes*4,)
    for c0 in range(0, n_var, col_batch):
        c1 = min(c0 + col_batch, n_var)
        sub = dos_block[:, c0:c1]                            # (n_ind, b) float32 view
        codes = _CODE_MAP[np.clip(sub, 0, 2).astype(np.uint8)]  # (n_ind, b) uint8
        if pad:
            codes = np.vstack([codes, np.zeros((pad, codes.shape[1]), dtype=np.uint8)])
        b = codes.shape[1]
        codes = codes.reshape(n_bytes, 4, b)
        packed = np.bitwise_or.reduce(codes << (np.arange(4, dtype=np.uint8) * 2).reshape(1, 4, 1), axis=1)
        fh.write(packed.T.reshape(-1).tobytes())             # var-major: b vars × n_bytes
        del sub, codes, packed
    return


def _pack_bed_block(dos_block: np.ndarray) -> np.ndarray:
    """In-memory variant used only by the unit test; same encoding as _write_bed_block."""
    n_ind, n_var = dos_block.shape
    n_bytes = (n_ind + 3) // 4
    codes = _CODE_MAP[np.clip(dos_block, 0, 2).astype(np.uint8)]
    pad = n_bytes * 4 - n_ind
    if pad:
        codes = np.vstack([codes, np.zeros((pad, n_var), dtype=np.uint8)])
    codes = codes.reshape(n_bytes, 4, n_var)
    shifts = (np.arange(4, dtype=np.uint8) * 2).reshape(1, 4, 1)
    packed = np.bitwise_or.reduce(codes << shifts, axis=1)   # (n_bytes, n_var)
    # .bed is variant-major: for each variant, its n_bytes sample-bytes contiguously
    return packed.T.reshape(-1)                              # (n_var * n_bytes,)


def simulate_stream(dG, samples, n_chunks, chunk_bp, recomb, mut, seed,
                    prefix, n_pca_keep, n_causal_keep, log=print):
    """Stream-simulate the genome, writing <prefix>.bed/.bim/.fam incrementally.

    Returns dict with: n_ind, n_loci, maf (n_loci,), Xpca (n_ind, kept_pca),
    Xcausal (n_ind, kept_causal), pca_global_idx, causal_global_idx (indices into
    the genome-wide variant order), and snp_ids list aligned to the .bim.
    """
    rng_pca = np.random.default_rng(7001 + seed)
    rng_cau = np.random.default_rng(8001 + seed)

    n_ind = None
    bim = open(f"{prefix}.bim", "w")
    bed = open(f"{prefix}.bed", "wb")
    bed.write(bytes([0x6c, 0x1b, 0x01]))  # magic + SNP-major

    maf_parts = []
    # reservoirs: store (global_idx, column) keeping dense columns
    pca_res_cols, pca_res_idx, pca_seen = [], [], 0
    cau_res_cols, cau_res_idx, cau_seen = [], [], 0
    gidx = 0  # running genome-wide variant index

    for ci in range(n_chunks):
        ts = msprime.sim_ancestry(samples=samples, demography=dG,
                                  sequence_length=float(chunk_bp), recombination_rate=recomb,
                                  ploidy=2, random_seed=101 + seed * 131 + ci * 7)
        ts = msprime.sim_mutations(ts, rate=mut, random_seed=102 + seed * 131 + ci * 7)
        g = ts.genotype_matrix()
        if g.shape[0] == 0:
            log(f"chunk {ci+1}/{n_chunks}: 0 sites")
            continue
        d = (g[:, 0::2] + g[:, 1::2]).T.astype(np.float32)   # (n_ind, n_var_chunk)
        del g                                                # free the haplotype matrix now
        if n_ind is None:
            n_ind = d.shape[0]
        nv = d.shape[1]
        positions = np.array([s.position for s in ts.sites()], dtype=float)
        del ts                                               # drop tree sequence before packing

        # write .bed block (streamed in column batches; bounded transient) + .bim lines
        _write_bed_block(d, bed)
        bim.write("".join(
            f"{ci+1}\tsnp{ci+1}_{int(positions[j])}_{gidx+j}\t0\t{int(positions[j])}\tA\tG\n"
            for j in range(nv)))

        af = d.mean(0) / 2.0
        maf = np.minimum(af, 1 - af).astype(np.float32)
        maf_parts.append(maf)

        # reservoir-sample dense columns for PCA (MAF>=0.05) and causal (MAF>=0.01)
        for jj in np.flatnonzero(maf >= 0.05):
            pca_seen += 1
            gi = gidx + int(jj)
            if len(pca_res_cols) < n_pca_keep:
                pca_res_cols.append(d[:, jj].copy()); pca_res_idx.append(gi)
            else:
                r = int(rng_pca.integers(0, pca_seen))
                if r < n_pca_keep:
                    pca_res_cols[r] = d[:, jj].copy(); pca_res_idx[r] = gi
        for jj in np.flatnonzero(maf >= 0.01):
            cau_seen += 1
            gi = gidx + int(jj)
            if len(cau_res_cols) < n_causal_keep:
                cau_res_cols.append(d[:, jj].copy()); cau_res_idx.append(gi)
            else:
                r = int(rng_cau.integers(0, cau_seen))
                if r < n_causal_keep:
                    cau_res_cols[r] = d[:, jj].copy(); cau_res_idx[r] = gi

        gidx += nv
        log(f"chunk {ci+1}/{n_chunks}: nInd={n_ind} sites={nv} (genome so far={gidx})")
        del d, af, maf, positions
        gc.collect()

    bim.close(); bed.close()
    with open(f"{prefix}.fam", "w") as f:
        f.writelines(f"i{i}\ti{i}\t0\t0\t0\t-9\n" for i in range(n_ind))

    maf = np.concatenate(maf_parts)
    Xpca = np.array(pca_res_cols, dtype=np.float32).T if pca_res_cols else np.empty((n_ind, 0), np.float32)
    Xcau = np.array(cau_res_cols, dtype=np.float32).T if cau_res_cols else np.empty((n_ind, 0), np.float32)
    return dict(n_ind=int(n_ind), n_loci=int(gidx), maf=maf,
                Xpca=Xpca, pca_global_idx=np.array(pca_res_idx, dtype=np.int64),
                Xcausal=Xcau, causal_global_idx=np.array(cau_res_idx, dtype=np.int64))
