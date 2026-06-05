"""Parallel streaming genotype simulation + incremental PLINK1 .bed writing.

genotype_matrix() returns int32 and is all-or-nothing, and at biobank sample
sizes a 5 Mb segment holds 2e5+ sites -> a single matrix would be 100s of GiB.
So the genome is simulated as MANY SMALL chunks (each an independent LD segment),
and -- because the chunks are independent -- they are processed IN PARALLEL across
the task's cores. Each worker owns a contiguous block of chunks and:
  - simulates ancestry+mutations,
  - decodes with the fast vectorised genotype_matrix, collapses int32 -> int8
    alt-dosage in row-batches (transient bounded),
  - appends each chunk to its OWN .bed/.bim SHARD (SNP-major, fast 2-bit packing),
  - reservoir-samples PCA-candidate (MAF>=0.05) and causal-candidate (MAF>=0.01)
    columns down to a per-block quota.
The parent then concatenates the shards in genome order and merges the reservoirs.
Per-worker RAM ~ one small chunk's int32 matrix; nothing ever holds the genome.

The .bed encoding matches PLINK1: 2 bits/sample, LSB-first, 4 samples/byte,
code 0b00=hom-A1(2 alt), 0b10=het, 0b11=hom-A2(0 alt), 0b01=missing; A1=alt.
"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import msprime


# index by alt-dosage 0,1,2 -> PLINK1 2-bit code
_CODE_MAP = np.array([0b11, 0b10, 0b00], dtype=np.uint8)

# Standard GWAS QC: only common variants enter the .bed (and thus the GWAS / P+T).
# At biobank n most simulated sites are singletons/ultra-rare that a P+T PGS would
# threshold out anyway; all causal variants are drawn from the MAF>=0.01 pool, so
# this removes only rare noise and cuts the variant set ~9x. Set 0.0 to keep all.
MAF_BED_MIN = 0.01


def _pack_variant_major(dos_vm: np.ndarray, fh, row_batch: int = 2048) -> None:
    """Append a (n_snp, n_ind) int8 alt-dosage block to an open .bed handle in
    SNP-major PLINK1 2-bit packing. Operates directly on the variant-major dosage
    (no transpose), using explicit shifts (no reshape+reduce) -- ~3x faster than a
    transposed gather. row_batch bounds the transient packing allocation."""
    n_snp, n_ind = dos_vm.shape
    n_bytes = (n_ind + 3) // 4
    pad = n_bytes * 4 - n_ind
    for s0 in range(0, n_snp, row_batch):
        sub = dos_vm[s0:s0 + row_batch]                      # (b, n_ind) int8, contiguous
        codes = _CODE_MAP[sub]                               # (b, n_ind) uint8
        if pad:
            codes = np.pad(codes, ((0, 0), (0, pad)))
        codes = codes.reshape(codes.shape[0], n_bytes, 4)
        packed = (codes[:, :, 0] | (codes[:, :, 1] << 2)
                  | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)).astype(np.uint8)
        fh.write(packed.reshape(-1).tobytes())               # (b, n_bytes) row-major = SNP-major


def _decode_chunk(ci, dG, samples, chunk_bp, recomb, mut, seed):
    """Simulate one chunk and return (n_ind, dos, pos) with dos a (n_sites, n_ind)
    int8 alt-dosage (variant-major), or None if the chunk is monomorphic."""
    ts = msprime.sim_ancestry(samples=samples, demography=dG,
                              sequence_length=float(chunk_bp), recombination_rate=recomb,
                              ploidy=2, random_seed=101 + seed * 131 + ci * 7)
    ts = msprime.sim_mutations(ts, rate=mut, random_seed=102 + seed * 131 + ci * 7)
    if ts.num_sites == 0:
        return None
    n_ind = ts.num_samples // 2
    G = ts.genotype_matrix()                                 # (n_sites, n_samples) int32
    nsite = G.shape[0]
    dos = np.empty((nsite, n_ind), dtype=np.int8)            # (n_sites, n_ind) int8 dosage {0,1,2}
    for r0 in range(0, nsite, 2048):                         # small batch: bounds the int32 sum temp
        r1 = min(r0 + 2048, nsite)
        blk = G[r0:r1]
        # recurrent mutation at biobank n makes some sites multi-allelic (allele
        # index >= 2), so a haplotype-pair sum can exceed 2; clip to a biallelic
        # alt-dosage {0,1,2} (keeps MAF in [0,1] and the .bed codes in range).
        dos[r0:r1] = np.clip(blk[:, 0::2] + blk[:, 1::2], 0, 2).astype(np.int8)
    del G
    pos = np.asarray(ts.tables.sites.position)
    return n_ind, dos, pos


def _process_block(block):
    """Worker: process a contiguous block of chunks into bed/bim shards + a per-block
    reservoir, written to files under shard_dir. Returns small metadata only."""
    (block_id, chunk_ids, dG, samples, chunk_bp, recomb, mut, seed,
     shard_dir, pca_quota, causal_quota) = block
    rng_pca = np.random.default_rng(7001 + seed * 1000 + block_id)
    rng_cau = np.random.default_rng(8001 + seed * 1000 + block_id)
    pca_cols, pca_seen = [], 0
    cau_cols, cau_seen = [], 0
    maf_parts = []
    bim_path = os.path.join(shard_dir, f"b{block_id}.bim")
    bed_path = os.path.join(shard_dir, f"b{block_id}.bed")
    n_ind = None
    nsite_total = 0

    def _reservoir(maf, dos, thr, cols, seen, rng, quota):
        sites = np.flatnonzero(maf >= thr)
        if sites.size:
            draws = rng.integers(0, seen + 1 + np.arange(sites.size))
            for k, jj in enumerate(sites):
                if len(cols) < quota:
                    cols.append(dos[jj].copy())
                elif int(draws[k]) < quota:
                    cols[int(draws[k])] = dos[jj].copy()
        return seen + int(sites.size)

    with open(bed_path, "wb") as bed, open(bim_path, "w") as bim:
        for ci in chunk_ids:
            r = _decode_chunk(ci, dG, samples, chunk_bp, recomb, mut, seed)
            if r is None:
                continue
            n_ind, dos, pos = r
            af = dos.mean(1) / 2.0
            maf = np.minimum(af, 1.0 - af).astype(np.float32)
            keep = maf >= MAF_BED_MIN                        # common-variant QC
            if not keep.any():
                del dos
                continue
            dos = dos[keep]                                  # (n_kept, n_ind) int8, contiguous copy
            pos = pos[keep]
            maf = maf[keep]
            nk = dos.shape[0]
            _pack_variant_major(dos, bed)
            # non-numeric contig name per chunk: plink's --allow-extra-chr accepts
            # arbitrary NAMES but rejects numeric codes above the human set (e.g. 27).
            bim.write("".join(
                f"c{ci+1}\ts{ci+1}_{k}\t0\t{int(pos[k])}\tA\tG\n" for k in range(nk)))
            maf_parts.append(maf)
            pca_seen = _reservoir(maf, dos, 0.05, pca_cols, pca_seen, rng_pca, pca_quota)
            cau_seen = _reservoir(maf, dos, 0.01, cau_cols, cau_seen, rng_cau, causal_quota)
            nsite_total += nk
            del dos

    maf_path = os.path.join(shard_dir, f"b{block_id}.maf.npy")
    pca_path = os.path.join(shard_dir, f"b{block_id}.pca.npy")
    cau_path = os.path.join(shard_dir, f"b{block_id}.cau.npy")
    np.save(maf_path, np.concatenate(maf_parts) if maf_parts else np.empty(0, np.float32))
    # store (n_ind, k) so the parent can hstack columns directly
    np.save(pca_path, (np.array(pca_cols, dtype=np.int8).T if pca_cols
                       else np.empty((n_ind or 0, 0), np.int8)))
    np.save(cau_path, (np.array(cau_cols, dtype=np.int8).T if cau_cols
                       else np.empty((n_ind or 0, 0), np.int8)))
    return dict(block_id=block_id, bed_path=bed_path, bim_path=bim_path, maf_path=maf_path,
                pca_path=pca_path, cau_path=cau_path, n_ind=n_ind, nsite=nsite_total,
                pca_seen=pca_seen, cau_seen=cau_seen)


def simulate_stream(dG, samples, n_chunks, chunk_bp, recomb, mut, seed,
                    prefix, n_pca_keep, n_causal_keep, log=print, workers=None):
    """Parallel-simulate the genome, writing <prefix>.bed/.bim/.fam.

    Chunks are independent LD segments, partitioned into contiguous blocks (one per
    worker) so shard concatenation reproduces genome order. Returns dict with:
    n_ind, n_loci, maf, Xpca (n_ind, kept_pca), Xcausal (n_ind, kept_causal), and
    pca/causal_global_idx (kept for the historical interface; unused downstream)."""
    if workers is None:
        try:
            workers = len(os.sched_getaffinity(0))
        except AttributeError:
            workers = os.cpu_count() or 1
    workers = max(1, min(int(workers), n_chunks))

    # CONTIGUOUS chunk blocks, one per worker, so shard concatenation = genome order;
    # per-block reservoir quota sums to the genome-wide keep target.
    edges = np.linspace(0, n_chunks, workers + 1).astype(int)
    blocks_idx = [list(range(edges[b], edges[b + 1])) for b in range(workers)]
    blocks_idx = [bk for bk in blocks_idx if bk]
    nb = len(blocks_idx)
    pca_quota = int(np.ceil(n_pca_keep / nb))
    causal_quota = int(np.ceil(n_causal_keep / nb))

    shard_dir = f"{prefix}_shards"
    shutil.rmtree(shard_dir, ignore_errors=True)
    os.makedirs(shard_dir, exist_ok=True)
    tasks = [(bid, bk, dG, samples, chunk_bp, recomb, mut, seed, shard_dir, pca_quota, causal_quota)
             for bid, bk in enumerate(blocks_idx)]
    log(f"parallel generate: {n_chunks} chunks x {chunk_bp}bp over {nb} workers")

    results = [None] * nb
    with ProcessPoolExecutor(max_workers=nb) as pool:
        for res in pool.map(_process_block, tasks):
            results[res["block_id"]] = res
            log(f"block {res['block_id']+1}/{nb} done: sites={res['nsite']} "
                f"pca_seen={res['pca_seen']} causal_seen={res['cau_seen']}")

    n_ind = next(r["n_ind"] for r in results if r["n_ind"] is not None)

    # merge bed shards (magic + bodies in block order), bim, maf, reservoirs
    with open(f"{prefix}.bed", "wb") as bed:
        bed.write(bytes([0x6c, 0x1b, 0x01]))                 # magic + SNP-major
        for r in results:
            with open(r["bed_path"], "rb") as sh:
                shutil.copyfileobj(sh, bed, length=1 << 24)
    with open(f"{prefix}.bim", "w") as bim:
        for r in results:
            with open(r["bim_path"]) as sh:
                shutil.copyfileobj(sh, bim)
    with open(f"{prefix}.fam", "w") as f:
        f.writelines(f"i{i}\ti{i}\t0\t0\t0\t-9\n" for i in range(n_ind))

    maf = np.concatenate([np.load(r["maf_path"]) for r in results])
    Xpca = np.hstack([np.load(r["pca_path"]) for r in results])
    Xcau = np.hstack([np.load(r["cau_path"]) for r in results])
    # trim to the genome-wide keep targets if quotas overshot (rounding)
    if Xpca.shape[1] > n_pca_keep:
        sel = np.random.default_rng(7001 + seed).choice(Xpca.shape[1], n_pca_keep, replace=False)
        Xpca = Xpca[:, sel]
    if Xcau.shape[1] > n_causal_keep:
        sel = np.random.default_rng(8001 + seed).choice(Xcau.shape[1], n_causal_keep, replace=False)
        Xcau = Xcau[:, sel]
    n_loci = int(sum(r["nsite"] for r in results))
    shutil.rmtree(shard_dir, ignore_errors=True)
    return dict(n_ind=int(n_ind), n_loci=n_loci, maf=maf,
                Xpca=Xpca, pca_global_idx=np.arange(Xpca.shape[1], dtype=np.int64),
                Xcausal=Xcau, causal_global_idx=np.arange(Xcau.shape[1], dtype=np.int64))
