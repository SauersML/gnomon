#!/usr/bin/env python3
"""
Generate PRS-CSx LD reference from PLINK training data.
Creates the HDF5 LD blocks and SNP info files expected by PRS-CSx.
"""
import sys
import os
import subprocess
import tempfile
import shutil
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

def read_plink_bed(bfile):
    """Read PLINK .bed file and return genotype matrix."""
    from pandas_plink import read_plink
    (bim, fam, bed) = read_plink(bfile)
    return bim, fam, bed.compute()


def _read_ld_matrix_from_file(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        ld = [[float(val) for val in line.strip().split()] for line in f if line.strip()]
    return np.asarray(ld, dtype=np.float64)


def _compute_ld_matrix_plink(bfile: str, snplist: list[str], work_dir: str) -> np.ndarray:
    # plink/plink2 --extract expects one variant ID per line
    snp_path = os.path.join(work_dir, "snplist.txt")
    with open(snp_path, "w", encoding="utf-8") as f:
        for snp in snplist:
            f.write(f"{snp}\n")

    out_prefix = os.path.join(work_dir, "ld")

    plink_exe = shutil.which("plink2") or shutil.which("plink")
    if plink_exe is None:
        raise RuntimeError("Neither plink2 nor plink is available on PATH")

    cmd = [
        plink_exe,
        "--bfile", bfile,
        "--extract", snp_path,
        "--r", "square",
        "--out", out_prefix,
        "--silent",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"PLINK LD computation failed: {result.stderr}")

    ld_file = out_prefix + ".ld"
    if not os.path.exists(ld_file):
        raise RuntimeError(f"Expected PLINK LD output not found: {ld_file}")
    return _read_ld_matrix_from_file(ld_file)

def compute_ld_blocks(bim, geno, block_size=1000):
    """
    Compute LD correlation matrices in blocks.
    Returns list of (start_idx, end_idx, ld_matrix) tuples.
    """
    n_snps = geno.shape[1]
    blocks = []

    ld_method = os.environ.get("PRSCSX_LD_METHOD", "numpy").strip().lower()
    bfile_for_plink = os.environ.get("PRSCSX_LD_BFILE", "").strip()
    block_size = int(os.environ.get("PRSCSX_LD_BLOCK_SIZE", block_size))

    if ld_method == "plink" and not bfile_for_plink:
        raise RuntimeError(
            "PRSCSX_LD_METHOD=plink requires PRSCSX_LD_BFILE to be set to a PLINK --bfile prefix"
        )
    
    for start in range(0, n_snps, block_size):
        end = min(start + block_size, n_snps)
        block_snps = bim['snp'].iloc[start:end].astype(str).tolist()
        if ld_method == "plink":
            with tempfile.TemporaryDirectory() as td:
                ld_matrix = _compute_ld_matrix_plink(bfile_for_plink, block_snps, td)
        else:
            block_geno = geno[:, start:end]

            # Standardize genotypes
            means = np.nanmean(block_geno, axis=0)
            stds = np.nanstd(block_geno, axis=0)
            stds[stds == 0] = 1  # Avoid division by zero
        
            block_std = (block_geno - means) / stds
            block_std = np.nan_to_num(block_std)
        
            # Compute correlation matrix
            ld_matrix = np.corrcoef(block_std.T)
            ld_matrix = np.nan_to_num(ld_matrix)

        blocks.append((start, end, ld_matrix, block_snps))
    
    return blocks

def write_prscsx_reference(bim, ld_blocks, chrom, output_dir):
    """Write LD blocks and SNP info in PRS-CSx format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PRS-CSx expects:
    #   - {ref_dir}/snpinfo_mult_1kg_hm3
    #   - {ref_dir}/ldblk_1kg_eur/ldblk_1kg_chr{chrom}.hdf5
    # where the hdf5 file has groups blk_1..blk_N, each containing datasets
    # 'ldblk' and 'snplist'.

    ldblk_dir = output_dir / "ldblk_1kg_eur"
    ldblk_dir.mkdir(parents=True, exist_ok=True)

    ld_file = ldblk_dir / f"ldblk_1kg_chr{chrom}.hdf5"
    with h5py.File(ld_file, 'w') as f:
        for idx, (_start, _end, ld_matrix, block_snps) in enumerate(ld_blocks, start=1):
            grp = f.create_group(f"blk_{idx}")
            grp.create_dataset("ldblk", data=ld_matrix, compression='gzip')
            snplist = np.asarray([s.encode("utf-8") for s in block_snps], dtype="S")
            grp.create_dataset("snplist", data=snplist, compression='gzip')

    snpinfo_file = output_dir / "snpinfo_mult_1kg_hm3"

    # Force chromosome to match the requested chromosome and write the full
    # multi-pop header expected by PRS-CSx parse_ref().
    chrom_values = np.full(shape=(bim.shape[0],), fill_value=int(chrom), dtype=np.int64)
    snp_values = bim['snp'].astype(str).values
    bp_values = bim['pos'].astype(int).values
    a1_values = bim['a0'].astype(str).values
    a2_values = bim['a1'].astype(str).values

    # Placeholder freqs/flip flags. parse_ref() only requires FRQ_*>0.
    frq = np.full(shape=(bim.shape[0],), fill_value=0.5, dtype=np.float64)
    flp = np.ones(shape=(bim.shape[0],), dtype=np.int64)

    snp_info = pd.DataFrame({
        'CHR': chrom_values,
        'SNP': snp_values,
        'BP': bp_values,
        'A1': a1_values,
        'A2': a2_values,
        'FRQ_AFR': frq,
        'FRQ_AMR': frq,
        'FRQ_EAS': frq,
        'FRQ_EUR': frq,
        'FRQ_SAS': frq,
        'FLP_AFR': flp,
        'FLP_AMR': flp,
        'FLP_EAS': flp,
        'FLP_EUR': flp,
        'FLP_SAS': flp,
    })
    snp_info.to_csv(snpinfo_file, sep='\t', index=False)

    print(f"Created LD reference: {ld_file}")
    print(f"Created SNP info: {snpinfo_file}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python create_ld_ref.py <bfile> <chrom> <output_dir>")
        sys.exit(1)
    
    bfile = sys.argv[1]
    chrom = sys.argv[2]
    output_dir = sys.argv[3]
    
    print(f"Reading PLINK files from {bfile}...")
    bim, fam, geno = read_plink_bed(bfile)
    
    print(f"Computing LD blocks for chromosome {chrom}...")
    ld_blocks = compute_ld_blocks(bim, geno)
    
    print(f"Writing PRS-CSx reference to {output_dir}...")
    write_prscsx_reference(bim, ld_blocks, chrom, output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
