#!/usr/bin/env python3
"""
Generate PRS-CSx LD reference from PLINK training data.
Creates the HDF5 LD blocks and SNP info files expected by PRS-CSx.
"""
import sys
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

def read_plink_bed(bfile):
    """Read PLINK .bed file and return genotype matrix."""
    from pandas_plink import read_plink
    (bim, fam, bed) = read_plink(bfile)
    return bim, fam, bed.compute()

def compute_ld_blocks(bim, geno, block_size=1000):
    """
    Compute LD correlation matrices in blocks.
    Returns list of (start_idx, end_idx, ld_matrix) tuples.
    """
    n_snps = geno.shape[1]
    blocks = []
    
    for start in range(0, n_snps, block_size):
        end = min(start + block_size, n_snps)
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
        
        blocks.append((start, end, ld_matrix))
    
    return blocks

def write_prscsx_reference(bim, ld_blocks, chrom, output_dir):
    """Write LD blocks and SNP info in PRS-CSx format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write LD blocks to HDF5
    ld_file = output_dir / f"ldblk_1kg_chr{chrom}.hdf5"
    with h5py.File(ld_file, 'w') as f:
        for idx, (start, end, ld_matrix) in enumerate(ld_blocks):
            blk_name = f"blk_{idx}"
            f.create_dataset(blk_name, data=ld_matrix, compression='gzip')
    
    # Write SNP info file
    snpinfo_file = output_dir / f"snpinfo_1kg_chr{chrom}"
    snp_info = pd.DataFrame({
        'CHR': bim['chrom'].values,
        'SNP': bim['snp'].values,
        'BP': bim['pos'].values,
        'A1': bim['a0'].values,
        'A2': bim['a1'].values,
        'MAF': 0.5  # Placeholder, PRS-CSx doesn't strictly require accurate MAF
    })
    snp_info.to_csv(snpinfo_file, sep='\t', index=False, header=False)
    
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
