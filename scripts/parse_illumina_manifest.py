#!/usr/bin/env python3
"""Parse Illumina manifest CSVs and extract chromosome + position for PCA variant lists.

Usage:
    python scripts/parse_illumina_manifest.py input.csv output.tsv

The output is a two-column TSV (chr, pos) suitable for gnomon fit --list.
"""

import csv
import gzip
import sys
from pathlib import Path


def parse_manifest(input_path: Path, output_path: Path) -> int:
    """Parse an Illumina manifest CSV and write chr/pos to output TSV.
    
    Returns the number of variants written.
    """
    # Handle gzipped input if needed
    if input_path.suffix == '.gz':
        open_fn = lambda p: gzip.open(p, 'rt', encoding='utf-8', errors='replace')
    else:
        open_fn = lambda p: open(p, 'r', encoding='utf-8', errors='replace')
    
    variants_written = 0
    in_assay_section = False
    header_found = False
    chr_idx = None
    mapinfo_idx = None
    
    with open_fn(input_path) as infile, open(output_path, 'w') as outfile:
        # Write header
        outfile.write("chr\tpos\n")
        
        for line in infile:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Look for [Assay] section marker
            if line == '[Assay]':
                in_assay_section = True
                continue
            
            # Skip until we're in the Assay section
            if not in_assay_section:
                continue
            
            # Parse as CSV
            row = next(csv.reader([line]))
            
            # First row after [Assay] is the header
            if not header_found:
                header_found = True
                # Find Chr and MapInfo columns
                for i, col in enumerate(row):
                    if col == 'Chr':
                        chr_idx = i
                    elif col == 'MapInfo':
                        mapinfo_idx = i
                
                if chr_idx is None or mapinfo_idx is None:
                    raise ValueError(f"Could not find Chr and MapInfo columns in header: {row}")
                
                print(f"Found Chr at index {chr_idx}, MapInfo at index {mapinfo_idx}", file=sys.stderr)
                continue
            
            # Data row - extract chr and position
            if len(row) <= max(chr_idx, mapinfo_idx):
                continue  # Skip malformed rows
            
            chrom = row[chr_idx]
            pos = row[mapinfo_idx]
            
            # Skip rows with missing chromosome or position
            if not chrom or not pos or chrom == '0' or pos == '0':
                continue
            
            # Validate position is numeric
            try:
                int(pos)
            except ValueError:
                continue
            
            outfile.write(f"{chrom}\t{pos}\n")
            variants_written += 1
    
    return variants_written


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.csv output.tsv", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Parsing {input_path}...", file=sys.stderr)
    count = parse_manifest(input_path, output_path)
    print(f"Wrote {count:,} variants to {output_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
