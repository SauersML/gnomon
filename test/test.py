import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
import re
from pathlib import Path

import pandas as pd
import numpy as np
import psutil

# ========================================================================================
#                                   CONFIGURATION
# ========================================================================================
CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK1_BINARY = CI_WORKDIR / "plink"
PLINK2_BINARY = CI_WORKDIR / "plink2"
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"

# This new prefix will point to genotype files with updated, position-based variant IDs
UPDATED_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50_pos_ids"

PLINK1_URL = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

GENOTYPE_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
GENOTYPE_FILES = [
    "chr22_subset50.bed.zip",
    "chr22_subset50.bim.zip",
    "chr22_subset50.fam.zip",
]

PGS_SCORES = {
    "PGS004696": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS004696/ScoringFiles/Harmonized/PGS004696_hmPOS_GRCh38.txt.gz",
    "PGS003725": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003725/ScoringFiles/Harmonized/PGS003725_hmPOS_GRCh38.txt.gz",
    "PGS001780": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS001780/ScoringFiles/Harmonized/PGS001780_hmPOS_GRCh38.txt.gz",
}

NUMERICAL_TOLERANCE = 1e-4
CORR_THRESHOLD = 0.9999

# ========================================================================================
#                                   HELPER FUNCTIONS
# ========================================================================================

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}")
    print(char * width)

def download_and_extract(url: str, dest_dir: Path):
    """Downloads and extracts a file, handling .zip and .gz."""
    base = url.split('?')[0]
    filename = Path(base.split('/')[-1])
    outpath = dest_dir / filename
    print(f"Downloading {filename}...")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(outpath, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    except Exception as e:
        print(f"❌ FAILED to download {url}: {e}")
        sys.exit(1)

    if outpath.suffix == '.zip':
        with zipfile.ZipFile(outpath, 'r') as z:
            z.extractall(dest_dir)
        print(f"Extracted ZIP: {filename}")
        outpath.unlink()
    elif outpath.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Extracted GZ: {filename} -> {dest.name}")
        outpath.unlink()

def run_and_measure(cmd: list, name: str):
    """Executes a command, measures its performance, and captures output."""
    print_header(f"RUNNING: {name}", char='-')
    print("Command:")
    print(f"  {' '.join(map(str, cmd))}\n")
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_mem = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak_mem = max(peak_mem, p.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    
    stdout, stderr = proc.communicate()
    duration = time.perf_counter() - start
    
    result = {
        'tool': name,
        'time_sec': duration,
        'peak_mem_mb': peak_mem / (1024*1024),
        'returncode': proc.returncode,
        'success': proc.returncode == 0
    }

    if result['success']:
        print(f"✅ {name} finished in {duration:.2f}s.")
    else:
        print(f"❌ {name} FAILED (Exit Code: {proc.returncode})")
        print_header("STDERR", char='.')
        print(stderr.strip() or "[EMPTY]")
        
    return result

def setup_environment():
    """Downloads all required binaries and data files for the CI test."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    
    # Download PLINK binaries
    for url, binary_path in [(PLINK1_URL, PLINK1_BINARY), (PLINK2_URL, PLINK2_BINARY)]:
        if not binary_path.exists():
            download_and_extract(url, CI_WORKDIR)
            # The zip files contain binaries with generic names, find and rename them
            for p in CI_WORKDIR.iterdir():
                if p.is_file() and p.name.startswith(binary_path.name.split('_')[0]):
                    p.rename(binary_path)
                    binary_path.chmod(0o755)
                    print(f"Renamed and chmod: {binary_path}")
                    break
    
    # Download genotype subset
    for zip_name in GENOTYPE_FILES:
        download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
        
    # Download PGS scores
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)
    
    print_header("WORKDIR CONTENTS AFTER DOWNLOADS", char='.')
    for p in sorted(CI_WORKDIR.iterdir()):
        print(f" - {p.name}")

def create_positional_bim(original_prefix: Path, new_prefix: Path):
    """
    Creates a new .bim file with variant IDs in 'chr:pos' format to ensure
    matching with harmonized PGS files. Also copies corresponding .bed/.fam files.
    """
    print_header("SYNCHRONIZING GENOTYPE VARIANT IDs", char='.')
    original_bim = original_prefix.with_suffix('.bim')
    new_bim = new_prefix.with_suffix('.bim')

    bim_df = pd.read_csv(
        original_bim,
        sep='\t',
        header=None,
        names=['chrom', 'rsid', 'cm', 'pos', 'a1', 'a2'],
        dtype={'chrom': str, 'rsid': str, 'cm': str, 'pos': str, 'a1': str, 'a2': str}
    )
    
    # Create the new canonical ID: chr:pos
    bim_df['rsid'] = bim_df['chrom'] + ':' + bim_df['pos']
    
    bim_df.to_csv(new_bim, sep='\t', header=False, index=False)
    print(f"Created new BIM file with positional IDs: {new_bim}")

    # Copy the .bed and .fam files to complete the new fileset
    shutil.copy(original_prefix.with_suffix('.bed'), new_prefix.with_suffix('.bed'))
    shutil.copy(original_prefix.with_suffix('.fam'), new_prefix.with_suffix('.fam'))
    print(f"Copied .bed and .fam to new prefix: {new_prefix}")

def create_plink_formatted_scorefile(score_file: Path, pgs_id: str) -> Path:
    """
    Reads a raw PGS harmonized file and creates a PLINK2-compatible score file.
    It uses harmonized positions (hm_chr:hm_pos) for robust variant matching.
    """
    print_header(f"PREPARING PLINK FORMAT FOR {pgs_id}", char='.')
    df = pd.read_csv(score_file, sep='\t', comment='#', dtype=str)

    # --- Use harmonized positions for robust variant matching ---
    # The schema guarantees hm_chr and hm_pos in harmonized files.
    if 'hm_chr' not in df.columns or 'hm_pos' not in df.columns:
        raise KeyError("Harmonized columns 'hm_chr' or 'hm_pos' not found.")
    
    df.dropna(subset=['hm_chr', 'hm_pos'], inplace=True)
    variant_id_col = "snp_id"
    df[variant_id_col] = df['hm_chr'] + ':' + df['hm_pos']
    print("  Using 'hm_chr':'hm_pos' as the canonical variant ID.")

    # --- Identify effect allele and effect weight columns ---
    effect_col = 'effect_allele'
    weight_col = 'effect_weight'
    if effect_col not in df.columns or weight_col not in df.columns:
        raise KeyError(f"Required columns '{effect_col}' or '{weight_col}' not found.")
    print(f"  Using effect-allele column: {effect_col}")
    print(f"  Using weight column: {weight_col}")
    
    # --- Subset and write the PLINK-compatible file ---
    plink_df = df[[variant_id_col, effect_col, weight_col]].copy()
    plink_df.columns = ['ID', 'A1', 'WEIGHT'] # Use standard column names for clarity
    
    out_path = CI_WORKDIR / f"{pgs_id}_plink_format.tsv"
    plink_df.to_csv(out_path, sep='\t', index=False, na_rep='NA')
    print(f"\n  Written PLINK2-compatible score file: {out_path}")

    # --- Debug: Check for matches against the updated BIM file ---
    bim_path = UPDATED_PLINK_PREFIX.with_suffix('.bim')
    bim_ids = pd.read_csv(bim_path, sep='\t', header=None, usecols=[1], dtype=str)[1].to_numpy()
    matches = plink_df['ID'].isin(bim_ids).sum()
    print(f"  Variants in score file matching updated BIM file: {matches} / {len(plink_df)}")
    if matches == 0:
        print("  ⚠️ WARNING: No matching variants found. PLINK will likely fail.")

    return out_path

def main():
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    
    # Create a single, positionally-keyed genotype fileset to be used by PLINK.
    create_positional_bim(ORIGINAL_PLINK_PREFIX, UPDATED_PLINK_PREFIX)

    all_results, failures = [], []
    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        raw_score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        print(f"Raw PGS file: {raw_score_file}")

        # 1) Gnomon (assumed to work with the raw PGS file format and original genotypes)
        res_g = run_and_measure([
            str(GNOMON_BINARY), "--score", str(raw_score_file), str(ORIGINAL_PLINK_PREFIX)
        ], f"gnomon_{pgs_id}")
        all_results.append(res_g)
        if not res_g['success']:
            failures.append(f"{pgs_id} (gnomon_failed)")
            # Continue to the next score if gnomon fails
            continue

        # 2) Prepare PLINK2-compatible score file
        try:
            plink_fmt_file = create_plink_formatted_scorefile(raw_score_file, pgs_id)
        except Exception as e:
            print(f"❌ Error preparing PLINK2 format for {pgs_id}: {e}")
            failures.append(f"{pgs_id} (format_error)")
            continue

        # 3) PLINK2
        out2 = CI_WORKDIR / f"plink2_{pgs_id}"
        # Use the genotype files with updated positional IDs for matching
        res_p2 = run_and_measure([
            str(PLINK2_BINARY),
            "--bfile", str(UPDATED_PLINK_PREFIX),
            "--score", str(plink_fmt_file), "header", "cols=maybesid,dosagesum,scoreavgs",
            "--out", str(out2)
        ], f"plink2_{pgs_id}")
        all_results.append(res_p2)
        if not res_p2['success']:
            failures.append(f"{pgs_id} (plink2_failed)")
            continue

    # Final summary
    print_header('PERFORMANCE SUMMARY')
    if all_results:
        df = pd.DataFrame(all_results)
        df['tool_base'] = df['tool'].str.split('_').str[0]
        summary = df[df['success']].groupby('tool_base').agg(
            mean_time_sec=('time_sec','mean'),
            mean_mem_mb=('peak_mem_mb','mean')
        ).reset_index()
        if not summary.empty:
            print(summary.to_markdown(index=False, floatfmt='.3f'))

    print_header(f"CI CHECK {'FAILED' if failures else 'PASSED'}")
    if failures:
        print('❌ Tests failed:')
        for f in sorted(set(failures)):
            print(f'  - {f}')
        sys.exit(1)
    else:
        print('🎉 All tests passed.')
        sys.exit(0)

if __name__ == '__main__':
    main()
