import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
import os
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

# Gnomon will use the original, raw files
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"
# Plink1/2 will use a new fileset with cleaned and synchronized variant IDs
PLINK_COMPAT_PREFIX = CI_WORKDIR / "chr22_subset50_plink_compat"

PLINK1_URL = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20231211.zip"
PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"

GENOTYPE_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
GENOTYPE_FILES = [
    "chr22_subset50.bed.zip", "chr22_subset50.bim.zip", "chr22_subset50.fam.zip"
]

PGS_SCORES = {
    "PGS004696": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS004696/ScoringFiles/Harmonized/PGS004696_hmPOS_GRCh38.txt.gz",
    "PGS003725": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003725/ScoringFiles/Harmonized/PGS003725_hmPOS_GRCh38.txt.gz",
    "PGS001780": "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS001780/ScoringFiles/Harmonized/PGS001780_hmPOS_GRCh38.txt.gz",
}

# ========================================================================================
#                                   HELPER FUNCTIONS
# ========================================================================================

def print_header(title: str, char: str = "="):
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}")
    print(char * width)

def print_debug_header(title: str):
    """Prints a smaller header for debug sections."""
    print("\n" + "." * 80)
    print(f".... DEBUG: {title} ....")
    print("." * 80)

def download_and_extract(url: str, dest_dir: Path):
    """Downloads and extracts a file, handling .zip and .gz."""
    base = url.split('?')[0]
    filename = Path(base.split('/')[-1])
    outpath = dest_dir / filename
    print(f"Downloading {filename}...")
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(outpath, 'wb') as f: shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print(f"❌ FAILED to download {url}: {e}"); sys.exit(1)

    if outpath.suffix == '.zip':
        with zipfile.ZipFile(outpath, 'r') as z: z.extractall(dest_dir)
        print(f"Extracted ZIP: {filename}"); outpath.unlink()
    elif outpath.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        print(f"Extracted GZ: {filename} -> {dest.name}"); outpath.unlink()

def run_and_measure(cmd: list, name: str, out_prefix: Path):
    """Executes a command, measures its performance, and captures output."""
    print_header(f"RUNNING: {name}", char='-')
    print(f"Command:\n  {' '.join(map(str, cmd))}\n")
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak_mem = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try: peak_mem = max(peak_mem, p.memory_info().rss)
            except psutil.NoSuchProcess: break
            time.sleep(0.01)
    except psutil.NoSuchProcess: pass
    
    _, stderr = proc.communicate()
    duration = time.perf_counter() - start
    
    result = {'tool': name, 'time_sec': duration, 'peak_mem_mb': peak_mem / (1024*1024), 'returncode': proc.returncode, 'success': proc.returncode == 0}

    if result['success']:
        print(f"✅ {name} finished in {duration:.2f}s.")
        if (out_prefix.with_suffix('.sscore')).exists(): print(f"  > PLINK2 output: {out_prefix}.sscore")
        if (out_prefix.with_suffix('.profile')).exists(): print(f"  > PLINK1 output: {out_prefix}.profile")
    else:
        print(f"❌ {name} FAILED (Exit Code: {proc.returncode})")
        print_debug_header("STDERR")
        print(stderr.strip() or "[EMPTY]")
        
    return result

def setup_environment():
    """Downloads all required binaries and data files for the CI test."""
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    
    for url, binary_path in [(PLINK1_URL, PLINK1_BINARY), (PLINK2_URL, PLINK2_BINARY)]:
        if not binary_path.exists():
            download_and_extract(url, CI_WORKDIR)
            for p in CI_WORKDIR.iterdir():
                if p.is_file() and p.name.startswith(binary_path.name.split('_')[0]):
                    p.rename(binary_path); p.chmod(0o755); print(f"Renamed and chmod: {binary_path}"); break

    for zip_name in GENOTYPE_FILES: download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    for url in PGS_SCORES.values(): download_and_extract(url, CI_WORKDIR)
    
    print_debug_header("WORKDIR CONTENTS")
    for root, _, files in os.walk(CI_WORKDIR):
        if Path(root) == CI_WORKDIR:
             for name in sorted(files): print(f" - {name}")

def create_synchronized_genotype_files(original_prefix: Path, new_prefix: Path):
    """
    Creates a new PLINK fileset with fully unique, allele-aware variant IDs
    in 'chr:pos:a1_sorted:a2_sorted' format for robust matching.
    """
    print_header("SYNCHRONIZING GENOTYPE VARIANT IDs FOR PLINK", char='.')
    original_bim, new_bim = original_prefix.with_suffix('.bim'), new_prefix.with_suffix('.bim')
    
    bim_df = pd.read_csv(original_bim, sep='\s+', header=None, names=['chrom', 'rsid', 'cm', 'pos', 'a1', 'a2'], dtype=str)
    
    print_debug_header(f"Original {original_bim.name} (first 5 rows)")
    print(bim_df.head().to_string(index=False))

    # Create a canonical, allele-aware ID by sorting alleles. This handles strand differences.
    # Ex: 22:12345:A:G will be the ID for a variant, regardless of which is REF/ALT.
    alleles = bim_df[['a1', 'a2']].apply(lambda x: sorted(x), axis=1, result_type='expand')
    bim_df['canonical_id'] = bim_df['chrom'] + ':' + bim_df['pos'] + ':' + alleles[0] + ':' + alleles[1]

    # Create the new fileset for PLINK
    bim_df[['chrom', 'canonical_id', 'cm', 'pos', 'a1', 'a2']].to_csv(new_bim, sep='\t', header=False, index=False)
    
    print_debug_header(f"New {new_bim.name} with canonical IDs (first 5 rows)")
    print(bim_df[['chrom', 'canonical_id', 'cm', 'pos', 'a1', 'a2']].head().to_string(index=False))
    print(f"\nCreated new BIM file: {new_bim}")

    shutil.copy(original_prefix.with_suffix('.bed'), new_prefix.with_suffix('.bed'))
    shutil.copy(original_prefix.with_suffix('.fam'), new_prefix.with_suffix('.fam'))
    print(f"Copied .bed and .fam to new prefix: {new_prefix}")

def create_plink_formatted_scorefile(score_file: Path, pgs_id: str) -> Path:
    """Reads a raw PGS file and creates a clean, de-duplicated, PLINK-compatible score file."""
    print_header(f"PREPARING PLINK FORMAT FOR {pgs_id}", char='.')
    
    # Define required columns for a minimal valid entry
    cols_to_use = ['hm_chr', 'hm_pos', 'effect_allele', 'other_allele', 'effect_weight']
    df = pd.read_csv(score_file, sep='\t', comment='#', usecols=lambda c: c in cols_to_use, dtype=str, low_memory=False)
    print(f"  Loaded {len(df)} raw entries from {score_file.name}")

    # --- Defensive Cleaning Pipeline ---
    # 1. Drop rows missing any essential information
    df.dropna(subset=cols_to_use, inplace=True)
    print(f"  Step 1: Kept {len(df)} rows with all required columns.")

    # 2. Clean weights: force to numeric and drop rows that fail
    df['effect_weight'] = pd.to_numeric(df['effect_weight'], errors='coerce')
    df.dropna(subset=['effect_weight'], inplace=True)
    print(f"  Step 2: Kept {len(df)} rows after removing non-numeric weights.")

    # 3. Create the same canonical ID as used for the BIM file
    alleles = df[['effect_allele', 'other_allele']].apply(lambda x: sorted(x), axis=1, result_type='expand')
    df['canonical_id'] = df['hm_chr'] + ':' + df['hm_pos'] + ':' + alleles[0] + ':' + alleles[1]
    print(f"  Step 3: Created canonical allele-aware IDs (e.g., chr:pos:A:G).")
    
    # 4. De-duplicate based on the new canonical ID
    df.drop_duplicates(subset=['canonical_id'], keep='first', inplace=True)
    print(f"  Step 4: Kept {len(df)} rows after de-duplicating by canonical ID.")

    # --- Prepare final file for PLINK ---
    out_path = CI_WORKDIR / f"{pgs_id}_plink_format.tsv"
    final_df = df[['canonical_id', 'effect_allele', 'effect_weight']].copy()
    final_df.columns = ['ID', 'EFFECT_ALLELE', 'WEIGHT']
    final_df.to_csv(out_path, sep='\t', index=False, na_rep='NA')
    
    print_debug_header("Final clean score data for PLINK (first 5 rows)")
    print(final_df.head().to_string(index=False))
    print(f"\nWritten clean PLINK-compatible file: {out_path} ({len(final_df)} variants)")

    # Final check for matches against the synchronized BIM file
    bim_ids = pd.read_csv(PLINK_COMPAT_PREFIX.with_suffix('.bim'), sep='\t', header=None, usecols=[1], dtype=str)[1].unique()
    matches = final_df['ID'].isin(bim_ids).sum()
    print(f"  > Variants matching synchronized genotype file: {matches} / {len(final_df)}")
    if matches == 0: print("  > ⚠️ WARNING: No matching variants found. PLINK will likely fail.")

    return out_path

def main():
    """Main function to run the CI test suite."""
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    create_synchronized_genotype_files(ORIGINAL_PLINK_PREFIX, PLINK_COMPAT_PREFIX)

    all_results, failures = [], []
    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        raw_score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem

        # 1. Gnomon uses the original, raw genotype files
        res_g = run_and_measure([str(GNOMON_BINARY), "--score", str(raw_score_file), str(ORIGINAL_PLINK_PREFIX)], f"gnomon_{pgs_id}", CI_WORKDIR / f"gnomon_{pgs_id}")
        all_results.append(res_g);
        if not res_g['success']: failures.append(f"{pgs_id} (gnomon_failed)"); continue

        # 2. Prepare a clean scoring file for the fragile PLINK tools
        try:
            plink_fmt_file = create_plink_formatted_scorefile(raw_score_file, pgs_id)
        except Exception as e:
            print(f"❌ Error preparing PLINK format for {pgs_id}: {e}"); failures.append(f"{pgs_id} (format_error)"); continue

        # 3. PLINK2 uses the synchronized genotype files
        out2_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        res_p2 = run_and_measure([str(PLINK2_BINARY), "--bfile", str(PLINK_COMPAT_PREFIX), "--score", str(plink_fmt_file), "header", "no-mean-imputation", "--out", str(out2_prefix)], f"plink2_{pgs_id}", out2_prefix)
        all_results.append(res_p2)
        if not res_p2['success']: failures.append(f"{pgs_id} (plink2_failed)")
        
        # 4. PLINK1 also uses the synchronized genotype files
        out1_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        res_p1 = run_and_measure([str(PLINK1_BINARY), "--bfile", str(PLINK_COMPAT_PREFIX), "--score", str(plink_fmt_file), "1", "2", "3", "header", "no-mean-imputation", "--out", str(out1_prefix)], f"plink1_{pgs_id}", out1_prefix)
        all_results.append(res_p1)
        if not res_p1['success']: failures.append(f"{pgs_id} (plink1_failed)")

    print_header('PERFORMANCE SUMMARY')
    if all_results:
        df = pd.DataFrame(all_results)
        df['tool_base'] = df['tool'].str.split('_').str[0]
        summary = df[df['success']].groupby('tool_base').agg(mean_time_sec=('time_sec','mean'), mean_mem_mb=('peak_mem_mb','mean')).reset_index()
        if not summary.empty: print(summary.to_markdown(index=False, floatfmt='.3f'))

    print_header(f"CI CHECK {'FAILED' if failures else 'PASSED'}")
    if failures:
        print('❌ Tests failed:'); [print(f'  - {f}') for f in sorted(list(set(failures)))]; sys.exit(1)
    else:
        print('🎉 All tests passed.'); sys.exit(0)

if __name__ == '__main__':
    main()
