import subprocess
import requests
import zipfile
import gzip
import shutil
import time
import sys
import os
from pathlib import Path
from itertools import combinations

import pandas as pd
import numpy as np
import psutil

# ========================================================================================
#                                     CONFIGURATION
# ========================================================================================
CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK1_BINARY = CI_WORKDIR / "plink"
PLINK2_BINARY = CI_WORKDIR / "plink2"

# --- ALL TOOLS will now use the same, cleaned genotype files for a fair comparison ---
# The variant ID in this fileset will be 'chr:pos'
SHARED_GENOTYPE_PREFIX = CI_WORKDIR / "chr22_subset50_compat"
# We still need the original files to create the cleaned version.
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"

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
#                                     HELPER FUNCTIONS
# ========================================================================================
def print_header(title: str, char: str = "="):
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def print_debug_header(title: str):
    print("\n" + "." * 80, flush=True)
    print(f".... DEBUG: {title} ....", flush=True)
    print("." * 80, flush=True)

def download_and_extract(url: str, dest_dir: Path):
    base = url.split('?')[0]
    filename = Path(base.split('/')[-1])
    outpath = dest_dir / filename
    print(f"[SCRIPT] Downloading {filename}...", flush=True)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(outpath, 'wb') as f: shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print(f"❌ FAILED to download {url}: {e}", flush=True); sys.exit(1)

    if outpath.suffix == '.zip':
        with zipfile.ZipFile(outpath, 'r') as z: z.extractall(dest_dir)
        print(f"[SCRIPT] Extracted ZIP: {filename}", flush=True); outpath.unlink()
    elif outpath.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        print(f"[SCRIPT] Extracted GZ: {filename} -> {dest.name}", flush=True); outpath.unlink()

def run_and_measure(cmd: list, name: str):
    print_header(f"RUNNING: {name}", char='-')
    print(f"Command:\n  {' '.join(map(str, cmd))}\n", flush=True)
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
    
    print_debug_header(f"{name} Real-time Output")
    with proc.stdout:
        for line in iter(proc.stdout.readline, ''):
            print(f"  [{name}] {line.strip()}", flush=True)
    
    returncode = proc.wait()
    duration = time.perf_counter() - start
    result = {'tool': name, 'time_sec': duration, 'returncode': returncode, 'success': returncode == 0}

    if result['success']:
        print(f"✅ {name} finished in {duration:.2f}s.", flush=True)
    else:
        print(f"❌ {name} FAILED (Exit Code: {returncode})", flush=True)
    return result

def setup_environment():
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    for url, binary_path in [(PLINK1_URL, PLINK1_BINARY), (PLINK2_URL, PLINK2_BINARY)]:
        if not binary_path.exists():
            download_and_extract(url, CI_WORKDIR)
            for p in CI_WORKDIR.iterdir():
                if p.is_file() and p.name.startswith(binary_path.name.split('_')[0]):
                    p.rename(binary_path); p.chmod(0o755); print(f"[SCRIPT] Renamed and chmod: {binary_path}", flush=True); break
    for zip_name in GENOTYPE_FILES: download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    for url in PGS_SCORES.values(): download_and_extract(url, CI_WORKDIR)

def create_synchronized_genotype_files(original_prefix: Path, final_prefix: Path):
    """Creates a new, de-duplicated PLINK fileset where variant IDs are 'chr:pos'."""
    print_header("SYNCHRONIZING GENOTYPE DATA", char='.')
    original_bim = original_prefix.with_suffix('.bim')
    bim_df = pd.read_csv(original_bim, sep='\s+', header=None, names=['chrom', 'rsid', 'cm', 'pos', 'a1', 'a2'], dtype=str)
    
    # Set the variant ID (col 2) to be 'chr:pos'. This is the key insight.
    bim_df['chr_pos_id'] = bim_df['chrom'] + ':' + bim_df['pos']
    
    # We must first remove variants with duplicate chr:pos, as these cannot be used as unique IDs.
    bim_df.drop_duplicates(subset=['chr_pos_id'], keep=False, inplace=True)
    print(f"[SCRIPT] Kept {len(bim_df)} variants with unique chr:pos identifiers.", flush=True)

    bim_df_to_save = bim_df[['chrom', 'chr_pos_id', 'cm', 'pos', 'a1', 'a2']]
    bim_df_to_save.to_csv(final_prefix.with_suffix('.bim'), sep='\t', header=False, index=False)
    
    # Now, filter the .bed file to keep only these non-duplicate variants.
    variants_to_keep_file = CI_WORKDIR / "variants_to_keep.txt"
    bim_df['chr_pos_id'].to_csv(variants_to_keep_file, index=False, header=False)

    print("[SCRIPT] Filtering original BED file to match unique chr:pos variants...", flush=True)
    filter_cmd = [str(PLINK2_BINARY), "--bfile", str(original_prefix), "--extract", str(variants_to_keep_file), "--make-bed", "--out", str(final_prefix)]
    filter_proc = subprocess.run(filter_cmd, capture_output=True, text=True)
    if filter_proc.returncode != 0:
        print("❌ FAILED to filter original genotype data.", flush=True); print(filter_proc.stderr, flush=True); sys.exit(1)
    
    print(f"[SCRIPT] Successfully created de-duplicated, synchronized fileset at: {final_prefix}", flush=True)

def create_unified_scorefile(score_file: Path, pgs_id: str) -> Path:
    """Creates a unified scorefile where the variant ID is 'chr:pos'."""
    print_header(f"CREATING UNIFIED SCORE FILE FOR {pgs_id}", char='.')
    
    df = pd.read_csv(score_file, sep='\t', comment='#', usecols=['hm_chr', 'hm_pos', 'effect_allele', 'other_allele', 'effect_weight'], dtype=str, low_memory=False)
    df.dropna(inplace=True)
    df['effect_weight'] = pd.to_numeric(df['effect_weight'], errors='coerce')
    df.dropna(inplace=True)
    
    df['chr_pos_id'] = df['hm_chr'] + ':' + df['hm_pos']
    df.drop_duplicates(subset=['chr_pos_id'], keep='first', inplace=True)

    final_df = df[['chr_pos_id', 'effect_allele', 'other_allele', 'effect_weight']].copy()
    final_df.columns = ['snp_id', 'effect_allele', 'other_allele', pgs_id]
    
    out_path = CI_WORKDIR / f"{pgs_id}_unified_format.tsv"
    print(f"[SCRIPT] Writing {len(final_df)} variants to unified score file: {out_path.name}", flush=True)
    final_df.to_csv(out_path, sep='\t', index=False)
    return out_path

def validate_results(pgs_id: str, results: list):
    print_header(f"VALIDATION & COMPARISON FOR {pgs_id}", char='-')
    tool_paths = {
        'gnomon': SHARED_GENOTYPE_PREFIX.with_suffix('.sscore'),
        'plink1': CI_WORKDIR / f"plink1_{pgs_id}.profile",
        'plink2': CI_WORKDIR / f"plink2_{pgs_id}.sscore",
    }
    dfs = []
    successful_tools = [res['tool'].split('_')[0] for res in results if res['success']]

    for tool_name in successful_tools:
        path = tool_paths.get(tool_name)
        if not path or not path.exists():
            print(f"  > WARNING: Output file for {tool_name} not found at {path}", flush=True); continue
        
        try:
            df = None
            if tool_name == 'gnomon':
                df_raw = pd.read_csv(path, sep='\t')
                avg_col = next(c for c in df_raw.columns if '_AVG' in c)
                # Gnomon SUM = AVG * num_variants_used. num_variants_used = total - missing
                # We extract total from the header string, e.g. "PGS004696(18722v)_AVG"
                total_vars_in_score = float(avg_col.split('(')[1].split('v)')[0])
                miss_pct_col = next(c for c in df_raw.columns if '_MISSING_PCT' in c)
                df_raw[f'SCORE_{tool_name}'] = df_raw[avg_col] * (total_vars_in_score * (1 - df_raw[miss_pct_col] / 100.0))
                df = df_raw[['#IID', f'SCORE_{tool_name}']].rename(columns={'#IID': 'IID'})
            elif tool_name == 'plink1':
                df = pd.read_csv(path, sep='\s+')[['IID', 'SCORE']].rename(columns={'SCORE': f'SCORE_{tool_name}'})
            elif tool_name == 'plink2':
                df = pd.read_csv(path, sep='\s+')[['#IID', 'SCORE1_SUM']].rename(columns={'#IID': 'IID', 'SCORE1_SUM': f'SCORE_{tool_name}'})
            
            if df is not None: dfs.append(df)
        except Exception as e:
            print(f"  > ERROR parsing {tool_name} output from {path}: {e}", flush=True)

    if len(dfs) < 2:
        print("Could not load enough results for comparison.", flush=True); return

    merged_df = dfs[0]
    for df_to_merge in dfs[1:]: merged_df = pd.merge(merged_df, df_to_merge, on='IID')
    
    score_cols = [c for c in merged_df.columns if c.startswith('SCORE_')]
    if len(score_cols) < 2: print("Not enough comparable scores after merging.", flush=True); return

    print_debug_header("Sample of Computed SUM Scores")
    print(merged_df.head(10).to_markdown(index=False, floatfmt='.6f'), flush=True)
    print_debug_header("Score Correlation Matrix")
    print(merged_df[score_cols].corr().to_markdown(floatfmt='.8f'), flush=True)
    print_debug_header("Mean Absolute Difference")
    for col1, col2 in combinations(score_cols, 2):
        diff = (merged_df[col1] - merged_df[col2]).abs().mean()
        print(f"  > {col1} vs {col2}: {diff:.8f}", flush=True)


def main():
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    create_synchronized_genotype_files(ORIGINAL_PLINK_PREFIX, SHARED_GENOTYPE_PREFIX)

    all_results, failures = [], []
    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        raw_score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        pgs_results = []

        unified_score_file = create_unified_scorefile(raw_score_file, pgs_id)
        
        # --- Gnomon Run ---
        # It reads the chr:pos from the unified file and matches the chr:pos index it builds from the bim file.
        gnomon_cmd = [str(GNOMON_BINARY), "--score", str(unified_score_file), str(SHARED_GENOTYPE_PREFIX)]
        res_g = run_and_measure(gnomon_cmd, f"gnomon_{pgs_id}")
        all_results.append(res_g); pgs_results.append(res_g)
        if not res_g['success']: failures.append(f"{pgs_id} (gnomon_failed)")

        # --- PLINK2 Run ---
        # It reads the chr:pos ID from the bim file and matches it against the chr:pos ID in the unified file.
        out2_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        # Use column numbers: 1=snp_id, 2=effect_allele, 4=score
        plink2_cmd = [str(PLINK2_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file), "1", "2", "4", "header", "no-mean-imputation", "--out", str(out2_prefix)]
        res_p2 = run_and_measure(plink2_cmd, f"plink2_{pgs_id}")
        all_results.append(res_p2); pgs_results.append(res_p2)
        if not res_p2['success']: failures.append(f"{pgs_id} (plink2_failed)")
        
        # --- PLINK1 Run ---
        # It also reads the chr:pos ID from the bim and matches against the score file.
        out1_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        plink1_cmd = [str(PLINK1_BINARY), "--bfile", str(SHARED_GENOTYPE_PREFIX), "--score", str(unified_score_file), "1", "2", "4", "header", "no-mean-imputation", "--out", str(out1_prefix)]
        res_p1 = run_and_measure(plink1_cmd, f"plink1_{pgs_id}")
        all_results.append(res_p1); pgs_results.append(res_p1)
        if not res_p1['success']: failures.append(f"{pgs_id} (plink1_failed)")

        validate_results(pgs_id, pgs_results)

    print_header('PERFORMANCE SUMMARY')
    if all_results:
        df = pd.DataFrame([r for r in all_results if r['success']])
        if not df.empty:
            df['tool_base'] = df['tool'].str.split('_').str[0]
            summary = df.groupby('tool_base').agg(mean_time_sec=('time_sec','mean')).reset_index()
            print(summary.to_markdown(index=False, floatfmt='.3f'), flush=True)

    print_header(f"CI CHECK {'FAILED' if failures else 'PASSED'}")
    if failures:
        print('❌ Tests failed:', flush=True); [print(f'  - {f}', flush=True) for f in sorted(list(set(failures)))]; sys.exit(1)
    else:
        print('🎉 All tests passed.', flush=True); sys.exit(0)

if __name__ == '__main__':
    main()
