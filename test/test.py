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
# --- All tools will now use the same, cleaned genotype files for a fair comparison ---
# This is the single source of truth for genotype data.
SHARED_GENOTYPE_PREFIX = CI_WORKDIR / "chr22_subset50_plink_compat"

GNOMON_BINARY = Path("./target/release/gnomon")
PLINK1_BINARY = CI_WORKDIR / "plink"
PLINK2_BINARY = CI_WORKDIR / "plink2"

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
    """Prints a formatted header to the console."""
    width = 80
    print("\n" + char * width, flush=True)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}", flush=True)
    print(char * width, flush=True)

def print_debug_header(title: str):
    """Prints a smaller header for debug sections."""
    print("\n" + "." * 80, flush=True)
    print(f".... DEBUG: {title} ....", flush=True)
    print("." * 80, flush=True)

def download_and_extract(url: str, dest_dir: Path):
    """Downloads and extracts a file, handling .zip and .gz."""
    base = url.split('?')[0]
    filename = Path(base.split('/')[-1])
    outpath = dest_dir / filename
    print(f"[SCRIPT] Downloading {filename}...", flush=True)
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(outpath, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    except Exception as e:
        print(f"❌ FAILED to download {url}: {e}", flush=True); sys.exit(1)

    if outpath.suffix == '.zip':
        with zipfile.ZipFile(outpath, 'r') as z: z.extractall(dest_dir)
        print(f"[SCRIPT] Extracted ZIP: {filename}", flush=True); outpath.unlink()
    elif outpath.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out: shutil.copyfileobj(f_in, f_out)
        print(f"[SCRIPT] Extracted GZ: {filename} -> {dest.name}", flush=True); outpath.unlink()


def run_and_measure(cmd: list, name: str, out_prefix: Path):
    """Executes a command, streaming its output in real-time, and measures performance."""
    print_header(f"RUNNING: {name}", char='-')
    print(f"Command:\n  {' '.join(map(str, cmd))}\n", flush=True)
    start = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1
    )

    print_debug_header(f"{name} Real-time Output")
    with proc.stdout:
        for line in iter(proc.stdout.readline, ''):
            print(f"  [{name}] {line.strip()}", flush=True)

    returncode = proc.wait()
    duration = time.perf_counter() - start

    peak_mem = 0
    # Note: peak memory measurement is not very reliable with Popen/wait
    try:
        p = psutil.Process(proc.pid)
        peak_mem = p.memory_info().rss
    except psutil.NoSuchProcess:
        pass

    result = {'tool': name, 'time_sec': duration, 'peak_mem_mb': peak_mem / (1024*1024), 'returncode': returncode, 'success': returncode == 0}

    if result['success']:
        print(f"✅ {name} finished in {duration:.2f}s.", flush=True)
    else:
        print(f"❌ {name} FAILED (Exit Code: {returncode})", flush=True)

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
                    p.rename(binary_path); p.chmod(0o755); print(f"[SCRIPT] Renamed and chmod: {binary_path}", flush=True); break

    for zip_name in GENOTYPE_FILES: download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    for url in PGS_SCORES.values(): download_and_extract(url, CI_WORKDIR)

    print_debug_header("WORKDIR CONTENTS")
    for root, _, files in os.walk(CI_WORKDIR):
        if Path(root) == CI_WORKDIR:
             for name in sorted(files): print(f" - {name}", flush=True)

def create_synchronized_genotype_files(original_prefix: Path, final_prefix: Path):
    """Creates a new, de-duplicated PLINK fileset with unique, allele-aware variant IDs."""
    print_header("SYNCHRONIZING GENOTYPE VARIANT IDs FOR PLINK", char='.')

    temp_prefix = CI_WORKDIR / "temp_bim_for_dedup"
    original_bim = original_prefix.with_suffix('.bim')
    temp_bim = temp_prefix.with_suffix('.bim')

    print(f"[SCRIPT] Reading original bim file: {original_bim}", flush=True)
    bim_df = pd.read_csv(original_bim, sep='\s+', header=None, names=['chrom', 'rsid', 'cm', 'pos', 'a1', 'a2'], dtype=str)

    print("[SCRIPT] Creating canonical allele-aware IDs (chr_pos_sorted_a1_sorted_a2)...", flush=True)
    alleles = bim_df[['a1', 'a2']].apply(lambda x: sorted(x), axis=1, result_type='expand')
    bim_df['canonical_id'] = bim_df['chrom'] + '_' + bim_df['pos'] + '_' + alleles[0] + '_' + alleles[1]

    bim_df_to_save = bim_df[['chrom', 'canonical_id', 'cm', 'pos', 'a1', 'a2']]
    bim_df_to_save.to_csv(temp_bim, sep='\t', header=False, index=False)

    shutil.copy(original_prefix.with_suffix('.bed'), temp_prefix.with_suffix('.bed'))
    shutil.copy(original_prefix.with_suffix('.fam'), temp_prefix.with_suffix('.fam'))
    print(f"[SCRIPT] Created temporary fileset at {temp_prefix} with new IDs.", flush=True)

    print("[SCRIPT] Using PLINK2 to remove duplicate variants from the genotype data...", flush=True)
    dedup_cmd = [str(PLINK2_BINARY), "--bfile", str(temp_prefix), "--rm-dup", "force-first", "--make-bed", "--out", str(final_prefix)]
    dedup_proc = subprocess.run(dedup_cmd, capture_output=True, text=True)
    if dedup_proc.returncode != 0:
        print("❌ FAILED to de-duplicate genotype data.", flush=True); print(dedup_proc.stderr, flush=True); sys.exit(1)

    print(f"[SCRIPT] Successfully created de-duplicated, synchronized fileset at: {final_prefix}", flush=True)

    for suffix in ['.bed', '.bim', '.fam', '.log', '.nosex']:
        if (temp_prefix.with_suffix(suffix)).exists():
            (temp_prefix.with_suffix(suffix)).unlink()


def create_plink_formatted_scorefile(score_file: Path, pgs_id: str) -> Path:
    """Reads a raw PGS file and creates a clean, de-duplicated, PLINK-compatible score file."""
    print_header(f"PREPARING PLINK FORMAT FOR {pgs_id}", char='.')

    cols_to_use = ['hm_chr', 'hm_pos', 'effect_allele', 'other_allele', 'effect_weight']
    print(f"[SCRIPT] Loading raw data from {score_file.name}...", flush=True)
    df = pd.read_csv(score_file, sep='\t', comment='#', usecols=lambda c: c in cols_to_use, dtype=str, low_memory=False)
    print(f"  > Loaded {len(df)} raw entries.", flush=True)

    print("[SCRIPT] Starting data cleaning pipeline...", flush=True)
    df.dropna(subset=cols_to_use, inplace=True, ignore_index=True)
    print(f"  > Step 1: Kept {len(df)} rows with all required columns.", flush=True)

    df['effect_weight'] = pd.to_numeric(df['effect_weight'], errors='coerce')
    df.dropna(subset=['effect_weight'], inplace=True, ignore_index=True)
    print(f"  > Step 2: Kept {len(df)} rows after removing non-numeric weights.", flush=True)

    alleles = df[['effect_allele', 'other_allele']].apply(lambda x: sorted(x), axis=1, result_type='expand')
    df['canonical_id'] = df['hm_chr'] + '_' + df['hm_pos'] + '_' + alleles[0] + '_' + alleles[1]
    print(f"  > Step 3: Created canonical allele-aware IDs.", flush=True)

    df.drop_duplicates(subset=['canonical_id'], keep='first', inplace=True, ignore_index=True)
    print(f"  > Step 4: Kept {len(df)} rows after de-duplicating by canonical ID.", flush=True)

    out_path = CI_WORKDIR / f"{pgs_id}_plink_format.tsv"
    final_df = df[['canonical_id', 'effect_allele', 'effect_weight']].copy()
    final_df.columns = ['ID', 'EFFECT_ALLELE', 'WEIGHT']

    print(f"[SCRIPT] Writing clean score file to {out_path}...", flush=True)
    final_df.to_csv(out_path, sep='\t', index=False, na_rep='NA')
    print(f"  > Done. Final file has {len(final_df)} unique, clean variants.", flush=True)
    return out_path


def validate_results(pgs_id: str, results: list):
    """Loads successful tool outputs, merges them, and calculates correlation and differences."""
    print_header(f"VALIDATION & COMPARISON FOR {pgs_id}", char='-')

    ### CHANGE ###
    # This dictionary maps the tool name to its *actual* output file path.
    # We now use the unique output prefixes for each run.
    all_tool_paths = {
        'gnomon': CI_WORKDIR / f"gnomon_{pgs_id}.sscore",
        'plink1': CI_WORKDIR / f"plink1_{pgs_id}.profile",
        'plink2': CI_WORKDIR / f"plink2_{pgs_id}.sscore",
    }

    successful_tools = {}
    for res in results:
        if res['success']:
            tool_name = res['tool'].split('_')[0]
            if tool_name in all_tool_paths:
                successful_tools[tool_name] = all_tool_paths[tool_name]

    if len(successful_tools) < 2:
        print("Fewer than two tools succeeded, cannot perform comparison.", flush=True)
        return

    print("Comparing scores from successful tools:", list(successful_tools.keys()), flush=True)

    data_frames = []
    for tool, path in successful_tools.items():
        if not path.exists():
            print(f"  > WARNING: Output file for {tool} not found at {path}", flush=True)
            continue

        try:
            df = None
            ### CHANGE ### More robust parsing of output files.
            if tool == 'gnomon':
                df_raw = pd.read_csv(path, sep='\t')
                id_col = '#IID' if '#IID' in df_raw.columns else 'IID'
                score_col = next((c for c in df_raw.columns if c.endswith('_AVG')), None)
                if not score_col: raise KeyError("Gnomon score column not found.")
                df = df_raw[[id_col, score_col]].rename(columns={id_col: 'IID', score_col: f'SCORE_{tool}'})
            elif tool == 'plink1':
                df_raw = pd.read_csv(path, sep='\s+')
                df = df_raw[['IID', 'SCORE']].rename(columns={'SCORE': f'SCORE_{tool}'})
            elif tool == 'plink2':
                df_raw = pd.read_csv(path, sep='\s+')
                id_col = '#IID' if '#IID' in df_raw.columns else 'IID'
                score_col = next((c for c in df_raw.columns if 'SCORE' in c), None)
                if not score_col: raise KeyError("PLINK2 score column not found.")
                df = df_raw[[id_col, score_col]].rename(columns={id_col: 'IID', score_col: f'SCORE_{tool}'})

            if df is not None: data_frames.append(df)
        except Exception as e:
            print(f"  > ERROR: Failed to parse output for {tool} from {path}: {e}", flush=True)

    if len(data_frames) < 2:
        print("Could not load enough valid dataframes for comparison.", flush=True)
        return

    merged_df = data_frames[0]
    for df_to_merge in data_frames[1:]:
        merged_df = pd.merge(merged_df, df_to_merge, on='IID')

    if merged_df.empty or merged_df.shape[1] < 3:
        print("Could not merge results for comparison.", flush=True)
        return

    score_cols = [col for col in merged_df.columns if 'SCORE' in col]

    print_debug_header("Sample of Computed Scores per Individual")
    print(merged_df.head(10).to_markdown(index=False, floatfmt='.6f'), flush=True)

    print_debug_header("Score Correlation Matrix")
    print(merged_df[score_cols].corr().to_markdown(floatfmt='.8f'), flush=True)

    print_debug_header("Mean Absolute Difference")
    for col1, col2 in combinations(score_cols, 2):
        abs_diff = (merged_df[col1] - merged_df[col2]).abs().mean()
        print(f"  > {col1} vs {col2}: {abs_diff:.6f}", flush=True)


def main():
    """Main function to run the CI test suite."""
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()
    # This one-time setup creates the clean, shared genotype data
    create_synchronized_genotype_files(ORIGINAL_PLINK_PREFIX, SHARED_GENOTYPE_PREFIX)

    all_results, failures = [], []
    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        raw_score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        pgs_results = []
        common_variants_file = None

        # --- STEP 1: Prepare a clean scoring file for all tools ---
        try:
            plink_fmt_file = create_plink_formatted_scorefile(raw_score_file, pgs_id)
        except Exception as e:
            print(f"❌ Error preparing PLINK format for {pgs_id}: {e}", flush=True)
            failures.append(f"{pgs_id} (format_error)")
            continue

        ### CHANGE ### Test logic is reordered to establish the "source of truth" first.

        # --- STEP 2: Run PLINK2 to define the exact set of variants to be used by everyone ---
        out2_prefix = CI_WORKDIR / f"plink2_{pgs_id}"
        plink2_cmd = [
            str(PLINK2_BINARY),
            "--bfile", str(SHARED_GENOTYPE_PREFIX),
            "--score", str(plink_fmt_file), "header", "no-mean-imputation", "list-variants",
            "--out", str(out2_prefix)
        ]
        res_p2 = run_and_measure(plink2_cmd, f"plink2_{pgs_id}", out2_prefix)
        all_results.append(res_p2); pgs_results.append(res_p2)

        if not res_p2['success']:
            failures.append(f"{pgs_id} (plink2_failed)")
            # If plink2 fails, we can't define the common variant set, so we can't continue this test.
            validate_results(pgs_id, pgs_results)
            continue
        else:
            # The list of variants that PLINK2 actually used is our source of truth.
            common_variants_file = out2_prefix.with_suffix('.sscore.vars')
            if not common_variants_file.exists():
                print(f"❌ CRITICAL: PLINK2 succeeded but did not produce the variant list at {common_variants_file}", flush=True)
                failures.append(f"{pgs_id} (plink2_no_vars_file)")
                continue

            print_debug_header(f"Common Variant List Established by PLINK2 for {pgs_id}")
            variant_count = sum(1 for line in open(common_variants_file))
            print(f"  > Found {variant_count} common variants to be used for all tools.", flush=True)
            print(f"  > File location: {common_variants_file}", flush=True)

        # --- STEP 3: Run Gnomon on the same genotype data, restricted to the common variants ---
        out_g_prefix = CI_WORKDIR / f"gnomon_{pgs_id}"
        # We now use the raw score file (Gnomon is robust) but crucially filter using the plink2-generated list.
        gnomon_cmd = [
            str(GNOMON_BINARY),
            "--score", str(raw_score_file),
            "--bfile", str(SHARED_GENOTYPE_PREFIX), # Use the same clean genotype data
            "--extract", str(common_variants_file), # Force use of the same variants
            "--out", str(out_g_prefix)
        ]
        res_g = run_and_measure(gnomon_cmd, f"gnomon_{pgs_id}", out_g_prefix)
        all_results.append(res_g); pgs_results.append(res_g)
        if not res_g['success']: failures.append(f"{pgs_id} (gnomon_failed)")


        # --- STEP 4: Run PLINK1 on the same data, restricted to the common variants ---
        out1_prefix = CI_WORKDIR / f"plink1_{pgs_id}"
        plink1_cmd = [
            str(PLINK1_BINARY),
            "--bfile", str(SHARED_GENOTYPE_PREFIX), # Use the same clean genotype data
            "--score", str(plink_fmt_file), "1", "2", "3", "header", "no-mean-imputation",
            "--extract", str(common_variants_file), # Force use of the same variants
            "--out", str(out1_prefix)
        ]
        res_p1 = run_and_measure(plink1_cmd, f"plink1_{pgs_id}", out1_prefix)
        all_results.append(res_p1); pgs_results.append(res_p1)
        if not res_p1['success']: failures.append(f"{pgs_id} (plink1_failed)")


        # --- STEP 5: Validate and compare results for this PGS ID ---
        validate_results(pgs_id, pgs_results)

    print_header('PERFORMANCE SUMMARY')
    if all_results:
        df = pd.DataFrame(all_results)
        df['tool_base'] = df['tool'].str.split('_').str[0]
        summary = df[df['success']].groupby('tool_base').agg(
            mean_time_sec=('time_sec','mean'),
            mean_mem_mb=('peak_mem_mb','mean'),
            runs=('tool_base', 'count')
        ).reset_index()
        if not summary.empty: print(summary.to_markdown(index=False, floatfmt='.3f'), flush=True)

    print_header(f"CI CHECK {'FAILED' if failures else 'PASSED'}")
    if failures:
        print('❌ Tests failed:', flush=True); [print(f'  - {f}', flush=True) for f in sorted(list(set(failures)))]; sys.exit(1)
    else:
        print('🎉 All tests passed.', flush=True); sys.exit(0)

if __name__ == '__main__':
    main()
