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
#                             CONFIGURATION
# ========================================================================================
CI_WORKDIR = Path("./ci_workdir")
GNOMON_BINARY = Path("./target/release/gnomon")
PLINK1_BINARY = CI_WORKDIR / "plink"
PLINK2_BINARY = CI_WORKDIR / "plink2"
ORIGINAL_PLINK_PREFIX = CI_WORKDIR / "chr22_subset50"

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
#                             HELPER FUNCTIONS
# ========================================================================================

def print_header(title: str, char: str = "="):
    width = 80
    print("\n" + char * width)
    print(f"{char*4} {title} {char*(width - len(title) - 6)}")
    print(char * width)

def download_and_extract(url: str, dest_dir: Path):
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

    if filename.suffix == '.zip':
        with zipfile.ZipFile(outpath, 'r') as z:
            z.extractall(dest_dir)
        print(f"Extracted ZIP: {filename}")
        outpath.unlink()
    elif filename.suffix == '.gz':
        dest = dest_dir / filename.stem
        with gzip.open(outpath, 'rb') as f_in, open(dest, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"Extracted GZ: {filename} -> {dest.name}")
        outpath.unlink()

def show_ambiguity_context(error_msg: str, score_file: Path, bim_file: Path):
    m = re.search(r"variant '([^']+)'", error_msg)
    if not m:
        return
    var = m.group(1)
    # Context in score file
    print_header(f"Context around {var} in score file", char='.')
    lines = score_file.read_text().splitlines()
    for i, line in enumerate(lines):
        if var in line:
            start = max(0, i-3)
            end = min(len(lines), i+4)
            for j in range(start, end):
                prefix = '>' if j == i else ' '
                print(f"{prefix} {j+1:5d}: {lines[j]}")
    # Context in BIM file
    print_header(f"Context around {var} in BIM file", char='.')
    bim_lines = bim_file.read_text().splitlines()
    for i, line in enumerate(bim_lines):
        if var in line:
            start = max(0, i-3)
            end = min(len(bim_lines), i+4)
            for j in range(start, end):
                prefix = '>' if j == i else ' '
                print(f"{prefix} {j+1:5d}: {bim_lines[j]}")

def run_and_measure(cmd: list, name: str, score_file: Path = None):
    print_header(f"RUNNING: {name}", char='-')
    print("Command:")
    print(f"  {' '.join(map(str, cmd))}\n")
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                peak = max(peak, p.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)
    except psutil.NoSuchProcess:
        pass
    out, err = proc.communicate()
    dur = time.perf_counter() - start
    result = {
        'tool': name,
        'time_sec': dur,
        'peak_mem_mb': peak / (1024*1024),
        'returncode': proc.returncode,
        'success': proc.returncode == 0
    }
    if result['success']:
        print(f"✅ {name} finished in {dur:.2f}s.")
    else:
        print(f"❌ {name} FAILED (Exit Code: {proc.returncode})")
        print_header("STDERR", char='.')
        print(err.strip() or "[EMPTY]")
        if 'FATAL AMBIGUITY' in err and score_file:
            show_ambiguity_context(err, score_file, ORIGINAL_PLINK_PREFIX.with_suffix('.bim'))
    return result

def setup_environment():
    print_header("ENVIRONMENT SETUP")
    CI_WORKDIR.mkdir(exist_ok=True)
    # Download PLINK binaries
    for url, binary in [(PLINK1_URL, PLINK1_BINARY), (PLINK2_URL, PLINK2_BINARY)]:
        download_and_extract(url, CI_WORKDIR)
        for p in CI_WORKDIR.iterdir():
            if p.is_file() and p.name.startswith(binary.name.split('_')[0]):
                p.rename(binary)
                binary.chmod(0o755)
                print(f"Renamed and chmod: {binary}")
                break
    # Download genotype subset
    for zip_name in GENOTYPE_FILES:
        download_and_extract(f"{GENOTYPE_URL_BASE}{zip_name}?raw=true", CI_WORKDIR)
    print_header("WORKDIR CONTENTS AFTER DOWNLOADS", char='.')
    for p in sorted(CI_WORKDIR.iterdir()):
        print(f" - {p.name}")
    # Download PGS scores
    for url in PGS_SCORES.values():
        download_and_extract(url, CI_WORKDIR)

def make_plink_format(score_file: Path, pgs_id: str) -> Path:
    """
    Read the raw PGS harmonized file, detect columns for variant ID, 
    effect allele, other allele, and weight; write out a 4-col TSV
    that PLINK2 --score will accept.
    """
    print_header(f"PREPARING PLINK FORMAT FOR {pgs_id}", char='.')
    df = pd.read_csv(score_file, sep='\t', comment='#', dtype=str)
    print(f"  Loaded PGS file with columns: {list(df.columns)}")

    # 1) Identify the variant ID column
    id_candidates = ['rsID', 'RSID', 'variant_id', 'variantID', 'ID']
    id_col = next((c for c in id_candidates if c in df.columns), None)
    if not id_col:
        raise KeyError(f"No variant ID column found among {id_candidates}")
    print(f"  Using variant-ID column: {id_col}")

    # 2) Identify effect / other allele columns
    eff_candidates = ['effect_allele', 'EA', 'A1', 'alt', 'a1']
    oth_candidates = ['other_allele', 'OA', 'A2', 'ref', 'a2']
    effect_col = next((c for c in eff_candidates if c in df.columns), None)
    other_col  = next((c for c in oth_candidates if c in df.columns), None)
    if not effect_col or not other_col:
        raise KeyError(f"Could not find effect/other allele columns")
    print(f"  Using effect-allele column: {effect_col}")
    print(f"  Using other-allele  column: {other_col}")

    # 3) Identify weight column (look for 'weight' or 'beta' or 'score')
    weight_col = next((c for c in df.columns 
                       if 'weight' in c.lower() or 'beta' in c.lower() or 'score' in c.lower()), None)
    if not weight_col:
        raise KeyError("No weight/beta/score column found")
    print(f"  Using weight column: {weight_col}")

    # 4) Subset and rename
    plink_df = df[[id_col, effect_col, other_col, weight_col]].copy()
    plink_df.columns = ['snp_id','effect_allele','other_allele','weight']
    out = CI_WORKDIR / f"{pgs_id}_plink_format.tsv"
    plink_df.to_csv(out, sep='\t', index=False)
    print(f"\n  Written PLINK2-compatible score file: {out}")

    # 5) Debug prints
    print_header("SAMPLE OF PLINK2 SCORE FILE", char='.')
    print(plink_df.head(5).to_markdown(index=False))
    print(f"\n  Total entries in score file: {len(plink_df)}")

    # 6) Check how many variant IDs actually match the .bim
    bim_path = ORIGINAL_PLINK_PREFIX.with_suffix('.bim')
    bim_ids = set()
    for line in open(bim_path):
        parts = line.split()
        if len(parts) >= 2:
            bim_ids.add(parts[1])
    matches = plink_df['snp_id'].isin(bim_ids).sum()
    print(f"  Variants matching BIM: {matches} / {len(plink_df)}")

    return out

def main():
    print_header("GNOMON CI TEST & BENCHMARK SUITE")
    setup_environment()

    all_results, failures = [], []
    for pgs_id, url in PGS_SCORES.items():
        print_header(f"TESTING SCORE: {pgs_id}")
        score_file = CI_WORKDIR / Path(url.split('/')[-1]).stem
        print(f"Raw PGS file: {score_file}")

        # 1) Gnomon
        res_g = run_and_measure([
            str(GNOMON_BINARY), "--score", str(score_file), str(ORIGINAL_PLINK_PREFIX)
        ], f"gnomon_{pgs_id}", score_file)
        all_results.append(res_g)
        if not res_g['success']:
            failures.append(f"{pgs_id} (gnomon_failed)")
            continue

        # 2) Prepare PLINK2-compatible score file
        try:
            fmt_file = make_plink_format(score_file, pgs_id)
        except Exception as e:
            print(f"❌ Error preparing PLINK2 format: {e}")
            failures.append(f"{pgs_id} (format_error)")
            continue

        # 3) PLINK2
        out2 = CI_WORKDIR / f"plink2_{pgs_id}"
        res_p2 = run_and_measure([
            str(PLINK2_BINARY),
            "--bfile", str(ORIGINAL_PLINK_PREFIX),
            "--score", str(fmt_file), "1", "2", "3", "header", "no-mean-imputation",
            "--out", str(out2)
        ], f"plink2_{pgs_id}", fmt_file)
        all_results.append(res_p2)
        if not res_p2['success']:
            failures.append(f"{pgs_id} (plink2_failed)")
            continue

        # 4) PLINK1: transform and score
        df = pd.read_csv(fmt_file, sep='\t')
        df['snp_id'] = df['snp_id'].astype(str)
        compat = CI_WORKDIR / f"{pgs_id}_plink1_format.tsv"
        df.to_csv(compat, sep='\t', index=False)
        print(f"Written PLINK1-compatible score file: {compat}")

        out1 = CI_WORKDIR / f"plink1_{pgs_id}"
        res_p1 = run_and_measure([
            str(PLINK1_BINARY),
            "--bfile", str(ORIGINAL_PLINK_PREFIX),
            "--score", str(compat), "1", "2", "3", "header", "sum",
            "--out", str(out1)
        ], f"plink1_{pgs_id}", compat)
        all_results.append(res_p1)
        if not res_p1['success']:
            failures.append(f"{pgs_id} (plink1_failed)")
            continue

        # 5) Validation (as before) …
        # [Your existing validation logic here, unchanged.]

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
            print(summary.to_markdown(index=False,floatfmt='.3f'))

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
