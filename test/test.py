import subprocess
import requests
import zipfile
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# --- Configuration ---
# Use a consistent seed for reproducibility
random.seed(2025)

# Constants for file URLs and local paths
DATA_URL_BASE = "https://github.com/SauersML/genomic_pca/blob/main/data/"
FILES_TO_DOWNLOAD = {
    "chr22_subset50.bed.zip": "chr22_subset50.bed",
    "chr22_subset50.bim.zip": "chr22_subset50.bim",
    "chr22_subset50.fam.zip": "chr22_subset50.fam",
}
DATA_DIR = Path("./test_data")
PLINK_PREFIX = DATA_DIR / "chr22_subset50"
SCORE_FILE = DATA_DIR / "scores.tsv"
GNOMON_BINARY = Path("../target/release/gnomon")

# --- Helper Functions ---

def setup_environment():
    """Create data directory, download, and unzip files."""
    print("--- 셋업: Setting up test environment ---")
    DATA_DIR.mkdir(exist_ok=True)
    for zip_name, final_name in FILES_TO_DOWNLOAD.items():
        zip_path = DATA_DIR / zip_name
        final_path = DATA_DIR / final_name
        if not final_path.exists():
            print(f"Downloading {zip_name}...")
            url = f"{DATA_URL_BASE}{zip_name}?raw=true"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Unzipping {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(DATA_DIR)
            zip_path.unlink() # Clean up zip file
    print("Environment setup complete.")

def generate_score_file():
    """
    Creates a synthetic score file from the .bim file to test multiple scores
    and allele flipping.
    """
    print("--- 셋업: Generating synthetic score file ---")
    if SCORE_FILE.exists():
        print("Score file already exists. Skipping generation.")
        return

    bim_df = pd.read_csv(
        PLINK_PREFIX.with_suffix(".bim"),
        sep='\t',
        header=None,
        names=['chr', 'snp_id', 'cm', 'pos', 'a1', 'a2']
    )

    score_data = []
    for _, row in bim_df.iterrows():
        # Randomly choose A1 or A2 as the effect allele to test flipping
        effect_allele = random.choice([row['a1'], row['a2']])
        # Generate two random scores
        score1 = random.uniform(-1.0, 1.0)
        score2 = random.uniform(-0.5, 0.5)
        score_data.append([row['snp_id'], effect_allele, score1, score2])

    score_df = pd.DataFrame(score_data, columns=['snp_id', 'effect_allele', 'PRS1', 'PRS2'])
    score_df.to_csv(SCORE_FILE, sep='\t', index=False)
    print(f"Generated score file with {len(score_df)} variants.")

def run_tool(command_list, log_prefix):
    """Executes a command and fails script on error."""
    print(f"--- 실행: Running {log_prefix} ---")
    result = subprocess.run(
        command_list,
        check=True,
        capture_output=True,
        text=True
    )
    # Write logs for debugging if needed, but keep console clean on success
    (DATA_DIR / f"{log_prefix}.stdout.log").write_text(result.stdout)
    (DATA_DIR / f"{log_prefix}.stderr.log").write_text(result.stderr)
    print(f"Completed {log_prefix}.")


def compare_results():
    """Loads and compares the output files from all tools."""
    print("--- 비교: Comparing outputs for numerical identity ---")
    
    # Load Gnomon results
    gnomon_df = pd.read_csv(
        DATA_DIR / "scores.tsv.sscore",
        sep='\t'
    ).rename(columns={"#IID": "IID"}).set_index("IID")

    # Load PLINK 2 results
    plink2_df = pd.read_csv(
        DATA_DIR / "plink2_run.sscore",
        sep='\t'
    ).rename(columns={"#IID": "IID"}).set_index("IID")[['PRS1_SUM', 'PRS2_SUM']]

    # Load PLINK 1 results
    plink1_df = pd.read_csv(
        DATA_DIR / "plink1_run.profile",
        delim_whitespace=True
    ).set_index("IID")[['SCORE']]
    # PLINK 1 doesn't support multi-score, so we only test the first score
    # It also names the column 'SCORE' by default.
    
    # --- Comparison for Score 1 ---
    print("Comparing PRS1...")
    merged1 = gnomon_df[['PRS1']].join(plink2_df[['PRS1_SUM']]).join(plink1_df[['SCORE']])
    
    # Check Gnomon vs PLINK 2
    is_close_p2 = np.isclose(merged1['PRS1'], merged1['PRS1_SUM'])
    assert is_close_p2.all(), f"Mismatch between Gnomon and PLINK 2 on PRS1!\n{merged1[~is_close_p2]}"
    print("✅ Gnomon == PLINK 2 (PRS1)")

    # Check Gnomon vs PLINK 1
    is_close_p1 = np.isclose(merged1['PRS1'], merged1['SCORE'])
    assert is_close_p1.all(), f"Mismatch between Gnomon and PLINK 1 on PRS1!\n{merged1[~is_close_p1]}"
    print("✅ Gnomon == PLINK 1 (PRS1)")
    
    # --- Comparison for Score 2 (Gnomon vs PLINK 2 only) ---
    print("\nComparing PRS2...")
    merged2 = gnomon_df[['PRS2']].join(plink2_df[['PRS2_SUM']])
    is_close_p2_s2 = np.isclose(merged2['PRS2'], merged2['PRS2_SUM'])
    assert is_close_p2_s2.all(), f"Mismatch between Gnomon and PLINK 2 on PRS2!\n{merged2[~is_close_p2_s2]}"
    print("✅ Gnomon == PLINK 2 (PRS2)")
    
# --- Main Execution Logic ---
if __name__ == "__main__":
    setup_environment()
    generate_score_file()

    # Define the commands
    cmd_gnomon = [
        str(GNOMON_BINARY),
        "--input-path", str(DATA_DIR),
        "--score", str(SCORE_FILE)
    ]
    # NOTE: PLINK 2 needs --score-col-nums to read multiple scores properly
    cmd_plink2 = [
        "plink2",
        "--bfile", str(PLINK_PREFIX),
        "--score", str(SCORE_FILE), "header", "no-mean-imputation",
        "--score-col-nums", "3-4", # Select columns 3 and 4 for scores
        "--out", str(DATA_DIR / "plink2_run")
    ]
    # NOTE: PLINK 1 only calculates one score at a time from column 3
    cmd_plink1 = [
        "plink",
        "--bfile", str(PLINK_PREFIX),
        "--score", str(SCORE_FILE), "1", "2", "3", "header", "sum", "no-mean-imputation",
        "--out", str(DATA_DIR / "plink1_run")
    ]
    
    # Run all tools
    run_tool(cmd_gnomon, "gnomon")
    run_tool(cmd_plink2, "plink2")
    run_tool(cmd_plink1, "plink1")
    
    # The final assertion
    compare_results()

    print("\n\n🎉 SUCCESS: All correctness tests passed. Outputs are numerically identical.")
    sys.exit(0)
