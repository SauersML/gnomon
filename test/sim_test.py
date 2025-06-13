import numpy as np
import pandas as pd
import sys
import os
import subprocess
import requests
import zipfile
import shutil
from pathlib import Path

# --- Configuration Parameters ---
N_VARIANTS = 500_000
N_INDIVIDUALS = 50
CHR = '22'
CHR_LENGTH = 40_000_000
ALT_EFFECT_PROB = 0.8
MISSING_RATE = 0.02

# --- CI/Validation Configuration ---
# Create an isolated workspace to prevent conflicts with other scripts.
WORKDIR = Path("./sim_workdir")
OUTPUT_PREFIX = WORKDIR / "simulated_data"

# --- Annotation simulation parameters ---
ANNOT_MEAN_DIST = 30
ANNOT_STD_DEV = 10
ANNOT_MIN_DIST = 2
ANNOT_MAX_DIST = 500

def generate_variants_and_weights():
    """
    FIRST: Derives normally and randomly distributed weights over 500K positions.
    """
    print(f"Step 1: Simulating {N_VARIANTS} variants on chr{CHR}...")

    positions = np.random.choice(np.arange(1, CHR_LENGTH + 1), N_VARIANTS, replace=False)
    positions.sort()

    alleles = np.array(['A', 'C', 'G', 'T'])
    ref_idx = np.random.randint(0, 4, N_VARIANTS)
    alt_offset = np.random.randint(1, 4, N_VARIANTS)
    alt_idx = (ref_idx + alt_offset) % 4
    
    ref_alleles = alleles[ref_idx]
    alt_alleles = alleles[alt_idx]

    is_alt_effect = np.random.rand(N_VARIANTS) < ALT_EFFECT_PROB
    effect_alleles = np.where(is_alt_effect, alt_alleles, ref_alleles)
    effect_weights = np.random.normal(loc=0, scale=0.05, size=N_VARIANTS)

    variants_df = pd.DataFrame({
        'chr': CHR,
        'pos': positions,
        'ref': ref_alleles,
        'alt': alt_alleles,
        'effect_allele': effect_alleles,
        'effect_weight': effect_weights
    })

    print("...Variant and weight simulation complete.")
    return variants_df

def generate_genotypes(variants_df):
    """
    THEN: Creates genotypes for 50 individuals based on HWE.
    """
    print(f"Step 2: Simulating genotypes for {N_INDIVIDUALS} individuals...")
    n_vars = len(variants_df)
    p = np.random.uniform(0.01, 0.99, size=n_vars)
    q = 1 - p
    hwe_probs = np.vstack([p**2, 2*p*q, q**2]).T
    rand_draws = np.random.rand(n_vars, N_INDIVIDUALS)
    cum_probs = hwe_probs.cumsum(axis=1)
    genotypes = (rand_draws > cum_probs[:, 0, np.newaxis]) + \
                (rand_draws > cum_probs[:, 1, np.newaxis])

    print("...Genotype simulation complete.")
    return genotypes.astype(int)

def simulate_annotations():
    """
    THEN: Simulates annotation positions along the chromosome.
    """
    print("Step 3: Simulating annotation positions (for demonstration)...")
    annotations = []
    current_pos = 1
    while current_pos <= CHR_LENGTH:
        annotations.append(current_pos)
        step = np.random.normal(ANNOT_MEAN_DIST, ANNOT_STD_DEV)
        step = np.clip(step, ANNOT_MIN_DIST, ANNOT_MAX_DIST)
        current_pos += int(round(step))
    
    print(f"...Generated {len(annotations)} annotations.")
    return annotations

def introduce_missingness(genotypes):
    """
    Inserts 2% random missingness into the genotype matrix.
    """
    print(f"Step 4: Introducing {MISSING_RATE*100:.1f}% missingness...")
    n_variants, n_individuals = genotypes.shape
    missing_mask = np.random.rand(n_variants, n_individuals) < MISSING_RATE
    genotypes_with_missing = genotypes.astype(float)
    genotypes_with_missing[missing_mask] = -1
    
    print("...Missingness introduced.")
    return genotypes_with_missing.astype(int)

def calculate_ground_truth_prs(genotypes, variants_df):
    """
    THEN: Calculates the 'ground truth' polygenic score for each individual.
    """
    print("Step 5: Calculating ground truth polygenic scores...")
    n_variants, n_individuals = genotypes.shape
    results = []
    is_alt_effect = (variants_df['effect_allele'] == variants_df['alt']).values

    for i in range(n_individuals):
        ind_genotypes = genotypes[:, i]
        valid_mask = ind_genotypes != -1
        valid_genotypes = ind_genotypes[valid_mask]
        
        if len(valid_genotypes) == 0:
            results.append({'FID': f"sample_{i+1}", 'IID': f"sample_{i+1}", 'PRS_AVG': 0})
            continue

        valid_weights = variants_df['effect_weight'][valid_mask].values
        dosages = np.zeros_like(valid_genotypes, dtype=float)
        valid_is_alt_effect = is_alt_effect[valid_mask]
        
        dosages[valid_is_alt_effect] = valid_genotypes[valid_is_alt_effect]
        dosages[~valid_is_alt_effect] = 2 - valid_genotypes[~valid_is_alt_effect]

        variant_count = len(valid_genotypes)
        score_sum = np.sum(dosages * valid_weights)
        score_avg = score_sum / variant_count if variant_count > 0 else 0
        
        results.append({
            'FID': f"sample_{i+1}", 
            'IID': f"sample_{i+1}", 
            'PRS_AVG': score_avg
        })
        
    print("...PRS calculation complete.")
    return pd.DataFrame(results)

def write_output_files(prs_results, variants_df, genotypes_with_missing, prefix: Path):
    """
    Writes all specified output files to the workspace directory.
    """
    print(f"Step 6: Writing all output files to prefix '{prefix}'...")

    # a. Ground Truth .sscore file
    sscore_filename = prefix.with_suffix(".truth.sscore")
    prs_results_ordered = prs_results[['FID', 'IID', 'PRS_AVG']]
    with open(sscore_filename, 'w') as f:
        f.write('#' + '\t'.join(prs_results_ordered.columns) + '\n')
        prs_results_ordered.to_csv(f, sep='\t', header=False, index=False, na_rep='NA')
    print(f"...Ground truth PRS results written to {sscore_filename}")

    # b. Gnomon-native Scorefile
    gnomon_scorefile = prefix.with_suffix(".gnomon.score")
    gnomon_df = variants_df.copy()
    gnomon_df['snp_id'] = gnomon_df['chr'].astype(str) + ':' + gnomon_df['pos'].astype(str)
    gnomon_df['other_allele'] = np.where(gnomon_df['effect_allele'] == gnomon_df['ref'], gnomon_df['alt'], gnomon_df['ref'])
    gnomon_df.rename(columns={'effect_weight': 'simulated_score'}, inplace=True)
    gnomon_df[['snp_id', 'effect_allele', 'other_allele', 'simulated_score']].to_csv(gnomon_scorefile, sep='\t', index=False)
    print(f"...Gnomon-native scorefile written to {gnomon_scorefile}")

    # c. PLINK fileset (.bed, .bim, .fam)
    with open(prefix.with_suffix(".fam"), 'w') as f:
        for i in range(N_INDIVIDUALS):
            f.write(f"sample_{i+1} sample_{i+1} 0 0 0 -9\n")
    
    bim_df = pd.DataFrame({
        'chr': variants_df['chr'],
        'id': variants_df['chr'].astype(str) + ':' + variants_df['pos'].astype(str),
        'cm': 0, 'pos': variants_df['pos'],
        'a1': variants_df['ref'], 'a2': variants_df['alt']
    })
    bim_df.to_csv(prefix.with_suffix(".bim"), sep='\t', header=False, index=False)
    
    code_map = {0: 0b00, -1: 0b01, 1: 0b10, 2: 0b11}
    with open(prefix.with_suffix(".bed"), 'wb') as f:
        f.write(bytes([0x6c, 0x1b, 0x01]))
        for i in range(genotypes_with_missing.shape[0]):
            for j in range(0, N_INDIVIDUALS, 4):
                byte = 0
                for k, geno in enumerate(genotypes_with_missing[i, j:j+4]):
                    byte |= (code_map[geno] << (k * 2))
                f.write(byte.to_bytes(1, 'little'))
    
    print(f"...PLINK files written: {prefix}.bed/.bim/.fam")

# Simple Dosage Test
def run_simple_dosage_test(workdir: Path, gnomon_path: Path, plink_path: Path, run_cmd_func):
    """
    Runs a minimal, hardcoded test case with 2 variants and 3 individuals.
    This test verifies basic dosage calculation AND correct handling of a missing genotype.
    """
    prefix = workdir / "simple_test"
    print("\n" + "="*80)
    print("= Running Simple Dosage Test Case (with missingness)")
    print("="*80)
    print(f"Test files will be prefixed: {prefix}")

    # --- 1. Define Ground Truth and File Contents ---
    # Manually calculated ground truth:
    # Scores are averaged over non-missing variants for each person.
    # Person 1 (hom_ref): (0*0.5 + 0*-0.2) / 2 = 0.0
    # Person 2 (het):     (1*0.5) / 1 = 0.5  (variant 2 is missing)
    # Person 3 (hom_alt): (2*0.5 + 2*-0.2) / 2 = (1.0 - 0.4) / 2 = 0.3
    truth_df = pd.DataFrame({
        'IID': ['id_hom_ref', 'id_het', 'id_hom_alt'],
        'SCORE_TRUTH': [0.0, 0.5, 0.3]
    })

    # --- 2. Write Input Files and Print Contents ---
    print("\n--- Writing Input Files ---")
    variant1_id = "1:1000"
    variant2_id = "1:2000"

    # .fam file
    fam_content = "id_hom_ref id_hom_ref 0 0 0 -9\nid_het id_het 0 0 0 -9\nid_hom_alt id_hom_alt 0 0 0 -9\n"
    with open(prefix.with_suffix(".fam"), "w") as f:
        f.write(fam_content)
    print(f"\nContents of {prefix.with_suffix('.fam')}:\n---\n{fam_content.strip()}\n---")

    # .bim file - two variants
    bim_content = f"1\t{variant1_id}\t0\t1000\tA\tG\n1\t{variant2_id}\t0\t2000\tC\tT\n"
    with open(prefix.with_suffix(".bim"), "w") as f:
        f.write(bim_content)
    print(f"\nContents of {prefix.with_suffix('.bim')}:\n---\n{bim_content.strip()}\n---")

    # score file - two variants
    score_content = (f"snp_id\teffect_allele\tother_allele\tsimple_score\n"
                     f"{variant1_id}\tG\tA\t0.5\n"
                     f"{variant2_id}\tT\tC\t-0.2\n")
    with open(prefix.with_suffix(".score"), "w") as f:
        f.write(score_content)
    print(f"\nContents of {prefix.with_suffix('.score')}:\n---\n{score_content.strip()}\n---")

    # .bed file (magic bytes + 2 variants for 3 people)
    # Variant 1 Genos: 0/0 (0b00), 0/1 (0b10), 1/1 (0b11). Byte is 00111000 -> 0x38
    # Variant 2 Genos: 0/0 (0b00), missing (0b01), 1/1 (0b11). Byte is 00110100 -> 0x34
    bed_bytes = bytes([0x6c, 0x1b, 0x01, 0x38, 0x34])
    with open(prefix.with_suffix(".bed"), "wb") as f:
        f.write(bed_bytes)
    print(f"\nContents of {prefix.with_suffix('.bed')} (in hex):\n---\n{bed_bytes.hex()}\n---")

    # --- 3. Run Tools ---
    run_cmd_func(
        [gnomon_path, "--score", prefix.with_suffix(".score").name, prefix.name],
        "Simple Gnomon Test", cwd=workdir
    )
    # The 'no-mean-imputation' flag is critical here. It tells PLINK to treat missing
    # genotypes as having zero contribution and to correctly adjust the denominator.
    run_cmd_func(
        [f"./{plink_path.name}", "--bfile", prefix.name, "--score", prefix.with_suffix(".score").name, "1", "2", "4", "header", "no-mean-imputation", "--out", "simple_plink_results"],
        "Simple PLINK2 Test", cwd=workdir
    )

    # --- 4. Compare Results ---
    print("\n--- Analyzing Simple Test Results ---")
    gnomon_results = pd.read_csv(workdir / "simple_test.sscore", sep='\t').rename(columns={'#IID': 'IID', 'simple_score_AVG': 'SCORE_GNOMON'})[['IID', 'SCORE_GNOMON']]
    plink_results_raw = pd.read_csv(workdir / "simple_plink_results.sscore", sep='\s+').rename(columns={'#IID': 'IID'})
    # PLINK's AVG is per-allele, not per-variant. We must multiply by 2 for comparison.
    plink_results = plink_results_raw.assign(SCORE_PLINK2=plink_results_raw['SCORE1_AVG'] * 2.0)[['IID', 'SCORE_PLINK2']]

    merged_df = pd.merge(truth_df, gnomon_results, on='IID').merge(plink_results, on='IID')
    print("\n--- Comparison of Scores ---")
    print(merged_df.to_markdown(index=False, floatfmt=".6f"))

    # --- 5. Final Validation ---
    is_gnomon_ok = np.allclose(merged_df['SCORE_TRUTH'], merged_df['SCORE_GNOMON'])
    is_plink_ok = np.allclose(merged_df['SCORE_TRUTH'], merged_df['SCORE_PLINK2'])

    if is_gnomon_ok and is_plink_ok:
        print("\n✅ Simple Dosage Test Successful: All tools produced the correct scores.")
    else:
        print("\n❌ Simple Dosage Test FAILED:")
        if not is_gnomon_ok: print("  - Gnomon scores do not match ground truth.")
        if not is_plink_ok: print("  - PLINK2 scores do not match ground truth.")
        sys.exit(1)

def run_and_validate_tools():
    """
    Downloads tools, runs them on the simulated data, and compares results.
    """
    # --- Configuration ---
    # URLs and paths are now relative to the isolated WORKDIR
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = WORKDIR / "plink2"
    # The gnomon binary is expected to be in the repo root relative to this script's execution
    GNOMON_BINARY_PATH = Path("./target/release/gnomon").resolve()

    def _print_header(title: str, char: str = "-"):
        width = 70
        print(f"\n{char*4} {title} {'-'*(width - len(title) - 5)}")

    def run_command(cmd: list, step_name: str, cwd: Path):
        _print_header(f"Executing: {step_name}")
        # Convert all command parts to strings for subprocess
        cmd_str = [str(c) for c in cmd]
        print(f"  > Command: {' '.join(cmd_str)}")
        print(f"  > CWD: {cwd}")
        print("--- OUTPUT ---")
        
        try:
        # Create a new environment for the subprocess, inheriting from the current one
        proc_env = os.environ.copy()
        # If we're running gnomon, force a backtrace on panic/crash
        if "gnomon" in str(cmd_str[0]):
            proc_env["RUST_BACKTRACE"] = "1"

        try:
            # Use Popen for real-time output, passing the modified environment
            process = subprocess.Popen(
                cmd_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', cwd=cwd,
                env=proc_env
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line, end='')  # end='' because line already has newline
                
            # Wait for process to complete
            return_code = process.wait()
            
            if return_code == 0:
                print(f"\n  > Success.")
            else:
                print(f"\n  > ❌ ERROR: {step_name} failed with exit code {return_code}.")
                sys.exit(1)
                
        except FileNotFoundError:
            print(f"  > ❌ ERROR: Command '{cmd[0]}' not found.")
            sys.exit(1)

    def setup_tools():
        _print_header("Step A: Setting up tools")
        if not GNOMON_BINARY_PATH.exists():
            print(f"  > ❌ ERROR: Gnomon binary not found at '{GNOMON_BINARY_PATH}'.")
            print("  > Please ensure 'cargo build --release' has completed in the parent directory.")
            sys.exit(1)
        print(f"  > Found 'gnomon' executable at: {GNOMON_BINARY_PATH}")

        if PLINK2_BINARY_PATH.exists():
            print("  > 'plink2' executable already exists. Skipping download.")
            return

        print(f"  > Downloading PLINK2 to '{PLINK2_BINARY_PATH}'...")
        zip_path = WORKDIR / "plink.zip"
        try:
            with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f: shutil.copyfileobj(r.raw, f)
            with zipfile.ZipFile(zip_path, 'r') as z:
                # Robustly find and extract just the binary
                for member in z.infolist():
                    if member.filename.endswith('plink2') and not member.is_dir():
                        with z.open(member) as source, open(PLINK2_BINARY_PATH, 'wb') as target:
                            shutil.copyfileobj(source, target)
                        break
            zip_path.unlink()
            PLINK2_BINARY_PATH.chmod(0o755)
            print("  > PLINK2 downloaded and extracted successfully.")
        except Exception as e:
            print(f"  > ❌ FAILED to download or extract PLINK2: {e}")
            sys.exit(1)

    def analyze_and_compare_large_scale_results():
        _print_header("Step D: Analyzing and Comparing Large-Scale Simulation Results")
        
        def _debug_print_df(df: pd.DataFrame, name: str):
            """Helper to print dataframe info for debugging."""
            print(f"\n--- DEBUG: Loaded '{name}' ---")
            print("Columns:", df.columns.tolist())
            print("Shape:", df.shape)
            print(df.head(3).to_markdown(index=False, floatfmt=".6f"))
            print("-" * 30)

        try:
            # --- Load truth_df ---
            truth_df_raw = pd.read_csv(OUTPUT_PREFIX.with_suffix(".truth.sscore"), sep='\t')
            _debug_print_df(truth_df_raw, "truth_df (raw)")
            if '#FID' in truth_df_raw.columns:
                truth_df_raw.rename(columns={'#FID': 'FID'}, inplace=True)
            truth_df = truth_df_raw[['IID', 'PRS_AVG']].rename(columns={'PRS_AVG': 'SCORE_TRUTH'})

            # --- Load gnomon_df ---
            gnomon_df_raw = pd.read_csv(OUTPUT_PREFIX.with_suffix(".sscore"), sep='\t')
            _debug_print_df(gnomon_df_raw, "gnomon_df (raw)")
            if '#IID' in gnomon_df_raw.columns:
                gnomon_df_raw.rename(columns={'#IID': 'IID'}, inplace=True)
            gnomon_df = gnomon_df_raw[['IID', 'simulated_score_AVG']].rename(columns={'simulated_score_AVG': 'SCORE_GNOMON'})

            # --- Load plink2_df ---
            plink2_df_raw = pd.read_csv(WORKDIR / "plink2_results.sscore", sep='\s+')
            _debug_print_df(plink2_df_raw, "plink2_df (raw)")
            # Clean up the '#IID' or '#FID' column name
            if '#IID' in plink2_df_raw.columns: plink2_df_raw.rename(columns={'#IID': 'IID'}, inplace=True)
            elif '#FID' in plink2_df_raw.columns: plink2_df_raw.rename(columns={'#FID': 'FID'}, inplace=True)
            # PLINK's AVG is per-allele, we multiply by 2 to compare with per-variant scores
            plink2_df = plink2_df_raw.assign(SCORE_PLINK2=plink2_df_raw['SCORE1_AVG'] * 2.0)[['IID', 'SCORE_PLINK2']]
            
        except (FileNotFoundError, KeyError) as e:
            print(f"  > ❌ ERROR: Failed to load or parse a result file. Error: {e}. Aborting comparison.")
            sys.exit(1)

        merged_df = pd.merge(truth_df, gnomon_df, on='IID').merge(plink2_df, on='IID')
        score_cols = ['SCORE_TRUTH', 'SCORE_GNOMON', 'SCORE_PLINK2']
        merged_df[score_cols] = merged_df[score_cols].astype(float)

        print("\n--- Sample of Computed Scores (Large-Scale) ---")
        print(merged_df.head(10).to_markdown(index=False, floatfmt=".8f"))

        print("\n--- Score Correlation Matrix (Large-Scale) ---")
        print(merged_df[score_cols].corr().to_markdown(floatfmt=".8f"))

        print("\n--- Mean Absolute Difference (MAD) vs. Ground Truth (Large-Scale) ---")
        mad_g_vs_t = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_GNOMON']).abs().mean()
        mad_p_vs_t = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_PLINK2']).abs().mean()
        print(f"Gnomon vs. Truth: {mad_g_vs_t:.10f}")
        print(f"PLINK2 vs. Truth: {mad_p_vs_t:.10f}")
        
        if mad_g_vs_t < 1e-9 and mad_p_vs_t < 1e-9:
            print("\n✅ Large-Scale Validation Successful: All tools produced scores identical to the ground truth.")
        else:
            print("\n⚠️ Large-Scale Validation Warning: Significant differences detected.")
            sys.exit(1)

    # --- Main execution flow for this function ---
    setup_tools()
    
    run_simple_dosage_test(WORKDIR, GNOMON_BINARY_PATH, PLINK2_BINARY_PATH, run_command)
    
    print("\n" + "="*80)
    print("= Running Large-Scale Simulation and Validation")
    print("="*80)

    # Run Gnomon on large-scale data
    run_command(
        [GNOMON_BINARY_PATH, "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name, OUTPUT_PREFIX.name],
        "Large-Scale Gnomon", cwd=WORKDIR
    )

    # Run PLINK2 on large-scale data
    run_command(
        [f"./{PLINK2_BINARY_PATH.name}", "--bfile", OUTPUT_PREFIX.name, "--score", OUTPUT_PREFIX.with_suffix(".gnomon.score").name, "1", "2", "4", "header", "no-mean-imputation", "--out", "plink2_results"],
        "Large-Scale PLINK2", cwd=WORKDIR
    )

    analyze_and_compare_large_scale_results()

def cleanup():
    """Removes the generated workspace directory."""
    print("\n" + "-"*80)
    print(f"Cleaning up workspace directory: {WORKDIR}")
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
        print("...Workspace cleaned up.")

def main():
    """Main function to run the entire simulation and file generation pipeline."""
    np.random.seed(42)
    # Ensure the isolated workspace exists and is empty
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir()

    print("--- Starting Full Simulation and File Writing Pipeline ---")
    print(f"All generated files will be placed in: {WORKDIR}")

    variants_df = generate_variants_and_weights()
    genotypes_pristine = generate_genotypes(variants_df)
    annotation_positions = simulate_annotations()
    with open(OUTPUT_PREFIX.with_suffix(".annotations.txt"), 'w') as f:
        for pos in annotation_positions: f.write(f"{pos}\n")

    genotypes_with_missing = introduce_missingness(genotypes_pristine)
    prs_results_df = calculate_ground_truth_prs(genotypes_with_missing, variants_df)
    
    write_output_files(prs_results_df, variants_df, genotypes_with_missing, OUTPUT_PREFIX)
    print("\n--- Simulation and File Writing Finished Successfully ---")

    try:
        run_and_validate_tools()
    finally:
        # Pass an argument like 'keep' to skip cleanup for debugging
        if 'keep' not in sys.argv:
            cleanup()
        else:
            print("\nSkipping cleanup because 'keep' argument was provided.")
    
    print("\n--- Full Simulation and Validation Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
