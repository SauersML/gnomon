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
OUTPUT_PREFIX = "simulated_data"

# Annotation simulation parameters THIS MUST BE USED. FIX.
ANNOT_MEAN_DIST = 30
ANNOT_STD_DEV = 10
ANNOT_MIN_DIST = 2
ANNOT_MAX_DIST = 500

def generate_variants_and_weights():
    """
    FIRST: Derives normally and randomly distributed weights over 500K positions.
    - Positions are random on chr22 (length 40M).
    - Alleles (ref, alt) are A,C,G,T. No multiallelics.
    - Effect alleles are chosen (80% alt, 20% ref).
    - Weights are from a normal distribution.
    """
    print(f"Step 1: Simulating {N_VARIANTS} variants on chr{CHR}...")

    # 1. Generate unique, sorted positions for the variants
    positions = np.random.choice(np.arange(1, CHR_LENGTH + 1), N_VARIANTS, replace=False)
    positions.sort()

    # 2. Generate ref and alt alleles (A, C, G, T) in a vectorized way
    alleles = np.array(['A', 'C', 'G', 'T'])
    ref_idx = np.random.randint(0, 4, N_VARIANTS)
    # Ensure alt is different from ref by adding a random offset (1, 2, or 3)
    alt_offset = np.random.randint(1, 4, N_VARIANTS)
    alt_idx = (ref_idx + alt_offset) % 4
    
    ref_alleles = alleles[ref_idx]
    alt_alleles = alleles[alt_idx]

    # 3. Choose the effect allele (80% alt, 20% ref)
    is_alt_effect = np.random.rand(N_VARIANTS) < ALT_EFFECT_PROB
    effect_alleles = np.where(is_alt_effect, alt_alleles, ref_alleles)

    # 4. Generate effect weights from a normal distribution
    # Effect weights are drawn from a normal distribution.
    # loc=0, scale=0.05 chosen as a reasonable default for effect size magnitudes.
    effect_weights = np.random.normal(loc=0, scale=0.05, size=N_VARIANTS)

    # 5. Assemble into a DataFrame for easy management
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
    - For each variant, a random allele frequency is generated.
    - Genotypes (0 for hom_ref, 1 for het, 2 for hom_alt) are drawn from HWE probabilities.
    Returns a numpy array of shape (n_variants, n_individuals).
    """
    print(f"Step 2: Simulating genotypes for {N_INDIVIDUALS} individuals...")
    
    n_vars = len(variants_df)

    # For each variant, simulate a reference allele frequency 'p'.
    # Drawing 'p' from U(0.01, 0.99) ensures variability in allele frequencies across loci,
    # preventing fixed alleles and providing a range for HWE calculations.
    p = np.random.uniform(0.01, 0.99, size=n_vars)
    q = 1 - p

    # HWE probabilities for genotypes [hom_ref, het, hom_alt]
    hwe_probs = np.vstack([p**2, 2*p*q, q**2]).T

    # Vectorized genotype simulation for efficiency
    # Draw a random number for each genotype to be simulated
    rand_draws = np.random.rand(n_vars, N_INDIVIDUALS)
    
    # Get cumulative probabilities for drawing
    cum_probs = hwe_probs.cumsum(axis=1)

    # Determine genotype by comparing random draw to cumulative probability thresholds
    # (rand_draws > p^2) is 1 for het/hom_alt, 0 for hom_ref
    # (rand_draws > p^2 + 2pq) is 1 for hom_alt, 0 for het/hom_ref
    # Summing them gives the desired 0, 1, or 2 code.
    genotypes = (rand_draws > cum_probs[:, 0, np.newaxis]) + \
                (rand_draws > cum_probs[:, 1, np.newaxis])

    print("...Genotype simulation complete.")
    return genotypes.astype(int)

def simulate_annotations():
    """
    THEN: Simulates annotation positions along the chromosome.
    This is a standalone step as per the prompt and is not used in other calculations.
    """
    print("Step 3: Simulating annotation positions (for demonstration)...")
    annotations = []
    current_pos = 1
    while current_pos <= CHR_LENGTH:
        annotations.append(current_pos)
        step = np.random.normal(ANNOT_MEAN_DIST, ANNOT_STD_DEV)
        # Clip the step size to be within the specified min/max
        step = np.clip(step, ANNOT_MIN_DIST, ANNOT_MAX_DIST)
        current_pos += int(round(step))
    
    print(f"...Generated {len(annotations)} annotations.")
    print(f"First 10 annotation positions: {annotations[:10]}")
    # This data is not used further, as per the prompt's structure.
    return annotations

def introduce_missingness(genotypes):
    """
    Inserts 2% random missingness into the genotype matrix.
    Missingness is represented by the integer -1 for computational convenience.
    """
    print(f"Step 4: Introducing {MISSING_RATE*100:.1f}% missingness...")
    
    n_variants, n_individuals = genotypes.shape
    
    # Create a boolean mask for values that will become missing
    missing_mask = np.random.rand(n_variants, n_individuals) < MISSING_RATE
    
    # Apply the mask. Convert to float to hold a marker, then back to int.
    genotypes_with_missing = genotypes.astype(float)
    genotypes_with_missing[missing_mask] = -1 # Use -1 as the missing marker
    # Using -1 as a numerical marker for missingness is specific to this script's
    # genotype processing and BED file conversion logic (via code_map).
    # In other contexts or with different tools, pandas.NA or np.nan might be preferred.
    
    print("...Missingness introduced.")
    return genotypes_with_missing.astype(int)

def calculate_ground_truth_prs(genotypes, variants_df):
    """
    THEN: Calculates the 'ground truth' polygenic score for each individual.
    Method: "sum, then divide by number of variants used in the calculation".
    """
    print("Step 5: Calculating ground truth polygenic scores...")

    n_variants, n_individuals = genotypes.shape
    results = []

    # Pre-calculate which variants have 'alt' as the effect allele
    is_alt_effect = (variants_df['effect_allele'] == variants_df['alt']).values

    for i in range(n_individuals):
        ind_genotypes = genotypes[:, i]
        
        # Create a mask to filter out missing genotypes (-1)
        valid_mask = ind_genotypes != -1
        valid_genotypes = ind_genotypes[valid_mask]
        
        # Handle the edge case of an individual with no valid genotypes
        if len(valid_genotypes) == 0:
            results.append({
                'FID': f"sample_{i+1}", 'IID': f"sample_{i+1}", 'SID': f"sample_{i+1}", 'PHENO1': 'NA',
                'ALLELE_CT': 0, 'DENOM': 0, 'NAMED_ALLELE_DOSAGE_SUM': 0,
                'PRS_SUM': 0, 'PRS_AVG': 0
            })
            continue

        # Get weights for the non-missing variants
        valid_weights = variants_df['effect_weight'][valid_mask].values
        
        # Determine the dosage of the effect allele for each valid variant
        dosages = np.zeros_like(valid_genotypes, dtype=float)
        valid_is_alt_effect = is_alt_effect[valid_mask]
        
        # Where effect allele is ALT, dosage is the genotype (0, 1, or 2)
        dosages[valid_is_alt_effect] = valid_genotypes[valid_is_alt_effect]
        # Where effect allele is REF, dosage is 2 minus the genotype
        dosages[~valid_is_alt_effect] = 2 - valid_genotypes[~valid_is_alt_effect]

        # Calculate score components as specified
        variant_count = len(valid_genotypes)
        allele_count = variant_count * 2
        named_allele_dosage_sum = np.sum(dosages)
        score_sum = np.sum(dosages * valid_weights)
        score_avg = score_sum / variant_count if variant_count > 0 else 0
        
        results.append({
            'FID': f"sample_{i+1}", 'IID': f"sample_{i+1}", 'SID': f"sample_{i+1}", 'PHENO1': 'NA',
            'ALLELE_CT': allele_count, 'DENOM': variant_count, 'NAMED_ALLELE_DOSAGE_SUM': named_allele_dosage_sum,
            'PRS_SUM': score_sum, 'PRS_AVG': score_avg
        })
        
    print("...PRS calculation complete.")
    return pd.DataFrame(results)

def write_output_files(prs_results, variants_df, genotypes_with_missing, prefix):
    """
    Writes all specified output files to disk:
    - The .sscore results file.
    - The scorefile with variant weights.
    - The PLINK .bed, .bim, and .fam files.
    """
    print("Step 6: Writing all output files...")

    # --- a. Write PRS Results File (.sscore format) ---
    sscore_filename = f"{prefix}.sscore"
    # Define column order for the .sscore file
    prs_cols_ordered = ['FID', 'IID', 'SID', 'PHENO1', 'ALLELE_CT', 'DENOM', 'NAMED_ALLELE_DOSAGE_SUM', 'PRS_AVG', 'PRS_SUM']

    with open(sscore_filename, 'w') as f:
        # Manually write the header line starting with #
        f.write('#' + '\t'.join(prs_cols_ordered) + '\n')
        # Write the DataFrame content without pandas header
        prs_results[prs_cols_ordered].to_csv(f, sep='\t', header=False, index=False, na_rep='NA')
    print(f"...PRS results written to {sscore_filename}")
    
    # --- b. Write Scorefile ---
    scorefile_filename = f"{prefix}.score"
    score_df = variants_df[['chr', 'pos', 'effect_allele', 'effect_weight']].copy()
    score_df.rename(columns={'chr': 'chr_name', 'pos': 'chr_position'}, inplace=True)
    score_df.to_csv(scorefile_filename, sep='\t', index=False)
    print(f"...Scorefile written to {scorefile_filename}")

    # --- c. Write PLINK BED/BIM/FAM files ---
    # --- .fam file ---
    fam_filename = f"{prefix}.fam"
    with open(fam_filename, 'w') as f:
        for i in range(N_INDIVIDUALS):
            f.write(f"sample_{i+1} sample_{i+1} 0 0 0 -9\n") # FID IID PAT MAT SEX PHENO
    
    # --- .bim file ---
    bim_filename = f"{prefix}.bim"
    bim_df = pd.DataFrame({
        'chr': variants_df['chr'],
        'id': variants_df['pos'].apply(lambda x: f"22:{x}"), # Standard variant ID format
        'cm': 0,
        'pos': variants_df['pos'],
        'a1': variants_df['ref'], # Allele 1 (corresponds to genotype code 00)
        'a2': variants_df['alt']  # Allele 2 (corresponds to genotype code 11)
    })
    bim_df.to_csv(bim_filename, sep='\t', header=False, index=False)
    
    # --- .bed file (binary) ---
    bed_filename = f"{prefix}.bed"
    # Map our internal genotype codes to PLINK's 2-bit codes
    # Our codes: 0=hom_ref(A1), 1=het, 2=hom_alt(A2), -1=missing
    # PLINK codes: 00=hom_A1, 01=missing, 10=het, 11=hom_A2
    code_map = {0: 0b00, -1: 0b01, 1: 0b10, 2: 0b11}

    with open(bed_filename, 'wb') as f:
        # Write PLINK magic number
        f.write(bytes([0x6c, 0x1b, 0x01]))

        # Write genotypes in variant-major order
        n_variants, n_individuals = genotypes_with_missing.shape
        for i in range(n_variants):
            # Process individuals in chunks of 4 to create bytes
            for j in range(0, n_individuals, 4):
                byte = 0
                chunk_genotypes = genotypes_with_missing[i, j:j+4]
                
                # Pack the 2-bit codes for 4 individuals into a single byte
                for k, geno in enumerate(chunk_genotypes):
                    plink_code = code_map[geno]
                    byte |= (plink_code << (k * 2))
                
                f.write(byte.to_bytes(1, 'little'))
    
    print(f"...PLINK files written: {fam_filename}, {bim_filename}, {bed_filename}")

def run_and_validate_tools():
    """
    ownloads PLINK2, runs it and gnomon on the simulated data,
    and compares their outputs against each other and the ground truth.
    """
    print("\n" + "="*80)
    print("= Running Tool Validation and Comparison Pipeline")
    print("="*80)

    # --- Configuration ---
    PLINK2_URL = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_avx2_20250609.zip"
    PLINK2_BINARY_PATH = Path("./plink2")
    GNOMON_BINARY_PATH = "gnomon" # Assumes gnomon is in the system's PATH
    COMPAT_PREFIX = "simulated_data_plink_compat"
    
    # --- Helper Functions (nested to keep them self-contained) ---

    def _print_header(title: str, char: str = "-"):
        width = 70
        print(f"\n{char*4} {title} {'-'*(width - len(title) - 5)}")

    def run_command(cmd: list, step_name: str):
        """Executes a command, printing its output and checking for errors."""
        _print_header(f"Executing: {step_name}")
        print(f"  > Command: {' '.join(map(str, cmd))}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,  # This will raise an exception on non-zero exit codes
                encoding='utf-8',
                errors='replace'
            )
            print(f"  > Success. Output logs can be found in files like '{cmd[-1]}.log'")
            # print(result.stdout) # Uncomment for verbose debugging
            return True
        except FileNotFoundError:
            print(f"  > ❌ ERROR: Command '{cmd[0]}' not found.")
            print(f"  > Please ensure '{cmd[0]}' is installed and in your system's PATH.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"  > ❌ ERROR: {step_name} failed with exit code {e.returncode}.")
            print("--- STDERR ---")
            print(e.stderr)
            print("--- STDOUT ---")
            print(e.stdout)
            sys.exit(1)

    def setup_tools():
        _print_header("Step A: Setting up tools")
        
        # 1. Check for gnomon
        if not shutil.which(GNOMON_BINARY_PATH):
            print(f"  > ❌ ERROR: '{GNOMON_BINARY_PATH}' not found in PATH.")
            print("  > Please install gnomon (e.g., via 'cargo install gnomon') and ensure it's in your PATH.")
            sys.exit(1)
        print("  > Found 'gnomon' executable.")

        # 2. Download and extract PLINK2 if it doesn't exist
        if PLINK2_BINARY_PATH.exists():
            print("  > 'plink2' executable already exists. Skipping download.")
            return

        print(f"  > Downloading PLINK2 from {PLINK2_URL}...")
        zip_path = Path("plink.zip")
        try:
            with requests.get(PLINK2_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(".")
            
            zip_path.unlink()
            PLINK2_BINARY_PATH.chmod(0o755) # Make it executable
            print("  > PLINK2 downloaded and extracted successfully.")

        except Exception as e:
            print(f"  > ❌ FAILED to download or extract PLINK2: {e}")
            sys.exit(1)

    def prepare_plink_compatible_files():
        """Creates a new fileset with variant IDs synchronized for PLINK."""
        _print_header("Step B: Preparing PLINK-compatible files")

        # 1. Read the original .bim file to get allele information
        bim_df = pd.read_csv(f"{OUTPUT_PREFIX}.bim", sep='\t', header=None, names=['chr', 'id', 'cm', 'pos', 'a1', 'a2'])
        
        # 2. Create canonical IDs (chr_pos_sortedA1_sortedA2) for robust matching
        alleles = bim_df[['a1', 'a2']].apply(sorted, axis=1, result_type='expand')
        bim_df['canonical_id'] = bim_df['chr'].astype(str) + '_' + bim_df['pos'].astype(str) + '_' + alleles[0] + '_' + alleles[1]
        
        # 3. Write new .bim file
        new_bim_df = bim_df[['chr', 'canonical_id', 'cm', 'pos', 'a1', 'a2']]
        new_bim_df.to_csv(f"{COMPAT_PREFIX}.bim", sep='\t', header=False, index=False)
        print(f"  > Created synchronized BIM file: {COMPAT_PREFIX}.bim")

        # 4. Copy .bed and .fam files
        shutil.copy(f"{OUTPUT_PREFIX}.bed", f"{COMPAT_PREFIX}.bed")
        shutil.copy(f"{OUTPUT_PREFIX}.fam", f"{COMPAT_PREFIX}.fam")

        # 5. Create a new score file with the same canonical IDs
        score_df = pd.read_csv(f"{OUTPUT_PREFIX}.score", sep='\t')
        # We need the other allele to create the same canonical ID
        # Join with original bim 'ref' and 'alt' to get both alleles for each variant
        bim_alleles = bim_df[['pos', 'a1', 'a2']].set_index('pos')
        score_df = score_df.join(bim_alleles, on='chr_position')
        
        alleles = score_df[['a1', 'a2']].apply(sorted, axis=1, result_type='expand')
        score_df['canonical_id'] = score_df['chr_name'].astype(str) + '_' + score_df['chr_position'].astype(str) + '_' + alleles[0] + '_' + alleles[1]

        # 6. Write new score file in the format PLINK expects (ID, EFFECT_ALLELE, WEIGHT)
        plink_score_df = score_df[['canonical_id', 'effect_allele', 'effect_weight']]
        plink_score_df.to_csv(f"{COMPAT_PREFIX}.score", sep='\t', index=False)
        print(f"  > Created synchronized score file: {COMPAT_PREFIX}.score")


    def analyze_and_compare_results():
        """Loads all results and prints a detailed comparison."""
        _print_header("Step D: Analyzing and Comparing Results")

        try:
            # 1. Load the ground truth calculated by this script
            truth_df = pd.read_csv(f"{OUTPUT_PREFIX}.sscore", sep='\t', comment='#')
            truth_df = truth_df[['IID', 'PRS_AVG']].rename(columns={'PRS_AVG': 'SCORE_TRUTH'})

            # 2. Load Gnomon's results
            gnomon_df = pd.read_csv("gnomon_results.sscore", sep='\t', comment='#')
            gnomon_df = gnomon_df[['IID', 'PRS_AVG']].rename(columns={'PRS_AVG': 'SCORE_GNOMON'})

            # 3. Load PLINK2's results (score column is SCORE1_AVG)
            plink2_df = pd.read_csv("plink2_results.sscore", sep='\t', comment='#')
            plink2_df = plink2_df[['IID', 'SCORE1_AVG']].rename(columns={'SCORE1_AVG': 'SCORE_PLINK2'})
            
        except FileNotFoundError as e:
            print(f"  > ❌ ERROR: Could not find a result file: {e.filename}")
            print("  > Aborting comparison.")
            return

        # 4. Merge all results into a single DataFrame
        merged_df = pd.merge(truth_df, gnomon_df, on='IID')
        merged_df = pd.merge(merged_df, plink2_df, on='IID')
        
        score_cols = ['SCORE_TRUTH', 'SCORE_GNOMON', 'SCORE_PLINK2']
        merged_df[score_cols] = merged_df[score_cols].astype(float)

        print("\n--- Sample of Computed Scores (first 10 individuals) ---")
        print(merged_df.head(10).to_markdown(index=False, floatfmt=".8f"))

        print("\n--- Score Correlation Matrix ---")
        correlation_matrix = merged_df[score_cols].corr()
        print(correlation_matrix.to_markdown(floatfmt=".8f"))

        print("\n--- Mean Absolute Difference (MAD) ---")
        mad_g_vs_t = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_GNOMON']).abs().mean()
        mad_p_vs_t = (merged_df['SCORE_TRUTH'] - merged_df['SCORE_PLINK2']).abs().mean()
        mad_g_vs_p = (merged_df['SCORE_GNOMON'] - merged_df['SCORE_PLINK2']).abs().mean()
        print(f"Gnomon vs. Truth: {mad_g_vs_t:.10f}")
        print(f"PLINK2 vs. Truth: {mad_p_vs_t:.10f}")
        print(f"Gnomon vs. PLINK2: {mad_g_vs_p:.10f}")

        if mad_g_vs_t < 1e-9 and mad_p_vs_t < 1e-9:
            print("\n✅ Validation Successful: All tools produced scores identical to the ground truth.")
        else:
            print("\n⚠️ Validation Warning: Significant differences detected between tool outputs and ground truth.")

    # --- Main execution flow for this function ---
    setup_tools()
    prepare_plink_compatible_files()

    # --- Step C: Run Scoring Tools ---
    # Gnomon uses original files (it's more flexible)
    run_command(
        [GNOMON_BINARY_PATH, "--score", f"{OUTPUT_PREFIX}.score", OUTPUT_PREFIX, "--out", "gnomon_results"],
        "Gnomon"
    )

    # PLINK2 uses the new, compatible files
    run_command(
        [
            str(PLINK2_BINARY_PATH),
            "--bfile", COMPAT_PREFIX,
            "--score", f"{COMPAT_PREFIX}.score", "1", "2", "3", "header", "no-mean-imputation",
            "--out", "plink2_results"
        ],
        "PLINK2"
    )

    analyze_and_compare_results()

def main():
    """Main function to run the entire simulation and file generation pipeline."""
    np.random.seed(42) # Set seed for reproducibility
    print("--- Starting Full Simulation and File Writing Pipeline ---")
    
    # Step 1: Simulate variants, weights, and effect alleles
    variants_df = generate_variants_and_weights()

    # Step 2: Simulate genotypes for all individuals based on HWE
    genotypes_pristine = generate_genotypes(variants_df)

    # Step 3: Simulate annotations (as a standalone demonstration)
    annotation_positions = simulate_annotations()

    # Step 3b: Writing annotation positions to file
    annotations_filename = f"{OUTPUT_PREFIX}.annotations.txt"
    print(f"Step 3b: Writing annotation positions to {annotations_filename}...")
    with open(annotations_filename, 'w') as f:
        for pos in annotation_positions:
            f.write(f"{pos}\n")
    print(f"...Annotation positions written to {annotations_filename}.")

    # Step 4: Introduce missingness into the genotype matrix
    genotypes_with_missing = introduce_missingness(genotypes_pristine)

    # Step 5: Calculate ground truth Polygenic Scores
    prs_results_df = calculate_ground_truth_prs(genotypes_with_missing, variants_df)

    print("\n--- Ground Truth Polygenic Scores (per individual) ---")
    print(prs_results_df.to_string())
    
    # Step 6: Write all specified output files to disk
    write_output_files(prs_results_df, variants_df, genotypes_with_missing, OUTPUT_PREFIX)

    print("\n--- Simulation and File Writing Finished Successfully ---")
    print(f"Generated filesets: '{OUTPUT_PREFIX}.bed/.bim/.fam', '{OUTPUT_PREFIX}.sscore', '{OUTPUT_PREFIX}.score', '{OUTPUT_PREFIX}.annotations.txt'")

    run_and_validate_tools()

    print("\n--- Pipeline Finished Successfully ---")
    print(f"Generated filesets: '{OUTPUT_PREFIX}.bed/.bim/.fam', '{OUTPUT_PREFIX}.sscore', '{OUTPUT_PREFIX}.score', '{OUTPUT_PREFIX}.annotations.txt'")

if __name__ == "__main__":
    main()
