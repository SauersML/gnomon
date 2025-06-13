import numpy as np
import pandas as pd
import sys
import os

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

    # Generate a random allele frequency 'p' for the reference allele for each variant
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
    header = ['#FID', 'IID', 'SID', 'PHENO1', 'ALLELE_CT', 'DENOM', 'NAMED_ALLELE_DOSAGE_SUM', 'PRS_AVG', 'PRS_SUM']
    # Reorder columns to match the standard PLINK output
    prs_cols_ordered = ['FID', 'IID', 'SID', 'PHENO1', 'ALLELE_CT', 'DENOM', 'NAMED_ALLELE_DOSAGE_SUM', 'PRS_AVG', 'PRS_SUM']
    prs_results[prs_cols_ordered].to_csv(sscore_filename, sep='\t', header=header, index=False, na_rep='NA')
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


def main():
    """Main function to run the entire simulation and file generation pipeline."""
    print("--- Starting Full Simulation and File Writing Pipeline ---")
    
    # Step 1: Simulate variants, weights, and effect alleles
    variants_df = generate_variants_and_weights()

    # Step 2: Simulate genotypes for all individuals based on HWE
    genotypes_pristine = generate_genotypes(variants_df)

    # Step 3: Simulate annotations (as a standalone demonstration)
    simulate_annotations()

    # Step 4: Introduce missingness into the genotype matrix
    genotypes_with_missing = introduce_missingness(genotypes_pristine)

    # Step 5: Calculate ground truth Polygenic Scores
    prs_results_df = calculate_ground_truth_prs(genotypes_with_missing, variants_df)
    
    # Step 6: Write all specified output files to disk
    write_output_files(prs_results_df, variants_df, genotypes_with_missing, OUTPUT_PREFIX)

    print("\n--- Pipeline Finished Successfully ---")
    print(f"Generated filesets: '{OUTPUT_PREFIX}.bed/.bim/.fam', '{OUTPUT_PREFIX}.sscore', '{OUTPUT_PREFIX}.score'")

if __name__ == "__main__":
    main()
