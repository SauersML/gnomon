""""
Example:
python3 pylink.py \
    --bfile ./ci_workdir/chr22_subset50_compat \
    --score ./ci_workdir/PGS003725_unified_format.tsv \
    --out ./ci_workdir/plink2_PGS003725 \
    1 2 4
"""

import sys
import math
import argparse
import pandas as pd
import numpy as np

def read_score_file(filepath, id_col, allele_col, score_col):
    """
    Reads the scoring file into a dictionary for quick lookup.
    Keys are (variant_id, effect_allele), values are the scores.
    """
    print(f"Reading score file: {filepath}...")
    try:
        # Adjust column numbers to be 0-indexed for pandas iloc
        id_col_idx, allele_col_idx, score_col_idx = id_col - 1, allele_col - 1, score_col - 1

        df = pd.read_csv(filepath, sep='\s+', header=0, low_memory=False,
                         comment='#')

        score_data = {}
        for _, row in df.iterrows():
            variant_id = str(row.iloc[id_col_idx])
            effect_allele = str(row.iloc[allele_col_idx])
            # Ensure the score is treated as a float
            score = float(row.iloc[score_col_idx])
            score_data[(variant_id, effect_allele)] = score

        print(f"  > Loaded {len(score_data)} variant scores.")
        return score_data
    except Exception as e:
        print(f"Error reading score file: {e}", file=sys.stderr)
        sys.exit(1)


def read_bim_file(filepath):
    """
    Reads a .bim file into a pandas DataFrame.
    """
    print(f"Reading variant info file (.bim): {filepath}...")
    try:
        bim_df = pd.read_csv(
            filepath, sep='\s+', header=None,
            names=['CHR', 'ID', 'CM', 'POS', 'A1', 'A2'],
            dtype={'CHR': str, 'ID': str, 'A1': str, 'A2': str}
        )
        print(f"  > Loaded info for {len(bim_df)} variants.")
        return bim_df
    except FileNotFoundError:
        print(f"Error: .bim file not found at {filepath}", file=sys.stderr)
        sys.exit(1)


def read_fam_file(filepath):
    """
    Reads a .fam file into a pandas DataFrame.
    """
    print(f"Reading sample info file (.fam): {filepath}...")
    try:
        fam_df = pd.read_csv(
            filepath, sep='\s+', header=None,
            names=['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHEN'],
            dtype={'FID': str, 'IID': str}
        )
        print(f"  > Loaded info for {len(fam_df)} samples.")
        return fam_df
    except FileNotFoundError:
        print(f"Error: .fam file not found at {filepath}", file=sys.stderr)
        sys.exit(1)

def find_ambiguous_samples(bed_filepath, num_samples, bim_df):
    """
    Pass 1: Identify samples with non-missing genotypes for multiple variants
    at the same genomic position.
    """
    print("\n--- Pass 1: Identifying ambiguous samples ---")
    # {sample_idx: set of (CHR, POS) tuples}
    position_tracker = [set() for _ in range(num_samples)]
    # Set of sample indices that must have a NaN score
    nan_sample_indices = set()
    bytes_per_variant = math.ceil(num_samples / 4)

    with open(bed_filepath, "rb") as f:
        # Skip magic number
        f.read(3)
        for var_idx, variant in bim_df.iterrows():
            pos_tuple = (variant['CHR'], variant['POS'])
            variant_bytes = f.read(bytes_per_variant)
            if not variant_bytes:
                break

            for sample_idx in range(num_samples):
                byte_idx = sample_idx // 4
                bit_shift = (sample_idx % 4) * 2
                genotype_code = (variant_bytes[byte_idx] >> bit_shift) & 0b11

                # 0b01 is the code for a missing genotype
                if genotype_code != 0b01:
                    if pos_tuple in position_tracker[sample_idx]:
                        nan_sample_indices.add(sample_idx)
                    else:
                        position_tracker[sample_idx].add(pos_tuple)

    if nan_sample_indices:
        print(f"  > Identified {len(nan_sample_indices)} sample(s) with ambiguous genotypes at single positions.")
    else:
        print("  > No ambiguous samples found.")
    return nan_sample_indices


def calculate_scores(bed_filepath, num_samples, bim_df, score_data):
    """
    Pass 2: Calculate scores for all samples.
    """
    print("\n--- Pass 2: Calculating scores ---")
    weighted_score_sum = np.zeros(num_samples)
    denominator_ct = np.zeros(num_samples) # This is NMISS_ALLELE_CT
    variants_processed_count = 0
    bytes_per_variant = math.ceil(num_samples / 4)

    # Genotype code mapping to dosage of Allele 2 (A2)
    # 00 (0b00): Homozygous A1 -> 0 dosage of A2
    # 10 (0b10): Heterozygous  -> 1 dosage of A2
    # 11 (0b11): Homozygous A2 -> 2 dosage of A2
    # 01 (0b01): Missing       -> -1 (sentinel)
    geno_to_a2_dosage = {0b00: 0, 0b10: 1, 0b11: 2, 0b01: -1}

    with open(bed_filepath, "rb") as f:
        # Skip magic number and variant-major mode byte
        magic = f.read(3)
        if magic != b'\x6c\x1b\x01':
            raise ValueError("Invalid .bed file: incorrect magic number.")

        for var_idx, variant in bim_df.iterrows():
            variant_id, a1, a2 = variant['ID'], variant['A1'], variant['A2']
            score_info = None

            # Check if the score is defined for A2 (alternate) or A1 (reference)
            if (variant_id, a2) in score_data:
                score_info = {'weight': score_data[(variant_id, a2)], 'effect_is_a2': True}
            elif (variant_id, a1) in score_data:
                score_info = {'weight': score_data[(variant_id, a1)], 'effect_is_a2': False}

            variant_bytes = f.read(bytes_per_variant)
            if not variant_bytes:
                break # End of file

            if score_info:
                variants_processed_count += 1
                for sample_idx in range(num_samples):
                    byte_idx = sample_idx // 4
                    bit_shift = (sample_idx % 4) * 2
                    genotype_code = (variant_bytes[byte_idx] >> bit_shift) & 0b11

                    dosage_a2 = geno_to_a2_dosage[genotype_code]
                    if dosage_a2 != -1: # If not missing
                        if score_info['effect_is_a2']:
                            effect_dosage = dosage_a2
                        else: # Effect allele is A1
                            effect_dosage = 2 - dosage_a2

                        weighted_score_sum[sample_idx] += effect_dosage * score_info['weight']
                        denominator_ct[sample_idx] += 2

    print(f"  > Processed {variants_processed_count} variants found in score file.")
    return weighted_score_sum, denominator_ct


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="A Python script to replicate 'plink --score' with "
                    "'no-mean-imputation', matching PLINK2's NaN behavior "
                    "for ambiguous positions.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--bfile', required=True,
        help='Prefix for the input PLINK .bed, .bim, and .fam fileset.'
    )
    parser.add_argument(
        '--score', required=True,
        help='Path to the scoring file. Must have a header.'
    )
    parser.add_argument(
        '--out', required=True,
        help='Prefix for the output file (e.g., my_scores).'
    )
    parser.add_argument(
        'columns', nargs=3, type=int,
        help="Three 1-based column numbers for: Variant_ID Effect_Allele Score"
    )

    args = parser.parse_args()

    # --- Load Metadata ---
    fam_df = read_fam_file(f"{args.bfile}.fam")
    bim_df = read_bim_file(f"{args.bfile}.bim")
    score_data = read_score_file(args.score, args.columns[0], args.columns[1], args.columns[2])
    
    num_samples = len(fam_df)
    bed_filepath = f"{args.bfile}.bed"

    # --- Pass 1 ---
    nan_indices = find_ambiguous_samples(bed_filepath, num_samples, bim_df)

    # --- Pass 2 ---
    weighted_sum, denom_ct = calculate_scores(bed_filepath, num_samples, bim_df, score_data)

    # --- Finalize and Write Output ---
    print("\n--- Finalizing results ---")
    
    # Calculate average score, handling division by zero
    final_scores_avg = np.divide(
        weighted_sum, denom_ct,
        out=np.zeros_like(weighted_sum, dtype=float),
        where=(denom_ct != 0)
    )

    # Apply NaN mask for ambiguous samples
    if nan_indices:
        final_scores_avg[list(nan_indices)] = np.nan
        print(f"  > Applied NaN to {len(nan_indices)} sample(s).")
    
    # Construct the output DataFrame
    output_df = pd.DataFrame({
        '#FID': fam_df['FID'],
        'IID': fam_df['IID'],
        'NMISS_ALLELE_CT': denom_ct.astype(int),
        'SCORE1_AVG': final_scores_avg
    })

    output_filepath = f"{args.out}.sscore"
    try:
        output_df.to_csv(
            output_filepath, sep='\t', index=False,
            float_format='%.6g', na_rep='nan'
        )
        print(f"Success! Results written to {output_filepath}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
