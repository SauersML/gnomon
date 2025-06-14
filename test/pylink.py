import sys
import math
import argparse
import polars as pl
import numpy as np
import pandas as pd # Used exclusively for its unique CSV writing capabilities.


def read_fam_file(filepath: str) -> pl.DataFrame:
    """
    Reads a .fam file using Polars, automatically handling tab or space delimiters.
    """
    print(f"Reading sample info file (.fam): {filepath}...")
    try:
        # PLINK .fam files are whitespace-delimited, often with variable spaces.
        # Reading as a single column and splitting is a robust way to handle this.
        df = pl.read_csv(filepath, has_header=False, separator='\t', new_columns=["data"])
        df = df.with_columns(
            pl.col("data").str.split_exact(" ", 5).alias("split_data")
        ).unnest("split_data")
        df = df.select(
            pl.col("field_0").alias("FID"),
            pl.col("field_1").alias("IID"),
            pl.col("field_2").alias("PAT"),
            pl.col("field_3").alias("MAT"),
            pl.col("field_4").alias("SEX"),
            pl.col("field_5").alias("PHEN"),
        )
        print(f"  > Loaded info for {df.height} samples.")
        return df
    except Exception as e:
        print(f"Error reading .fam file: {e}", file=sys.stderr)
        sys.exit(1)


def read_bim_file(filepath: str) -> pl.DataFrame:
    """
    Reads a .bim file using Polars, automatically handling tab or space delimiters.
    """
    print(f"Reading variant info file (.bim): {filepath}...")
    try:
        # .bim files are typically tab-delimited but can be space-delimited.
        # Polars' read_csv with separator='\s+' is not yet stable, so we try tab then space.
        df = pl.read_csv(
            filepath, separator='\t', has_header=False,
            new_columns=['CHR', 'ID', 'CM', 'POS', 'A1', 'A2'],
            dtypes={'CHR': pl.Utf8, 'ID': pl.Utf8, 'CM': pl.Utf8, 'POS': pl.Int64, 'A1': pl.Utf8, 'A2': pl.Utf8}
        )
        print(f"  > Loaded info for {df.height} variants.")
        return df
    except pl.ShapeError: # This error suggests the delimiter was wrong.
        print("  > Tab delimiter failed, trying space delimiter...")
        df = pl.read_csv(
            filepath, separator=' ', has_header=False,
            new_columns=['CHR', 'ID', 'CM', 'POS', 'A1', 'A2'],
            dtypes={'CHR': pl.Utf8, 'ID': pl.Utf8, 'CM': pl.Utf8, 'POS': pl.Int64, 'A1': pl.Utf8, 'A2': pl.Utf8}
        )
        print(f"  > Loaded info for {df.height} variants.")
        return df
    except Exception as e:
        print(f"Error reading .bim file: {e}", file=sys.stderr)
        sys.exit(1)


def read_score_file(filepath: str, id_col: int, allele_col: int, score_col: int) -> dict:
    """
    Reads the scoring file efficiently using Polars and creates a dictionary lookup.
    """
    print(f"Reading score file: {filepath}...")
    try:
        id_col_idx, allele_col_idx, score_col_idx = id_col - 1, allele_col - 1, score_col - 1
        df = pl.read_csv(filepath, separator='\s+', has_header=True, comment_prefix='#', try_parse_dates=False)
        
        # Select columns by their integer index to match original logic
        id_series = df.select(pl.col(df.columns[id_col_idx]).cast(pl.Utf8)).to_series()
        allele_series = df.select(pl.col(df.columns[allele_col_idx]).cast(pl.Utf8)).to_series()
        score_series = df.select(pl.col(df.columns[score_col_idx]).cast(pl.Float64)).to_series()

        # Create the dictionary lookup using a fast zip comprehension
        score_data = {
            (vid, allele): score
            for vid, allele, score in zip(id_series, allele_series, score_series)
        }
        
        print(f"  > Loaded {len(score_data)} variant scores.")
        return score_data
    except Exception as e:
        print(f"Error reading score file: {e}", file=sys.stderr)
        sys.exit(1)


def find_ambiguous_samples(bed_filepath: str, num_samples: int, bim_df: pl.DataFrame) -> set:
    """
    Pass 1: Identifies samples with non-missing genotypes for multiple variants
    at the same genomic position using a fully vectorized approach.
    """
    print("\n--- Pass 1: Identifying ambiguous samples ---")
    
    # Find positions that have more than one variant mapped to them
    ambiguous_pos_df = bim_df.group_by(['CHR', 'POS']).count().filter(pl.col('count') > 1)
    
    if ambiguous_pos_df.height == 0:
        print("  > No variants at duplicated positions found. Skipping ambiguity check.")
        return set()
        
    # Get the indices of all variants located at these ambiguous positions
    variants_to_check_df = bim_df.with_row_index().join(
        ambiguous_pos_df, on=['CHR', 'POS']
    )
    indices_to_check = variants_to_check_df['index'].to_numpy()
    pos_map = {(row['CHR'], row['POS']): [] for row in variants_to_check_df.iter_rows(named=True)}
    for row in variants_to_check_df.iter_rows(named=True):
        pos_map[(row['CHR'], row['POS'])].append(row['index'])
        
    bytes_per_variant = math.ceil(num_samples / 4)
    genotype_counts_at_pos = {pos: np.zeros(num_samples, dtype=np.uint8) for pos in pos_map}
    
    # Pre-calculate bit shifts for unpacking 4 genotypes from each byte
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)

    with open(bed_filepath, "rb") as f:
        magic = f.read(3)
        if magic != b'\x6c\x1b\x01':
            raise ValueError("Invalid .bed file: incorrect magic number.")
            
        # Iterate through all variants but only process the ones at ambiguous positions
        for i in range(bim_df.height):
            variant_bytes = f.read(bytes_per_variant)
            if not variant_bytes: break
            
            if i in indices_to_check:
                # --- Vectorized Genotype Unpacking ---
                byte_array = np.frombuffer(variant_bytes, dtype=np.uint8)
                unpacked = (byte_array[:, np.newaxis] >> shifts) & 0b11
                genotypes = unpacked.flatten()[:num_samples]
                
                # We only care about non-missing genotypes (code is not 0b01)
                non_missing_mask = (genotypes != 0b01)
                
                # Find which position this variant belongs to and add its non-missing counts
                variant_info = bim_df[i]
                pos_tuple = (variant_info['CHR'][0], variant_info['POS'][0])
                genotype_counts_at_pos[pos_tuple] += non_missing_mask

    # A sample is ambiguous if its count of non-missing genotypes at any position is > 1
    is_ambiguous = np.zeros(num_samples, dtype=bool)
    for pos_counts in genotype_counts_at_pos.values():
        is_ambiguous |= (pos_counts > 1)
        
    nan_indices = set(np.where(is_ambiguous)[0])
    
    if nan_indices:
        print(f"  > Identified {len(nan_indices)} sample(s) with ambiguous genotypes at single positions.")
    else:
        print("  > No ambiguous samples found.")
        
    return nan_indices


def calculate_scores(bed_filepath: str, num_samples: int, bim_df: pl.DataFrame, score_data: dict) -> tuple:
    """
    Pass 2: Calculates scores for all samples using a fully vectorized approach.
    """
    print("\n--- Pass 2: Calculating scores ---")
    weighted_score_sum = np.zeros(num_samples, dtype=np.float64)
    denominator_ct = np.zeros(num_samples, dtype=np.int64)
    variants_processed_count = 0
    bytes_per_variant = math.ceil(num_samples / 4)

    # Pre-computation for speed
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
    # Maps PLINK genotype codes [00, 01, 10, 11] to A2 dosage. Missing (01) is -1.
    geno_to_a2_dosage = np.array([0, -1, 1, 2], dtype=np.int8)

    # Use Polars series for fast iteration
    bim_iter = zip(bim_df['ID'], bim_df['A1'], bim_df['A2'])

    with open(bed_filepath, "rb") as f:
        # Skip magic number
        f.seek(3)
        for variant_id, a1, a2 in bim_iter:
            score_info = None
            # Check for score matching either allele
            if (variant_id, a2) in score_data:
                score_info = {'weight': score_data[(variant_id, a2)], 'effect_is_a2': True}
            elif (variant_id, a1) in score_data:
                score_info = {'weight': score_data[(variant_id, a1)], 'effect_is_a2': False}

            variant_bytes = f.read(bytes_per_variant)
            if not variant_bytes: break

            if score_info:
                variants_processed_count += 1
                
                # --- Vectorized Genotype Processing ---
                byte_array = np.frombuffer(variant_bytes, dtype=np.uint8)
                unpacked_codes = (byte_array[:, np.newaxis] >> shifts) & 0b11
                genotype_codes = unpacked_codes.flatten()[:num_samples]

                dosages_a2 = geno_to_a2_dosage[genotype_codes]
                
                # Create a boolean mask for non-missing genotypes
                non_missing_mask = dosages_a2 != -1
                
                # Skip variant if no one has a non-missing genotype
                if not np.any(non_missing_mask): continue
                
                if score_info['effect_is_a2']:
                    effect_dosages = dosages_a2
                else:  # Effect allele is A1, so dosage is 2 - dosage_A2
                    effect_dosages = 2 - dosages_a2
                
                # Apply calculations only to non-missing samples using the mask
                weight = score_info['weight']
                weighted_score_sum[non_missing_mask] += effect_dosages[non_missing_mask] * weight
                denominator_ct[non_missing_mask] += 2

    print(f"  > Processed {variants_processed_count} variants found in score file.")
    return weighted_score_sum, denominator_ct


def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(
        description="A high-performance Python script to replicate 'plink --score' with "
                    "'no-mean-imputation', matching PLINK2's NaN behavior "
                    "for ambiguous positions.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--bfile', required=True, help='Prefix for the input PLINK .bed, .bim, and .fam fileset.')
    parser.add_argument('--score', required=True, help='Path to the scoring file. Must have a header.')
    parser.add_argument('--out', required=True, help='Prefix for the output file (e.g., my_scores).')
    parser.add_argument('columns', nargs=3, type=int, help="Three 1-based column numbers for: Variant_ID Effect_Allele Score")
    args = parser.parse_args()

    # --- Load Metadata using high-speed Polars readers ---
    fam_df = read_fam_file(f"{args.bfile}.fam")
    bim_df = read_bim_file(f"{args.bfile}.bim")
    score_data = read_score_file(args.score, args.columns[0], args.columns[1], args.columns[2])
    
    num_samples = fam_df.height
    bed_filepath = f"{args.bfile}.bed"

    # --- Pass 1 & 2 using vectorized functions ---
    nan_indices = find_ambiguous_samples(bed_filepath, num_samples, bim_df)
    weighted_sum, denom_ct = calculate_scores(bed_filepath, num_samples, bim_df, score_data)

    # --- Finalize and Write Output ---
    print("\n--- Finalizing results ---")
    
    # Calculate average score, handling division by zero by defaulting to NaN
    final_scores_avg = np.divide(
        weighted_sum, denom_ct,
        out=np.full(num_samples, np.nan, dtype=float),
        where=(denom_ct != 0)
    )

    # Apply NaN mask for ambiguous samples identified in Pass 1
    if nan_indices:
        final_scores_avg[list(nan_indices)] = np.nan
        print(f"  > Applied NaN to {len(nan_indices)} sample(s).")
    
    # Construct final DataFrame in Polars for memory efficiency
    output_df_pl = pl.DataFrame({
        '#FID': fam_df['FID'],
        'IID': fam_df['IID'],
        'NMISS_ALLELE_CT': denom_ct,
        'SCORE1_AVG': final_scores_avg
    })

    output_filepath = f"{args.out}.sscore"
    try:
        # CRITICAL STEP FOR EXACT BEHAVIOR:
        # Convert to pandas JUST for the to_csv call to perfectly replicate
        # the '%.6g' float formatting and 'nan' string representation.
        output_df_pd = output_df_pl.to_pandas()
        
        output_df_pd.to_csv(
            output_filepath, 
            sep='\t', 
            index=False,
            float_format='%.6g', 
            na_rep='nan'
        )
        print(f"Success! Results written to {output_filepath}")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
