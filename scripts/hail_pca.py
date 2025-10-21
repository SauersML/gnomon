import hail as hl


DATA_PATH = "gs://gcp-public-data--gnomad/release/3.1/secondary_analyses/hgdp_1kg_v2/mt/genotypes.mt"

SNP_LIST_TSV = "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv"

OUT = "pca_out"

K = 20
# ================================================================


def _read_genotypes(path: str) -> hl.MatrixTable:
    """
    Read a Hail MatrixTable or import VCFs from `path`.
    This function assumes:
      - If `path` endswith ".mt" -> hl.read_matrix_table(path)
      - Else -> hl.import_vcf(path, reference_genome="GRCh38")
    """
    if path.endswith(".mt"):
        mt = hl.read_matrix_table(path)
    else:
        # Import VCF(s). If your source is BCF, convert to VCF.bgz first.
        mt = hl.import_vcf(
            path,
            force_bgz=True,
            reference_genome="GRCh38"
        )
    return mt


def _read_sites_tsv(url: str) -> hl.Table:
    """
    Read the SNP list TSV and return a Hail Table keyed by (locus, alleles).
    Expected columns in the TSV (case-insensitive): chrom/chr, pos, ref, alt.
    """
    ht = hl.import_table(url, impute=True, force=True)

    # Normalize header names to lower-case for flexible selection
    lower_map = {k: k.lower() for k in ht.row_value}
    ht = ht.rename(lower_map)

    # Expect these canonical names after lowering
    chrom_col = "chrom" if "chrom" in ht.row else "chr" if "chr" in ht.row else "chromosome" if "chromosome" in ht.row else None
    pos_col = "pos" if "pos" in ht.row else "position" if "position" in ht.row else "bp" if "bp" in ht.row else None
    ref_col = "ref" if "ref" in ht.row else "ref_allele" if "ref_allele" in ht.row else None
    alt_col = "alt" if "alt" in ht.row else "alt_allele" if "alt_allele" in ht.row else None

    ht = ht.select(
        chrom = hl.str(ht[chrom_col]),
        pos   = hl.int32(ht[pos_col]),
        ref   = hl.str(ht[ref_col]),
        alt   = hl.str(ht[alt_col])
    )

    # Ensure GRCh38-style contigs: Hail's GRCh38 uses "chr1", ..., "chrX".
    # Construct a "chr<contig>:pos" string robust to inputs that may or may not include "chr".
    norm_chr = hl.if_else(ht.chrom.startswith("chr"), ht.chrom, hl.str("chr") + ht.chrom)
    locus = hl.parse_locus(hl.format("{}:{}", norm_chr, ht.pos), reference_genome="GRCh38")
    alleles = hl.array([ht.ref, ht.alt])

    sites_ht = ht.select(locus=locus, alleles=alleles)
    sites_ht = sites_ht.key_by("locus", "alleles")
    return sites_ht


def main():
    # Initialize Hail with GRCh38. Adjust driver/executor memory externally as needed.
    hl.init(app_name="hgdp1kg_pca_defaults", default_reference="GRCh38")

    # Read genotype data
    mt = _read_genotypes(DATA_PATH)

    # Keep only biallelic SNPs
    mt = mt.filter_rows(
        (hl.len(mt.alleles) == 2) & hl.is_snp(mt.alleles[0], mt.alleles[1])
    )

    # Intersect to provided SNP list
    sites_ht = _read_sites_tsv(SNP_LIST_TSV)
    mt = mt.semi_join_rows(sites_ht)

    # Optional: ensure rows are keyed canonically for joins/exports
    mt = mt.key_rows_by(mt.locus, mt.alleles)

    # LD prune with Hail defaults (r2=0.2, window=1,000,000 bp)
    # Docs: hl.ld_prune(..., r2=0.2, bp_window_size=1_000_000, keep_higher_maf=True)
    pruned = hl.ld_prune(mt.GT)  # <- DEFAULTS

    # Filter MT to pruned variants
    mt_pruned = mt.filter_rows(hl.is_defined(pruned[mt.row_key]))

    # HWE-normalized PCA with k=20 (compute loadings = True)
    eigenvalues, scores, loadings = hl.hwe_normalized_pca(
        mt_pruned.GT, k=K, compute_loadings=True
    )

    # ---------- Write outputs ----------
    # Create output directory (local or GCS)
    fs = hl.current_backend().fs
    if not fs.exists(OUT):
        fs.mkdir(OUT)

    # 1) Sample scores (expand array -> PC1..PCK)
    pc_cols = {f"PC{i+1}": scores.scores[i] for i in range(K)}
    scores_exp = scores.select("s", **pc_cols)
    scores_exp.export(f"{OUT}/pca_scores.tsv.bgz")

    # 2) Variant loadings (expand array -> PC1..PCK)
    load_cols = {f"PC{i+1}": loadings.loadings[i] for i in range(K)}
    loadings_exp = loadings.select(
        chrom = loadings.locus.contig,
        pos   = loadings.locus.position,
        ref   = loadings.alleles[0],
        alt   = loadings.alleles[1],
        **load_cols
    )
    loadings_exp.export(f"{OUT}/pca_loadings.tsv.bgz")

    # 3) Eigenvalues
    ev_ht = hl.Table.parallelize(
        hl.enumerate(hl.literal(eigenvalues)).map(
            lambda x: hl.struct(component=x[0] + 1, eigenvalue=x[1])
        ),
        key="component"
    )
    ev_ht.export(f"{OUT}/eigenvalues.tsv")

    # 4) Save pruned sites list for provenance
    pruned_sites = pruned.key_by().select(
        chrom = pruned.locus.contig,
        pos   = pruned.locus.position,
        ref   = pruned.alleles[0],
        alt   = pruned.alleles[1]
    )
    pruned_sites.export(f"{OUT}/pruned_sites.tsv.bgz")

    # 5) Save a small QC summary to text
    n_samples = mt_pruned.count_cols()
    n_variants = mt_pruned.count_rows()
    with fs.open(f"{OUT}/summary.txt", "w") as f:
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Variants after LD-prune: {n_variants}\n")
        f.write(f"PCs computed (k): {K}\n")


if __name__ == "__main__":
    main()
