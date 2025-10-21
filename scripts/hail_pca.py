from __future__ import annotations

import argparse
import time

import numpy as np
import hail as hl  # type: ignore


# Path to the public phased haplotypes. Only used when Hail is available.
DATA_PATH = "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/"

SNP_LIST_TSV = "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv"

OUT = "pca_out"

K = 20
# ================================================================


def _read_genotypes(path: str):
    """Read a Hail MatrixTable or import VCFs from ``path``."""

    if path.endswith(".mt"):
        mt = hl.read_matrix_table(path)
    else:
        mt = hl.import_vcf(
            path,
            force_bgz=True,
            reference_genome="GRCh38",
        )
    return mt


def _read_sites_tsv(url: str):
    """Read the SNP list TSV and return a Hail Table keyed by (locus, alleles)."""

    ht = hl.import_table(url, impute=True, force=True)

    lower_map = {k: k.lower() for k in ht.row_value}
    ht = ht.rename(lower_map)

    chrom_col = "chrom" if "chrom" in ht.row else "chr" if "chr" in ht.row else "chromosome" if "chromosome" in ht.row else None
    pos_col = "pos" if "pos" in ht.row else "position" if "position" in ht.row else "bp" if "bp" in ht.row else None
    ref_col = "ref" if "ref" in ht.row else "ref_allele" if "ref_allele" in ht.row else None
    alt_col = "alt" if "alt" in ht.row else "alt_allele" if "alt_allele" in ht.row else None

    ht = ht.select(
        chrom=hl.str(ht[chrom_col]),
        pos=hl.int32(ht[pos_col]),
        ref=hl.str(ht[ref_col]),
        alt=hl.str(ht[alt_col]),
    )

    norm_chr = hl.if_else(ht.chrom.startswith("chr"), ht.chrom, hl.str("chr") + ht.chrom)
    locus = hl.parse_locus(hl.format("{}:{}", norm_chr, ht.pos), reference_genome="GRCh38")
    alleles = hl.array([ht.ref, ht.alt])

    sites_ht = ht.select(locus=locus, alleles=alleles)
    sites_ht = sites_ht.key_by("locus", "alleles")
    return sites_ht


def _hail_main(loop_seconds: float | None = None) -> None:
    """Run the PCA pipeline using the Hail backend."""

    hl.init(app_name="hgdp1kg_pca_defaults", default_reference="GRCh38")

    mt = _read_genotypes(DATA_PATH)

    mt = mt.filter_rows(
        (hl.len(mt.alleles) == 2) & hl.is_snp(mt.alleles[0], mt.alleles[1])
    )

    sites_ht = _read_sites_tsv(SNP_LIST_TSV)
    mt = mt.semi_join_rows(sites_ht)

    mt = mt.key_rows_by(mt.locus, mt.alleles)

    pruned = hl.ld_prune(mt.GT)

    mt_pruned = mt.filter_rows(hl.is_defined(pruned[mt.row_key]))

    eigenvalues, scores, loadings = hl.hwe_normalized_pca(
        mt_pruned.GT, k=K, compute_loadings=True
    )

    fs = hl.current_backend().fs
    if not fs.exists(OUT):
        fs.mkdir(OUT)

    pc_cols = {f"PC{i+1}": scores.scores[i] for i in range(K)}
    scores_exp = scores.select("s", **pc_cols)
    scores_exp.export(f"{OUT}/pca_scores.tsv.bgz")

    load_cols = {f"PC{i+1}": loadings.loadings[i] for i in range(K)}
    loadings_exp = loadings.select(
        chrom=loadings.locus.contig,
        pos=loadings.locus.position,
        ref=loadings.alleles[0],
        alt=loadings.alleles[1],
        **load_cols,
    )
    loadings_exp.export(f"{OUT}/pca_loadings.tsv.bgz")

    ev_ht = hl.Table.parallelize(
        hl.enumerate(hl.literal(eigenvalues)).map(
            lambda x: hl.struct(component=x[0] + 1, eigenvalue=x[1])
        ),
        key="component",
    )
    ev_ht.export(f"{OUT}/eigenvalues.tsv")

    pruned_sites = pruned.key_by().select(
        chrom=pruned.locus.contig,
        pos=pruned.locus.position,
        ref=pruned.alleles[0],
        alt=pruned.alleles[1],
    )
    pruned_sites.export(f"{OUT}/pruned_sites.tsv.bgz")

    n_samples = mt_pruned.count_cols()
    n_variants = mt_pruned.count_rows()
    with fs.open(f"{OUT}/summary.txt", "w") as f:
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Variants after LD-prune: {n_variants}\n")
        f.write(f"PCs computed (k): {K}\n")

    if loop_seconds is not None:
        _busy_wait(loop_seconds)


def _busy_wait(seconds: float) -> None:
    """Perform deterministic numerical work for at least ``seconds`` seconds."""

    target = time.perf_counter() + seconds
    rng = np.random.default_rng(0)
    payload = rng.random((1024, 256), dtype=np.float64)
    while time.perf_counter() < target:
        payload = payload @ payload.T
        payload = np.tanh(payload)
        payload = payload[:1024, :256]


def main(loop_seconds: float | None = None) -> None:
    _hail_main(loop_seconds=loop_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA on HGDP/1KG data using Hail."
    )
    parser.add_argument(
        "--loop-seconds",
        type=float,
        default=None,
        help=(
            "Continue performing dense linear algebra for at least this many "
            "seconds after PCA outputs are written."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(loop_seconds=args.loop_seconds)
