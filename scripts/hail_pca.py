"""Run PCA on genomic data using Hail."""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
from pathlib import Path
from typing import Iterable, List
from urllib.error import URLError
from urllib.request import urlopen

import hail as hl

DEFAULT_DATA_PATH = (
    "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/*.bcf"
)
GCS_CONNECTOR_URL = "https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-latest.jar"
DEFAULT_NUM_PCS = 10
DEFAULT_OUTPUT_PREFIX = "hail_pca"


def _infer_variant_format(path: str) -> str | None:
    """Return the obvious variant format for *path* if one is implied."""

    lowered = path.lower()
    if lowered.endswith((".bcf", ".bcf.bgz", ".bcf.gz")):
        return "bcf"
    if lowered.endswith((".vcf", ".vcf.gz", ".vcf.bgz")):
        return "vcf"
    return None


def _detect_variant_format_from_file(path: str) -> str | None:
    """Return ``"bcf"`` or ``"vcf"`` if the file content makes it obvious."""

    try:
        with hl.hadoop_open(path, "rb") as raw:
            prefix = raw.read(4)
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise RuntimeError(f"Unable to inspect '{path}' to determine its format") from exc

    if prefix.startswith(b"BCF"):
        return "bcf"
    if prefix.startswith(b"##"):
        return "vcf"
    if prefix[:2] != b"\x1f\x8b":
        return None

    try:
        with hl.hadoop_open(path, "rb") as compressed, gzip.GzipFile(fileobj=compressed) as gz:
            header = gz.read(4)
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise RuntimeError(
            f"Unable to inspect compressed file '{path}' to determine its format"
        ) from exc

    if header.startswith(b"BCF"):
        return "bcf"
    if header.startswith(b"##"):
        return "vcf"
    return None


def _resolve_variant_path(path: str) -> tuple[str, str]:
    """Return a glob that selects variants under *path* alongside the format."""

    trimmed = path.rstrip("/")
    inferred = _infer_variant_format(trimmed)
    if inferred is not None:
        return trimmed, inferred

    if "*" in trimmed:
        raise ValueError(
            "Unable to infer variant format from glob pattern. "
            "Please include a file extension in the --data-path value."
        )

    listing_target = path if path.endswith("/") else f"{path}/"
    try:
        entries = hl.hadoop_ls(listing_target)
    except Exception as exc:  # pragma: no cover - filesystem errors
        raise FileNotFoundError(
            f"Failed to list files under '{listing_target}'. Ensure the path exists."
        ) from exc

    files = [entry["path"] for entry in entries if not entry.get("is_dir")]
    if not files:
        raise FileNotFoundError(f"No files were found under '{listing_target}'.")

    for suffix, fmt in (
        (".bcf", "bcf"),
        (".bcf.bgz", "bcf"),
        (".bcf.gz", "bcf"),
        (".vcf.bgz", "vcf"),
        (".vcf.gz", "vcf"),
        (".vcf", "vcf"),
    ):
        matching = [name for name in files if name.lower().endswith(suffix)]
        if not matching:
            continue

        detected = _detect_variant_format_from_file(matching[0]) or fmt
        base = listing_target.rstrip("/")
        return f"{base}/*{suffix}", detected

    raise FileNotFoundError(
        "Unable to determine whether the input files are VCF or BCF. "
        "Please pass a path with an explicit extension."
    )


def _ensure_gcs_connector() -> str:
    """Download the GCS connector jar if it is not already cached."""

    destination = Path.home() / ".hail" / "gcs-connector-hadoop3-latest.jar"
    if destination.exists():
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")

    try:
        with urlopen(GCS_CONNECTOR_URL) as response, temp_path.open("wb") as output:
            shutil.copyfileobj(response, output)
    except URLError as exc:  # pragma: no cover - network failure reporting
        raise RuntimeError("Failed to download the GCS connector required for gs:// support") from exc

    temp_path.replace(destination)
    return str(destination)


def _spark_conf_for_path(path: str) -> dict[str, str]:
    """Return Spark configuration needed for the input path."""

    if path.startswith("gs://"):
        connector_path = _ensure_gcs_connector()
        return {
            "spark.hadoop.fs.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            "spark.hadoop.fs.AbstractFileSystem.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
            "spark.hadoop.fs.gs.auth.service.account.enable": "false",
            "spark.hadoop.fs.gs.auth.null.enable": "true",
            "spark.jars": connector_path,
        }

    return {}


def _read_genotypes(path: str) -> hl.MatrixTable:
    """Read genotype data from either a MatrixTable or variant inputs."""

    if path.endswith(".mt"):
        logging.info("Reading MatrixTable from %s", path)
        return hl.read_matrix_table(path)

    variant_path, variant_format = _resolve_variant_path(path)

    if variant_format == "bcf":
        logging.info("Importing BCFs from %s", variant_path)
        return hl.import_bcf(variant_path)

    logging.info("Importing VCFs from %s", variant_path)
    return hl.import_vcf(variant_path, force_bgz=True, array_elements_required=False)


def _export_scores(scores: hl.Table, output_prefix: str) -> None:
    scores.export(f"{output_prefix}.scores.tsv.bgz")


def _export_loadings(loadings: hl.Table, output_prefix: str) -> None:
    loadings.export(f"{output_prefix}.loadings.tsv.bgz")


def _export_eigenvalues(eigenvalues: Iterable[float], output_prefix: str) -> None:
    rows = [
        {"component": i + 1, "eigenvalue": float(value)}
        for i, value in enumerate(eigenvalues)
    ]
    eigenvalue_ht = hl.Table.parallelize(rows).key_by("component")
    eigenvalue_ht.export(f"{output_prefix}.eigenvalues.tsv.bgz")


def run_pca(data_path: str, output_prefix: str, n_pcs: int) -> None:
    """Run PCA on the provided data and export the results."""

    logging.info(
        "Starting PCA with data_path=%s, output_prefix=%s, n_pcs=%d",
        data_path,
        output_prefix,
        n_pcs,
    )
    hl.init(spark_conf=_spark_conf_for_path(data_path))

    mt = _read_genotypes(data_path)

    if "GT" not in mt.entry:
        raise ValueError("Genotype MatrixTable must contain a 'GT' entry field")

    eigenvalues, scores, loadings = hl.hwe_normalized_pca(mt.GT, k=n_pcs)

    _export_eigenvalues(eigenvalues, output_prefix)
    _export_scores(scores, output_prefix)
    _export_loadings(loadings, output_prefix)

    logging.info("PCA complete")


def parse_args(args: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=(
            "Path to a Hail MatrixTable (.mt) or directory/glob of VCF/BCF files. "
            "Defaults to the public HGDP/1kG BCF directory."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Prefix used for the exported eigenvalues, scores, and loadings. "
            f"Defaults to '{DEFAULT_OUTPUT_PREFIX}'."
        ),
    )
    parser.add_argument(
        "--num-pcs",
        type=int,
        default=DEFAULT_NUM_PCS,
        help="Number of principal components to compute.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity.",
    )
    return parser.parse_args(args=args)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    run_pca(args.data_path, args.output_prefix, args.num_pcs)


if __name__ == "__main__":
    main()
