#!/opt/homebrew/bin/python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_NAME = "hwe_1kg_hgdp_gsa_v3"
MODEL_URL = (
    "https://github.com/SauersML/gnomon/releases/download/models-v1/"
    "hwe_1kg_hgdp_gsa_v3.json.zst"
)
ROOT = Path("/tmp/gnomon_project_prefit_microarray")
DATA_DIR = ROOT / "data"
MODEL_JSON = ROOT / f"{MODEL_NAME}.json"
MODEL_ZST = ROOT / f"{MODEL_NAME}.json.zst"
RESULTS_CSV = ROOT / "project_scaling.csv"
PLOT_PNG = ROOT / "project_scaling.png"
REALISTIC_RESULTS_CSV = ROOT / "project_scaling_realistic.csv"
REALISTIC_PLOT_PNG = ROOT / "project_scaling_realistic.png"
GNOMON = Path.home() / ".local" / "bin" / "gnomon"
COUNTS = [1, 4, 16, 64, 256, 512, 1024]
REPEATS = 1

DEFAULT_MODE = "realistic"

PLINK_MAGIC_HEADER = bytes([0x6C, 0x1B, 0x01])
def run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def ensure_gnomon() -> None:
    if GNOMON.exists():
        return
    raise SystemExit(f"missing gnomon binary at {GNOMON}")


def ensure_model_json() -> Path:
    ROOT.mkdir(parents=True, exist_ok=True)
    if MODEL_JSON.exists():
        return MODEL_JSON
    if not MODEL_ZST.exists():
        run(["curl", "-L", "-o", str(MODEL_ZST), MODEL_URL])
    with MODEL_JSON.open("wb") as out:
        proc = subprocess.run(
            ["zstd", "-d", "-c", str(MODEL_ZST)],
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr)
    return MODEL_JSON


def extract_variant_keys(model_json: Path) -> list[tuple[str, int, str, str]]:
    decoder = json.JSONDecoder()
    keys: list[tuple[str, int, str, str]] = []
    started = False
    with model_json.open("r", encoding="utf-8") as fh:
        buffer = ""
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk and not buffer:
                break
            buffer += chunk
            if not started:
                marker = '"variant_keys"'
                idx = buffer.find(marker)
                if idx == -1:
                    if chunk:
                        buffer = buffer[-(len(marker) + 4) :]
                        continue
                    raise RuntimeError("variant_keys not found in model json")
                bracket_idx = buffer.find("[", idx + len(marker))
                if bracket_idx == -1:
                    if chunk:
                        buffer = buffer[idx:]
                        continue
                    raise RuntimeError("variant_keys array start not found in model json")
                buffer = buffer[bracket_idx + 1 :]
                started = True
            pos = 0
            while True:
                while pos < len(buffer) and buffer[pos] in " \n\r\t,":
                    pos += 1
                if pos >= len(buffer):
                    buffer = ""
                    break
                if buffer[pos] == "]":
                    return keys
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    buffer = buffer[pos:]
                    break
                alleles = obj.get("alleles")
                if not isinstance(alleles, list) or len(alleles) < 2:
                    raise RuntimeError("variant_key is missing allele pair")
                keys.append(
                    (
                        obj["chromosome"],
                        int(obj["position"]),
                        str(alleles[0]),
                        str(alleles[1]),
                    )
                )
                pos = end
    raise RuntimeError("unterminated variant_keys array")


def per_variant_maf(variant_idx: int, chrom: str, pos: int) -> float:
    # Heavy skew toward common/low-frequency array loci with some rare loci mixed in.
    seed = variant_idx * 1315423911 + pos * 2654435761 + sum(ch.encode("ascii")[0] for ch in chrom)
    u = ((seed ^ (seed >> 13) ^ (seed << 7)) & 0xFFFFFFFF) / 0xFFFFFFFF
    if u < 0.55:
        maf = 0.20 * (u / 0.55) ** 0.55
    elif u < 0.90:
        maf = 0.20 + 0.25 * ((u - 0.55) / 0.35)
    else:
        maf = 0.45 + 0.045 * ((u - 0.90) / 0.10)
    return min(max(maf, 0.001), 0.495)


def sample_dropout_rate(person_idx: int) -> float:
    bucket = person_idx % 32
    if bucket == 0:
        return 0.03
    if bucket < 4:
        return 0.012
    return 0.004


def pseudo_uniform(*parts: int) -> float:
    x = 0x9E3779B97F4A7C15
    for part in parts:
        x ^= (part + 0x9E3779B97F4A7C15 + ((x << 6) & 0xFFFFFFFFFFFFFFFF) + (x >> 2)) & 0xFFFFFFFFFFFFFFFF
    return (x & 0xFFFFFFFF) / 0xFFFFFFFF


def make_fam(prefix: Path, n_samples: int) -> None:
    fam_path = prefix.with_suffix(".fam")
    with fam_path.open("w", encoding="ascii") as fh:
        for i in range(n_samples):
            sex = 1 + (i % 2)
            fh.write(f"F{i:06}\tI{i:06}\t0\t0\t{sex}\t-9\n")


def make_bim(prefix: Path, variant_keys: list[tuple[str, int, str, str]]) -> None:
    bim_path = prefix.with_suffix(".bim")
    with bim_path.open("w", encoding="ascii") as fh:
        for chrom, pos, ref_allele, alt_allele in variant_keys:
            fh.write(f"{chrom}\t{chrom}:{pos}\t0\t{pos}\t{alt_allele}\t{ref_allele}\n")


def plink_code(genotype: int | None) -> int:
    if genotype is None:
        return 0b01
    if genotype == 0:
        return 0b00
    if genotype == 1:
        return 0b10
    if genotype == 2:
        return 0b11
    raise ValueError(f"unexpected genotype {genotype}")


def synthetic_genotype(person_idx: int, variant_idx: int, chrom: str, pos: int) -> int | None:
    u_missing = pseudo_uniform(person_idx, variant_idx, pos, 17)
    if u_missing < sample_dropout_rate(person_idx):
        return None

    maf = per_variant_maf(variant_idx, chrom, pos)

    # Mild batch effect and chromosome-specific drift to avoid overly clean synthetic structure.
    batch_shift = ((person_idx % 8) - 3.5) * 0.0025
    chrom_shift = ((sum(ch.encode("ascii")[0] for ch in chrom) % 5) - 2) * 0.0015
    p = min(max(maf + batch_shift + chrom_shift, 0.001), 0.495)

    u = pseudo_uniform(person_idx, variant_idx, pos, 29)
    p0 = (1.0 - p) * (1.0 - p)
    p1 = 2.0 * p * (1.0 - p)
    if u < p0:
        return 0
    if u < p0 + p1:
        return 1
    return 2


def make_bed(prefix: Path, n_samples: int, variant_keys: list[tuple[str, int, str, str]]) -> None:
    bed_path = prefix.with_suffix(".bed")
    bytes_per_variant = math.ceil(n_samples / 4)
    with bed_path.open("wb") as fh:
        fh.write(PLINK_MAGIC_HEADER)
        row = bytearray(bytes_per_variant)
        for variant_idx, (chrom, pos, _a1, _a2) in enumerate(variant_keys):
            for i in range(bytes_per_variant):
                row[i] = 0
            for person_idx in range(n_samples):
                gt = synthetic_genotype(person_idx, variant_idx, chrom, pos)
                row[person_idx // 4] |= plink_code(gt) << ((person_idx % 4) * 2)
            fh.write(row)


def make_dataset(prefix: Path, n_samples: int, variant_keys: list[tuple[str, int, str, str]]) -> None:
    make_fam(prefix, n_samples)
    make_bim(prefix, variant_keys)
    make_bed(prefix, n_samples, variant_keys)


def benchmark_one(prefix: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    n_samples = sum(1 for _ in prefix.with_suffix(".fam").open("r", encoding="ascii"))
    for repeat in range(1, REPEATS + 1):
        print(f"benchmark samples={n_samples} repeat={repeat}", flush=True)
        out_path = prefix.with_suffix(".projection_scores.tsv")
        if out_path.exists():
            out_path.unlink()
        start = time.perf_counter()
        proc = subprocess.run(
            [str(GNOMON), "project", "--model", MODEL_NAME, str(prefix.with_suffix(".bed"))],
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise SystemExit(f"projection failed for samples={n_samples}, repeat={repeat}")
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "samples": n_samples,
                "repeat": repeat,
                "seconds": elapsed,
                "samples_per_sec": n_samples / elapsed,
                "ms_per_sample": elapsed * 1000.0 / n_samples,
            }
        )
        print(
            "METRIC "
            f"samples={n_samples} "
            f"repeat={repeat} "
            f"seconds={elapsed:.6f} "
            f"samples_per_sec={n_samples / elapsed:.6f} "
            f"ms_per_sample={elapsed * 1000.0 / n_samples:.6f}",
            flush=True,
        )
        sys.stderr.write(proc.stderr)
    return rows


def save_results(rows: list[dict[str, float | int]], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["samples", "repeat", "seconds", "samples_per_sec", "ms_per_sample"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: list[dict[str, float | int]], output_png: Path, title_suffix: str) -> None:
    grouped: dict[int, list[dict[str, float | int]]] = {}
    for row in rows:
        grouped.setdefault(int(row["samples"]), []).append(row)
    xs = sorted(grouped)
    median_throughput = []
    for x in xs:
        throughput = sorted(float(r["samples_per_sec"]) for r in grouped[x])
        mid = len(throughput) // 2
        median_throughput.append(throughput[mid])

    plt.figure(figsize=(9, 5.5))
    plt.plot(xs, median_throughput, marker="o", linewidth=2)
    plt.xscale("log", base=2)
    plt.xlabel("Sample count")
    plt.ylabel("Samples/sec")
    plt.title(f"gnomon project --model {MODEL_NAME} scaling ({title_suffix})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        default=",".join(str(n) for n in COUNTS),
        help="comma-separated sample counts to benchmark",
    )
    parser.add_argument(
        "--output-csv",
        default=str(REALISTIC_RESULTS_CSV),
        help="where to write benchmark CSV",
    )
    parser.add_argument(
        "--output-png",
        default=str(REALISTIC_PLOT_PNG),
        help="where to write benchmark plot",
    )
    parser.add_argument(
        "--title-suffix",
        default=DEFAULT_MODE,
        help="suffix appended to plot title",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = [int(part) for part in args.counts.split(",") if part.strip()]
    output_csv = Path(args.output_csv)
    output_png = Path(args.output_png)
    ensure_gnomon()
    model_json = ensure_model_json()
    variant_keys = extract_variant_keys(model_json)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    for n_samples in counts:
        prefix = DATA_DIR / f"gsa_v3_{n_samples}"
        if prefix.with_suffix(".bed").exists():
            for ext in (".bed", ".bim", ".fam", ".projection_scores.tsv"):
                path = prefix.with_suffix(ext)
                if path.exists():
                    path.unlink()
        make_dataset(prefix, n_samples, variant_keys)
        rows.extend(benchmark_one(prefix))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    save_results(rows, output_csv)
    plot_results(rows, output_png, args.title_suffix)

    print(f"results_csv={output_csv}")
    print(f"plot_png={output_png}")


if __name__ == "__main__":
    main()
