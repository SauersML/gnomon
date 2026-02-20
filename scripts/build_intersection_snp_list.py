#!/usr/bin/env python3
"""Download SNP source files, parse chr:pos loci, and compute global intersection.

Default URL source is a markdown file containing one URL per line.
The script assumes all sources are on the same genome build.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator


URL_RE = re.compile(r"^https?://\S+$")

# Hardcoded source URLs for intersection/build-inference runs.
SOURCE_URLS = [
    "https://raw.githubusercontent.com/psbaltar/rawDNA2vcf/master/filter/23andme_v4.tsv",
    "https://github.com/SauersML/gnomon/raw/refs/heads/main/data/autosomal.txt",
    "https://raw.githubusercontent.com/SauersML/gnomon/refs/heads/main/data/genome_Joshua_Yoakem_v5_Full_20250129211749.txt",
    "https://github.com/SauersML/reagle/raw/refs/heads/main/data/kat_suricata/ancestrydna.txt",
    "https://raw.githubusercontent.com/SauersML/reagle/refs/heads/main/data/kat_suricata/23andme_genome_kat_suricata_v5_full_20171221130201.txt",
    "https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/genome_Christopher_Smith_v5_Full_20230926164611.zip",
    "https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/dna-data-2023-09-26.zip",
    "https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/humanomniexpress-24/v1-4/InfiniumOmniExpress-24v1-4_A1_csv.zip",
    "https://webdata.illumina.com/downloads/productfiles/infinium-omni5-4/v1-2/infinium-omni5-4-v1-2-a2-manifest-file-csv.zip",
    "https://www.chg.ox.ac.uk/~wrayner/strand/BDCHP-1X10-HUMANHAP550_11218540_C-b37-strand.zip",
    "https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium-gsa-with-gcra/GSA-48v4-0_20085471_D2.csv",
    "https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/global-screening-array-24/v3-0/GSA-24v3-0-A2-manifest-file-csv.zip",
]

HG19_TO_HG38_CHAIN_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"

# GRCh37 (hg19) chromosome lengths (1-based inclusive max)
GRCH37_MAX = {
    "1": 249_250_621,
    "2": 243_199_373,
    "3": 198_022_430,
    "4": 191_154_276,
    "5": 180_915_260,
    "6": 171_115_067,
    "7": 159_138_663,
    "8": 146_364_022,
    "9": 141_213_431,
    "10": 135_534_747,
    "11": 135_006_516,
    "12": 133_851_895,
    "13": 115_169_878,
    "14": 107_349_540,
    "15": 102_531_392,
    "16": 90_354_753,
    "17": 81_195_210,
    "18": 78_077_248,
    "19": 59_128_983,
    "20": 63_025_520,
    "21": 48_129_895,
    "22": 51_304_566,
    "X": 155_270_560,
    "Y": 59_373_566,
    "MT": 16_571,  # UCSC chrM; chrMT (rCRS) is 16_569
}

# GRCh38 (hg38) chromosome lengths (1-based inclusive max)
GRCH38_MAX = {
    "1": 248_956_422,
    "2": 242_193_529,
    "3": 198_295_559,
    "4": 190_214_555,
    "5": 181_538_259,
    "6": 170_805_979,
    "7": 159_345_973,
    "8": 145_138_636,
    "9": 138_394_717,
    "10": 133_797_422,
    "11": 135_086_622,
    "12": 133_275_309,
    "13": 114_364_328,
    "14": 107_043_718,
    "15": 101_991_189,
    "16": 90_338_345,
    "17": 83_257_441,
    "18": 80_373_285,
    "19": 58_617_616,
    "20": 64_444_167,
    "21": 46_709_983,
    "22": 50_818_468,
    "X": 156_040_895,
    "Y": 57_227_415,
    "MT": 16_569,
}


def _build_hint_patterns() -> tuple[list[tuple[str, re.Pattern[str]]], list[tuple[str, re.Pattern[str]]]]:
    b37 = [
        ("hg19", re.compile(r"(^|[^a-z0-9])hg19([^a-z0-9]|$)", re.IGNORECASE)),
        ("grch37", re.compile(r"(^|[^a-z0-9])grch37([^a-z0-9]|$)", re.IGNORECASE)),
        ("build37", re.compile(r"(^|[^a-z0-9])build[_ -]?37([^a-z0-9]|$)", re.IGNORECASE)),
        ("b37", re.compile(r"(^|[^a-z0-9])b37([^a-z0-9]|$)", re.IGNORECASE)),
    ]
    b38 = [
        ("hg38", re.compile(r"(^|[^a-z0-9])hg38([^a-z0-9]|$)", re.IGNORECASE)),
        ("grch38", re.compile(r"(^|[^a-z0-9])grch38([^a-z0-9]|$)", re.IGNORECASE)),
        ("build38", re.compile(r"(^|[^a-z0-9])build[_ -]?38([^a-z0-9]|$)", re.IGNORECASE)),
        ("b38", re.compile(r"(^|[^a-z0-9])b38([^a-z0-9]|$)", re.IGNORECASE)),
    ]
    return b37, b38


BUILD37_HINT_PATTERNS, BUILD38_HINT_PATTERNS = _build_hint_patterns()


def detect_build_hints_from_name(url: str, member_name: str | None = None) -> dict[str, object]:
    hay = " ".join([url, member_name or ""])
    hits37 = [name for name, pat in BUILD37_HINT_PATTERNS if pat.search(hay)]
    hits38 = [name for name, pat in BUILD38_HINT_PATTERNS if pat.search(hay)]
    suggested = None
    if hits37 and not hits38:
        suggested = "GRCh37"
    elif hits38 and not hits37:
        suggested = "GRCh38"
    elif hits37 and hits38:
        suggested = "conflict"
    return {
        "suggested_build": suggested,
        "hits_grch37": hits37,
        "hits_grch38": hits38,
    }


def detect_build_indicators(markers: Iterable[str]) -> dict[str, object]:
    """Infer build by checking coordinates against max chromosome lengths.

    Returns counts of loci incompatible with each build and a suggested build,
    but can be ambiguous if no markers hit discriminating regions.
    """
    too_big_37 = 0
    too_big_38 = 0
    max_by_chr: dict[str, int] = {}
    for m in markers:
        chrom, pos_s = m.split(":")
        pos = int(pos_s)
        prev = max_by_chr.get(chrom)
        if prev is None or pos > prev:
            max_by_chr[chrom] = pos

        max37 = GRCH37_MAX.get(chrom)
        max38 = GRCH38_MAX.get(chrom)
        if max37 is not None and pos > max37:
            too_big_37 += 1
        if max38 is not None and pos > max38:
            too_big_38 += 1

    suggested = "ambiguous"
    if too_big_37 > 0 and too_big_38 == 0:
        suggested = "GRCh38"
    elif too_big_38 > 0 and too_big_37 == 0:
        suggested = "GRCh37"
    elif too_big_37 > 0 and too_big_38 > 0:
        suggested = "conflict"

    implies_37: list[dict[str, int | str]] = []
    implies_38: list[dict[str, int | str]] = []
    for chrom, max_pos in sorted(max_by_chr.items(), key=lambda kv: chrom_sort_key(f"{kv[0]}:{kv[1]}")):
        max37 = GRCH37_MAX.get(chrom)
        max38 = GRCH38_MAX.get(chrom)
        if max37 is None or max38 is None:
            continue
        if max37 < max38 and (max37 < max_pos <= max38):
            implies_38.append(
                {
                    "chrom": chrom,
                    "max_pos": max_pos,
                    "grch37_max": max37,
                    "grch38_max": max38,
                }
            )
        elif max38 < max37 and (max38 < max_pos <= max37):
            implies_37.append(
                {
                    "chrom": chrom,
                    "max_pos": max_pos,
                    "grch37_max": max37,
                    "grch38_max": max38,
                }
            )

    positional_suggested = "ambiguous"
    if implies_37 and not implies_38:
        positional_suggested = "GRCh37"
    elif implies_38 and not implies_37:
        positional_suggested = "GRCh38"
    elif implies_37 and implies_38:
        positional_suggested = "conflict"

    return {
        "suggested_build": suggested,
        "too_big_grch37": too_big_37,
        "too_big_grch38": too_big_38,
        "max_by_chr": max_by_chr,
        "positional_suggested_build": positional_suggested,
        "implies_grch37_by_chrmax": implies_37,
        "implies_grch38_by_chrmax": implies_38,
    }


def detect_build_from_url(url: str, member_name: str | None = None) -> str | None:
    name_hint = detect_build_hints_from_name(url=url, member_name=member_name)
    suggested = name_hint["suggested_build"]
    if suggested in {"GRCh37", "GRCh38"}:
        return str(suggested)
    return None


def detect_build_from_text(lines: Iterable[str]) -> str | None:
    joined = "\n".join([ln.strip().lower() for ln in lines if ln.strip()])
    if "grch37" in joined or "hg19" in joined or "build 37" in joined or "build37" in joined:
        return "GRCh37"
    if "grch38" in joined or "hg38" in joined or "build 38" in joined or "build38" in joined:
        return "GRCh38"
    # Specific known metadata keys
    if "genome-version=hg19" in joined or "genome-version-ncbi=37" in joined:
        return "GRCh37"
    if "genome-version=hg38" in joined or "genome-version-ncbi=38" in joined:
        return "GRCh38"
    if "genomebuild,38" in joined or "genomebuild\t38" in joined:
        return "GRCh38"
    if "genomebuild,37" in joined or "genomebuild\t37" in joined:
        return "GRCh37"
    return None


def read_urls(md_path: Path) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for raw in md_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not URL_RE.match(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        urls.append(line)
    return urls


def local_name_for_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    base = Path(parsed.path).name or "download.bin"
    if "." not in base:
        base = f"{base}.bin"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{base}"


def is_valid_cached_download(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False

    name = path.name.lower()
    try:
        if name.endswith(".zip"):
            if not zipfile.is_zipfile(path):
                return False
            with zipfile.ZipFile(path) as zf:
                if not zf.namelist():
                    return False
                bad_member = zf.testzip()
                return bad_member is None
        if name.endswith(".gz"):
            with gzip.open(path, "rb") as fh:
                fh.read(1)
            return True
        with path.open("rb") as fh:
            return len(fh.read(1)) == 1
    except Exception:
        return False


def download_one(url: str, out_dir: Path, timeout: int = 1800) -> tuple[str, Path, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / local_name_for_url(url)
    if out_path.exists():
        if is_valid_cached_download(out_path):
            return (url, out_path, True)
        out_path.unlink()

    tmp = out_path.with_suffix(out_path.suffix + ".part")
    curl = shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            str(timeout),
            "-A",
            "gnomon-snp-intersection/1.0",
            "-o",
            str(tmp),
            url,
        ]
        subprocess.run(cmd, check=True)
    else:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "gnomon-snp-intersection/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as dst:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    tmp.replace(out_path)
    return (url, out_path, False)


def download_to_path(url: str, out_path: Path, timeout: int = 1800) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    curl = shutil.which("curl")
    if curl:
        cmd = [
            curl,
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            str(timeout),
            "-A",
            "gnomon-snp-intersection/1.0",
            "-o",
            str(tmp),
            url,
        ]
        subprocess.run(cmd, check=True)
    else:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "gnomon-snp-intersection/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as dst:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    tmp.replace(out_path)
    return out_path


def normalize_chrom(chrom: str) -> str | None:
    c = chrom.strip()
    if not c:
        return None
    if c.lower().startswith("chr"):
        c = c[3:]
    c = c.upper()
    if c in {"M", "MTDNA"}:
        c = "MT"
    if c == "23":
        c = "X"
    elif c == "24":
        c = "Y"
    elif c in {"25", "XY"}:
        c = "XY"
    elif c in {"26"}:
        c = "MT"

    if c.isdigit():
        i = int(c)
        if 1 <= i <= 22:
            return str(i)
        return None
    if c in {"X", "Y", "MT", "XY"}:
        return c
    return None


def parse_pos(pos: str) -> int | None:
    p = pos.strip()
    if not p:
        return None
    try:
        val = int(float(p))
    except ValueError:
        return None
    if val <= 0:
        return None
    return val


def make_marker(chrom: str, pos: str) -> str | None:
    c = normalize_chrom(chrom)
    p = parse_pos(pos)
    if c is None or p is None:
        return None
    return f"{c}:{p}"


def write_marker_list(path: Path, markers: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(markers, key=chrom_sort_key)
    with path.open("w", encoding="utf-8") as out:
        out.write("chr:pos\n")
        for marker in ordered:
            out.write(marker)
            out.write("\n")


def marker_to_ucsc(marker: str) -> tuple[str, int]:
    chrom, pos_s = marker.split(":")
    pos = int(pos_s)
    if chrom == "MT":
        ucsc_chrom = "chrM"
    else:
        ucsc_chrom = f"chr{chrom}"
    return ucsc_chrom, pos


def liftover_markers_hg19_to_hg38(markers: set[str], chain_file: Path) -> tuple[set[str], dict[str, int]]:
    try:
        from pyliftover import LiftOver  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "pyliftover is required for hg19->hg38 conversion. "
            "Install with: python3 -m pip install pyliftover"
        ) from exc

    lo = LiftOver(str(chain_file))
    lifted: set[str] = set()
    stats = {
        "input_markers": len(markers),
        "mapped_rows": 0,
        "unmapped_rows": 0,
        "multimap_rows": 0,
        "invalid_target_rows": 0,
    }
    for marker in markers:
        chrom, pos = marker_to_ucsc(marker)
        res = lo.convert_coordinate(chrom, pos - 1)
        if not res:
            stats["unmapped_rows"] += 1
            continue
        stats["mapped_rows"] += 1
        if len(res) > 1:
            stats["multimap_rows"] += 1
        tgt_chrom_raw, tgt_pos0 = res[0][0], int(res[0][1])
        tgt_marker = make_marker(tgt_chrom_raw, str(tgt_pos0 + 1))
        if tgt_marker is None:
            stats["invalid_target_rows"] += 1
            continue
        tgt_chrom, tgt_pos_s = tgt_marker.split(":")
        tgt_pos = int(tgt_pos_s)
        max38 = GRCH38_MAX.get(tgt_chrom)
        if max38 is None or tgt_pos > max38:
            stats["invalid_target_rows"] += 1
            continue
        lifted.add(tgt_marker)

    stats["lifted_unique_markers"] = len(lifted)
    stats["lost_after_dedup_or_filter"] = stats["mapped_rows"] - len(lifted)
    return lifted, stats


def iter_text_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for line in fh:
            yield line.rstrip("\n")


def iter_gzip_lines(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        for line in fh:
            yield line.rstrip("\n")


def choose_zip_member(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")]
    if not names:
        raise RuntimeError("ZIP has no file members")

    def score(name: str) -> tuple[int, int]:
        lower = name.lower()
        s = 0
        if lower.endswith(".txt"):
            s += 6
        if lower.endswith(".tsv"):
            s += 6
        if lower.endswith(".csv"):
            s += 5
        if lower.endswith(".strand"):
            s += 4
        for token, pts in (
            ("genome_", 6),
            ("ancestry", 6),
            ("23andme", 6),
            ("manifest", 4),
            ("annot", 4),
            ("strand", 3),
        ):
            if token in lower:
                s += pts
        return (s, -len(name))

    return max(names, key=score)


def iter_zip_member_lines(path: Path) -> tuple[str, Iterator[str]]:
    zf = zipfile.ZipFile(path)
    member = choose_zip_member(zf)

    def _iter() -> Iterator[str]:
        with zf.open(member, "r") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as txt:
                for line in txt:
                    yield line.rstrip("\n")
        zf.close()

    return member, _iter()


def sample_text_lines(path: Path, max_lines: int = 200) -> list[str]:
    out: list[str] = []
    for line in iter_text_lines(path):
        out.append(line)
        if len(out) >= max_lines:
            break
    return out


def sample_gzip_lines(path: Path, max_lines: int = 200) -> list[str]:
    out: list[str] = []
    for line in iter_gzip_lines(path):
        out.append(line)
        if len(out) >= max_lines:
            break
    return out


def sample_zip_member_lines(path: Path, max_lines: int = 200) -> tuple[str, list[str]]:
    with zipfile.ZipFile(path) as zf:
        member = choose_zip_member(zf)
        out: list[str] = []
        with zf.open(member, "r") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as txt:
                for line in txt:
                    out.append(line.rstrip("\n"))
                    if len(out) >= max_lines:
                        break
    return member, out


def sniff_delim(line: str) -> str:
    if line.count("\t") >= line.count(","):
        return "\t"
    return ","


def parse_illumina_assay_section(lines: Iterable[str]) -> set[str]:
    markers: set[str] = set()
    in_assay = False
    header: list[str] | None = None
    chr_idx = None
    pos_idx = None
    for line in lines:
        s = line.strip("\r")
        if not s:
            continue
        if not in_assay:
            if s.strip() == "[Assay]":
                in_assay = True
            continue
        row = next(csv.reader([s]))
        if header is None:
            header = [h.strip() for h in row]
            lowered = {h.lower(): i for i, h in enumerate(header)}
            chr_idx = lowered.get("chr")
            pos_idx = lowered.get("mapinfo")
            continue
        if chr_idx is None or pos_idx is None:
            continue
        if len(row) <= max(chr_idx, pos_idx):
            continue
        marker = make_marker(row[chr_idx], row[pos_idx])
        if marker:
            markers.add(marker)
    return markers


def parse_generic_csv_headered(lines: Iterable[str], delim: str) -> set[str]:
    markers: set[str] = set()
    def clean_lines() -> Iterator[str]:
        for line in lines:
            s = line.strip("\r")
            if not s:
                continue
            if s.startswith("#"):
                continue
            yield s

    reader = csv.reader(clean_lines(), delimiter=delim)
    header = None
    chr_idx = None
    pos_idx = None
    for row in reader:
        if not row:
            continue
        if header is None:
            header = [c.strip().lower() for c in row]
            for i, col in enumerate(header):
                if col in {"chr", "chrom", "chromosome"}:
                    chr_idx = i
                if col in {"mapinfo", "position", "pos", "physical position", "physical_position"}:
                    pos_idx = i
            continue
        if chr_idx is None or pos_idx is None:
            continue
        if len(row) <= max(chr_idx, pos_idx):
            continue
        marker = make_marker(row[chr_idx], row[pos_idx])
        if marker:
            markers.add(marker)
    return markers


def parse_text_schema(lines: Iterable[str], source_name: str) -> set[str]:
    markers: set[str] = set()
    it = iter(lines)
    sample: list[str] = []
    for _ in range(200):
        try:
            sample.append(next(it))
        except StopIteration:
            break

    nonempty_sample = [ln.strip("\r") for ln in sample if ln.strip()]
    if not nonempty_sample:
        return markers

    def full_iter() -> Iterator[str]:
        return itertools.chain(sample, it)

    first = nonempty_sample[0]
    schema = None

    if any(ln.strip() == "[Assay]" for ln in nonempty_sample[:60]):
        schema = "illumina_assay"
    else:
        for ln in nonempty_sample[:30]:
            if ln.startswith("#CHROM") and "POS" in ln and "RSID" in ln:
                schema = "rawdna2vcf"
                break
    if schema is None:
        for ln in nonempty_sample[:80]:
            if ln.startswith("#"):
                continue
            low = ln.lower().replace(" ", "")
            if low.startswith("rsid\tchromosome\tposition\tallele1\tallele2"):
                schema = "ancestry5"
                break
    if schema is None and "\t" in first and first.lower().startswith("name\tchr\tposition"):
        schema = "name_chr_pos"
    if schema is None:
        first_data = next((ln for ln in nonempty_sample if not ln.startswith("#")), "")
        if first_data:
            parts0 = first_data.split("\t")
            if len(parts0) >= 4 and parse_pos(parts0[2] if len(parts0) > 2 else "") is not None:
                if parts0[0].lower() != "rsid":
                    schema = "dtc4"
            if (
                schema is None
                and len(parts0) >= 4
                and parts0[1].lower().startswith("chr")
                and parse_pos(parts0[2]) is not None
                and parse_pos(parts0[3]) is not None
            ):
                schema = "ucsc"
    if schema is None and "," in first:
        schema = "generic_csv"

    if schema == "illumina_assay":
        return parse_illumina_assay_section(full_iter())

    if schema == "rawdna2vcf":
        for row in full_iter():
            s = row.strip("\r")
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            if len(parts) < 2:
                continue
            marker = make_marker(parts[0], parts[1])
            if marker:
                markers.add(marker)
        return markers

    if schema == "ancestry5":
        saw_header = False
        for row in full_iter():
            s = row.strip("\r")
            if not s or s.startswith("#"):
                continue
            low = s.lower().replace(" ", "")
            if not saw_header and low.startswith("rsid\tchromosome\tposition\tallele1\tallele2"):
                saw_header = True
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            marker = make_marker(parts[1], parts[2])
            if marker:
                markers.add(marker)
        return markers

    if schema == "name_chr_pos":
        first_line = True
        for row in full_iter():
            s = row.strip("\r")
            if not s:
                continue
            if first_line:
                first_line = False
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            marker = make_marker(parts[1], parts[2])
            if marker:
                markers.add(marker)
        return markers

    if schema == "dtc4":
        for row in full_iter():
            s = row.strip("\r")
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            if len(parts) < 3:
                continue
            marker = make_marker(parts[1], parts[2])
            if marker:
                markers.add(marker)
        return markers

    if schema == "ucsc":
        for row in full_iter():
            s = row.strip("\r")
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            if len(parts) < 4:
                continue
            marker = make_marker(parts[1], parts[3])
            if marker:
                markers.add(marker)
        return markers

    if schema == "generic_csv":
        return parse_generic_csv_headered((ln.strip("\r") for ln in full_iter()), delim=",")

    raise RuntimeError(f"Could not identify parse schema for {source_name}")


def parse_source(path: Path, url: str) -> tuple[set[str], str]:
    name = path.name.lower()
    if name.endswith(".zip"):
        member, lines = iter_zip_member_lines(path)
        markers = parse_text_schema(lines, f"{url}::{member}")
        return markers, member
    if name.endswith(".gz"):
        markers = parse_text_schema(iter_gzip_lines(path), url)
        return markers, "(gzip text)"
    markers = parse_text_schema(iter_text_lines(path), url)
    return markers, "(text)"


def detect_declared_build(path: Path, url: str) -> dict[str, object]:
    name = path.name.lower()
    member = None
    text_lines: list[str] = []
    if name.endswith(".zip"):
        member, text_lines = sample_zip_member_lines(path)
    elif name.endswith(".gz"):
        text_lines = sample_gzip_lines(path)
    else:
        text_lines = sample_text_lines(path)

    by_url = detect_build_from_url(url, member)
    by_name_hint = detect_build_hints_from_name(url=url, member_name=member)
    by_text = detect_build_from_text(text_lines)
    declared = by_text or by_url
    return {
        "declared_build": declared,
        "declared_from": "text" if by_text else ("url" if by_url else None),
        "member_name": member,
        "url_name_hint_build": by_name_hint["suggested_build"],  # type: ignore[index]
        "url_name_hint_hits_grch37": by_name_hint["hits_grch37"],  # type: ignore[index]
        "url_name_hint_hits_grch38": by_name_hint["hits_grch38"],  # type: ignore[index]
    }


def infer_build_from_sources(
    build_check: dict[str, object],
    declared_build: str | None,
    url_name_hint_build: str | None,
) -> dict[str, object]:
    votes: dict[str, list[str]] = {"GRCh37": [], "GRCh38": []}

    if declared_build in votes:
        votes[declared_build].append("declared")
    if url_name_hint_build in votes:
        votes[url_name_hint_build].append("url_or_filename")

    suggested = build_check.get("suggested_build")
    if suggested in votes:
        votes[str(suggested)].append("max_pos_count")

    positional = build_check.get("positional_suggested_build")
    if positional in votes:
        votes[str(positional)].append("chrmax_window")

    support37 = len(votes["GRCh37"])
    support38 = len(votes["GRCh38"])
    if support37 > 0 and support38 == 0:
        final = "GRCh37"
    elif support38 > 0 and support37 == 0:
        final = "GRCh38"
    elif support37 > 0 and support38 > 0:
        final = "conflict"
    else:
        final = "ambiguous"

    return {"final_build": final, "evidence": votes}


def chrom_sort_key(marker: str) -> tuple[int, int]:
    chrom, pos_s = marker.split(":")
    pos = int(pos_s)
    if chrom.isdigit():
        return (int(chrom), pos)
    order = {"X": 23, "Y": 24, "XY": 25, "MT": 26}
    return (order.get(chrom, 99), pos)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--download-dir",
        default="data/snp_intersection_downloads",
        help="Download/cache directory.",
    )
    ap.add_argument(
        "--output",
        default="data/snp_intersection_final_hg38_chrpos.txt",
        help="Output file for final hg38 intersected chr:pos markers.",
    )
    ap.add_argument(
        "--hg19-output",
        default="data/snp_intersection_hg19_chrpos.txt",
        help="Output file for native hg19 intersection (before liftover).",
    )
    ap.add_argument(
        "--hg38-output",
        default="data/snp_intersection_hg38_native_chrpos.txt",
        help="Output file for native hg38 intersection.",
    )
    ap.add_argument(
        "--hg19-lifted-output",
        default="data/snp_intersection_hg19_lifted_to_hg38_chrpos.txt",
        help="Output file for hg19 intersection after liftover to hg38.",
    )
    ap.add_argument(
        "--chain-file",
        default="data/hg19ToHg38.over.chain.gz",
        help="Path to hg19->hg38 chain file (downloaded if missing).",
    )
    ap.add_argument(
        "--summary-json",
        default="data/snp_intersection_summary.json",
        help="Summary JSON report path.",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=6,
        help="Parallel download workers.",
    )
    ap.add_argument(
        "--fail-on-build-mismatch",
        action="store_true",
        help="Exit non-zero if any source has build conflicts or mismatches.",
    )
    args = ap.parse_args()

    download_dir = Path(args.download_dir)
    output_path = Path(args.output)
    hg19_output_path = Path(args.hg19_output)
    hg38_output_path = Path(args.hg38_output)
    hg19_lifted_output_path = Path(args.hg19_lifted_output)
    summary_path = Path(args.summary_json)
    chain_path = Path(args.chain_file)

    urls = list(dict.fromkeys(SOURCE_URLS))
    if not urls:
        print("No URLs configured in SOURCE_URLS", file=sys.stderr)
        return 1

    print(f"Using {len(urls)} hardcoded source URLs", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hg19_output_path.parent.mkdir(parents=True, exist_ok=True)
    hg38_output_path.parent.mkdir(parents=True, exist_ok=True)
    hg19_lifted_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        futures = {ex.submit(download_one, url, download_dir): url for url in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                _, out_path, from_cache = fut.result()
                downloaded[url] = out_path
                tag = "cache-hit" if from_cache else "downloaded"
                print(f"[{tag}] {url} -> {out_path.name}", file=sys.stderr)
            except Exception as exc:  # pragma: no cover
                print(f"[download-failed] {url}: {exc}", file=sys.stderr)
                return 2

    source_stats: list[dict[str, object]] = []
    hg19_intersection: set[str] | None = None
    hg38_intersection: set[str] | None = None
    hg19_steps: list[dict[str, object]] = []
    hg38_steps: list[dict[str, object]] = []
    skipped_sources: list[dict[str, object]] = []

    for idx, url in enumerate(urls, start=1):
        path = downloaded[url]
        try:
            markers, member = parse_source(path, url)
        except Exception as exc:  # pragma: no cover
            print(f"[parse-failed] {url}: {exc}", file=sys.stderr)
            return 3
        build_info = detect_build_indicators(markers)
        build_decl = detect_declared_build(path, url)
        combined = infer_build_from_sources(
            build_check=build_info,
            declared_build=build_decl.get("declared_build"),  # type: ignore[arg-type]
            url_name_hint_build=build_decl.get("url_name_hint_build"),  # type: ignore[arg-type]
        )
        n = len(markers)
        build_bucket = str(combined["final_build"])
        bucket_after = None
        bucket_lost = None
        if build_bucket == "GRCh37":
            if hg19_intersection is None:
                hg19_intersection = set(markers)
                before = len(markers)
                after = len(hg19_intersection)
                lost = 0
            else:
                before = len(hg19_intersection)
                hg19_intersection.intersection_update(markers)
                after = len(hg19_intersection)
                lost = before - after
            bucket_after = after
            bucket_lost = lost
            hg19_steps.append(
                {
                    "index": idx,
                    "url": url,
                    "markers": n,
                    "before": before,
                    "after": after,
                    "lost": lost,
                }
            )
            print(
                f"[intersect-hg19] idx={idx} before={before:,} after={after:,} lost={lost:,}",
                file=sys.stderr,
            )
        elif build_bucket == "GRCh38":
            if hg38_intersection is None:
                hg38_intersection = set(markers)
                before = len(markers)
                after = len(hg38_intersection)
                lost = 0
            else:
                before = len(hg38_intersection)
                hg38_intersection.intersection_update(markers)
                after = len(hg38_intersection)
                lost = before - after
            bucket_after = after
            bucket_lost = lost
            hg38_steps.append(
                {
                    "index": idx,
                    "url": url,
                    "markers": n,
                    "before": before,
                    "after": after,
                    "lost": lost,
                }
            )
            print(
                f"[intersect-hg38] idx={idx} before={before:,} after={after:,} lost={lost:,}",
                file=sys.stderr,
            )
        else:
            skipped_sources.append(
                {"index": idx, "url": url, "final_build": build_bucket, "markers": n}
            )
            print(
                f"[skip-build] idx={idx} final_build={build_bucket} url={url}",
                file=sys.stderr,
            )
        source_stats.append(
            {
                "index": idx,
                "url": url,
                "local_path": str(path),
                "parsed_member": member,
                "markers": n,
                "build_check": build_info,
                "declared_build": build_decl["declared_build"],
                "declared_from": build_decl["declared_from"],
                "declared_member": build_decl["member_name"],
                "url_name_hint_build": build_decl["url_name_hint_build"],
                "url_name_hint_hits_grch37": build_decl["url_name_hint_hits_grch37"],
                "url_name_hint_hits_grch38": build_decl["url_name_hint_hits_grch38"],
                "build_inference": combined,
                "build_bucket": build_bucket,
                "bucket_running_intersection": bucket_after,
                "bucket_step_lost": bucket_lost,
            }
        )
        print(
            f"[parsed {idx}/{len(urls)}] {url} markers={n:,} build={build_bucket}",
            file=sys.stderr,
        )
        if (
            build_info["suggested_build"] != "ambiguous"
            or build_info["positional_suggested_build"] != "ambiguous"
            or build_decl["url_name_hint_build"] is not None
        ):
            print(
                f"[build] {url} final={combined['final_build']} "
                f"url_hint={build_decl['url_name_hint_build']} "
                f"chrmax_window={build_info['positional_suggested_build']} "
                f"count_suggested={build_info['suggested_build']} "
                f"too_big_grch37={build_info['too_big_grch37']} "
                f"too_big_grch38={build_info['too_big_grch38']}",
                file=sys.stderr,
            )
        if (
            combined["final_build"] == "conflict"
            or combined["final_build"] == "ambiguous"
        ) and args.fail_on_build_mismatch:
            print(f"[build-conflict] {url}", file=sys.stderr)
            return 4

    hg19_native = hg19_intersection if hg19_intersection is not None else set()
    hg38_native = hg38_intersection if hg38_intersection is not None else set()
    write_marker_list(hg19_output_path, hg19_native)
    write_marker_list(hg38_output_path, hg38_native)
    print(f"[step1] hg38 native intersection: {len(hg38_native):,}", file=sys.stderr)
    print(f"[step2] hg19 native intersection: {len(hg19_native):,}", file=sys.stderr)

    print(
        f"[step3] ensuring chain file for liftover: {chain_path} ({HG19_TO_HG38_CHAIN_URL})",
        file=sys.stderr,
    )
    chain_file = download_to_path(HG19_TO_HG38_CHAIN_URL, chain_path)
    lifted_hg19, liftover_stats = liftover_markers_hg19_to_hg38(hg19_native, chain_file)
    write_marker_list(hg19_lifted_output_path, lifted_hg19)
    print(
        f"[step3] liftover hg19->hg38 input={liftover_stats['input_markers']:,} "
        f"mapped={liftover_stats['mapped_rows']:,} "
        f"unmapped={liftover_stats['unmapped_rows']:,} "
        f"multimap={liftover_stats['multimap_rows']:,} "
        f"invalid_target={liftover_stats['invalid_target_rows']:,} "
        f"lifted_unique={liftover_stats['lifted_unique_markers']:,}",
        file=sys.stderr,
    )

    before_final = min(len(hg38_native), len(lifted_hg19))
    final_intersection = hg38_native.intersection(lifted_hg19)
    final_lost_vs_hg38 = len(hg38_native) - len(final_intersection)
    final_lost_vs_lifted = len(lifted_hg19) - len(final_intersection)
    write_marker_list(output_path, final_intersection)
    print(
        f"[step4] final intersection (hg38_native ∩ lifted_hg19): {len(final_intersection):,} "
        f"lost_vs_hg38={final_lost_vs_hg38:,} lost_vs_lifted={final_lost_vs_lifted:,}",
        file=sys.stderr,
    )
    print(
        f"[step5] check note: this is the hg38-normalized final list built via "
        f"intersection-then-liftover path (reference size {before_final:,}).",
        file=sys.stderr,
    )

    summary = {
        "num_sources": len(urls),
        "hg19_source_count": len(hg19_steps),
        "hg38_source_count": len(hg38_steps),
        "skipped_source_count": len(skipped_sources),
        "final_intersection_count": len(final_intersection),
        "hg19_intersection_count": len(hg19_native),
        "hg38_native_intersection_count": len(hg38_native),
        "hg19_lifted_intersection_count": len(lifted_hg19),
        "output_file": str(output_path),
        "hg19_output_file": str(hg19_output_path),
        "hg38_output_file": str(hg38_output_path),
        "hg19_lifted_output_file": str(hg19_lifted_output_path),
        "liftover_chain_file": str(chain_file),
        "liftover_stats": liftover_stats,
        "hg19_intersection_steps": hg19_steps,
        "hg38_intersection_steps": hg38_steps,
        "skipped_sources": skipped_sources,
        "final_intersection_loss": {
            "lost_vs_hg38_native": final_lost_vs_hg38,
            "lost_vs_hg19_lifted": final_lost_vs_lifted,
        },
        "sources": source_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] final intersection: {len(final_intersection):,}", file=sys.stderr)
    print(f"[done] wrote output: {output_path}", file=sys.stderr)
    print(f"[done] wrote summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
