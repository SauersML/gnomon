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

    return {
        "suggested_build": suggested,
        "too_big_grch37": too_big_37,
        "too_big_grch38": too_big_38,
        "max_by_chr": max_by_chr,
    }


def detect_build_from_url(url: str, member_name: str | None = None) -> str | None:
    hay = " ".join([url, member_name or ""]).lower()
    if "hg19" in hay or "grch37" in hay or "build37" in hay or "build_37" in hay:
        return "GRCh37"
    if "hg38" in hay or "grch38" in hay or "build38" in hay or "build_38" in hay:
        return "GRCh38"
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


def download_one(url: str, out_dir: Path, timeout: int = 1800) -> tuple[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / local_name_for_url(url)
    if out_path.exists() and out_path.stat().st_size > 0:
        return (url, out_path)

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
    return (url, out_path)


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


def detect_declared_build(path: Path, url: str) -> dict[str, str | None]:
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
    by_text = detect_build_from_text(text_lines)
    declared = by_text or by_url
    return {
        "declared_build": declared,
        "declared_from": "text" if by_text else ("url" if by_url else None),
        "member_name": member,
    }


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
        "--source-md",
        default="examples/snp_list_subset.md",
        help="Markdown file with URL lines.",
    )
    ap.add_argument(
        "--download-dir",
        default="data/snp_intersection_downloads",
        help="Download/cache directory.",
    )
    ap.add_argument(
        "--output",
        default="data/snp_intersection_chrpos.txt",
        help="Output file for intersected chr:pos markers.",
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

    md_path = Path(args.source_md)
    download_dir = Path(args.download_dir)
    output_path = Path(args.output)
    summary_path = Path(args.summary_json)

    urls = read_urls(md_path)
    if not urls:
        print(f"No URLs found in {md_path}", file=sys.stderr)
        return 1

    print(f"Found {len(urls)} unique URLs in {md_path}", file=sys.stderr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
        futures = {ex.submit(download_one, url, download_dir): url for url in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            try:
                _, out_path = fut.result()
                downloaded[url] = out_path
                print(f"[downloaded] {url} -> {out_path.name}", file=sys.stderr)
            except Exception as exc:  # pragma: no cover
                print(f"[download-failed] {url}: {exc}", file=sys.stderr)
                return 2

    intersection: set[str] | None = None
    source_stats: list[dict[str, object]] = []
    for idx, url in enumerate(urls, start=1):
        path = downloaded[url]
        try:
            markers, member = parse_source(path, url)
        except Exception as exc:  # pragma: no cover
            print(f"[parse-failed] {url}: {exc}", file=sys.stderr)
            return 3
        build_info = detect_build_indicators(markers)
        build_decl = detect_declared_build(path, url)
        n = len(markers)
        if intersection is None:
            intersection = set(markers)
        else:
            intersection.intersection_update(markers)
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
                "running_intersection": len(intersection),
            }
        )
        print(
            f"[parsed {idx}/{len(urls)}] {url} markers={n:,} running_intersection={len(intersection):,}",
            file=sys.stderr,
        )
        if build_info["suggested_build"] != "ambiguous":
            print(
                f"[build] {url} suggested={build_info['suggested_build']} "
                f"too_big_grch37={build_info['too_big_grch37']} "
                f"too_big_grch38={build_info['too_big_grch38']}",
                file=sys.stderr,
            )
        if build_info["suggested_build"] == "conflict" and args.fail_on_build_mismatch:
            print(f"[build-conflict] {url}", file=sys.stderr)
            return 4

    assert intersection is not None
    ordered = sorted(intersection, key=chrom_sort_key)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("chr:pos\n")
        for marker in ordered:
            out.write(marker)
            out.write("\n")

    summary = {
        "source_markdown": str(md_path),
        "num_sources": len(urls),
        "final_intersection_count": len(ordered),
        "output_file": str(output_path),
        "sources": source_stats,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] final intersection: {len(ordered):,}", file=sys.stderr)
    print(f"[done] wrote output: {output_path}", file=sys.stderr)
    print(f"[done] wrote summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
