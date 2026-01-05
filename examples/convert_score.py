#!/usr/bin/env python3
"""Score direct-to-consumer genotype exports using gnomon with Beagle imputation.

Pipeline:
1. Convert DTC text → PLINK (via gnomon)
2. Convert PLINK → VCF
3. Run Beagle imputation with 1000 Genomes reference panel
4. Concatenate imputed chromosomes
5. Score imputed VCF with gnomon

Required dependencies:
- gnomon (auto-installed via install.sh or built from source)
- Java 8+ (for Beagle)
- plink2 (conda install -c bioconda plink2)
- bcftools (conda install -c bioconda bcftools)

All Beagle dependencies (JAR, genetic maps, reference panels) are auto-downloaded.

Usage:
    python examples/convert_score.py
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "convert_score_output"
CACHE_DIR = Path.home() / ".gnomon"

GNOMON_BIN = shutil.which("gnomon") or REPO_ROOT / "target" / "release" / "gnomon"

PGS_IDS = ("PGS000007", "PGS000317", "PGS004869", "PGS000507")
PGS_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"
ASSEMBLY = "GRCh37"

GENOME_FILES = (
    ("genome_Joshua_Yoakem_v5_Full_20250129211749.txt", "Joshua_Yoakem"),
    ("autosomal.txt", "LivingDNA_Autosomal"),
)

# Beagle configuration
BEAGLE_JAR_URL = "https://faculty.washington.edu/browning/beagle/beagle.27Feb25.75f.jar"
CONFORM_GT_URL = "https://faculty.washington.edu/browning/conform-gt/conform-gt.24May16.cee.jar"
GENETIC_MAP_URL = "https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.GRCh37.map.zip"

# 1000 Genomes reference panel (VCF format for conform-gt compatibility)
REF_PANEL_BASE = "https://bochet.gcc.biostat.washington.edu/beagle/1000_Genomes_phase3_v5a/b37.vcf"
CHROMOSOMES = [str(i) for i in range(1, 23)]  # chr1-22

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


@dataclass
class ScoreResult:
    pgs_id: str
    average: float
    missing_pct: float


def debug(msg: str) -> None:
    print(f"[convert_score] {msg}", flush=True)


def run(cmd: list) -> subprocess.CompletedProcess:
    debug(f"$ {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    return result


def download(url: str, dest: Path) -> None:
    if dest.exists():
        debug(f"Using cached {dest.name}")
        return
    debug(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Download failed: {e}") from e


def get_platform() -> tuple[str, str]:
    """Get platform identifier for binary downloads."""
    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        if machine == "arm64":
            return "mac_arm64", "darwin-arm64"
        return "mac", "darwin-x64"
    elif system == "linux":
        if machine == "aarch64":
            return "linux_aarch64", "linux-arm64"
        return "linux_x86_64", "linux-x64"
    else:
        raise RuntimeError(f"Unsupported platform: {system} {machine}")


def ensure_plink2() -> Path:
    """Download plink2 if not found."""
    # Check if already in PATH
    existing = shutil.which("plink2")
    if existing:
        return Path(existing)

    # Check our tools dir
    tools_dir = CACHE_DIR / "tools"
    plink2_path = tools_dir / "plink2"
    if plink2_path.exists():
        return plink2_path

    plat, _ = get_platform()
    # plink2 alpha builds from S3
    url = f"https://s3.amazonaws.com/plink2-assets/alpha6/plink2_{plat}_20250122.zip"

    debug(f"Downloading plink2 from {url}")
    zip_path = tools_dir / "plink2.zip"
    download(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tools_dir)
    zip_path.unlink()

    plink2_path.chmod(0o755)
    return plink2_path


def ensure_bcftools() -> Path:
    """Download bcftools if not found."""
    # Check if already in PATH
    existing = shutil.which("bcftools")
    if existing:
        return Path(existing)

    # Check our tools dir
    tools_dir = CACHE_DIR / "tools"
    bcftools_path = tools_dir / "bcftools"
    if bcftools_path.exists():
        return bcftools_path

    _, plat = get_platform()
    # Pre-built bcftools from conda-forge (static builds)
    url = f"https://github.com/samtools/bcftools/releases/download/1.21/bcftools-1.21_{plat}.tar.bz2"

    debug(f"Downloading bcftools from {url}")
    import tarfile
    tar_path = tools_dir / "bcftools.tar.bz2"
    tools_dir.mkdir(parents=True, exist_ok=True)

    try:
        download(url, tar_path)
        with tarfile.open(tar_path, "r:bz2") as tf:
            # Extract just the bcftools binary
            for member in tf.getmembers():
                if member.name.endswith("/bcftools") or member.name == "bcftools":
                    member.name = "bcftools"
                    tf.extract(member, tools_dir)
                    break
        tar_path.unlink()
        bcftools_path.chmod(0o755)
        return bcftools_path
    except Exception:
        # Fallback: try conda-forge static build URL
        tar_path.unlink(missing_ok=True)
        raise RuntimeError(
            "bcftools download failed. Install manually:\n"
            "  macOS: brew install bcftools\n"
            "  Ubuntu: sudo apt install bcftools\n"
            "  conda: conda install -c bioconda bcftools"
        )


def ensure_java() -> Path:
    """Check Java is available."""
    java = shutil.which("java")
    if java:
        # Verify it actually works (macOS stub returns 0 but doesn't work)
        result = subprocess.run([java, "-version"], capture_output=True)
        if result.returncode == 0:
            return Path(java)

    # Check homebrew location on macOS
    homebrew_java = Path("/opt/homebrew/opt/openjdk/bin/java")
    if homebrew_java.exists():
        return homebrew_java

    raise RuntimeError(
        "Java not found. Install Java 8+ for Beagle:\n"
        "  macOS: brew install openjdk\n"
        "  Ubuntu: sudo apt install default-jdk"
    )


def check_dependencies() -> dict[str, Path]:
    """Check/install all required dependencies."""
    deps = {}

    # gnomon
    gnomon = Path(GNOMON_BIN) if isinstance(GNOMON_BIN, str) else GNOMON_BIN
    if not gnomon.exists():
        raise RuntimeError(
            "gnomon not found. Install with:\n"
            "  curl -fsSL https://raw.githubusercontent.com/SauersML/gnomon/main/install.sh | bash"
        )
    deps["gnomon"] = gnomon

    # Java (for Beagle) - must be installed by user
    deps["java"] = ensure_java()
    debug(f"Java: {deps['java']}")

    # plink2 - auto-download if needed
    deps["plink2"] = ensure_plink2()
    debug(f"plink2: {deps['plink2']}")

    # bcftools - auto-download if needed
    deps["bcftools"] = ensure_bcftools()
    debug(f"bcftools: {deps['bcftools']}")

    debug("All dependencies ready.")
    return deps


# --------------------------------------------------------------------------------------
# Setup: Download Beagle, genetic maps, reference panels
# --------------------------------------------------------------------------------------


def ensure_beagle() -> Path:
    """Download Beagle JAR if not cached."""
    tools_dir = CACHE_DIR / "tools"
    jar_path = tools_dir / "beagle.jar"
    download(BEAGLE_JAR_URL, jar_path)
    return jar_path


def ensure_conform_gt() -> Path:
    """Download conform-gt JAR if not cached."""
    tools_dir = CACHE_DIR / "tools"
    jar_path = tools_dir / "conform-gt.jar"
    download(CONFORM_GT_URL, jar_path)
    return jar_path


def ensure_genetic_maps() -> Path:
    """Download and extract genetic maps if not cached."""
    maps_dir = CACHE_DIR / "maps" / "GRCh37"
    marker_file = maps_dir / "plink.chr1.GRCh37.map"

    if marker_file.exists():
        debug("Using cached genetic maps")
        return maps_dir

    zip_path = CACHE_DIR / "maps" / "plink.GRCh37.map.zip"
    download(GENETIC_MAP_URL, zip_path)

    debug("Extracting genetic maps...")
    maps_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(maps_dir)

    return maps_dir


def ensure_reference_panel(chrom: str, bcftools: Path) -> Path:
    """Download reference panel for a chromosome and filter to SNPs only."""
    ref_dir = CACHE_DIR / "refs" / "1kg_b37_vcf"
    raw_file = ref_dir / f"chr{chrom}.1kg.phase3.v5a.raw.vcf.gz"
    filtered_file = ref_dir / f"chr{chrom}.1kg.phase3.v5a.snps.vcf.gz"

    if filtered_file.exists():
        return filtered_file

    # Download raw file
    if not raw_file.exists():
        url = f"{REF_PANEL_BASE}/chr{chrom}.1kg.phase3.v5a.vcf.gz"
        download(url, raw_file)

    # Filter to SNPs only (remove structural variants)
    debug(f"Filtering chr{chrom} reference to SNPs only...")
    result = subprocess.run([
        bcftools, "view",
        "-v", "snps",  # SNPs only
        "-O", "z",
        "-o", filtered_file,
        raw_file,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"bcftools filter failed: {result.stderr}")

    # Index the filtered file
    subprocess.run([bcftools, "index", "-t", filtered_file], capture_output=True)

    return filtered_file


# --------------------------------------------------------------------------------------
# Conversion and imputation
# --------------------------------------------------------------------------------------


def convert_dtc_to_plink(genome_path: Path, gnomon: Path, work_dir: Path) -> Path:
    """Convert DTC text file to PLINK using gnomon."""
    cache_dir = genome_path.parent / f"{genome_path.stem}.gnomon_cache"
    plink_prefix = cache_dir / "genotypes"

    if plink_prefix.with_suffix(".bed").exists():
        debug(f"Using cached PLINK files for {genome_path.name}")
        return plink_prefix

    # Run gnomon with a dummy score to trigger conversion
    dummy_score = work_dir / "dummy.txt"
    dummy_score.write_text(
        "rsID\tchr_name\tchr_position\teffect_allele\teffect_weight\n"
        "rs1\t1\t1000\tA\t0.0\n"
    )

    # gnomon will convert and cache PLINK files
    subprocess.run(
        [gnomon, "score", dummy_score, genome_path, "--build", "37"],
        capture_output=True,
    )

    if not plink_prefix.with_suffix(".bed").exists():
        raise RuntimeError(f"PLINK conversion failed for {genome_path}")

    return plink_prefix


def plink_to_vcf(plink_prefix: Path, vcf_path: Path, plink2: Path) -> None:
    """Convert PLINK to VCF."""
    if vcf_path.exists():
        debug(f"Using cached {vcf_path.name}")
        return

    run([
        plink2,
        "--bfile", plink_prefix,
        "--recode", "vcf",
        "--out", vcf_path.with_suffix(""),
    ])


def run_conform_gt(
    vcf_in: Path,
    vcf_out: Path,
    ref_panel: Path,
    conform_gt_jar: Path,
    chrom: str,
    java: Path,
) -> None:
    """Run conform-gt to align target alleles with reference panel."""
    if vcf_out.exists():
        debug(f"Using cached {vcf_out.name}")
        return

    out_prefix = vcf_out.with_suffix("").with_suffix("")  # Remove .vcf.gz

    run([
        java, "-Xmx2g", "-jar", conform_gt_jar,
        f"gt={vcf_in}",
        f"ref={ref_panel}",
        f"chrom={chrom}",
        f"out={out_prefix}",
    ])


def run_beagle_imputation(
    vcf_in: Path,
    vcf_out: Path,
    beagle_jar: Path,
    conform_gt_jar: Path,
    maps_dir: Path,
    chrom: str,
    java: Path,
    bcftools: Path,
    work_dir: Path,
) -> None:
    """Run conform-gt then Beagle imputation for one chromosome."""
    if vcf_out.exists():
        debug(f"Using cached {vcf_out.name}")
        return

    ref_panel = ensure_reference_panel(chrom, bcftools)
    map_file = maps_dir / f"plink.chr{chrom}.GRCh37.map"

    # Step 1: Run conform-gt to align alleles
    conformed_vcf = work_dir / f"conformed.chr{chrom}.vcf.gz"
    run_conform_gt(vcf_in, conformed_vcf, ref_panel, conform_gt_jar, chrom, java)

    # Step 2: Run Beagle imputation on conformed data
    out_prefix = vcf_out.with_suffix("").with_suffix("")  # Remove .vcf.gz

    run([
        java, "-Xmx4g", "-jar", beagle_jar,
        f"gt={conformed_vcf}",
        f"ref={ref_panel}",
        f"map={map_file}",
        f"out={out_prefix}",
        "impute=true",
    ])


def concatenate_vcfs(vcf_files: list[Path], output_vcf: Path, bcftools: Path) -> None:
    """Concatenate multiple VCF files into one."""
    if output_vcf.exists():
        debug(f"Using cached {output_vcf.name}")
        return

    # Sort by chromosome number
    sorted_vcfs = sorted(vcf_files, key=lambda p: int(p.name.split(".chr")[1].split(".")[0]))

    run([
        bcftools, "concat",
        "-O", "z",  # Output gzipped VCF
        "-o", output_vcf,
        *sorted_vcfs,
    ])

    # Index the output
    run([bcftools, "index", "-t", output_vcf])


def impute_genome(
    genome_path: Path,
    sample_id: str,
    deps: dict[str, Path],
    beagle_jar: Path,
    conform_gt_jar: Path,
    maps_dir: Path,
    work_dir: Path,
) -> Path:
    """Full imputation pipeline: DTC → PLINK → VCF → conform-gt → Beagle → merged VCF."""
    debug(f"Processing {genome_path.name}...")

    # Step 1: DTC → PLINK
    debug("Step 1: Converting DTC to PLINK...")
    plink_prefix = convert_dtc_to_plink(genome_path, deps["gnomon"], work_dir)

    # Step 2: PLINK → VCF
    debug("Step 2: Converting PLINK to VCF...")
    vcf_path = work_dir / f"{sample_id}.vcf"
    plink_to_vcf(plink_prefix, vcf_path, deps["plink2"])

    # Step 3: Impute each chromosome (includes conform-gt step)
    debug("Step 3: Running conform-gt + Beagle imputation per chromosome...")
    imputed_vcfs = []

    sample_work_dir = work_dir / sample_id
    sample_work_dir.mkdir(exist_ok=True)

    for chrom in CHROMOSOMES:
        debug(f"  Chromosome {chrom}...")
        chrom_vcf = sample_work_dir / f"chr{chrom}.imputed.vcf.gz"
        run_beagle_imputation(
            vcf_in=vcf_path,
            vcf_out=chrom_vcf,
            beagle_jar=beagle_jar,
            conform_gt_jar=conform_gt_jar,
            maps_dir=maps_dir,
            chrom=chrom,
            java=deps["java"],
            bcftools=deps["bcftools"],
            work_dir=sample_work_dir,
        )
        imputed_vcfs.append(chrom_vcf)

    # Step 4: Concatenate all chromosomes
    debug("Step 4: Concatenating imputed chromosomes...")
    merged_vcf = work_dir / f"{sample_id}.imputed.vcf.gz"
    concatenate_vcfs(imputed_vcfs, merged_vcf, deps["bcftools"])

    debug(f"Imputation complete: {merged_vcf}")
    return merged_vcf


# --------------------------------------------------------------------------------------
# Score helpers
# --------------------------------------------------------------------------------------


def get_pgs_score(pgs_id: str, cache_dir: Path) -> Path:
    """Download PGS score file if not cached."""
    filename = f"{pgs_id}_hmPOS_{ASSEMBLY}.txt"
    target = cache_dir / filename

    if target.exists():
        debug(f"Using cached {target.name}")
        return target

    gz_file = cache_dir / f"{filename}.gz"
    url = f"{PGS_BASE_URL}/{pgs_id}/ScoringFiles/Harmonized/{filename}.gz"
    download(url, gz_file)

    debug(f"Decompressing {gz_file.name}")
    with gzip.open(gz_file, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_file.unlink()

    return target


def parse_sscore(path: Path) -> list[ScoreResult]:
    """Parse gnomon .sscore file."""
    results = []
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#REGION"):
                continue
            if header is None:
                header = line.strip().lstrip("#").split("\t")
                continue
            values = line.strip().split("\t")
            i = 1
            while i < len(values):
                if header[i].endswith("_AVG"):
                    results.append(ScoreResult(
                        pgs_id=header[i].replace("_AVG", ""),
                        average=float(values[i]),
                        missing_pct=float(values[i + 1]),
                    ))
                    i += 2
                else:
                    i += 1
    return results


def score_vcf(vcf_path: Path, score_file: Path, gnomon: Path) -> Path:
    """Score a VCF file with gnomon and return the sscore path."""
    run([gnomon, "score", score_file, vcf_path, "--build", "37"])

    # gnomon outputs sscore next to input file
    sscore = vcf_path.parent / f"{vcf_path.stem}_{score_file.stem}.sscore"

    # Handle .vcf.gz → remove both suffixes
    if vcf_path.suffix == ".gz":
        stem = vcf_path.stem  # foo.imputed.vcf
        stem = Path(stem).stem  # foo.imputed
        sscore = vcf_path.parent / f"{stem}_{score_file.stem}.sscore"

    if not sscore.exists():
        raise RuntimeError(f"Score output not found: {sscore}")

    return sscore


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    # Check dependencies upfront
    debug("Checking dependencies...")
    deps = check_dependencies()

    # Setup Beagle resources
    debug("Setting up Beagle imputation pipeline...")
    beagle_jar = ensure_beagle()
    conform_gt_jar = ensure_conform_gt()
    maps_dir = ensure_genetic_maps()

    # Prepare directories
    score_cache = OUTPUT_DIR / "scores"
    score_cache.mkdir(parents=True, exist_ok=True)
    work_dir = OUTPUT_DIR / "imputation"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Find genome files
    genomes = [(DATA_DIR / f, name) for f, name in GENOME_FILES if (DATA_DIR / f).exists()]
    if len(genomes) < 2:
        raise RuntimeError(f"Need at least 2 genome files in {DATA_DIR}")

    all_results: dict[str, dict[str, ScoreResult]] = defaultdict(dict)

    for genome_path, sample_id in genomes:
        debug(f"\n{'='*60}")
        debug(f"Processing {sample_id}")
        debug(f"{'='*60}")

        # Impute the genome
        imputed_vcf = impute_genome(
            genome_path, sample_id, deps, beagle_jar, conform_gt_jar, maps_dir, work_dir
        )

        # Score with each PGS
        for pgs_id in PGS_IDS:
            debug(f"Scoring {pgs_id}...")
            score_file = get_pgs_score(pgs_id, score_cache)
            sscore_path = score_vcf(imputed_vcf, score_file, deps["gnomon"])

            for result in parse_sscore(sscore_path):
                all_results[sample_id][result.pgs_id] = result

    # Print summary
    print("\n" + "=" * 60)
    print("SCORE SUMMARY (with imputation)")
    print("=" * 60)
    for sample, scores in all_results.items():
        print(f"\n{sample}:")
        for pgs_id, r in sorted(scores.items()):
            print(f"  {pgs_id}: avg={r.average:.6f}, missing={r.missing_pct:.1f}%")

    # Verify all scores present
    for sample, scores in all_results.items():
        missing = set(PGS_IDS) - set(scores)
        if missing:
            raise AssertionError(f"{sample} missing scores: {missing}")

    debug("\nAll assertions passed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
