In order to have a widely usable set of SNVs for PCA, we should intersect multiple files. Since PCA is not affected much by thinning, we don't have to worry about deleting SNVs.

- Sites with low missingness in the All of Us cohort's microarray data
- Sites present in GDA, all GSA versions
- Sites present in common DTC microarray chips

First, let's get sites with low missingness (99.99%+ call rate) in All of Us:
```
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display


def _vmiss_path_for_outprefix(out_prefix: Path) -> Path:
    # plink2 writes "<out>.vmiss" (string concat), NOT suffix replacement
    return Path(str(out_prefix) + ".vmiss")


def _looks_like_vmiss(p: Path) -> tuple[bool, str]:
    """
    plink2 .vmiss header (whitespace-delimited) looks like:
      #CHROM  ID  MISSING_CT  OBS_CT  F_MISS
    """
    try:
        if not p.exists():
            return (False, "missing")
        if p.stat().st_size < 64:
            return (False, "too_small")
        with p.open("r") as f:
            hdr = f.readline().strip().split()
        hs = set(hdr)
        ok = (("F_MISS" in hs) and ("ID" in hs) and (("#CHROM" in hs) or ("CHROM" in hs)))
        if not ok:
            return (False, "bad_header:" + " ".join(hdr[:8]))
        return (True, "ok")
    except Exception as e:
        return (False, f"error:{type(e).__name__}")


def _search_dirs(prefix: Path, out_dir_p: Path) -> list[Path]:
    # Real dirs in your environment; ordered by likelihood
    dirs = [
        out_dir_p.resolve(),                    # e.g. ../../workspaces -> /home/jupyter/workspaces
        Path.cwd().resolve(),
        Path.cwd().parent.resolve(),
        (Path.cwd() / "../..").resolve(),
        prefix.parent.resolve(),                # /home/jupyter
        (prefix.parent / "workspaces").resolve(),  # /home/jupyter/workspaces
        Path.home().resolve(),                  # /home/jupyter
        (Path.home() / "workspaces").resolve(), # /home/jupyter/workspaces
        Path("/home/jupyter/workspaces"),
        Path("/home/jupyter"),
        Path("/home"),
        Path("/tmp"),
    ]
    out = []
    seen = set()
    for d in dirs:
        s = str(d)
        if s in seen:
            continue
        seen.add(s)
        if d.exists():
            out.append(d)
    return out


def _find_cached_vmiss(prefix_name: str, out_prefix: Path, search_dirs: list[Path]) -> tuple[Path | None, list[tuple[Path, str]]]:
    checked: list[tuple[Path, str]] = []

    # 0) Most likely: exactly where THIS invocation would write it
    expected = _vmiss_path_for_outprefix(out_prefix)
    ok, why = _looks_like_vmiss(expected)
    checked.append((expected, why))
    if ok:
        return expected, checked

    # 1) Common known names
    want_names = [
        f"{prefix_name}.missingness.vmiss",
        f"{prefix_name}.missingness.variantonly.vmiss",
        f"{prefix_name}.vmiss",
    ]

    candidates: list[Path] = []

    for d in search_dirs:
        for nm in want_names:
            p = d / nm
            ok, why = _looks_like_vmiss(p)
            if p.exists():
                checked.append((p, why))
            if ok:
                candidates.append(p)

    # 2) Glob fallback
    for d in search_dirs:
        for p in sorted(d.glob(f"{prefix_name}*.vmiss")):
            ok, why = _looks_like_vmiss(p)
            checked.append((p, why))
            if ok:
                candidates.append(p)

    if not candidates:
        return None, checked

    best = sorted(set(candidates), key=lambda x: x.stat().st_mtime, reverse=True)[0]
    return best, checked


def _run_plink2_variant_missingness(prefix: Path, out_prefix: Path, threads: int) -> Path:
    plink2 = shutil.which("plink2")
    if not plink2:
        raise RuntimeError("plink2 not found on PATH.")

    cmd = [
        plink2,
        "--bfile", str(prefix),
        "--missing", "variant-only",
        "--threads", str(threads),
        "--out", str(out_prefix),
    ]

    vmiss_path = _vmiss_path_for_outprefix(out_prefix)
    log_path = Path(str(out_prefix) + ".log")
    stdout_path = Path(str(out_prefix) + ".stdout.txt")
    stderr_path = Path(str(out_prefix) + ".stderr.txt")

    print("STEP 2/4: CACHE MISS -> running plink2 now (PER-VARIANT missingness only).")
    print("         " + " ".join(cmd))
    print(f"         expected output: {vmiss_path}")

    status = display("  0%  |  plink2 starting", display_id=True)
    t0 = time.time()
    last_update = 0.0
    last_pct: int | None = 0
    last_stage = "plink2 running"

    with stdout_path.open("w") as f_out, stderr_path.open("w") as f_err:
        p = subprocess.Popen(
            cmd,
            stdout=f_out,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert p.stderr is not None

        for line in p.stderr:
            f_err.write(line)
            s = line.strip()
            if not s:
                continue

            if s.startswith("Calculating "):
                last_stage = s.split("...")[0].strip()

            m = re.search(r"(\d{1,3})%\s*$", s)
            if m:
                last_pct = int(m.group(1))

            now = time.time()
            if (now - last_update) >= 0.75:
                elapsed = now - t0
                pct_txt = f"{last_pct:3d}%" if last_pct is not None else "   ?%"
                status.update(f"{pct_txt}  |  {last_stage}  |  {elapsed:,.1f}s elapsed")
                last_update = now

        rc = p.wait()

    if rc != 0:
        raise RuntimeError(
            f"plink2 failed (exit={rc}). See:\n"
            f"  {log_path}\n  {stderr_path}\n  {stdout_path}"
        )

    ok, why = _looks_like_vmiss(vmiss_path)
    if not ok:
        raise FileNotFoundError(f"Expected vmiss not found/invalid: {vmiss_path} ({why}). See {log_path}")

    status.update("100%  |  done")
    print(f"STEP 2/4: plink2 DONE -> {vmiss_path}")
    return vmiss_path


def _derive_pos_from_id_if_possible(vmiss_df: pd.DataFrame) -> tuple[pd.Series, float]:
    """
    If ID is like chr1:598858:A:C, extract 598858.
    Returns (pos_series_int32_with_NA, success_fraction).
    """
    # extract digits between first and second colon
    pos = vmiss_df["ID"].astype("string").str.extract(r"^[^:]+:(\d+):", expand=False)
    ok_mask = pos.notna()
    ok_frac = float(ok_mask.mean()) if len(pos) else 0.0
    pos_int = pd.to_numeric(pos, errors="coerce").astype("Int64")  # nullable
    return pos_int, ok_frac


def snp_callrate_lists(
    bfile_prefix: str,
    out_dir: str,
    callrate_lo: float = 0.9999,
    threads: int | None = None,
    use_cache: bool = True,
) -> dict[str, Path]:
    prefix = Path(bfile_prefix).expanduser().resolve()
    bed = prefix.with_suffix(".bed")
    bim = prefix.with_suffix(".bim")
    fam = prefix.with_suffix(".fam")

    print("STEP 1/4: INPUT CHECK")
    print(f"  bfile prefix: {prefix}")
    if not (bed.exists() and bim.exists() and fam.exists()):
        raise FileNotFoundError(f"Missing bed/bim/fam:\n  {bed}\n  {bim}\n  {fam}")

    out_dir_p = Path(out_dir).expanduser().resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)
    if threads is None:
        threads = max(1, (os.cpu_count() or 1))

    out_prefix = out_dir_p / f"{prefix.name}.missingness"
    expected_vmiss = _vmiss_path_for_outprefix(out_prefix)

    print("STEP 2/4: FIND OR BUILD .vmiss (so we can skip plink2 if already present)")
    print(f"  expected vmiss: {expected_vmiss}")
    print(f"  cache enabled:  {use_cache}")

    search_dirs = _search_dirs(prefix=prefix, out_dir_p=out_dir_p)
    print("  searching dirs:")
    for d in search_dirs:
        print(f"   - {d}")

    vmiss_path: Path | None = None
    checked: list[tuple[Path, str]] = []
    if use_cache:
        vmiss_path, checked = _find_cached_vmiss(prefix_name=prefix.name, out_prefix=out_prefix, search_dirs=search_dirs)

    if vmiss_path is not None:
        print("STEP 2/4: CACHE HIT -> using existing vmiss (plink2 will NOT run).")
        print(f"  vmiss: {vmiss_path}")
    else:
        # if we saw anything and rejected it, print one concrete reason
        rejected = [(p, why) for (p, why) in checked if p.exists() and why != "ok"]
        if rejected:
            p, why = rejected[0]
            print(f"  note: found candidate but rejected it: {p} ({why})")
        vmiss_path = _run_plink2_variant_missingness(prefix=prefix, out_prefix=out_prefix, threads=threads)

    print("STEP 3/4: LOAD .vmiss + PRODUCE chr,pos LISTS")
    vmiss = pd.read_csv(
        vmiss_path,
        sep=r"\s+",
        usecols=["#CHROM", "ID", "F_MISS"],
        dtype={"#CHROM": "string", "ID": "string", "F_MISS": "float32"},
        engine="c",
    ).rename(columns={"#CHROM": "chr", "F_MISS": "f_miss"})

    # Fast path: parse pos from ID if it encodes it (your file does)
    pos_int, ok_frac = _derive_pos_from_id_if_possible(vmiss)
    if ok_frac >= 0.999:
        vmiss["pos"] = pos_int.astype("int32")
        using_bim = False
        print(f"  pos source: parsed from ID (success {ok_frac*100:.3f}%) -> NOT reading .bim")
        base_df = vmiss[["chr", "pos", "f_miss"]]
        n_total = int(base_df.shape[0])
    else:
        # Fallback: map ID -> BP via .bim (works for rsIDs etc.)
        using_bim = True
        print(f"  pos source: ID parsing insufficient ({ok_frac*100:.3f}%) -> mapping via .bim")

        bim_df = pd.read_csv(
            bim,
            sep=r"\s+",
            header=None,
            usecols=[0, 1, 3],
            names=["chr", "ID", "pos"],
            dtype={"chr": "string", "ID": "string", "pos": "int32"},
            engine="c",
        )
        fmiss_series = vmiss.set_index("ID")["f_miss"]
        bim_df["f_miss"] = bim_df["ID"].map(fmiss_series)

        n_unmapped = int(bim_df["f_miss"].isna().sum())
        if n_unmapped:
            print(f"  note: {n_unmapped:,} variants had no f_miss mapping; excluded from denominator.")
        base_df = bim_df.dropna(subset=["f_miss"])[["chr", "pos", "f_miss"]]
        n_total = int(base_df.shape[0])

    fmiss = base_df["f_miss"].to_numpy(dtype=np.float32, copy=False)
    mask_100 = (fmiss == 0.0)
    fmiss_max = np.float32((1.0 - float(callrate_lo)) + 1e-15)
    mask_lo = (fmiss <= fmiss_max)

    out_100 = out_dir_p / f"{prefix.name}.snps_callrate_100.tsv"
    out_lo  = out_dir_p / f"{prefix.name}.snps_callrate_ge_{callrate_lo:.4f}.tsv"

    base_df.loc[mask_100, ["chr", "pos"]].to_csv(out_100, sep="\t", index=False)
    base_df.loc[mask_lo,  ["chr", "pos"]].to_csv(out_lo,  sep="\t", index=False)

    c100 = int(mask_100.sum())
    clo = int(mask_lo.sum())
    p100 = (c100 / n_total) * 100.0 if n_total else 0.0
    plo = (clo / n_total) * 100.0 if n_total else 0.0

    print("STEP 4/4: SUMMARY")
    print(f"  Total SNPs considered:      {n_total:,}")
    print(f"  100% call rate (F_MISS=0):  {c100:,} ({p100:.4f}%)  -> {out_100}")
    print(f"  >= {callrate_lo*100:.4f}% call rate:   {clo:,} ({plo:.4f}%)  -> {out_lo}")
    print(f"  used .bim mapping:          {using_bim}")
    print("DONE")

    return {
        "missingness_file": vmiss_path,
        "snps_callrate_100_tsv": out_100,
        f"snps_callrate_ge_{callrate_lo:.4f}_tsv": out_lo,
    }


paths = snp_callrate_lists(
    bfile_prefix="../../arrays",
    out_dir="../../workspaces",
    callrate_lo=0.9999,
    threads=4,
    use_cache=True,
)
paths
```

23andMe v4 (Illumina HTS iSelect HD chip):
https://raw.githubusercontent.com/psbaltar/rawDNA2vcf/master/filter/23andme_v4.tsv

DTC reference set (Living DNA autosomal):
https://github.com/SauersML/gnomon/raw/refs/heads/main/data/autosomal.txt

DTC reference set (23andMe, Joshua Yoakem v5):
https://raw.githubusercontent.com/SauersML/gnomon/refs/heads/main/data/genome_Joshua_Yoakem_v5_Full_20250129211749.txt

DTC reference set (AncestryDNA, kat_suricata):
https://github.com/SauersML/reagle/raw/refs/heads/main/data/kat_suricata/ancestrydna.txt

DTC reference set (23andMe, kat_suricata v5):
https://raw.githubusercontent.com/SauersML/reagle/refs/heads/main/data/kat_suricata/23andme_genome_kat_suricata_v5_full_20171221130201.txt

DTC reference set (23andMe, Christopher Smith v5 ZIP):
https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/genome_Christopher_Smith_v5_Full_20230926164611.zip

DTC reference set (AncestryDNA, Christopher Smith ZIP):
https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/dna-data-2023-09-26.zip

Illumina HumanHap550 v3 BeadChip:
https://www.openbioinformatics.org/gengen/download/hh550v3_snptable.txt.gz

Illumina HumanHap550 (Human Hap 550v3):
https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayIllumina550.txt.gz

Illumina OmniExpress (Infinium OmniExpress-24 v1.4):
https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/humanomniexpress-24/v1-4/InfiniumOmniExpress-24v1-4_A1_csv.zip

Affymetrix Axiom UK Biobank array:
https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/Array_UKB_34.zip

Illumina Omni family:
https://webdata.illumina.com/downloads/productfiles/infinium-omni5-4/v1-2/infinium-omni5-4-v1-2-a2-manifest-file-csv.zip

Illumina’s OmniExpress-24 v1.4:
https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/humanomniexpress-24/v1-4/InfiniumOmniExpress-24v1-4_A1_csv.zip

Illumina HumanHap550 (possibly Quad+):
https://www.chg.ox.ac.uk/~wrayner/strand/BDCHP-1X10-HUMANHAP550_11218540_C-b37-strand.zip

Infinium DNA Analysis BeadChip (Global Screening Array v4.0, Build 38):
https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium-gsa-with-gcra/GSA-48v4-0_20085471_D2.csv

Genome-Wide DNA Analysis BeadChips:
https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/global-screening-array-24/v3-0/GSA-24v3-0-A2-manifest-file-csv.zip

Affymetrix GeneChip SNP 6.0:
https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayAffy6.txt.gz

---

## Compression check from first chunk (64 KiB range), run on 2026-02-11

Method: `curl -L -r 0-65535` per URL, then infer format from magic bytes (`PK..`, `1F 8B 08`, plain text headers).

| Source URL | First-chunk compression | Notes |
|---|---|---|
| https://raw.githubusercontent.com/psbaltar/rawDNA2vcf/master/filter/23andme_v4.tsv | none (plain text TSV) | Starts with `#CHROM POS RSID` |
| https://github.com/SauersML/gnomon/raw/refs/heads/main/data/autosomal.txt | none (plain text) | GitHub raw resolves to `raw.githubusercontent.com`; starts with `# Living DNA ...` |
| https://raw.githubusercontent.com/SauersML/gnomon/refs/heads/main/data/genome_Joshua_Yoakem_v5_Full_20250129211749.txt | none (plain text) | Starts with `# This data file generated by 23andMe ...` |
| https://github.com/SauersML/reagle/raw/refs/heads/main/data/kat_suricata/ancestrydna.txt | none (plain text) | GitHub raw resolves to `raw.githubusercontent.com`; starts with `#AncestryDNA raw data download` |
| https://raw.githubusercontent.com/SauersML/reagle/refs/heads/main/data/kat_suricata/23andme_genome_kat_suricata_v5_full_20171221130201.txt | none (plain text) | Starts with `# This data file generated by 23andMe ...` |
| https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/genome_Christopher_Smith_v5_Full_20230926164611.zip | zip | GitHub raw resolves to `raw.githubusercontent.com`; magic `PK 03 04` |
| https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/dna-data-2023-09-26.zip | zip | GitHub raw resolves to `raw.githubusercontent.com`; magic `PK 03 04` |
| https://www.openbioinformatics.org/gengen/download/hh550v3_snptable.txt.gz | gzip | Magic `1F 8B 08` |
| https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayIllumina550.txt.gz | gzip | Magic `1F 8B 08` |
| https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/humanomniexpress-24/v1-4/InfiniumOmniExpress-24v1-4_A1_csv.zip | zip | Magic `PK 03 04` |
| https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/Array_UKB_34.zip | zip | Magic `PK 03 04` |
| https://webdata.illumina.com/downloads/productfiles/infinium-omni5-4/v1-2/infinium-omni5-4-v1-2-a2-manifest-file-csv.zip | zip | Magic `PK 03 04` |
| https://www.chg.ox.ac.uk/~wrayner/strand/BDCHP-1X10-HUMANHAP550_11218540_C-b37-strand.zip | zip | Magic `PK 03 04` |
| https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium-gsa-with-gcra/GSA-48v4-0_20085471_D2.csv | none (plain text CSV) | Starts with `Illumina, Inc.` |
| https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/global-screening-array-24/v3-0/GSA-24v3-0-A2-manifest-file-csv.zip | zip | Magic `PK 03 04` |
| https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayAffy6.txt.gz | gzip | Magic `1F 8B 08` |

Important findings:
- `InfiniumOmniExpress-24v1-4_A1_csv.zip` appears twice in this list.

## File format/schema notes (parse-oriented), run on 2026-02-11

| Source URL | Container / encoding | Layout | Columns / schema | Parse notes |
|---|---|---|---|---|
| https://raw.githubusercontent.com/psbaltar/rawDNA2vcf/master/filter/23andme_v4.tsv | Plain text TSV | Header row, then records | `#CHROM`, `POS`, `RSID` | Treat as marker list (site catalog), not per-sample genotypes. |
| https://github.com/SauersML/gnomon/raw/refs/heads/main/data/autosomal.txt | Plain text TSV | Comment preamble (`# ...`), then data rows | Commented schema line shows `rsid`, `chromosome`, `position`, `genotype`; first data rows match 4 columns | Skip all `#` lines. `genotype` is diploid call like `AA`/`GG`. |
| https://raw.githubusercontent.com/SauersML/gnomon/refs/heads/main/data/genome_Joshua_Yoakem_v5_Full_20250129211749.txt | Plain text TSV | Comment preamble (`# ...`), then data rows | Commented schema line `rsid chromosome position genotype`; data rows are 4 columns | 23andMe style. Missing/no-call can be `--`. IDs may be rsIDs or internal IDs. |
| https://github.com/SauersML/reagle/raw/refs/heads/main/data/kat_suricata/ancestrydna.txt | Plain text TSV | Comment preamble (`# ...`), explicit header row, then data rows | `rsid`, `chromosome`, `position`, `allele1`, `allele2` | Skip `#` lines, then parse headered 5-column table. |
| https://raw.githubusercontent.com/SauersML/reagle/refs/heads/main/data/kat_suricata/23andme_genome_kat_suricata_v5_full_20171221130201.txt | Plain text TSV | Comment preamble (`# ...`), then data rows | Commented schema line `rsid chromosome position genotype`; data rows are 4 columns | 23andMe style; observed `--` no-call example. |
| https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/genome_Christopher_Smith_v5_Full_20230926164611.zip | ZIP | Single-file archive (first member name visible) | First member appears to be `genome_Christopher_Smith_v5_Full_20230926164611.txt` | After unzip, parse as 23andMe raw format (`rsid/chromosome/position/genotype` with `#` preamble). |
| https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/dna-data-2023-09-26.zip | ZIP | Archive; first member name not fully clear from 64 KiB sample | Strings suggest an Ancestry-like text payload (contains `AncestryDNA.txt`) | Unzip before parsing; likely Ancestry schema (`rsid/chromosome/position/allele1/allele2`). Confirm member/header at full extract time. |
| https://www.openbioinformatics.org/gengen/download/hh550v3_snptable.txt.gz | Gzip (text TSV inside) | Header row, then records | `Name`, `Chr`, `Position`, `SNP`, `ILMN Strand`, `Customer Strand` | Decompress then parse TSV. `SNP` values are allele pairs like `[T/C]`. |
| https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayIllumina550.txt.gz | Gzip (text TSV inside) | No header in sample; fixed-width tab fields | Inferred UCSC schema: `bin`, `chrom`, `chromStart`, `chromEnd`, `name`, `score`, `strand`, `observed` | Decompress then parse TSV with explicit column names. `chrom` is `chrN` style. |
| https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/humanomniexpress-24/v1-4/InfiniumOmniExpress-24v1-4_A1_csv.zip | ZIP | Archive with CSV member | First member visible: `InfiniumOmniExpress-24v1-4_A1.csv` | Unzip then parse Illumina manifest CSV (large assay table, comma-delimited). |
| https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/Array_UKB_34.zip | ZIP | Archive with CSV member | First member visible: `Axiom_UKB_WCSG.na34.annot.csv` | Unzip then parse Affymetrix annotation CSV. |
| https://webdata.illumina.com/downloads/productfiles/infinium-omni5-4/v1-2/infinium-omni5-4-v1-2-a2-manifest-file-csv.zip | ZIP | Archive with CSV member | First member visible: `InfiniumOmni5-4v1-2_A2.csv` | Unzip then parse Illumina manifest CSV. |
| https://www.chg.ox.ac.uk/~wrayner/strand/BDCHP-1X10-HUMANHAP550_11218540_C-b37-strand.zip | ZIP | Archive with strand-map member | Member string visible: `BDCHP-1X10-HUMANHAP550_11218540_C-b37.strand` | Unzip then parse strand mapping table (nonstandard extension; delimiter/header should be confirmed after extraction). |
| https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium-gsa-with-gcra/GSA-48v4-0_20085471_D2.csv | Plain text CSV | Sectioned file: metadata block then assay table | Starts with `[Heading]` key/value rows; `[Assay]` introduces comma-delimited table (header begins `IlmnID,Name,...`) | Parser should detect `[Assay]` line and parse subsequent CSV as main table. |
| https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/global-screening-array-24/v3-0/GSA-24v3-0-A2-manifest-file-csv.zip | ZIP | Archive with CSV member | First member visible: `GSA-24v3-0_A2.csv` | Unzip then parse Illumina manifest CSV. |
| https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayAffy6.txt.gz | Gzip (text TSV inside) | No header in sample; fixed-width tab fields | Inferred UCSC schema: `bin`, `chrom`, `chromStart`, `chromEnd`, `name`, `score`, `strand`, `observed`, `rsId` | Decompress then parse TSV with explicit column names. |

## Combined genome-build evidence table (URL inspection + max-position), run on 2026-02-11

Decision rule from `scripts/build_intersection_snp_list.py`:
- URL/filename hints: tokens like `hg19`, `grch37`, `b37`, `hg38`, `grch38`, `b38`
- Max-position counts: `too_big_grch37 > 0 && too_big_grch38 == 0 -> GRCh38`; `too_big_grch38 > 0 && too_big_grch37 == 0 -> GRCh37`
- Chr-window implication: if per-chromosome max falls in the region unique to one build, that build is implied

| # | Source URL | URL/filename hint | Declared build | Max-position counts (`too_big37`,`too_big38`) | Chr-window implication | Final call |
|---|---|---|---|---|---|---|
| 1 | https://raw.githubusercontent.com/psbaltar/rawDNA2vcf/master/filter/23andme_v4.tsv | none | none | (0, 2716) | GRCh37 | GRCh37 |
| 2 | https://github.com/SauersML/gnomon/raw/refs/heads/main/data/autosomal.txt | none | GRCh37 (text) | (0, 2563) | GRCh37 | GRCh37 |
| 3 | https://raw.githubusercontent.com/SauersML/gnomon/refs/heads/main/data/genome_Joshua_Yoakem_v5_Full_20250129211749.txt | none | GRCh37 (text) | (0, 3002) | GRCh37 | GRCh37 |
| 4 | https://github.com/SauersML/reagle/raw/refs/heads/main/data/kat_suricata/ancestrydna.txt | none | GRCh37 (text) | (0, 3010) | GRCh37 | GRCh37 |
| 5 | https://raw.githubusercontent.com/SauersML/reagle/refs/heads/main/data/kat_suricata/23andme_genome_kat_suricata_v5_full_20171221130201.txt | none | GRCh37 (text) | (0, 3064) | GRCh37 | GRCh37 |
| 6 | https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/genome_Christopher_Smith_v5_Full_20230926164611.zip | none | GRCh37 (text) | (0, 3003) | GRCh37 | GRCh37 |
| 7 | https://github.com/SauersML/reagle/raw/refs/heads/main/data/christopher_smith/dna-data-2023-09-26.zip | none | GRCh37 (text) | (0, 3131) | GRCh37 | GRCh37 |
| 8 | https://www.openbioinformatics.org/gengen/download/hh550v3_snptable.txt.gz | none | none | (1228, 2035) | GRCh37 | GRCh37 (count-conflict) |
| 9 | https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayIllumina550.txt.gz | `hg19` | GRCh37 (url/text) | (0, 1740) | GRCh37 | GRCh37 |
| 10 | https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/humanomniexpress-24/v1-4/InfiniumOmniExpress-24v1-4_A1_csv.zip | none | none | (0, 2785) | GRCh37 | GRCh37 |
| 11 | https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/Array_UKB_34.zip | none | GRCh37 (text) | (0, 5178) | GRCh37 | GRCh37 |
| 12 | https://webdata.illumina.com/downloads/productfiles/infinium-omni5-4/v1-2/infinium-omni5-4-v1-2-a2-manifest-file-csv.zip | none | none | (10848, 0) | GRCh38 | GRCh38 |
| 13 | https://www.chg.ox.ac.uk/~wrayner/strand/BDCHP-1X10-HUMANHAP550_11218540_C-b37-strand.zip | `b37` | GRCh37 (url/text) | (0, 1725) | GRCh37 | GRCh37 |
| 14 | https://support.illumina.com/content/dam/illumina-support/documents/documentation/chemistry_documentation/infinium_assays/infinium-gsa-with-gcra/GSA-48v4-0_20085471_D2.csv | none | none | (1602, 0) | GRCh38 | GRCh38 |
| 15 | https://support.illumina.com/content/dam/illumina-support/documents/downloads/productfiles/global-screening-array-24/v3-0/GSA-24v3-0-A2-manifest-file-csv.zip | none | none | (1720, 0) | GRCh38 | GRCh38 |
| 16 | https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayAffy6.txt.gz | `hg19` | GRCh37 (url/text) | (0, 2243) | GRCh37 | GRCh37 |

Note on source #8 (`hh550v3_snptable.txt.gz`): automatic max-position count checks were conflicting, but manual rsID spot-checks align to hg19/GRCh37, so it is treated as GRCh37.

## Production run notes (2026-02-11, reduced hg19 set)

For final production, the hg19 subset was reduced by dropping these three hg19 sources:
- `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayIllumina550.txt.gz`
- `https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/Array_UKB_34.zip`
- `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/snpArrayAffy6.txt.gz`

This yields the target nonzero hg19 intersection:
- hg19 native intersection: `58,234`

Liftover and final hg38 merge results:
- hg19->hg38 liftover input: `58,234`
- liftover mapped: `58,234`
- liftover invalid target rows: `6`
- lifted hg38 unique markers: `58,228`
- native hg38 intersection: `416,842`
- final intersection (`hg38_native ∩ lifted_hg19`): `56,330`

Final outputs:
- `data/snp_intersection_hg19_chrpos.txt`
- `data/snp_intersection_hg19_lifted_to_hg38_chrpos.txt`
- `data/snp_intersection_hg38_native_chrpos.txt`
- `data/snp_intersection_final_hg38_chrpos.txt`
- `data/snp_intersection_summary.json`
