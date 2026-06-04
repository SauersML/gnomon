#!/usr/bin/env python3
"""Fit survival marginal-slope GAMs on All of Us microarray data.

For each disease the script:
  1. Looks up the SNOMED standard Condition concept by name in the
     OMOP/OHDSI `concept` table on the AoU CDR.
  2. Pulls everyone whose `condition_occurrence.condition_concept_id`
     descends from that concept via `concept_ancestor`.
  3. Fits `Surv(entry_age, exit_age, event) ~ duchon(PC1..PC10) + sex`
     with the hard-coded PGS feeding the marginal-slope latent z and the same
     isotropic Duchon smooth on the log-slope channel. No `linkwiggle()` on
     either channel (link-deviation / score-warp both hang, SauersML/gam#683).
  4. Compares against Z_norm2 and raw-PRS+PC Cox PH baselines on the same split.
  5. Runs leave-one-group-out OOD refits by care site, Census region, and
     AoU inferred genetic ancestry category.

Reported per disease: IPCW concordance, integrated Brier score, and Graf-style
integrated-Brier pseudo-R^2 on held-out rows.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import struct
import subprocess
import sys
import urllib.request
import zipfile
import faulthandler
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
faulthandler.enable(file=sys.stderr, all_threads=True)

SCRIPT_DIR = Path(__file__).resolve().parent
HOME = Path.home()

_CUDA_SONAME_FAMILIES = {
    "libcuda",
    "libcudart",
    "libcublas",
    "libcublasLt",
    "libcusparse",
    "libcusolver",
    "libnvJitLink",
    "libnvrtc",
}


def user_cache_root() -> Path:
    """Return the cache root used by Rust's `dirs::cache_dir()` on Linux."""
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser()
    return HOME / ".cache"


def cuda_library_family(basename: str) -> str | None:
    stem = basename.split(".so", 1)[0]
    return stem if stem in _CUDA_SONAME_FAMILIES else None


def mapped_cuda_libraries() -> dict[str, list[str]]:
    maps = Path("/proc/self/maps")
    try:
        content = maps.read_text()
    except OSError:
        return {}
    grouped: dict[str, set[str]] = {}
    for line in content.splitlines():
        fields = line.split()
        if not fields:
            continue
        raw_path = fields[-1]
        if not raw_path.startswith("/"):
            continue
        family = cuda_library_family(Path(raw_path).name)
        if family is None:
            continue
        grouped.setdefault(family, set()).add(raw_path)
    return {family: sorted(paths) for family, paths in sorted(grouped.items())}


def print_cuda_process_diagnostics(prefix: str) -> None:
    print(f"{prefix} CUDA process maps:")
    mapped = mapped_cuda_libraries()
    if not mapped:
        print(f"{prefix}   <none>")
        return
    for family, paths in mapped.items():
        conflict = " CONFLICT" if len(paths) > 1 else ""
        print(f"{prefix}   {family}:{conflict}")
        for path in paths:
            print(f"{prefix}     {path}")


def print_gamfit_diagnostics(gamfit_module) -> None:
    print("gamfit diagnostics:")
    print(f"  module_file: {getattr(gamfit_module, '__file__', '<unknown>')}")
    try:
        print(gamfit_module.format_cuda_diagnostics())
    except Exception as exc:
        print(f"  cuda_diagnostics_error: {type(exc).__name__}: {exc}")
    try:
        info = gamfit_module.build_info()
    except Exception as exc:
        print(f"  build_info_error: {type(exc).__name__}: {exc}")
        return
    for key in ("available", "module", "crate", "engine_crate", "python_module", "version"):
        if key in info:
            print(f"  build_info.{key}: {info[key]}")


def _dedup_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _search_roots() -> list[Path]:
    return _dedup_paths([Path.cwd(), SCRIPT_DIR, HOME])


def _python_cuda_lib_dirs() -> list[Path]:
    """Find CUDA shared-library directories from Python nvidia wheels."""
    nvidia_roots: list[Path] = []
    try:
        import nvidia
    except Exception:
        pass
    else:
        for raw_path in getattr(nvidia, "__path__", []):
            nvidia_roots.append(Path(raw_path))
    for raw_path in sys.path:
        if not raw_path:
            continue
        nvidia_roots.append(Path(raw_path) / "nvidia")

    lib_dirs: list[Path] = []
    for root in _dedup_paths(nvidia_roots):
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            lib_dir = child / "lib"
            if not lib_dir.is_dir():
                continue
            try:
                has_shared_library = any(
                    entry.is_file() and ".so" in entry.name
                    for entry in lib_dir.iterdir()
                )
            except OSError:
                continue
            if has_shared_library:
                lib_dirs.append(lib_dir)

    order = {
        "cuda_nvrtc": 0,
        "cuda_runtime": 1,
        "nvjitlink": 2,
        "cublas": 3,
        "cusparse": 4,
        "cusolver": 5,
    }
    return sorted(
        _dedup_paths(lib_dirs),
        key=lambda path: (order.get(path.parent.name, 100), path.parent.name),
    )


def _gnomon_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    cuda_lib_dirs = _python_cuda_lib_dirs()
    if not cuda_lib_dirs:
        return env

    existing = [
        Path(raw_path)
        for raw_path in env.get("LD_LIBRARY_PATH", "").split(":")
        if raw_path
    ]
    ld_paths = _dedup_paths([*cuda_lib_dirs, *existing])
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in ld_paths)
    print("[score] CUDA library path for gnomon subprocess:", flush=True)
    for path in cuda_lib_dirs:
        print(f"[score]   {path}", flush=True)
    return env


def _iter_files_with_suffix(suffix: str, max_depth: int = 4):
    suffix = suffix.lower()
    skip_dirs = {
        ".git",
        "__pycache__",
        ".ipynb_checkpoints",
        "node_modules",
        "target",
        "vendor",
    }
    for root in _search_roots():
        queue = deque([(root, 0)])
        while queue:
            current, depth = queue.popleft()
            try:
                entries = sorted(
                    current.iterdir(),
                    key=lambda p: (not p.is_dir(), p.name.lower()),
                )
            except OSError:
                continue
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(suffix):
                    yield entry
            if depth >= max_depth:
                continue
            dirs = [
                entry for entry in entries
                if entry.is_dir() and entry.name not in skip_dirs
            ]

            def dir_priority(path: Path) -> tuple[int, str]:
                name = path.name.lower()
                if "plink" in name or "array" in name:
                    return (0, name)
                if "pgs" in name or "genotype" in name:
                    return (1, name)
                if "cache" in name:
                    return (2, name)
                return (3, name)

            dirs.sort(key=dir_priority)
            for directory in dirs:
                queue.append((directory, depth + 1))


def _plink_prefix_from_bed(bed: Path) -> Path | None:
    prefix = bed.with_suffix("")
    if not prefix.with_suffix(".bim").exists():
        return None
    if not prefix.with_suffix(".fam").exists():
        return None
    return prefix


def find_plink_prefix() -> Path:
    """Find the first local PLINK1 bed/bim/fam triplet by shallow BFS."""
    for bed in _iter_files_with_suffix(".bed"):
        prefix = _plink_prefix_from_bed(bed)
        if prefix is None:
            continue
        print(f"plink: prefix={prefix}")
        return prefix
    roots = ", ".join(str(p) for p in _search_roots())
    raise FileNotFoundError(
        "could not find a local PLINK bed/bim/fam triplet under: "
        f"{roots}"
    )


# Local PLINK1 bed/bim/fam, typically staged once from AoU controlled storage.
# Discovery intentionally looks for an existing triplet instead of assuming any
# particular cache or repository layout.
PLINK_PREFIX = find_plink_prefix()
WORKDIR = PLINK_PREFIX.parent
FITS_DIR = WORKDIR / "biobank_fits"
NUM_PCS = 3
DUCHON_CENTERS = 10  # > linear nullspace (d+1=4) in d=3
TRAIN_FRACTION = 0.80  # per-class 80/20 split
RNG_SEED = 0
MAX_LOSO_CARE_SITES = 5
MIN_LOSO_TRAIN_EVENTS = 50
MIN_LOSO_TRAIN_CENSORS = 50
# IPCW concordance and Brier curves are event-limited; below ~1000 held-out
# events the LOSO fold noise dominates the GAM-minus-baseline delta.
MIN_LOSO_TEST_EVENTS = 1000
MIN_LOSO_TEST_CENSORS = 1000
MIN_LOSO_TEST_N = 2000
LOSO_AXES = ("care_site", "census_region", "ancestry")
LOSO_AXIS_TO_COLUMN = {
    "care_site": "care_site_group",
    "census_region": "census_region",
    "ancestry": "ancestry_category",
}
GNOMON_BIN = os.environ.get("GNOMON_BIN", "gnomon")
PGS_ID_PATTERN = re.compile(r"^PGS\d{6}$")
BOOTSTRAP_RESAMPLES = 399
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

# AoU controlled CDR layout (per the "Controlled CDR Directory" doc and "How
# the All of Us Genomic data are organized"):
#   $WORKSPACE_CDR       BigQuery dataset    e.g. fc-aou-cdr-prod.C2024Q3R3
#   $CDR_STORAGE_PATH    GCS root            gs://fc-aou-datasets-controlled/v8
#   $GOOGLE_PROJECT      billing project     used for requester-pays GCS reads
#
# All of the file paths below are sub-paths of CDR_STORAGE_PATH so the script
# follows the CDR version pinned by the workspace instead of hardcoding "v8".
# When CDR_STORAGE_PATH is unset (e.g. running outside a workspace), we fall
# back to the v8 paths documented at the time of writing.
CDR_STORAGE_PATH = os.environ.get(
    "CDR_STORAGE_PATH", "gs://fc-aou-datasets-controlled/v8"
).rstrip("/")
ANCESTRY_DIR_URI = f"{CDR_STORAGE_PATH}/wgs/short_read/snpindel/aux/ancestry"
# AoU's "How the genomic data are organized" doc lists this file as bare
# `ancestry_preds.tsv`, but the actual object shipped in v8 (both
# `fc-aou-datasets-controlled` and the Verily Workbench mirror
# `vwb-aou-datasets-controlled`) is build-prefixed, e.g.
# `echo_v4_r2.ancestry_preds.tsv`. We resolve the real name lazily via
# `_resolve_ancestry_preds_uri` so the URL works in both layouts.
_DEFAULT_ANCESTRY_PREDS_URI = f"{ANCESTRY_DIR_URI}/ancestry_preds.tsv"
_resolved_ancestry_preds_uri: str | None = None


def _resolve_ancestry_preds_uri() -> str:
    """Return the actual GCS URI for `*ancestry_preds.tsv` under the CDR.

    Falls back to the documented bare-name path if a directory listing isn't
    possible (e.g. gcsfs not importable, no bucket-level list permission).
    The caller is still responsible for handling the eventual read error.
    """
    global _resolved_ancestry_preds_uri
    if _resolved_ancestry_preds_uri is not None:
        return _resolved_ancestry_preds_uri
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem(
            project=os.environ.get("GOOGLE_PROJECT", ""),
            requester_pays=True,
        )
        matches = fs.glob(f"{ANCESTRY_DIR_URI}/*ancestry_preds.tsv")
        # Prefer the bare name if present, else the build-prefixed file.
        preferred = f"{ANCESTRY_DIR_URI.replace('gs://', '', 1)}/ancestry_preds.tsv"
        chosen = next((m for m in matches if m.endswith(preferred)), None)
        if chosen is None and matches:
            chosen = matches[0]
        if chosen:
            uri = chosen if chosen.startswith("gs://") else f"gs://{chosen}"
            _resolved_ancestry_preds_uri = uri
            return uri
    except Exception as exc:
        print(f"  ancestry_preds: glob failed ({type(exc).__name__}: {exc}); "
              f"using documented path {_DEFAULT_ANCESTRY_PREDS_URI}")
    _resolved_ancestry_preds_uri = _DEFAULT_ANCESTRY_PREDS_URI
    return _resolved_ancestry_preds_uri


# Backwards-compatible name; callers read this and may also call
# `_resolve_ancestry_preds_uri()` directly for the just-in-time resolution.
ANCESTRY_PREDS_URI = _DEFAULT_ANCESTRY_PREDS_URI
ADMIXTURE_Q_URI = (
    f"{CDR_STORAGE_PATH}/wgs/short_read/snpindel/aux/admixture_estimates/"
    "aou_admixture_estimates_rye_v8.Q"
)
MICROARRAY_PLINK_PREFIX_URI = f"{CDR_STORAGE_PATH}/microarray/plink/arrays"
ANCESTRY_PREDS_CACHE = WORKDIR / "ancestry_preds.tsv"

# Curated SNOMED -> PGS map, keyed by SNOMED concept_code. The runtime
# disease set is built by intersecting this map with the OHDSI Phenotype
# Library's canonical Reference disease cohorts and then ranking the
# intersection by case prevalence (descendant expansion via
# `concept_ancestor`) in the active CDR; the top TOP_N_DISEASES survive.
# Extending this map is the only thing needed to grow the runtime set.
SNOMED_PGS_MAP: dict[str, dict[str, str]] = {
    # COPD -- Jung et al. metaPRS for J44. Not trained in AoU.
    "13645005": {"slug": "copd", "pgs": "PGS004536"},
    # Hypertension -- Privé et al. 2022 sparse hypertension PRS. Not trained in AoU.
    "38341003": {"slug": "hypertension", "pgs": "PGS001320"},
    # Obesity -- Kim et al. 2026 O_MetPRS_EUR; LDpred2 over multi-ancestry GWAS.
    "414916001": {"slug": "obesity", "pgs": "PGS005331"},
}

TOP_N_DISEASES = 15

# OHDSI Phenotype Library: canonical Reference disease cohorts.
SNOMED_DISEASE_CODE = "64572001"  # SNOMED root for 'Disease (disorder)'
PL_ZIP_URL = "https://github.com/OHDSI/PhenotypeLibrary/archive/refs/heads/main.zip"
PL_CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "ohdsi_phenotype_library"
)
PL_LOCAL_DIR = PL_CACHE_DIR / "PhenotypeLibrary-main"


# --- loaders ---------------------------------------------------------------

def _canonical_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


_SEX_MAP = {
    "0": 0, "1": 1, "2": 0,
    "f": 0, "female": 0, "m": 1, "male": 1,
    "xx": 0, "xy": 1,
}


def _load_sex_from_aou(client: "bigquery.Client", cdr: str) -> pd.DataFrame | None:
    """Pull biological sex per participant from AoU's OMOP `person` table.

    Tries several columns in order of reliability:
      1. `gender_concept_id` — OMOP standard, always 8507/8532 when populated.
      2. `sex_at_birth_concept_id` — same 8507/8532 codes when AoU populates it.
      3. `sex_at_birth_source_value` — raw text ("Male"/"Female"), used when
         AoU uses non-standard / PMI custom concept ids upstream.

    Returns None on any error so the caller can fall back to PLINK/gnomon.
    """
    try:
        q = f"""
            SELECT CAST(person_id AS STRING) AS person_id,
                   gender_concept_id,
                   sex_at_birth_concept_id,
                   LOWER(sex_at_birth_source_value) AS sex_at_birth_source_value
            FROM `{cdr}.person`
        """
        df = client.query(q).result().to_dataframe()
    except Exception as exc:
        print(f"  sex (AoU): unavailable ({type(exc).__name__}: {exc}); "
              f"will fall back to gnomon terms --sex")
        return None

    n_total = len(df)
    # AoU uses PPI concept ids alongside the OMOP standards:
    #   8507  = MALE   (OMOP standard)
    #   8532  = FEMALE (OMOP standard)
    #   45880669 = "SexAtBirth_Male"   (PPI, dominant in AoU)
    #   45878463 = "SexAtBirth_Female" (PPI, dominant in AoU)
    code_map = {8507: 1, 8532: 0, 45880669: 1, 45878463: 0}
    # source_value strings observed in AoU CDR are "sexatbirth_male" /
    # "sexatbirth_female" (lowercased). Accept any value containing "male"
    # not preceded by "fe".
    def _text_to_sex(s):
        if not isinstance(s, str):
            return None
        s = s.strip().lower()
        if "female" in s or s in ("f", "xx"):
            return 0
        if "male" in s or s in ("m", "xy"):
            return 1
        return None
    sex = df["gender_concept_id"].map(code_map)
    n_from_gender = int(sex.notna().sum())
    fill_sab_code = df["sex_at_birth_concept_id"].map(code_map)
    sex = sex.fillna(fill_sab_code)
    n_after_sab_code = int(sex.notna().sum())
    fill_sab_text = df["sex_at_birth_source_value"].map(_text_to_sex)
    sex = sex.fillna(fill_sab_text)
    n_after_sab_text = int(sex.notna().sum())

    if n_after_sab_text == 0:
        gc = df["gender_concept_id"].value_counts(dropna=False).head(5).to_dict()
        sc = df["sex_at_birth_concept_id"].value_counts(dropna=False).head(5).to_dict()
        sv = df["sex_at_birth_source_value"].value_counts(dropna=False).head(5).to_dict()
        print(f"  sex (AoU): no usable values in {cdr}.person "
              f"(rows={n_total:,}); top gender_concept_id={gc} "
              f"sex_at_birth_concept_id={sc} sex_at_birth_source_value={sv}; "
              f"falling back")
        return None

    keep = sex.notna()
    out = pd.DataFrame({
        "person_id": _canonical_id(df.loc[keep, "person_id"]),
        "sex": sex[keep].astype(int).values,
    })
    print(
        f"  sex (AoU): n={len(out):,} of {n_total:,} "
        f"(gender_concept_id={n_from_gender:,}, "
        f"+sex_at_birth_concept_id={n_after_sab_code - n_from_gender:,}, "
        f"+source_value={n_after_sab_text - n_after_sab_code:,})  "
        f"src={cdr}.person"
    )
    return out


def load_sex(client: "bigquery.Client | None" = None, cdr: str | None = None) -> pd.DataFrame:
    # 1. Prefer any already-present gnomon-derived sex TSV (free, instant).
    path = PLINK_PREFIX.with_name(f"{PLINK_PREFIX.name}.sex.tsv")
    if not path.exists():
        # The PLINK triplet may have been staged in a different directory than
        # the one find_plink_prefix() picked; search common locations for any
        # `*.sex.tsv` written next to a previous staging.
        sidecar_candidates: list[Path] = []
        for sidecar in _iter_files_with_suffix(".sex.tsv"):
            sidecar_candidates.append(sidecar)
        if sidecar_candidates:
            sidecar_candidates.sort(key=lambda p: p.stat().st_mtime)
            chosen = sidecar_candidates[-1]
            print(f"  sex: reusing sidecar cache {chosen}")
            path = chosen
    if not path.exists():
        legacy_dir = Path.home() / ".aou_cache" / "sex_terms"
        legacy_hits = sorted(legacy_dir.glob("sex_*.tsv"), key=lambda p: p.stat().st_mtime) if legacy_dir.is_dir() else []
        if legacy_hits:
            legacy = legacy_hits[-1]
            print(f"  sex: reusing legacy cache {legacy}")
            path = legacy
    # 2. Otherwise pull from AoU's OMOP person table (also instant; canonical).
    if not path.exists() and client is not None and cdr:
        df = _load_sex_from_aou(client, cdr)
        if df is not None:
            return df
    # 3. Last resort — invoke gnomon terms --sex (slow, ~1h on full AoU).
    if not path.exists():
        cmd = [GNOMON_BIN, "terms", "--sex", str(PLINK_PREFIX)]
        print(f"  sex: running {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    if not path.exists():
        raise FileNotFoundError(f"gnomon terms did not write expected sex TSV: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    id_col = next(c for c in df.columns if c.lower() in {"research_id", "sample_id", "iid", "person_id", "#iid"})
    sex_col = next(c for c in df.columns if "sex" in c.lower())
    out = pd.DataFrame({
        "person_id": _canonical_id(df[id_col]),
        "sex": df[sex_col].astype(str).str.strip().str.lower().map(_SEX_MAP),
    }).dropna()
    out["sex"] = out["sex"].astype(int)
    print(f"  sex:  file={path.name}  n={len(out):,}")
    return out


def _load_pcs_from_gnomon(num_pcs: int) -> pd.DataFrame:
    """Read gnomon's `projection_scores.bin` (GNPRJ001/GNPSID01) as a fallback."""
    bin_path = PLINK_PREFIX.with_name(f"{PLINK_PREFIX.name}.projection_scores.bin")
    meta_path = PLINK_PREFIX.with_name(f"{PLINK_PREFIX.name}.projection_scores.metadata.json")
    meta = json.loads(meta_path.read_text())
    rows, cols = int(meta["rows"]), int(meta["cols"])
    with bin_path.open("rb") as fh:
        assert fh.read(8) == b"GNPRJ001"
        fh.read(4 + 8 + 8 + 4)
        data = np.fromfile(fh, dtype="<f8", count=rows * cols).reshape(cols, rows).T
        assert fh.read(8) == b"GNPSID01"
        fh.read(4 + 4)
        count = struct.unpack("<Q", fh.read(8))[0]
        sb = struct.unpack("<Q", fh.read(8))[0]
        offsets = np.frombuffer(fh.read(8 * (count + 1)), dtype="<u8")
        blob = fh.read(sb)
    ids = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(count)]
    df = pd.DataFrame(data[:, :num_pcs], columns=[f"PC{i+1}" for i in range(num_pcs)])
    df.insert(0, "person_id", _canonical_id(pd.Series(ids)))
    return df


def _load_pcs_from_aou(num_pcs: int) -> pd.DataFrame | None:
    """Pull the AoU-published 16 PCs from `ancestry_preds.tsv:pca_features`.

    Per the "How the All of Us Genomic data are organized" doc (Table 9), the
    ancestry preds TSV carries a `pca_features` Array[number] column of length
    16 for every srWGS sample, produced by Hail's `hwe_normalized_pca` against
    1KGP+HGDP training data. Using these instead of gnomon's locally-computed
    PCs aligns the model with AoU's canonical genetic-similarity coordinates.

    Returns None if `ANCESTRY_PREDS_CACHE` is missing or fetch from
    `ANCESTRY_PREDS_URI` fails (e.g. workspace lacks bucket access -- the
    same 403 that disables the ancestry LOSO axis). Caller falls back to
    gnomon's projection scores.
    """
    if num_pcs > 16:
        return None
    src = None
    if ANCESTRY_PREDS_CACHE.exists() and ANCESTRY_PREDS_CACHE.stat().st_size > 0:
        src = ANCESTRY_PREDS_CACHE
        df_raw = pd.read_csv(src, sep="\t", usecols=["research_id", "pca_features"], dtype=str)
    else:
        try:
            uri = _resolve_ancestry_preds_uri()
            print(f"  pcs (AoU): GET {uri} -> {ANCESTRY_PREDS_CACHE}")
            # Cache all three columns so load_genetic_ancestry_labels can reuse the same file.
            df_full = pd.read_csv(
                uri,
                sep="\t",
                usecols=["research_id", "ancestry_pred", "pca_features"],
                dtype=str,
                storage_options={
                    "project": os.environ.get("GOOGLE_PROJECT", ""),
                    "requester_pays": True,
                },
            )
            ANCESTRY_PREDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            df_full.to_csv(ANCESTRY_PREDS_CACHE, sep="\t", index=False)
            df_raw = df_full[["research_id", "pca_features"]]
            src = ANCESTRY_PREDS_URI
        except Exception as exc:
            print(f"  pcs (AoU): unavailable ({type(exc).__name__}: {exc}); "
                  f"will fall back to gnomon projection scores")
            return None

    # pca_features is serialized as e.g. "[8.1232, 0.0123, ..., 0.001]" (16 floats).
    parsed = df_raw["pca_features"].map(
        lambda s: json.loads(s) if isinstance(s, str) and s.startswith("[") else None
    )
    mask = parsed.notna() & parsed.map(lambda v: isinstance(v, list) and len(v) >= num_pcs)
    if not mask.any():
        print("  pcs (AoU): no usable pca_features rows; falling back to gnomon")
        return None
    arr = np.array([v[:num_pcs] for v in parsed[mask].tolist()], dtype=float)
    df = pd.DataFrame(arr, columns=[f"PC{i+1}" for i in range(num_pcs)])
    df.insert(0, "person_id", _canonical_id(df_raw.loc[mask, "research_id"].reset_index(drop=True)))
    print(f"  pcs (AoU): n={len(df):,}  src={src}  using top {num_pcs} of 16 PCs")
    return df


def load_pcs(num_pcs: int) -> pd.DataFrame:
    """Load `num_pcs` PCs per participant.

    Prefers AoU's official `pca_features` from `ancestry_preds.tsv` (the 16
    PCs from `hwe_normalized_pca` against 1KGP+HGDP, packaged with the CDR);
    falls back to gnomon's local `<plink-prefix>.projection_scores.bin` if the
    AoU file is unreachable (e.g. the same 403 that disables the ancestry LOSO
    axis when the workspace can't read `fc-aou-datasets-controlled`).
    """
    df = _load_pcs_from_aou(num_pcs)
    if df is None:
        df = _load_pcs_from_gnomon(num_pcs)
        print(
            f"  pcs (gnomon): n={len(df):,}  "
            f"source={PLINK_PREFIX.name}.projection_scores.bin"
        )
    return df


def _iter_sscore_files():
    seen: set[Path] = set()
    local_patterns = [
        f"{PLINK_PREFIX.name}_*.sscore",
        "arrays_*.sscore",
        "*.sscore",
    ]
    for pattern in local_patterns:
        for path in PLINK_PREFIX.parent.glob(pattern):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path
    for path in _iter_files_with_suffix(".sscore"):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        yield path


def _find_sscore_for(pgs_id: str) -> Path | None:
    """Find any `.sscore` whose header contains `{pgs_id}_AVG`.

    gnomon names per-PGS outputs differently depending on how `--score` is
    invoked:
      - directory of pre-downloaded score files -> `arrays_<PGS_ID>.sscore`
      - inline comma-separated IDs              -> `arrays_pgs<N>_<hash>.sscore`
    Filename matching alone would miss the second form, so we look at column
    headers instead.
    """
    target = f"{pgs_id}_AVG"
    for path in _iter_sscore_files():
        try:
            with path.open() as fh:
                header = fh.readline().rstrip("\n").split("\t")
        except OSError:
            continue
        if target in header:
            return path
    return None


def ensure_scored(pgs_ids: list[str]) -> None:
    """Run a single `gnomon score` call for any PGS not yet scored on disk.

    "Already scored" = some `.sscore` visible to the discovery scan has a `<PGS>_AVG`
    column. This catches both gnomon's per-PGS naming and the hashed inline
    naming, so we don't re-score a file that's already there under a
    different filename and trip gnomon's overwrite refusal.
    """
    missing = [p for p in pgs_ids if _find_sscore_for(p) is None]
    if not missing:
        return
    # Score every missing PGS in ONE gnomon invocation by joining the IDs
    # with commas (gnomon score parses the SCORE_PATH argument as a
    # comma-separated list of PGS Catalog IDs at score/main.rs:128-139).
    # Single-call multi-PGS amortizes the ~1 s preparation, the ~7 s
    # CUDA context init + NVRTC compile, and reuses the cuBLAS handle
    # across all PGS, instead of paying those costs N times in a
    # per-PGS subprocess loop.
    #
    # Earlier revisions of this script split each PGS into its own
    # subprocess as a workaround for an AoU CUDA teardown abort
    # ("double free or corruption (!prev)" at ~100% progress). If that
    # bug resurfaces, this call will fail without writing all outputs.
    score_arg = ",".join(missing)
    env = _gnomon_subprocess_env()
    print(f"[score] running {GNOMON_BIN} score {score_arg} {PLINK_PREFIX}")
    result = subprocess.run(
        [GNOMON_BIN, "score", score_arg, str(PLINK_PREFIX)], check=False, env=env,
    )
    still_missing = [p for p in missing if _find_sscore_for(p) is None]
    if still_missing:
        print(
            f"[score] subprocess returned {result.returncode} and the "
            f"sscore output(s) for {still_missing} were not written",
            flush=True,
        )
        raise subprocess.CalledProcessError(result.returncode, result.args)
    if result.returncode != 0:
        print(
            f"[score] subprocess returned {result.returncode} but every "
            f"requested `.sscore` is on disk; treating as success "
            f"(known cudarc atexit abort, fixed in gnomon >= 1896221e).",
            flush=True,
        )


def load_one_pgs(pgs_id: str) -> pd.DataFrame:
    path = _find_sscore_for(pgs_id)
    if path is None:
        raise FileNotFoundError(
            f"no discovered sscore file carries column {pgs_id}_AVG"
        )
    avg_col = f"{pgs_id}_AVG"
    # Some hits are the 1+ GB bulk 533-PGS file; only read the id col + the
    # one PGS column so the load is fast regardless of file width.
    with path.open() as fh:
        id_col = fh.readline().rstrip("\n").split("\t")[0]
    print(f"  pgs:  file={path.name}  col={avg_col}  reading ...", flush=True)
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[id_col, avg_col],
        dtype={id_col: str, avg_col: float},
        low_memory=False,
    )
    df = df.rename(columns={id_col: "person_id", avg_col: "pgs"})
    df["person_id"] = _canonical_id(df["person_id"])
    print(f"  pgs:  file={path.name}  col={avg_col}  n={len(df):,}", flush=True)
    return df


# --- OHDSI canonical disease extraction -------------------------------------
# Pulls the OHDSI Phenotype Library and filters cohorts to canonical
# disease phenotypes:
#   (a) isReferenceCohort = 1
#   (b) primary criterion is a ConditionOccurrence
#   (c) the cohort's concept set has exactly one include and zero excludes
#   (d) the include root descends from SNOMED 'Disease (disorder)'
# These four filters together yield ~235-240 single-SNOMED-root disease
# cohorts per OHDSI release. We then intersect with SNOMED_PGS_MAP and
# rank the intersection by case prevalence in the active CDR.

def ensure_phenotype_library() -> Path:
    """Download and extract OHDSI PhenotypeLibrary to PL_CACHE_DIR if missing."""
    cohorts_dir = PL_LOCAL_DIR / "inst" / "cohorts"
    if cohorts_dir.exists():
        return PL_LOCAL_DIR
    PL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ohdsi: downloading PhenotypeLibrary from {PL_ZIP_URL}")
    with urllib.request.urlopen(PL_ZIP_URL) as resp:
        data = resp.read()
    print(f"  ohdsi: {len(data)/1e6:.1f} MB; extracting to {PL_CACHE_DIR}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(PL_CACHE_DIR)
    if not cohorts_dir.exists():
        raise RuntimeError(f"OHDSI PhenotypeLibrary extraction failed: missing {cohorts_dir}")
    return PL_LOCAL_DIR


def _load_cohort_json(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("latin-1"))


def extract_ohdsi_canonical_disease_concepts(client: bigquery.Client, cdr: str) -> set[int]:
    """Return the OMOP concept_id set of canonical OHDSI disease phenotype roots."""
    pl_root = ensure_phenotype_library()
    json_dir = pl_root / "inst" / "cohorts"
    cohorts_csv = pl_root / "inst" / "Cohorts.csv"

    meta = pd.read_csv(cohorts_csv)
    meta["cohortId"] = meta["cohortId"].astype(int)
    meta["isReferenceCohort"] = meta["isReferenceCohort"].fillna(0).astype(int)
    ref_ids = sorted(meta.loc[meta["isReferenceCohort"] == 1, "cohortId"].tolist())
    print(f"  ohdsi: cohorts={len(meta)} reference={len(ref_ids)}")

    roots: set[int] = set()
    skipped = {"non_condition_primary": 0, "multi_include_or_exclude": 0, "parse": 0}
    for cid in ref_ids:
        path = json_dir / f"{cid}.json"
        try:
            j = _load_cohort_json(path)
        except Exception:
            skipped["parse"] += 1
            continue
        codeset_id = None
        for crit in j.get("PrimaryCriteria", {}).get("CriteriaList", []):
            if "ConditionOccurrence" in crit:
                codeset_id = crit["ConditionOccurrence"].get("CodesetId")
                break
        if codeset_id is None:
            skipped["non_condition_primary"] += 1
            continue
        cs = next((c for c in j.get("ConceptSets", []) if c.get("id") == codeset_id), None)
        if cs is None:
            skipped["parse"] += 1
            continue
        items = cs.get("expression", {}).get("items", [])
        incs = [it for it in items if not it.get("isExcluded")]
        excs = [it for it in items if it.get("isExcluded")]
        if len(incs) != 1 or len(excs) != 0:
            skipped["multi_include_or_exclude"] += 1
            continue
        roots.add(int(incs[0]["concept"]["CONCEPT_ID"]))
    print(f"  ohdsi: single-root condition cohorts={len(roots)} skipped={skipped}")

    # Filter (d): include-root must descend from SNOMED 'Disease (disorder)'.
    disease_root = lookup_snomed_code(client, cdr, SNOMED_DISEASE_CODE)
    roots_arr = sorted(roots)
    df = client.query(
        f"""
        WITH roots AS (SELECT concept_id FROM UNNEST(@roots) AS concept_id),
             dis   AS (SELECT descendant_concept_id AS concept_id
                       FROM `{cdr}.concept_ancestor`
                       WHERE ancestor_concept_id = @disease_root)
        SELECT r.concept_id, dis.concept_id IS NOT NULL AS is_disease
        FROM roots r LEFT JOIN dis USING (concept_id)
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("roots", "INT64", roots_arr),
            bigquery.ScalarQueryParameter("disease_root", "INT64", disease_root),
        ]),
    ).to_dataframe()
    disease_roots = set(df.loc[df["is_disease"], "concept_id"].astype(int).tolist())
    print(
        f"  ohdsi: canonical disease phenotypes={len(disease_roots)} "
        f"(dropped {len(roots) - len(disease_roots)} symptom-rooted)"
    )
    return disease_roots


def lookup_snomed_code(client: bigquery.Client, cdr: str, code: str) -> int:
    """Return the OMOP concept_id for a SNOMED standard concept by its concept_code."""
    sql = f"""
    SELECT concept_id
    FROM `{cdr}.concept`
    WHERE vocabulary_id = 'SNOMED' AND standard_concept = 'S'
      AND concept_code = @code
    ORDER BY concept_id
    LIMIT 1
    """
    rows = list(client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("code", "STRING", code),
        ]),
    ).result())
    if not rows:
        raise ValueError(f"no standard SNOMED concept with code {code!r}")
    return int(rows[0]["concept_id"])


def resolve_snomed_codes(client: bigquery.Client, cdr: str, codes: list[str]) -> pd.DataFrame:
    """Resolve SNOMED concept_codes to (concept_id, concept_name)."""
    df = client.query(
        f"""
        SELECT concept_code, concept_id, concept_name
        FROM `{cdr}.concept`
        WHERE vocabulary_id = 'SNOMED' AND standard_concept = 'S'
          AND concept_code IN UNNEST(@codes)
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("codes", "STRING", codes),
        ]),
    ).to_dataframe()
    df["concept_id"] = df["concept_id"].astype(int)
    return df


def rank_disease_concepts_by_prevalence(
    client: bigquery.Client, cdr: str, concept_ids: list[int]
) -> pd.DataFrame:
    """Distinct case count per ancestor concept_id, descending."""
    if not concept_ids:
        return pd.DataFrame(columns=["concept_id", "case_count"])
    df = client.query(
        f"""
        SELECT ca.ancestor_concept_id AS concept_id,
               COUNT(DISTINCT co.person_id) AS case_count
        FROM `{cdr}.condition_occurrence` AS co
        JOIN `{cdr}.concept_ancestor` AS ca
          ON ca.descendant_concept_id = co.condition_concept_id
        WHERE ca.ancestor_concept_id IN UNNEST(@ancestors)
        GROUP BY ca.ancestor_concept_id
        """,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ancestors", "INT64", concept_ids),
        ]),
    ).to_dataframe()
    df["concept_id"] = df["concept_id"].astype(int)
    df["case_count"] = df["case_count"].astype(int)
    return df.sort_values("case_count", ascending=False).reset_index(drop=True)


def select_runtime_diseases(client: bigquery.Client, cdr: str) -> dict[str, dict]:
    """Build the per-run diseases dict by intersecting OHDSI canonical disease
    phenotypes with SNOMED_PGS_MAP and taking the top TOP_N_DISEASES by case
    prevalence in the active CDR.
    """
    print("\n=== DISEASE SELECTION ===")
    print(f"  snomed_pgs_map_size={len(SNOMED_PGS_MAP)}  top_n={TOP_N_DISEASES}")
    mapped_codes = sorted(SNOMED_PGS_MAP.keys())
    resolved = resolve_snomed_codes(client, cdr, mapped_codes)
    resolved_codes = set(resolved["concept_code"].astype(str))
    missing = [c for c in mapped_codes if c not in resolved_codes]
    if missing:
        raise ValueError(
            f"SNOMED codes in SNOMED_PGS_MAP not resolvable in {cdr}.concept: {missing}"
        )
    code_to_id = dict(zip(resolved["concept_code"].astype(str), resolved["concept_id"]))
    code_to_name = dict(zip(resolved["concept_code"].astype(str), resolved["concept_name"]))
    print(f"  resolved {len(code_to_id)} SNOMED code(s) to OMOP concept_ids")

    canonical = extract_ohdsi_canonical_disease_concepts(client, cdr)

    mapped_ids = {int(code_to_id[c]): c for c in mapped_codes}
    survivors = {cid: code for cid, code in mapped_ids.items() if cid in canonical}
    dropped = [
        (code, code_to_name[code]) for cid, code in mapped_ids.items() if cid not in canonical
    ]
    print(f"  mapped ∩ OHDSI-canonical: {len(survivors)}/{len(mapped_ids)}")
    for code, name in dropped:
        print(f"    dropped (not OHDSI-canonical disease): {code} {name!r}")

    ranked = rank_disease_concepts_by_prevalence(client, cdr, list(survivors.keys()))
    counts = dict(zip(ranked["concept_id"], ranked["case_count"]))
    chosen_concept_ids: list[int] = []
    for cid in ranked["concept_id"].tolist():
        if cid in survivors:
            chosen_concept_ids.append(int(cid))
        if len(chosen_concept_ids) >= TOP_N_DISEASES:
            break

    diseases: dict[str, dict] = {}
    print(f"  selected top-{len(chosen_concept_ids)} by prevalence:")
    for cid in chosen_concept_ids:
        code = survivors[cid]
        cfg = SNOMED_PGS_MAP[code]
        slug = cfg["slug"]
        diseases[slug] = {
            "snomed_code": code,
            "concept_id": int(cid),
            "snomed_name": code_to_name[code],
            "pgs": cfg["pgs"],
            "case_count": int(counts.get(cid, 0)),
        }
        print(
            f"    {slug:<24}  concept_id={cid:>8}  cases={counts.get(cid, 0):>9,}  "
            f"pgs={cfg['pgs']}  ({code_to_name[code]})"
        )
    print("=== /DISEASE SELECTION ===\n")
    return diseases


# --- cases -----------------------------------------------------------------


def fetch_cases(client: bigquery.Client, cdr: str, ancestor_id: int) -> pd.DataFrame:
    """Per-case earliest qualifying condition date.

    Returns columns `person_id` (str) and `event_date` (datetime64). The event
    date is the earliest `condition_start_date` across any descendant concept,
    which is what age-as-time-scale survival wants as the failure time.
    """
    sql = f"""
    SELECT CAST(co.person_id AS STRING) AS person_id,
           MIN(co.condition_start_date) AS event_date
    FROM `{cdr}.condition_occurrence` AS co
    JOIN `{cdr}.concept_ancestor` AS ca
      ON ca.descendant_concept_id = co.condition_concept_id
    WHERE ca.ancestor_concept_id = @ancestor
    GROUP BY co.person_id
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("ancestor", "INT64", ancestor_id),
        ]),
    )
    df = job.to_dataframe()
    df["person_id"] = _canonical_id(df["person_id"])
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def fetch_person_times(client: bigquery.Client, cdr: str) -> pd.DataFrame:
    """Per-person birth date and observation-window bounds.

    Returns columns `person_id`, `birth_datetime`, `obs_start`, `obs_end`.
    `obs_start` is the earliest observation_period start; `obs_end` the latest
    end. Used to build age-as-time-scale entry/exit for survival.
    """
    sql = f"""
    SELECT CAST(p.person_id AS STRING) AS person_id,
           p.birth_datetime,
           MIN(op.observation_period_start_date) AS obs_start,
           MAX(op.observation_period_end_date)   AS obs_end
    FROM `{cdr}.person` AS p
    JOIN `{cdr}.observation_period` AS op USING(person_id)
    GROUP BY p.person_id, p.birth_datetime
    """
    df = client.query(sql).to_dataframe()
    df["person_id"] = _canonical_id(df["person_id"])
    df["birth_datetime"] = pd.to_datetime(df["birth_datetime"], utc=True).dt.tz_convert(None)
    df["obs_start"] = pd.to_datetime(df["obs_start"])
    df["obs_end"] = pd.to_datetime(df["obs_end"])
    print(f"  times: n={len(df):,}")
    return df


STATE_TO_CENSUS_REGION = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "MI": "Midwest", "OH": "Midwest",
    "WI": "Midwest", "IA": "Midwest", "KS": "Midwest", "MN": "Midwest",
    "MO": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "DE": "South", "FL": "South", "GA": "South", "MD": "South",
    "NC": "South", "SC": "South", "VA": "South", "DC": "South",
    "WV": "South", "AL": "South", "KY": "South", "MS": "South",
    "TN": "South", "AR": "South", "LA": "South", "OK": "South",
    "TX": "South",
    "AZ": "West", "CO": "West", "ID": "West", "MT": "West",
    "NV": "West", "NM": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "OR": "West",
    "WA": "West",
    "AS": "Territory", "GU": "Territory", "MP": "Territory", "PR": "Territory",
    "VI": "Territory",
}


def _clean_group_label(s: pd.Series) -> pd.Series:
    return (
        s.fillna("unknown")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace("", "unknown")
    )


# State of residence comes from `person_ext`; EHR site comes from the
# per-row clinical extension tables:
#
# - `person_ext.state_of_residence_concept_id` holds the PPI PIIState_<XX>
#   concept whose `concept_code` is `PIIState_<XX>`, or 2000000011 ("State
#   information suppressed for privacy") when the participant's state has
#   fewer than 200 enrollees. Curation writes it from `observation.
#   value_source_concept_id` at `observation_source_concept_id = 1585249`
#   (see all-of-us/curation create_person_ext_table.py).
#
# - `person_ext.src_id` is declared in the schema but is *not* reliably
#   populated in the CDR (AoU's own controlled-tier QC --
#   `controlled_tier_qc/check_controlled_tier_part2.py` -- identifies
#   EHR-consented persons by UNION'ing across the six per-row `*_ext`
#   tables, never via `person_ext`). The canonical EHR-site marker is
#   `<row>_ext.src_id` matching `(?i)EHR site` (e.g. "EHR site 100"), a
#   masked HPO identifier present on every EHR-uploaded row. We take the
#   modal site per person across `visit_occurrence_ext` (semantically the
#   "care site") and `condition_occurrence_ext` (broadest coverage for
#   EHR-consented participants with few recorded visits).
STATE_OBSERVATION_CONCEPT_ID = 1585249  # PII state-of-residence PPI concept


def fetch_person_context(client: bigquery.Client, cdr: str) -> pd.DataFrame:
    """Per-person OOD grouping context (state, census region, EHR site).

    State: `person_ext.state_of_residence_concept_id` joined to `concept`.
    Rows generalized for privacy (concept_id 2000000011, code !=
    `PIIState_<XX>`) fall through as "unknown".

    Care site: modal `src_id` per person from `visit_occurrence_ext`
    UNION ALL `condition_occurrence_ext`, restricted to values matching
    `(?i)EHR site`. Participants without EHR consent (no matching rows in
    either table) bucket as "unknown".
    """
    sql = f"""
    WITH per_person_site_evidence AS (
      SELECT
        CAST(vo.person_id AS STRING) AS person_id,
        ve.src_id,
        COUNT(*) AS weight
      FROM `{cdr}.visit_occurrence` AS vo
      JOIN `{cdr}.visit_occurrence_ext` AS ve USING (visit_occurrence_id)
      WHERE REGEXP_CONTAINS(ve.src_id, r'(?i)EHR site')
      GROUP BY 1, 2

      UNION ALL

      SELECT
        CAST(co.person_id AS STRING) AS person_id,
        ce.src_id,
        COUNT(*) AS weight
      FROM `{cdr}.condition_occurrence` AS co
      JOIN `{cdr}.condition_occurrence_ext` AS ce USING (condition_occurrence_id)
      WHERE REGEXP_CONTAINS(ce.src_id, r'(?i)EHR site')
      GROUP BY 1, 2
    ),
    site_totals AS (
      SELECT person_id, src_id, SUM(weight) AS n
      FROM per_person_site_evidence
      GROUP BY 1, 2
    ),
    dominant_site AS (
      SELECT person_id, src_id AS care_site_src_id
      FROM site_totals
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY person_id
        ORDER BY n DESC, src_id
      ) = 1
    )
    SELECT
      CAST(pe.person_id AS STRING) AS person_id,
      c.concept_code AS state_concept_code,
      ds.care_site_src_id AS care_site_src_id
    FROM `{cdr}.person_ext` AS pe
    LEFT JOIN `{cdr}.concept` AS c
      ON c.concept_id = pe.state_of_residence_concept_id
    LEFT JOIN dominant_site AS ds
      ON ds.person_id = CAST(pe.person_id AS STRING)
    """
    df = client.query(sql).to_dataframe()
    df["person_id"] = _canonical_id(df["person_id"])

    # PIIState_<XX> -> "XX"; anything else (incl. the 2000000011 generalization
    # whose code is "GeneralizedForPrivacy") becomes "UNKNOWN".
    code = df["state_concept_code"].fillna("").astype(str)
    state = code.str.extract(r"^PIIState_([A-Z]{2})$", expand=False).fillna("")
    df["state"] = np.where(state.ne(""), state.str.upper(), "UNKNOWN")
    df["census_region"] = df["state"].map(STATE_TO_CENSUS_REGION).fillna("unknown")
    # The SQL pre-filtered to matching `(?i)EHR site` values, so anything
    # present is a real masked HPO id; NULLs mean no EHR evidence found.
    src = df["care_site_src_id"].fillna("").astype(str).str.strip()
    df["care_site_group"] = np.where(src.ne(""), src, "unknown")

    # --- diagnostics --------------------------------------------------------
    # When LOSO downstream reports `groups=0`, the user needs to see *why*:
    # how many people are "unknown" on each axis, and what the actual
    # populated buckets look like.
    n = len(df)
    n_state_known = int((df["state"].astype(str).str.upper() != "UNKNOWN").sum())
    n_region_known = int(df["census_region"].astype(str).str.lower().ne("unknown").sum())
    n_care_known = int(df["care_site_group"].astype(str).str.lower().ne("unknown").sum())

    state_top = df.loc[df["state"].astype(str).str.upper() != "UNKNOWN", "state"].value_counts().head(8)
    region_top = df.loc[df["census_region"].astype(str).str.lower() != "unknown", "census_region"].value_counts()
    care_top = df.loc[df["care_site_group"].astype(str).str.lower() != "unknown", "care_site_group"].value_counts().head(8)

    print(f"  context: n={n:,}")
    print(
        f"    state    : known={n_state_known:,} ({100.0*n_state_known/max(n,1):.1f}%)  "
        f"distinct={df['state'].nunique():,}"
    )
    if not state_top.empty:
        print(f"      top states: {dict(state_top)}")
    print(
        f"    region   : known={n_region_known:,} ({100.0*n_region_known/max(n,1):.1f}%)  "
        f"distinct={df['census_region'].nunique():,}"
    )
    if not region_top.empty:
        print(f"      regions: {dict(region_top)}")
    print(
        f"    care_site: known={n_care_known:,} ({100.0*n_care_known/max(n,1):.1f}%)  "
        f"distinct={df['care_site_group'].nunique():,}"
    )
    if not care_top.empty:
        print(f"      top care sites: {dict(care_top)}")
    if n_state_known == 0 and n_care_known == 0:
        print("    [WARN] all LOSO context columns are 'unknown' -> no LOSO axes will run.")

    return df[["person_id", "state", "census_region", "care_site_group"]]


def load_genetic_ancestry_labels() -> pd.DataFrame:
    """Load AoU inferred genetic ancestry labels.

    Uses `ANCESTRY_PREDS_CACHE` if present; otherwise reads `ANCESTRY_PREDS_URI`
    directly via fsspec/gcsfs with the requester-pays + workspace-project
    storage options — the canonical AoU pattern (see
    `SauersML/ferromic` `phewas/iox.py:load_ancestry_labels`) — and caches the
    result locally for future runs. Any failure propagates and aborts the run.
    AoU's labels are reference-panel-derived categories (`AFR`, `AMR`, `EAS`,
    `EUR`, `MID`, `SAS`, and sometimes `OTH`), not self-reported race/ethnicity.
    """
    # Keep `pca_features` in the cache so `_load_pcs_from_aou` can reuse it.
    cache_cols = ["research_id", "ancestry_pred", "pca_features"]
    have_usable_cache = False
    if ANCESTRY_PREDS_CACHE.exists() and ANCESTRY_PREDS_CACHE.stat().st_size > 0:
        df = pd.read_csv(ANCESTRY_PREDS_CACHE, sep="\t", dtype=str)
        if "ancestry_pred" in df.columns:
            if "pca_features" not in df.columns:
                df["pca_features"] = ""
            df = df[["research_id", "ancestry_pred", "pca_features"]]
            have_usable_cache = True
        else:
            # Cache was written by _load_pcs_from_aou before it learned to
            # include ancestry_pred; refetch from the URI to repopulate.
            print(f"  cache {ANCESTRY_PREDS_CACHE} lacks ancestry_pred; refetching")
    if not have_usable_cache:
        uri = _resolve_ancestry_preds_uri()
        print(f"  gcsfs GET {uri} -> {ANCESTRY_PREDS_CACHE}")
        df = pd.read_csv(
            uri,
            sep="\t",
            usecols=cache_cols,
            dtype=str,
            storage_options={
                "project": os.environ["GOOGLE_PROJECT"],
                "requester_pays": True,
            },
        )
        ANCESTRY_PREDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ANCESTRY_PREDS_CACHE, sep="\t", index=False)
    labels = _clean_group_label(df["ancestry_pred"]).str.upper()
    # MID is too small to support its own LOSO fold; fold it into OTH.
    labels = labels.where(labels != "MID", "OTH")
    out = pd.DataFrame({
        "person_id": _canonical_id(df["research_id"]),
        "ancestry_category": labels,
    })
    counts = out["ancestry_category"].value_counts().sort_index()
    print(
        "  ancestry: "
        + " ".join(f"{k}={v:,}" for k, v in counts.items())
    )
    return out


# --- model -----------------------------------------------------------------

@dataclass(frozen=True)
class SurvivalMetricStats:
    n: int
    n_events: int
    median_exit_age: float
    c_ipcw: float
    c_ipcw_ci_low: float
    c_ipcw_ci_high: float
    dynamic_auc_mean: float
    ibs: float
    null_ibs: float
    r2_ibs: float
    tau: float
    ibs_start: float
    ibs_stop: float
    n_times: int


@dataclass(frozen=True)
class CoxFit:
    names: list[str]
    coefs: dict[str, float]
    result: object


@dataclass(frozen=True)
class BinaryStats:
    n: int
    n_events: int
    prevalence: float
    auroc: float
    average_precision: float
    log_loss: float
    brier: float
    nagelkerke_r2: float
    liability_r2: float


def survival_model_columns(num_pcs: int) -> list[str]:
    """Columns the survival marginal-slope fit and its predict path both see.

    Single source of truth: `fit_marginal_slope` selects this subset of the
    training frame before handing it to `gamfit.fit`, and prediction helpers
    select the same subset before calling `model.predict`. Keeping fit-time and
    predict-time schemas in one place avoids the class of bug where they
    drift (e.g., predict missing `entry_age`/`exit_age` after a formula
    change). `event` is included even at predict time: gamfit's predict
    requires only that required columns be present, tolerates extras, and
    both train and test frames already carry `event`.
    """
    return ["entry_age", "exit_age", "event", "sex", "prs_z"] + [
        f"PC{i+1}" for i in range(num_pcs)
    ]


def print_fit_frame_diagnostics(df: pd.DataFrame, label: str) -> None:
    print(f"  fit_frame[{label}]: rows={len(df):,} cols={len(df.columns)}")
    print(f"  fit_frame[{label}]: columns={list(df.columns)}")
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    print(f"  fit_frame[{label}]: duplicate_columns={duplicate_columns or '<none>'}")
    for col in df.columns:
        series = df[col]
        nulls = int(series.isna().sum())
        dtype = str(series.dtype)
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
            finite = np.isfinite(values)
            finite_n = int(finite.sum())
            if finite_n:
                print(
                    f"  fit_frame[{label}].{col}: dtype={dtype} null={nulls} "
                    f"finite={finite_n:,}/{len(series):,} "
                    f"min={float(np.nanmin(values)):.6g} "
                    f"max={float(np.nanmax(values)):.6g} "
                    f"mean={float(np.nanmean(values)):.6g}"
                )
            else:
                print(
                    f"  fit_frame[{label}].{col}: dtype={dtype} null={nulls} "
                    f"finite=0/{len(series):,}"
                )
        else:
            top = series.astype(str).value_counts(dropna=False).head(5).to_dict()
            print(f"  fit_frame[{label}].{col}: dtype={dtype} null={nulls} top={top}")


def pc_duchon_term(num_pcs: int) -> str:
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    # Pure isotropic scale-free Duchon over the leading PCs. Both
    # `length_scale` and `scale_dims` are omitted on purpose:
    #
    # * Omitting `length_scale` selects the polyharmonic spectrum
    #   `||w||^(2(p+s))` (basis.rs: `pure = length_scale.is_none()`) -- the
    #   canonical scale-free Duchon, recommended in docs/formulas.md for
    #   PC-space smoothing.
    # * Omitting `scale_dims` keeps `aniso_log_scales=None`, i.e. the kernel
    #   uses the isotropic distance `r = ||x - c||` (basis.rs:1307,2810).
    #   On the prior run with `scale_dims=true` the optimized per-axis
    #   kappas landed at 0.88 / 1.04 / 1.09 -- <15% anisotropy, so we're
    #   giving up almost nothing in expressivity, and we eliminate the
    #   `build_psi_hyper_coords` / anisotropic-kappa code path where the
    #   outer BFGS gradient previously exploded (|g| jumped 6 orders of
    #   magnitude with the objective flat). Outer optimizer dim drops from
    #   rho_dim+log_kappa_dim = 8+6 = 14 down to 8+0 = 8.
    #
    # At dim=NUM_PCS=3, order=1, max_op=2 (stiffness active),
    # `resolve_duchon_orders` returns (Linear, s=1) -- the CPD gate
    # `2s < d` is satisfied without escalation, so the polynomial nullspace
    # stays at d+1 = 4 cols, well below DUCHON_CENTERS.
    return f"duchon({pcs}, centers={DUCHON_CENTERS}, order=1)"


def pc_marginal_surface_term_binary(num_pcs: int) -> str:
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    # Polyharmonic Duchon marginal surface, sharing one basis with the logslope
    # channel (the scientific target: PC-varying PRS log-OR).
    #
    # KNOWN gam-side issue in the probit binary path: the Duchon penalty leaves
    # a polynomial nullspace unpenalized. Its CONSTANT direction collides with
    # the GLM marginal block's explicit intercept (the audit FATAL of #531,
    # closed); orthogonalization now clears that, but the surviving LINEAR PC
    # directions are unpenalized AND unconstrained on a probit fit, so the outer
    # REML/ARC solve fails to converge (β→∞, λ→0) -- SauersML/gam#754. The escape
    # hatch is a strictly-PD Matern marginal (no polynomial nullspace); we keep
    # Duchon here pending the gam-side fix so both channels share a basis.
    return f"duchon({pcs}, centers={DUCHON_CENTERS})"


def fit_marginal_slope(train_df: pd.DataFrame, num_pcs: int):  # -> gamfit.Model
    """Survival marginal-slope GAM with joint Duchon over PCs in both the
    baseline hazard surface and the log-slope (log-HR) channel; sex linear;
    prs_z is the latent score (z_column) so its hazard ratio varies in PC space.
    No `linkwiggle()` on either channel: link-deviation and logslope score-warp
    both hang the BMS cell-moment path (SauersML/gam#683).

    Age-as-time-scale: `Surv(entry_age, exit_age, event)` is left-truncated at
    each person's age at AoU observation start.

    Re-running this on identical training data is near-instant thanks to
    gamfit's persistent warm-start cache at `XDG_CACHE_HOME/gam/warm/v1/`, which
    auto-resumes the outer (rho) and inner (beta) iterates from the prior
    fit. The cache is keyed on (gamfit version, n_rows, n_cols, likelihood,
    link, y, weights, ...), so the only refits are: first fit on this data,
    or first fit after a gamfit version bump.
    """
    import gamfit  # lazy: lets the linear baseline import this module without dragging gamfit in
    duchon = pc_duchon_term(num_pcs)
    # No linkwiggle (link-deviation) or logslope linkwiggle (score-warp): both hang
    # bernoulli/survival marginal-slope in the BMS cell-moment path (SauersML/gam#683).
    logslope_formula = duchon
    formula = f"Surv(entry_age, exit_age, event) ~ {duchon} + sex"
    cols = survival_model_columns(num_pcs)
    model_df = train_df[cols]
    print("  fit_spec: family=survival marginal-slope")
    print(f"  fit_spec: formula={formula!r}")
    print(f"  fit_spec: z_column='prs_z'  logslope_formula={logslope_formula!r}")
    print(f"  fit_spec: num_pcs={num_pcs}  duchon_centers={DUCHON_CENTERS}  n_train={len(train_df)}")
    print(f"  fit_spec: gamfit={gamfit.__version__}")
    print_fit_frame_diagnostics(model_df, "survival_marginal_slope_train")
    print_cuda_process_diagnostics("  fit_spec:")
    return gamfit.fit(
        model_df,
        formula,
        survival_likelihood="marginal-slope",
        z_column="prs_z",
        logslope_formula=logslope_formula,
    )


# Binary-path age handling — matched basis across all four models so
# age is a nuisance held fixed, not a source of GAM-vs-baseline variance.
#
# Disease incidence is non-linear in current_age (~exponential past
# midlife), so every model needs a non-linear age adjustment. Both the
# Bernoulli marginal-slope GAM and the three logistic baselines consume
# the same simple quadratic in standardized current_age
# (`current_age_z`, `current_age_z2`), with `entry_age_z` and
# `birth_year_z` (year-of-birth cohort control) linear.
#
# A quadratic — not a spline. A `cr(current_age, df=4)` cubic-regression
# basis is a partition of unity (its columns sum to 1), so it aliases the
# model intercept and gamfit's identifiability audit refuses the fit
# (marginal block range_rank 15/16, intra-block deficient). A plain
# quadratic in z-scored age is linearly independent of the intercept,
# well-conditioned, and captures the midlife-onward curvature that matters
# here without spending spline degrees of freedom.
#
# With age matched, every GAM-vs-baseline delta reflects PC/PRS treatment and
# the requested flexible marginal-slope deviations: a smooth PC surface and a
# PC-varying PRS log-OR (gamfit's `logslope_formula`). Link-deviation and
# score-warp (`linkwiggle()`) are omitted — both hang (SauersML/gam#683).
#
# Both the GAM and the logistic baselines consume these same columns, so
# age is a fully matched nuisance term and any GAM-vs-baseline delta
# reflects PC/PRS treatment only.
BASELINE_AGE_FEATURES = [
    "entry_age_z",
    "current_age_z",
    "current_age_z2",
    "birth_year_z",
]
GAM_AGE_FEATURES = list(BASELINE_AGE_FEATURES)


def _add_binary_age_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the train-only standardized age representations the binary path
    needs.

    Produces these columns on train and test:

    * `entry_age_z` — linear z-scored entry age (train mean/std). Used by
      every model.
    * `current_age_z`, `current_age_z2` — current age z-scored on train
      (mean/std), and its square: a simple quadratic age adjustment shared
      by both the logistic baselines and the marginal-slope GAM. Not a
      spline: a `cr()` cubic-regression basis is a partition of unity whose
      columns sum to 1, aliasing the model intercept (gamfit's
      identifiability audit then refuses the fit). A quadratic in
      standardized age is independent of the intercept and well-conditioned.
    * `birth_year_z` — year of birth (from `birth_datetime`), z-scored on
      train: a cohort/period control. Identifiable alongside the age terms
      because observation windows vary across people, so it is not an exact
      linear function of entry/current age.

    Why these columns and not `followup_years`/`exit_age`:

    `current_age = (obs_end - birth)/365.25` and `entry_age = (obs_start -
    birth)/365.25` are *label-independent* — they depend only on
    enrollment dates and birth date, never on event status. `exit_age` (=
    event onset for cases, obs_end for censors) and `followup_years =
    exit_age - entry_age` are *endogenous to the outcome*: for cases they
    stop at diagnosis, for censors they run to obs_end. Using either as a
    binary covariate would leak the label and inflate every metric.
    """
    train = train.copy()
    test = test.copy()
    for col in ("entry_age", "current_age", "birth_datetime"):
        if col not in train.columns:
            raise ValueError(
                f"binary path requires column {col!r}; main() must compute "
                f"it before splitting"
            )
    # Year of birth as a cohort/period control — the calendar year captures
    # the cohort effect; fractional-year resolution is unnecessary.
    train["birth_year"] = train["birth_datetime"].dt.year.astype("float64")
    test["birth_year"] = test["birth_datetime"].dt.year.astype("float64")
    # z-score each age + birth year on train (mean/std), applied to test with
    # train stats.
    for src, dst in (
        ("entry_age", "entry_age_z"),
        ("current_age", "current_age_z"),
        ("birth_year", "birth_year_z"),
    ):
        mean = float(train[src].mean())
        std = float(train[src].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"training column {src!r} has zero or invalid variance")
        train[dst] = (train[src] - mean) / std
        test[dst] = (test[src] - mean) / std
    # Quadratic curvature in current age (the nuisance the spline used to carry).
    train["current_age_z2"] = train["current_age_z"] ** 2
    test["current_age_z2"] = test["current_age_z"] ** 2
    return train, test


def binary_model_columns(num_pcs: int) -> list[str]:
    return ["event", "sex", "prs_z", *GAM_AGE_FEATURES] + [
        f"PC{i+1}" for i in range(num_pcs)
    ]


def binary_predict_columns(num_pcs: int) -> list[str]:
    return ["sex", "prs_z", *GAM_AGE_FEATURES] + [
        f"PC{i+1}" for i in range(num_pcs)
    ]


def fit_binary_marginal_slope(train_df: pd.DataFrame, num_pcs: int):  # -> gamfit.Model
    """Bernoulli marginal-slope GAM with a matched-basis age nuisance.

    Age enters as the same quadratic in standardized current age
    (`current_age_z`, `current_age_z2`, plus linear `entry_age_z`) used by
    the logistic baselines, fed in as plain linear features. Age is a
    nuisance we adjust for, not a quantity we want gamfit to spend REML
    degrees of freedom estimating, and matching the basis makes every
    GAM-vs-baseline delta attributable to PC/PRS treatment alone. PCs
    enter via `matern(PCs)` and the `prs_z` log-OR varies as the same
    `matern(PCs)` surface (its `logslope_formula`) — matched basis, with
    matern required on the marginal channel for probit identifiability
    (gam#531). No `linkwiggle()` on either channel: link-deviation and
    logslope score-warp both hang the BMS cell-moment path (SauersML/gam#683).
    """
    import gamfit  # lazy

    age_terms = " + ".join(BASELINE_AGE_FEATURES)
    # Marginal nuisance surface uses a strictly-PD Matern (no polynomial
    # nullspace) to stay identifiable against the probit intercept; the
    # logslope target keeps the polyharmonic Duchon. See gam#531 and
    # pc_marginal_surface_term_binary().
    marginal_surface = pc_marginal_surface_term_binary(num_pcs)
    # Matched matern logslope (matern marginal is required for probit identifiability,
    # gam#531 — so duchon-marginal is out, hence matern on both channels). No linkwiggle
    # link-deviation and no logslope linkwiggle score-warp: both hang the BMS
    # cell-moment path (SauersML/gam#683).
    logslope_formula = marginal_surface
    formula = f"event ~ {marginal_surface} + sex + {age_terms}"
    cols = binary_model_columns(num_pcs)
    print("  binary_fit_spec: family=bernoulli-marginal-slope  link=probit")
    print(f"  binary_fit_spec: formula={formula!r}")
    print(f"  binary_fit_spec: z_column='prs_z'  logslope_formula={logslope_formula!r}")
    print(
        f"  binary_fit_spec: num_pcs={num_pcs}  duchon_centers={DUCHON_CENTERS}  "
        f"n_train={len(train_df)}"
    )
    print(f"  binary_fit_spec: gamfit={gamfit.__version__}")
    return gamfit.fit(
        train_df[cols],
        formula,
        family="bernoulli-marginal-slope",
        link="probit",
        z_column="prs_z",
        logslope_formula=logslope_formula,
    )


def z_norm2(
    train_pgs: np.ndarray,
    train_pcs: np.ndarray,
    test_pgs: np.ndarray,
    test_pcs: np.ndarray,
    pgs_id: str = "PGS",
) -> tuple[np.ndarray, np.ndarray]:
    """pgscatalog-calc Z_norm2 via `pgs_adjust` (mean+var, 2-step, no pop labels).

    Calls the upstream `pgscatalog.calc.lib._ancestry.tools.pgs_adjust` so the
    adjustment is bit-for-bit identical to the pgsc_calc / eMERGE pipeline.
    Regressions are fit on train (`ref_df`) and applied to both train and test
    (`target_df`) with train coefficients only — no test leakage.

    The pgs_adjust API requires `pop` columns even for continuous-ancestry
    methods that ignore them, so we pass a constant dummy label. The Z_norm2
    math itself does not consult them.
    """
    from pgscatalog.calc.lib._ancestry.tools import pgs_adjust  # lazy: heavy import

    n_pcs = train_pcs.shape[1]
    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]

    def _frame(pgs: np.ndarray, pcs: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(pcs, columns=pc_cols)
        df["pop"] = "all"
        df[pgs_id] = pgs
        return df

    ref_df = _frame(train_pgs, train_pcs)
    target_df = _frame(
        np.concatenate([train_pgs, test_pgs]),
        np.vstack([train_pcs, test_pcs]),
    )
    kwargs = dict(
        ref_df=ref_df,
        scorecols=[pgs_id],
        ref_pop_col="pop",
        target_pop_col="pop",
        use_method=["mean+var"],
        norm2_2step=True,
        n_pcs=n_pcs,
    )
    adj_train, adj_target, _ = pgs_adjust(target_df=target_df, **kwargs)
    z_col = f"Z_norm2|{pgs_id}"
    return (
        adj_train[z_col].to_numpy(),
        adj_target[z_col].to_numpy()[len(train_pgs):],
    )


def fit_baseline_cox(
    entry: np.ndarray,
    exit_: np.ndarray,
    event: np.ndarray,
    X: np.ndarray,
    names: list[str],
) -> CoxFit:
    """Generic left-truncated Cox PH baseline.

    `X` is an `(n, p)` covariate matrix; `names` labels its columns. Returns
    `{name: beta}` for every column.

    This script calls it twice per fold so the GAM is benchmarked against
    *two* sensible PRS-+-PC adjustment strategies:
      A. `Z_norm2 + sex`                           -- PCs folded into the
         PRS via train-only mean+var standardization (pgsc_calc style).
      B. `prs_z + sex + PC1 + ... + PC_NUM_PCS`    -- raw train-z-scored
         PRS plus PCs as linear additive Cox covariates.
    Both share `+ sex` with the GAM so no model has an extra free covariate;
    they differ only in how PC structure enters the score. The GAM's
    `duchon(PC1..PC_NUM_PCS)` is the third, smooth, alternative.

    Uses `statsmodels.duration.hazard_regression.PHReg` -- the only fast
    Python Cox PH that natively supports left-truncation via `entry=`.
    """
    from statsmodels.duration.hazard_regression import PHReg  # lazy

    result = PHReg(
        endog=np.asarray(exit_, dtype=np.float64),
        exog=np.asarray(X, dtype=np.float64),
        status=np.asarray(event, dtype=np.float64),
        entry=np.asarray(entry, dtype=np.float64),
        ties="breslow",
    ).fit(method="newton", maxiter=30, tol=1e-5, disp=0)
    if len(names) != len(result.params):
        raise ValueError(
            f"names length {len(names)} != coef length {len(result.params)}"
        )
    return CoxFit(
        names=list(names),
        coefs={n: float(p) for n, p in zip(names, result.params)},
        result=result,
    )


def fit_logit_binary(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict[str, object]:
    import statsmodels.api as sm  # lazy

    X_const = sm.add_constant(np.asarray(X, dtype=np.float64), has_constant="add")
    result = sm.GLM(
        np.asarray(y, dtype=np.float64),
        X_const,
        family=sm.families.Binomial(),
    ).fit(maxiter=100)
    coefs = {"const": float(result.params[0])}
    coefs.update({n: float(p) for n, p in zip(names, result.params[1:])})
    return {"result": result, "names": list(names), "coefs": coefs}


def predict_logit_binary(fit: dict[str, object], X: np.ndarray) -> np.ndarray:
    import statsmodels.api as sm  # lazy

    result = fit["result"]
    X_const = sm.add_constant(np.asarray(X, dtype=np.float64), has_constant="add")
    return np.asarray(result.predict(X_const), dtype=float)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Average-rank tiebreaker (positions 1..n; ties get the mean of their
    positions). Used by the Sun & Xu 2014 fast-DeLong algorithm.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def fast_delong_auc(
    scores: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong — O(n log n) AUROC + covariance.

    Returns:
        aucs:  shape (K,) per-model AUROC.
        cov:   shape (K, K) covariance matrix of AUROCs across models on
               the *same* held-out rows. From this:
                 SE(AUC_k)        = sqrt(cov[k, k])
                 Var(AUC_k - AUC_j) = cov[k, k] + cov[j, j] - 2*cov[k, j]
               i.e. paired Δ AUROC CIs come straight from `cov`.

    `scores` may be 1-D (a single model) or 2-D (one row per model). All
    models must be scored on the same `y`. Ties are handled by mid-rank.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None, :]
    K, n = scores.shape
    y = np.asarray(y).astype(bool)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    m = int(pos.size)
    n0 = int(neg.size)
    if m < 1 or n0 < 1:
        return np.full(K, float("nan")), np.full((K, K), float("nan"))

    V10 = np.empty((K, m), dtype=np.float64)
    V01 = np.empty((K, n0), dtype=np.float64)
    aucs = np.empty(K, dtype=np.float64)
    for k in range(K):
        sp = scores[k][pos]
        sn = scores[k][neg]
        rx = _midrank(sp)
        ry = _midrank(sn)
        rall = _midrank(np.concatenate([sp, sn]))
        tx = rall[:m]
        ty = rall[m:]
        V10[k] = (tx - rx) / n0
        V01[k] = 1.0 - (ty - ry) / m
        aucs[k] = V10[k].mean()

    if K == 1:
        s10 = float(np.var(V10[0], ddof=1)) if m > 1 else 0.0
        s01 = float(np.var(V01[0], ddof=1)) if n0 > 1 else 0.0
        cov = np.array([[s10 / m + s01 / n0]], dtype=np.float64)
    else:
        S10 = np.cov(V10, ddof=1) if m > 1 else np.zeros((K, K))
        S01 = np.cov(V01, ddof=1) if n0 > 1 else np.zeros((K, K))
        cov = S10 / m + S01 / n0
    return aucs, cov


def _extract_binary_mean(prediction: object) -> np.ndarray:
    """Coerce a gamfit bernoulli-marginal-slope `model.predict(...)` result to a
    1-D probability vector.

    Why: gamfit 0.1.124's `shape_prediction_response` only takes its 1-D
    early-return branch when the predict payload's `model_class` equals
    `"bernoulli marginal-slope"`. The Rust `PredictionPayload` does not emit
    `model_class`, so the Python layer falls back to the serde-kebab-case
    `model_kind` from the saved payload (`"marginal-slope"`), misses the
    branch, and returns a 2-column pandas DataFrame containing both `eta` and
    `mean`. We want just `mean`.
    """
    import pandas as pd

    if isinstance(prediction, pd.DataFrame):
        return np.asarray(prediction["mean"].to_numpy(), dtype=float)
    arr = np.asarray(prediction, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 2:
        # ordered_prediction_column_values puts "eta" before "mean"
        return arr[:, 1]
    raise ValueError(
        f"unexpected binary-GAM prediction shape {arr.shape}; expected 1-D or (n, 2)"
    )


def binary_stats(y: np.ndarray, p: np.ndarray, population_prevalence: float | None) -> BinaryStats:
    from scipy.stats import norm
    from sklearn.metrics import (  # lazy: only needed after binary predictions exist
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    prevalence = float(y.mean())
    ll_model = -float(log_loss(y, p, labels=[0, 1], normalize=False))
    null_p = np.repeat(prevalence, len(y))
    ll_null = -float(log_loss(y, null_p, labels=[0, 1], normalize=False))
    cox_snell = 1.0 - np.exp((2.0 / len(y)) * (ll_null - ll_model))
    max_cox_snell = 1.0 - np.exp((2.0 / len(y)) * ll_null)
    nagelkerke = cox_snell / max_cox_snell if max_cox_snell > 0 else float("nan")
    liability_r2 = float("nan")
    if population_prevalence is not None and 0.0 < population_prevalence < 1.0 and 0.0 < prevalence < 1.0:
        z = float(norm.pdf(norm.ppf(population_prevalence)))
        if z > 0.0:
            liability_r2 = float(
                nagelkerke
                * population_prevalence**2
                * (1.0 - population_prevalence) ** 2
                / (z**2 * prevalence * (1.0 - prevalence))
            )
    return BinaryStats(
        n=int(len(y)),
        n_events=int(y.sum()),
        prevalence=prevalence,
        auroc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        nagelkerke_r2=float(nagelkerke),
        liability_r2=float(liability_r2),
    )


def fmt_binary(stats: BinaryStats) -> str:
    return (
        f"AUROC={stats.auroc:.4f}  AP={stats.average_precision:.4f}  "
        f"logloss={stats.log_loss:.4f}  Brier={stats.brier:.4f}  "
        f"Nagelkerke_R2={stats.nagelkerke_r2:.4f}  "
        f"liability_R2={stats.liability_r2:.4f}"
    )


def _binary_fields(prefix: str, stats: BinaryStats) -> dict[str, float | int]:
    return {
        f"{prefix}_n": stats.n,
        f"{prefix}_events": stats.n_events,
        f"{prefix}_prevalence": stats.prevalence,
        f"{prefix}_auroc": stats.auroc,
        f"{prefix}_average_precision": stats.average_precision,
        f"{prefix}_log_loss": stats.log_loss,
        f"{prefix}_brier": stats.brier,
        f"{prefix}_nagelkerke_r2": stats.nagelkerke_r2,
        f"{prefix}_liability_r2": stats.liability_r2,
    }


def gam_survival_outputs(
    model,
    df: pd.DataFrame,
    num_pcs: int,
    followup_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row risk and conditional survival on a follow-up time grid.

    The GAM is fit on `Surv(entry_age, exit_age, event)`, i.e. left-truncated
    age-as-time scale. sksurv's IPCW machinery ignores entry times, so we
    convert age-scale predictions to follow-up scale before measuring
    discrimination/calibration. The right quantity for each individual is the
    conditional survival

        S_cond_i(s | entry_i) = P(T_i > entry_i + s | T_i > entry_i)
                              = S_i(entry_i + s) / S_i(entry_i),

    paired against the right-censored follow-up time ``s = exit_i - entry_i``.

    Implementation: gamfit's ``survival_at`` takes one 1-D age grid and
    returns S(age|x) per row, so we batch-evaluate on a dense common age
    grid covering every per-row prediction window, then per row linearly
    interpolate to ages ``entry_i`` and ``entry_i + s_k``.

    Returns ``(risk, surv_cond, kept)``:
      * ``risk = -log S_cond_i(s_K)``  — monotone-in-hazard ranking score for
        IPCW C / dynamic AUC; same ordering as the cumulative hazard accrued
        during the follow-up window up to ``s_K``.
      * ``surv_cond`` shape ``(n_kept, len(followup_times))``.
      * ``kept`` is a bool mask over input rows that survived gamfit's
        predict guard; all callers downstream must subset by ``kept``.
    """
    from gamfit._exceptions import PredictionError  # lazy

    cols = survival_model_columns(num_pcs)
    n = len(df)
    entry = df["entry_age"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(entry)):
        raise ValueError("entry_age has non-finite values")
    if followup_times.size < 1:
        raise ValueError("followup_times must have at least one point")
    kept = np.ones(n, dtype=bool)

    # Dense common age grid spans every per-row prediction window
    # (entry_i, entry_i + s_K). 8x oversampling of the requested follow-up
    # grid keeps per-row interpolation error in S well below ~1e-4 on the
    # gamfit-emitted survival curve.
    age_lo = float(np.min(entry))
    age_hi = float(np.max(entry) + float(followup_times[-1]))
    common_ages = np.linspace(age_lo, age_hi, max(200, 8 * len(followup_times)))
    S_age = np.full((n, common_ages.size), np.nan, dtype=np.float64)

    def _predict_slice(start: int, end: int) -> None:
        if start >= end:
            return
        sub = df.iloc[start:end]
        try:
            pred = model.predict(sub[cols])
        except PredictionError as exc:
            if end - start == 1:
                # Single offender: drop it.
                kept[start] = False
                # Don't log the person_id (individual-level); the positional
                # index is enough for diagnostics across reruns on the same
                # split.
                print(f"  predict: dropping row at split-position {start} — {exc}")
                return
            mid = (start + end) // 2
            _predict_slice(start, mid)
            _predict_slice(mid, end)
            return
        S_age[start:end, :] = np.asarray(pred.survival_at(common_ages), dtype=np.float64)

    _predict_slice(0, n)

    if not kept.all():
        print(f"  predict: dropped {int((~kept).sum())} of {n} rows that tripped the guard")

    keep_idx = np.flatnonzero(kept)
    if keep_idx.size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty((0, len(followup_times)), dtype=np.float64),
            kept,
        )
    S_age_kept = S_age[keep_idx]
    entry_kept = entry[keep_idx]

    # Per-row interp to (entry_i, entry_i + s_1, ..., entry_i + s_K).
    S_cond = np.empty((keep_idx.size, len(followup_times)), dtype=np.float64)
    for i in range(keep_idx.size):
        S_at_entry = float(np.interp(entry_kept[i], common_ages, S_age_kept[i]))
        denom = max(S_at_entry, 1e-12)
        S_target = np.interp(entry_kept[i] + followup_times, common_ages, S_age_kept[i])
        S_cond[i] = np.clip(S_target / denom, 0.0, 1.0)

    risk = -np.log(np.clip(S_cond[:, -1], 1e-12, 1.0))
    return risk, S_cond, kept


def _surv_array(event: np.ndarray, exit_: np.ndarray):
    from sksurv.util import Surv  # lazy

    return Surv.from_arrays(
        event=np.asarray(event, dtype=bool),
        time=np.asarray(exit_, dtype=np.float64),
    )


def _survival_metric_times(
    train_followup: np.ndarray,
    test_followup: np.ndarray,
    test_event: np.ndarray | None = None,
) -> np.ndarray:
    """Quantile-spaced follow-up grid (years from entry) for sksurv metrics.

    All sksurv estimators consume right-censored data with no entry time, so
    age-scale left-truncated fits must be reduced to follow-up scale
    (``s = exit_age - entry_age``) before discrimination/calibration is
    measured. The grid lives strictly inside the intersection of train and
    test follow-up support, where the IPCW censoring estimator
    ``G(s) = P(C > s)`` stays finite. Quantile-spaced at test event
    follow-ups so the Brier and AUC estimators integrate against a
    well-supported axis.
    """
    train_followup = np.asarray(train_followup, dtype=np.float64)
    test_followup = np.asarray(test_followup, dtype=np.float64)
    finite_train = train_followup[np.isfinite(train_followup) & (train_followup > 0)]
    finite_test = test_followup[np.isfinite(test_followup) & (test_followup > 0)]
    if finite_train.size < 2 or finite_test.size < 2:
        raise ValueError("survival metrics need at least two finite positive follow-ups in train and test")
    lo = float(np.min(finite_test))
    hi = min(float(np.max(finite_train)), float(np.max(finite_test)))
    lo = float(np.nextafter(lo, np.inf))
    hi = float(np.nextafter(hi, -np.inf))
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"invalid follow-up grid: lower={lo}, upper={hi}")
    if test_event is None:
        spine = finite_test
    else:
        event_mask = np.asarray(test_event, dtype=bool)
        # event_mask aligns with the original test_followup; recover positions.
        all_idx = np.isfinite(test_followup) & (test_followup > 0)
        spine = test_followup[all_idx & event_mask]
    spine = np.unique(spine[(spine > lo) & (spine < hi)])
    if spine.size < 2:
        return np.linspace(lo, hi, 2)
    max_times = max(2, int(np.sqrt(finite_test.size)))
    if spine.size <= max_times:
        return spine
    return np.unique(np.quantile(spine, np.linspace(0.0, 1.0, max_times)))


def cox_survival_matrix(
    fit: CoxFit,
    X: np.ndarray,
    entry_age: np.ndarray,
    followup_times: np.ndarray,
) -> np.ndarray:
    """Conditional S_cond_i(s | entry_i) on a follow-up time grid.

    The statsmodels PHReg fit gives back the Breslow baseline cumulative
    hazard ``H0(age)`` at the linear predictor 0. For individual i with
    covariate vector x_i, the conditional cumulative hazard accrued during
    the follow-up window is

        H_cond_i(s) = (H0(entry_i + s) - H0(entry_i)) * exp(beta . x_i)

    and the conditional survival is ``exp(-H_cond_i(s))``. This is the
    canonical way to map a left-truncated age-scale Cox fit onto the
    follow-up-from-entry scale that sksurv's right-censored metric
    estimators expect.
    """
    base = fit.result.baseline_cumulative_hazard
    if not base:
        raise ValueError("Cox fit did not expose a baseline cumulative hazard")
    base_t = np.asarray(base[0][0], dtype=np.float64)
    base_H = np.asarray(base[0][1], dtype=np.float64)
    entry = np.asarray(entry_age, dtype=np.float64)
    follow = np.asarray(followup_times, dtype=np.float64)
    H0_entry = np.interp(entry, base_t, base_H, left=0.0, right=float(base_H[-1]))
    target_ages = entry[:, None] + follow[None, :]
    H0_target = np.interp(
        target_ages.ravel(),
        base_t, base_H, left=0.0, right=float(base_H[-1]),
    ).reshape(target_ages.shape)
    beta = np.asarray([fit.coefs[n] for n in fit.names], dtype=np.float64)
    eta = np.asarray(X, dtype=np.float64) @ beta
    H_cond = np.clip((H0_target - H0_entry[:, None]) * np.exp(eta)[:, None], 0.0, None)
    return np.exp(-H_cond)


def null_survival_matrix(
    train: pd.DataFrame,
    n_rows: int,
    followup_times: np.ndarray,
) -> np.ndarray:
    """KM null reference S(s) on follow-up time, broadcast to ``n_rows`` rows.

    KM is fit on training (event, follow-up) — no covariates — so it is the
    proper no-skill survival curve for the Brier null reference against which
    each model's conditional S_cond_i(s) is compared.
    """
    from sksurv.nonparametric import SurvivalFunctionEstimator

    train_f = (train["exit_age"] - train["entry_age"]).to_numpy(dtype=np.float64)
    estimator = SurvivalFunctionEstimator().fit(
        _surv_array(train["event"].to_numpy(), train_f)
    )
    null_curve = np.asarray(estimator.predict_proba(followup_times), dtype=np.float64)
    return np.tile(null_curve, (n_rows, 1))


def _stratified_bootstrap_ci(
    event_idx: np.ndarray,
    censor_idx: np.ndarray,
    statistic,
) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    values: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_events = rng.choice(event_idx, size=event_idx.size, replace=True)
        sampled_censors = rng.choice(censor_idx, size=censor_idx.size, replace=True)
        value = float(statistic(sampled_events, sampled_censors))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    alpha = 100.0 * (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [alpha, 100.0 - alpha])
    return float(low), float(high)


def _ipcw_c_for_indices(
    train_y,
    event: np.ndarray,
    followup: np.ndarray,
    risk: np.ndarray,
    indices: np.ndarray,
    tau: float,
) -> float:
    from sksurv.metrics import concordance_index_ipcw

    idx = np.asarray(indices, dtype=np.int64)
    test_y = _surv_array(event[idx], followup[idx])
    return float(concordance_index_ipcw(train_y, test_y, np.asarray(risk, dtype=np.float64)[idx], tau=tau)[0])


def _train_followup_surv(train: pd.DataFrame):
    train_f = (train["exit_age"] - train["entry_age"]).to_numpy(dtype=np.float64)
    return _surv_array(train["event"].to_numpy(), train_f)


def bootstrap_ipcw_c_ci(train: pd.DataFrame, test: pd.DataFrame, risk: np.ndarray, tau: float) -> tuple[float, float]:
    train_y = _train_followup_surv(train)
    event = test["event"].to_numpy(dtype=bool)
    followup = (test["exit_age"] - test["entry_age"]).to_numpy(dtype=np.float64)
    event_idx = np.flatnonzero(event)
    censor_idx = np.flatnonzero(~event)
    if event_idx.size == 0 or censor_idx.size == 0:
        return float("nan"), float("nan")

    def statistic(sampled_events: np.ndarray, sampled_censors: np.ndarray) -> float:
        idx = np.concatenate([
            np.asarray(sampled_events, dtype=np.int64),
            np.asarray(sampled_censors, dtype=np.int64),
        ])
        return _ipcw_c_for_indices(train_y, event, followup, risk, idx, tau)

    return _stratified_bootstrap_ci(event_idx, censor_idx, statistic)


def bootstrap_ipcw_delta_ci(
    train: pd.DataFrame,
    test: pd.DataFrame,
    left_risk: np.ndarray,
    right_risk: np.ndarray,
    tau: float,
) -> tuple[float, float]:
    train_y = _train_followup_surv(train)
    event = test["event"].to_numpy(dtype=bool)
    followup = (test["exit_age"] - test["entry_age"]).to_numpy(dtype=np.float64)
    event_idx = np.flatnonzero(event)
    censor_idx = np.flatnonzero(~event)
    if event_idx.size == 0 or censor_idx.size == 0:
        return float("nan"), float("nan")

    def statistic(sampled_events: np.ndarray, sampled_censors: np.ndarray) -> float:
        idx = np.concatenate([
            np.asarray(sampled_events, dtype=np.int64),
            np.asarray(sampled_censors, dtype=np.int64),
        ])
        left = _ipcw_c_for_indices(train_y, event, followup, left_risk, idx, tau)
        right = _ipcw_c_for_indices(train_y, event, followup, right_risk, idx, tau)
        return left - right

    return _stratified_bootstrap_ci(event_idx, censor_idx, statistic)


def survival_library_metrics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    risk: np.ndarray,
    surv_cond: np.ndarray,
    followup_times: np.ndarray,
    tau_followup: float | None = None,
) -> SurvivalMetricStats:
    """sksurv discrimination + calibration metrics on follow-up scale.

    Both ``train`` and ``test`` must carry ``entry_age``, ``exit_age``, and
    ``event``; the follow-up time ``s = exit_age - entry_age`` is what the
    sksurv estimators see (right-censored, no entry). ``risk`` is a per-row
    monotone-in-hazard ranking score (e.g. ``-log S_cond(s_K)``) and
    ``surv_cond`` is the per-row conditional survival on ``followup_times``.

    Returns a :class:`SurvivalMetricStats` with IPCW Harrell's C (Uno) plus
    its bootstrap 95% CI, mean cumulative/dynamic AUC, integrated Brier
    score, IBS under the KM null reference, and IPA pseudo R² =
    ``1 - IBS / IBS_null``.
    """
    from sksurv.metrics import (
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )

    train_f = (train["exit_age"] - train["entry_age"]).to_numpy(dtype=np.float64)
    test_f = (test["exit_age"] - test["entry_age"]).to_numpy(dtype=np.float64)
    train_y = _surv_array(train["event"].to_numpy(), train_f)
    test_y = _surv_array(test["event"].to_numpy(), test_f)
    risk = np.asarray(risk, dtype=np.float64)
    surv_cond = np.asarray(surv_cond, dtype=np.float64)
    if surv_cond.shape != (len(test), len(followup_times)):
        raise ValueError(
            f"conditional survival shape {surv_cond.shape} does not match "
            f"(n_test={len(test)}, n_times={len(followup_times)})"
        )
    # IPCW C-index truncation time: 80th percentile of *observed event*
    # follow-up times in the held-out set, clipped to the interior of the
    # follow-up metric grid. scikit-survival recommends a conservative
    # percentile rather than max(times) because the IPCW weight 1/G(s)
    # becomes unstable as the censoring KM approaches zero. The grid is
    # already pre-restricted to the train/test follow-up overlap by
    # ``_survival_metric_times``.
    event_f = test_f[test["event"].to_numpy(dtype=bool)]
    if tau_followup is not None:
        tau_candidate = float(tau_followup)
    elif event_f.size >= 1:
        tau_candidate = float(np.quantile(event_f, 0.80))
    else:
        tau_candidate = float(followup_times[-1])
    tau = float(np.clip(tau_candidate, float(followup_times[0]), float(followup_times[-1])))
    c_ipcw = float(concordance_index_ipcw(train_y, test_y, risk, tau=tau)[0])
    c_ipcw_ci_low, c_ipcw_ci_high = bootstrap_ipcw_c_ci(train, test, risk, tau)
    _, mean_auc = cumulative_dynamic_auc(train_y, test_y, risk, followup_times)
    ibs = float(integrated_brier_score(train_y, test_y, surv_cond, followup_times))
    null_ibs = float(
        integrated_brier_score(
            train_y,
            test_y,
            null_survival_matrix(train, len(test), followup_times),
            followup_times,
        )
    )
    r2_ibs = float(1.0 - ibs / null_ibs) if null_ibs > 0 else float("nan")
    return SurvivalMetricStats(
        n=int(len(test)),
        n_events=int(test["event"].sum()),
        median_exit_age=float(np.median(test["exit_age"].to_numpy(dtype=np.float64))),
        c_ipcw=c_ipcw,
        c_ipcw_ci_low=c_ipcw_ci_low,
        c_ipcw_ci_high=c_ipcw_ci_high,
        dynamic_auc_mean=float(mean_auc),
        ibs=ibs,
        null_ibs=null_ibs,
        r2_ibs=r2_ibs,
        tau=tau,
        ibs_start=float(followup_times[0]),
        ibs_stop=float(followup_times[-1]),
        n_times=int(len(followup_times)),
    )


def fmt_survival(stats: SurvivalMetricStats) -> str:
    return (
        f"C_ipcw={stats.c_ipcw:.4f} "
        f"95%CI=[{stats.c_ipcw_ci_low:.4f},{stats.c_ipcw_ci_high:.4f}]  "
        f"iAUC={stats.dynamic_auc_mean:.4f}  "
        f"IBS={stats.ibs:.5f}  R2_IBS={stats.r2_ibs:.4f}"
    )


def _survival_fields(prefix: str, stats: SurvivalMetricStats) -> dict[str, float | int]:
    return {
        f"{prefix}_n": stats.n,
        f"{prefix}_events": stats.n_events,
        f"{prefix}_median_exit_age": stats.median_exit_age,
        f"{prefix}_c_ipcw": stats.c_ipcw,
        f"{prefix}_c_ipcw_ci_low": stats.c_ipcw_ci_low,
        f"{prefix}_c_ipcw_ci_high": stats.c_ipcw_ci_high,
        f"{prefix}_dynamic_auc_mean": stats.dynamic_auc_mean,
        f"{prefix}_ibs": stats.ibs,
        f"{prefix}_null_ibs": stats.null_ibs,
        f"{prefix}_r2_ibs": stats.r2_ibs,
        f"{prefix}_tau": stats.tau,
        f"{prefix}_ibs_start": stats.ibs_start,
        f"{prefix}_ibs_stop": stats.ibs_stop,
        f"{prefix}_n_times": stats.n_times,
    }


def save_fit_cache(
    model,
    fit_path: Path | None,
    meta_path: Path | None,
    save_info: dict[str, object] | None,
    pc_cols: list[str],
    pgs_mean: float,
    pgs_std: float,
    train_n: int,
) -> None:
    if save_info is None or fit_path is None or meta_path is None:
        return
    try:
        model.save(str(fit_path))
    except Exception as e:
        print(f"  save: model.save failed ({e}); skipping persist")
        return
    meta = {
        **save_info,
        "num_pcs": len(pc_cols),
        "duchon_centers": DUCHON_CENTERS,
        "train_fraction": TRAIN_FRACTION,
        "rng_seed": RNG_SEED,
        "pgs_mean": pgs_mean,
        "pgs_std": pgs_std,
        "n_train": int(train_n),
        "survival_likelihood": "marginal-slope",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  save: model -> {fit_path}  meta -> {meta_path}")


def prepare_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pc_cols: list[str],
    pgs_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Train-only PGS standardization for GAM plus train-only Z_norm2 baseline."""
    train = train.copy()
    test = test.copy()
    pgs_mean = float(train["pgs"].mean())
    pgs_std = float(train["pgs"].std(ddof=0))
    if not np.isfinite(pgs_std) or pgs_std <= 0:
        raise ValueError("training PGS has zero or invalid variance")
    train["prs_z"] = (train["pgs"] - pgs_mean) / pgs_std
    test["prs_z"] = (test["pgs"] - pgs_mean) / pgs_std

    z_tr, z_te = z_norm2(
        train["pgs"].to_numpy(),
        train[pc_cols].to_numpy(),
        test["pgs"].to_numpy(),
        test[pc_cols].to_numpy(),
        pgs_id=pgs_id,
    )
    train["z_norm2"] = z_tr
    test["z_norm2"] = z_te
    return train, test, pgs_mean, pgs_std


def evaluate_binary_model_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pc_cols: list[str],
    pgs_id: str,
    label: str,
    population_prevalence: float | None,
    score_train: bool,
) -> dict[str, float]:
    """Fit Bernoulli marginal-slope GAM and matched logistic baselines on
    the same balanced split.

    Age handling is fixed across models; the GAM adds the marginal-slope
    flexible PC surface and PC-varying log-slope:

      * `binaryGAM` — Bernoulli marginal-slope GAM with
        `matern(PC1..PCk)` in the baseline-risk surface, the same
        `matern(...)` in the log-slope channel (so `prs_z`'s log-OR
        varies by ancestry), and the matched age nuisance basis
        (quadratic `current_age_z`/`current_age_z2` + linear `entry_age_z`,
        identical to baselines N/A/B) plus `sex`. matern is required on the
        marginal channel for probit identifiability (gam#531); no
        `linkwiggle()` link-deviation or score-warp (both hang, gam#683).
      * `binaryN` — logistic GLM on the matched age nuisance basis
        (quadratic `current_age_z`/`current_age_z2` + linear
        `entry_age_z`) and `sex`, with *no PRS information*. The
        age+sex-only ceiling.
      * `binaryA` — `binaryN + z_norm2` (pgsc_calc PC mean+var adjusted
        PRS, fit on train, applied to test).
      * `binaryB` — `binaryN + prs_z + PC1..PCk` (train-z-scored PRS
        plus linear PCs).

    Discrimination is reported with AUROC + paired-DeLong 95% CI, AP,
    log-loss, Brier, Nagelkerke R², and Lee 2012 liability R² (taking
    the supplied `population_prevalence` so the 1:1 balanced split's
    sample prevalence of 0.5 is correctly mapped back to the population
    scale). The headline `GAM−N` delta is the *clinical-utility* lift,
    `A−N` and `B−N` are PRS-incremental lifts over age+sex alone, and
    `GAM−B` isolates the value of the GAM-native smooth basis given the
    same PRS and PC information.
    """
    train, test = _add_binary_age_features(train, test)
    y_train = train["event"].to_numpy()
    y_test = test["event"].to_numpy()

    model = fit_binary_marginal_slope(train, len(pc_cols))
    gam_p_test = _extract_binary_mean(model.predict(test[binary_predict_columns(len(pc_cols))]))
    gam_test_m = binary_stats(y_test, gam_p_test, population_prevalence)
    gam_train_m = None
    if score_train:
        gam_p_train = _extract_binary_mean(model.predict(train[binary_predict_columns(len(pc_cols))]))
        gam_train_m = binary_stats(y_train, gam_p_train, population_prevalence)

    nuisance_names = [*BASELINE_AGE_FEATURES, "sex"]
    X_N_tr = train[nuisance_names].to_numpy()
    X_N_te = test[nuisance_names].to_numpy()
    N_fit = fit_logit_binary(y_train, X_N_tr, nuisance_names)
    p_N_test = predict_logit_binary(N_fit, X_N_te)
    N_test_m = binary_stats(y_test, p_N_test, population_prevalence)

    A_names = [*nuisance_names, "z_norm2"]
    X_A_tr = train[A_names].to_numpy()
    X_A_te = test[A_names].to_numpy()
    A_fit = fit_logit_binary(y_train, X_A_tr, A_names)
    p_A_test = predict_logit_binary(A_fit, X_A_te)
    A_test_m = binary_stats(y_test, p_A_test, population_prevalence)

    B_names = [*nuisance_names, "prs_z", *pc_cols]
    X_B_tr = train[B_names].to_numpy()
    X_B_te = test[B_names].to_numpy()
    B_fit = fit_logit_binary(y_train, X_B_tr, B_names)
    p_B_test = predict_logit_binary(B_fit, X_B_te)
    B_test_m = binary_stats(y_test, p_B_test, population_prevalence)

    N_train_m = A_train_m = B_train_m = None
    if score_train:
        N_train_m = binary_stats(y_train, predict_logit_binary(N_fit, X_N_tr), population_prevalence)
        A_train_m = binary_stats(y_train, predict_logit_binary(A_fit, X_A_tr), population_prevalence)
        B_train_m = binary_stats(y_train, predict_logit_binary(B_fit, X_B_tr), population_prevalence)

    A_coefs = A_fit["coefs"]
    B_coefs = B_fit["coefs"]
    print(
        "  binary_baseline_spec: logistic GLM nuisance=("
        + " + ".join(nuisance_names)
        + "); compare nuisance-only (N), +Z_norm2 (A), and +prs_z+linear PCs (B)"
    )
    print(
        f"  binary_baselineA: log_OR(z_norm2)={A_coefs['z_norm2']:+.4f}  "
        f"OR/SD={np.exp(A_coefs['z_norm2']):.4f}"
    )
    pc_str = "  ".join(f"{c}={B_coefs[c]:+.3f}" for c in pc_cols)
    print(
        f"  binary_baselineB: log_OR(prs_z)={B_coefs['prs_z']:+.4f}  "
        f"OR/SD={np.exp(B_coefs['prs_z']):.4f}  [{pc_str}]"
    )
    print(
        f"  binary {label}  train_n={len(train):,}  test_n={gam_test_m.n:,}  "
        f"test_events={gam_test_m.n_events:,}  test_P={gam_test_m.prevalence:.4f}  "
        f"K_crude={population_prevalence if population_prevalence is not None else float('nan'):.6f}"
    )
    if score_train:
        assert gam_train_m is not None and N_train_m is not None and A_train_m is not None and B_train_m is not None
        print(f"  binaryGAM   train {fmt_binary(gam_train_m)}  test {fmt_binary(gam_test_m)}")
        print(f"  binaryN     train {fmt_binary(N_train_m)}  test {fmt_binary(N_test_m)}")
        print(f"  binaryA     train {fmt_binary(A_train_m)}  test {fmt_binary(A_test_m)}")
        print(f"  binaryB     train {fmt_binary(B_train_m)}  test {fmt_binary(B_test_m)}")
    else:
        print(f"  binaryGAM   test {fmt_binary(gam_test_m)}")
        print(f"  binaryN     test {fmt_binary(N_test_m)}")
        print(f"  binaryA     test {fmt_binary(A_test_m)}")
        print(f"  binaryB     test {fmt_binary(B_test_m)}")

    # Paired DeLong (Sun & Xu 2014) on the same held-out rows: per-model AUROC
    # SE/CI and paired Δ AUROC SE/CI (GAM minus each baseline). Single pass —
    # all four models share the AUROC covariance matrix.
    stack = np.vstack([gam_p_test, p_N_test, p_A_test, p_B_test])
    aucs, cov = fast_delong_auc(stack, y_test)
    Z975 = 1.959963984540054

    def _auc_ci(k: int) -> tuple[float, float, float]:
        se = float(np.sqrt(max(cov[k, k], 0.0)))
        return se, max(0.0, float(aucs[k]) - Z975 * se), min(1.0, float(aucs[k]) + Z975 * se)

    def _delta_ci(k: int, j: int) -> tuple[float, float, float, float]:
        d = float(aucs[k] - aucs[j])
        var = float(cov[k, k] + cov[j, j] - 2.0 * cov[k, j])
        se = float(np.sqrt(max(var, 0.0)))
        return d, se, d - Z975 * se, d + Z975 * se

    gam_auc_se, gam_auc_lo, gam_auc_hi = _auc_ci(0)
    N_auc_se, N_auc_lo, N_auc_hi = _auc_ci(1)
    A_auc_se, A_auc_lo, A_auc_hi = _auc_ci(2)
    B_auc_se, B_auc_lo, B_auc_hi = _auc_ci(3)
    d_GAM_N = _delta_ci(0, 1)
    d_GAM_A = _delta_ci(0, 2)
    d_GAM_B = _delta_ci(0, 3)
    d_A_N = _delta_ci(2, 1)
    d_B_N = _delta_ci(3, 1)

    print(
        f"  binary_auroc_ci  GAM=[{gam_auc_lo:.4f},{gam_auc_hi:.4f}]"
        f" N=[{N_auc_lo:.4f},{N_auc_hi:.4f}]"
        f" A=[{A_auc_lo:.4f},{A_auc_hi:.4f}]"
        f" B=[{B_auc_lo:.4f},{B_auc_hi:.4f}]"
    )
    print(
        f"  binary_delta_auroc_ci  GAM-N={d_GAM_N[0]:+.4f}"
        f" 95%CI=[{d_GAM_N[2]:+.4f},{d_GAM_N[3]:+.4f}] SE={d_GAM_N[1]:.4f}  "
        f"GAM-A={d_GAM_A[0]:+.4f}"
        f" 95%CI=[{d_GAM_A[2]:+.4f},{d_GAM_A[3]:+.4f}] SE={d_GAM_A[1]:.4f}  "
        f"GAM-B={d_GAM_B[0]:+.4f}"
        f" 95%CI=[{d_GAM_B[2]:+.4f},{d_GAM_B[3]:+.4f}] SE={d_GAM_B[1]:.4f}"
    )
    print(
        f"  binary_prs_lift_ci  A-N={d_A_N[0]:+.4f}"
        f" 95%CI=[{d_A_N[2]:+.4f},{d_A_N[3]:+.4f}]  "
        f"B-N={d_B_N[0]:+.4f}"
        f" 95%CI=[{d_B_N[2]:+.4f},{d_B_N[3]:+.4f}]  "
        f"(PRS-incremental discrimination over age+sex nuisance)"
    )
    print(
        "  binary_delta "
        f"logloss(GAM-N)={gam_test_m.log_loss - N_test_m.log_loss:+.4f}  "
        f"Brier(GAM-N)={gam_test_m.brier - N_test_m.brier:+.4f}  "
        f"Nagelkerke_R2(GAM-N)={gam_test_m.nagelkerke_r2 - N_test_m.nagelkerke_r2:+.4f}  "
        f"liability_R2(GAM-N)={gam_test_m.liability_r2 - N_test_m.liability_r2:+.4f}"
    )

    result: dict[str, float | int | str] = {
        "label": label,
        "pgs": pgs_id,
        **_binary_fields("binary_gam_test", gam_test_m),
        **_binary_fields("binary_nuisance_test", N_test_m),
        **_binary_fields("binary_baselineA_test", A_test_m),
        **_binary_fields("binary_baselineB_test", B_test_m),
        "binary_gam_test_auroc_se": gam_auc_se,
        "binary_gam_test_auroc_ci_low": gam_auc_lo,
        "binary_gam_test_auroc_ci_high": gam_auc_hi,
        "binary_nuisance_test_auroc_se": N_auc_se,
        "binary_nuisance_test_auroc_ci_low": N_auc_lo,
        "binary_nuisance_test_auroc_ci_high": N_auc_hi,
        "binary_baselineA_test_auroc_se": A_auc_se,
        "binary_baselineA_test_auroc_ci_low": A_auc_lo,
        "binary_baselineA_test_auroc_ci_high": A_auc_hi,
        "binary_baselineB_test_auroc_se": B_auc_se,
        "binary_baselineB_test_auroc_ci_low": B_auc_lo,
        "binary_baselineB_test_auroc_ci_high": B_auc_hi,
        "binary_delta_auroc_vs_nuisance": d_GAM_N[0],
        "binary_delta_auroc_vs_nuisance_se": d_GAM_N[1],
        "binary_delta_auroc_vs_nuisance_ci_low": d_GAM_N[2],
        "binary_delta_auroc_vs_nuisance_ci_high": d_GAM_N[3],
        "binary_delta_auroc_vs_baselineA": d_GAM_A[0],
        "binary_delta_auroc_vs_baselineA_se": d_GAM_A[1],
        "binary_delta_auroc_vs_baselineA_ci_low": d_GAM_A[2],
        "binary_delta_auroc_vs_baselineA_ci_high": d_GAM_A[3],
        "binary_delta_auroc_vs_baselineB": d_GAM_B[0],
        "binary_delta_auroc_vs_baselineB_se": d_GAM_B[1],
        "binary_delta_auroc_vs_baselineB_ci_low": d_GAM_B[2],
        "binary_delta_auroc_vs_baselineB_ci_high": d_GAM_B[3],
        "binary_baselineA_minus_nuisance_auroc": d_A_N[0],
        "binary_baselineA_minus_nuisance_auroc_se": d_A_N[1],
        "binary_baselineA_minus_nuisance_auroc_ci_low": d_A_N[2],
        "binary_baselineA_minus_nuisance_auroc_ci_high": d_A_N[3],
        "binary_baselineB_minus_nuisance_auroc": d_B_N[0],
        "binary_baselineB_minus_nuisance_auroc_se": d_B_N[1],
        "binary_baselineB_minus_nuisance_auroc_ci_low": d_B_N[2],
        "binary_baselineB_minus_nuisance_auroc_ci_high": d_B_N[3],
        "binary_delta_log_loss_vs_nuisance": gam_test_m.log_loss - N_test_m.log_loss,
        "binary_delta_brier_vs_nuisance": gam_test_m.brier - N_test_m.brier,
        "binary_delta_nagelkerke_vs_nuisance": gam_test_m.nagelkerke_r2 - N_test_m.nagelkerke_r2,
        "binary_delta_liability_r2_vs_nuisance": gam_test_m.liability_r2 - N_test_m.liability_r2,
    }
    if score_train:
        assert gam_train_m is not None and N_train_m is not None and A_train_m is not None and B_train_m is not None
        result.update(_binary_fields("binary_gam_train", gam_train_m))
        result.update(_binary_fields("binary_nuisance_train", N_train_m))
        result.update(_binary_fields("binary_baselineA_train", A_train_m))
        result.update(_binary_fields("binary_baselineB_train", B_train_m))
    print("  BINARY_RESULT " + json.dumps(result, sort_keys=True))
    return {
        k: float(v)
        for k, v in result.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def evaluate_model_pair(
    train: pd.DataFrame,
    test: pd.DataFrame,
    pc_cols: list[str],
    pgs_id: str,
    label: str,
    population_prevalence: float | None = None,
    save_info: dict[str, object] | None = None,
    score_train: bool = True,
    mode: str = "both",
) -> dict[str, float]:
    """Fit GAM + Cox baselines and print survival validation metrics.

    `mode` selects which fits to run: ``"both"`` (default) runs the
    survival marginal-slope GAM with Cox PH baselines *and* the binary
    marginal-slope GAM with logistic baselines; ``"survival"`` skips the
    binary path; ``"binary"`` skips the survival path (no Cox fits, no
    sksurv metrics, no ``.gamfit`` artifact write).
    """
    train, test, pgs_mean, pgs_std = prepare_scores(train, test, pc_cols, pgs_id)

    if mode == "binary":
        # Short-circuit: skip the survival GAM, Cox baselines, sksurv
        # metrics, and artifact save entirely. `evaluate_binary_model_pair`
        # is self-contained — it adds its own age features and fits its
        # own logistic baselines.
        if save_info is not None:
            print("  save: skipped (mode=binary does not persist artifacts)")
        binary_result = evaluate_binary_model_pair(
            train, test, pc_cols, pgs_id,
            label=label,
            population_prevalence=population_prevalence,
            score_train=score_train,
        )
        result: dict[str, float | int | str] = {
            "label": label, "pgs": pgs_id, **binary_result,
        }
        print("  RESULT " + json.dumps(result, sort_keys=True))
        return {
            k: float(v)
            for k, v in result.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

    # Fit. Resume is handled transparently by gamfit's persistent warm-start
    # cache at `XDG_CACHE_HOME/gam/warm/v1/`: on identical (training data, gamfit
    # version) the outer (rho) and inner (beta) iterates from the prior fit
    # are loaded and PIRLS converges in 1-2 cycles. We don't try to manage
    # resume ourselves through `.gamfit` files — those are for artifact
    # persistence, not resume, and they're a separate code path with a
    # separate set of failure modes (e.g., the gamfit 0.1.69 marginal-slope
    # save bug fixed in fa7f7b6c / 0.1.70).
    model = fit_marginal_slope(train, len(pc_cols))

    if save_info is not None:
        # Best-effort artifact write: prediction below uses the in-memory
        # `model`, not the file, so a save failure (e.g., 0.1.69 omitting
        # survival_time_* metadata) doesn't fail this run.
        FITS_DIR.mkdir(parents=True, exist_ok=True)
        disease = str(save_info["disease"])
        save_fit_cache(
            model,
            FITS_DIR / f"{disease}.gamfit",
            FITS_DIR / f"{disease}.meta.json",
            save_info, pc_cols, pgs_mean, pgs_std, len(train),
        )

    print(
        "  baseline_spec: two left-truncated Cox PH benchmarks; both share "
        "`+ sex` with the GAM."
    )
    entry_tr = train["entry_age"].to_numpy()
    exit_tr  = train["exit_age"].to_numpy()
    event_tr = train["event"].to_numpy()
    sex_tr   = train["sex"].to_numpy()
    sex_te   = test["sex"].to_numpy()

    # Baseline A: Z_norm2 (PC mean+var adjusted PRS) + sex
    X_A_tr = np.column_stack([train["z_norm2"].to_numpy(), sex_tr])
    X_A_te = np.column_stack([test["z_norm2"].to_numpy(), sex_te])
    A_names = ["z_norm2", "sex"]
    A_fit = fit_baseline_cox(entry_tr, exit_tr, event_tr, X_A_tr, A_names)
    A_coefs = A_fit.coefs
    print(
        f"  baselineA   Cox PH on  z_norm2 + sex          "
        f"log_HR(z_norm2)={A_coefs['z_norm2']:+.4f}  "
        f"HR/SD={np.exp(A_coefs['z_norm2']):.4f}  "
        f"log_HR(sex)={A_coefs['sex']:+.4f}"
    )

    # Baseline B: prs_z + sex + PC1..PC_k (linear PCs as Cox covariates)
    X_B_tr = np.column_stack([
        train["prs_z"].to_numpy(), sex_tr, train[pc_cols].to_numpy(),
    ])
    X_B_te = np.column_stack([
        test["prs_z"].to_numpy(), sex_te, test[pc_cols].to_numpy(),
    ])
    B_names = ["prs_z", "sex", *pc_cols]
    B_fit = fit_baseline_cox(entry_tr, exit_tr, event_tr, X_B_tr, B_names)
    B_coefs = B_fit.coefs
    pc_str = "  ".join(f"{c}={B_coefs[c]:+.3f}" for c in pc_cols)
    print(
        f"  baselineB   Cox PH on  prs_z + sex + {'+'.join(pc_cols)}  "
        f"log_HR(prs_z)={B_coefs['prs_z']:+.4f}  "
        f"HR/SD={np.exp(B_coefs['prs_z']):.4f}  "
        f"log_HR(sex)={B_coefs['sex']:+.4f}  [{pc_str}]"
    )

    # Follow-up time scale for all sksurv metrics (left-truncation respected).
    # gam_survival_outputs / cox_survival_matrix produce conditional survival
    # S_cond_i(s | entry_i); ranking risk is -log S_cond(s_K), the per-row
    # cumulative hazard accrued during the follow-up window.
    train_followup = (train["exit_age"] - train["entry_age"]).to_numpy(dtype=np.float64)
    test_followup = (test["exit_age"] - test["entry_age"]).to_numpy(dtype=np.float64)
    metric_times = _survival_metric_times(
        train_followup,
        test_followup,
        test["event"].to_numpy(dtype=bool),
    )
    gam_risk_test, gam_surv_test, kept_test = gam_survival_outputs(
        model, test, len(pc_cols), metric_times
    )
    test_metric = test.loc[kept_test].reset_index(drop=True)
    train_followup_max = float(np.max(train_followup))
    test_metric_followup = (
        test_metric["exit_age"] - test_metric["entry_age"]
    ).to_numpy(dtype=np.float64)
    support_mask = test_metric_followup < np.nextafter(train_followup_max, -np.inf)
    if int(support_mask.sum()) < 2:
        raise ValueError("fewer than two held-out rows inside the training censoring support")
    if not bool(support_mask.all()):
        print(
            f"  survival_metrics: dropped {int((~support_mask).sum())} held-out rows "
            "outside the training follow-up censoring support"
        )
        test_metric = test_metric.loc[support_mask].reset_index(drop=True)
        test_metric_followup = test_metric_followup[support_mask]
        gam_risk_test = gam_risk_test[support_mask]
        gam_surv_test = gam_surv_test[support_mask, :]
        kept_positions = np.flatnonzero(kept_test)[support_mask]
        kept_test = np.zeros_like(kept_test, dtype=bool)
        kept_test[kept_positions] = True
    valid_time = (
        (metric_times > float(test_metric_followup.min()))
        & (metric_times < min(float(test_metric_followup.max()), train_followup_max))
    )
    if int(valid_time.sum()) < 2:
        raise ValueError("fewer than two valid follow-up metric time points after prediction filtering")
    if not bool(valid_time.all()):
        metric_times = metric_times[valid_time]
        gam_surv_test = gam_surv_test[:, valid_time]

    A_entry_te = test_metric["entry_age"].to_numpy(dtype=np.float64)
    B_entry_te = A_entry_te  # same rows
    A_surv_test = cox_survival_matrix(A_fit, X_A_te[kept_test], A_entry_te, metric_times)
    B_surv_test = cox_survival_matrix(B_fit, X_B_te[kept_test], B_entry_te, metric_times)
    # Per-row conditional cumulative hazard at the last follow-up time, same
    # ranking-score convention as the GAM.
    A_risk_test = -np.log(np.clip(A_surv_test[:, -1], 1e-12, 1.0))
    B_risk_test = -np.log(np.clip(B_surv_test[:, -1], 1e-12, 1.0))
    A_test_m = survival_library_metrics(
        train, test_metric, A_risk_test, A_surv_test, metric_times,
    )
    B_test_m = survival_library_metrics(
        train, test_metric, B_risk_test, B_surv_test, metric_times,
    )
    gam_test_m = survival_library_metrics(
        train, test_metric, gam_risk_test, gam_surv_test, metric_times,
    )
    gam_train_m = A_train_m = B_train_m = None
    if score_train:
        train_times = _survival_metric_times(
            train_followup,
            train_followup,
            train["event"].to_numpy(dtype=bool),
        )
        gam_risk_train, gam_surv_train, kept_train = gam_survival_outputs(
            model, train, len(pc_cols), train_times
        )
        train_metric = train.loc[kept_train].reset_index(drop=True)
        train_metric_followup = (
            train_metric["exit_age"] - train_metric["entry_age"]
        ).to_numpy(dtype=np.float64)
        train_support_mask = train_metric_followup < np.nextafter(train_followup_max, -np.inf)
        if int(train_support_mask.sum()) < 2:
            raise ValueError("fewer than two train rows inside the training censoring support")
        if not bool(train_support_mask.all()):
            train_metric = train_metric.loc[train_support_mask].reset_index(drop=True)
            train_metric_followup = train_metric_followup[train_support_mask]
            gam_risk_train = gam_risk_train[train_support_mask]
            gam_surv_train = gam_surv_train[train_support_mask, :]
            kept_train_positions = np.flatnonzero(kept_train)[train_support_mask]
            kept_train = np.zeros_like(kept_train, dtype=bool)
            kept_train[kept_train_positions] = True
        valid_train_time = (
            (train_times > float(train_metric_followup.min()))
            & (train_times < float(train_metric_followup.max()))
        )
        if int(valid_train_time.sum()) < 2:
            raise ValueError("fewer than two valid train follow-up metric time points")
        if not bool(valid_train_time.all()):
            train_times = train_times[valid_train_time]
            gam_surv_train = gam_surv_train[:, valid_train_time]
        gam_train_m = survival_library_metrics(
            train, train_metric, gam_risk_train, gam_surv_train, train_times,
        )
        A_entry_tr = train_metric["entry_age"].to_numpy(dtype=np.float64)
        B_entry_tr = A_entry_tr
        A_surv_train = cox_survival_matrix(A_fit, X_A_tr[kept_train], A_entry_tr, train_times)
        B_surv_train = cox_survival_matrix(B_fit, X_B_tr[kept_train], B_entry_tr, train_times)
        A_risk_train = -np.log(np.clip(A_surv_train[:, -1], 1e-12, 1.0))
        B_risk_train = -np.log(np.clip(B_surv_train[:, -1], 1e-12, 1.0))
        A_train_m = survival_library_metrics(
            train, train_metric, A_risk_train, A_surv_train, train_times,
        )
        B_train_m = survival_library_metrics(
            train, train_metric, B_risk_train, B_surv_train, train_times,
        )
    delta_A_c = gam_test_m.c_ipcw - A_test_m.c_ipcw
    delta_B_c = gam_test_m.c_ipcw - B_test_m.c_ipcw
    delta_A_c_ci_low, delta_A_c_ci_high = bootstrap_ipcw_delta_ci(
        train, test_metric, gam_risk_test, A_risk_test, gam_test_m.tau
    )
    delta_B_c_ci_low, delta_B_c_ci_high = bootstrap_ipcw_delta_ci(
        train, test_metric, gam_risk_test, B_risk_test, gam_test_m.tau
    )
    delta_A_ibs = gam_test_m.ibs - A_test_m.ibs
    delta_B_ibs = gam_test_m.ibs - B_test_m.ibs
    delta_A_r2 = gam_test_m.r2_ibs - A_test_m.r2_ibs
    delta_B_r2 = gam_test_m.r2_ibs - B_test_m.r2_ibs
    print(
        f"  {label}  train_n={len(train):,}  test_n={gam_test_m.n:,}  "
        f"test_events={gam_test_m.n_events:,}  "
        f"median_exit_age={gam_test_m.median_exit_age:.2f}  "
        f"metric_followup_grid=[{gam_test_m.ibs_start:.2f},{gam_test_m.ibs_stop:.2f}] yr "
        f"tau={gam_test_m.tau:.2f} yr  n_times={gam_test_m.n_times}"
    )
    if score_train:
        assert gam_train_m is not None and A_train_m is not None and B_train_m is not None
        print(f"  GAM        train {fmt_survival(gam_train_m)}  test {fmt_survival(gam_test_m)}")
        print(f"  baselineA  train {fmt_survival(A_train_m)}  test {fmt_survival(A_test_m)}")
        print(f"  baselineB  train {fmt_survival(B_train_m)}  test {fmt_survival(B_test_m)}")
    else:
        print(f"  GAM        test {fmt_survival(gam_test_m)}")
        print(f"  baselineA  test {fmt_survival(A_test_m)}")
        print(f"  baselineB  test {fmt_survival(B_test_m)}")
    print(
        f"  delta      GAM-baselineA: ΔC_ipcw={delta_A_c:+.4f} "
        f"95%CI=[{delta_A_c_ci_low:+.4f},{delta_A_c_ci_high:+.4f}] "
        f"ΔIBS={delta_A_ibs:+.5f} ΔR2_IBS={delta_A_r2:+.4f}  "
        f"GAM-baselineB: ΔC_ipcw={delta_B_c:+.4f} "
        f"95%CI=[{delta_B_c_ci_low:+.4f},{delta_B_c_ci_high:+.4f}] "
        f"ΔIBS={delta_B_ibs:+.5f} ΔR2_IBS={delta_B_r2:+.4f}"
    )
    if mode != "survival":
        binary_result = evaluate_binary_model_pair(
            train,
            test,
            pc_cols,
            pgs_id,
            label=label,
            population_prevalence=population_prevalence,
            score_train=score_train,
        )
    else:
        binary_result = {}

    result: dict[str, float | int | str] = {
        "label": label,
        "pgs": pgs_id,
        **_survival_fields("gam_test", gam_test_m),
        **_survival_fields("baselineA_test", A_test_m),
        **_survival_fields("baselineB_test", B_test_m),
        "delta_test_c_ipcw_vs_baselineA": delta_A_c,
        "delta_test_c_ipcw_vs_baselineA_ci_low": delta_A_c_ci_low,
        "delta_test_c_ipcw_vs_baselineA_ci_high": delta_A_c_ci_high,
        "delta_test_c_ipcw_vs_baselineB": delta_B_c,
        "delta_test_c_ipcw_vs_baselineB_ci_low": delta_B_c_ci_low,
        "delta_test_c_ipcw_vs_baselineB_ci_high": delta_B_c_ci_high,
        "delta_test_ibs_vs_baselineA": delta_A_ibs,
        "delta_test_ibs_vs_baselineB": delta_B_ibs,
        "delta_test_r2_ibs_vs_baselineA": delta_A_r2,
        "delta_test_r2_ibs_vs_baselineB": delta_B_r2,
    }
    result.update(binary_result)
    if score_train:
        assert gam_train_m is not None and A_train_m is not None and B_train_m is not None
        result.update(_survival_fields("gam_train", gam_train_m))
        result.update(_survival_fields("baselineA_train", A_train_m))
        result.update(_survival_fields("baselineB_train", B_train_m))
    else:
        result.update({
            "gam_train_c_ipcw": float("nan"),
            "baselineA_train_c_ipcw": float("nan"),
            "baselineB_train_c_ipcw": float("nan"),
        })
    print("  RESULT " + json.dumps(result, sort_keys=True))
    return {
        k: float(v)
        for k, v in result.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def loso_groups(
    df: pd.DataFrame,
    col: str,
    min_train_events: int,
    min_train_censors: int,
    min_test_events: int,
    min_test_censors: int,
    min_test_n: int,
    max_groups: int | None = None,
    verbose: bool = True,
) -> list[str]:
    """Eligible held-out groups, sorted by size and event-count constraints.

    When `verbose`, prints a breakdown of how many candidate groups exist
    before filtering and how many were rejected for each reason. This is the
    diagnostic that previously got swallowed by the laconic "no eligible
    groups" message.
    """
    n_rows = len(df)
    raw_distinct = df[col].nunique(dropna=False)
    known = df[col].notna() & df[col].astype(str).str.lower().ne("unknown")
    n_known = int(known.sum())
    total_events = int(df["event"].sum())
    total_censors = int(len(df) - total_events)
    if verbose:
        print(
            f"  LOSO[{col}] candidates: n_rows={n_rows:,} known_rows={n_known:,} "
            f"({100.0*n_known/max(n_rows,1):.1f}%) distinct_values={raw_distinct:,}"
        )
    if n_known == 0:
        if verbose:
            print(f"  LOSO[{col}] no known (non-'unknown') values; column is empty.")
        return []

    summary = (
        df[known]
        .groupby(col, sort=False)
        .agg(n=("event", "size"), events=("event", "sum"))
        .reset_index()
    )
    summary["censors"] = summary["n"] - summary["events"]

    fail_n         = (summary["n"] < min_test_n)
    fail_events    = (summary["events"] < min_test_events)
    fail_censors   = (summary["censors"] < min_test_censors)
    fail_train_ev  = ((total_events - summary["events"]) < min_train_events)
    fail_train_cn  = ((total_censors - summary["censors"]) < min_train_censors)

    if verbose:
        print(
            f"  LOSO[{col}] groups_before_filter={len(summary):,}  "
            f"reject: n<{min_test_n}={int(fail_n.sum())} "
            f"events<{min_test_events}={int(fail_events.sum())} "
            f"censors<{min_test_censors}={int(fail_censors.sum())} "
            f"train_events<{min_train_events}={int(fail_train_ev.sum())} "
            f"train_censors<{min_train_censors}={int(fail_train_cn.sum())}"
        )

    summary = summary[~(fail_n | fail_events | fail_censors | fail_train_ev | fail_train_cn)]
    summary = summary.sort_values(["n", "events"], ascending=False, kind="stable")
    if verbose and not summary.empty:
        preview = summary.head(5)[[col, "n", "events", "censors"]].to_dict("records")
        print(f"  LOSO[{col}] eligible (top 5 by size): {preview}")
    if max_groups is not None:
        summary = summary.head(max_groups)
    return [str(x) for x in summary[col].tolist()]


def run_loso_axis(
    df_full: pd.DataFrame,
    axis_name: str,
    group_col: str,
    pc_cols: list[str],
    pgs_id: str,
    min_train_events: int,
    min_train_censors: int,
    min_test_events: int,
    min_test_censors: int,
    min_test_n: int,
    max_groups: int | None = None,
    score_train: bool = False,
    mode: str = "both",
) -> None:
    """Leave one group out: refit both models on all other groups.

    `mode` is threaded into `evaluate_model_pair`. The LOSO summary
    metric is the survival ΔC-ipcw vs baselineA when survival fits ran,
    and the binary ΔAUROC vs baselineB when only binary fits ran.
    """
    if mode == "binary":
        summary_key = "binary_delta_auroc_vs_baselineB"
        summary_label = "binary_delta_auroc_GAM-B"
    else:
        summary_key = "delta_test_c_ipcw_vs_baselineA"
        summary_label = "delta_C_ipcw"
    groups = loso_groups(
        df_full,
        group_col,
        min_train_events=min_train_events,
        min_train_censors=min_train_censors,
        min_test_events=min_test_events,
        min_test_censors=min_test_censors,
        min_test_n=min_test_n,
        max_groups=max_groups,
    )
    print(
        f"  LOSO axis={axis_name} col={group_col} groups={len(groups)} "
        f"min_test_n={min_test_n} min_test_events={min_test_events} "
        f"min_test_censors={min_test_censors}"
    )
    if not groups:
        print(f"  LOSO axis={axis_name} skipped=no eligible groups")
        return

    deltas: list[float] = []
    for group in groups:
        holdout_mask = df_full[group_col].eq(group)
        train = df_full.loc[~holdout_mask].reset_index(drop=True)
        test = df_full.loc[holdout_mask].reset_index(drop=True)
        group_short = group[:96]
        print(
            f"  LOSO fold axis={axis_name} holdout={group_short!r}  "
            f"train_n={len(train):,} test_n={len(test):,} "
            f"test_events={int(test['event'].sum()):,}"
        )
        result = evaluate_model_pair(
            train,
            test,
            pc_cols,
            pgs_id,
            label=f"LOSO[{axis_name}] holdout={group_short!r}",
            population_prevalence=float(df_full["event"].mean()),
            score_train=score_train,
            mode=mode,
        )
        deltas.append(result[summary_key])

    print(
        f"  LOSO summary axis={axis_name} folds={len(deltas)}  "
        f"mean_{summary_label}={float(np.mean(deltas)):+.4f}  "
        f"worst_{summary_label}={float(np.min(deltas)):+.4f}  "
        f"best_{summary_label}={float(np.max(deltas)):+.4f}"
    )


# --- main ------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Fit marginal-slope GAMs on All of Us microarray data.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "survival", "binary"),
        default="both",
        help=(
            "Which fits to run per disease. 'both' (default): survival GAM "
            "+ Cox baselines and binary GAM + logistic baselines. "
            "'survival': skip the binary path. 'binary': skip the survival "
            "GAM, Cox baselines, sksurv metrics, and .gamfit artifact write."
        ),
    )
    args = parser.parse_args()
    mode = args.mode
    print(f"run mode: {mode}")
    import gamfit
    print(f"gamfit version: {gamfit.__version__}")
    print_gamfit_diagnostics(gamfit)
    # The real "did the previous fit survive?" cache is gamfit's persistent
    # warm-start store, not the .gamfit files in FITS_DIR. It auto-resumes
    # any fit on identical (data, version) keys; reruns are typically a
    # handful of PIRLS cycles instead of hundreds. .gamfit files are for
    # artifact persistence (sharing the fitted model out-of-process).
    warm_cache_root = user_cache_root() / "gam" / "warm" / "v1"
    print(
        f"warm-start cache: {warm_cache_root} "
        f"({'present' if warm_cache_root.exists() else 'will be created on first fit'})"
    )
    print(f"artifact cache:   {FITS_DIR}")
    active_axes = list(LOSO_AXES)
    print(f"loso_axes: {active_axes}")

    cdr = os.environ["WORKSPACE_CDR"]
    client = bigquery.Client()

    diseases = select_runtime_diseases(client, cdr)
    diseases = {k: v for k, v in diseases.items() if PGS_ID_PATTERN.match(v["pgs"])}
    if not diseases:
        raise SystemExit(
            "no diseases survived OHDSI canonical ∩ SNOMED_PGS_MAP — extend "
            "SNOMED_PGS_MAP or verify OHDSI extraction."
        )
    print(f"diseases with real PGS IDs: {list(diseases)}")

    ensure_scored([cfg["pgs"] for cfg in diseases.values()])

    print("loading PCs and sex ...")
    pcs = load_pcs(NUM_PCS)
    sex = load_sex(client=client, cdr=cdr)
    base = pcs.merge(sex, on="person_id")
    print(f"base: n={len(base):,}")

    print("loading person times (birth + observation period) ...")
    times = fetch_person_times(client, cdr)
    base = base.merge(times, on="person_id")
    print(f"base (with times): n={len(base):,}")

    print("loading geography and care-site context ...")
    context = fetch_person_context(client, cdr)
    base = base.merge(context, on="person_id", how="left")
    base["census_region"] = _clean_group_label(base["census_region"])
    base["care_site_group"] = _clean_group_label(base["care_site_group"])
    print(f"base (with context): n={len(base):,}")

    print("loading AoU inferred genetic ancestry labels ...")
    try:
        ancestry = load_genetic_ancestry_labels()
    except Exception as exc:
        # Bare-metal VMs sit outside the AoU VPC-SC perimeter, so the controlled
        # bucket is unreachable. If a prior workbench-side `gsutil cp` hasn't
        # staged ANCESTRY_PREDS_CACHE here, drop the `ancestry` LOSO axis and
        # finish the rest of the run (random split + care_site + census_region).
        first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        print(f"  WARNING: ancestry labels unavailable -> dropping 'ancestry' LOSO axis. detail: {first_line}")
        active_axes = [a for a in active_axes if a != "ancestry"]
    else:
        base = base.merge(ancestry, on="person_id", how="left")
        base["ancestry_category"] = _clean_group_label(base["ancestry_category"]).str.upper()
        print(f"base (with ancestry): n={len(base):,}")

    rng = np.random.default_rng(RNG_SEED)
    pc_cols = [f"PC{i+1}" for i in range(NUM_PCS)]

    # --- pre-flight diagnostics: print upfront, aggregate-only, no row-level
    # data. Tells the operator what LOSO axes will have anything to work with
    # before any fits start.
    print("\n=== PRE-FLIGHT DIAGNOSTICS ===")
    print(f"base cohort           : n={len(base):,}")
    print(f"PC columns            : {pc_cols}")
    print(f"diseases to fit       : {list(diseases.keys())}")
    print(f"LOSO axes (configured): {LOSO_AXES}")
    print(f"LOSO axes (active)    : {active_axes}")
    print(f"LOSO thresholds       : min_test_n={MIN_LOSO_TEST_N} "
          f"min_test_events={MIN_LOSO_TEST_EVENTS} "
          f"min_test_censors={MIN_LOSO_TEST_CENSORS} "
          f"min_train_events={MIN_LOSO_TRAIN_EVENTS} "
          f"min_train_censors={MIN_LOSO_TRAIN_CENSORS} "
          f"max_groups={MAX_LOSO_CARE_SITES}")
    for axis in active_axes:
        col = LOSO_AXIS_TO_COLUMN[axis]
        if col not in base.columns:
            print(f"  axis={axis:<14} column={col:<24} status=MISSING (column not in base)")
            continue
        s = base[col].astype(str)
        known_mask = s.notna() & s.str.lower().ne("unknown") & s.ne("")
        n_known = int(known_mask.sum())
        n_total = len(base)
        distinct = int(s[known_mask].nunique())
        pct = 100.0 * n_known / max(n_total, 1)
        # Show only group sizes that meet the LOSO min_test_n threshold so we
        # never emit a low-count bucket. Top 8 by population.
        if n_known > 0:
            vc = s[known_mask].value_counts()
            big = vc[vc >= MIN_LOSO_TEST_N]
            print(
                f"  axis={axis:<14} column={col:<24} "
                f"known={n_known:,} ({pct:.1f}%) distinct={distinct:,} "
                f"buckets>={MIN_LOSO_TEST_N}: {len(big):,}"
            )
            if len(big) > 0:
                preview = dict(big.head(8).items())
                print(f"      top buckets (n>={MIN_LOSO_TEST_N}): {preview}")
            else:
                print(f"      no bucket has >={MIN_LOSO_TEST_N} people — axis will be empty downstream")
        else:
            print(
                f"  axis={axis:<14} column={col:<24} "
                f"known=0 ({pct:.1f}%)  -> axis will be skipped"
            )
    print("=== /PRE-FLIGHT ===\n")

    for name, cfg in diseases.items():
        print(f"\n=== {name.upper()} ===")
        pgs_df = load_one_pgs(cfg["pgs"])
        df_full = base.merge(pgs_df, on="person_id")
        print(
            f"  merge[base+pgs]: base={len(base):,} pgs={len(pgs_df):,} "
            f"merged={len(df_full):,}"
        )
        if len(df_full) == 0:
            base_ids = set(base["person_id"].head(5))
            pgs_ids = set(pgs_df["person_id"].head(5))
            print(f"    base sample person_id={sorted(base_ids)}")
            print(f"    pgs  sample person_id={sorted(pgs_ids)}")
            print(
                f"    base dtype={base['person_id'].dtype} "
                f"pgs dtype={pgs_df['person_id'].dtype}"
            )
        ancestor = int(cfg["concept_id"])
        case_dates = fetch_cases(client, cdr, ancestor)
        before_cases = len(df_full)
        df_full = df_full.merge(case_dates, on="person_id", how="left")
        matched_cases = int(df_full["event_date"].notna().sum())
        print(
            f"  merge[+cases]: pre={before_cases:,} cases_returned={len(case_dates):,} "
            f"matched={matched_cases:,}"
        )
        if matched_cases == 0 and len(case_dates) > 0 and before_cases > 0:
            base_ids = set(df_full["person_id"].head(5))
            case_ids = set(case_dates["person_id"].head(5))
            print(f"    df_full sample person_id={sorted(base_ids)}")
            print(f"    cases   sample person_id={sorted(case_ids)}")
            print(
                f"    df_full dtype={df_full['person_id'].dtype} "
                f"cases dtype={case_dates['person_id'].dtype}"
            )
        df_full["event"] = df_full["event_date"].notna().astype(int)

        # Age-as-time-scale (years). Entry: age at AoU obs-period start.
        # Exit: age at first qualifying condition for events; age at obs-period
        # end for censors. Left-truncation respected via `Surv(entry, exit, event)`.
        #
        # `current_age` (age at obs_end) is the fourth age column. Unlike
        # `exit_age` (event onset for cases, obs_end for censors -- endogenous
        # to the label) and `followup_years = exit_age - entry_age`, which
        # would be label-leaky if used as a binary covariate, `current_age`
        # depends only on birth_datetime and obs_end and is exogenous to the
        # outcome. The binary path then materialises a single matched age
        # basis used by every model — a quadratic in z-scored `current_age`
        # (`current_age_z`, `current_age_z2`) plus linear `entry_age_z` —
        # so age is held identical across the GAM and the logistic
        # baselines.
        days_per_year = 365.25
        df_full["entry_age"] = (
            (df_full["obs_start"] - df_full["birth_datetime"]).dt.days / days_per_year
        )
        exit_date = df_full["event_date"].fillna(df_full["obs_end"])
        df_full["exit_age"] = (
            (exit_date - df_full["birth_datetime"]).dt.days / days_per_year
        )
        df_full["current_age"] = (
            (df_full["obs_end"] - df_full["birth_datetime"]).dt.days / days_per_year
        )

        # Drop rows with non-positive or invalid intervals: missing birth/obs,
        # prevalent cases whose event predates obs_start (entry >= exit), etc.
        before = len(df_full)
        miss = {c: int(df_full[c].isna().sum()) for c in
                ("pgs", "entry_age", "exit_age", "current_age",
                 "birth_datetime", "obs_start", "obs_end")}
        df_full = df_full.dropna(
            subset=["pgs", "entry_age", "exit_age", "current_age"]
        ).copy()
        after_dropna = len(df_full)
        df_full = df_full[df_full["exit_age"] > df_full["entry_age"]].copy()
        after_exit_gt_entry = len(df_full)
        df_full = df_full[df_full["entry_age"] >= 0].copy()
        after_entry_nonneg = len(df_full)
        df_full = df_full[df_full["current_age"] >= df_full["entry_age"]].copy()
        dropped = before - len(df_full)
        print(
            f"  filters: pre={before:,} "
            f"after_dropna={after_dropna:,} "
            f"after_exit>entry={after_exit_gt_entry:,} "
            f"after_entry>=0={after_entry_nonneg:,} "
            f"final={len(df_full):,}  na_counts={miss}"
        )
        n_event = int(df_full["event"].sum())
        n_censor = len(df_full) - n_event
        K = n_event / max(1, len(df_full))
        print(
            f"  snomed={cfg['snomed_name']!r}  concept_id={ancestor}  "
            f"events={n_event:,}  censors={n_censor:,}  K(crude)={K:.6f}  "
            f"dropped_bad_intervals={dropped:,}"
        )

        # Balanced events:censors = 1:1, then 80/20 train/test per class.
        # Survival likelihood *can* use every censor, but per-cycle PIRLS
        # cost scales with n_train and statistical power is event-bound,
        # so downsample censors to match event count up front.
        event_idx_all = rng.permutation(df_full.index[df_full["event"] == 1].to_numpy())
        censor_idx_all = rng.permutation(df_full.index[df_full["event"] == 0].to_numpy())
        n_keep = min(len(event_idx_all), len(censor_idx_all))
        event_idx = event_idx_all[:n_keep]
        censor_idx = censor_idx_all[:n_keep]
        n_train_event = int(round(n_keep * TRAIN_FRACTION))
        n_train_censor = int(round(n_keep * TRAIN_FRACTION))
        train_pick = np.concatenate([
            event_idx[:n_train_event], censor_idx[:n_train_censor],
        ])
        test_pick = np.concatenate([
            event_idx[n_train_event:], censor_idx[n_train_censor:],
        ])
        print(
            f"  split: n={len(df_full):,}  "
            f"train_events={n_train_event:,} train_censors={n_train_censor:,}  "
            f"test_events={len(event_idx) - n_train_event:,} "
            f"test_censors={len(censor_idx) - n_train_censor:,}"
        )
        train = df_full.loc[train_pick].reset_index(drop=True)
        test = df_full.loc[test_pick].reset_index(drop=True)

        evaluate_model_pair(
            train,
            test,
            pc_cols,
            cfg["pgs"],
            label=f"PGS={cfg['pgs']} random_split K(crude)={K:.6f}",
            population_prevalence=K,
            save_info={
                "disease": name,
                "pgs": cfg["pgs"],
                "snomed_name": cfg["snomed_name"],
                "concept_id": ancestor,
                "K_crude": K,
            },
            mode=mode,
        )

        print("  OOD: leave-one-group-out refits")
        for axis in active_axes:
            run_loso_axis(
                df_full,
                axis_name=axis,
                group_col=LOSO_AXIS_TO_COLUMN[axis],
                pc_cols=pc_cols,
                pgs_id=cfg["pgs"],
                min_train_events=MIN_LOSO_TRAIN_EVENTS,
                min_train_censors=MIN_LOSO_TRAIN_CENSORS,
                min_test_events=MIN_LOSO_TEST_EVENTS,
                min_test_censors=MIN_LOSO_TEST_CENSORS,
                min_test_n=MIN_LOSO_TEST_N,
                max_groups=MAX_LOSO_CARE_SITES if axis == "care_site" else None,
                score_train=False,
                mode=mode,
            )


if __name__ == "__main__":
    main()
