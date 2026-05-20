"""Core implementation: subprocess wrappers + sscore parser."""

from __future__ import annotations

import csv
import enum
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GnomonError(Exception):
    """Base class for all gnomon wrapper errors."""


class GnomonBinaryNotFound(GnomonError, FileNotFoundError):
    """The `gnomon` binary could not be located."""


class InvalidConfig(GnomonError, ValueError):
    """An argument combination is invalid before launching."""


class GnomonFailed(GnomonError, RuntimeError):
    """gnomon exited non-zero."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class SscoreParseError(GnomonError, ValueError):
    """Failed to parse a .sscore file."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InferredSex(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Binary location
# ---------------------------------------------------------------------------


def locate_binary(override: Optional[PathLike] = None, *, name: str = "gnomon") -> Path:
    """Locate the `gnomon` (or `gnomon-score`/etc.) binary or raise.

    Resolution order:
      1. ``override`` argument.
      2. ``$GNOMON_BIN`` env var (only when ``name`` is ``"gnomon"``).
      3. ``name`` on ``PATH``.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise GnomonBinaryNotFound(f"gnomon binary not at {p}")
        return p
    if name == "gnomon":
        env = os.environ.get("GNOMON_BIN")
        if env:
            p = Path(env)
            if not p.exists():
                raise GnomonBinaryNotFound(f"$GNOMON_BIN -> nonexistent {p}")
            return p
    which = shutil.which(name)
    if which:
        return Path(which)
    raise GnomonBinaryNotFound(
        f"{name} not found. Install with: cargo install gnomon"
    )


# ---------------------------------------------------------------------------
# Sscore table parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreTable:
    """Parsed contents of a ``.sscore`` file.

    The file is a TSV with FID/IID identity columns and per-score columns
    suffixed ``_AVG``, ``_SUM``, and ``_DENOM`` (the latter two omitted in
    some builds). We expose:

      * ``fids`` / ``iids`` — parallel string sequences (length N samples).
      * ``score_names`` — names with the suffix stripped, in column order.
      * ``avg`` — list-of-lists ``N x K`` (or ``None`` if column missing).
      * ``sum`` / ``denom`` — same shape, may be ``None`` if absent.

    ``__len__`` returns the number of samples; ``shape`` returns
    ``(n_samples, n_columns_including_ids)``.
    """

    fids: Tuple[str, ...]
    iids: Tuple[str, ...]
    score_names: Tuple[str, ...]
    avg: Optional[Tuple[Tuple[float, ...], ...]] = None
    sum: Optional[Tuple[Tuple[float, ...], ...]] = None
    denom: Optional[Tuple[Tuple[float, ...], ...]] = None

    def __len__(self) -> int:
        return len(self.iids)

    @property
    def shape(self) -> Tuple[int, int]:
        cols = 2 + len(self.score_names) * sum(x is not None for x in (self.avg, self.sum, self.denom))
        return (len(self.iids), cols)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fids": list(self.fids),
            "iids": list(self.iids),
            "score_names": list(self.score_names),
            "avg": [list(r) for r in self.avg] if self.avg else None,
            "sum": [list(r) for r in self.sum] if self.sum else None,
            "denom": [list(r) for r in self.denom] if self.denom else None,
        }

    def to_pandas(self):
        """Return a pandas DataFrame. Requires pandas to be installed."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("pandas is required for to_pandas()") from e
        cols: Dict[str, Any] = {"FID": list(self.fids), "IID": list(self.iids)}
        for suffix, matrix in (("AVG", self.avg), ("SUM", self.sum), ("DENOM", self.denom)):
            if matrix is None:
                continue
            for i, name in enumerate(self.score_names):
                cols[f"{name}_{suffix}"] = [row[i] for row in matrix]
        return pd.DataFrame(cols)


def read_sscore(path: PathLike) -> ScoreTable:
    """Parse a ``.sscore`` file produced by ``gnomon score``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as e:
            raise SscoreParseError(f"{p}: empty file") from e
        # The header's first column starts with '#'.
        if not header or not header[0].lstrip("#").upper().startswith(("FID", "IID")):
            raise SscoreParseError(
                f"{p}: first column should start with #FID or #IID, got {header[0]!r}"
            )
        # Identify FID / IID columns.
        norm = [c.lstrip("#") for c in header]
        try:
            iid_idx = norm.index("IID")
        except ValueError as e:
            raise SscoreParseError(f"{p}: no IID column in header {header}") from e
        fid_idx = norm.index("FID") if "FID" in norm else None

        # Bucket score columns by suffix.
        avg_cols: List[Tuple[int, str]] = []
        sum_cols: List[Tuple[int, str]] = []
        den_cols: List[Tuple[int, str]] = []
        for i, name in enumerate(norm):
            if name in ("FID", "IID"):
                continue
            if name.endswith("_AVG"):
                avg_cols.append((i, name[:-4]))
            elif name.endswith("_SUM"):
                sum_cols.append((i, name[:-4]))
            elif name.endswith("_DENOM"):
                den_cols.append((i, name[:-6]))
            # Anything else gets silently dropped — gnomon historically
            # only writes these three suffixes.

        # Canonical score name order: union, preferring AVG order then SUM/DENOM.
        score_names_in_order: List[str] = []
        seen = set()
        for bucket in (avg_cols, sum_cols, den_cols):
            for _, n in bucket:
                if n not in seen:
                    seen.add(n)
                    score_names_in_order.append(n)

        # Pre-compute column index per score per suffix.
        def index_for(cols, name):
            for i, n in cols:
                if n == name:
                    return i
            return None

        idx_avg = [index_for(avg_cols, n) for n in score_names_in_order]
        idx_sum = [index_for(sum_cols, n) for n in score_names_in_order]
        idx_den = [index_for(den_cols, n) for n in score_names_in_order]
        have_avg = any(i is not None for i in idx_avg)
        have_sum = any(i is not None for i in idx_sum)
        have_den = any(i is not None for i in idx_den)

        fids: List[str] = []
        iids: List[str] = []
        rows_avg: List[Tuple[float, ...]] = []
        rows_sum: List[Tuple[float, ...]] = []
        rows_den: List[Tuple[float, ...]] = []

        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                iids.append(row[iid_idx])
            except IndexError as e:
                raise SscoreParseError(f"{p}:{line_no}: row shorter than header") from e
            fids.append(row[fid_idx] if fid_idx is not None else row[iid_idx])
            if have_avg:
                rows_avg.append(tuple(_safe_float(row[i]) if i is not None else float("nan") for i in idx_avg))
            if have_sum:
                rows_sum.append(tuple(_safe_float(row[i]) if i is not None else float("nan") for i in idx_sum))
            if have_den:
                rows_den.append(tuple(_safe_float(row[i]) if i is not None else float("nan") for i in idx_den))

    return ScoreTable(
        fids=tuple(fids),
        iids=tuple(iids),
        score_names=tuple(score_names_in_order),
        avg=tuple(rows_avg) if have_avg else None,
        sum=tuple(rows_sum) if have_sum else None,
        denom=tuple(rows_den) if have_den else None,
    )


def _safe_float(s: str) -> float:
    if s in ("", "NA", "nan", "NaN"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of a ``gnomon score`` run."""

    output_path: Path
    """The ``.sscore`` file emitted by the run."""

    scores: ScoreTable
    """Parsed contents of ``output_path``."""

    stdout: str
    stderr: str
    returncode: int

    @property
    def n_samples(self) -> int:
        return len(self.scores)

    @property
    def score_names(self) -> Tuple[str, ...]:
        return self.scores.score_names


@dataclass(frozen=True)
class TermsResult:
    """Outcome of a ``gnomon terms`` run."""

    inferred_sex: Optional[InferredSex]
    sex_output_path: Optional[Path]
    stdout: str
    stderr: str
    returncode: int
    sex_table: Optional[Tuple[Tuple[str, str], ...]] = None


@dataclass(frozen=True)
class MapResult:
    output_paths: Tuple[Path, ...]
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class CalibrateResult:
    output_paths: Tuple[Path, ...]
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class AllResult:
    score: Optional[ScoreResult]
    map: Optional[MapResult]
    terms: Optional[TermsResult]
    stdout: str
    stderr: str
    returncode: int


# ---------------------------------------------------------------------------
# Run helper
# ---------------------------------------------------------------------------


def _run(
    binary: Path,
    argv: Sequence[str],
    *,
    timeout: Optional[float],
    env: Optional[Mapping[str, str]],
    cwd: Optional[PathLike],
    check: bool = True,
) -> subprocess.CompletedProcess:
    full_argv = [str(binary), *map(str, argv)]
    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)
    try:
        completed = subprocess.run(
            full_argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=proc_env,
            cwd=cwd,
        )
    except FileNotFoundError as e:
        raise GnomonBinaryNotFound(str(e)) from e
    if check and completed.returncode != 0:
        # Pull out the last error line from stderr, if there is one, for the
        # exception message.
        last = ""
        for line in reversed((completed.stderr or "").splitlines()):
            if line.strip():
                last = line.strip()
                break
        raise GnomonFailed(
            f"gnomon exited {completed.returncode}: {last}" if last else f"gnomon exited {completed.returncode}",
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            returncode=completed.returncode,
        )
    return completed


# ---------------------------------------------------------------------------
# `score` subcommand
# ---------------------------------------------------------------------------


# Score is invoked as: `gnomon score <SCORE_PATH> <GENOTYPE_PATH> [flags...]`
# Output file naming logic (from score/main.rs):
#   - For a PGS-style argument like "PGS004536,PGS001320":
#       <genotype_prefix>_PGS004536-PGS001320.sscore (joined by `-`).
#   - For a single score file: <genotype_prefix>_<score_stem>.sscore
#   - For a directory of score files: <genotype_prefix>.sscore (no suffix).
# We replicate this logic so the wrapper can locate the output file.


_PGS_ID_RE = re.compile(r"^PGS\d{6}$", re.IGNORECASE)


def _looks_like_pgs_arg(score_arg: str) -> bool:
    parts = [p.strip() for p in score_arg.split(",")]
    return bool(parts) and all(_PGS_ID_RE.match(p) for p in parts)


_COMPOUND_INPUT_SUFFIXES = (".vcf.bgz", ".vcf.gz", ".bcf.bgz", ".bcf.gz")
_SIMPLE_INPUT_SUFFIXES = (".bed", ".vcf", ".bcf")


def _score_output_stem(path: Path) -> str:
    """Mirror score::main::score_output_stem (Rust) byte-for-byte."""
    name = path.name
    if not name:
        return "gnomon_results"
    lower = name.lower()
    for suffix in _COMPOUND_INPUT_SUFFIXES + _SIMPLE_INPUT_SUFFIXES:
        if lower.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return name


def _expected_sscore_path(input_path: Path, score_arg: str) -> Path:
    """Mirror score::main::score_output_path."""
    stem = _score_output_stem(input_path)
    parent = input_path.parent if str(input_path.parent) else Path(".")
    if _looks_like_pgs_arg(score_arg):
        suffix = "-".join(p.strip().upper() for p in score_arg.split(","))
        return parent / f"{stem}_{suffix}.sscore"
    score_path = Path(score_arg)
    if score_path.exists() and score_path.is_dir():
        return parent / f"{stem}.sscore"
    score_stem = score_path.stem
    return parent / f"{stem}_{score_stem}.sscore"


def score(
    score: Union[str, PathLike, Sequence[str]],
    input_path: PathLike,
    *,
    keep: Optional[PathLike] = None,
    reference: Optional[PathLike] = None,
    build: Optional[str] = None,
    panel: Optional[PathLike] = None,
    inferred_sex: Optional[Union[InferredSex, str]] = None,
    output_path: Optional[PathLike] = None,
    binary: Optional[PathLike] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
    read_output: bool = True,
) -> ScoreResult:
    """Run ``gnomon score``.

    ``score`` is either:
      * A comma-separated list of PGS Catalog IDs: ``"PGS004536,PGS001320"``.
      * A list/tuple of PGS IDs: ``["PGS004536", "PGS001320"]``.
      * A path to a single score file or directory of score files.

    ``input_path`` is the genotype source (PLINK prefix, VCF, BCF, DTC).
    """
    if isinstance(score, (list, tuple)):
        score_arg = ",".join(score)
    else:
        score_arg = str(score)
    input_p = Path(input_path)
    bin_path = locate_binary(binary)

    argv: List[str] = ["score", score_arg, str(input_p)]
    if keep is not None:
        argv += ["--keep", str(keep)]
    if reference is not None:
        argv += ["--reference", str(reference)]
    if build is not None:
        argv += ["--build", str(build)]
    if panel is not None:
        argv += ["--panel", str(panel)]
    if inferred_sex is not None:
        argv += ["--inferred-sex", _coerce_enum(inferred_sex, InferredSex).value]
    if extra_args:
        argv += list(extra_args)

    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd)

    expected = Path(output_path) if output_path else _expected_sscore_path(input_p, score_arg)
    if not expected.exists():
        # Try a fallback: scan stdout for a path-like artefact.
        m = re.search(r"([^\s]+\.sscore)\b", completed.stdout + "\n" + completed.stderr)
        candidate = Path(m.group(1)) if m else expected
        if not candidate.exists():
            raise GnomonFailed(
                f"gnomon score exited 0 but no .sscore output found "
                f"(expected {expected}).",
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )
        expected = candidate

    table = read_sscore(expected) if read_output else ScoreTable((), (), ())
    return ScoreResult(
        output_path=expected,
        scores=table,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


# ---------------------------------------------------------------------------
# `terms` subcommand
# ---------------------------------------------------------------------------


_SEX_LINE_RE = re.compile(
    r"^(?P<iid>\S+)\s+(?P<sex>male|female|unknown|indeterminate)\b",
    re.IGNORECASE,
)


def terms(
    genotype_path: PathLike,
    *,
    sex: bool = True,
    binary: Optional[PathLike] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> TermsResult:
    """Run ``gnomon terms``. Returns inferred sex and the TSV output path if any."""
    if not sex:
        raise InvalidConfig("terms() currently requires sex=True (the only available term).")

    gp = Path(genotype_path)
    bin_path = locate_binary(binary)
    argv = ["terms", str(gp), "--sex"]
    if extra_args:
        argv += list(extra_args)
    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd)

    # The CLI writes the sex TSV next to the genotype prefix as
    # `<stem>_sex.tsv`. Resolve it.
    stem = gp.stem if gp.suffix in {".bed", ".vcf", ".bcf", ".gz"} else gp.name
    candidates = [
        gp.parent / f"{stem}_sex.tsv",
        gp.parent / f"{stem}.sex.tsv",
        gp.parent / f"{gp.name}_sex.tsv",
    ]
    out_path: Optional[Path] = next((c for c in candidates if c.exists()), None)

    sex_table: Optional[Tuple[Tuple[str, str], ...]] = None
    inferred: Optional[InferredSex] = None
    if out_path is not None:
        with open(out_path) as f:
            reader = csv.reader(f, delimiter="\t")
            rows: List[Tuple[str, str]] = []
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    continue
                # Skip a header line if present (IID/FID/SEX in the second cell).
                if row[1].strip().lower() in {"sex", "inferred_sex", "inferredsex"}:
                    continue
                rows.append((row[0], row[1]))
            sex_table = tuple(rows)
            if len(rows) == 1:
                inferred = _coerce_sex(rows[0][1])
    else:
        # Single-sample dataset: try parsing stdout.
        for line in (completed.stdout or "").splitlines():
            m = _SEX_LINE_RE.match(line)
            if m:
                inferred = _coerce_sex(m.group("sex"))
                break

    return TermsResult(
        inferred_sex=inferred,
        sex_output_path=out_path,
        sex_table=sex_table,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def _coerce_sex(s: str) -> InferredSex:
    s = s.strip().lower()
    if s == "male":
        return InferredSex.MALE
    if s == "female":
        return InferredSex.FEMALE
    return InferredSex.UNKNOWN


def _coerce_enum(value, enum_cls):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.lower())
        except ValueError:
            for member in enum_cls:
                if member.name.lower() == value.lower():
                    return member
    raise InvalidConfig(f"Cannot coerce {value!r} to {enum_cls.__name__}")


# ---------------------------------------------------------------------------
# `all` subcommand
# ---------------------------------------------------------------------------


def run_all(
    score: Union[str, PathLike, Sequence[str]],
    input_path: PathLike,
    model: str,
    *,
    keep: Optional[PathLike] = None,
    reference: Optional[PathLike] = None,
    build: Optional[str] = None,
    panel: Optional[PathLike] = None,
    output_manifest: Optional[PathLike] = None,
    binary: Optional[PathLike] = None,
    extra_args: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[PathLike] = None,
) -> AllResult:
    """Run ``gnomon all`` — score + project + terms in one pass."""
    score_arg = ",".join(score) if isinstance(score, (list, tuple)) else str(score)
    input_p = Path(input_path)
    bin_path = locate_binary(binary)
    argv: List[str] = ["all", score_arg, str(input_p), "--model", model]
    if keep is not None:
        argv += ["--keep", str(keep)]
    if reference is not None:
        argv += ["--reference", str(reference)]
    if build is not None:
        argv += ["--build", str(build)]
    if panel is not None:
        argv += ["--panel", str(panel)]
    if output_manifest is not None:
        argv += ["--output-manifest", str(output_manifest)]
    if extra_args:
        argv += list(extra_args)

    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd)

    # Best-effort: pick up the score artefact if it exists.
    expected = _expected_sscore_path(input_p, score_arg)
    score_result: Optional[ScoreResult] = None
    if expected.exists():
        score_result = ScoreResult(
            output_path=expected,
            scores=read_sscore(expected),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    return AllResult(
        score=score_result,
        map=None,
        terms=None,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
