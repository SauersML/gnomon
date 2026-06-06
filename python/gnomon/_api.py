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
      2. ``name`` on ``PATH``.

    No environment-variable indirection — pass ``binary=`` explicitly
    when you want to use a non-PATH copy.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise GnomonBinaryNotFound(f"gnomon binary not at {p}")
        return p
    which = shutil.which(name)
    if which:
        return Path(which)
    raise GnomonBinaryNotFound(
        f"{name} not found. Install with `cargo install gnomon`, "
        f"or pass binary=... explicitly."
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

    # -- ergonomic accessors --------------------------------------------------

    def index_of(self, iid: str) -> int:
        """Return the row index for ``iid``. Raises KeyError if not present."""
        try:
            return self.iids.index(iid)
        except ValueError as e:
            raise KeyError(iid) from e

    def __contains__(self, iid: object) -> bool:
        return isinstance(iid, str) and iid in self.iids

    def __getitem__(self, iid: str) -> Dict[str, Dict[str, Optional[float]]]:
        """Look up one sample by IID.

        Returns a dict shaped::

            {score_name: {"avg": float | None, "sum": float | None, "denom": float | None}}

        Each sub-value is ``None`` if the corresponding column was missing
        from the .sscore file.
        """
        idx = self.index_of(iid)
        out: Dict[str, Dict[str, Optional[float]]] = {}
        for i, name in enumerate(self.score_names):
            out[name] = {
                "avg": self.avg[idx][i] if self.avg is not None else None,
                "sum": self.sum[idx][i] if self.sum is not None else None,
                "denom": self.denom[idx][i] if self.denom is not None else None,
            }
        return out

    def score_for(self, iid: str, score_name: str, *, kind: str = "avg") -> float:
        """Return one cell. ``kind`` is ``'avg'``, ``'sum'``, or ``'denom'``."""
        kind = kind.lower()
        matrix_map = {"avg": self.avg, "sum": self.sum, "denom": self.denom}
        if kind not in matrix_map:
            raise ValueError(f"kind must be 'avg', 'sum', or 'denom', got {kind!r}")
        matrix = matrix_map[kind]
        if matrix is None:
            raise KeyError(
                f"{kind!r} column not present in this .sscore file "
                f"(file had columns: {self._available_kinds()})"
            )
        try:
            col = self.score_names.index(score_name)
        except ValueError as e:
            raise KeyError(score_name) from e
        return matrix[self.index_of(iid)][col]

    def _available_kinds(self) -> Tuple[str, ...]:
        return tuple(
            k for k, v in (("avg", self.avg), ("sum", self.sum), ("denom", self.denom)) if v is not None
        )


def read_sscore(path: PathLike) -> ScoreTable:
    """Parse a ``.sscore`` file produced by ``gnomon score``.

    Streams the TSV with the stdlib ``csv`` module — fast and dependency-
    free for typical biobank-scale outputs (millions of rows, dozens of
    scores). Missing values (``""``, ``NA``, ``nan``) become ``NaN``.
    """
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
class SexMetrics:
    """Per-sample sex-inference metrics, parsed from a ``.sex.tsv`` row.

    These mirror the metric columns the gnomon CLI writes in
    ``terms/sex.rs`` (``SEX_TSV_HEADER``):

      ``IID  Build  Sex  Y_Density  X_AutoHet_Ratio  Composite_Index
       Auto_Valid  Auto_Het  X_NonPAR_Valid  X_NonPAR_Het
       Y_NonPAR_Valid  Y_PAR_Valid``

    The string ``"NA"`` in any numeric column becomes ``None``.

    ``composite_index`` is gnomon's single combined sex-discriminant statistic
    (a continuous male/female separation score); it is surfaced as
    :attr:`confidence` for callers that just want one number alongside the
    call. The raw ratios (``y_density``, ``x_autosome_het_ratio``) and the
    underlying counts let downstream code reproduce gnomon's thresholding or
    apply its own QC without re-reading the TSV.
    """

    iid: str
    sex: InferredSex
    build: Optional[str] = None
    y_density: Optional[float] = None
    x_autosome_het_ratio: Optional[float] = None
    composite_index: Optional[float] = None
    auto_valid: Optional[int] = None
    auto_het: Optional[int] = None
    x_non_par_valid: Optional[int] = None
    x_non_par_het: Optional[int] = None
    y_non_par_valid: Optional[int] = None
    y_par_valid: Optional[int] = None

    @property
    def confidence(self) -> Optional[float]:
        """Alias for :attr:`composite_index` — gnomon's combined sex score.

        gnomon does not emit a 0–1 probability; the composite index is the
        closest single confidence-like statistic it computes. Returns
        ``None`` when the column was ``NA`` (insufficient informative loci).
        """
        return self.composite_index


@dataclass(frozen=True)
class TermsResult:
    """Outcome of a ``gnomon terms`` run."""

    inferred_sex: Optional[InferredSex]
    sex_output_path: Optional[Path]
    stdout: str
    stderr: str
    returncode: int
    sex_table: Optional[Tuple[Tuple[str, str], ...]] = None
    sex_metrics: Optional[Tuple[SexMetrics, ...]] = None
    """Per-sample structured metrics parsed from the ``.sex.tsv`` columns.

    ``None`` only when no TSV was produced (e.g. a single-sample dataset
    where the call was recovered from stdout). For a single-sample run with
    a TSV this is a 1-tuple; :attr:`metrics` returns that sole entry.
    """

    @property
    def metrics(self) -> Optional[SexMetrics]:
        """The sole :class:`SexMetrics` for a single-sample run, else ``None``.

        Convenience for the common one-sample case (the pipeline's per-sample
        VCFs). When the dataset has multiple samples, use
        :attr:`sex_metrics` and index by IID.
        """
        if self.sex_metrics and len(self.sex_metrics) == 1:
            return self.sex_metrics[0]
        return None


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
    capture: bool = True,
) -> subprocess.CompletedProcess:
    """Run the gnomon binary.

    When ``capture=False``, stdout/stderr inherit from the parent so the
    user sees live progress bars; the returned `CompletedProcess` then
    has empty `.stdout`/`.stderr`.
    """
    full_argv = [str(binary), *map(str, argv)]
    proc_env = dict(os.environ)
    if env:
        proc_env.update(env)
    try:
        completed = subprocess.run(
            full_argv,
            capture_output=capture,
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


def _fnv1a64_hex8(data: bytes) -> str:
    """8-hex-digit FNV-1a hash — byte-identical to score::main::fnv1a64_hex8."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"[-8:]


def _inline_pgs_output_suffix(score_arg: str) -> str:
    """Mirror score::main::inline_pgs_output_suffix."""
    ids = []
    for item in score_arg.split(","):
        item = item.strip()
        if not item:
            continue
        first = item.split("|", 1)[0].strip()
        if first.startswith("PGS"):
            ids.append(first)
    return f"pgs{len(ids)}_{_fnv1a64_hex8(score_arg.encode())}"


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


def expected_sscore_path(
    input_path: Union[str, os.PathLike],
    score_arg: Union[str, os.PathLike, Sequence[str]],
    *,
    score_exists: Optional[bool] = None,
) -> Path:
    """Predict the ``.sscore`` path the gnomon CLI will write to.

    Mirrors ``score::main::score_output_path`` byte-for-byte. Useful for
    pre-flighting where the output will land, or for clean-up scripts
    that need to enumerate produced files.

    The Rust logic for the output filename is:
      * If the SCORE argument does not exist on disk AND it contains
        ``"PGS"`` somewhere in its text, use the inline-PGS suffix
        ``pgs<count>_<fnv1a64_hex8>``.
      * Otherwise, use the file stem of the SCORE argument
        (or ``"scores"`` if the argument has no filename component).

    Pass ``score_exists`` explicitly to skip the on-disk ``.exists()``
    check (handy in dry-runs or when the score file is remote).
    """
    input_p = Path(input_path)
    if isinstance(score_arg, (list, tuple)):
        score_text = ",".join(score_arg)
    else:
        score_text = str(score_arg)
    return _expected_sscore_path_impl(input_p, score_text, score_exists=score_exists)


def _expected_sscore_path(
    input_path: Path,
    score_arg: str,
    *,
    score_exists: Optional[bool] = None,
) -> Path:
    """Internal alias — kept for backwards compatibility with tests."""
    return _expected_sscore_path_impl(input_path, score_arg, score_exists=score_exists)


def _expected_sscore_path_impl(
    input_path: Path,
    score_arg: str,
    *,
    score_exists: Optional[bool] = None,
) -> Path:
    stem = _score_output_stem(input_path)
    parent = input_path.parent if str(input_path.parent) else Path(".")

    score_path = Path(score_arg)
    if score_exists is None:
        score_exists = score_path.exists()

    if not score_exists and "PGS" in score_arg:
        suffix = _inline_pgs_output_suffix(score_arg)
        return parent / f"{stem}_{suffix}.sscore"

    score_stem = score_path.stem if score_path.name else "scores"
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
    capture: bool = True,
) -> ScoreResult:
    """Run ``gnomon score``.

    Parameters
    ----------
    score : str | path | list of str
        Either:
          * A comma-separated list of PGS Catalog IDs (``"PGS004536,PGS001320"``).
          * A list/tuple of PGS IDs (``["PGS004536", "PGS001320"]``).
          * A path to a single score file or a directory of score files.
    input_path : path-like
        Genotype source — PLINK prefix, .vcf/.vcf.gz, .bcf, or DTC text.
    keep, reference, build, panel, inferred_sex
        Pre-supply these to skip the corresponding inference / download
        steps inside the binary.
    output_path : path-like, optional
        Override where *the wrapper* looks for the output ``.sscore``.
        The Rust binary always derives its own output path from the
        SCORE + GENOTYPE positionals (see ``_expected_sscore_path``), so
        this argument is purely a search hint — useful in tests or when
        the path computation in this wrapper drifts from the CLI's. To
        actually relocate gnomon's output, pre-stage the genotype prefix
        in the destination directory instead.
    extra_args
        Forwarded verbatim to the CLI after the recognised flags.
    timeout, env, cwd
        Standard subprocess controls.
    read_output : bool, default True
        Parse the ``.sscore`` into a ``ScoreTable``. Set to False to
        skip the parse — handy when scores are huge and you just want
        the file path.

    Returns
    -------
    ScoreResult
        ``output_path``, parsed ``scores`` (or an empty ``ScoreTable``
        when ``read_output=False``), plus captured stdout/stderr/exit.
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

    completed = _run(bin_path, argv, timeout=timeout, env=env, cwd=cwd, capture=capture)

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

# Compound suffixes the gnomon CLI strips when deriving an output stem
# (mirrors ``derive_output_stem_name`` in ``map/io.rs``). Order matters:
# the compound (two-dot) suffixes are tried before the simple ones.
_TERMS_COMPOUND_SUFFIXES = (".vcf.bgz", ".vcf.gz", ".bcf.bgz", ".bcf.gz")
_TERMS_SIMPLE_SUFFIXES = (
    ".bed", ".bim", ".fam", ".pgen", ".pvar", ".psam", ".vcf", ".bcf",
)


def _terms_output_stem(path: Path) -> str:
    """Derive the gnomon output stem for ``path``.

    Byte-for-byte port of ``derive_output_stem_name`` in ``map/io.rs`` so the
    wrapper finds the exact ``<stem>.sex.tsv`` the CLI writes (e.g.
    ``sample.vcf.gz`` -> ``sample``, not ``sample.vcf``).
    """
    name = path.name
    if not name:
        return "dataset"
    lower = name.lower()
    for suffix in _TERMS_COMPOUND_SUFFIXES + _TERMS_SIMPLE_SUFFIXES:
        if lower.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    stem = Path(name).stem
    return stem if stem else "dataset"


def _parse_float_or_none(s: str) -> Optional[float]:
    s = s.strip()
    if s in ("", "NA", "nan", "NaN", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_int_or_none(s: str) -> Optional[int]:
    s = s.strip()
    if s in ("", "NA", "nan", "NaN", "."):
        return None
    try:
        return int(s)
    except ValueError:
        f = _parse_float_or_none(s)
        return int(f) if f is not None else None


# Column order written by ``terms/sex.rs::SEX_TSV_HEADER``.
_SEX_TSV_COLUMNS = (
    "IID", "Build", "Sex", "Y_Density", "X_AutoHet_Ratio", "Composite_Index",
    "Auto_Valid", "Auto_Het", "X_NonPAR_Valid", "X_NonPAR_Het",
    "Y_NonPAR_Valid", "Y_PAR_Valid",
)


def _row_to_sex_metrics(row: Sequence[str]) -> SexMetrics:
    """Build a :class:`SexMetrics` from one ``.sex.tsv`` data row.

    Tolerates short rows (older CLI builds with fewer metric columns) by
    defaulting any missing trailing column to ``None``.
    """
    def cell(i: int) -> str:
        return row[i] if i < len(row) else ""

    return SexMetrics(
        iid=cell(0),
        sex=_coerce_sex(cell(2)),
        build=cell(1).strip() or None,
        y_density=_parse_float_or_none(cell(3)),
        x_autosome_het_ratio=_parse_float_or_none(cell(4)),
        composite_index=_parse_float_or_none(cell(5)),
        auto_valid=_parse_int_or_none(cell(6)),
        auto_het=_parse_int_or_none(cell(7)),
        x_non_par_valid=_parse_int_or_none(cell(8)),
        x_non_par_het=_parse_int_or_none(cell(9)),
        y_non_par_valid=_parse_int_or_none(cell(10)),
        y_par_valid=_parse_int_or_none(cell(11)),
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
    # `<stem>.sex.tsv` (see `terms/sex.rs::infer_sex_to_tsv` ->
    # `dataset.output_path("sex.tsv")` and `map/io.rs::append_output_filename`,
    # which joins `<base>.<filename>`). The historical `<stem>_sex.tsv`
    # candidate is kept last as a defensive fallback only — the CLI never
    # writes it. Resolve against the correct stem first.
    stem = _terms_output_stem(gp)
    candidates = [
        gp.parent / f"{stem}.sex.tsv",
        # Defensive fallbacks (never produced by the current CLI):
        gp.parent / f"{gp.name}.sex.tsv",
        gp.parent / f"{stem}_sex.tsv",
    ]
    out_path: Optional[Path] = next((c for c in candidates if c.exists()), None)

    sex_table: Optional[Tuple[Tuple[str, str], ...]] = None
    sex_metrics: Optional[Tuple[SexMetrics, ...]] = None
    inferred: Optional[InferredSex] = None
    if out_path is not None:
        with open(out_path) as f:
            reader = csv.reader(f, delimiter="\t")
            rows: List[Tuple[str, str]] = []
            metrics: List[SexMetrics] = []
            for row in reader:
                if not row or row[0].lstrip("#").strip().upper() == "IID":
                    # Header line. The first column header is "IID" (the CLI
                    # writes it un-commented; tolerate a leading '#' too).
                    continue
                if not row or row[0].startswith("#"):
                    continue
                if len(row) < 2:
                    continue
                m = _row_to_sex_metrics(row)
                metrics.append(m)
                # The Sex call is column index 2 in the real CLI output
                # (`IID  Build  Sex  ...`); fall back to column 1 only for a
                # hypothetical 2-column file.
                sex_str = m.sex.value if len(row) > 2 else row[1]
                rows.append((row[0], sex_str))
            sex_table = tuple(rows)
            sex_metrics = tuple(metrics)
            if len(metrics) == 1:
                inferred = metrics[0].sex
    else:
        # Single-sample dataset with no TSV: try parsing stdout.
        for line in (completed.stdout or "").splitlines():
            m = _SEX_LINE_RE.match(line)
            if m:
                inferred = _coerce_sex(m.group("sex"))
                break

    return TermsResult(
        inferred_sex=inferred,
        sex_output_path=out_path,
        sex_table=sex_table,
        sex_metrics=sex_metrics,
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
