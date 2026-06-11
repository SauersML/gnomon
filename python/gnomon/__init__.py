"""gnomon — Python bindings for the SauersML/gnomon engine.

gnomon is a high-performance polygenic score engine. This package ships a
**native, in-process** binding to the Rust core (built with maturin/PyO3): the
compiled extension ``gnomon._gnomon`` is loaded directly into the Python
process, so the core subcommands run with no subprocess to a ``gnomon`` binary
and no ``install.sh`` step.

In-process core (native, the default)
-------------------------------------

These call the Rust engine directly, in-process:

  * ``gnomon.score(score, input_path, ...)``  – compute raw PGS
  * ``gnomon.project(genotype_path, ...)``    – project onto an HWE-PCA model
  * ``gnomon.terms(genotype_path, sex=True)`` – write a sex-inference TSV
  * ``gnomon.infer_sex(genotype_path)``       – first sample's sex call, returned
                                                directly (no TSV, no CLI run)
  * ``gnomon.model(name)``                    – a model's variant keys as JSON

When the native extension is present (a normal wheel install or ``maturin
develop``), the top-level ``score``/``terms`` names bind to the in-process
implementations. The legacy typed subprocess wrappers remain importable from
``gnomon._api`` (and the rich result dataclasses are re-exported below) as a
fallback for environments without the compiled extension.

Quick start
-----------

>>> import gnomon
>>> gnomon.infer_sex("/data/sample.vcf.gz")
'male'
>>> gnomon.score("PGS004536,PGS001320", "/data/arrays")
'/data/arrays'
"""

# --- legacy typed subprocess API + shared result/exception types -------------
# Imported unconditionally: these define the public dataclasses/exceptions and
# provide a fallback path when the native extension is unavailable.
from ._api import (
    score as _subprocess_score,
    terms as _subprocess_terms,
    run_all,
    read_sscore,
    expected_sscore_path,
    ScoreResult,
    ScoreTable,
    TermsResult,
    SexMetrics,
    MapResult,
    CalibrateResult,
    AllResult,
    InferredSex,
    GnomonError,
    GnomonBinaryNotFound,
    GnomonFailed,
    InvalidConfig,
    SscoreParseError,
    locate_binary,
)
from . import map, calibrate

# --- native, in-process core (preferred) -------------------------------------
try:
    from . import _gnomon as _native
except ImportError:  # pragma: no cover - only when the extension isn't built
    _native = None

if _native is not None:
    # The native extension is the default in-process implementation.
    score = _native.score
    project = _native.project
    terms = _native.terms
    infer_sex = _native.infer_sex
    model = _native.model
    HAVE_NATIVE = True
else:
    # Fall back to the typed subprocess wrappers. ``project``/``infer_sex``/
    # ``model`` are native-only conveniences and are not provided here.
    score = _subprocess_score
    terms = _subprocess_terms
    project = None
    infer_sex = None
    model = None
    HAVE_NATIVE = False

# Always expose the subprocess wrappers explicitly for callers that want the
# typed dataclass results regardless of which path ``score``/``terms`` take.
subprocess_score = _subprocess_score
subprocess_terms = _subprocess_terms

__all__ = [
    # in-process core (native when available)
    "score",
    "project",
    "terms",
    "infer_sex",
    "model",
    "HAVE_NATIVE",
    # typed subprocess wrappers + helpers
    "subprocess_score",
    "subprocess_terms",
    "run_all",
    "read_sscore",
    "expected_sscore_path",
    "locate_binary",
    # result types
    "ScoreResult",
    "ScoreTable",
    "TermsResult",
    "SexMetrics",
    "MapResult",
    "CalibrateResult",
    "AllResult",
    "InferredSex",
    # exceptions
    "GnomonError",
    "GnomonBinaryNotFound",
    "GnomonFailed",
    "InvalidConfig",
    "SscoreParseError",
    # submodules
    "map",
    "calibrate",
]

__version__ = "0.1.0"
