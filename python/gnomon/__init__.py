"""gnomon — Python bindings for the SauersML/gnomon CLI.

gnomon is a high-performance polygenic score engine. This Python package
wraps its CLI surface as a small set of typed functions:

  * ``gnomon.score(score, input_path, ...)``    – compute raw PGS
  * ``gnomon.read_sscore(path)``                – parse a .sscore output
  * ``gnomon.terms(genotype_path, sex=True)``   – terms inference (sex etc.)
  * ``gnomon.map.fit(...)`` / ``gnomon.map.project(...)``  – HWE PCA
  * ``gnomon.calibrate.train(...)`` / ``gnomon.calibrate.infer(...)``
  * ``gnomon.run_all(...)``                     – score + project + terms

Each subcommand has its own typed kwargs and returns a frozen dataclass
result (e.g. ``ScoreResult`` carries the output path, the parsed
``ScoreTable``, plus captured stdout/stderr for inspection).

Quick start
-----------

>>> from gnomon import score
>>> result = score(
...     score="PGS004536,PGS001320,PGS005331",
...     input_path="/data/aou_array_plink/arrays",
... )
>>> result.output_path
PosixPath('/data/aou_array_plink/arrays_PGS004536-PGS001320-PGS005331.sscore')
>>> result.scores.shape
(245678, 4)            # IID + 3 scores
"""

from ._api import (
    score,
    terms,
    run_all,
    read_sscore,
    expected_sscore_path,
    ScoreResult,
    ScoreTable,
    TermsResult,
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

__all__ = [
    "score",
    "terms",
    "run_all",
    "read_sscore",
    "expected_sscore_path",
    "ScoreResult",
    "ScoreTable",
    "TermsResult",
    "MapResult",
    "CalibrateResult",
    "AllResult",
    "InferredSex",
    "GnomonError",
    "GnomonBinaryNotFound",
    "GnomonFailed",
    "InvalidConfig",
    "SscoreParseError",
    "locate_binary",
    "map",
    "calibrate",
]

__version__ = "0.1.0"
