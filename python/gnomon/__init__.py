"""Native, in-process Python bindings for the gnomon polygenic-score engine.

The package is the compiled PyO3 extension. Importing :mod:`gnomon` therefore
either loads the Rust core or fails immediately; there is no subprocess API and
no second execution path.

>>> import gnomon
>>> gnomon.infer_sex("/data/sample.vcf.gz")
'male'
>>> gnomon.score("PGS004536,PGS001320", "/data/arrays")
'/data/arrays'
"""

from ._gnomon import infer_sex, model, project, score, terms

__all__ = ["score", "project", "terms", "infer_sex", "model"]

__version__ = "0.1.3"
