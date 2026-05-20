# gnomon (Python)

Python wrapper for the [`SauersML/gnomon`](https://github.com/SauersML/gnomon)
high-performance polygenic score engine. Each gnomon subcommand is a
typed Python function with kwargs that mirror the CLI flags one-to-one,
plus a parsed-result dataclass.

```python
import gnomon

result = gnomon.score(
    "PGS004536,PGS001320,PGS005331",
    "/data/aou_array_plink/arrays",
)

result.output_path             # PosixPath('.../arrays_pgs3_<hash>.sscore')
result.n_samples               # 245678
result.score_names             # ('PGS004536', 'PGS001320', 'PGS005331')

# Indexable / queryable:
result.scores["NWD_001"]                          # {score: {avg, sum, denom}}
result.scores.score_for("NWD_001", "PGS004536")   # float
result.scores.to_pandas()                         # DataFrame
```

## Install

```bash
pip install gnomon
cargo install gnomon
```

Binary located via `binary=`, `$GNOMON_BIN`, or `PATH`.

## Subcommands

```python
gnomon.score(score, input_path, *, keep=None, reference=None, build=None,
             panel=None, inferred_sex=None, ...)  -> ScoreResult
gnomon.terms(genotype_path, *, sex=True)          -> TermsResult
gnomon.run_all(score, input_path, model)          -> AllResult

import gnomon.map as gmap
gmap.fit(genotype_path, components=20, ld=True, bp_window=500_000)
gmap.project(genotype_path, model="hwe_1kg_hgdp_gsa_v3")

import gnomon.calibrate as gcal
gcal.train("train.tsv", num_pcs=10, model_family="gam")
gcal.infer("test.tsv", model="model.toml")
```

## Shortcuts: avoid downloads and re-inference

Every kwarg the CLI exposes is on the Python API — none of these
overrides require touching the binary directly.

```python
gnomon.score(
    "PGS004536,PGS001320",
    "/data/cohort/arrays",
    reference="/cache/hg38.fa",     # skip reference auto-download
    build="38",                      # skip build auto-detection
    panel="/cache/1kg_panel.vcf",    # supply harmonisation panel
    inferred_sex="male",             # skip the in-pipeline sex scan
    keep="/data/keep.iids.txt",      # restrict to a sample subset
)
```

`inferred_sex` accepts `"male"`, `"female"`, `"unknown"`, the matching
`InferredSex` enum members, or any string returned by `infer_sex`.

## ScoreTable

`gnomon.read_sscore(path)` returns a `ScoreTable`. The result of
`gnomon.score(...)` carries one on `.scores`.

```python
table.iids                            # tuple of sample IDs
table.score_names                     # tuple of PGS names (suffix-stripped)
table.fids                            # tuple of family IDs
table.avg / .sum / .denom             # tuple-of-tuples or None per column

# Membership / lookup:
"NWD_001" in table
table.index_of("NWD_001")              # int row index
table["NWD_001"]                       # {score: {avg, sum, denom}}
table.score_for("NWD_001", "PGS004536", kind="avg")  # float

# Pandas adapter (optional, requires `pip install gnomon[pandas]`):
table.to_pandas()
```

Missing columns (e.g. older builds without `_SUM` / `_DENOM`) return
`None` — no fake zeros.

## Path inference

Sscore output filenames follow `score::main::score_output_path` exactly.
For inline PGS arguments the wrapper computes the same
`pgs<count>_<fnv1a64_hex8>` suffix the Rust binary writes — you don't
have to guess where the file lands.

```python
from gnomon import expected_sscore_path

expected_sscore_path("/data/arrays.vcf.gz", "PGS001,PGS002")
# PosixPath('/data/arrays_pgs2_<hash>.sscore')
```

## Errors

* `GnomonBinaryNotFound` — CLI not installed / not on PATH.
* `InvalidConfig` — argument combination rejected before launching.
* `GnomonFailed` — CLI exited non-zero. The exception preserves
  `stdout`, `stderr`, `returncode`. Includes the last non-empty stderr
  line in the message so the failure mode is obvious without spelunking.
* `SscoreParseError` — corrupt or unexpectedly-shaped `.sscore` output.

All subclass `GnomonError`.
