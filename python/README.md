# gnomon (Python)

Python wrapper for the [`SauersML/gnomon`](https://github.com/SauersML/gnomon)
high-performance polygenic score engine.

The wrapper drives the upstream Rust CLI via subprocess, parses
`.sscore` outputs into typed dataclasses, and exposes each subcommand
as a Pythonic function with validated kwargs.

## Install

```bash
pip install gnomon
# also install the Rust binary:
cargo install gnomon
```

The wrapper finds the binary on `PATH`, via `$GNOMON_BIN`, or through
the `binary=` keyword argument.

## Quick start

```python
import gnomon

result = gnomon.score(
    "PGS004536,PGS001320,PGS005331",
    "/data/aou_array_plink/arrays",
)

print(result.output_path)          # PosixPath('.../arrays_PGS004536-...-PGS005331.sscore')
print(result.n_samples)            # 245678
print(result.score_names)          # ('PGS004536', 'PGS001320', 'PGS005331')

# Pandas, if you want it:
df = result.scores.to_pandas()
```

## Shortcuts: avoid downloads and inference

Every kwarg the upstream CLI exposes is on the Python API. You don't have
to let gnomon download references or re-infer state you already know.

```python
gnomon.score(
    "PGS004536,PGS001320",
    "/data/cohort/arrays",
    reference="/cache/hg38.fa",   # skip reference auto-download
    build="38",                    # skip build auto-detection
    panel="/cache/1kg_panel.vcf",  # supply your harmonisation panel
    inferred_sex="male",           # skip the in-pipeline sex scan
    keep="/data/keep.iids.txt",    # restrict to a sample subset
)
```

`gnomon.terms`, `gnomon.run_all`, `gnomon.map.fit/project`,
`gnomon.calibrate.train/infer` all expose the same precomputed-input
kwargs.

## Subcommands

```python
gnomon.score(score, input_path, *, keep=None, build=None, panel=None, ...)
gnomon.terms(genotype_path, *, sex=True)
gnomon.run_all(score, input_path, model)

import gnomon.map as gmap
gmap.fit(genotype_path, components=20, ld=True, bp_window=500_000)
gmap.project(genotype_path, model="hwe_1kg_hgdp_gsa_v3")

import gnomon.calibrate as gcal
gcal.train("train.tsv", num_pcs=10, model_family="gam")
gcal.infer("test.tsv", model="model.toml")
```

## Sscore parsing

```python
from gnomon import read_sscore

table = read_sscore("arrays_PGS001.sscore")
table.iids            # ('S1', 'S2', ...)
table.score_names     # ('PGS001',)
table.avg             # ((0.10,), (0.20,), ...)
table.sum             # ((5.0,),  (10.0,), ...)
table.to_pandas()
```

## Errors

* `GnomonBinaryNotFound` — CLI not installed / not on PATH.
* `InvalidConfig` — argument combination rejected before launching.
* `GnomonFailed` — CLI exited non-zero (stdout/stderr/returncode preserved).
* `SscoreParseError` — corrupt or unexpectedly-shaped `.sscore` output.

All subclass `GnomonError`.
