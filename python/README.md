# gnomon (Python)

Native, in-process Python bindings for the
[`SauersML/gnomon`](https://github.com/SauersML/gnomon) polygenic-score engine.
The wheel contains the Rust core through PyO3; it does not invoke a command-line
binary.

## Install

```bash
pip install gnomon-pgs
```

## API

```python
import gnomon

# Compute raw polygenic scores. The result is the input prefix beside which
# the engine writes its score artifacts.
gnomon.score(
    "PGS004536,PGS001320",
    "/data/cohort/arrays",
    reference="/cache/hg38.fa",
    build="38",
    panel="/cache/1kg_panel.vcf",
    inferred_sex="male",
    emit_components=False,
)

# Project genotypes onto a built-in HWE-PCA model.
gnomon.project(
    "/data/cohort/arrays",
    build="38",
    model="hwe_1kg_hgdp_gsa_v3",
    output_manifest="/data/cohort/projection.json",
)

# Write inferred-sex terms or return the first sample's call directly.
gnomon.terms("/data/cohort/arrays", sex=True)
gnomon.infer_sex("/data/cohort/arrays")

# Return a built-in model's variant-key document as JSON.
gnomon.model("hwe_1kg_hgdp_gsa_v3")
```

Rust failures are raised as Python exceptions by the native extension. Import
failure means the wheel or extension build is incomplete; no alternate runtime
is selected.
