"""Add truth_short.parquet (cif and death at 0.25 and 0.5 y) to simulator sets, from the set's own generator source.

    python examples/biobank/study/truth_short.py <generator src> <root> <scenario> [small,medium,large]

- <generator src> is a checkout or `git archive` of the commit that wrote the set; its examples/biobank/study/ holds
  simulate.py and diseases.json. It must be byte-identical to the set's recorded simulate.py
  (simulator.script_sha256) and diseases file, and the set must have used the reference PCs in STUDY_SIM_REFERENCE.
  Otherwise the set is refused.
- <root> is a published version (with MANIFEST.json; the sizes argument is then required) or a
  `simulate.py generate --out` directory.

The file has truth.parquet's rows, keys and order, with the cif and death columns null outside each rule's survival
frame. A 1-y horizon from the same run must match truth.parquet's cif_1y to 1e-8 (the grids differ) before anything
is written. Prints one JSON line per file, with its sha256, and exits non-zero if any set was refused. MSI only; one
core.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HORIZONS = (0.25, 0.5, 1.0)     # the 1-y value is only the consistency check against truth.parquet
DEFAULT_REFERENCE = "/scratch.global/sauer354/aou-study/study-sim/reference_pcs.parquet"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _groups(sim, root: Path, scenario: str, sizes):
    """{censoring: {dir, n, seed}} per size, and the world seed."""
    if (root / "MANIFEST.json").exists():
        manifest = json.loads((root / "MANIFEST.json").read_text())
        groups = [{x["censoring"]: x for x in manifest["sets"] if x["dir"].startswith(f"{size}/{scenario}_")}
                  for size in sizes]
        return groups, manifest["world_seed"]
    group, world_seed = {}, None
    for rule in sim.CENSORING:
        mf = json.loads((root / f"{scenario}_{rule}" / "manifest.json").read_text())
        group[rule] = {"dir": f"{scenario}_{rule}", "n": mf["simulator"]["n"], "seed": mf["seed"], "censoring": rule}
        world_seed = mf["simulator"]["world_seed"]
    return [group], world_seed


def _refuse_foreign(sim, root: Path, groups, reference: str):
    script, diseases = _sha(sim.__file__), _sha(sim.DEFAULT_DISEASES)
    # No reference path: the set must have used the synthetic mixture (sim.load_reference).
    ref_sha = _sha(reference) if reference else "synthetic mixture"
    for group in groups:
        for entry in group.values():
            meta = json.loads((root / entry["dir"] / "manifest.json").read_text())["simulator"]
            if meta["script_sha256"] != script:
                sys.exit(f"REFUSED: {entry['dir']} was written by simulate.py {meta['script_sha256'][:12]}, "
                         f"the source here is {script[:12]}")
            if meta["diseases_sha256"] != diseases:
                sys.exit(f"REFUSED: {entry['dir']} used diseases {meta['diseases_sha256'][:12]}, "
                         f"the source here has {diseases[:12]}")
            if ref_sha not in meta["reference"]:
                sys.exit(f"REFUSED: {entry['dir']} used another reference: {meta['reference']}")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src, root, scenario = argv[0], Path(argv[1]), argv[2]
    sizes = argv[3].split(",") if len(argv) > 3 else []
    sys.path.insert(0, str(Path(src) / "examples" / "biobank"))
    from study import simulate as sim
    sim.parallel_truth = lambda *a, **k: None       # the draws only; the short-horizon truth is computed below
    reference = os.environ.get("STUDY_SIM_REFERENCE", DEFAULT_REFERENCE)
    groups, world_seed = _groups(sim, root, scenario, sizes)
    _refuse_foreign(sim, root, groups, reference)
    ref, ref_src = sim.load_reference(reference, 20260918)
    world = sim.make_world(world_seed, scenario, sim.load_diseases(sim.DEFAULT_DISEASES), ref, ref_src)
    refused = []
    for sets in groups:
        t0 = time.time()
        s = sim.simulate(world, sets["independent"]["n"], sets["independent"]["seed"], workers=1, log=lambda *a: None)
        per_rule = {rule: sim.build_frames(s, rule)[1] for rule in sim.CENSORING}
        p = s["people"]
        truths = {}
        for rec in s["diseases"]:
            slug = rec["dp"].spec.slug
            if rec["dp"].spec.pgs is None:
                continue
            pop = per_rule["independent"][slug][0]
            union = per_rule["independent"][slug][1] | per_rule["lastcontact"][slug][1]
            truths[slug] = (pop, sim.compute_truth(world, rec["dp"], rec["terms"], rec["s"], p, HORIZONS,
                                                   rows=(np.zeros_like(pop), union)))
        for rule, entry in sets.items():
            d = root / entry["dir"]
            published = pq.read_table(d / "truth.parquet",
                                      columns=["disease", "person_id", "in_survival", "cif_1y"]).to_pandas()
            parts = []
            for slug, (pop, t) in truths.items():
                j = np.flatnonzero(pop)
                sj = per_rule[rule][slug][1][j]
                parts.append(pd.DataFrame({
                    "disease": slug, "person_id": p["person_id"][j], "in_survival": sj,
                    "cif_0.25y": np.where(sj, t["cif"][j, 0], np.nan), "cif_0.5y": np.where(sj, t["cif"][j, 1], np.nan),
                    "death_0.25y": np.where(sj, t["death"][j, 0], np.nan),
                    "death_0.5y": np.where(sj, t["death"][j, 1], np.nan),
                    "_check_1y": np.where(sj, t["cif"][j, 2], np.nan)}))
            new = pd.concat(parts, ignore_index=True)
            keys_equal = bool(len(new) == len(published)
                              and (new.disease.to_numpy() == published.disease.to_numpy()).all()
                              and (new.person_id.to_numpy() == published.person_id.to_numpy()).all()
                              and (new.in_survival.to_numpy() == published.in_survival.to_numpy()).all())
            a, b = new._check_1y.to_numpy(), published.cif_1y.to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else float("nan")
            nulls_equal = bool((np.isfinite(a) == np.isfinite(b)).all())
            if not (keys_equal and nulls_equal and diff < 1e-8):
                print(json.dumps({"dir": entry["dir"], "REFUSED": True, "keys_equal": keys_equal,
                                  "nulls_equal": nulls_equal, "cif_1y_max_abs_diff": diff}), flush=True)
                refused.append(entry["dir"])
                continue
            path = d / "truth_short.parquet"
            pq.write_table(sim._arrow_frame(new.drop(columns="_check_1y")), path, compression="zstd")
            print(json.dumps({"dir": entry["dir"], "file": "truth_short.parquet", "rows": len(new),
                              "sha256": _sha(path), "cif_1y_max_abs_diff": diff, "keys_equal": keys_equal,
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    if refused:
        sys.exit(f"REFUSED: {', '.join(refused)}")


if __name__ == "__main__":
    main()
