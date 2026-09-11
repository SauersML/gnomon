"""Fit one PC-conditional CTN on an external genetic reference panel.

Run on MSI, never on AoU participant rows. The score and PC projection must
use the same frozen definitions as the target cohort. This estimates the
reference-panel score distribution, not outcome risk or target calibration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import pandas as pd


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_manifest(manifest, pgs, num_pcs, projection_sha256):
    if manifest.get("schema") != "external-reference-ctn-v1":
        raise ValueError("unknown external CTN manifest schema")
    if manifest.get("reference_population") not in {"1000G", "HGDP+1000G"}:
        raise ValueError("CTN must identify its external reference population")
    if manifest.get("pgs_id") != pgs or not re.fullmatch(r"PGS\d{6}", pgs):
        raise ValueError("reference CTN belongs to a different PGS")
    if manifest.get("pc_columns") != [f"PC{i + 1}" for i in range(num_pcs)]:
        raise ValueError("reference CTN PC columns do not match the outcome model")
    if manifest.get("projection_model_sha256") != projection_sha256:
        raise ValueError("reference and target must use the same PC projection")
    for key in ("projection_model_sha256", "score_file_sha256", "training_table_sha256", "model_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))):
            raise ValueError(f"reference CTN lacks a valid {key}")
    if manifest.get("score_column") != pgs + "_AVG":
        raise ValueError("reference CTN must use the matching Gnomon average-score column")


def load_reference(archives, output, pgs, num_pcs, projection_sha256):
    """Load one immutable external transform; there is no fit or batch adaptation."""
    import gamfit
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    matches = []
    for archive in archives:
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            if sorted(m.name for m in members) != ["manifest.json", "transform.gamfit"]:
                raise ValueError("reference CTN archive must contain exactly its manifest and model")
            if any(not m.isfile() for m in members) or sum(m.size for m in members) > 128 * 1024**2:
                raise ValueError("invalid or oversized reference CTN archive")
            metadata = json.load(source.extractfile("manifest.json"))
            if metadata.get("pgs_id") == pgs:
                matches.append(archive)
    if len(matches) != 1:
        raise ValueError("exactly one frozen reference CTN is required for each requested PGS")
    with tarfile.open(matches[0], "r:gz") as source:
        members = source.getmembers()
        names = [member.name for member in members]
        if sorted(names) != ["manifest.json", "transform.gamfit"]:
            raise ValueError("reference CTN archive must contain exactly its manifest and model")
        if any(not member.isfile() for member in members) or sum(m.size for m in members) > 128 * 1024**2:
            raise ValueError("invalid or oversized reference CTN archive")
        manifest = json.load(source.extractfile("manifest.json"))
        validate_manifest(manifest, pgs, num_pcs, projection_sha256)
        model_bytes = source.extractfile("transform.gamfit").read()
    if hashlib.sha256(model_bytes).hexdigest() != manifest["model_sha256"]:
        raise ValueError("reference CTN checksum mismatch")
    model_path = output / "transform.gamfit"
    model_path.write_bytes(model_bytes)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return gamfit.load(model_path), model_path, manifest


def train(args):
    import gamfit
    metadata = json.loads(args.metadata.read_text())
    if metadata["reference_population"] not in {"1000G", "HGDP+1000G"}:
        raise ValueError("training data must be an external reference panel")
    pgs = metadata["pgs_id"]
    if not re.fullmatch(r"PGS\d{6}", pgs):
        raise ValueError("invalid PGS ID")
    for key in ("projection_model_sha256", "score_file_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get(key, ""))):
            raise ValueError(f"reference metadata lacks a valid {key}")
    if args.num_pcs < 1 or args.centers <= args.num_pcs + 1:
        raise ValueError("reference Duchon basis size must exceed the PC count plus one")
    pcs = [f"PC{i + 1}" for i in range(args.num_pcs)]
    frame = pd.read_parquet(args.table)
    required = ["sample_id", pgs + "_AVG", *pcs]
    if set(frame.columns) != set(required):
        raise ValueError("reference table must contain only sample IDs, the PGS and projected PCs")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any() or len(frame) < 200:
        raise ValueError("reference panel requires at least 200 unique identified samples")
    data = frame[[pgs + "_AVG", *pcs]].rename(columns={pgs + "_AVG": "PGS"})
    if not np.isfinite(data.to_numpy(float)).all() or data.PGS.std() <= 0:
        raise ValueError("reference data must be finite with a variable score")
    rhs = f"duchon({', '.join(pcs)}, centers={args.centers}, scale_dims=true)"
    args.output.mkdir(parents=True, exist_ok=True)
    model = gamfit.fit(data, f"PGS ~ {rhs}", transformation_normal=True,
                      config={"transformation_normal_config": {"response_num_internal_knots": 2}},
                      persistent_warm_start_root=args.output / "warm")
    scores = np.asarray(model.transformation_score(data))
    if not np.isfinite(scores).all():
        raise ValueError("reference CTN produced nonfinite scores")
    path = args.output / "transform.gamfit"
    model.save(path)
    restored = gamfit.load(path)
    np.testing.assert_allclose(restored.transformation_score(data.iloc[:3]), scores[:3], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(restored.transformation_score(data.iloc[[1]]), scores[[1]], rtol=1e-8, atol=1e-10)
    for index in np.linspace(0, len(data) - 1, 20, dtype=int):
        grid = data.iloc[[index] * 25].copy()
        grid["PGS"] = np.linspace(data.PGS.min(), data.PGS.max(), 25)
        if not (np.diff(restored.transformation_score(grid)) > 0).all():
            raise ValueError("reference CTN is not monotone on the observed score range")
    manifest = {**metadata, "schema": "external-reference-ctn-v1", "pc_columns": pcs,
                "score_column": pgs + "_AVG", "formula": f"PGS ~ {rhs}",
                "training_table_sha256": digest(args.table), "model_sha256": digest(path),
                "training_rows": len(frame), "gamfit_build": gamfit.build_info(),
                "target_population_normality_claim": False,
                "pc_min": data[pcs].min().tolist(), "pc_max": data[pcs].max().tolist()}
    validate_manifest(manifest, pgs, args.num_pcs, metadata["projection_model_sha256"])
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(args.output / "reference_ctn.tar.gz", "w:gz") as archive:
        for item in (manifest_path, path):
            archive.add(item, arcname=item.name, recursive=False)
    print("External reference CTN fitted, saved and replay-checked.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("table", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--num-pcs", type=int, default=6)
    parser.add_argument("--centers", type=int, default=8)
    train(parser.parse_args())
