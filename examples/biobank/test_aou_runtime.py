"""MSI acceptance: frozen public-reference CTN and synthetic survival outcomes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from aou_survival import bounded_fit, cif_from_hazards
from reference_ctn import load_reference
from aou_score_transform import transformed_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-ctn", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--projection-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(915)
    n = 240
    reference = pd.read_parquet(args.reference_table)
    score_columns = [c for c in reference if c.endswith("_AVG")]
    if len(score_columns) != 1:
        raise ValueError("reference table must identify one score")
    pgs_id = score_columns[0].removesuffix("_AVG")
    pcs = [c for c in reference if c.startswith("PC")]
    transform, transform_path, _ = load_reference(
        [args.reference_ctn], args.output / "reference", pgs_id, len(pcs), args.projection_sha256)
    frame = reference.sample(n=n, random_state=915).rename(columns={score_columns[0]: "PGS"})
    frame = frame[["PGS", *pcs]].reset_index(drop=True)
    frame["baseline"] = pd.Timestamp("2020-01-01")
    frame["death_date"] = pd.NaT
    frame["Z_ctn"] = transformed_score(transform, "ctn", frame, len(pcs))
    event_time = rng.exponential(np.exp(-.25 * frame.Z_ctn.to_numpy()) * 2)
    censor_time = rng.uniform(1, 5, n)
    frame = frame.assign(**{"age0": rng.uniform(40, 70, n),
                          "sex": rng.integers(0, 2, n), "entry": np.zeros(n),
                          "followup": np.minimum(event_time, censor_time),
                          "event_code": (event_time <= censor_time).astype(int),
                          "is_train": np.arange(n) % 5 != 0,
                          "split_group": np.arange(n)})
    frame_path = args.output / "synthetic.parquet"
    frame.to_parquet(frame_path, index=False)
    config = {"num_pcs": len(pcs), "baseline_centers": len(pcs) + 2, "slope_centers": len(pcs) + 2,
              "time_num_internal_knots": 2,
              "horizons_years": [1., 2.], "grid_intervals": 20}
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(config))
    bounded_fit([sys.executable, str(Path(__file__).with_name("aou_survival.py")), "fit",
                 "--frame", str(frame_path), "--config", str(config_path),
                 "--kind", "pc_varying", "--normalizer", "ctn",
                 "--transform-model", str(transform_path),
                 "--cause", "1", "--output", str(args.output)],
                90, args.output / "fit.log")
    with np.load(args.output / "hazards.npz") as artifact:
        cif = cif_from_hazards(artifact["hazards"][None, :, :])
    assert np.isfinite(cif).all() and (np.diff(cif, axis=2) >= -1e-10).all()
    print("Frozen external CTN, synthetic survival, save/load and batch invariance passed.")


if __name__ == "__main__":
    main()
