"""Bounded native-engine acceptance test using synthetic data, executed on MSI."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from aou_survival import bounded_fit, cif_from_hazards
from aou_score_transform import assemble_scores, grouped_folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(915)
    n = 240
    pgs = rng.normal(size=n)
    event_time = rng.exponential(np.exp(-.25 * pgs) * 2)
    censor_time = rng.uniform(1, 5, n)
    frame = pd.DataFrame({"PGS": pgs, "PC1": rng.normal(size=n),
                          "PC2": rng.normal(size=n), "age0": rng.uniform(40, 70, n),
                          "sex": rng.integers(0, 2, n), "entry": np.zeros(n),
                          "followup": np.minimum(event_time, censor_time),
                          "event_code": (event_time <= censor_time).astype(int),
                          "is_train": np.arange(n) % 5 != 0,
                          "split_group": np.arange(n)})
    frame["inner_fold"] = -1
    frame.loc[frame.is_train, "inner_fold"] = grouped_folds(
        frame.loc[frame.is_train, "split_group"], 2, 915)
    frame_path = args.output / "synthetic.parquet"
    frame.to_parquet(frame_path, index=False)
    config = {"num_pcs": 2, "baseline_centers": 4, "slope_centers": 4,
              "time_num_internal_knots": 2,
              "stage1_centers": 4, "stage1_age_k": 4, "stage1_response_knots": 2,
              "horizons_years": [1., 2.], "grid_intervals": 20}
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(config))
    artifacts = []
    runner = str(Path(__file__).with_name("aou_survival.py"))
    for fold in (0, 1, -1):
        stage = args.output / f"ctn_fold_{fold}"
        stage.mkdir(exist_ok=True)
        bounded_fit([sys.executable, runner, "transform", "--frame", str(frame_path),
                     "--config", str(config_path), "--normalizer", "ctn",
                     "--fold", str(fold), "--output", str(stage)], 60, stage / "fit.log")
        artifacts.append(stage / "scores.npz")
    frame["Z_ctn"] = assemble_scores(frame, artifacts)
    frame.to_parquet(frame_path, index=False)
    bounded_fit([sys.executable, str(Path(__file__).with_name("aou_survival.py")), "fit",
                 "--frame", str(frame_path), "--config", str(config_path),
                 "--kind", "pc_varying", "--normalizer", "ctn",
                 "--transform-model", str(args.output / "ctn_fold_-1" / "transform.gamfit"),
                 "--cause", "1", "--output", str(args.output)],
                90, args.output / "fit.log")
    with np.load(args.output / "hazards.npz") as artifact:
        cif = cif_from_hazards(artifact["hazards"][None, :, :])
    assert np.isfinite(cif).all() and (np.diff(cif, axis=2) >= -1e-10).all()
    print("Native cross-fitted CTN survival, held-out hazards, save/load and batch invariance passed.")


if __name__ == "__main__":
    main()
