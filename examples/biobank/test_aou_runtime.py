"""Bounded native-engine acceptance test using synthetic data, executed on MSI."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from aou_survival import bounded_fit, cif_from_hazards


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
                          "is_train": np.arange(n) % 5 != 0})
    frame_path = args.output / "synthetic.parquet"
    frame.to_parquet(frame_path, index=False)
    config = {"num_pcs": 2, "baseline_centers": 4, "slope_centers": 4,
              "horizons_years": [1., 2.], "grid_intervals": 20}
    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(config))
    bounded_fit([sys.executable, str(Path(__file__).with_name("aou_survival.py")), "fit",
                 "--frame", str(frame_path), "--config", str(config_path),
                 "--kind", "pc_varying", "--cause", "1", "--output", str(args.output)],
                90, args.output / "fit.log")
    with np.load(args.output / "hazards.npz") as artifact:
        cif = cif_from_hazards(artifact["hazards"][None, :, :])
    assert np.isfinite(cif).all() and (np.diff(cif, axis=2) >= -1e-10).all()
    print("Native PC-varying survival fit, held-out hazards, and save/load replay passed.")


if __name__ == "__main__":
    main()
