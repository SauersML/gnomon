"""Aggregate the per-dataset eval CSVs into study-level tables.

Reads results/{binary,survival}/*_{acc,cal}.csv (one pair per dataset) and writes,
under results/:
  accuracy_binary.csv, calibration_binary.csv,
  accuracy_survival.csv, calibration_survival.csv   -- all seeds concatenated
  summary_table.csv  -- per (dem, pheno, method) mean+/-SD over seeds of the
                        headline metrics: discrimination, and calibration BSS in
                        the held-out training-ancestry vs the other ancestries.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("sims/results_hpc/ancestry_calibration")
RES = OUT / "results"


def _seed_from(name: str) -> int:
    for tok in name.replace(".csv", "").split("_"):
        if tok.startswith("s") and tok[1:].isdigit():
            return int(tok[1:])
    return -1


def _concat(subdir: str, which: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RES / subdir).glob(f"*_{which}.csv")):
        df = pd.read_csv(f)
        if "seed" not in df.columns:
            df["seed"] = _seed_from(f.name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    tables = {}
    for arm in ("binary", "survival"):
        acc = _concat(arm, "acc")
        cal = _concat(arm, "cal")
        if not acc.empty:
            acc.to_csv(RES / f"accuracy_{arm}.csv", index=False)
        if not cal.empty:
            cal.to_csv(RES / f"calibration_{arm}.csv", index=False)
        tables[arm] = (acc, cal)

    # summary: headline discrimination + BSS(train) + BSS(other), mean/SD over seeds
    summary = []
    disc_metric = {"binary": "auc", "survival": "cindex"}
    for arm, (acc, cal) in tables.items():
        if acc.empty:
            continue
        keys = ["dem", "pheno", "method"]
        dm = disc_metric[arm]
        d = acc[acc.metric == dm].groupby(keys).value.agg(["mean", "std"]).reset_index()
        merged = d.rename(columns={"mean": f"{dm}_mean", "std": f"{dm}_sd"})
        if not cal.empty:
            for binval, tag in [("train_deme", "bss_train"), ("other_deme", "bss_other")]:
                sub = cal[(cal.metric == "bss") & (cal.ancestry_bin_kind == "train_ancestry")
                          & (cal.ancestry_bin == binval)]
                g = sub.groupby(keys).value.agg(["mean", "std"]).reset_index()
                g = g.rename(columns={"mean": f"{tag}_mean", "std": f"{tag}_sd"})
                merged = merged.merge(g, on=keys, how="left")
        merged.insert(0, "arm", arm)
        summary.append(merged)
    if summary:
        out = pd.concat(summary, ignore_index=True)
        out.to_csv(RES / "summary_table.csv", index=False)
        print(out.round(4).to_string(index=False))
    print("ANALYZE_DONE")


if __name__ == "__main__":
    main()
