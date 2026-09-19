#!/usr/bin/env python3
"""Print the study's single results table in the terminal from its digest names.

  tabulate_study.py TOKENS [--stratum overall] [--metrics auc,brier,...]

TOKENS is a text file of token names, one per line: study.py writes
work/tokens.txt, and `submit_study.py tokens RUN` prints a finished AoU run's.
Only aggregate names are read, so this runs anywhere, with no participant data.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from study import digest  # noqa: E402

KEY_FIELDS = set(digest.KEYS)


def cell(value):
    if value is None:
        return "."
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def print_table(title, header, rows):
    if not rows:
        return
    widths = [max(len(str(h)), *(len(cell(r[i])) for r in rows)) for i, h in enumerate(header)]
    print(f"\n{title}")
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(cell(v).ljust(w) for v, w in zip(row, widths)))


def exclusion_caveats(results):
    """study-audit: an exclusion match censors, and where the exclusion is not
    independent of the target codes (T1D among T2D-coded people, bipolar among
    MDD) that censoring can be dependent. A disease whose exclusion exits pass
    1% of its survival frame says so, from released counts only."""
    lines = []
    for row in results:
        if row["model"] != "flow_survival":
            continue
        steps = sorted(m for m in row if digest.STEP.fullmatch(m))
        end, exits = (row[steps[-1]] if steps else None), row.get("exclusion_exits_count")
        if exits is None:
            lines.append(f"CAVEAT {row['disease']}: exclusion exits withheld (small-cell rule); "
                         "their share of the survival frame is not shown")
        elif end and exits > 0.01 * end:
            lines.append(f"CAVEAT {row['disease']}: exclusion exits are {exits / end:.1%} of the survival frame; "
                         "censoring at an exclusion match may be dependent")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--stratum", default="overall", help="stratum of the main tables (slugged label)")
    parser.add_argument("--metrics", help="comma-separated metric columns (default: every one present)")
    args = parser.parse_args()
    results, operations = digest.parse(args.tokens.read_text().splitlines())

    ops = defaultdict(dict)
    for row in operations:
        ops[row["scope"]][row["item"]] = row
    run = ops.get("study", {}).get("run", {})
    print("study config {config_sha256_12}  vcpus {vcpus}  threads {threads}  memory_gb {memory_budget_gb}  "
          "attempts {attempts}  vcpu_hours {vcpu_hours}  bigquery_bytes {bigquery_bytes_billed}  "
          "outer_test_looks {outer_test_looks}  "
          "horizons {horizons}".format_map(defaultdict(lambda: "?", run)))
    print("label {label}  run {run_kind}  gam {gam_commit_12}  tables {tables_source} {tables_sha256_12} "
          "seed {tables_seed} {tables_scenario}  cdr_cutoff {cdr_cutoff} ({cdr_cutoff_source})".format_map(
        defaultdict(lambda: "?", run)))
    if run.get("caveats"):
        print("CAVEATS: " + str(run["caveats"]).replace("_and_", "; "))
    for line in exclusion_caveats(results):
        print(line)
    for row in results:
        if row["model"] == "pc_scale":
            print("PC SD over the base: " + "  ".join(f"{m[3:].upper()} {cell(v)}" for m, v in sorted(
                ((m, v) for m, v in row.items() if m.startswith("sd_pc")), key=lambda item: int(item[0][5:]))))
    timings = [(item, row.get("wall_seconds")) for item, row in ops.get("timing", {}).items()]
    if timings:
        total = sum(s for _, s in timings if isinstance(s, (int, float)))
        print("stage wall (s): " + "  ".join(f"{item} {cell(s)}" for item, s in timings) + f"  total {cell(total)}")

    model_rows = [r for r in results if not digest.special(r["model"])]
    metrics = args.metrics.split(",") if args.metrics else sorted(
        {m for row in model_rows for m in row if m not in KEY_FIELDS})
    for model in ("binary", "survival"):
        chosen = [r for r in model_rows if r["model"] == model and r["fit"] == "pooled"
                  and r["stratum"] == args.stratum]
        rows = [[r["disease"], r["variant"], r["horizon"], *(r.get(m) for m in metrics)]
                for r in sorted(chosen, key=lambda r: (r["disease"], r["horizon"], r["variant"]))]
        print_table(f"{model}: pooled fits, stratum {args.stratum}", ["disease", "variant", "horizon", *metrics], rows)
        logo = [r for r in model_rows if r["model"] == model and r["fit"].startswith("logo_")]
        rows = [[r["disease"], r["variant"], r["fit"][len("logo_"):], r["horizon"], *(r.get(m) for m in metrics)]
                for r in sorted(logo, key=lambda r: (r["disease"], r["fit"], r["horizon"], r["variant"]))]
        print_table(f"{model}: leave-one-group-out refits, scored on the held-out group's test rows",
                    ["disease", "variant", "held_out", "horizon", *metrics], rows)

    fit_fields = ["fits", "ok", "failed", "median_seconds", "max_seconds", "cpu_seconds", "threads", "max_rss_mb"]
    print_table("fits (disease_model_variant_component)", ["fit", *fit_fields],
                [[item, *(row.get(f) for f in fit_fields)] for item, row in sorted(ops.get("fits", {}).items())])
    print_table("fit failures (status_category: count)", ["fit", "failures"],
                [[item, "  ".join(f"{k} {v}" for k, v in sorted(row.items()) if k not in ("scope", "item"))]
                 for item, row in sorted(ops.get("fit_failures", {}).items())])

    for row in sorted((r for r in results if digest.special(r["model"])), key=lambda r: (r["model"], r["disease"])):
        values = [(k, v) for k, v in sorted(row.items()) if k not in KEY_FIELDS]
        print(f"\n{row['model']} {row['disease']} {row['stratum']}: " + "  ".join(f"{k} {cell(v)}" for k, v in values))


if __name__ == "__main__":
    main()
