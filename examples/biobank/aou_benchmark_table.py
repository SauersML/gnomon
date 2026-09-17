#!/usr/bin/env python3
"""Tabulate a benchmark run's digest tokens, African and admixed American ancestry first.

The workspace task publishes each aggregate result as the name of an empty
object; a listing of those names is all this reads. The headline tables are the
held-out gains in African and admixed American ancestry, where a
European-trained score gains least; European ancestry and all held-out
participants follow as comparators. A cell is printed only over at least the
reporting minimum of participants, cases and controls.

  aou_benchmark_table.py TOKENS   one object name per line, e.g. a gsutil ls listing
"""
from __future__ import annotations

import argparse
from pathlib import Path

HEADLINE = (("ancestry_afr", "African ancestry"), ("ancestry_amr", "admixed American ancestry"))
COMPARATORS = (("ancestry_eur", "European ancestry"), ("overall", "all held-out participants"))
DEVELOPMENT_ORDER = ("multi_ancestry", "european")
MINIMUM = 20
# The smallest true AUC difference a two-sided 5 % DeLong test detects with
# 80 % power at a cell's own support: (z_0.975 + z_0.80) standard errors.
DETECTABLE_SE_MULTIPLE = 1.959963984540054 + 0.8416212335729143


def number(text):
    """A digest value: signs travel as m and p."""
    return float(text.replace("m", "-").replace("p", "+"))


def parse(names):
    """Score facts, per-method cells, paired deltas and pre-cap support from digest object names."""
    scores, cells, deltas, support = {}, {}, {}, {}
    for raw in names:
        name = raw.strip().rsplit("/", 1)[-1]
        if not (name.startswith("digest__") and name.endswith(".txt")):
            continue
        parts = name[len("digest__"):-len(".txt")].split("__")
        if len(parts) < 3 or parts[1] == "cohort":
            continue
        disease, pgs, rest = parts[0], parts[1], parts[2:]
        if pgs == "support":
            if len(rest) == 3 and rest[1] in ("eligible", "cases"):
                support.setdefault((disease, rest[0]), {})[rest[1]] = number(rest[2])
            continue
        facts = scores.setdefault((disease, pgs), {})
        if rest[0] == "development" and len(rest) == 2:
            facts["development"] = rest[1]
        elif rest[0] == "gnomon" and rest[1] in ("status", "fit_wall") and len(rest) == 3:
            facts[f"gnomon_{rest[1]}"] = rest[2]
        elif rest[0] == "delta" and len(rest) == 7 and rest[2] == "vs":
            _, method, _, reference, group, key, value = rest
            deltas.setdefault((disease, pgs, method, reference, group), {})[key] = number(value)
        elif len(rest) == 4 and rest[0] != "all":
            method, group, metric, value = rest
            cells.setdefault((disease, pgs, method, group), {})[metric] = number(value)
    return scores, cells, deltas, support


def support_table(support, minimum=MINIMUM):
    rows = ["| disease | ancestry | eligible participants | cases |", "| --- | --- | --- | --- |"]
    for (disease, group), found in sorted(support.items()):
        n, cases = found.get("eligible"), found.get("cases")
        if n is None or cases is None or min(cases, n - cases) < minimum:
            continue
        rows.append(f"| {disease.replace('_', ' ')} | {group.removeprefix('ancestry_').upper()} "
                    f"| {int(n):,} | {int(cases):,} |")
    return "\n".join(rows)


def difference(deltas, key):
    found = deltas.get(key, {})
    if "auc_difference" not in found or "auc_difference_se" not in found:
        return "n/a", "n/a"
    se = found["auc_difference_se"]
    return f"{found['auc_difference']:+.4f} ± {se:.4f}", f"{DETECTABLE_SE_MULTIPLE * se:.4f}"


def slope(cells, key):
    found = cells.get(key, {})
    return f"{found['calibration_slope']:.3f}" if "calibration_slope" in found else "n/a"


def table(scores, cells, deltas, group, minimum=MINIMUM):
    rows = ["| disease | score | development | n | cases | AUC covariates | ΔAUC standard − covariates "
            "| detectable | ΔAUC gnomon − standard | detectable | calibration slope standard "
            "| calibration slope gnomon | gnomon fit |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    def order(item):
        (disease, pgs), facts = item
        development = facts.get("development")
        rank = DEVELOPMENT_ORDER.index(development) if development in DEVELOPMENT_ORDER else len(DEVELOPMENT_ORDER)
        return disease, rank, pgs
    for (disease, pgs), facts in sorted(scores.items(), key=order):
        support = cells.get((disease, pgs, "standard", group), {})
        n, cases = support.get("n"), support.get("cases")
        label = [disease.replace("_", " "), pgs.upper(), facts.get("development", "not recorded").replace("_", "-")]
        if n is None or cases is None or min(cases, n - cases) < minimum:
            rows.append("| " + " | ".join(label + ["insufficient support"] + ["n/a"] * 9) + " |")
            continue
        covariates = cells.get((disease, pgs, "covariates", group), {}).get("auc")
        rows.append("| " + " | ".join(label + [
            f"{int(n):,}", f"{int(cases):,}", "n/a" if covariates is None else f"{covariates:.4f}",
            *difference(deltas, (disease, pgs, "standard", "covariates", group)),
            *difference(deltas, (disease, pgs, "gnomon", "standard", group)),
            slope(cells, (disease, pgs, "standard", group)), slope(cells, (disease, pgs, "gnomon", group)),
            facts.get("gnomon_status", "n/a").replace("_", " ")]) + " |")
    rows.append("\n'detectable' is the smallest true AUC difference a two-sided 5 % DeLong test detects "
                f"with 80 % power at that cell's support ({DETECTABLE_SE_MULTIPLE:.4f} standard errors).")
    return "\n".join(rows)


def render(names, minimum=MINIMUM):
    scores, cells, deltas, support = parse(names)
    sections = []
    if support:
        sections.append("### Eligible support before the per-ancestry cap\n\n" + support_table(support, minimum))
    for kind, groups in (("headline", HEADLINE), ("comparator", COMPARATORS)):
        for group, title in groups:
            sections.append(f"### Held-out gains in {title} ({kind})\n\n"
                            + table(scores, cells, deltas, group, minimum))
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tokens", type=Path)
    args = parser.parse_args()
    print(render(args.tokens.read_text().splitlines()), end="")


if __name__ == "__main__":
    main()
