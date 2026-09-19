"""Aggregate result tokens: the only study output that leaves the workspace.

The workspace blocks downloads, so results leave as object NAMES:

    c__<ver>__<k>__<metric>__<metric>...        the model rows' column order, in parts k
    s__<ver>__<status>__<status>...             their status vocabulary
    r__<ver>__<disease>__<model>__<fit>__<stratum>__<horizon>__<variant>-<v>_<v>_...__<variant>-...
    d__<disease>__<model>__<fit>__<stratum>__<horizon>__p<k>__<variant>.<metric>-<value>__...
    o__<scope>__<item>__p<k>__<metric>-<value>__...

A model row (evaluate's) is positional: one value per column, "x" for none and
"s<i>" for the i-th status word; a name holds as many variants of one cell as
fit. Flow, follow-up and descendant rows, whose metric names vary, use named
pairs (d__), and operation rows (o__: timings, fit outcomes, cost; no
participant data) likewise. Labels are [a-z0-9_] ("__" separates fields) and
numbers have four significant digits with "-"/"+" spelled "m"/"p". `parse`
reads every form back into rows keyed like evaluate's.

AoU policy: no participant count from 1 to 20 may be published or derived.
`suppress` (SPEC section 8, S9) removes counts and proportions wherever one
would be, to a fixed point:
- a cell whose counts, within-cell differences or horizon increments fall in
  1..20 publishes only its status;
- an axis whose categories partition a cell withholds every category's counts
  when one is withheld, the unlisted remainder is small, or the overall cell is
  withheld; a withheld region axis withholds the division axis nested in it;
- a nested cell (survival inside binary; an outer-test evaluation cell inside
  its whole-cohort cell) withholds its counts when its n or cases at a horizon
  sit 1..20 below the outer cell's;
- leave-one-group-out rows repeat pooled cells, so they publish scores only;
- a proportion goes with its cell's counts.
Exclusion flows are count chains that never step by 1..20 (`flow_rows`).
`encode` then runs study-audit's disclosure.audit on exactly the names that
would leave, per disease, and refuses on any finding: that audit is the gate.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import numbers
import re

from study import disclosure

LIMIT = disclosure.LIMIT
KEYS = ("disease", "model", "variant", "fit", "stratum", "horizon")
CELL = ("disease", "model", "fit", "stratum", "horizon")
# GCS object names are at most 1024 bytes, and the run's prefix takes some.
NAME_BUDGET = 900
LABEL = re.compile(r"[a-z0-9_]+")
NUMBER = re.compile(r"m?\d+(\.\d+)?(e[mp]\d+)?")
AXES = ("ancestry", "region", "division", "ehr_site", "baseline_year", "entry_year", "sex", "age_band",
        "ses_quartile", "lookback_tertile", "risk")
INSUFFICIENT = "insufficient_support"
WITHHELD = "counts_withheld"
# Flow rows: step_<index>_<label> per shown exclusion step (the auditor's chain), plus subgroup counts.
STEP = re.compile(r"step_(\d+)_[a-z0-9_]+")
SUBGROUP = re.compile(r"[a-z0-9_]+_count")
DESCENDANT = re.compile(r"c\d+_n")
REACH = re.compile(r"reach_[a-z0-9_]+")
# Exact fractions of a row's n, per model: the people each counts and the rest are both released-safe.
FRACTIONS = {"followup": REACH, "ehr_domains": re.compile(r"extended_by_[a-z0-9_]+|end_from_long_visit")}
# Whole-cohort counts per frame (phenotypes flow by_ancestry): ordinary cells, so
# suppress and the auditor partition them like evaluation cells.
COHORT_COUNTS = frozenset({"n", "cases", "deaths"})
# The follow-up rows' fixed metrics; an exact proportion names its count ratio.
FOLLOWUP = {
    "followup": {"n": {"type": "count"}},
    "ehr_domains": {"n": {"type": "count"}},
    "followup_ehr": {"n": {"type": "count"}, "with_ehr_n": {"type": "count"},
                     "fraction_without_ehr": {"type": "proportion", "of": "without_ehr_count", "per": "n"},
                     "fraction_obs_end_after_ehr_end": {"type": "proportion", "of": "obs_end_after_count",
                                                        "per": "with_ehr_n"},
                     "median_gap_years": {"type": "score"}, "median_positive_gap_years": {"type": "score"}},
    # The SD of each PC the models see, over the base cohort: whole-base aggregates, no counts.
    "pc_scale": {f"sd_pc{index}": {"type": "score"} for index in range(1, 65)},  # CohortConfig allows 1..64 PCs
}
# Inner model -> the models whose cells contain its cells' people.
NESTED = {"survival": ("binary", "cohort_survival"), "binary": ("cohort_binary",),
          "cohort_survival": ("cohort_binary",)}


def slug(text):
    """A label in [a-z0-9_], never truncated: `pack` bounds each name by NAME_BUDGET."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if not cleaned:
        raise ValueError("empty digest label")
    return cleaned


def check_caveats(caveats):
    """A run's caveats reach its digest verbatim, joined by "_and_", which tabulate splits on."""
    for caveat in caveats:
        if not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", caveat) or "_and_" in f"_{caveat}_":
            raise ValueError(f"a caveat is a fixed label of [a-z0-9] words, none of them 'and': {caveat!r}")


def label(text, length=40):
    """A fixed-vocabulary word cut to `length` characters (never a number)."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:length].strip("_")
    return cleaned if cleaned and not NUMBER.fullmatch(cleaned) else f"x_{cleaned or 'empty'}"[:length]


def token(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("digest values must be numbers")
    if isinstance(value, numbers.Integral):
        text = str(int(value))
    else:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("digest values must be finite")
        text = f"{value:.4g}"
    return text.replace("-", "m").replace("+", "p")


def untoken(text):
    text = text.replace("m", "-").replace("p", "+")
    return int(text) if re.fullmatch(r"-?\d+", text) else float(text)


def horizon_label(horizon):
    return "all" if horizon is None else "h" + token(float(horizon))


def is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def is_missing(value):
    return value is None or (is_number(value) and not isinstance(value, numbers.Integral) and math.isnan(value))


def small(value, limit=LIMIT):
    return is_number(value) and 1 <= abs(value) <= limit


def special(model):
    """Rows whose metric names vary (flows, descendants, follow-up): named pairs,
    passed through suppress as built (their builders keep them safe)."""
    return str(model).startswith("flow_") or model in ("descendants", "followup", "followup_ehr", "ehr_domains",
                                                       "pc_scale")


# --------------------------------------------------------------------------- #
# metric registry
# --------------------------------------------------------------------------- #
class Registry:
    """metric -> {"type": count | proportion | score}, from evaluate.METRICS
    (a bare kind, or a dict with an exact "of"/"per" where one exists), plus
    the flow, follow-up and descendant counts. Default deny: a metric the
    registry does not name is refused."""
    def __init__(self, metrics=None):
        if metrics is None:
            from study import evaluate
            metrics = getattr(evaluate, "REGISTRY", None) or getattr(evaluate, "METRICS", {})
        self.specs = {m: (dict(v) if isinstance(v, dict) else {"type": v}) for m, v in metrics.items()}
        for metric, spec in self.specs.items():
            if spec["type"] not in ("count", "proportion", "score"):
                raise ValueError(f"metric {metric} has unknown type {spec['type']}")

    def kind(self, metric, row=None):
        if row is not None and row["model"] in FOLLOWUP:
            if metric in FOLLOWUP[row["model"]]:
                return FOLLOWUP[row["model"]][metric]["type"]
            if row["model"] in FRACTIONS and FRACTIONS[row["model"]].fullmatch(metric):
                return "proportion"
            raise ValueError(f"{row['model']} metric {metric} is not registered")
        if row is not None and str(row["model"]).startswith("cohort_"):
            if metric in COHORT_COUNTS:
                return "count"
            raise ValueError(f"cohort metric {metric} is not one of {sorted(COHORT_COUNTS)}")
        if row is not None and special(row["model"]):
            if STEP.fullmatch(metric) or SUBGROUP.fullmatch(metric) or DESCENDANT.fullmatch(metric):
                return "count"
            raise ValueError(f"{row['model']} metric {metric} is not a flow step, subgroup or descendant count")
        if metric not in self.specs:
            raise ValueError(f"metric {metric} is not in the registry")
        return self.specs[metric]["type"]

    def audit_registry(self, rows):
        """The auditor's registry for these rows. A proportion stays bare unless
        it is an exact count ratio (the follow-up reach fractions): the auditor
        then requires it beside all of its cell's counts."""
        specs = dict(self.specs)
        for row in rows:
            if row["model"] in FOLLOWUP:
                specs.update({m: FOLLOWUP[row["model"]][m] for m in row if m in FOLLOWUP[row["model"]]})
                specs.update({m: {"type": "proportion", "of": f"{m}_count", "per": "n"}
                              for m in row if row["model"] in FRACTIONS and FRACTIONS[row["model"]].fullmatch(m)})
            elif str(row["model"]).startswith("cohort_"):
                specs.update({m: {"type": "count"} for m in row if m in COHORT_COUNTS})
            elif special(row["model"]):
                specs.update({m: {"type": "count"} for m, v in row.items()
                              if m not in KEYS and not isinstance(v, str)})
        return specs


def split_stratum(stratum):
    """(axis, value) of a stratum label; "overall" is (None, None)."""
    text = slug(stratum)
    if text == "overall":
        return None, None
    for axis in sorted(AXES, key=len, reverse=True):
        if text.startswith(axis + "_") and len(text) > len(axis) + 1:
            return axis, text[len(axis) + 1:]
    raise ValueError(f"stratum {stratum!r} names no known axis")


def logo_group(fit):
    """The slugged stratum a LOGO fit holds out ("logo:ancestry:afr" -> "ancestry_afr")."""
    _, axis, group = fit.split(":", 2)
    return slug(f"{axis}_{group}")


# --------------------------------------------------------------------------- #
# suppression
# --------------------------------------------------------------------------- #
def suppress(rows, limit=LIMIT, registry=None):
    """The rows safe to publish: counts and proportions removed wherever a count
    from 1 to `limit` would be shown or derivable, statuses in their place."""
    registry = Registry() if registry is None else registry
    rows = [dict(row) for row in rows]

    def cell(row):
        return (row["disease"], row["model"], slug(row["stratum"]))

    # Cells: (disease, model, stratum) over all horizons. Counts do not depend on
    # the variant, so every pooled row of one cell must carry the same counts.
    cells = defaultdict(dict)
    for row in rows:
        kinds = {m: registry.kind(m, row) for m, v in row.items()
                 if m not in KEYS and v is not None and not isinstance(v, str)}
        if row["fit"] != "pooled" or special(row["model"]):
            continue
        horizon = row.get("horizon")
        for metric, kind in kinds.items():
            if kind != "count" or not is_number(row[metric]):
                continue
            seen = cells[cell(row)].setdefault(horizon, {}).get(metric)
            if seen is not None and seen != row[metric]:
                raise ValueError(f"cell {cell(row)} at {horizon} has two values of {metric}")
            cells[cell(row)][horizon][metric] = row[metric]

    def derivable_small(by_horizon):
        for values in by_horizon.values():
            counts = list(values.values())
            if any(small(v, limit) for v in counts):
                return True
            if any(small(a - b, limit) for i, a in enumerate(counts) for b in counts[i + 1:]):
                return True
            if "n" in values and small(values["n"] - sum(v for m, v in values.items() if m != "n"), limit):
                return True
        horizons = sorted(h for h in by_horizon if h is not None)
        for metric in {m for values in by_horizon.values() for m in values}:
            series = [by_horizon[h][metric] for h in horizons if metric in by_horizon[h]]
            if any(small(a - b, limit) for i, a in enumerate(series) for b in series[i + 1:]):
                return True
        return False

    insufficient = {key for key, by_horizon in cells.items() if derivable_small(by_horizon)}
    withheld = set(insufficient)
    families = defaultdict(set)
    for key in cells:
        axis, _ = split_stratum(key[2])
        if axis is not None:
            families[(key[0], key[1], axis)].add(key)

    def remainder_small(disease, model, members):
        overall = cells.get((disease, model, "overall"))
        if overall is None:
            return True
        for horizon, values in overall.items():
            for metric, total in values.items():
                parts = [cells[m].get(horizon, {}).get(metric) for m in members]
                if any(p is None for p in parts) or small(total - sum(parts), limit):
                    return True
        return False

    changed = True
    while changed:
        changed = False
        for (disease, model, axis), members in families.items():
            if members <= withheld:
                continue
            if (members & withheld or remainder_small(disease, model, members)
                    or (disease, model, "overall") in withheld
                    or (axis == "division" and families.get((disease, model, "region"), set()) & withheld)):
                withheld |= members
                changed = True
        # Nested cohorts, stratum by stratum: survival inside binary (checked for
        # every disease; withholding where C3 breaks nesting is only conservative),
        # and each evaluation cell (outer-test rows) inside its whole-cohort cell.
        # The inner cell is withheld when its n or cases at a horizon sit 1..20
        # below the outer cell's.
        for key in list(cells):
            disease, model, stratum = key
            if model not in NESTED or key in withheld:
                continue
            for outer_model in NESTED[model]:
                outer_key = (disease, outer_model, stratum)
                outer = cells.get(outer_key, {}).get(None, {})
                if outer_key in withheld or any(
                        metric in values and metric in outer and small(outer[metric] - values[metric], limit)
                        for values in cells[key].values() for metric in ("n", "cases")):
                    withheld.add(key)
                    changed = True
                    break

    def scores_only(row, **status):
        kept = {k: v for k, v in row.items()
                if k in KEYS or isinstance(v, str) or (v is not None and registry.kind(k, row) == "score")}
        return {**kept, **status}

    out = []
    for row in rows:
        if special(row["model"]):
            out.append(row)
        elif row["fit"] != "pooled":
            # A LOGO cell is the pooled cell of its held-out group: scores only,
            # and none at all where that group's pooled cell is insufficient.
            if split_stratum(row["stratum"])[0] is not None:
                continue
            if (row["disease"], row["model"], logo_group(row["fit"])) in insufficient:
                continue
            out.append(scores_only(row))
        elif cell(row) in insufficient:
            out.append({**{k: row[k] for k in KEYS if k in row}, "support": INSUFFICIENT})
        elif cell(row) in withheld:
            out.append(scores_only(row, counts=WITHHELD))
        else:
            out.append(row)
    return out




def flow_rows(disease, flows, limit=LIMIT, subgroups=None):
    """Exclusion flows as count chains that never step by 1..limit.

    A chain's ends are fixed (its first count is the enclosing cohort, its last
    the frame itself); an intermediate step shows only when it is safely apart
    from the last shown step and from the end, and the end only when it is
    safely apart from the last shown step. Shown steps are step_<index>_<label>
    (study-audit's chain pattern), and a subgroup count shows only when it and
    its complement in the end count are both safe."""
    rows = []
    for frame, steps in flows.items():
        if isinstance(steps, dict):
            steps = steps.get("steps", [])
        if not isinstance(steps, list) or not steps:
            continue
        row = {"disease": disease, "model": f"flow_{slug(frame)}", "variant": "all", "fit": "pooled",
               "stratum": "overall", "horizon": None}
        counts = [int(step["n"]) for step in steps]
        end, shown = counts[-1], None
        for index, n in enumerate(counts):
            last = index == len(counts) - 1
            if small(n, limit) or (shown is not None and small(shown - n, limit)):
                continue
            if not last and small(n - end, limit):
                continue
            row[f"step_{index:02d}_{label(steps[index]['step'], 36)}"] = n
            shown = n
        if shown == end and not small(end, limit):
            for name, count in (subgroups or {}).get(frame, {}).items():
                if not small(count, limit) and not small(end - count, limit):
                    row[f"{label(name, 36)}_count"] = int(count)
        if len(row) > len(KEYS):
            rows.append(row)
    return rows


def subgroup_parts(rows):
    """The auditor's relations for flow subgroups: every shown step contains
    each subgroup of its frame's end count."""
    parts = defaultdict(list)
    for row in rows:
        if not str(row["model"]).startswith("flow_"):
            continue
        steps = [m for m in row if STEP.fullmatch(m)]
        for metric in row:
            if SUBGROUP.fullmatch(metric):
                for step in steps:
                    relation = (step, [metric, f"rest_{step}_{metric}"])
                    if relation not in parts[row["model"]]:
                        parts[row["model"]].append(relation)
    for row in rows:
        if row["model"] in FRACTIONS:
            for metric in row:
                relation = ("n", [f"{metric}_count", f"{metric}_short"])
                if FRACTIONS[row["model"]].fullmatch(metric) and relation not in parts[row["model"]]:
                    parts[row["model"]].append(relation)
        if row["model"] == "followup_ehr" and "with_ehr_n" in row:
            parts["followup_ehr"] = [("n", ["with_ehr_n", "without_ehr_count"]),
                                     ("with_ehr_n", ["obs_end_after_count", "obs_end_not_after_count"])]
    return dict(parts)


def descendant_rows(table, limit=LIMIT):
    """The outcome-blind phenotype check: the concepts recorded for the most
    people under each root, and under each excluded branch (disease label
    <root>_<branch>). Concepts leave by OMOP concept_id only; people overlap
    across concepts, so the counts form no partition."""
    rows = {}
    for record in table.to_dict("records"):
        if small(record["n_persons"], limit):
            continue
        root, code = str(record["root_code"] or record["snomed_code"]), str(record["snomed_code"])
        name = root if record["role"] == "root" else f"{root}_{code}"
        row = rows.setdefault(name, {"disease": name, "model": "descendants", "variant": "all",
                                     "fit": "pooled", "stratum": "overall", "horizon": None})
        row[f"c{int(record['concept_id'])}_n"] = int(record["n_persons"])
    return list(rows.values())


def cohort_rows(disease, by_ancestry):
    """Whole-cohort people and outcomes per frame, overall and by ancestry
    (phenotypes' by_ancestry partitions each frame), as ordinary cells that
    suppress and the auditor treat like any other: model cohort_binary (n,
    cases) and cohort_survival (n, cases = disease events, deaths). An exclusion match
    censors (study-cohort); its count is the survival flow's exclusion_exits subgroup."""
    fields = {"cohort_binary": {"n": "binary_n", "cases": "binary_cases"},
              "cohort_survival": {"n": "survival_n", "cases": "survival_disease", "deaths": "survival_death"}}
    rows = []
    for model, names_ in fields.items():
        cells = {f"ancestry_{label(group)}": {metric: int(counts[source]) for metric, source in names_.items()}
                 for group, counts in sorted(by_ancestry.items())}
        overall = {metric: sum(cell[metric] for cell in cells.values()) for metric in names_}
        for stratum, counts in [("overall", overall), *cells.items()]:
            rows.append({"disease": disease, "model": model, "variant": "all", "fit": "pooled", "stratum": stratum,
                         "horizon": None, **counts})
    return rows


def followup_rows(administrative, limit=LIMIT):
    """The outcome-blind follow-up behind the horizon rule: how many of the
    survival-eligible cohort can reach each candidate horizon. A fraction shows
    only when both the people reaching it and those who cannot are safely more
    than the small-cell maximum, with a margin for its four-digit rounding."""
    n = int(administrative["n"])
    if n == 0 or small(n, limit):
        return []
    row = {"disease": "base", "model": "followup", "variant": "all", "fit": "pooled", "stratum": "overall",
           "horizon": None, "n": n}
    for horizon, fraction in administrative["fraction_reaching"].items():
        if safe_fraction(fraction, n, limit):
            row[f"reach_{label('h' + str(horizon))}"] = float(fraction)
    return [row]


def pc_scale_rows(sds):
    """The SD of each PC the models see over the base cohort (study-sim, lead
    09-19): it confirms the PC geometry the joint Duchon tests assume. Whole-base
    aggregates carry no count, so there is nothing to suppress."""
    return [{"disease": "base", "model": "pc_scale", "variant": "all", "fit": "pooled", "stratum": "overall",
             "horizon": None, **{f"sd_{label(pc)}": float(sd) for pc, sd in sds.items()}}]


def safe_fraction(fraction, n, limit=LIMIT):
    """Whether a fraction of n people pins no count of 1..limit: the people it
    counts and the rest are both safely more than limit, with a margin for its
    four-digit rounding. An unknown (null) fraction is never released."""
    if fraction is None or not is_number(fraction) or not math.isfinite(fraction):
        return False
    reach, margin = fraction * n, 1 + n * 5e-4
    return reach - margin > limit and n - reach - margin > limit


def ehr_rows(ehr, limit=LIMIT):
    """The outcome-blind EHR follow-up check (SPEC section 3, ehr_end censoring):
    who has no EHR at all, how often obs_end runs past ehr_end, and by how much.
    The share of deaths after ehr_end is not released: its denominator (base
    deaths) is not a released count."""
    n = int(ehr["n"])
    if n == 0 or small(n, limit):
        return []
    row = {"disease": "base", "model": "followup_ehr", "variant": "all", "fit": "pooled", "stratum": "overall",
           "horizon": None, "n": n}
    if safe_fraction(ehr.get("fraction_without_ehr"), n, limit):
        with_ehr = n - round(n * ehr["fraction_without_ehr"])
        row.update(with_ehr_n=with_ehr, fraction_without_ehr=float(ehr["fraction_without_ehr"]))
        if safe_fraction(ehr.get("fraction_obs_end_after_ehr_end"), with_ehr, limit):
            row["fraction_obs_end_after_ehr_end"] = float(ehr["fraction_obs_end_after_ehr_end"])
    for name in ("median_gap_years", "median_positive_gap_years"):
        if ehr.get(name) is not None and math.isfinite(ehr[name]):
            row[name] = float(ehr[name])
    return [row]


def ehr_domain_rows(manifest, limit=LIMIT):
    """The outcome-blind EHR-domain check over the whole CDR: how often each
    extra domain moved a person's ehr_end later, and how often ehr_end is a
    long visit's end, as exact fractions of manifest["ehr_people"] (the CDR
    people with any EHR-sourced row). Each shows only under safe_fraction."""
    n = manifest.get("ehr_people")
    if not n or small(n, limit):
        return []
    row = {"disease": "cdr", "model": "ehr_domains", "variant": "all", "fit": "pooled", "stratum": "overall",
           "horizon": None, "n": int(n)}
    for domain, fraction in (manifest.get("ehr_extended_by") or {}).items():
        if safe_fraction(fraction, n, limit):
            row[f"extended_by_{label(domain, 24)}"] = float(fraction)
    if safe_fraction(manifest.get("ehr_end_from_long_visit"), n, limit):
        row["end_from_long_visit"] = float(manifest["ehr_end_from_long_visit"])
    return [row]


# --------------------------------------------------------------------------- #
# names
# --------------------------------------------------------------------------- #
def checked(row, metric, value, registry, limit):
    """The token for one value, refusing any count from 1 to `limit`."""
    if isinstance(value, str):
        if not LABEL.fullmatch(value) or NUMBER.fullmatch(value):
            raise ValueError(f"digest status {metric} is not a fixed label")
        return value
    if registry.kind(metric, row) == "count" and small(value, limit):
        raise ValueError(f"refusing to publish a {metric} of 1 to {limit}")
    return token(value)


def cell_key(row):
    return [slug(row["disease"]), slug(row["model"]), slug(row["fit"]), slug(row["stratum"]),
            horizon_label(row.get("horizon"))]


def pack(head, items, separator="__"):
    """Names of `head__items...`, each within the name budget."""
    names, current = [], []
    for item in items:
        if len(head) + len(item) + 2 > NAME_BUDGET:
            raise ValueError("one digest item is longer than a name")
        if current and len(head) + sum(len(x) + 2 for x in current) + len(item) + 2 > NAME_BUDGET:
            names.append(separator.join([head, *current]))
            current = []
        current.append(item)
    if current:
        names.append(separator.join([head, *current]))
    return names


def numbered(base, items):
    """Names `base__p<k>__items...`, budgeted as if every part label were the widest."""
    widest = f"{base}__p99"
    return [f"{base}__p{k}" + name[len(widest):] for k, name in enumerate(pack(widest, items))]


def names(rows, limit=LIMIT, registry=None):
    """Result names for suppressed rows; any count 1..limit left is refused."""
    registry = Registry() if registry is None else registry
    model_rows = [row for row in rows if not special(row["model"])]
    columns = sorted({m for row in model_rows for m, v in row.items() if m not in KEYS and not is_missing(v)})
    statuses = sorted({v for row in model_rows for m, v in row.items() if m not in KEYS and isinstance(v, str)})
    version = hashlib.sha256(" ".join(columns + ["|"] + statuses).encode()).hexdigest()[:8]
    out = []
    if model_rows:
        out += numbered(f"c__{version}", [slug(c) for c in columns])
        out += numbered(f"s__{version}", statuses) if statuses else []
    code = {status: f"s{i}" for i, status in enumerate(statuses)}
    cells = defaultdict(list)
    for row in model_rows:
        values = []
        for metric in columns:
            value = row.get(metric)
            if is_missing(value):
                values.append("x")
            elif isinstance(value, str):
                checked(row, metric, value, registry, limit)
                values.append(code[value])
            else:
                values.append(checked(row, metric, value, registry, limit))
        cells[tuple(cell_key(row))].append(f"{slug(row['variant'])}-{'_'.join(values)}")
    for key, items in sorted(cells.items()):
        if len({item.split("-", 1)[0] for item in items}) != len(items):
            raise ValueError(f"cell {key} repeats a variant")
        out += pack("__".join([f"r__{version}", *key]), items)
    for row in rows:
        if special(row["model"]):
            items = [f"{slug(row['variant'])}.{slug(m)}-{checked(row, m, v, registry, limit)}"
                     for m, v in sorted(row.items()) if m not in KEYS and not is_missing(v)]
            out += numbered("__".join(["d", *cell_key(row)]), items)
    if len(set(out)) != len(out):
        raise ValueError("two digest names coincide")
    return out


def operation_names(rows):
    """o__<scope>__<item>__<metric>-<value>__...: no participant data."""
    out, keys = [], set()
    for row in rows:
        # parse merges an operation row by (scope, item): two rows whose keys
        # slug alike ("a.b_c", "a_b.c") would read back as one.
        key = (slug(row["scope"]), slug(row["item"]))
        if key in keys:
            raise ValueError(f"two operation rows share the key {key}")
        keys.add(key)
        items = []
        for metric in sorted(set(row) - {"scope", "item"}):
            value = row[metric]
            if isinstance(value, str):
                if not LABEL.fullmatch(value) or NUMBER.fullmatch(value):
                    raise ValueError(f"operation status {metric} is not a fixed label")
                items.append(f"{slug(metric)}-{value}")
            else:
                items.append(f"{slug(metric)}-{token(value)}")
        out += pack("__".join(["o", *key]), items)
    if len(set(out)) != len(out):
        raise ValueError("two operation names coincide")
    return out


def audit(rows, registry, nested=()):
    """study-audit's differencing audit of exactly the rows `parse` gives back,
    one disease at a time. Evaluation cells (outer-test rows) nest in their
    whole-cohort cells for every disease; `nested` names the diseases whose
    survival cohort is also a subset of the binary cohort (no exclusion roots,
    SPEC C3)."""
    parsed, _ = parse(names(rows, LIMIT, registry))
    by_disease = defaultdict(list)
    for row in parsed:
        by_disease[row["disease"]].append(row)
    findings = []
    for disease, group in sorted(by_disease.items()):
        findings += disclosure.audit(
            group, registry.audit_registry(group), axes=AXES, parts=subgroup_parts(group),
            cumulative={"survival": {"n": "decreasing"}}, horizon_invariant=(), chain=STEP,
            nested_models=[("binary", "cohort_binary", ("n", "cases")), ("survival", "cohort_survival", ("n", "cases")),
                           *([("survival", "binary", ("n", "cases")), ("cohort_survival", "cohort_binary", ("n", "cases"))]
                             if disease in nested else [])])
    return findings


def encode(rows, operations, registry=None, nested=()):
    """(result names, operation names) after suppression, refused on any audit finding."""
    registry = Registry() if registry is None else registry
    safe = suppress(rows, LIMIT, registry)
    findings = audit(safe, registry, nested)
    if findings:
        described = "; ".join(f"{f['kind']} {f['what']}" for f in findings[:5])
        raise ValueError(f"disclosure audit found {len(findings)} derivable small counts: {described}")
    return names(safe, LIMIT, registry), operation_names(operations)


def parse(tokens):
    """(result rows, operation rows) back from names (bare, or paths/URIs)."""
    names_ = [text.strip().rsplit("/", 1)[-1] for text in tokens if text.strip()]
    columns, statuses = defaultdict(dict), defaultdict(dict)
    for name in names_:
        fields = name.split("__")
        if fields[0] in ("c", "s"):
            (columns if fields[0] == "c" else statuses)[fields[1]][int(fields[2][1:])] = fields[3:]
    order = {version: [c for k in sorted(parts) for c in parts[k]] for version, parts in columns.items()}
    vocabulary = {version: [s for k in sorted(parts) for s in parts[k]] for version, parts in statuses.items()}
    results, operations = {}, {}
    for name in names_:
        fields = name.split("__")
        if fields[0] == "r":
            version, key, items = fields[1], tuple(fields[2:7]), fields[7:]
            if version not in order:
                raise ValueError(f"row names of version {version} arrived without their column header")
            for item in items:
                variant, _, packed = item.partition("-")
                values = packed.split("_")
                if len(values) != len(order[version]):
                    raise ValueError(f"a row of version {version} has {len(values)} values, not {len(order[version])}")
                row = results.setdefault((*key, variant), dict(zip(CELL, key), variant=variant))
                for metric, value in zip(order[version], values):
                    if value == "x":
                        continue
                    row[metric] = (vocabulary[version][int(value[1:])] if value.startswith("s")
                                   else untoken(value))
        elif fields[0] == "d" and len(fields) >= 7:
            key, items = tuple(fields[1:6]), fields[7:]
            for item in items:
                left, _, value = item.partition("-")
                variant, _, metric = left.partition(".")
                row = results.setdefault((*key, variant), dict(zip(CELL, key), variant=variant))
                row[metric] = untoken(value) if NUMBER.fullmatch(value) else value
        elif fields[0] == "o" and len(fields) >= 3:
            key = tuple(fields[1:3])
            row = operations.setdefault(key, {"scope": key[0], "item": key[1]})
            for item in fields[3:]:
                metric, _, value = item.partition("-")
                row[metric] = untoken(value) if NUMBER.fullmatch(value) else value
    return list(results.values()), list(operations.values())
