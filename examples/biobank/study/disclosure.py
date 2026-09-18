"""Differencing audit of the digest: no participant count of 1 to 20 may be released or derivable.

The All of Us Data and Statistics Dissemination Policy: "No participant count of 1 to 20 can be published or
distributed directly (a count of 0 is permitted)", and "No data or statistics can be reported that allow a
participant count of 1 to 20 to be derived from other reported cells or information". Suppressing each small
cell is not enough: an overall total minus the released categories of an axis gives the suppressed one back,
and so do nested divisions, cumulative event counts at successive horizons, a proportion times its released
denominator, and a leave-one-group-out cell against its pooled counterparts.

`audit` reads the rows that `digest.parse` returns (every label slugged, numbers as the tokens print them) and
models every count as an unknown with the linear relations that hold between the released cells:
- an axis partitions a cell: its categories plus a residual for any category not listed sum to the cell;
- a division lies inside its Census region;
- a count splits into declared parts (n = cases + controls);
- a cumulative count grows (or shrinks) across horizons by a non-negative increment;
- a count that does not depend on the horizon is one unknown for all horizons;
- a leave-one-group-out fit's cells are the pooled cells crossed with its held-out group;
- a nested model's persons are a subset of the enclosing model's in the same cell;
- a proportion printed to four significant digits pins its numerator when only one integer rounds to it.
Exact rational Gauss-Jordan elimination then finds every count, residual and increment the released numbers
determine, and every group of unknown counts whose sum they determine. Each one in 1..LIMIT is a finding. A
contradiction between released numbers and the relations is an error, since the relations model the digest
wrongly or the digest is inconsistent; either way nothing is safe to publish.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math

LIMIT = 20

# Census divisions inside their regions, as slugged labels. "unknown" maps to itself.
CENSUS = {"division": ("region", {
    "new_england": "northeast", "middle_atlantic": "northeast",
    "east_north_central": "midwest", "west_north_central": "midwest",
    "south_atlantic": "south", "east_south_central": "south", "west_south_central": "south",
    "mountain": "west", "pacific": "west", "unknown": "unknown"})}


def _split(label, axes, what):
    """(axis, value) of a slugged 'axis_value' label, matching the longest known axis."""
    for axis in sorted(axes, key=len, reverse=True):
        if label.startswith(axis + "_") and len(label) > len(axis) + 1:
            return axis, label[len(axis) + 1:]
    raise ValueError(f"{what} {label!r} names no known axis")


def _horizon(label):
    if label == "all":
        return None
    if not label.startswith("h"):
        raise ValueError(f"bad horizon label {label!r}")
    return float(label[1:].replace("m", "-").replace("p", "+"))


def _shown(value):
    """A number as a token prints it: four significant digits."""
    return float(f"{value:.4g}")


class _System:
    def __init__(self):
        self.index, self.names, self.known, self.equations = {}, [], {}, []

    def var(self, key):
        if key not in self.index:
            self.index[key] = len(self.names)
            self.names.append(key)
        return self.index[key]

    def fix(self, key, value, source):
        v = self.var(key)
        value = Fraction(value)
        if v in self.known and self.known[v][0] != value:
            raise ValueError(f"inconsistent released values for {key}: {self.known[v][0]} vs {value} ({source})")
        self.known.setdefault(v, (value, source))

    def equate(self, terms, description):
        """sum(coefficient * variable) == 0 over (key, coefficient) terms."""
        row = defaultdict(Fraction)
        for key, coefficient in terms:
            row[self.var(key)] += coefficient
        self.equations.append(({v: c for v, c in row.items() if c}, description))

    def solve(self):
        """Values of every determined variable, and the reduced rows over the undetermined ones."""
        pivots = {}
        for v, (value, _) in self.known.items():
            pivots[v] = ({v: Fraction(1)}, value)
        occurs = defaultdict(set)
        for v in pivots:
            occurs[v].add(v)
        for terms, description in self.equations:
            row, constant = dict(terms), Fraction(0)
            for v in [v for v in row if v in pivots]:
                c = row.pop(v)
                prow, pconst = pivots[v]
                constant -= c * pconst
                for u, d in prow.items():
                    if u != v:
                        row[u] = row.get(u, 0) - c * d
                        if not row[u]:
                            del row[u]
            if not row:
                if constant != 0:
                    raise ValueError(f"released numbers contradict the relation: {description}")
                continue
            p = min(row, key=lambda u: len(occurs[u]))
            scale = row[p]
            row = {u: c / scale for u, c in row.items()}
            constant /= scale
            for other in list(occurs[p]):
                if other == p or other not in pivots:
                    continue
                orow, oconst = pivots[other]
                c = orow.pop(p, 0)
                if not c:
                    continue
                oconst -= c * constant
                for u, d in row.items():
                    if u != p:
                        orow[u] = orow.get(u, 0) - c * d
                        if orow[u]:
                            occurs[u].add(other)
                        else:
                            del orow[u]
                            occurs[u].discard(other)
                pivots[other] = (orow, oconst)
            occurs[p] = {p}
            for u in row:
                occurs[u].add(p)
            pivots[p] = (row, constant)
        determined = {v: const for v, (row, const) in pivots.items() if len(row) == 1}
        return determined, pivots


def audit(rows, registry, *, axes, parts=None, cumulative=None, horizon_invariant=("n",),
          nested_models=(), hierarchy=CENSUS, limit=LIMIT):
    """Findings: every released or derivable participant count in 1..limit, as dicts.

    rows       digest.parse output: KEYS (disease, model, variant, fit, stratum, horizon) plus metrics.
    registry   metric -> {"type": "count"} | {"type": "proportion", "of": count, "per": metric} |
               {"type": "score"}; an unregistered numeric metric is an error (default deny). A count may
               carry "population": "train" (training rows); the default is the evaluation rows.
    axes       stratum axes (slugged), e.g. ancestry, region, division, ehr_site, sex, age_band, risk.
               A "risk" stratum is a per-variant predicted-risk bin.
    parts      model -> [(whole, [part, ...])]: counts that split a cell (parts may be unreleased).
    cumulative model -> {count: "increasing" | "decreasing"} across horizons.
    nested_models  [(inner, outer)]: the inner model's persons are a subset of the outer model's, per cell,
               for the counts in horizon_invariant.
    """
    parts, cumulative = parts or {}, cumulative or {}
    system, findings = _System(), []
    axes = set(axes)
    parent_axis = {child: parent for child, (parent, _) in hierarchy.items()}

    def canonical(conditions):
        out = dict(conditions)
        for child, (parent, mapping) in hierarchy.items():
            if child in out:
                if out[child] not in mapping:
                    raise ValueError(f"{child} {out[child]!r} has no {parent}")
                if out.get(parent, mapping[out[child]]) != mapping[out[child]]:
                    return None  # an empty cell: a division outside the stated region
                out[parent] = mapping[out[child]]
        return frozenset(out.items())

    def key(disease, model, metric, horizon, conditions, population):
        if metric in horizon_invariant:
            horizon = None
        return ("count", disease, model, metric, horizon, conditions, population)

    cells = set()  # (disease, model, metric, horizon, conditions, population) that exist
    proportions = []
    for row in rows:
        disease, model, variant, fit = row["disease"], row["model"], row["variant"], row["fit"]
        horizon = _horizon(row["horizon"])
        conditions = {}
        logo = None
        if fit != "pooled":
            if not fit.startswith("logo_"):
                raise ValueError(f"unknown fit {fit!r}")
            logo = _split(fit[len("logo_"):], axes, "fit")
            conditions[logo[0]] = logo[1]
        if row["stratum"] != "overall":
            axis, value = _split(row["stratum"], axes, "stratum")
            if axis == "risk":
                axis = ("risk", variant, fit)
            if axis in conditions and conditions[axis] != value:
                continue  # another group's stratum inside a held-out group: empty by construction
            conditions[axis] = value
        conditions = canonical(conditions)
        if conditions is None:
            continue
        for metric, value in row.items():
            if metric in ("disease", "model", "variant", "fit", "stratum", "horizon") or isinstance(value, str):
                continue
            spec = registry.get(metric)
            if spec is None:
                raise ValueError(f"metric {metric!r} is not registered")
            if spec["type"] == "score":
                continue
            if spec["type"] == "proportion":
                proportions.append((disease, model, horizon, conditions, spec, value, logo, row))
                continue
            if spec["type"] != "count":
                raise ValueError(f"metric {metric!r} has unknown type {spec['type']!r}")
            population = spec.get("population", "evaluation")
            if population == "train" and logo is not None:
                # A LOGO fit trains on everyone outside its held-out group.
                conditions_train = dict(conditions)
                conditions_train[logo[0]] = "not_" + logo[1]
                cell_conditions = frozenset(conditions_train.items())
            else:
                cell_conditions = conditions
            label = f"{disease}/{model}/{variant}/{fit}/{row['stratum']}/{row['horizon']}/{metric}"
            if value != int(value) or value < 0:
                raise ValueError(f"count {label} = {value} is not a non-negative integer")
            if 1 <= value <= limit:
                findings.append({"kind": "released", "what": label, "value": int(value)})
            system.fix(key(disease, model, metric, horizon, cell_conditions, population), int(value), label)
            cells.add((disease, model, metric, horizon if metric not in horizon_invariant else None,
                       cell_conditions, population))

    # LOGO train complements: a group's cell plus its complement is the pooled cell.
    for disease, model, metric, horizon, conditions, population in list(cells):
        for axis, value in conditions:
            if isinstance(value, str) and value.startswith("not_"):
                rest = conditions - {(axis, value)}
                group = rest | {(axis, value[len("not_"):])}
                system.equate([(key(disease, model, metric, horizon, group, population), 1),
                               (key(disease, model, metric, horizon, conditions, population), 1),
                               (key(disease, model, metric, horizon, rest, population), -1)],
                              f"{disease}/{model}/{metric}: {axis} {value[4:]} plus the rest is the pooled cell")
                cells.add((disease, model, metric, horizon, rest, population))

    # Every cell's marginal cells exist, released or not, so each axis family lists every category it has:
    # a division implies its region's cell, a LOGO cross cell both of its marginals.
    frontier = list(cells)
    while frontier:
        disease, model, metric, horizon, conditions, population = frontier.pop()
        for axis, value in conditions:
            if isinstance(value, str) and value.startswith("not_"):
                continue
            if any(child in dict(conditions) for child, (p, _) in hierarchy.items() if p == axis):
                continue  # the region stays while its division does
            smaller = (disease, model, metric, horizon, conditions - {(axis, value)}, population)
            if smaller not in cells:
                cells.add(smaller)
                frontier.append(smaller)

    # Partitions: every cell's categories along one axis, within the rest of its conditions.
    families = defaultdict(set)
    for disease, model, metric, horizon, conditions, population in cells:
        for axis, value in conditions:
            if isinstance(value, str) and value.startswith("not_"):
                continue
            if axis in parent_axis:
                parent = frozenset(c for c in conditions if c[0] != axis)  # keeps the region
            elif any(child in dict(conditions) for child, (p, _) in hierarchy.items() if p == axis):
                continue  # region is implied by the division; its partition is the region's own
            else:
                parent = conditions - {(axis, value)}
            families[(disease, model, metric, horizon, population, parent, axis)].add(value)
    for (disease, model, metric, horizon, population, parent, axis), values in families.items():
        terms = [(key(disease, model, metric, horizon, canonical(set(parent) | {(axis, v)}), population), 1)
                 for v in sorted(values)]
        residual = ("residual", disease, model, metric, horizon, parent, axis, population)
        system.equate(terms + [(residual, 1), (key(disease, model, metric, horizon, parent, population), -1)],
                      f"{disease}/{model}/{metric}/{horizon}: {axis} categories within {sorted(parent, key=str)}")

    # Declared parts, cumulative growth across horizons, nesting between models.
    horizons = defaultdict(set)
    conditions_seen = defaultdict(set)
    for disease, model, metric, horizon, conditions, population in cells:
        horizons[(disease, model)].add(horizon)
        conditions_seen[(disease, model, population)].add(conditions)
    for (disease, model, population), seen in conditions_seen.items():
        hs = sorted(h for h in horizons[(disease, model)] if h is not None)
        for conditions in seen:
            for whole, pieces in parts.get(model, []):
                for h in hs or [None]:
                    system.equate([(key(disease, model, whole, h, conditions, population), -1)]
                                  + [(key(disease, model, p, h, conditions, population), 1) for p in pieces],
                                  f"{disease}/{model}: {whole} = {' + '.join(pieces)}")
            for metric, direction in cumulative.get(model, {}).items():
                for a, b in zip(hs, hs[1:]):
                    small, large = (a, b) if direction == "increasing" else (b, a)
                    step = ("increment", disease, model, metric, a, b, conditions, population)
                    system.equate([(key(disease, model, metric, small, conditions, population), 1), (step, 1),
                                   (key(disease, model, metric, large, conditions, population), -1)],
                                  f"{disease}/{model}/{metric} between horizons {a:g} and {b:g}")
    for inner, outer in nested_models:
        for disease, model, metric, horizon, conditions, population in list(cells):
            if model == inner and metric in horizon_invariant:
                extra = ("nesting", disease, inner, outer, metric, conditions, population)
                system.equate([(key(disease, inner, metric, None, conditions, population), 1), (extra, 1),
                               (key(disease, outer, metric, None, conditions, population), -1)],
                              f"{disease}: {inner} {metric} inside {outer}")

    # A proportion pins its numerator when exactly one integer prints as it; iterate as denominators resolve.
    added = True
    while added:
        determined, pivots = system.solve()
        added = False
        for disease, model, horizon, conditions, spec, value, logo, row in proportions:
            per_spec = registry.get(spec["per"], {"type": "count"})
            if per_spec["type"] == "count":
                per_key = key(disease, model, spec["per"], horizon, conditions,
                              per_spec.get("population", "evaluation"))
                if per_key not in system.index or system.index[per_key] not in determined:
                    continue
                per = determined[system.index[per_key]]
            else:
                if spec["per"] not in row:
                    continue
                per = Fraction(row[spec["per"]]).limit_denominator(10**9)
            if per <= 0:
                continue
            centre = float(value) * float(per)
            low, high = max(0, math.floor(centre * (1 - 1e-3)) - 1), math.ceil(centre * (1 + 1e-3)) + 1
            candidates = [k for k in range(low, high + 1) if _shown(k / float(per)) == float(value)]
            of_key = key(disease, model, spec["of"], horizon, conditions, "evaluation")
            label = f"{disease}/{model}/{row['variant']}/{row['fit']}/{row['stratum']}/{row['horizon']}/{spec['of']}"
            if len(candidates) == 1 and (of_key not in system.index or system.index[of_key] not in system.known):
                system.fix(of_key, candidates[0], f"{label} from its proportion")
                added = True

    labels = {i: name for i, name in enumerate(system.names)}
    for v, value in determined.items():
        if v in system.known:
            continue  # released numbers were checked as released
        if value < 0:
            raise ValueError(f"released numbers make a count negative: {_describe(labels[v])} = {value}")
        if value.denominator == 1 and 1 <= value <= limit:
            findings.append({"kind": "derived", "what": _describe(labels[v]), "value": int(value)})
    for v, (source_value, source) in system.known.items():
        if source.endswith("from its proportion") and 1 <= source_value <= limit:
            findings.append({"kind": "proportion", "what": source, "value": int(source_value)})
    # Groups of unknowns with a determined sum: a reduced row, or an original relation after substitution.
    seen = set()
    rows_to_check = [(row, const) for row, const in pivots.values() if len(row) > 1]
    for terms, _ in system.equations:
        free, constant = {}, Fraction(0)
        for u, c in terms.items():
            if u in determined:
                constant -= c * determined[u]
            else:
                free[u] = c
        rows_to_check.append((free, constant))
    for free, constant in rows_to_check:
        coefficients = set(free.values())
        if len(free) < 2 or len(coefficients) != 1:
            continue
        c = coefficients.pop()
        total = constant / c
        group = frozenset(free)
        if group in seen or total.denominator != 1 or not 1 <= total <= limit:
            continue
        seen.add(group)
        findings.append({"kind": "group_sum", "what": " + ".join(sorted(_describe(labels[u]) for u in group)),
                         "value": int(total)})
    return findings


def _describe(name):
    kind, *rest = name
    return f"{kind}:" + "/".join(str(sorted(x, key=str)) if isinstance(x, frozenset) else str(x) for x in rest)
