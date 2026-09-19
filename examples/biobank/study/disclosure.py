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
- an exclusion flow's steps are a decreasing chain, each step removing a non-negative count;
- a proportion printed to four significant digits pins its numerator when only one integer rounds to it;
- a twin model's rows (the cutoff-censoring survival rows) carry no counts of their own: their proportions stand
  beside their primary model's cell counts, and a twin row published beside a suppressed or missing primary row
  is a finding.
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
COUNT_SPEC = {"type": "count"}
# The group-sum search adds up to this many relations, and refuses a digest whose search grows past the limit.
GROUP_DEPTH, GROUP_SEARCH_LIMIT = 4, 2_000_000

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
          nested_models=(), chain=None, hierarchy=CENSUS, twins=None, limit=LIMIT):
    """Findings: every released or derivable participant count in 1..limit, as dicts.

    rows       digest.parse output: KEYS (disease, model, variant, fit, stratum, horizon) plus metrics.
    registry   metric -> {"type": "count"} | {"type": "proportion", "of": count, "per": metric} |
               {"type": "score"}, or the bare word "count", "proportion" or "score" (evaluate.METRICS); a
               proportion without "of" may appear only beside all of its cell's counts. An unregistered
               numeric metric is an error (default deny). A count may
               carry "population": "train" (training rows); the default is the evaluation rows.
    axes       stratum axes (slugged), e.g. ancestry, region, division, ehr_site, sex, age_band, risk.
               A "risk" stratum is a per-variant predicted-risk bin.
    parts      model -> [(whole, [part, ...])]: counts that split a cell (parts may be unreleased).
    cumulative model -> {count: "increasing" | "decreasing"} across horizons.
    nested_models  [(inner, outer)] or [(inner, outer, counts)]: each named count (default: those in
               horizon_invariant) of the inner model's cell is a subset of the outer model's same cell, at the
               same horizon or at the outer model's horizon-free cell.
    chain      a compiled pattern whose first group orders a row's exclusion-flow steps (e.g.
               step_(\\d+)_\\w+): each matching metric is a count, and consecutive steps differ by a count.
    twins      twin model -> primary model, e.g. {"survival_cutoff": "survival"}: the twin's rows are the
               primary's cells evaluated another way, with the same persons and counts, which the digest publishes
               once, on the primary. A twin row carrying a count is an error. Its proportions are checked, and
               pin numerators, against the primary's cell. A twin row with any number is a finding
               ("twin_beside_suppressed") unless the primary row with the same keys carries one too: the digest
               gives every twin row its primary cell's decision.
    """
    parts, cumulative, twins = parts or {}, cumulative or {}, twins or {}
    registry = {metric: {"type": spec} if isinstance(spec, str) else spec for metric, spec in registry.items()}
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
    # A proportion without a declared numerator ("proportion" alone, as evaluate.METRICS writes it) times its
    # cell's counts gives a count, so it may appear only in a cell whose counts are all released.
    shown, released, vocabulary = [], defaultdict(set), defaultdict(set)
    # (disease, model, variant, fit, stratum, horizon) -> whether the row shows any number, for the twin check.
    numbered = {}
    for row in rows:
        disease, model, variant, fit = row["disease"], row["model"], row["variant"], row["fit"]
        primary = twins.get(model)
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
        steps = []
        numbered[(disease, model, variant, fit, row["stratum"], row["horizon"])] = False
        for metric, value in row.items():
            if metric in ("disease", "model", "variant", "fit", "stratum", "horizon") or isinstance(value, str):
                continue
            step = chain.fullmatch(metric) if chain is not None else None
            if step is not None:
                steps.append((int(step.group(1)), metric))
            spec = COUNT_SPEC if step is not None else registry.get(metric)
            if spec is None:
                raise ValueError(f"metric {metric!r} is not registered")
            numbered[(disease, model, variant, fit, row["stratum"], row["horizon"])] = True
            if primary is not None and spec["type"] == "count":
                raise ValueError(f"{disease}/{model}/{variant}/{fit}/{row['stratum']}/{row['horizon']} carries the "
                                 f"count {metric}: a twin model's counts are its primary's, published there alone")
            if spec["type"] == "score":
                continue
            if spec["type"] == "proportion":
                # A twin row's proportion stands beside its primary cell's counts.
                cell_model = primary or model
                if "of" in spec:
                    proportions.append((disease, cell_model, horizon, conditions, spec, value, logo, row))
                else:
                    shown.append((disease, cell_model, horizon, conditions,
                                  f"{disease}/{model}/{variant}/{fit}/{row['stratum']}/{row['horizon']}/{metric}"))
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
            if population == "evaluation":
                released[(disease, model, cell_conditions)].add(
                    (metric, None if metric in horizon_invariant else horizon))
                vocabulary[model].add(metric)
        steps.sort()
        for (_, before), (_, after) in zip(steps, steps[1:]):
            system.equate([(key(disease, model, before, horizon, conditions, "evaluation"), 1),
                           (key(disease, model, after, horizon, conditions, "evaluation"), -1),
                           (("removed", disease, model, before, after, conditions), -1)],
                          f"{disease}/{model}: {before} to {after} removes a count")

    # LOGO train complements: a group's cell plus its complement is the pooled cell.
    for disease, model, metric, horizon, conditions, population in sorted(cells, key=_order):
        for axis, value in sorted(conditions, key=repr):
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
    frontier = sorted(cells, key=_order)
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
    for disease, model, metric, horizon, conditions, population in sorted(cells, key=_order):
        for axis, value in sorted(conditions, key=repr):
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
    for disease, model, metric, horizon, conditions, population in sorted(cells, key=_order):
        horizons[(disease, model)].add(horizon)
        conditions_seen[(disease, model, population)].add(conditions)
    for (disease, model, population), seen in sorted(conditions_seen.items(), key=_order):
        hs = sorted(h for h in horizons[(disease, model)] if h is not None)
        for conditions in sorted(seen, key=_order):
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
    for inner, outer, *counted in nested_models:
        counted = set(counted[0]) if counted else set(horizon_invariant)
        for disease, model, metric, horizon, conditions, population in sorted(cells, key=_order):
            if model == inner and metric in counted:
                # The inner cell at a horizon lies inside the outer cell at that horizon, or inside the outer
                # model's only (horizon-free) cell when it has no horizons.
                outer_horizon = horizon if horizon in horizons[(disease, outer)] else None
                extra = ("nesting", disease, inner, outer, metric, horizon, conditions, population)
                system.equate([(key(disease, inner, metric, horizon, conditions, population), 1), (extra, 1),
                               (key(disease, outer, metric, outer_horizon, conditions, population), -1)],
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
    # Groups of unknowns with a determined sum. Which reduced rows hold such a group depends on the pivot order,
    # so the search runs over the original relations instead: each after substituting the determined values, and
    # every signed sum of up to GROUP_DEPTH of them chained through an unknown that cancels (a suppressed cell
    # between its own family and its parent's family, say).
    seen = set()
    reduced = []
    for terms, _ in system.equations:
        free, constant = {}, Fraction(0)
        for u, c in terms.items():
            if u in determined:
                constant -= c * determined[u]
            else:
                free[u] = c
        if free:
            reduced.append((free, constant))
    where = defaultdict(list)
    for i, (free, _) in enumerate(reduced):
        for u in free:
            where[u].append(i)

    def consider(free, constant):
        coefficients = set(free.values())
        if len(free) < 2 or len(coefficients) != 1:
            return
        total = constant / coefficients.pop()
        group = frozenset(free)
        if group in seen or total.denominator != 1 or not 1 <= total <= limit:
            return
        seen.add(group)
        findings.append({"kind": "group_sum", "what": " + ".join(sorted(_describe(labels[u]) for u in group)),
                         "value": int(total)})

    for row, const in pivots.values():
        consider(row, const)
    frontier = [((i, 1),) for i in range(len(reduced))]
    visited = {frozenset(combo) for combo in frontier}
    for depth in range(1, GROUP_DEPTH + 1):
        grown = []
        for combo in frontier:
            free, constant = defaultdict(Fraction), Fraction(0)
            for i, sign in combo:
                for u, c in reduced[i][0].items():
                    free[u] += sign * c
                constant += sign * reduced[i][1]
            free = {u: c for u, c in free.items() if c}
            consider(free, constant)
            if depth == GROUP_DEPTH:
                continue
            members = {i for i, _ in combo}
            for u, c in free.items():
                for j in where[u]:
                    if j in members:
                        continue
                    sign = -1 if reduced[j][0][u] == c else 1 if reduced[j][0][u] == -c else 0
                    if not sign:
                        continue
                    extended = combo + ((j, sign),)
                    # A combination and its negation are the same relation.
                    canonical_key = min(frozenset(extended), frozenset((k, -s) for k, s in extended), key=sorted)
                    if canonical_key not in visited:
                        visited.add(canonical_key)
                        grown.append(extended)
            if len(visited) > GROUP_SEARCH_LIMIT:
                raise ValueError("the group-sum search outgrew its bound; audit this digest in smaller parts")
        frontier = grown
    for disease, model, horizon, conditions, label in shown:
        needed = {(m, None if m in horizon_invariant else horizon) for m in vocabulary[model]}
        missing = sorted(m for m, _ in needed - released[(disease, model, conditions)])
        if missing:
            findings.append({"kind": "proportion_beside_withheld", "value": 0,
                             "what": f"{label} beside withheld {', '.join(missing)}"})
    for (disease, model, variant, fit, stratum, horizon), shows in sorted(numbered.items(), key=_order):
        if model not in twins or not shows:
            continue
        label = f"{disease}/{model}/{variant}/{fit}/{stratum}/{horizon}"
        twin = numbered.get((disease, twins[model], variant, fit, stratum, horizon))
        if not twin:
            findings.append({"kind": "twin_beside_suppressed", "value": 0,
                             "what": f"{label} published beside {'no' if twin is None else 'a suppressed'} "
                                     f"{twins[model]} row"})
    return findings


def _order(item):
    """A canonical sort key: sets by their sorted members, so no iteration order depends on string hashing."""
    if isinstance(item, (set, frozenset)):
        return "{" + ",".join(sorted(_order(x) for x in item)) + "}"
    if isinstance(item, tuple):
        return "(" + ",".join(_order(x) for x in item) + ")"
    return repr(item)


def _describe(name):
    kind, *rest = name
    return f"{kind}:" + "/".join(str(sorted(x, key=str)) if isinstance(x, frozenset) else str(x) for x in rest)
