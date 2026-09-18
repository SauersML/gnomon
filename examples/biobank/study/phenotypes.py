"""Per-disease analysis frames for the binary and survival models (SPEC sections 2-3).

Everything here reads the SCHEMA.md tables through a `cohort` source and is
vectorized over people: dates are day numbers (float, NaN = none), and every
join is an index lookup on person_id. The frame columns are documented in
SCHEMA.md under "Analysis frames".

Phenotype (uniform, prespecified): a confirmed case has qualifying records on
at least two distinct dates, and its event date is the second of them. People
with a single record are kept as non-cases (binary) or at risk (survival).
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .cohort import EPOCH, days

YEAR = 365.25
SEX_CODES = {45880669: 1, 8507: 1, 45878463: 0, 8532: 0}  # PPI SexAtBirth_* and OMOP gender
AGE_BANDS = ("18-39", "40-59", "60-74", "75+")
AGE_EDGES = (40.0, 60.0, 75.0)
SES_QUARTILES = ("Q1", "Q2", "Q3", "Q4")
LOOKBACK_TERTILES = ("T1", "T2", "T3")
UNKNOWN = "unknown"
REGIONS = ("Northeast", "Midwest", "South", "West")
DIVISIONS = {
    "New England": ("CT", "ME", "MA", "NH", "RI", "VT"),
    "Middle Atlantic": ("NJ", "NY", "PA"),
    "East North Central": ("IL", "IN", "MI", "OH", "WI"),
    "West North Central": ("IA", "KS", "MN", "MO", "NE", "ND", "SD"),
    "South Atlantic": ("DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"),
    "East South Central": ("AL", "KY", "MS", "TN"),
    "West South Central": ("AR", "LA", "OK", "TX"),
    "Mountain": ("AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"),
    "Pacific": ("AK", "CA", "HI", "OR", "WA"),
}
DIVISION_REGION = {
    "New England": "Northeast", "Middle Atlantic": "Northeast",
    "East North Central": "Midwest", "West North Central": "Midwest",
    "South Atlantic": "South", "East South Central": "South", "West South Central": "South",
    "Mountain": "West", "Pacific": "West",
}
STATE_DIVISION = {state: division for division, states in DIVISIONS.items() for state in states}
SHARED_COLUMNS = ("person_id", "pgs", "sex")
STRATA = ("ancestry", "region", "division", "ehr_site", "age_band", "ses_quartile", "lookback_tertile")


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
# Split seeds whose outer test rows earlier AoU runs have already seen (audit S8).
SPENT_SEEDS = (20260910, 20260914)


@dataclass(frozen=True)
class CohortConfig:
    seed: int
    num_pcs: int = 6
    lookback_days: int = 365
    landmark_days: int = 180
    min_age: float = 18.0
    test_fraction: float = 0.2
    dev_folds: int = 5
    max_prune_unmatched: int = 0
    # Exclusion-rule exits after the landmark become a competing event (3) once
    # they exceed this fraction of a disease's survival cohort; below it they censor.
    exclusion_competing_fraction: float = 0.01
    # The prespecified horizon rule (`choose_horizons`): every candidate that at
    # least this fraction of the survival-eligible cohort can reach administratively.
    horizon_candidates: tuple = (1, 2, 3, 4, 5)
    horizon_min_reach: float = 0.5

    def __post_init__(self):
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("the split seed must be a non-negative integer")
        if self.seed in SPENT_SEEDS:
            raise ValueError(f"split seed {self.seed} has already seen AoU outer-test rows; record a new one")
        if self.max_prune_unmatched < 0:
            raise ValueError("max_prune_unmatched is a non-negative count")
        if not 1 <= self.num_pcs <= 64 or self.dev_folds < 2 or not 0 < self.test_fraction < 1:
            raise ValueError("unsupported PC count, fold count or test fraction")
        if self.lookback_days < 0 or self.landmark_days < 0:
            raise ValueError("lookback and landmark are non-negative day counts")
        if not 0 <= self.exclusion_competing_fraction <= 1 or not 0 < self.horizon_min_reach <= 1:
            raise ValueError("exclusion_competing_fraction and horizon_min_reach are fractions")
        candidates = tuple(self.horizon_candidates)
        if not candidates or list(candidates) != sorted(set(candidates)) or candidates[0] <= 0:
            raise ValueError("horizon_candidates are distinct increasing positive years")
        object.__setattr__(self, "horizon_candidates", candidates)

    @classmethod
    def from_json(cls, block):
        unknown = set(block) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown cohort configuration keys {sorted(unknown)}")
        return cls(**block)


# diseases.json keys: the ones that define the phenotype, and free text. Any
# other key is refused, so a new phenotype field cannot be skipped silently.
DISEASE_KEYS = {"slug", "snomed_code", "pgs", "sex", "exclusions", "excluded_branches", "omop_concept_id"}
DISEASE_TEXT = {"audit", "cached", "concept_name", "expected_cases", "genome_build", "interpretation_note",
                "multi_ancestry", "multi_ancestry_note", "pgs_name", "pgs_trait", "root_note",
                "sensitivity_analyses", "sex_reason", "variants_number"}
CODE_KEYS = {"snomed_code", "reason", "omop_concept_id"}
CODE_TEXT = {"bias_note", "browser_any_record", "concept_name"}


def _codes(slug, entries, what):
    """((snomed_code, reason, declared OMOP concept_id), ...) for exclusions or excluded branches."""
    out = []
    for entry in entries or ():
        unknown = set(entry) - CODE_KEYS - CODE_TEXT
        if unknown:
            raise ValueError(f"{slug}: {what} carry keys this pipeline does not implement: {sorted(unknown)}")
        code, reason = str(entry["snomed_code"]), entry.get("reason")
        if not re.fullmatch(r"\d{6,18}", code) or not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{slug}: each of the {what} needs a SNOMED code and a written reason")
        out.append((code, reason, entry.get("omop_concept_id")))
    return tuple(out)


@dataclass(frozen=True)
class Disease:
    slug: str
    snomed_code: str  # the root concept's SNOMED concept_code
    pgs: str
    sex: str | None = None
    exclusions: tuple = ()  # (snomed_code, reason[, declared concept_id]): a case rule that removes people
    excluded_branches: tuple = ()  # (snomed_code, reason[, declared concept_id]): records that do not qualify
    omop_concept_id: int | None = None

    @classmethod
    def from_json(cls, slug, entry):
        """A diseases.json entry. Unknown keys are refused; the listed free-text keys are ignored."""
        unknown = set(entry) - DISEASE_KEYS - DISEASE_TEXT
        if unknown:
            raise ValueError(f"{slug}: diseases.json keys this pipeline does not implement: {sorted(unknown)}")
        code, pgs, sex = str(entry["snomed_code"]), entry["pgs"], entry.get("sex")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", slug):
            raise ValueError(f"disease slug {slug!r} is not a lowercase identifier")
        if not re.fullmatch(r"\d{6,18}", code) or not re.fullmatch(r"PGS\d{6}", pgs):
            raise ValueError(f"{slug}: snomed_code must be a SNOMED code and pgs a PGS Catalog ID")
        if sex not in (None, "female", "male"):
            raise ValueError(f"{slug}: sex restriction must be female, male or null")
        return cls(slug, code, pgs, sex, _codes(slug, entry.get("exclusions"), "exclusions"),
                   _codes(slug, entry.get("excluded_branches"), "excluded branches"),
                   entry.get("omop_concept_id"))

    @property
    def snomed_codes(self):
        """The roots whose qualifying records the frames read: the disease, then its exclusions."""
        return (self.snomed_code, *(code for code, *_ in self.exclusions))

    @property
    def branch_codes(self):
        return tuple(sorted(code for code, *_ in self.excluded_branches))


def phenotype_codes(diseases):
    """`cohort.AouSource` keyword arguments: every root and excluded branch, with declared concept IDs.

    {"snomed_codes": {root: concept_id | None}, "excluded_branches": {root: {branch: concept_id | None}}}
    """
    roots, branches = {}, {}

    def declare(table, code, concept):
        if table.get(code) is not None and concept is not None and table[code] != concept:
            raise ValueError(f"{code} is declared as two OMOP concepts")
        table[code] = table.get(code) if concept is None else concept

    for disease in diseases:
        declare(roots, disease.snomed_code, disease.omop_concept_id)
        for code, _, *concept in disease.exclusions:
            declare(roots, code, (concept or [None])[0])
        if disease.excluded_branches:
            mine = branches.setdefault(disease.snomed_code, {})
            if mine and set(mine) != set(disease.branch_codes):
                raise ValueError(f"{disease.snomed_code} is given two different sets of excluded branches")
            for code, _, *concept in disease.excluded_branches:
                declare(mine, code, (concept or [None])[0])
    for disease in diseases:
        if not disease.excluded_branches and disease.snomed_code in branches:
            raise ValueError(f"{disease.snomed_code} has excluded branches in one disease but not another")
    return {"snomed_codes": roots, "excluded_branches": branches}


def load_diseases(block):
    """Diseases from a {slug: entry} mapping, a list of entries with "slug", or {"diseases": ...}."""
    if isinstance(block, dict) and "diseases" in block:
        block = block["diseases"]
    items = block.items() if isinstance(block, dict) else ((entry["slug"], entry) for entry in block)
    diseases = [Disease.from_json(slug, entry) for slug, entry in items]
    if len({d.slug for d in diseases}) != len(diseases):
        raise ValueError("disease slugs repeat")
    return diseases


# --------------------------------------------------------------------------- #
# split: a seeded hash of person_id alone
# --------------------------------------------------------------------------- #
_MASK64 = (1 << 64) - 1


def _splitmix64(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _fnv1a64(text):
    h = 0xCBF29CE484222325
    for byte in text.encode():
        h = ((h ^ byte) * 0x100000001B3) & _MASK64
    return h


def unit_hash(person_id, seed, purpose):
    """Uniform [0, 1) per person: splitmix64(person_id XOR splitmix64(seed XOR fnv1a64(purpose)))."""
    key = _splitmix64(np.array([(seed & _MASK64) ^ _fnv1a64(purpose)], dtype=np.uint64))[0]
    x = _splitmix64(np.asarray(person_id, dtype=np.int64).view(np.uint64) ^ key)
    return (x >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


def split(person_id, config):
    """(test, fold): the locked outer test and development folds; fold is -1 on test rows."""
    test = unit_hash(person_id, config.seed, "test") < config.test_fraction
    fold = np.minimum((unit_hash(person_id, config.seed, "fold") * config.dev_folds).astype(np.int64),
                      config.dev_folds - 1).astype(np.int8)
    fold[test] = -1
    return test, fold


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _year(day):
    """Calendar year of whole day numbers."""
    calendar = np.asarray(day).astype(np.int64).astype("datetime64[D]").astype("datetime64[Y]")
    return (calendar.astype(np.int64) + 1970).astype(np.int16)


def _floats(column):
    """Arrow float column -> float64 numpy, NaN where null."""
    return column.to_pandas().to_numpy(dtype=np.float64, na_value=np.nan)


def _positions(index, ids):
    """Row of each id in `index` (a unique pd.Index), -1 when absent."""
    return index.get_indexer(ids)


def _take(values, positions, fill):
    out = np.full(len(positions), fill, dtype=np.result_type(values.dtype, np.min_scalar_type(fill)))
    found = positions >= 0
    out[found] = values[positions[found]]
    return out


def _categorical(labels, categories):
    return pd.Categorical(labels, categories=list(categories))


def _quantile_bins(values, probabilities, labels):
    """Labels by cut points over the known values; a value equal to a cut goes lower."""
    known = ~np.isnan(values)
    cuts = np.quantile(values[known], probabilities) if known.any() else np.array([])
    out = np.full(len(values), UNKNOWN, dtype=object)
    if known.any():
        out[known] = np.asarray(labels, dtype=object)[np.searchsorted(cuts, values[known], side="left")]
    return out, [float(c) for c in cuts]


class Flow:
    """Ordered exclusion steps with the people remaining after each."""

    def __init__(self, start_label, n):
        self.steps = [{"step": start_label, "n": int(n), "removed": 0}]

    def step(self, label, keep):
        n = int(keep.sum())
        self.steps.append({"step": label, "n": n, "removed": self.steps[-1]["n"] - n})


# --------------------------------------------------------------------------- #
# base cohort
# --------------------------------------------------------------------------- #
@dataclass
class Base:
    frame: pd.DataFrame  # one row per base participant; "_"-prefixed columns are internal day numbers
    flow: list
    cuts: dict
    cdr_cutoff_day: float
    config: CohortConfig
    index: pd.Index = field(repr=False, default=None)


def base_cohort(source, config):
    """The cohort shared by every disease and both models (SCHEMA.md "Base cohort")."""
    manifest = source.manifest
    if config.num_pcs > manifest["num_pcs"]:
        raise ValueError(f"config asks for {config.num_pcs} PCs; the tables carry {manifest['num_pcs']}")
    # Audit M15: the prune must act on the ancestry universe. Prune IDs outside
    # the ancestry file mean the two universes differ, so relatives of them
    # could enter unpruned.
    unmatched = int(manifest.get("prune_unmatched", 0))
    if unmatched > config.max_prune_unmatched:
        raise ValueError(f"{unmatched} relatedness-prune IDs are outside the ancestry universe "
                         f"(allowed: {config.max_prune_unmatched})")
    person = source.table("person")
    person_id = person.column("person_id").to_numpy()
    flow = Flow("cdr_persons", len(person_id))

    ancestry = source.table("ancestry")
    at = _positions(pd.Index(ancestry.column("person_id").to_numpy()), person_id)
    keep = at >= 0
    flow.step("in_ancestry", keep)
    related = ancestry.column("related_excluded").to_numpy()
    keep &= ~_take(related, at, True)
    flow.step("not_related_excluded", keep)
    pcs = source.table("pcs")
    pc_at = _positions(pd.Index(pcs.column("person_id").to_numpy()), person_id)
    keep &= pc_at >= 0
    flow.step("has_pcs", keep)
    scored = _positions(pd.Index(source.table("scores").column("person_id").to_numpy()), person_id)
    keep &= scored >= 0
    flow.step("scored", keep)
    sex = pd.Series(person.column("sex_at_birth_concept_id").to_numpy()).map(SEX_CODES).to_numpy()
    keep &= ~np.isnan(sex)
    flow.step("sex_male_or_female", keep)
    birth, baseline = days(person.column("birth_date")), days(person.column("baseline_date"))
    keep &= ~np.isnan(baseline)
    flow.step("has_baseline", keep)
    age = (baseline - birth) / YEAR
    keep &= age >= config.min_age
    flow.step("adult_at_baseline", keep)
    obs_start, obs_end = days(person.column("obs_start")), days(person.column("obs_end"))
    keep &= ~np.isnan(obs_start)
    flow.step("covering_observation_period", keep)
    lookback = baseline - obs_start
    keep &= lookback >= config.lookback_days
    flow.step("lookback", keep)

    rows = np.flatnonzero(keep)
    base_id = person_id[rows]
    cutoff = float((datetime.date.fromisoformat(manifest["cdr_cutoff"]) - EPOCH).days)

    frame = pd.DataFrame({"person_id": base_id, "sex": sex[rows].astype(np.int8)})
    for i in range(1, config.num_pcs + 1):
        frame[f"PC{i}"] = pcs.column(f"PC{i}").to_numpy()[pc_at[rows]]
    frame["age_baseline"] = age[rows]
    frame["baseline_year"] = _year(baseline[rows])
    frame["lookback_days"] = lookback[rows].astype(np.int32)
    # Window covariates known at baseline (audit N3): the pre-baseline record
    # and the administrative window to the CDR cutoff.
    frame["lookback_years"] = lookback[rows] / YEAR
    frame["admin_years"] = (cutoff - baseline[rows]) / YEAR

    labels = ancestry.column("ancestry_pred").to_numpy()[at[rows]]
    frame["ancestry"] = _categorical(labels, sorted(set(labels)))
    state = person.column("state").to_pandas().to_numpy(dtype=object)[rows]
    division = pd.Series(state, dtype=object).map(STATE_DIVISION).fillna(UNKNOWN).to_numpy(dtype=object)
    frame["division"] = _categorical(division, [*DIVISIONS, UNKNOWN])
    region = pd.Series(division, dtype=object).map(DIVISION_REGION).fillna(UNKNOWN)
    frame["region"] = _categorical(region, [*REGIONS, UNKNOWN])
    site = person.column("ehr_site").to_pandas().fillna(UNKNOWN).to_numpy(dtype=object)[rows]
    frame["ehr_site"] = _categorical(site, sorted(set(site) - {UNKNOWN}) + [UNKNOWN])
    frame["age_band"] = _categorical(np.asarray(AGE_BANDS, dtype=object)[np.searchsorted(
        AGE_EDGES, age[rows], side="right")], AGE_BANDS)
    deprivation = _floats(person.column("deprivation_index"))[rows]
    quartile, ses_cuts = _quantile_bins(deprivation, (0.25, 0.5, 0.75), SES_QUARTILES)
    frame["ses_quartile"] = _categorical(quartile, [*SES_QUARTILES, UNKNOWN])
    tertile, lookback_cuts = _quantile_bins(lookback[rows], (1 / 3, 2 / 3), LOOKBACK_TERTILES)
    frame["lookback_tertile"] = _categorical(tertile, LOOKBACK_TERTILES)
    frame["test"], frame["fold"] = split(base_id, config)
    frame["_birth"], frame["_baseline"] = birth[rows], baseline[rows]
    frame["_obs_end"], frame["_death"] = obs_end[rows], days(person.column("death_date"))[rows]
    frame["_ehr_end"] = days(person.column("ehr_end"))[rows]
    return Base(frame, flow.steps, {"ses_quartile": ses_cuts, "lookback_tertile": lookback_cuts},
                cutoff, config, pd.Index(base_id))


# --------------------------------------------------------------------------- #
# per-disease rows, then the two frames
# --------------------------------------------------------------------------- #
@dataclass
class DiseaseRows:
    disease: Disease
    # Base columns plus pgs and n_dates, and internal day numbers: _first and
    # _second (the root's qualifying dates) and, per exclusion code, the date
    # its case rule is met (_exclusion_<code>, NaN when never).
    frame: pd.DataFrame
    flow: list


def disease_rows(base, source, disease):
    """Base participants with the disease's score, inside its sex restriction (SCHEMA.md "Per disease").

    Exclusion roots are applied by each frame, because the two models apply
    them over different windows (audit C3)."""
    frame = base.frame
    flow = Flow("base", len(frame))
    scores = source.scores(disease.pgs)
    at = _positions(pd.Index(scores.column("person_id").to_numpy()), frame.person_id.to_numpy())
    pgs = _take(_floats(scores.column(disease.pgs)), at, np.nan)
    keep = ~np.isnan(pgs)
    flow.step("pgs_present", keep)
    if disease.sex is not None:
        keep &= frame.sex.to_numpy() == (1 if disease.sex == "male" else 0)
        flow.step(f"sex_{disease.sex}", keep)

    conditions = source.condition(disease.snomed_codes)
    code_of = conditions.column("snomed_code").to_numpy()
    ids = conditions.column("person_id").to_numpy()
    first_all, second_all = days(conditions.column("first_date")), days(conditions.column("second_date"))
    count_all = conditions.column("n_dates").to_numpy()

    def aligned(code):
        mine = code_of == code
        where = _positions(base.index, ids[mine])
        inside = where >= 0
        first = np.full(len(frame), np.nan)
        second = np.full(len(frame), np.nan)
        count = np.zeros(len(frame), dtype=np.int32)
        first[where[inside]] = first_all[mine][inside]
        second[where[inside]] = second_all[mine][inside]
        count[where[inside]] = count_all[mine][inside]
        return first, second, count

    first, second, count = aligned(disease.snomed_code)
    rows = frame.loc[keep].copy()
    rows.insert(1, "pgs", pgs[keep])
    rows["n_dates"] = count[keep]
    rows["_first"], rows["_second"] = first[keep], second[keep]
    for code, *_ in disease.exclusions:
        rows[f"_exclusion_{code}"] = aligned(code)[1][keep]  # the second date: the case rule is met
    return DiseaseRows(disease, rows.reset_index(drop=True), flow.steps)


def _shared(num_pcs):
    return [*SHARED_COLUMNS, *(f"PC{i}" for i in range(1, num_pcs + 1)), "age_baseline", "baseline_year",
            "lookback_days", "lookback_years", "admin_years", "n_dates", *STRATA, "test", "fold"]


def binary_frame(rows, base, config):
    """Confirmed case by the CDR cutoff (a second distinct date on or before it) among per-disease rows.

    Covariates are measured at baseline only (audit S2). An exclusion root
    removes anyone meeting its case rule by the cutoff."""
    frame = rows.frame
    flow = Flow("disease_rows", len(frame))
    keep = np.ones(len(frame), dtype=bool)
    for code, *_ in rows.disease.exclusions:
        keep &= ~(frame[f"_exclusion_{code}"].to_numpy() <= base.cdr_cutoff_day)
        flow.step(f"exclusion_{code}", keep)
    out = frame.loc[keep, _shared(config.num_pcs)].copy()
    out["y"] = (frame._second.to_numpy() <= base.cdr_cutoff_day)[keep].astype(np.int8)
    out = out.reset_index(drop=True)
    counts = {"steps": flow.steps, "cases": int(out.y.sum()), "non_cases": int((out.y == 0).sum()),
              "single_record": int((out.n_dates == 1).sum())}
    return out, counts


def survival_frame(rows, base, config, *, onset="second", censor="ehr_end", exclusion="auto"):
    """Incident confirmed case on the age scale, entry at the landmark, death competing.

    onset="second" (primary) puts the event at the second distinct qualifying
    date; "first" is the first-date sensitivity. censor="ehr_end" (primary)
    censors at the last EHR-sourced record (SPEC section 3, 21:22Z: AoU's
    observation period also counts survey and physical-measurement dates),
    capped at the CDR cutoff because no record exists after it; "cutoff"
    censors at min(death, cutoff) alone. Follow-up needs ehr_end past the landmark.

    Exclusion roots (audit C3) remove only people who meet their case rule by
    the landmark. Meeting it later ends follow-up at that date: as a censoring
    (exclusion="censor"), or as competing event 3 ("competing"). "auto"
    chooses competing when those exits exceed config
    exclusion_competing_fraction of the frame, else censoring. Same-day exits:
    a disease event beats death, which beats censoring; an exclusion met that
    day voids the disease event (the case rule no longer holds) but not death.
    """
    if onset not in ("second", "first") or censor not in ("ehr_end", "cutoff"):
        raise ValueError("onset is second|first and censor is ehr_end|cutoff")
    if exclusion not in ("auto", "censor", "competing"):
        raise ValueError("exclusion is auto|censor|competing")
    frame = rows.frame
    landmark = frame._baseline.to_numpy() + config.landmark_days
    first, death = frame._first.to_numpy(), frame._death.to_numpy()
    cutoff = np.full(len(frame), base.cdr_cutoff_day)
    observed_to = np.minimum(frame._ehr_end.to_numpy(), cutoff)  # NaN (no EHR) stays NaN
    end = observed_to if censor == "ehr_end" else cutoff

    flow = Flow("disease_rows", len(frame))
    keep = np.ones(len(frame), dtype=bool)
    excluded = np.full(len(frame), np.nan)
    for code, *_ in rows.disease.exclusions:
        met = frame[f"_exclusion_{code}"].to_numpy()
        keep &= ~(met <= landmark)
        flow.step(f"exclusion_{code}_by_landmark", keep)
        excluded = np.fmin(excluded, met)
    keep &= ~(first <= landmark)
    flow.step("no_record_by_landmark", keep)
    keep &= ~(death <= landmark)
    flow.step("alive_at_landmark", keep)
    keep &= observed_to > landmark
    flow.step("ehr_past_landmark", keep)
    if censor == "cutoff":
        keep &= cutoff > landmark
        flow.step("cutoff_past_landmark", keep)

    confirmed = frame.n_dates.to_numpy() >= 2
    event_day = np.where(confirmed, frame._second.to_numpy() if onset == "second" else first, np.nan)
    exit_day = np.fmin(np.fmin(np.fmin(event_day, death), end), excluded)
    voided = excluded == exit_day
    event = np.where((event_day == exit_day) & ~voided, 1, np.where(death == exit_day, 2, 0)).astype(np.int8)
    by_exclusion = voided & (event == 0)
    exits = int(by_exclusion[keep].sum())
    if exclusion == "auto":
        exclusion = "competing" if exits > config.exclusion_competing_fraction * keep.sum() else "censor"
    if exclusion == "competing":
        event[by_exclusion] = 3
    birth = frame._birth.to_numpy()

    out = frame.loc[keep, _shared(config.num_pcs)].copy()
    out["entry_age"] = ((landmark - birth) / YEAR)[keep]
    out["entry_year"] = _year(landmark[keep])
    out["exit_age"] = ((exit_day - birth) / YEAR)[keep]
    out["event"] = event[keep]
    out["followup"] = out.exit_age.to_numpy() - out.entry_age.to_numpy()
    if not (out.followup.to_numpy() > 0).all():
        raise AssertionError("a survival row exits at or before its entry")
    out = out.reset_index(drop=True)
    counts = {"steps": flow.steps,
              "events": {"censored": int((out.event == 0).sum()), "disease": int((out.event == 1).sum()),
                         "death": int((out.event == 2).sum()), "exclusion": int((out.event == 3).sum())},
              "exclusion_exits": exits, "exclusion_as": exclusion,
              "single_record_at_risk": int((out.n_dates == 1).sum())}
    return out, counts


def ancestry_counts(binary, survival):
    """Per-ancestry people, cases and events: outcome counts for the digest, which suppresses small cells."""
    counts = {}
    for label in binary.ancestry.cat.categories:
        b, s = binary.loc[binary.ancestry == label], survival.loc[survival.ancestry == label]
        counts[str(label)] = {"binary_n": len(b), "binary_cases": int(b.y.sum()), "survival_n": len(s),
                              "survival_disease": int((s.event == 1).sum()),
                              "survival_death": int((s.event == 2).sum()),
                              "survival_exclusion": int((s.event == 3).sum())}
    return counts


@dataclass
class DiseaseFrames:
    binary: pd.DataFrame
    survival: pd.DataFrame
    flow: dict


def build_frames(source, diseases, config, *, censor="ehr_end", exclusion="auto"):
    """(base, {slug: DiseaseFrames}) for every disease: the binary and survival frames.

    `censor` and `exclusion` choose the survival frames' censoring rule and
    exclusion treatment (see `survival_frame`); the defaults are the primary analysis."""
    base = base_cohort(source, config)
    missing = sorted({code for d in diseases for code in d.snomed_codes} - set(source.manifest["snomed_codes"]))
    if missing:
        raise ValueError(f"the tables lack condition roots {missing}")
    # The tables' qualifying records must be net of exactly the branches the
    # disease list excludes, or the phenotype is not the declared one.
    extracted = source.manifest.get("excluded_branches", {})
    for disease in diseases:
        if tuple(sorted(extracted.get(disease.snomed_code, ()))) != disease.branch_codes:
            raise ValueError(f"{disease.slug}: the tables exclude branches "
                             f"{extracted.get(disease.snomed_code, [])} of {disease.snomed_code}; "
                             f"the disease list excludes {list(disease.branch_codes)}")
    frames = {}
    for disease in diseases:
        rows = disease_rows(base, source, disease)
        binary, binary_counts = binary_frame(rows, base, config)
        survival, survival_counts = survival_frame(rows, base, config, censor=censor, exclusion=exclusion)
        frames[disease.slug] = DiseaseFrames(binary, survival, {
            "disease": rows.flow, "binary": binary_counts, "survival": survival_counts,
            "by_ancestry": ancestry_counts(binary, survival)})
    return base, frames


HORIZON_CANDIDATES = (0.5, 1, 1.5, 2, 2.5, 3, 4, 5)
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _key(years):
    return f"{years:g}"


def _summary(years, horizons):
    return {"n": int(len(years)),
            "quantiles": [float(v) for v in np.quantile(years, QUANTILES)] if len(years) else [],
            "fraction_reaching": {_key(h): float((years >= h).mean()) if len(years) else 0.0 for h in horizons}}


def followup_distribution(base, config):
    """Outcome-blind follow-up from the landmark, used to fix the horizons (audit S6, N4).

    The population is base participants alive at the landmark whose EHR runs
    past it (the survival frame's rules without any disease record). The
    horizon basis is ADMINISTRATIVE follow-up, cutoff - landmark, because a
    case's own records extend the EHR. Observed follow-up (ehr_end capped at
    the cutoff) is reported beside it, and administrative follow-up is broken
    down by entry year. `ehr` describes, over the whole base cohort, how the
    EHR end relates to AoU's observation-period end, which also counts survey
    and physical-measurement dates (SPEC section 3, 21:22Z).
    """
    frame = base.frame
    landmark = frame._baseline.to_numpy() + config.landmark_days
    ehr_end, obs_end, death = frame._ehr_end.to_numpy(), frame._obs_end.to_numpy(), frame._death.to_numpy()
    observed_to = np.minimum(ehr_end, base.cdr_cutoff_day)
    eligible = ~(death <= landmark) & (observed_to > landmark)
    administrative = (base.cdr_cutoff_day - landmark)[eligible] / YEAR
    observed = (observed_to - landmark)[eligible] / YEAR
    entry_year = _year(landmark[eligible])
    horizons = sorted(set(HORIZON_CANDIDATES) | set(config.horizon_candidates))
    both = ~np.isnan(ehr_end)
    gap = (obs_end - ehr_end)[both] / YEAR
    died = ~np.isnan(death)

    def fraction(mask, within):
        return float(mask[within].mean()) if within.any() else 0.0

    def median(values):
        return float(np.median(values)) if len(values) else 0.0

    return {
        "quantiles": list(QUANTILES),
        "administrative": _summary(administrative, horizons),
        "observed": _summary(observed, horizons),
        "by_entry_year": {int(year): {"n": int((entry_year == year).sum()),
                                      "administrative_max": float(administrative[entry_year == year].max()),
                                      "administrative_min": float(administrative[entry_year == year].min())}
                          for year in np.unique(entry_year)},
        "ehr": {"n": int(len(frame)),
                "fraction_without_ehr": fraction(~both, np.ones(len(frame), dtype=bool)),
                "fraction_obs_end_after_ehr_end": fraction(obs_end > ehr_end, both),
                "median_gap_years": median(gap),
                "median_positive_gap_years": median(gap[gap > 0]),
                "fraction_deaths_after_ehr_end": fraction(death > ehr_end, died & both)},
    }


def choose_horizons(distribution, config):
    """The prespecified, outcome-blind horizon rule (SPEC section 3; audits S6, N4).

    Keep every config horizon_candidates year that at least horizon_min_reach
    of the survival-eligible cohort can reach on ADMINISTRATIVE follow-up
    (cutoff - landmark). It reads `followup_distribution` only, so it is
    computed in the same run, before any outcome is seen.
    """
    reach = distribution["administrative"]["fraction_reaching"]
    chosen = [h for h in config.horizon_candidates if reach[_key(h)] >= config.horizon_min_reach]
    if not chosen:
        raise ValueError(f"no horizon in {list(config.horizon_candidates)} is reached by "
                         f"{config.horizon_min_reach:.0%} of the survival-eligible cohort")
    return chosen
