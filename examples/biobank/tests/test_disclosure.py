"""The differencing auditor: each planted disclosure must fire, and a safe digest must pass."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import disclosure  # noqa: E402

COUNT, TRAIN = {"type": "count"}, {"type": "count", "population": "train"}
REGISTRY = {"n": COUNT, "cases": COUNT, "controls": COUNT, "events": COUNT, "noncases": COUNT,
            "censored": COUNT, "n_train": TRAIN, "auc": {"type": "score"},
            "prevalence": {"type": "proportion", "of": "cases", "per": "n"}}
AXES = ["ancestry", "region", "division", "sex", "age_band", "ehr_site", "risk"]
PARTS = {"binary": [("n", ["cases", "controls"])], "survival": [("n", ["events", "noncases", "censored"])]}
CUMULATIVE = {"survival": {"events": "increasing", "censored": "increasing", "noncases": "decreasing"}}


def row(stratum="overall", fit="pooled", model="binary", horizon="all", variant="ours", **metrics):
    return {"disease": "t2d", "model": model, "variant": variant, "fit": fit, "stratum": stratum,
            "horizon": horizon, **metrics}


def suppressed(stratum, **keys):
    return row(stratum, status="insufficient_support", **keys)


def run(rows, **options):
    options.setdefault("parts", PARTS)
    options.setdefault("cumulative", CUMULATIVE)
    return disclosure.audit(rows, REGISTRY, axes=AXES, **options)


def values(findings, kind=None):
    return sorted(f["value"] for f in findings if kind is None or f["kind"] == kind)


def safe_rows():
    return [row(n=1000, cases=300, controls=700, prevalence=0.3, auc=0.71),
            row("ancestry_afr", n=400, cases=100, controls=300),
            row("ancestry_eur", n=500, cases=150, controls=350),
            suppressed("ancestry_amr"), suppressed("ancestry_mid"),
            row("sex_0", n=600, cases=170, controls=430),
            row("sex_1", n=400, cases=130, controls=270)]


def test_safe_digest_has_no_findings():
    assert run(safe_rows()) == []


def test_suppressed_rows_absent_or_marked_audit_the_same():
    rows = [r for r in safe_rows() if "status" not in r]
    assert run(rows) == []


def test_a_released_count_of_twenty_is_a_breach_and_twenty_one_is_not():
    # The policy bans 1 to 20 inclusive.
    assert values(run([row(n=520, cases=20, controls=500)]), "released") == [20]
    assert run([row(n=521, cases=21, controls=500)]) == []


def test_one_suppressed_category_is_recovered_from_the_total():
    rows = [row(n=1000), row("ancestry_afr", n=400), row("ancestry_eur", n=585), suppressed("ancestry_amr")]
    assert values(run(rows), "derived") == [15]


def test_an_unreleased_part_is_recovered_from_its_whole():
    assert values(run([row(n=500, cases=490)]), "derived") == [10]


def test_a_division_is_recovered_inside_its_region():
    rows = [row(n=1000), row("region_south", n=100), row("division_south_atlantic", n=88),
            suppressed("division_east_south_central"), suppressed("division_west_south_central")]
    assert 12 in values(run(rows), "derived")


def test_a_small_group_sum_across_region_and_division_fires():
    # South is suppressed; the unlisted regions plus South's unlisted divisions total 10.
    rows = [row(n=1000), row("region_northeast", n=300), row("region_midwest", n=300), row("region_west", n=300),
            suppressed("region_south"), row("division_south_atlantic", n=50),
            row("division_east_south_central", n=40)]
    assert 10 in values(run(rows), "group_sum")


def test_a_group_sum_of_twenty_one_or_more_is_safe():
    rows = [row(n=1000), row("region_northeast", n=300), row("region_midwest", n=300), row("region_west", n=300),
            suppressed("region_south"), row("division_south_atlantic", n=40),
            row("division_east_south_central", n=39)]
    assert run(rows) == []


def test_cumulative_events_between_horizons_are_counts():
    rows = [row(model="survival", horizon=h, n=500, events=e) for h, e in (("h1", 25), ("h2", 40), ("h3", 70))]
    assert values(run(rows), "derived") == [15]


def test_a_proportion_with_its_denominator_pins_a_suppressed_numerator():
    assert values(run([row(n=1500, prevalence=float(f"{18 / 1500:.4g}"))]), "proportion") == [18]


def test_a_proportion_without_its_denominator_pins_nothing():
    assert run([row(prevalence=float(f"{18 / 1500:.4g}"))]) == []


def test_a_large_numerator_is_not_pinned_by_four_digits():
    assert run([row(n=400000, prevalence=0.375)]) == []


def test_a_logo_cross_cell_is_recovered_from_pooled_and_other_groups():
    rows = [row(n=1000), row("region_south", n=400), row("ancestry_afr", n=300), row("ancestry_eur", n=700),
            row(fit="logo_ancestry_afr", n=300), row("region_south", fit="logo_ancestry_afr", n=190),
            row(fit="logo_ancestry_eur", n=700), row("region_south", fit="logo_ancestry_eur", n=195)]
    assert 15 in values(run(rows), "derived")


def test_the_logo_training_complement_is_a_count():
    rows = [row(n_train=800), row(fit="logo_ancestry_afr", n_train=790)]
    assert values(run(rows), "derived") == [10]


def test_the_survival_cohort_inside_the_binary_cohort_differences_to_a_count():
    rows = [row("sex_0", n=100), row("sex_0", model="survival", horizon="h1", n=85)]
    assert values(run(rows, nested_models=[("survival", "binary")]), "derived") == [15]


def test_per_variant_risk_bins_partition_their_variant():
    rows = [row(n=1000), row("risk_0_0_01", variant="ours", n=700), row("risk_0_01_0_05", variant="ours", n=288),
            row("risk_0_0_01", variant="standard", n=500)]
    assert values(run(rows), "derived") == [12]


def test_counts_that_differ_across_variants_are_refused():
    with pytest.raises(ValueError, match="inconsistent"):
        run([row(n=100, variant="ours"), row(n=99, variant="standard")])


def test_numbers_that_make_a_count_negative_are_refused():
    with pytest.raises(ValueError, match="negative"):
        run([row(n=100), row("ancestry_afr", n=60), row("ancestry_eur", n=50)])


def test_an_unregistered_metric_is_refused():
    with pytest.raises(ValueError, match="not registered"):
        run([row(n=100, noncase_rate=0.5)])
