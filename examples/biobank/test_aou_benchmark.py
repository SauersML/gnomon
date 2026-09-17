import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aou_benchmark as bench
import aou_benchmark_table as table


def test_auc_counts_ties_as_half_and_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.normal(size=300) + y * 0.8, 1)
    pos, neg = p[y == 1], p[y == 0]
    pairwise = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert bench.auc(y, p) == pytest.approx(pairwise)


def test_delong_variance_matches_paired_bootstrap_scale():
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 2, n)
    a = rng.normal(size=n) + y
    b = a + rng.normal(scale=0.5, size=n)
    aucs, cov = bench.delong(y, np.vstack([a, b]))
    assert aucs[0] == pytest.approx(bench.auc(y, a))
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    boot = []
    for _ in range(300):
        rows = rng.integers(0, n, n)
        boot.append(bench.auc(y[rows], a[rows]) - bench.auc(y[rows], b[rows]))
    assert se == pytest.approx(np.std(boot), rel=0.25)


def test_logistic_fit_recovers_coefficients():
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(20000), rng.normal(size=(20000, 2))])
    beta = np.array([-1.0, 0.8, -0.4])
    y = (rng.random(20000) < bench.expit(X @ beta)).astype(float)
    assert bench.logistic_fit(X, y) == pytest.approx(beta, abs=0.08)


def test_calibration_of_true_probabilities_has_unit_slope():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.6, 40000)
    y = (rng.random(40000) < p).astype(float)
    metrics = bench.cell_metrics(y, p)
    assert metrics["calibration_slope"] == pytest.approx(1.0, abs=0.08)
    assert metrics["calibration_intercept"] == pytest.approx(0.0, abs=0.08)
    assert metrics["observed"] == pytest.approx(metrics["mean_predicted"], abs=0.01)


def test_pc_adjusted_score_removes_linear_ancestry_mean_and_scale():
    rng = np.random.default_rng(5)
    n = 30000
    pc = rng.normal(size=n)
    s = 2.0 * pc + np.exp(0.5 * pc) * rng.normal(size=n)
    df = pd.DataFrame({"PC1": pc, "ancestry": np.where(pc > 0, "a", "b"), "PGS": s})
    z = bench.score_transforms(df, "PGS", np.ones(n, dtype=bool), 1)["pc_adjusted"]
    assert abs(np.corrcoef(z, pc)[0, 1]) < 0.02
    assert np.std(z[pc > 1]) == pytest.approx(np.std(z[pc < -1]), rel=0.1)


def test_small_cells_are_suppressed_and_tokens_never_carry_small_counts():
    rng = np.random.default_rng(6)
    n = 400
    test = pd.DataFrame({"y": (rng.random(n) < 0.3).astype(float), "sex": rng.integers(0, 2, n),
                         "age0": rng.uniform(20, 80, n),
                         "ancestry": ["eur"] * 380 + ["eas"] * 20})
    predictions = {m: np.clip(rng.uniform(0.1, 0.5, n), 0, 1) for m in ("covariates", "standard", "gnomon")}
    with tempfile.TemporaryDirectory() as tmp:
        digest = bench.Digest(tmp)
        bench.report(digest, "hypertension", "PGS000001", test, predictions, {"min_report_count": 20})
        names = {p.name for p in Path(tmp).iterdir()}
    assert "digest__hypertension__pgs000001__all__ancestry_eas__insufficient_support.txt" in names
    assert not any("ancestry_eas" in name and "insufficient" not in name for name in names)
    assert any(name.startswith("digest__hypertension__pgs000001__delta__gnomon__vs__standard__overall__auc_difference__")
               for name in names)
    counts = [int(name.split("__")[-1][:-4]) for name in names if "__cases__" in name or "__n__" in name]
    assert min(counts) >= 20


def test_config_rejects_disease_spam_and_small_reporting_minimum():
    scores = {"PGS004236": "multi_ancestry", "PGS004525": "european"}
    config = {"google_project": "p", "workspace_cdr": "p.d", "seed": 1, "num_pcs": 6,
              "max_rows_per_disease": 40000, "max_rows_per_ancestry": 20000, "test_fraction": 0.25,
              "min_report_count": 20, "maximum_bytes_billed": 10**10, "query_timeout_seconds": 300,
              "lookback_days": 365, "gnomon_timeout_seconds": 600, "gnomon_centers": 8,
              "diseases": {"hypertension": {"snomed_code": "38341003", "scores": scores}}}
    bench.validate_config(config)
    with pytest.raises(ValueError):
        bench.validate_config(dict(config, min_report_count=5))
    with pytest.raises(ValueError):
        bench.validate_config(dict(config, diseases={f"d{i}x": {"snomed_code": "38341003", "scores": scores}
                                                     for i in range(6)}))
    with pytest.raises(ValueError):
        bench.validate_config(dict(config, max_rows_per_ancestry=50000))


def test_config_requires_development_ancestry_and_a_multi_ancestry_lead():
    config = {"google_project": "p", "workspace_cdr": "p.d", "seed": 1, "num_pcs": 6,
              "max_rows_per_disease": 40000, "max_rows_per_ancestry": 20000, "test_fraction": 0.25,
              "min_report_count": 20, "maximum_bytes_billed": 10**10, "query_timeout_seconds": 300,
              "lookback_days": 365, "gnomon_timeout_seconds": 600, "gnomon_centers": 8}
    for scores in (["PGS004236", "PGS004525"], {"PGS004525": "european"}, {"PGS004236": "trans_ethnic"}):
        with pytest.raises(ValueError):
            bench.validate_config(dict(config, diseases={"hypertension": {"snomed_code": "38341003",
                                                                          "scores": scores}}))


def test_sample_caps_each_ancestry_without_looking_at_outcomes():
    rng = np.random.default_rng(7)
    n = 5000
    ancestry = np.where(np.arange(n) < 4000, "eur", "afr")
    base = pd.DataFrame({"person_id": [str(i) for i in range(n)], "ancestry": ancestry})
    scores = {"PGS000001": pd.DataFrame({"person_id": base.person_id, "PGS000001": rng.normal(size=n)})}
    config = {"seed": 3, "max_rows_per_disease": 2500, "max_rows_per_ancestry": 1500, "test_fraction": 0.25}
    first = bench.disease_cohort(base, set(base.person_id[:100]), scores, config)
    second = bench.disease_cohort(base, set(base.person_id[-2000:]), scores, config)
    assert first.ancestry.value_counts().to_dict() == {"eur": 1500, "afr": 1000}
    assert list(first.person_id) == list(second.person_id)
    assert list(first.is_test) == list(second.is_test)


def test_failure_class_reports_only_the_exception_class_name():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp, "fit.log")
        log.write_text("gnomon_fit_started\nTraceback...\ngamfit._exceptions.IntegrationError: participant 123 did x\n")
        assert bench.failure_class(log) == "integrationerror"


def test_failure_class_names_the_fixed_category_of_a_solver_message():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp, "fit.log")
        log.write_text("Traceback...\ngamfit._exceptions.GamError: gam error: exact two-block spatial "
                       "optimization failed: no candidate seeds passed outer startup validation\n")
        assert bench.failure_class(log) == "gamerror_startup_seeds"
        log.write_text("Traceback...\ngamfit._exceptions.GamError: resource policy refused: "
                       "refusing to densify operator-backed design\n")
        assert bench.failure_class(log) == "gamerror_resource_policy"
        log.write_text("Traceback...\ngamfit._exceptions.GamError: something new entirely\n")
        assert bench.failure_class(log) == "gamerror"


def test_failure_stage_is_the_last_marker_the_worker_printed():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp, "fit.log")
        for text, stage in (("Traceback...\nImportError: x\n", "before_fit"),
                            ("gnomon_fit_started\n[3s] outer iter\nGamError: x\n", "fit"),
                            ("gnomon_fit_started\ngnomon_fit_saved\nGamError: x\n", "predict"),
                            ("gnomon_fit_started\ngnomon_fit_saved\ngnomon_predict_complete\n", "after_predict")):
            log.write_text(text)
            assert bench.failure_stage(log) == stage


def test_table_leads_with_african_and_admixed_american_gains_and_never_prints_small_cells():
    rng = np.random.default_rng(8)
    ancestry = ["afr"] * 300 + ["amr"] * 300 + ["eur"] * 300 + ["eas"] * 15
    n = len(ancestry)
    test = pd.DataFrame({"y": (rng.random(n) < 0.3).astype(float), "sex": rng.integers(0, 2, n),
                         "age0": rng.uniform(20, 80, n), "ancestry": ancestry})
    with tempfile.TemporaryDirectory() as tmp:
        digest = bench.Digest(tmp)
        for pgs, development in (("PGS000002", "european"), ("PGS000001", "multi_ancestry")):
            predictions = {m: rng.uniform(0.1, 0.5, n) for m in ("covariates", "standard", "gnomon")}
            digest.emit("type_2_diabetes", bench.slug(pgs), "development", development)
            digest.emit("type_2_diabetes", bench.slug(pgs), "gnomon", "status", "ok")
            bench.report(digest, "type_2_diabetes", pgs, test, predictions, {"min_report_count": 20})
        names = [f"gs://bucket/run/call-bench/digest/{p.name}" for p in Path(tmp).iterdir()]
    text = table.render(names)
    sections = text.split("### ")
    assert sections[1].startswith("Held-out gains in African ancestry (headline)")
    assert sections[2].startswith("Held-out gains in admixed American ancestry (headline)")
    assert sections[3].startswith("Held-out gains in European ancestry (comparator)")
    rows = [line for line in sections[1].splitlines() if line.startswith("| type 2 diabetes")]
    assert [row.split(" | ")[1:3] for row in rows] == [["PGS000001", "multi-ancestry"], ["PGS000002", "european"]]
    cells = rows[0].split(" | ")
    assert cells[3] == "300"
    for delta, detectable in ((cells[6], cells[7]), (cells[8], cells[9])):
        se = float(delta.split(" ± ")[1])
        assert float(detectable) == pytest.approx(2.8016 * se, abs=3e-4)
    assert len(sections) == 5 and "ancestry_eas" not in text
    counts = [int(cell.replace(",", "")) for row in text.splitlines() if row.startswith("| type")
              for cell in row.split(" | ")[3:5] if cell.replace(",", "").isdigit()]
    assert min(counts) >= 20


def test_table_suppresses_a_cell_below_the_minimum_even_if_a_token_carries_it():
    names = ["digest__hypertension__pgs000001__development__multi_ancestry.txt",
             "digest__hypertension__pgs000001__standard__ancestry_afr__n__400.txt",
             "digest__hypertension__pgs000001__standard__ancestry_afr__cases__12.txt",
             "digest__hypertension__pgs000001__delta__standard__vs__covariates__ancestry_afr__auc_difference__m1.5em05.txt",
             "digest__hypertension__pgs000001__delta__standard__vs__covariates__ancestry_afr__auc_difference_se__0.004.txt"]
    assert table.number("m1.5em05") == -1.5e-05
    row = next(line for line in table.render(names).split("### ")[1].splitlines() if line.startswith("| hyper"))
    assert "insufficient support" in row and "12" not in row


def test_support_counts_every_eligible_participant_before_the_cap_and_suppresses_small_groups():
    rng = np.random.default_rng(9)
    ancestry = ["afr"] * 3000 + ["eur"] * 5000 + ["mid"] * 30
    n = len(ancestry)
    base = pd.DataFrame({"person_id": [str(i) for i in range(n)], "ancestry": ancestry})
    scores = {"PGS000001": pd.DataFrame({"person_id": base.person_id, "PGS000001": rng.normal(size=n)})}
    cases = set(base.person_id[rng.random(n) < 0.3])
    config = {"seed": 3, "max_rows_per_disease": 3000, "max_rows_per_ancestry": 1000, "test_fraction": 0.25}
    scored = bench.scored_participants(base, cases, scores)
    assert bench.disease_cohort(base, cases, scores, config).ancestry.value_counts().to_dict() == {"afr": 1000,
                                                                                                "eur": 1000,
                                                                                                "mid": 30}
    with tempfile.TemporaryDirectory() as tmp:
        digest = bench.Digest(tmp)
        bench.report_support(digest, "hypertension", scored, 20)
        names = sorted(p.name for p in Path(tmp).iterdir())
    afr_cases = int(scored.y[scored.ancestry == "afr"].sum())
    assert "digest__hypertension__support__ancestry_afr__eligible__3000.txt" in names
    assert f"digest__hypertension__support__ancestry_afr__cases__{afr_cases}.txt" in names
    assert "digest__hypertension__support__ancestry_mid__insufficient_support.txt" in names
    assert not any("ancestry_mid" in name and "insufficient" not in name for name in names)
    text = table.render(names)
    assert text.startswith("### Eligible support before the per-ancestry cap")
    assert "| hypertension | AFR | 3,000 |" in text and "| MID |" not in text


def test_every_status_the_benchmark_publishes_is_a_public_label():
    import re as regex
    import aou_status
    source = Path(bench.__file__).read_text()
    published = set(regex.findall(r'publish_status\(status, "([a-z_]+)"\)', source))
    assert "benchmark_support_completed" in published and "benchmark_completed" in published
    assert published <= aou_status.LABELS
