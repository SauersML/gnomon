import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aou_benchmark as bench


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
    config = {"google_project": "p", "workspace_cdr": "p.d", "seed": 1, "num_pcs": 6,
              "max_rows_per_disease": 40000, "test_fraction": 0.25, "min_report_count": 20,
              "maximum_bytes_billed": 10**10, "query_timeout_seconds": 300, "lookback_days": 365,
              "gnomon_timeout_seconds": 600, "gnomon_centers": 8,
              "diseases": {"hypertension": {"snomed_code": "38341003", "scores": ["PGS004525"]}}}
    bench.validate_config(config)
    with pytest.raises(ValueError):
        bench.validate_config(dict(config, min_report_count=5))
    with pytest.raises(ValueError):
        bench.validate_config(dict(config, diseases={f"d{i}x": {"snomed_code": "38341003", "scores": ["PGS004525"]}
                                                     for i in range(6)}))


def test_failure_class_reports_only_the_exception_class_name():
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp, "fit.log")
        log.write_text("gnomon_fit_started\nTraceback...\ngamfit._exceptions.IntegrationError: participant 123 did x\n")
        assert bench.failure_class(log) == "integrationerror"
