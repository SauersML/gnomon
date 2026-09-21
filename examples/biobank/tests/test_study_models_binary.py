"""study/models/binary.py: every variant fits, persists and replays; slopes, forms and refusals hold.

Runs on MSI only (gamfit fits). The data are a small logistic world, off every
fitted probit family (a linear probit truth sends a link wiggle's lambda to the
boundary, gam#2978), with a known score coefficient, so each variant's reported
probit slope can be checked against the true local probit slope.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.models import binary  # noqa: E402

LOGIT_SLOPE = 0.7
N = 4000
SETTINGS = {"num_pcs": 6, "q_centers": 10, "slope_centers": 10, "windows": ["admin_years", "lookback_years"],
            "latent_law": "global-empirical", "slope_age_k": 4}
DISEASE = {"slug": "test", "sex": None}


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(20260918)
    pcs = rng.normal(size=(N, 6))
    frame = pd.DataFrame({f"PC{i + 1}": pcs[:, i] for i in range(6)})
    frame["age_baseline"] = 18 + 72 * rng.beta(2.2, 2.0, N)
    frame["sex"] = (rng.uniform(size=N) < 0.4).astype(np.int8)
    frame["admin_years"] = rng.uniform(0.3, 6.3, N)
    frame["lookback_years"] = 1 + rng.gamma(2.0, 3.0, N)
    frame["z"] = rng.standard_normal(N)
    eta = (-1.9 + 0.05 * (frame.age_baseline - 50) + 0.3 * frame.sex + 0.3 * np.tanh(frame.PC1)
           + 0.08 * frame.admin_years + LOGIT_SLOPE * frame.z)
    p = expit(eta)
    frame["y"] = (rng.uniform(size=N) < p).astype(np.int8)
    # The true local probit slope d probit(p) / dz of each row.
    probit = np.array([NormalDist().inv_cdf(v) for v in p])
    frame["true_slope"] = LOGIT_SLOPE * p * (1 - p) / (np.exp(-0.5 * probit ** 2) / math.sqrt(2 * math.pi))
    train, test = frame.iloc[:3000].reset_index(drop=True), frame.iloc[3000:].reset_index(drop=True)
    return train, test.drop(columns="y")


@pytest.fixture(scope="module")
def fitted(frames, tmp_path_factory):
    """One fit per variant, made on first use, so one variant's failure fails only its own tests."""
    train, test = frames
    cache = {}

    def get(variant):
        if variant not in cache:
            directory = tmp_path_factory.mktemp(variant)
            info = binary.fit(variant, "disease", train, SETTINGS, directory, disease=DISEASE)
            prediction = binary.predict(variant, {"disease": directory}, test, SETTINGS, None, disease=DISEASE)
            cache[variant] = (directory, info, prediction)
        return cache[variant]
    return get


# shipped carries gam's link wiggles, the slowest fit, so its cases run last.
LAST_SHIPPED = [v for v in binary.VARIANTS if v != "shipped"] + ["shipped"]
NO_SLOPE = ("shipped", "calpred")


@pytest.mark.parametrize("variant", LAST_SHIPPED)
def test_variant_fits_and_replays(frames, fitted, variant):
    _, test = frames
    directory, info, first = fitted(variant)
    assert info["variant"] == variant and info["rows"] == 3000
    assert info["converged"] is True and info["convergence"]["certified"] is True
    assert first["risk"].shape == (len(test),) and first["slope"].shape == (len(test),)
    assert np.all((first["risk"] > 0) & (first["risk"] < 1))
    # gam reports no score derivative through shipped's wiggles or for calpred's location-scale fit.
    assert np.isnan(first["slope"]).all() if variant in NO_SLOPE else np.isfinite(first["slope"]).all()
    again = binary.predict(variant, {"disease": directory}, test, SETTINGS, None, disease=DISEASE)
    assert np.array_equal(first["risk"], again["risk"])
    assert np.array_equal(first["slope"], again["slope"], equal_nan=True)
    # Row order does not change a row's prediction.
    reversed_rows = binary.predict(variant, {"disease": directory}, test.iloc[::-1], SETTINGS, None,
                                   disease=DISEASE)
    assert np.allclose(reversed_rows["risk"][::-1], first["risk"], rtol=1e-12, atol=0)


@pytest.mark.parametrize("variant", [v for v in LAST_SHIPPED if v in binary.MARGINAL_SLOPE])
def test_marginal_slope_fits_anchor_on_the_empirical_law(fitted, variant):
    payload = json.loads((fitted(variant)[0] / "model.gamfit").read_text())["model"]
    assert payload["latent_measure"]["kind"] == "global-empirical"
    assert payload["latent_z_rank_int_calibration"] is None and payload["latent_z_conditional_calibration"] is None


def test_covariates_slope_is_zero(fitted):
    assert np.all(fitted("covariates")[2]["slope"] == 0)


def test_standard_slope_is_constant(frames, fitted):
    # A constant slope b: the local probit slope is b, shrunk a little under the posterior mean.
    _, test = frames
    standard = fitted("standard")[2]["slope"]
    assert np.std(standard) < 0.02 * np.mean(standard)
    assert abs(np.mean(standard) - test.true_slope.mean()) < 0.12


@pytest.mark.parametrize("variant", ["ours", "z_pc"])
def test_slope_recovers_the_true_mean_slope(frames, fitted, variant):
    _, test = frames
    assert abs(np.mean(fitted(variant)[2]["slope"]) - test.true_slope.mean()) < 0.12


@pytest.mark.parametrize("variant", ["ours", "standard", "z_pc"])
def test_slope_is_the_derivative_of_the_reported_risk(frames, fitted, variant):
    """gam's analytic probit slope against the reported risks themselves, z moved a step either way."""
    _, test = frames
    directory, _, first = fitted(variant)
    probit = np.vectorize(NormalDist().inv_cdf)
    step = 0.05
    up, down = (binary.predict(variant, {"disease": directory}, test.assign(z=test.z + shift), SETTINGS, None,
                               disease=DISEASE)["risk"] for shift in (step, -step))
    assert np.allclose(first["slope"], (probit(up) - probit(down)) / (2 * step), rtol=1e-3, atol=1e-5)


def test_covariates_design_has_no_z(frames):
    _, test = frames
    assert "z" not in binary.design("covariates", test, binary.settings_of(SETTINGS))


def test_competitors_differ_from_ours_only_in_how_z_enters():
    s = binary.settings_of(SETTINGS)
    forms = {v: binary.formulas(v, s) for v in binary.VARIANTS}
    covariate_part = forms["covariates"][0]
    assert all(forms[v][0].startswith(covariate_part) for v in binary.VARIANTS if v != "shipped")
    assert covariate_part.count("duchon(") == 1 and "s(admin_years)" in covariate_part
    slopes = {v: k.get("slope_formula") for v, (_, k) in forms.items()}
    assert slopes["standard"] == "1" and slopes["z_pc"] == "1 + PC1 + PC2 + PC3 + PC4 + PC5 + PC6"
    assert slopes["ours"] == "1 + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=10) + s(age_baseline, k=4)"
    assert all(k["link"] == "probit" for v, (_, k) in forms.items() if v in ("covariates", "calpred"))


def test_shipped_is_calibrate_as_shipped():
    s = binary.settings_of(SETTINGS)
    formula, keywords = binary.formulas("shipped", s)
    # calibrate/model.rs PcSmoothConfig::for_pcs(6) = (9, 8); for_pcs(16) = (24, 20).
    assert binary.shipped_centers(6) == (9, 8) and binary.shipped_centers(16) == (24, 20)
    assert formula == "y ~ sex + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=9) + linkwiggle()"
    assert keywords["slope_formula"] == "1 + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=8) + linkwiggle()"
    assert binary.columns("shipped", s) == ["z", "sex", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]


def test_refusals(frames, tmp_path):
    train, _ = frames
    for law in ("standard-normal", "auto", "conditional-location-scale"):
        with pytest.raises(ValueError, match="unsupported binary latent law"):
            binary.settings_of({**SETTINGS, "latent_law": law})
    with pytest.raises(ValueError, match="unknown \\['length_scale'\\]"):
        binary.settings_of({**SETTINGS, "length_scale": 1.0})
    with pytest.raises(ValueError, match="missing \\['q_centers'\\]"):
        binary.settings_of({k: v for k, v in SETTINGS.items() if k != "q_centers"})
    with pytest.raises(ValueError, match="Duchon null space"):
        binary.settings_of({**SETTINGS, "q_centers": 7})
    with pytest.raises(ValueError, match="cold"):
        binary.fit("standard", "disease", train, SETTINGS, tmp_path / "a", reference=tmp_path, disease=DISEASE)
    with pytest.raises(ValueError, match="lacks"):
        binary.fit("standard", "disease", train.drop(columns="admin_years"), SETTINGS, tmp_path / "b",
                   disease=DISEASE)
    bad = train.copy()
    bad.loc[0, "y"] = 2
    with pytest.raises(ValueError, match="0/1"):
        binary.fit("standard", "disease", bad, SETTINGS, tmp_path / "c", disease=DISEASE)


def test_single_sex_disease_has_no_sex_term(frames, tmp_path):
    train, test = frames
    female = {"slug": "breast_cancer", "sex": "female"}
    info = binary.fit("standard", "disease", train.assign(sex=0), SETTINGS, tmp_path, disease=female)
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert info["sex_term"] is False and "sex" not in spec["formula"]
    risk = binary.predict("standard", {"disease": tmp_path}, test.assign(sex=0), SETTINGS, None, disease=female)["risk"]
    assert np.all((risk > 0) & (risk < 1))
    # A model fitted under one declaration never predicts under another.
    with pytest.raises(ValueError, match="sex_term"):
        binary.predict("standard", {"disease": tmp_path}, test, SETTINGS, None, disease=DISEASE)
    s = binary.settings_of(SETTINGS)
    for variant in binary.VARIANTS:
        assert "sex" not in binary.covariates(variant, "disease", SETTINGS, female)
        assert "sex" in binary.covariates(variant, "disease", SETTINGS, DISEASE)
        formula, keywords = binary.formulas(variant, s, sex=False)
        assert "sex" not in formula and "sex" not in (keywords.get("slope_formula") or "")
        assert "sex" not in (keywords.get("noise_formula") or "")


def test_predict_refuses_another_variants_directory(frames, fitted):
    _, test = frames
    with pytest.raises(ValueError, match="holds"):
        binary.predict("standard", {"disease": fitted("z_pc")[0]}, test, SETTINGS, None, disease=DISEASE)
