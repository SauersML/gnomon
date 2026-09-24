"""study/models/survival.py: every variant fits, persists and replays; CIFs are proper; refusals hold.

Runs on MSI only (gamfit fits). The data are an age-scale world with delayed
entry, a skewed score and competing death.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study.models import survival  # noqa: E402

N = 4000
SETTINGS = {"num_pcs": 6, "q_centers": 10, "slope_centers": 10, "windows": ["admin_years", "lookback_years"],
            "latent_law": "global-empirical", "death_model": "location-scale"}
DISEASE = {"slug": "test", "sex": None}
HORIZONS = [1.0, 3.0, 5.0]


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(20260923)
    pcs = rng.normal(size=(N, 6))
    eps = (rng.gamma(4.0, 1.0, N) - 4.0) / 2.0
    male = (rng.random(N) < 0.4).astype(np.int8)
    entry = 30 + 45 * rng.random(N)
    lin = 0.4 * eps + 0.2 * np.tanh(pcs[:, 0]) + 0.1 * male
    t1 = np.log(np.exp(0.05 * entry) + 0.05 * rng.exponential(size=N) / (0.03 * np.exp(-2.75 + lin))) / 0.05
    t2 = np.log(np.exp(0.085 * entry) + 0.085 * rng.exponential(size=N) / (0.004 * np.exp(-3.4))) / 0.085
    cutoff = rng.uniform(1.5, 6.5, N)
    exit_ = np.minimum(np.minimum(t1, t2), entry + cutoff - 0.5)
    frame = pd.DataFrame({"entry_age": entry, "exit_age": exit_,
                          "event": np.select([t1 <= exit_, t2 <= exit_], [1, 2], 0).astype(np.int8),
                          "z": (eps - eps.mean()) / eps.std(), "sex": male, "admin_years": cutoff,
                          "lookback_years": rng.uniform(1.0, 14.0, N)})
    for j in range(6):
        frame[f"PC{j + 1}"] = pcs[:, j]
    frame = frame[frame.exit_age > frame.entry_age].reset_index(drop=True)
    train, test = frame.iloc[:3000].reset_index(drop=True), frame.iloc[3000:].reset_index(drop=True)
    # A prediction frame carries no outcome; horizons past the training ages are not predicted.
    test = test[test.entry_age + max(HORIZONS) <= train.exit_age.max()].reset_index(drop=True)
    return train, test.drop(columns=["exit_age", "event"])


@pytest.fixture(scope="module")
def fitted(frames, tmp_path_factory):
    train, test = frames
    death = tmp_path_factory.mktemp("death")
    survival.fit("shared", "death", train, SETTINGS, death, disease=DISEASE)
    cache = {}

    def get(variant):
        if variant not in cache:
            directory = tmp_path_factory.mktemp(variant)
            info = survival.fit(variant, "disease", train, SETTINGS, directory, disease=DISEASE)
            prediction = survival.predict(variant, {"disease": directory, "death": death}, test, SETTINGS,
                                          HORIZONS, disease=DISEASE)
            cache[variant] = (directory, info, prediction, death)
        return cache[variant]
    return get


LAST_SHIPPED = [v for v in survival.VARIANTS if v != "shipped"] + ["shipped"]


@pytest.mark.parametrize("variant", LAST_SHIPPED)
def test_variant_fits_and_replays(frames, fitted, variant):
    _, test = frames
    directory, info, first, death = fitted(variant)
    assert info["variant"] == variant and info["rows"] == 3000 and info["converged"] is True
    for key in ("risk", "death"):
        cif = first[key]
        assert cif.shape == (len(test), len(HORIZONS)) and np.isfinite(cif).all()
        assert (cif >= 0).all() and (cif <= 1).all()
        assert (np.diff(cif, axis=1) >= -1e-12).all(), "a cumulative incidence never decreases in the horizon"
    assert (first["risk"] + first["death"] <= 1 + 1e-10).all()
    assert (first["risk"][:, -1] > 0).any()
    again = survival.predict(variant, {"disease": directory, "death": death}, test, SETTINGS, HORIZONS, disease=DISEASE)
    assert np.array_equal(first["risk"], again["risk"])
    # A row's prediction depends on that row alone.
    reversed_rows = survival.predict(variant, {"disease": directory, "death": death}, test.iloc[::-1], SETTINGS,
                                     HORIZONS, disease=DISEASE)
    assert np.allclose(reversed_rows["risk"][::-1], first["risk"], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("variant", MARGINAL := ["ours", "shipped"])
def test_marginal_slope_fits_anchor_on_the_empirical_law(fitted, variant):
    payload = json.loads((fitted(variant)[0] / "model.gamfit").read_text())["model"]
    assert payload["latent_measure"]["kind"] == "global-empirical"
    assert payload["latent_z_rank_int_calibration"] is None and payload["latent_z_conditional_calibration"] is None


def test_the_score_raises_the_risk_where_it_enters(frames, fitted):
    """In this world a higher score means more disease: ours and standard order the rows so."""
    _, test = frames
    for variant in ("ours", "standard"):
        risk = fitted(variant)[2]["risk"][:, -1]
        assert np.corrcoef(risk, test.z)[0, 1] > 0.2, variant


def test_death_is_shared_and_zero_hazard_when_absent(frames, tmp_path):
    train, test = frames
    none = tmp_path / "no_deaths"
    info = survival.fit("shared", "death", train.assign(event=train.event.where(train.event != 2, 0)), SETTINGS,
                        none, disease=DISEASE)
    assert info["events"] == 0 and info["converged"] is True
    spec = json.loads((none / "spec.json").read_text())
    assert spec["zero_hazard"] is True
    disease = tmp_path / "standard"
    survival.fit("standard", "disease", train, SETTINGS, disease, disease=DISEASE)
    out = survival.predict("standard", {"disease": disease, "death": none}, test, SETTINGS, HORIZONS, disease=DISEASE)
    assert np.all(out["death"] == 0)


def test_formulas_share_one_covariate_part():
    s = survival.settings_of(SETTINGS)
    forms = {v: survival.formulas(v, "disease", s) for v in survival.VARIANTS}
    part = forms["covariates"][0]
    assert part.startswith("Surv(entry_age, exit_age, event) ~ sex + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=10)")
    assert all(forms[v][0].startswith(part) for v in survival.VARIANTS if v != "shipped")
    assert forms["ours"][1]["slope_formula"] == "1 + duchon(PC1, PC2, PC3, PC4, PC5, PC6, centers=10)"
    assert forms["ours"][1]["survival_likelihood"] == "marginal-slope"
    assert forms["standard"][1] == {"survival_likelihood": "location-scale"}
    assert forms["calpred"][1]["noise_formula"] == "PC1 + PC2 + PC3 + PC4 + PC5 + PC6"
    assert "admin_years" not in forms["shipped"][0]
    death, keywords = survival.formulas("shared", "death", s)
    assert death.endswith("+ z") and keywords == {"survival_likelihood": "location-scale"}


def test_refusals(frames, tmp_path):
    train, _ = frames
    for law in ("standard-normal", "auto", "conditional-location-scale"):
        with pytest.raises(ValueError, match="unsupported survival latent law"):
            survival.settings_of({**SETTINGS, "latent_law": law})
    with pytest.raises(ValueError, match="unsupported death model"):
        survival.settings_of({**SETTINGS, "death_model": "cox"})
    with pytest.raises(ValueError, match="Duchon null space"):
        survival.settings_of({**SETTINGS, "q_centers": 7})
    with pytest.raises(ValueError, match="shared by every variant"):
        survival.formulas("ours", "death", survival.settings_of(SETTINGS))
    bad = train.copy()
    bad.loc[0, "exit_age"] = bad.loc[0, "entry_age"]
    with pytest.raises(ValueError, match="zero-length"):
        survival.fit("standard", "disease", bad, SETTINGS, tmp_path / "a", disease=DISEASE)
    bad = train.copy()
    bad.loc[0, "event"] = 3
    with pytest.raises(ValueError, match="event codes"):
        survival.fit("standard", "disease", bad, SETTINGS, tmp_path / "b", disease=DISEASE)


def test_single_sex_disease_has_no_sex_term():
    female = {"slug": "breast_cancer", "sex": "female"}
    for variant in survival.VARIANTS:
        assert "sex" not in survival.covariates(variant, "disease", SETTINGS, female)
        assert "sex" in survival.covariates(variant, "disease", SETTINGS, DISEASE)
        formula, keywords = survival.formulas(variant, "disease", survival.settings_of(SETTINGS), sex=False)
        assert "sex" not in formula and "sex" not in (keywords.get("slope_formula") or "")
    assert "sex" not in survival.covariates("shared", "death", SETTINGS, female)
