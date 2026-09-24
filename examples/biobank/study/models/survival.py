"""Survival models: the cumulative incidence of a confirmed diagnosis by entry + h,
from alive and disease-free at the landmark, with death competing.

Age is the time axis with delayed entry at the landmark (`Surv(entry_age,
exit_age, event)`, event 1 disease, 2 death, 0 censored). Every method shares
one covariate part in the marginal index,

    sex + duchon(PC1..PCk) + admin_years + lookback_years   (+ s(entry_age))

and differs only in how z enters (SPEC section 4), as in binary.py:
- ours: gam's survival marginal slope, the slope 1 + duchon(PCs), the anchor
  on the training rows' empirical z law, no time dependence in the slope.
- shipped: the same without the window covariates, on calibrate's time block.
- covariates / standard / z_pc / calpred: gam's probit location-scale survival
  model with no z, z, z and z x PC, and z with a log scale linear in the PCs.

Death is the one shared component: fitted once per (disease, fit) as
`death_model` and given to every variant, so every method's CIF composes its
own disease hazard with the same death hazard (SPEC section 8, M14). A cause
with no events in the training rows has zero hazard.

Prediction: gam's posterior-mean cumulative hazard H_c(age) of each cause, read
at each row's entry age and at entry + tau on a follow-up grid tau, and gam's
competing-risks kernel turns the increments H_c(entry + tau) - H_c(entry) into
the cause-specific cumulative incidence from entry. The rows are predicted in
entry-age bands led by one row at the band's edge, so the 64-point age grid gam
returns starts at the same age whichever rows share a call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

VARIANTS = ("ours", "shipped", "covariates", "standard", "z_pc", "calpred")
CAUSES = {"disease": 1, "death": 2}
MARGINAL_SLOPE = ("ours", "shipped")

# Model choices study.json must declare, each with its reason (study-pipe's
# check_reasons): the PC count, the joint Duchon smooths' centers in q and in
# ours' slope, the window covariates, the latent law, and the shared death
# model: "location-scale" (q + z) or "marginal-slope" (q, a constant slope).
REQUIRED = ("num_pcs", "q_centers", "slope_centers", "windows", "latent_law", "death_model")
DEFAULTS = {
    # s(entry_age) in every method's q: given attained age, a time-since-entry effect.
    "entry_age_smooth": False,
    # Follow-up step of the CIF grid, in years; every horizon is on the grid.
    "cif_step_years": 0.02,
    # Width, in years, of the entry-age bands rows are predicted in.
    "predict_band_years": 5.0,
    # Rows per prediction call, bounding the (rows x grid) surfaces held at once.
    "predict_rows": 20000,
    # Spacing, in years, of the ages a band's cumulative hazard is read at; H is
    # linear between them (the hazard is constant within a step).
    "predict_knot_years": 1.0,
}
LATENT_LAW = "global-empirical"
DEATH_MODELS = ("location-scale", "marginal-slope")


def settings_of(settings):
    settings = settings or {}
    unknown = set(settings) - set(DEFAULTS) - set(REQUIRED)
    missing = set(REQUIRED) - set(settings)
    if unknown or missing:
        raise ValueError(f"survival settings: unknown {sorted(unknown)}, missing {sorted(missing)}")
    s = {**DEFAULTS, **settings}
    if s["latent_law"] != LATENT_LAW:
        raise ValueError(f"unsupported survival latent law {s['latent_law']!r}; the study anchors on {LATENT_LAW!r}")
    if s["death_model"] not in DEATH_MODELS:
        raise ValueError(f"unsupported death model {s['death_model']!r}; use one of {DEATH_MODELS}")
    for key in ("q_centers", "slope_centers"):
        if s[key] <= s["num_pcs"] + 1:
            raise ValueError(f"{key} must exceed the Duchon null space ({s['num_pcs'] + 1} columns)")
    return s


def pc_columns(s):
    return [f"PC{i + 1}" for i in range(s["num_pcs"])]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def components(variant, settings):
    if variant not in VARIANTS:
        raise ValueError(f"unknown survival variant {variant!r}")
    return ["disease"]


def shared_components(settings):
    return ["death"]


def has_sex_term(disease):
    if disease["sex"] not in (None, "female", "male"):
        raise ValueError(f"disease sex must be null, female or male, not {disease['sex']!r}")
    return disease["sex"] is None


def duchon(s, centers):
    return f"duchon({', '.join(pc_columns(s))}, centers={centers})"


def covariate_part(s, sex=True, windows=True):
    terms = [*(["sex"] if sex else []), duchon(s, s["q_centers"]), *(s["windows"] if windows else [])]
    if s["entry_age_smooth"]:
        terms.append("s(entry_age)")
    return " + ".join(terms)


def shipped_centers(num_pcs):
    """calibrate's PcSmoothConfig::for_pcs, as in binary.py."""
    import math
    floor = num_pcs + 2
    return max(math.ceil(3 * num_pcs / 2), floor), max(math.ceil(5 * num_pcs / 4), floor)


def formulas(variant, component, s, sex=True):
    """(formula, extra gamfit.fit keywords) of one fit."""
    response = "Surv(entry_age, exit_age, event)"
    marginal = {"survival_likelihood": "marginal-slope", "z_column": "z",
                "config": {"latent_measure": s["latent_law"]}}
    if component == "death":
        if variant != "shared":
            raise ValueError("the death fit is shared by every variant")
        main = f"{response} ~ {covariate_part(s, sex)}"
        if s["death_model"] == "location-scale":
            return f"{main} + z", {"survival_likelihood": "location-scale"}
        return main, {**marginal, "slope_formula": "1"}
    if component != "disease":
        raise ValueError(f"survival models have no {component!r} component")
    if variant == "shipped":
        context, slope = shipped_centers(s["num_pcs"])
        # calibrate's time block: the shipped arm is that model under test.
        return (f"{response} ~ {'sex + ' if sex else ''}{duchon(s, context)}",
                {**marginal, "slope_formula": f"1 + {duchon(s, slope)}",
                 "config": {**marginal["config"], "time_num_internal_knots": 6, "time_degree": 3}})
    main = f"{response} ~ {covariate_part(s, sex)}"
    if variant == "ours":
        return main, {**marginal, "slope_formula": f"1 + {duchon(s, s['slope_centers'])}"}
    scale = {"survival_likelihood": "location-scale"}
    if variant == "covariates":
        return main, scale
    if variant == "standard":
        return f"{main} + z", scale
    if variant == "z_pc":
        return f"{main} + z + {' + '.join(f'z_{pc}' for pc in pc_columns(s))}", scale
    if variant == "calpred":
        return f"{main} + z", {**scale, "noise_formula": " + ".join(pc_columns(s))}
    raise ValueError(f"unknown survival variant {variant!r}")


def columns(variant, component, s, sex=True):
    """The frame columns a fit's design uses."""
    sexes = ["sex"] if sex else []
    entry = ["entry_age"] if s["entry_age_smooth"] else []
    if component == "death":
        return [*(["z"] if s["death_model"] == "location-scale" else []), *sexes, *s["windows"], *entry,
                *pc_columns(s)]
    if variant == "shipped":
        return ["z", *sexes, *pc_columns(s)]
    return [*([] if variant == "covariates" else ["z"]), *sexes, *s["windows"], *entry, *pc_columns(s)]


def covariates(variant, component, settings, disease):
    """The frame columns a fit's design uses."""
    if variant != "shared":
        components(variant, settings)
    return columns(variant, component, settings_of(settings), has_sex_term(disease))


def design(variant, component, frame, s, sex=True):
    """A fit's input columns, with the z x PC products z_pc adds."""
    data = {c: frame[c].to_numpy(float) for c in columns(variant, component, s, sex)}
    data["entry_age"] = frame.entry_age.to_numpy(float)
    if variant == "z_pc" and component == "disease":
        for pc in pc_columns(s):
            data[f"z_{pc}"] = data["z"] * data[pc]
    return data


def check_frame(frame, s, with_response):
    needed = ["entry_age", "z", "sex", *s["windows"], *pc_columns(s)] + (["exit_age", "event"] if with_response else [])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"survival frame lacks {missing}")
    if not np.isfinite(frame[needed].to_numpy(float)).all():
        raise ValueError("survival frame has non-finite inputs")
    if not with_response:
        return
    codes = frame.event.to_numpy()
    if not np.isin(codes, [0, *CAUSES.values()]).all():
        raise ValueError(f"survival event codes must be 0 or one of {sorted(CAUSES.values())}")
    entry, exit_ = frame.entry_age.to_numpy(float), frame.exit_age.to_numpy(float)
    if (entry > exit_).any():
        raise ValueError(f"{int((entry > exit_).sum())} survival row(s) leave before they enter")
    at_entry = entry == exit_
    if (at_entry & (codes != 0)).any():
        raise ValueError(f"{int((at_entry & (codes != 0)).sum())} survival row(s) have an event at their entry age")
    # A row censored at its entry age adds nothing to the likelihood but would
    # enter the z standardization and gam's empirical z law: the driver removes
    # these rows once, before it standardizes z.
    if at_entry.any():
        raise ValueError(f"{int(at_entry.sum())} zero-length rows must be removed before the fit")


def fit(variant, component, train, settings, out_dir, reference=None, *, disease):
    import gamfit
    s = settings_of(settings)
    if component == "disease":
        components(variant, s)
    elif component != "death" or variant != "shared":
        raise ValueError(f"survival models have no ({variant!r}, {component!r}) fit")
    if reference is not None:
        raise ValueError("survival LOGO refits are cold: no reference fit is used")
    check_frame(train, s, True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sex = has_sex_term(disease)
    formula, keywords = formulas(variant, component, s, sex)
    data = design(variant, component, train, s, sex)
    data["exit_age"] = train.exit_age.to_numpy(float)
    data["event"] = (train.event.to_numpy() == CAUSES[component]).astype(float)
    events = int(data["event"].sum())
    info = {"variant": variant, "component": component, "rows": len(train), "events": events, "sex_term": sex}
    started = time.perf_counter()
    if component != "disease" and events == 0:
        # A competing cause that never happens in these rows has zero hazard.
        spec = {"zero_hazard": True, "formula": formula, **keywords}
        info.update(converged=True, seconds=0.0)
    else:
        model = gamfit.fit(data, formula, **keywords)
        info["seconds"] = time.perf_counter() - started
        model.save(out / "model.gamfit")
        payload = json.loads((out / "model.gamfit").read_text())["model"]
        if keywords["survival_likelihood"] == "marginal-slope":
            # No CTN: the score enters as the driver standardized it, and the
            # anchor is the requested empirical law, never the standard normal.
            if (payload.get("latent_z_rank_int_calibration") is not None
                    or payload.get("latent_z_conditional_calibration") is not None):
                raise ValueError("the marginal-slope fit transformed the score")
            measure = (payload.get("latent_measure") or {}).get("kind")
            if measure != s["latent_law"]:
                raise ValueError(f"the marginal-slope fit anchored on {measure!r}, not {s['latent_law']!r}")
            # A follow-up-varying slope is improper on an unbounded latent law (gam#2765).
            if payload.get("slope_time_basis") not in (None, {}, []):
                raise ValueError("the marginal-slope fit carries a time-varying slope basis")
        # gam's own typed verdict on the optimization, recorded as gam reports it.
        convergence = model.convergence
        # gam's time basis ends at its last knot (log age): past it no hazard is fitted.
        knots = [float(k) for k in (payload.get("survival_time_knots") or []) if np.isfinite(k)]
        if not knots:
            raise ValueError("the survival fit saved no time knots")
        spec = {"zero_hazard": False, "formula": formula, **keywords,
                "survival_time_anchor": payload.get("survival_time_anchor"),
                "time_upper": float(np.exp(max(knots))),
                "lambdas": [float(v) for v in model.smoothing_parameters().values()]}
        info.update(converged=convergence["certified"], convergence=convergence)
    write_json(out / "spec.json", {**spec, **info, "settings": s})
    return info


def follow_up_grid(horizons, step):
    """Follow-up years from 0 to the last horizon, with every horizon on the grid."""
    edges = np.concatenate([[0.0], np.sort(np.asarray(horizons, dtype=float))])
    grid = np.unique(np.concatenate([np.linspace(a, b, max(2, int(np.ceil((b - a) / step - 1e-9)) + 1))
                                     for a, b in zip(edges[:-1], edges[1:])]))
    return grid, np.searchsorted(grid, horizons)


def cumulative_hazard_increments(directory, variant, component, frame, s, sex, grid):
    """(H(entry + grid) - H(entry) of each row, which rows reach past the fitted ages) from
    the saved fit: gam's posterior-mean cumulative hazard at the ages of each entry-age
    band, `predict_knot_years` apart from the band's edge to its last follow-up age, read
    at each row's own ages on the piecewise-linear interpolant between them."""
    import gamfit
    spec = json.loads((directory / "spec.json").read_text())
    if spec["zero_hazard"]:
        return np.zeros((len(frame), len(grid))), np.zeros(len(frame), dtype=bool)
    model = gamfit.load(directory / "model.gamfit")
    entry = frame.entry_age.to_numpy(float)
    bands = np.floor(entry / s["predict_band_years"])
    out = np.empty((len(frame), len(grid)))
    beyond = np.zeros(len(frame), dtype=bool)
    for band in np.unique(bands):
        rows = np.flatnonzero(bands == band)
        edge = band * s["predict_band_years"]
        # Past the last training age the surface is read at its end: no hazard is
        # fitted there, and the rows are counted (`beyond_fit` in the prediction).
        last = min(edge + s["predict_band_years"] + grid[-1], spec["time_upper"])
        knots = np.unique(np.append(np.arange(edge, last, s["predict_knot_years"]), last))
        for start in range(0, len(rows), s["predict_rows"]):
            chunk = rows[start:start + s["predict_rows"]]
            part = pd.DataFrame(design(variant, component, frame.iloc[chunk], s, sex))
            # Placeholders: prediction reads no outcome, and the exit is the last age read.
            part["exit_age"] = np.maximum(last, part.entry_age.to_numpy(float))
            part["event"] = 0.0
            prediction = model.predict(part, time_grid=knots)
            surface = np.asarray(prediction.cumulative_hazard_at(knots), dtype=float)
            if surface.shape != (len(chunk), len(knots)) or not np.isfinite(surface).all():
                raise ValueError(f"gam returned an invalid cumulative hazard surface for {directory}")
            ages = entry[chunk, None] + np.concatenate([[0.0], grid])[None, :]
            beyond[chunk] = ages[:, -1] > spec["time_upper"]
            at = np.vstack([np.interp(ages[i], knots, surface[i]) for i in range(len(chunk))])
            out[chunk] = at[:, 1:] - at[:, :1]
    return out, beyond


def predict(variant, model_dirs, frame, settings, horizons, *, disease):
    """rows x horizons cumulative incidence of the disease (and of death) by entry + h."""
    from gamfit._binding import rust_module
    s = settings_of(settings)
    components(variant, s)
    check_frame(frame, s, False)
    sex = has_sex_term(disease)
    dirs = {c: Path(model_dirs[c]) for c in ("disease", "death")}
    spec = json.loads((dirs["disease"] / "spec.json").read_text())
    if spec["variant"] != variant or spec["sex_term"] != sex:
        raise ValueError(f"{dirs['disease']} holds {spec['variant']!r} with sex_term {spec['sex_term']}, not "
                         f"{variant!r} for a disease declared for sex {disease['sex']!r}")
    grid, at = follow_up_grid(horizons, s["cif_step_years"])
    disease_h, beyond = cumulative_hazard_increments(dirs["disease"], variant, "disease", frame, s, sex, grid)
    death_h, beyond_death = cumulative_hazard_increments(dirs["death"], "shared", "death", frame, s, sex, grid)
    increments = [disease_h, death_h]
    # From entry, each cause's cumulative hazard starts at zero, so the CIF from
    # entry is gam's competing-risks composition on the follow-up grid.
    times = np.concatenate([[0.0], grid])
    hazards = [np.concatenate([np.zeros((len(frame), 1)), h], axis=1) for h in increments]
    cif, _ = rust_module().competing_risks_cif_from_predictions(times, hazards, ["disease", "death"])
    cif = np.asarray(cif, dtype=float)
    return {"risk": cif[0][:, 1 + at], "death": cif[1][:, 1 + at],
            "beyond_fit": (beyond | beyond_death).astype(float)}
