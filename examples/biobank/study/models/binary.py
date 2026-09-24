"""Binary models: P(a confirmed case by the CDR cutoff | baseline covariates, z).

ours and the competitors share one covariate part, fitted by gamfit with the
probit link (SPEC section 4):

    s(age_baseline) + sex + duchon(PC1..PCk) + s(admin_years) + s(lookback_years)

(the PCs enter as ONE joint Duchon smooth; a disease declared for one sex has
no sex term in any fit), and differ only in how z enters:
- ours: Bernoulli marginal slope. The covariate part is the marginal index
  q(x), the slope is 1 + duchon(PCs) (+ s(age) where the simulator shows it
  helps), and the anchor integrates the empirical law of the training z,
  never the Gaussian one.
- covariates: z does not enter (a probit GAM).
- standard: ours with a constant slope.
- z_pc: ours with the slope 1 + PC1 + ... + PCk.
- calpred: + z, with the log SD of the liability linear in the PCs (a binomial
  location-scale GAM), CalPred-style.
- shipped: gnomon calibrate's binary model exactly as shipped (calibrate/
  estimate.rs and construction.rs at the study's calibrate pin): sex and the
  joint PC smooth in q with calibrate's center counts, no age or windows, and
  gam's link deviation and score warp (`linkwiggle()` in both formulas).

standard and z_pc are restricted-slope ablations of ours, not the probit GAMs
"covariate part + z" and "+ z + z x PC": the anchored intercept a(q, b) of a
finite-basis q and b is not itself in q's basis, so the two families span
different risks. On study-sim v1 small, hypertension (8k rows), the
constant-slope fit and the probit GAM "+ z" differ by at most 0.008 in risk
(mean 0.0013). gamfit at gam 6fc5ad9c1c refuses the calpred fit at every
startup seed ("dense Hessian shape mismatch") and a smooth log SD by design, so
calpred fails until gam fixes it.

Every prediction is gam's posterior mean. Besides the risk, a marginal-slope
variant reports its score slope on the probit scale, d probit(risk) / dz at
each row's own z: gam's analytic `probit_score_derivative`, integrated at the
posterior nodes that give the risk. shipped's link deviation and score warp
leave gam no such derivative and calpred is not a marginal-slope fit, so their
slope is NaN; covariates' is 0.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

VARIANTS = ("ours", "shipped", "covariates", "standard", "z_pc", "calpred", "no_pc")

AGE = "age_baseline"

# Model choices study.json must declare, each with its reason (study-pipe's
# check_reasons): the PC count, the joint Duchon smooths' centers in q and the
# slope of ours (a linear null space in k dimensions needs more than k + 1, and
# gam's own default grows with n, gam#2993), the window covariates, the latent
# law, and the basis size of the age term in ours' slope (null: no age term).
# Everything else is gam's own behaviour.
REQUIRED = ("num_pcs", "q_centers", "slope_centers", "windows", "latent_law", "slope_age_k")
DEFAULTS = {
    # The log-smoothing levels gam's marginal-slope multistart searches from beside
    # its derived start (gnomon#2359); null keeps gam's five. Each level is one
    # certified outer search, so the set is most of a fit's cost.
    "outer_start_levels": None,
}

# The anchor integrates the training rows' own z law, on the score's own axis.
# Never "auto": it takes the Gaussian closed form after a fixed screen (SPEC
# section 4). gam's "conditional-location-scale" anchors a transformed residual
# score instead, which this study's raw-axis slope is not defined on.
LATENT_LAW = "global-empirical"
MARGINAL_SLOPE = ("ours", "shipped", "standard", "z_pc")


def settings_of(settings):
    settings = settings or {}
    unknown = set(settings) - set(REQUIRED) - set(DEFAULTS)
    missing = set(REQUIRED) - set(settings)
    if unknown or missing:
        raise ValueError(f"binary settings: unknown {sorted(unknown)}, missing {sorted(missing)}")
    s = {**DEFAULTS, **settings}
    levels = s["outer_start_levels"]
    if levels is not None and (not levels or any(not isinstance(v, (int, float)) or v != v for v in levels)):
        raise ValueError("outer_start_levels is null or a non-empty list of log-smoothing levels")
    if s["latent_law"] != LATENT_LAW:
        raise ValueError(f"unsupported binary latent law {s['latent_law']!r}; the study anchors on {LATENT_LAW!r}")
    for key in ("q_centers", "slope_centers"):
        if s[key] <= s["num_pcs"] + 1:
            raise ValueError(f"{key} must exceed the Duchon null space ({s['num_pcs'] + 1} columns)")
    if s["slope_age_k"] is not None and s["slope_age_k"] < 3:
        raise ValueError("slope_age_k is null or a basis size of at least 3")
    return s


def pc_columns(s):
    return [f"PC{i + 1}" for i in range(s["num_pcs"])]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def components(variant, settings):
    if variant not in VARIANTS:
        raise ValueError(f"unknown binary variant {variant!r}")
    return ["disease"]


def has_sex_term(disease):
    """The single-sex rule: a disease declared for one sex has no sex term in any fit."""
    if disease["sex"] not in (None, "female", "male"):
        raise ValueError(f"disease sex must be null, female or male, not {disease['sex']!r}")
    return disease["sex"] is None


def covariates(variant, component, settings, disease):
    """The frame columns a fit's design uses."""
    components(variant, settings)
    return columns(variant, settings_of(settings), has_sex_term(disease))


def duchon(s, centers):
    return f"duchon({', '.join(pc_columns(s))}, centers={centers})"


def covariate_part(s, sex=True):
    return " + ".join([f"s({AGE})", *(["sex"] if sex else []), duchon(s, s["q_centers"]),
                       *(f"s({w})" for w in s["windows"])])


def shipped_centers(num_pcs):
    """calibrate's PcSmoothConfig::for_pcs: ceil(3k/2) centers in the context and
    ceil(5k/4) in the slope, never fewer than k + 2."""
    floor = num_pcs + 2
    return max(math.ceil(3 * num_pcs / 2), floor), max(math.ceil(5 * num_pcs / 4), floor)


def formulas(variant, s, sex=True):
    """(formula, extra gamfit.fit keywords) of a variant. `sex` is False for a
    disease declared for one sex: no fit then has a sex term."""
    if variant == "shipped":
        context, slope = shipped_centers(s["num_pcs"])
        return (f"y ~ {'sex + ' if sex else ''}{duchon(s, context)} + linkwiggle()",
                {"family": "bernoulli-marginal-slope", "z_column": "z",
                 "slope_formula": f"1 + {duchon(s, slope)} + linkwiggle()",
                 "config": marginal_slope_config(s)})
    main = f"y ~ {covariate_part(s, sex)}"
    probit = {"family": "binomial", "link": "probit"}
    if variant == "covariates":
        return main, probit
    if variant == "no_pc":
        # The PC-free baseline: the score with age, sex and the observation windows, no
        # ancestry information anywhere (user, 2026-09-24: what do PCs as predictors buy).
        terms = [f"s({AGE})", *(["sex"] if sex else []), *(f"s({w})" for w in s["windows"]), "z"]
        return "y ~ " + " + ".join(terms), probit
    if variant == "calpred":
        return f"{main} + z", {**probit, "noise_formula": " + ".join(pc_columns(s))}
    if variant == "standard":
        slope = "1"
    elif variant == "z_pc":
        slope = " + ".join(["1", *pc_columns(s)])
    elif variant == "ours":
        slope = f"1 + {duchon(s, s['slope_centers'])}"
        if s["slope_age_k"] is not None:
            slope += f" + s({AGE}, k={s['slope_age_k']})"
    else:
        raise ValueError(f"unknown binary variant {variant!r}")
    return main, {"family": "bernoulli-marginal-slope", "z_column": "z", "slope_formula": slope,
                  "config": marginal_slope_config(s)}


def marginal_slope_config(s):
    """gam's fit config for a marginal-slope variant: the anchor's law, and the
    multistart's levels where the study chose them."""
    config = {"latent_measure": s["latent_law"]}
    if s["outer_start_levels"] is not None:
        config["outer_start_levels"] = [float(v) for v in s["outer_start_levels"]]
    return config


def columns(variant, s, sex=True):
    """The frame columns a variant reads."""
    sexes = ["sex"] if sex else []
    if variant == "shipped":
        return ["z", *sexes, *pc_columns(s)]
    if variant == "no_pc":
        return ["z", *sexes, AGE, *s["windows"]]
    return ([] if variant == "covariates" else ["z"]) + [*sexes, AGE, *s["windows"], *pc_columns(s)]


def design(variant, frame, s, sex=True):
    """A variant's input columns."""
    return {c: frame[c].to_numpy(float) for c in columns(variant, s, sex)}


def check_frame(frame, s, with_response):
    needed = ["z", "sex", AGE, *s["windows"], *pc_columns(s)] + (["y"] if with_response else [])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"binary frame lacks {missing}")
    if not np.isfinite(frame[needed].to_numpy(float)).all():
        raise ValueError("binary frame has non-finite inputs")
    if with_response and not set(np.unique(frame.y)) <= {0, 1}:
        raise ValueError("the binary response must be 0/1")


def fit(variant, component, train, settings, out_dir, reference=None, *, disease):
    import gamfit
    s = settings_of(settings)
    components(variant, s)
    if component != "disease":
        raise ValueError(f"binary models have no {component!r} component")
    if reference is not None:
        raise ValueError("binary LOGO refits are cold: no reference fit is used")
    check_frame(train, s, True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sex = has_sex_term(disease)
    formula, keywords = formulas(variant, s, sex)
    data = design(variant, train, s, sex)
    data["y"] = train.y.to_numpy(float)
    started = time.perf_counter()
    model = gamfit.fit(data, formula, **keywords)
    seconds = time.perf_counter() - started
    model.save(out / "model.gamfit")
    # gam's saved document is {"kind", "version", "model"}.
    payload = json.loads((out / "model.gamfit").read_text())["model"]
    if variant in MARGINAL_SLOPE:
        # No CTN: the score enters as the driver standardized it, and the
        # anchor is the requested empirical law, never the standard normal.
        if (payload.get("latent_z_rank_int_calibration") is not None
                or payload.get("latent_z_conditional_calibration") is not None):
            raise ValueError("the marginal-slope fit transformed the score")
        measure = (payload.get("latent_measure") or {}).get("kind")
        if measure != s["latent_law"]:
            raise ValueError(f"the marginal-slope fit anchored on {measure!r}, not {s['latent_law']!r}")
    # gam's own typed verdict on the optimization, recorded as gam reports it.
    convergence = model.convergence
    info = {"variant": variant, "rows": len(train), "events": int(train.y.sum()), "seconds": seconds,
            "sex_term": sex, "converged": convergence["certified"], "convergence": convergence,
            "lambdas": [float(v) for v in model.smoothing_parameters().values()]}
    write_json(out / "spec.json", {"formula": formula, **keywords, **info, "settings": s})
    return info


def predict(variant, model_dirs, frame, settings, horizons=None, *, disease):
    import gamfit
    s = settings_of(settings)
    check_frame(frame, s, False)
    out = Path(model_dirs["disease"])
    spec = json.loads((out / "spec.json").read_text())
    if spec["variant"] != variant or spec["sex_term"] != has_sex_term(disease):
        raise ValueError(f"{out} holds {spec['variant']!r} with sex_term {spec['sex_term']}, not {variant!r} "
                         f"for a disease declared for sex {disease['sex']!r}")
    model = gamfit.load(out / "model.gamfit")

    table = model.predict(design(variant, frame, s, spec["sex_term"]), return_type="dict")
    risk = np.asarray(table["mean" if variant in MARGINAL_SLOPE else "posterior_mean"], dtype=float)
    if risk.shape != (len(frame),) or not np.all((risk > 0) & (risk < 1)):
        raise ValueError(f"{variant} predictions are invalid")
    if variant == "covariates":
        return {"risk": risk, "slope": np.zeros(len(frame))}
    if "probit_score_derivative" not in table:
        return {"risk": risk, "slope": np.full(len(frame), np.nan)}
    return {"risk": risk, "slope": np.asarray(table["probit_score_derivative"], dtype=float)}
