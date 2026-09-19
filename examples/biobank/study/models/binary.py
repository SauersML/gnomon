"""Binary models: P(a confirmed case by the CDR cutoff | baseline covariates, z).

ours and the competitors share one covariate part, fitted by gamfit with the
probit link (SPEC section 4):

    s(age_baseline) + sex + duchon(PC1..PCk) + s(admin_years) + s(lookback_years)

(the PCs enter as ONE joint Duchon smooth; a sex-restricted disease, whose rows
share one sex, has no sex term in any fit), and differ only in how z enters:
- ours: Bernoulli marginal slope. The covariate part is the marginal index
  q(x), the slope is 1 + duchon(PCs) (+ s(age) where the simulator shows it
  helps), and the anchor integrates an empirical latent law of the training z,
  never the Gaussian one.
- covariates: z does not enter (a probit GAM).
- standard: z linear, as a marginal-slope fit with a constant slope.
- z_pc: z linear and z times each PC, as a marginal-slope fit with the slope
  1 + PC1 + ... + PCk.
- calpred: + z, with the log SD of the liability linear in the PCs (a binomial
  location-scale GAM), CalPred-style.
- shipped: gnomon calibrate's binary model exactly as shipped (calibrate/
  estimate.rs and construction.rs at the study's calibrate pin): sex and the
  joint PC smooth in q with calibrate's center counts, no age or windows, and
  gam's link deviation and score warp (`linkwiggle()` in both formulas).

standard and z_pc are the probit GAMs "covariate part + z" and "+ z + z x PC",
reparametrized: with a constant or PC-linear slope, the anchored index a(q, b) + b z
spans the same risks as alpha(x) + b z, and only the penalty sits on q rather than
on alpha. On study-sim v1 small, hypertension (8k rows), the constant-slope fit and
the probit GAM "+ z" differ by at most 0.008 in risk (mean 0.0013), less than the
GAM moves when its double penalty is dropped (0.027), while gamfit's standard-REML
path timed out at 900 s on two of three diseases at 8k rows (gam#2817, comment
5737819338). gamfit at gam 6fc5ad9c1c refuses the calpred fit at every startup seed
("dense Hessian shape mismatch") and a smooth log SD by design, so calpred fails
until gam fixes it.

Besides the risk, every variant reports its local score slope on the probit
scale, d probit(p) / dz at each row's own z, as a central difference of its own
predictions, so slope recovery is scored on one scale for every method.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np

VARIANTS = ("ours", "shipped", "covariates", "standard", "z_pc", "calpred")

AGE = "age_baseline"

# Model choices study.json must declare, each with its reason (study-pipe's
# check_reasons): the PC count, the joint Duchon smooths' centers in q and the
# slope of ours (a linear null space in k dimensions needs more than k + 1, and
# gam's own default grows with n, gam#2993), the window covariates, the latent
# law, and the basis size of the age term in ours' slope (null: no age term).
REQUIRED = ("num_pcs", "q_centers", "slope_centers", "windows", "latent_law", "slope_age_k")

# Everything else is gam's own behaviour, or a numerical setting of prediction.
DEFAULTS = {
    # z step of the central-difference slope.
    "slope_step": 0.05,
}

# latent_law: the training rows' own z law, or gam's estimated law of z given
# the context (gam#2926). Never "auto": it takes the Gaussian closed form after
# a fixed screen (SPEC section 4).
LATENT_LAWS = ("global-empirical", "conditional-location-scale")
MARGINAL_SLOPE = ("ours", "shipped", "standard", "z_pc")


def settings_of(settings):
    settings = settings or {}
    unknown = set(settings) - set(DEFAULTS) - set(REQUIRED)
    missing = set(REQUIRED) - set(settings)
    if unknown or missing:
        raise ValueError(f"binary settings: unknown {sorted(unknown)}, missing {sorted(missing)}")
    s = {**DEFAULTS, **settings}
    if s["latent_law"] not in LATENT_LAWS:
        raise ValueError(f"unsupported binary latent law {s['latent_law']!r}; use one of {LATENT_LAWS}")
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
    sex-restricted disease, whose rows all share one sex: no fit then has a sex term."""
    if variant == "shipped":
        context, slope = shipped_centers(s["num_pcs"])
        pcs = ", ".join(pc_columns(s))
        return (f"y ~ {'sex + ' if sex else ''}s({pcs}, type=duchon, centers={context}) + linkwiggle()",
                {"family": "bernoulli-marginal-slope", "z_column": "z",
                 "slope_formula": f"1 + s({pcs}, type=duchon, centers={slope}) + linkwiggle()",
                 "config": {"latent_measure": s["latent_law"]}})
    main = f"y ~ {covariate_part(s, sex)}"
    probit = {"family": "binomial", "link": "probit"}
    if variant == "covariates":
        return main, probit
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
                  "config": {"latent_measure": s["latent_law"]}}


def columns(variant, s, sex=True):
    """The frame columns a variant reads."""
    sexes = ["sex"] if sex else []
    if variant == "shipped":
        return ["z", *sexes, *pc_columns(s)]
    return ([] if variant == "covariates" else ["z"]) + [*sexes, AGE, *s["windows"], *pc_columns(s)]


def design(variant, frame, s, sex=True, shift=0.0):
    """A variant's input columns, with z moved by `shift`."""
    data = {c: frame[c].to_numpy(float) for c in columns(variant, s, sex)}
    if "z" in data:
        data["z"] = data["z"] + shift
    return data


def check_frame(frame, s, with_response):
    needed = ["z", "sex", AGE, *s["windows"], *pc_columns(s)] + (["y"] if with_response else [])
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError(f"binary frame lacks {missing}")
    if not np.isfinite(frame[needed].to_numpy(float)).all():
        raise ValueError("binary frame has non-finite inputs")
    if with_response and not set(np.unique(frame.y)) <= {0, 1}:
        raise ValueError("the binary response must be 0/1")


def convergence_summary(payload):
    """gam's own convergence and certificate verdict, reported, never overridden.

    A fit gam keeps without certifying carries that word in its convergence
    record; `certified` is then False and the driver's convergence gate must not
    count the fit as converged."""
    result = payload.get("fit_result") or {}
    record = result.get("convergence") or {}
    certificate = (result.get("artifacts") or {}).get("criterion_certificate") or {}
    gradient = next(iter((certificate.get("stationarity") or {}).values()), {})
    return {"inner_status": record.get("inner_status"), "outer_iterations": record.get("outer_iterations"),
            "projected_grad_norm": gradient.get("projected_grad_norm"), "bound": gradient.get("bound"),
            "certified": record.get("inner_status") == "Converged" and "ncertified" not in json.dumps(record)}


def fit(variant, component, train, settings, out_dir, reference=None):
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
    # A sex-restricted disease's rows share one sex; a sex term would be identically constant.
    sex = bool(train.sex.nunique() > 1)
    formula, keywords = formulas(variant, s, sex)
    data = design(variant, train, s, sex)
    data["y"] = train.y.to_numpy(float)
    started = time.perf_counter()
    model = gamfit.fit(data, formula, **keywords)
    seconds = time.perf_counter() - started
    model.save(out / "model.gamfit")
    payload = json.loads((out / "model.gamfit").read_text())["payload"]
    if variant in MARGINAL_SLOPE:
        # No CTN: the score enters as the driver standardized it, and the
        # anchor is the requested empirical law, never the standard normal.
        if (payload.get("latent_z_rank_int_calibration") is not None
                or payload.get("latent_z_conditional_calibration") is not None):
            raise ValueError("the marginal-slope fit transformed the score")
        measure = (payload.get("latent_measure") or {}).get("kind")
        if measure != s["latent_law"]:
            raise ValueError(f"the marginal-slope fit anchored on {measure!r}, not {s['latent_law']!r}")
    convergence = convergence_summary(payload)
    info = {"variant": variant, "rows": len(train), "events": int(train.y.sum()), "seconds": seconds,
            "sex_term": sex, "converged": convergence["certified"], "convergence": convergence,
            "lambdas": [float(v) for v in model.smoothing_parameters().values()]}
    write_json(out / "spec.json", {"formula": formula, **keywords, **info, "settings": s})
    return info


_INV_CDF = np.frompyfunc(NormalDist().inv_cdf, 1, 1)


def ndtri(p):
    """Standard normal quantile, to double precision (Wichura AS241 in the stdlib)."""
    p = np.asarray(p, float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("ndtri needs probabilities strictly inside (0, 1)")
    return _INV_CDF(p).astype(float)


def predict(variant, model_dirs, frame, settings, horizons=None):
    import gamfit
    s = settings_of(settings)
    check_frame(frame, s, False)
    out = Path(model_dirs["disease"])
    spec = json.loads((out / "spec.json").read_text())
    if spec["variant"] != variant:
        raise ValueError(f"{out} holds {spec['variant']!r}, not {variant!r}")
    model = gamfit.load(out / "model.gamfit")

    def risk_at(shift):
        p = np.asarray(model.predict(design(variant, frame, s, spec["sex_term"], shift)), dtype=float)
        if p.shape != (len(frame),) or not np.all((p > 0) & (p < 1)):
            raise ValueError(f"{variant} predictions are invalid")
        return p

    risk = risk_at(0.0)
    if variant == "covariates":
        return {"risk": risk, "slope": np.zeros(len(frame))}
    h = s["slope_step"]
    return {"risk": risk, "slope": (ndtri(risk_at(h)) - ndtri(risk_at(-h))) / (2 * h)}
