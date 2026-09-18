"""The study's models, one module per outcome kind behind the one interface study.py calls.

binary.py (study-model-bin) and survival.py (study-model-surv) each export:

    VARIANTS                                    ours first, then the competitors
    components(variant, settings)               the fits a variant makes, e.g. ["disease"]
    shared_components(settings)                 optional: fitted once per (disease, fit) as
                                                variant "shared" and given to every variant
                                                (survival: the competing causes, death and
                                                exclusion, so all methods share one CIF basis)
    fit(variant, component, train, settings, out_dir, reference=None) -> small JSON-able dict
    predict(variant, model_dirs, frame, settings, horizons) -> {"risk": array, ...}

The driver runs each fit in its own process with its thread budget set, gives
`train` with z standardized on exactly that fit's training rows, and gives
`predict` a frame without outcome columns and {component: directory} for every
component the variant needs. Import gamfit inside fit/predict, not at module
import: the driver imports this package for VARIANTS alone.
"""
from __future__ import annotations

from importlib import import_module

KINDS = ("binary", "survival")


def _module(kind):
    if kind not in KINDS:
        raise ValueError(f"unknown model kind {kind!r}")
    return import_module(f"study.models.{kind}")


class _Variants(dict):
    """VARIANTS[kind], read from the kind's module when first asked for, so the
    package imports even before both kind modules exist."""
    def __missing__(self, kind):
        declared = _module(kind).VARIANTS
        self[kind] = tuple(declared[kind] if isinstance(declared, dict) else declared)
        return self[kind]


VARIANTS = _Variants()


def components(kind, variant, settings):
    return list(_module(kind).components(variant, settings))


def shared_components(kind, settings):
    module = _module(kind)
    return list(module.shared_components(settings)) if hasattr(module, "shared_components") else []


def fit(kind, variant, component, train, settings, out_dir, reference=None):
    return _module(kind).fit(variant, component, train, settings, out_dir, reference)


def predict(kind, variant, model_dirs, frame, settings, horizons):
    return _module(kind).predict(variant, model_dirs, frame, settings, horizons)
