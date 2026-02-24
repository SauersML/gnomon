"""Method implementations for PGS calibration."""
from .base import PGSMethod
from .raw_pgs import RawPGSMethod
from .linear_interaction import LinearInteractionMethod
from .normalization import NormalizationMethod

GAMMethod = None
ThinPlateMethod = None


def __getattr__(name):
    if name == "GAMMethod":
        from .gam_mgcv import GAMMethod as _GAMMethod
        return _GAMMethod
    if name == "ThinPlateMethod":
        from .thinplate_mgcv import ThinPlateMethod as _ThinPlateMethod
        return _ThinPlateMethod
    raise AttributeError(name)

__all__ = [
    'PGSMethod',
    'RawPGSMethod',
    'LinearInteractionMethod',
    'NormalizationMethod',
    'GAMMethod',
    'ThinPlateMethod',
]
