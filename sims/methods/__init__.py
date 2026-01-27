"""Method implementations for PGS calibration."""
from .base import PGSMethod
from .raw_pgs import RawPGSMethod
from .linear_interaction import LinearInteractionMethod
from .normalization import NormalizationMethod

__all__ = [
    'PGSMethod',
    'RawPGSMethod',
    'LinearInteractionMethod',
    'NormalizationMethod',
]

try:
    from .gam_mgcv import GAMMethod
    __all__.append('GAMMethod')
except ImportError:
    pass
