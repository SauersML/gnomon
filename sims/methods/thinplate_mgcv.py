"""
Method 5: Joint Duchon-spline GAM using R's mgcv.
"""
import numpy as np
from .gam_mgcv import GAMMethod


class ThinPlateMethod(GAMMethod):
    """
    Joint Duchon-spline variant of the mgcv GAM.

    This fits one smooth over PGS and the retained PCs together instead of
    separate marginal + interaction terms. The basis uses mgcv's Duchon spline
    (`bs="ds"`) and keeps double-penalty selection enabled.
    """

    def __init__(
        self,
        k_joint: int = 80,
        method: str = "REML",
    ):
        super().__init__(
            k_pgs=k_joint,
            k_pc=k_joint,
            k_interaction=k_joint,
            method=method,
            use_ti=False,
            use_double_penalty=True,
        )
        self.k_joint = int(k_joint)
        self.name = f"Duchon spline (mgcv, {self.n_pcs} PCs)"

    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> "ThinPlateMethod":
        self.use_double_penalty = True
        self.use_ti = False
        super().fit(P, PC, y)
        return self

    def _build_formula(self, n_pcs_actual: int) -> str:
        pc_vars = ", ".join([f"PC{i+1}" for i in range(n_pcs_actual)])
        return f'y ~ s(P, {pc_vars}, bs="ds", k={self.k_joint})'
