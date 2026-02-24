"""
Method 5: Thin-plate spline GAM using R's mgcv.
"""
import numpy as np
from .gam_mgcv import GAMMethod


class ThinPlateMethod(GAMMethod):
    """
    Thin-plate spline variant of the mgcv GAM.

    Uses mgcv's `bs="tp"` basis, which is constructed from unique covariate
    locations in the observed data (no fixed knot grid), and always enables
    double-penalty term selection (`select=TRUE`).
    """

    def __init__(
        self,
        k_pgs: int = 10,
        k_pc: int = 10,
        k_interaction: int = 5,
        method: str = "REML",
        use_ti: bool = True,
    ):
        super().__init__(
            k_pgs=k_pgs,
            k_pc=k_pc,
            k_interaction=k_interaction,
            method=method,
            use_ti=use_ti,
            use_double_penalty=True,
        )
        self.name = f"Thin-plate spline (mgcv, {self.n_pcs} PCs)"

    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> "ThinPlateMethod":
        # Guard against accidental external mutation of this setting.
        self.use_double_penalty = True
        super().fit(P, PC, y)
        return self

    def _build_formula(self, n_pcs_actual: int) -> str:
        if self.use_ti:
            terms = []
            terms.append(f's(P, k={self.k_pgs}, bs="tp")')
            for i in range(n_pcs_actual):
                terms.append(f's(PC{i+1}, k={self.k_pc}, bs="tp")')
            for i in range(n_pcs_actual):
                terms.append(
                    f'ti(P, PC{i+1}, k=c({self.k_interaction},{self.k_interaction}), bs=c("tp","tp"))'
                )
            return "y ~ " + " + ".join(terms)

        pc_vars = ", ".join([f"PC{i+1}" for i in range(n_pcs_actual)])
        k_dims = ", ".join([str(self.k_pgs)] + [str(self.k_pc)] * n_pcs_actual)
        bs_types = ", ".join(['"tp"'] * (1 + n_pcs_actual))
        return f"y ~ te(P, {pc_vars}, k=c({k_dims}), bs=c({bs_types}))"
