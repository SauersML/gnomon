"""
Method 4: GAM with tensor product splines using R's mgcv.

Implements the model from the paper:
    η_j = Σ_{m=0}^{p} α_m(PC_j) · B_m(P_j)
    
Where:
    α_m(PC_j) = γ_{m0} + Σ_{l=1}^{k} f_{ml}(PC_{jl})

This decomposes into:
    1. Ancestry-specific baseline: γ_{00} + Σ f_{0l}(PC_l)
    2. Main PGS effects: Σ γ_{m0} · B_m(P)
    3. Non-linear P×PC interactions: Σ Σ f_{ml}(PC_l) · B_m(P)

In mgcv syntax:
    y ~ s(P, bs='cr') + s(PC1, bs='cr') + s(PC2, bs='cr') + ti(P, PC1) + ti(P, PC2)
    
Or equivalently with tensor product:
    y ~ te(P, PC1, PC2, bs=c('cr','cr','cr'))
"""
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from .base import PGSMethod


class GAMMethod(PGSMethod):
    """
    GAM with tensor product splines via R's mgcv package.
    
    Implements the phenotype-informed calibration model with:
    - Main effect of PGS: s(P)
    - Main effects of ancestry PCs: s(PC1), s(PC2), ...
    - Tensor product interactions: ti(P, PC1), ti(P, PC2), ...
    
    The model learns how the PGS-phenotype relationship varies
    across the ancestry spectrum, using REML for automatic
    smoothing parameter selection.
    """
    
    def __init__(
        self,
        k_pgs: int = 10,
        k_pc: int = 10,
        k_interaction: int = 5,
        method: str = 'REML',
        use_ti: bool = True,
    ):
        """
        Parameters
        ----------
        k_pgs : int  
            Basis dimension for PGS spline (default 10)
        k_pc : int
            Basis dimension for PC splines (default 10)
        k_interaction : int
            Basis dimension for interaction terms (default 5)
        method : str
            Smoothing parameter estimation method ('REML', 'GCV.Cp', 'ML')
        use_ti : bool
            If True, use decomposed form s() + ti() for clearer interpretation.
            If False, use single te() tensor product.
        """
        self.n_pcs = 3
        super().__init__(name=f"GAM (mgcv, {self.n_pcs} PCs)")
        self.k_pgs = k_pgs
        self.k_pc = k_pc
        self.k_interaction = k_interaction
        self.method = method
        self.use_ti = use_ti
        self.r_model = None
        
        # Import R packages
        self.mgcv = importr('mgcv')
        self.stats = importr('stats')
    
    def _build_formula(self, n_pcs_actual: int) -> str:
        """
        Build the mgcv formula string.
        
        Uses decomposed form for interpretability:
            y ~ s(P) + s(PC1) + s(PC2) + ... + ti(P, PC1) + ti(P, PC2) + ...
            
        This separates:
        - Main effect of PGS: s(P)
        - Main effects of ancestry: s(PC1), s(PC2), ...
        - Pure interactions (effect modification): ti(P, PC1), ti(P, PC2), ...
        """
        if self.use_ti:
            # Decomposed form: main effects + tensor product interactions
            terms = []
            
            # Main effect of PGS with cubic regression spline
            terms.append(f's(P, k={self.k_pgs}, bs="cr")')
            
            # Main effects of each PC
            for i in range(n_pcs_actual):
                terms.append(f's(PC{i+1}, k={self.k_pc}, bs="cr")')
            
            # Tensor product interactions between P and each PC
            # ti() gives the pure interaction (excludes main effects)
            for i in range(n_pcs_actual):
                terms.append(f'ti(P, PC{i+1}, k=c({self.k_interaction},{self.k_interaction}), bs=c("cr","cr"))')
            
            formula_str = 'y ~ ' + ' + '.join(terms)
        else:
            # Single tensor product (includes main effects implicitly)
            pc_vars = ', '.join([f'PC{i+1}' for i in range(n_pcs_actual)])
            k_dims = ', '.join([str(self.k_pgs)] + [str(self.k_pc)] * n_pcs_actual)
            bs_types = ', '.join(['"cr"'] * (1 + n_pcs_actual))
            formula_str = f'y ~ te(P, {pc_vars}, k=c({k_dims}), bs=c({bs_types}))'
        
        return formula_str

    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> 'GAMMethod':
        """
        Fit GAM using R's mgcv with proper tensor product structure.
        
        Uses REML for smoothing parameter estimation, which provides
        a good balance between fit and smoothness.
        """
        # Prepare data frame for R
        data_dict = {'y': y.astype(float), 'P': P.astype(float)}
        
        if PC.shape[1] < self.n_pcs:
            raise RuntimeError(
                f"GAM requires at least {self.n_pcs} PCs, got {PC.shape[1]}."
            )
        n_pcs_actual = self.n_pcs
        for i in range(n_pcs_actual):
            data_dict[f'PC{i+1}'] = PC[:, i].astype(float)
        
        # Convert to R data frame
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            r_df = ro.DataFrame(data_dict)

        # Build formula
        formula_str = self._build_formula(n_pcs_actual)
        formula = ro.Formula(formula_str)
        self.r_model = self.mgcv.gam(
            formula,
            data=r_df,
            family=self.stats.binomial(),
            method=self.method,
        )

        self.is_fitted = True
        self._formula_str = formula_str
        return self
    
    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        """Predict probabilities using fitted GAM."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Prepare new data
        data_dict = {'P': P.astype(float)}
        if PC.shape[1] < self.n_pcs:
            raise RuntimeError(
                f"GAM requires at least {self.n_pcs} PCs, got {PC.shape[1]}."
            )
        n_pcs_actual = self.n_pcs
        for i in range(n_pcs_actual):
            data_dict[f'PC{i+1}'] = PC[:, i].astype(float)
        
        # Predict
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            r_newdata = ro.DataFrame(data_dict)
            
            predictions = ro.r['predict'](
                self.r_model,
                newdata=r_newdata,
                type='response'
            )
            
            return np.array(predictions)
    
    def get_summary(self) -> str:
        """Get R summary of the fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting summary")
        
        summary_func = ro.r['summary']
        capture_output = ro.r['capture.output']
        summary_text = capture_output(summary_func(self.r_model))
        return '\n'.join(list(summary_text))
    
    def get_edf(self) -> dict:
        """
        Get effective degrees of freedom for each smooth term.
        
        Higher EDF indicates more non-linear relationship.
        EDF close to 1 suggests linear relationship.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted")
        
        # Extract smooth term names and EDFs
        smooth_names = list(ro.r['names'](self.r_model.rx2('smooth')))
        edfs = np.array(self.r_model.rx2('edf'))
        
        return dict(zip(smooth_names, edfs))
