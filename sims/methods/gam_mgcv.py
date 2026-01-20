"""
Method 4: GAM with tensor product splines using R's mgcv.

Implements: gam(y ~ te(P, PC1, PC2, ...), family=binomial())
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
    
    Uses rpy2 to fit: gam(y ~ te(P, PC1, PC2, ...), family=binomial())
    """
    
    def __init__(self, n_pcs: int = 2):
        """
        Parameters
        ----------
        n_pcs : int
            Number of PCs to include in tensor product
        """
        super().__init__(name=f"GAM (mgcv, {n_pcs} PCs)")
        self.n_pcs = n_pcs
        self.r_model = None
        
        # Import R packages (will fail fast if R or mgcv not available)
        self.mgcv = importr('mgcv')
        self.stats = importr('stats')
    
    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> 'GAMMethod':
        """Fit GAM using R's mgcv."""
        # Prepare data frame for R
        n_samples = len(P)
        data_dict = {'y': y, 'P': P}
        
        # Add PCs
        for i in range(min(self.n_pcs, PC.shape[1])):
            data_dict[f'PC{i+1}'] = PC[:, i]
        
        # Convert to R data frame using conversion context
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            r_df = ro.DataFrame(data_dict)
        
        # Build formula string
        pc_vars = ', '.join([f'PC{i+1}' for i in range(min(self.n_pcs, PC.shape[1]))])
        formula_str = f'y ~ te(P, {pc_vars})'
        
        # Fit GAM
        formula = ro.Formula(formula_str)
        self.r_model = self.mgcv.gam(
            formula,
            data=r_df,
            family=self.stats.binomial()
        )
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        """Predict probabilities using fitted GAM."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Prepare new data for prediction
        data_dict = {'P': P}
        for i in range(min(self.n_pcs, PC.shape[1])):
            data_dict[f'PC{i+1}'] = PC[:, i]
        
        # Convert and predict using conversion context
        with localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            r_newdata = ro.DataFrame(data_dict)
            
            # Predict on response scale (probabilities)
            predictions = self.stats.predict_glm(
                self.r_model,
                newdata=r_newdata,
                type='response'
            )
            
            # Convert R vector to numpy array
            return np.array(predictions)
