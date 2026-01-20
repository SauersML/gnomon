"""
Method 3: Phenotype-agnostic normalization using pgscatalog.calc.

Uses standard normalization methods (empirical, mean, mean+var) then fits
logistic regression on the normalized score.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from .base import PGSMethod


class NormalizationMethod(PGSMethod):
    """
    Phenotype-agnostic PGS normalization.
    
    Implements standard normalization approaches:
    - 'empirical': Residuals from P ~ PC regression
    - 'mean': Standardize mean by PC percentiles
    - 'mean+var': Standardize mean and variance by PC percentiles
    """
    
    def __init__(self, method: str = 'empirical', **kwargs):
        """
        Parameters
        ----------
        method : str
            Normalization method: 'empirical', 'mean', or 'mean+var'
        """
        if method not in ['empirical', 'mean', 'mean+var']:
            raise ValueError(f"Unknown normalization method: {method}")
        
        super().__init__(name=f"Normalization ({method})")
        self.norm_method = method
        self.norm_model = None  # For empirical method
        self.model = LogisticRegression(**kwargs)
    
    def _normalize_empirical(self, P: np.ndarray, PC: np.ndarray, fit: bool = True) -> np.ndarray:
        """Empirical normalization: residuals from P ~ PC regression."""
        if fit:
            self.norm_model = LinearRegression()
            self.norm_model.fit(PC, P)
        
        # Normalized score = residuals
        P_normalized = P - self.norm_model.predict(PC)
        return P_normalized
    
    def _normalize_mean(self, P: np.ndarray, PC: np.ndarray, fit: bool = True) -> np.ndarray:
        """Mean normalization: standardize mean along PC axis."""
        # Use PC1 as the main ancestry axis for binning
        pc1 = PC[:, 0]
        
        if fit:
            # Store percentile-based bins from training data
            self.pc_percentiles = np.percentile(pc1, np.linspace(0, 100, 11))
            self.bin_means = []
            
            # Compute mean P in each PC bin
            for i in range(len(self.pc_percentiles) - 1):
                mask = (pc1 >= self.pc_percentiles[i]) & (pc1 < self.pc_percentiles[i + 1])
                if mask.sum() > 0:
                    self.bin_means.append(P[mask].mean())
                else:
                    self.bin_means.append(0.0)
        
        # Normalize: subtract bin-specific mean
        P_normalized = P.copy()
        for i in range(len(self.pc_percentiles) - 1):
            mask = (pc1 >= self.pc_percentiles[i]) & (pc1 < self.pc_percentiles[i + 1])
            P_normalized[mask] -= self.bin_means[i]
        
        return P_normalized
    
    def _normalize_mean_var(self, P: np.ndarray, PC: np.ndarray, fit: bool = True) -> np.ndarray:
        """Mean+Var normalization: standardize both moments."""
        pc1 = PC[:, 0]
        
        if fit:
            self.pc_percentiles = np.percentile(pc1, np.linspace(0, 100, 11))
            self.bin_means = []
            self.bin_stds = []
            
            for i in range(len(self.pc_percentiles) - 1):
                mask = (pc1 >= self.pc_percentiles[i]) & (pc1 < self.pc_percentiles[i + 1])
                if mask.sum() > 1:
                    self.bin_means.append(P[mask].mean())
                    self.bin_stds.append(P[mask].std())
                else:
                    self.bin_means.append(0.0)
                    self.bin_stds.append(1.0)
        
        # Normalize: (P - mean) / std for each bin
        P_normalized = P.copy()
        for i in range(len(self.pc_percentiles) - 1):
            mask = (pc1 >= self.pc_percentiles[i]) & (pc1 < self.pc_percentiles[i + 1])
            P_normalized[mask] = (P[mask] - self.bin_means[i]) / self.bin_stds[i]
        
        return P_normalized
    
    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> 'NormalizationMethod':
        """
        Fit normalization on genotype data, then logistic regression on phenotype.
        
        Note: Normalization uses P and PC only (phenotype-agnostic),
        then the normalized score is used to predict y.
        """
        # Apply normalization (fit=True stores normalization parameters)
        if self.norm_method == 'empirical':
            P_normalized = self._normalize_empirical(P, PC, fit=True)
        elif self.norm_method == 'mean':
            P_normalized = self._normalize_mean(P, PC, fit=True)
        else:  # mean+var
            P_normalized = self._normalize_mean_var(P, PC, fit=True)
        
        # Fit logistic regression on normalized score
        X = P_normalized.reshape(-1, 1)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        """Predict probability using normalized score."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Apply normalization (fit=False uses stored parameters)
        if self.norm_method == 'empirical':
            P_normalized = self._normalize_empirical(P, PC, fit=False)
        elif self.norm_method == 'mean':
            P_normalized = self._normalize_mean(P, PC, fit=False)
        else:  # mean+var
            P_normalized = self._normalize_mean_var(P, PC, fit=False)
        
        X = P_normalized.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]
