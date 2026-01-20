"""
Method 1: Raw PGS baseline.

Simple logistic regression: logit(P(Y=1)) = β₀ + β₁·P
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .base import PGSMethod


class RawPGSMethod(PGSMethod):
    """Baseline method using raw PGS only, ignoring ancestry."""
    
    def __init__(self, **kwargs):
        super().__init__(name="Raw PGS")
        self.model = LogisticRegression(**kwargs)
    
    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> 'RawPGSMethod':
        """Fit logistic regression on P only."""
        X = P.reshape(-1, 1)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        """Predict probability using fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = P.reshape(-1, 1)
        # Return probability of class 1
        return self.model.predict_proba(X)[:, 1]
