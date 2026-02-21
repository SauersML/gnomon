"""
Method 2: Linear P×PC interaction.

Logistic regression with linear interactions: 
logit(P(Y=1)) = β₀ + β₁·P + Σᵢ(β₂ᵢ·PCᵢ + β₃ᵢ·P×PCᵢ)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .base import PGSMethod


class LinearInteractionMethod(PGSMethod):
    """Linear PGS×PC interaction model."""
    
    def __init__(self, **kwargs):
        super().__init__(name="Linear Interaction")
        self.model = LogisticRegression(**kwargs)
    
    def _create_features(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        """
        Create feature matrix with linear interactions.
        
        Features: [P, PC₁, PC₂, ..., P×PC₁, P×PC₂, ...]
        """
        P = P.reshape(-1, 1)
        
        # Interaction terms: P × each PC
        interactions = P * PC
        
        # Concatenate: [P, PC₁, PC₂, ..., P×PC₁, P×PC₂, ...]
        X = np.hstack([P, PC, interactions])
        return X
    
    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> 'LinearInteractionMethod':
        """Fit logistic regression with P×PC interactions."""
        X = self._create_features(P, PC)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        """Predict probability using fitted model."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        X = self._create_features(P, PC)
        return self.model.predict_proba(X)[:, 1]
