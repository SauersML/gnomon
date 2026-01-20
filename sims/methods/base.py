"""
Base class for PGS calibration methods.
"""
from abc import ABC, abstractmethod
import numpy as np


class PGSMethod(ABC):
    """Abstract base class for PGS calibration methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> 'PGSMethod':
        """
        Fit the method on training data.
        
        Parameters
        ----------
        P : np.ndarray, shape (n_samples,)
            Raw polygenic scores
        PC : np.ndarray, shape (n_samples, n_pcs)
            Principal components (ancestry)
        y : np.ndarray, shape (n_samples,)
            Binary phenotype (0 or 1)
            
        Returns
        -------
        self : PGSMethod
            Fitted estimator
        """
        pass
    
    @abstractmethod
    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        """
        Predict probability of outcome.
        
        Parameters
        ----------
        P : np.ndarray, shape (n_samples,)
            Raw polygenic scores
        PC : np.ndarray, shape (n_samples, n_pcs)
            Principal components (ancestry)
            
        Returns
        -------
        proba : np.ndarray, shape (n_samples,)
            Predicted probabilities for class 1
        """
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
