"""
Evaluation metrics for PGS calibration methods.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Optional, Dict, Tuple


def compute_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    population_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute AUC overall and by population.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    population_labels : np.ndarray, optional
        Population labels for stratified analysis
        
    Returns
    -------
    aucs : Dict[str, float]
        Dictionary with 'overall' and population-specific AUCs
    """
    results = {}
    
    # Overall AUC
    results['overall'] = roc_auc_score(y_true, y_pred_proba)
    
    # Population-specific AUC
    if population_labels is not None:
        for pop in np.unique(population_labels):
            mask = population_labels == pop
            if mask.sum() > 0 and len(np.unique(y_true[mask])) > 1:
                results[str(pop)] = roc_auc_score(y_true[mask], y_pred_proba[mask])
    
    return results


def compute_brier_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    population_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute Brier score (MSE of probabilities) overall and by population.
    
    Lower is better.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    population_labels : np.ndarray, optional
        Population labels for stratified analysis
        
    Returns
    -------
    brier_scores : Dict[str, float]
        Dictionary with 'overall' and population-specific Brier scores
    """
    results = {}
    
    # Overall Brier score
    results['overall'] = brier_score_loss(y_true, y_pred_proba)
    
    # Population-specific Brier score
    if population_labels is not None:
        for pop in np.unique(population_labels):
            mask = population_labels == pop
            if mask.sum() > 0:
                results[str(pop)] = brier_score_loss(y_true[mask], y_pred_proba[mask])
    
    return results


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve (binned predicted vs observed).
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred_proba : np.ndarray
        Predicted probabilities
    n_bins : int
        Number of bins
        
    Returns
    -------
    bin_edges : np.ndarray
        Bin edges
    observed_freq : np.ndarray
        Observed frequency in each bin
    predicted_mean : np.ndarray
        Mean predicted probability in each bin
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    observed_freq = np.zeros(n_bins)
    predicted_mean = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
        
        if mask.sum() > 0:
            observed_freq[i] = y_true[mask].mean()
            predicted_mean[i] = y_pred_proba[mask].mean()
        else:
            observed_freq[i] = np.nan
            predicted_mean[i] = bin_centers[i]
    
    return bin_edges, observed_freq, predicted_mean


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    population_labels: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute all evaluation metrics.
    
    Returns
    -------
    metrics : Dict
        Dictionary containing AUC, Brier score, and calibration info
    """
    return {
        'auc': compute_auc(y_true, y_pred_proba, population_labels),
        'brier': compute_brier_score(y_true, y_pred_proba, population_labels),
    }
