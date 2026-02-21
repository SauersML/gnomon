"""
Normalization method using pgscatalog-calc ancestry adjustments (Z_norm2).

This follows the pgscatalog normalization approach: regress PGS on PCs to
remove ancestry-related mean shifts, then use a second regression to normalize
variance (Z_norm2). We then fit a logistic regression on the normalized score.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base import PGSMethod
from pgscatalog.calc.lib._ancestry.tools import pgs_adjust


class NormalizationMethod(PGSMethod):
    def __init__(self, n_pcs: int = 5, **kwargs):
        super().__init__(name="Normalization (Z_norm2)")
        self.n_pcs = n_pcs
        self.model = LogisticRegression(**kwargs)
        self._ref_df = None
        self._score_col = "PGS"
        self._pop_labels = None

    def set_pop_labels(self, pop_labels):
        self._pop_labels = np.asarray(pop_labels)

    def _make_df(self, P: np.ndarray, PC: np.ndarray, pop_labels) -> pd.DataFrame:
        if pop_labels is None:
            raise RuntimeError("Normalization requires pop labels to be set.")
        pc_cols = {f"PC{i+1}": PC[:, i] for i in range(self.n_pcs)}
        df = pd.DataFrame(pc_cols)
        df["pop"] = pop_labels
        df[self._score_col] = P
        return df

    def fit(self, P: np.ndarray, PC: np.ndarray, y: np.ndarray) -> "NormalizationMethod":
        if self._pop_labels is None:
            raise RuntimeError("Call set_pop_labels() before fit for NormalizationMethod.")
        ref_df = self._make_df(P, PC, self._pop_labels)
        self._ref_df = ref_df

        _, target_adj, _ = pgs_adjust(
            ref_df=ref_df,
            target_df=ref_df,
            scorecols=[self._score_col],
            ref_pop_col="pop",
            target_pop_col="pop",
            use_method=["mean+var"],
            norm2_2step=True,
            n_pcs=self.n_pcs,
        )

        z_col = f"Z_norm2|{self._score_col}"
        if z_col not in target_adj.columns:
            raise RuntimeError(f"Normalization output missing required column {z_col}")
        z = target_adj[z_col].to_numpy()
        self.model.fit(z.reshape(-1, 1), y)
        self.is_fitted = True
        return self

    def predict_proba(self, P: np.ndarray, PC: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        if self._pop_labels is None:
            raise RuntimeError("Call set_pop_labels() before predict for NormalizationMethod.")
        target_df = self._make_df(P, PC, self._pop_labels)
        _, target_adj, _ = pgs_adjust(
            ref_df=self._ref_df,
            target_df=target_df,
            scorecols=[self._score_col],
            ref_pop_col="pop",
            target_pop_col="pop",
            use_method=["mean+var"],
            norm2_2step=True,
            n_pcs=self.n_pcs,
        )

        z_col = f"Z_norm2|{self._score_col}"
        if z_col not in target_adj.columns:
            raise RuntimeError(f"Normalization output missing required column {z_col}")
        z = target_adj[z_col].to_numpy()
        return self.model.predict_proba(z.reshape(-1, 1))[:, 1]
