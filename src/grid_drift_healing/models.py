"""Baseline forecasters. All expose fit(X, y) and predict_quantiles(X) -> {q: array}."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

QUANTILES = (0.1, 0.5, 0.9)


class SeasonalNaive:
    """y_hat(t) = y(t - season). Quantiles from empirical residuals of the training window."""

    name = "seasonal_naive"

    def __init__(self, lag_col: str = "lag_24", quantiles=QUANTILES):
        self.lag_col = lag_col
        self.quantiles = tuple(quantiles)
        self.offsets: dict[float, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        resid = np.asarray(y) - X[self.lag_col].to_numpy()
        self.offsets = {q: float(np.quantile(resid, q)) for q in self.quantiles}
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        base = X[self.lag_col].to_numpy()
        return {q: np.clip(base + self.offsets.get(q, 0.0), 0, None) for q in self.quantiles}


class GBMQuantile:
    """One HistGradientBoostingRegressor per quantile (loss='quantile'), monotone-sorted output."""

    name = "gbm_quantile"

    def __init__(self, quantiles=QUANTILES, max_iter: int = 200, learning_rate: float = 0.08,
                 max_leaf_nodes: int = 31, random_state: int = 0):
        self.quantiles = tuple(quantiles)
        self.kw = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                       random_state=random_state)
        self.models: dict[float, HistGradientBoostingRegressor] = {}
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GBMQuantile":
        self.columns = list(X.columns)
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **self.kw)
            m.fit(X.to_numpy(), np.asarray(y))
            self.models[q] = m
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        Xn = X[self.columns].to_numpy()
        raw = np.vstack([np.clip(self.models[q].predict(Xn), 0, None) for q in self.quantiles])
        raw.sort(axis=0)  # avoid quantile crossing
        return {q: raw[i] for i, q in enumerate(self.quantiles)}


def make_model(name: str, **kw):
    if name == "seasonal_naive":
        return SeasonalNaive(**kw)
    if name == "gbm_quantile":
        return GBMQuantile(**kw)
    raise ValueError(f"unknown model {name!r}")
