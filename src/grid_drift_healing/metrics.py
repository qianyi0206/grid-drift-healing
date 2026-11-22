"""Probabilistic and point forecast metrics."""
from __future__ import annotations

import numpy as np


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> np.ndarray:
    """Elementwise pinball (quantile) loss."""
    d = np.asarray(y) - np.asarray(q_pred)
    return np.maximum(q * d, (q - 1) * d)


def mean_pinball(y: np.ndarray, preds: dict[float, np.ndarray]) -> np.ndarray:
    """Per-observation pinball averaged over quantile levels."""
    return np.mean([pinball(y, p, q) for q, p in preds.items()], axis=0)


def crps_from_quantiles(y: np.ndarray, preds: dict[float, np.ndarray]) -> np.ndarray:
    """Quantile-based CRPS approximation: CRPS ~= 2 * mean_q pinball_q."""
    return 2.0 * mean_pinball(y, preds)


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def interval_coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))
