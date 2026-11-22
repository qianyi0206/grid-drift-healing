"""Diebold-Mariano promotion gate (with Harvey-Leybourne-Newbold small-sample correction)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DMResult:
    stat: float
    p_value: float
    mean_diff: float  # mean(loss_incumbent - loss_candidate); >0 means candidate better
    n: int


def diebold_mariano(loss_incumbent: np.ndarray, loss_candidate: np.ndarray, h: int = 1) -> DMResult:
    """One-sided DM test of H0: E[d]<=0 vs H1: E[d]>0 where d = L_inc - L_cand.

    Uses a Newey-West (Bartlett) long-run variance with h-1 lags and the HLN correction.
    """
    d = np.asarray(loss_incumbent, float) - np.asarray(loss_candidate, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        raise ValueError("need at least 10 paired losses")
    dbar = d.mean()
    dc = d - dbar
    lrv = np.dot(dc, dc) / n
    for k in range(1, max(h, 1)):
        w = 1 - k / h
        lrv += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    if lrv <= 0:
        lrv = np.dot(dc, dc) / n
    if lrv <= 0:
        return DMResult(float("inf") if dbar > 0 else 0.0, 0.0 if dbar > 0 else 1.0, float(dbar), n)
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = float(dm * hln)
    p = float(1 - stats.t.cdf(stat, df=n - 1))
    return DMResult(stat, p, float(dbar), n)


def promote(loss_incumbent, loss_candidate, alpha: float = 0.05, h: int = 24) -> tuple[bool, DMResult]:
    """Promote the candidate only if it is significantly better at level alpha."""
    res = diebold_mariano(loss_incumbent, loss_candidate, h=h)
    return (res.p_value < alpha and res.mean_diff > 0), res
