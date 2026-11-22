"""Drift detection: PSI covariate drift and rolling pinball-loss degradation."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins from the reference sample.

    Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 significant shift.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class PSIDriftDetector:
    """Covariate drift on selected columns vs. the incumbent's training window."""

    columns: list[str]
    threshold: float = 0.25
    bins: int = 10
    reference: pd.DataFrame | None = None

    def set_reference(self, X: pd.DataFrame) -> None:
        self.reference = X[self.columns].copy()

    def score(self, X_recent: pd.DataFrame) -> dict[str, float]:
        if self.reference is None:
            raise RuntimeError("reference not set")
        return {c: psi(self.reference[c].to_numpy(), X_recent[c].to_numpy(), self.bins) for c in self.columns}

    def check(self, X_recent: pd.DataFrame) -> tuple[bool, dict[str, float]]:
        s = self.score(X_recent)
        return any(v > self.threshold for v in s.values() if np.isfinite(v)), s


@dataclass
class PinballDegradationDetector:
    """Fires when the rolling loss statistic exceeds ``ratio`` x the baseline established on
    the incumbent's validation data.

    If reference losses (e.g. a persistence model's) are passed to ``update``, the statistic is
    the *relative* loss sum(model)/sum(reference), which removes seasonal changes in difficulty
    (summer PV errors are naturally larger than winter ones). Otherwise it is the mean loss.
    """

    window: int = 14 * 24
    ratio: float = 1.25
    baseline: float | None = None
    _buf: deque = field(default_factory=deque)
    _ref: deque = field(default_factory=deque)

    def reset(self, baseline: float) -> None:
        self.baseline = float(baseline)
        self._buf = deque(maxlen=self.window)
        self._ref = deque(maxlen=self.window)

    def update(self, losses: np.ndarray, ref_losses: np.ndarray | None = None) -> None:
        if self._buf.maxlen != self.window:
            self._buf = deque(self._buf, maxlen=self.window)
            self._ref = deque(self._ref, maxlen=self.window)
        self._buf.extend(np.asarray(losses, dtype=float).tolist())
        if ref_losses is not None:
            self._ref.extend(np.asarray(ref_losses, dtype=float).tolist())

    @property
    def rolling(self) -> float:
        if not self._buf:
            return float("nan")
        if len(self._ref) == len(self._buf):
            return float(np.sum(self._buf) / max(np.sum(self._ref), 1e-9))
        return float(np.mean(self._buf))

    def check(self) -> bool:
        if self.baseline is None or len(self._buf) < self.window:
            return False
        return self.rolling > self.ratio * max(self.baseline, 1e-9)
