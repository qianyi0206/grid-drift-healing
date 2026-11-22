"""Self-healing loop: monitor -> trigger -> retrain candidate -> DM gate -> promote/reject."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .drift import PinballDegradationDetector, PSIDriftDetector
from .gate import promote
from .metrics import mean_pinball, pinball
from .models import make_model

log = logging.getLogger(__name__)


@dataclass
class HealingConfig:
    model: str = "gbm_quantile"
    recent_days: int = 365         # candidate training window (full year keeps seasons covered)
    holdout_days: int = 14         # gate evaluation window (incumbent's genuine live losses)
    baseline_days: int = 14        # validation slice for the degradation baseline
    cooldown_days: int = 7         # min days between healing attempts
    psi_threshold: float = 0.25    # floor; the live threshold self-calibrates (see below)
    psi_window_days: int = 28
    psi_calib_quantile: float = 0.95
    psi_warmup: int = 8            # PSI checks needed before PSI may trigger on its own
    psi_columns: tuple[str, ...] = ("temperature_2m", "cloud_cover", "relative_humidity_2m")
    degradation_ratio: float = 1.25
    degradation_window_days: int = 14
    alpha: float = 0.05
    dm_lag: int = 24
    model_kwargs: dict = field(default_factory=dict)


@dataclass
class HealingEvent:
    time: pd.Timestamp
    trigger: str
    psi_max: float
    rolling_loss: float
    baseline_loss: float
    dm_stat: float
    p_value: float
    promoted: bool


def seasonal_reference(X_train: pd.DataFrame, recent_index: pd.DatetimeIndex, half_width_days: int = 21) -> pd.DataFrame:
    """Rows of the training window with day-of-year near the recent window (removes seasonality
    from the PSI comparison). Falls back to the whole training window if too few rows."""
    centre = int(np.median(recent_index.day_of_year))
    doy = X_train.index.day_of_year.values
    dist = np.minimum(np.abs(doy - centre), 366 - np.abs(doy - centre))
    ref = X_train[dist <= half_width_days]
    return ref if len(ref) >= 24 * 14 else X_train


def persistence_loss(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Absolute error of the 24h-persistence forecast (= 2 x median pinball); seasonal reference."""
    return 2 * pinball(y.to_numpy(), X["lag_24"].to_numpy(), 0.5)


def relative_loss(model_loss: np.ndarray, X: pd.DataFrame, y: pd.Series) -> float:
    return float(np.sum(model_loss) / max(np.sum(persistence_loss(X, y)), 1e-9))


class HealingController:
    def __init__(self, X: pd.DataFrame, y: pd.Series, cfg: HealingConfig | None = None):
        self.X, self.y = X, y
        self.cfg = cfg or HealingConfig()
        self.psi = PSIDriftDetector(list(self.cfg.psi_columns), threshold=self.cfg.psi_threshold)
        self.deg = PinballDegradationDetector(window=self.cfg.degradation_window_days * 24,
                                              ratio=self.cfg.degradation_ratio)
        self.events: list[HealingEvent] = []
        self.live_losses = pd.Series(dtype=float)
        self.incumbent = None
        self.train_slice: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self.last_attempt: pd.Timestamp | None = None
        self.n_promotions = 0
        self.psi_history: list[float] = []

    # -- training helpers ---------------------------------------------------
    def _fit(self, start, end):
        mask = (self.X.index >= start) & (self.X.index < end)
        m = make_model(self.cfg.model, **self.cfg.model_kwargs)
        return m.fit(self.X[mask], self.y[mask])

    def _deploy(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        """Fit on [start, end - baseline) to get an out-of-sample baseline, then refit on [start, end)."""
        b0 = end - pd.Timedelta(days=self.cfg.baseline_days)
        probe = self._fit(start, b0)
        vmask = (self.X.index >= b0) & (self.X.index < end)
        Xv, yv = self.X[vmask], self.y[vmask]
        base = relative_loss(mean_pinball(yv.to_numpy(), probe.predict_quantiles(Xv)), Xv, yv)
        self.incumbent = self._fit(start, end)
        self.train_slice = (start, end)
        self.deg.reset(base)
        tmask = (self.X.index >= start) & (self.X.index < end)
        self._X_train = self.X[tmask]

    def initial_fit(self, start: pd.Timestamp, end: pd.Timestamp) -> None:
        self._deploy(start, end)

    # -- online step ----------------------------------------------------------
    def predict(self, X_day: pd.DataFrame) -> dict[float, np.ndarray]:
        return self.incumbent.predict_quantiles(X_day)

    def observe(self, X_day: pd.DataFrame, y_day: pd.Series, preds: dict[float, np.ndarray]) -> HealingEvent | None:
        losses = mean_pinball(y_day.to_numpy(), preds)
        self.live_losses = pd.concat([self.live_losses, pd.Series(losses, index=X_day.index)])
        self.deg.update(losses, persistence_loss(X_day, y_day))
        now = X_day.index[-1] + pd.Timedelta(hours=1)  # end of observed day
        if self.last_attempt is not None and now - self.last_attempt < pd.Timedelta(days=self.cfg.cooldown_days):
            return None
        if len(self.live_losses) < self.cfg.holdout_days * 24:
            return None

        recent = self.X[(self.X.index >= now - pd.Timedelta(days=self.cfg.psi_window_days)) & (self.X.index < now)]
        self.psi.set_reference(seasonal_reference(self._X_train, recent.index))
        psi_max = max(self.psi.score(recent).values())
        psi_flag = psi_max > self.psi_threshold
        self.psi_history.append(psi_max)
        deg_flag = self.deg.check()
        if not (psi_flag or deg_flag):
            return None
        trigger = "+".join(t for t, f in (("psi", psi_flag), ("pinball", deg_flag)) if f)
        return self._attempt_heal(now, trigger, psi_max)

    @property
    def psi_threshold(self) -> float:
        """Self-calibrating threshold. Weather is strongly autocorrelated, so the textbook
        PSI>0.25 rule fires on ordinary synoptic variability (measured null q95 ~0.5-1.8 on
        Hefei data). We flag only scores above the q95 of this site's own PSI history,
        never below the configured floor, and not at all during warm-up."""
        h = self.psi_history
        if len(h) < self.cfg.psi_warmup:
            return float("inf")
        return max(self.cfg.psi_threshold, float(np.quantile(h, self.cfg.psi_calib_quantile)))

    def _attempt_heal(self, now: pd.Timestamp, trigger: str, psi_max: float) -> HealingEvent:
        cfg = self.cfg
        self.last_attempt = now
        h0 = now - pd.Timedelta(days=cfg.holdout_days)
        cand = self._fit(h0 - pd.Timedelta(days=cfg.recent_days), h0)
        hmask = (self.X.index >= h0) & (self.X.index < now)
        Xh, yh = self.X[hmask], self.y[hmask]
        inc_loss = self.live_losses.reindex(Xh.index).to_numpy()  # genuine live losses
        cand_loss = mean_pinball(yh.to_numpy(), cand.predict_quantiles(Xh))
        ok, res = promote(inc_loss, cand_loss, alpha=cfg.alpha, h=cfg.dm_lag)
        ev = HealingEvent(now, trigger, float(psi_max), self.deg.rolling, float(self.deg.baseline or np.nan),
                          res.stat, res.p_value, bool(ok))
        self.events.append(ev)
        if ok:
            self.n_promotions += 1
            self._deploy(now - pd.Timedelta(days=cfg.recent_days), now)
        log.info("heal attempt %s trigger=%s dm=%.2f p=%.4f promoted=%s", now, trigger, res.stat, res.p_value, ok)
        return ev
