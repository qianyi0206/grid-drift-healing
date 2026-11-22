"""Rolling-origin backtest harness + CLI.

Example (Hefei):
    python -m grid_drift_healing.backtest --lat 31.82 --lon 117.23 \
        --start 2024-01-01 --end 2026-10-31 --train-days 365
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .features import build_features, pv_power
from .gate import diebold_mariano
from .healing import HealingConfig, HealingController
from .ingestion import load_weather
from .metrics import crps_from_quantiles, interval_coverage, mae, mean_pinball, rmse
from .models import make_model

log = logging.getLogger("grid_drift_healing.backtest")


def prepare(weather: pd.DataFrame, capacity_kw: float = 1000.0, inject_drift: str | None = None,
            lat: float = 31.82, lon: float = 117.23):
    df = weather.copy()
    df["target"] = pv_power(df, capacity_kw=capacity_kw)
    if inject_drift:
        date, factor = inject_drift.split(":")
        mask = df.index >= pd.Timestamp(date, tz="UTC")
        df.loc[mask, "target"] *= float(factor)
    X = build_features(df, lat=lat, lon=lon)
    valid = X.notna().all(axis=1)
    return X[valid], df["target"][valid]


def run_backtest(X: pd.DataFrame, y: pd.Series, train_days: int = 365, retrain_every: int = 30,
                 cfg: HealingConfig | None = None, max_days: int | None = None) -> dict:
    cfg = cfg or HealingConfig()
    t0 = X.index[0].normalize() + pd.Timedelta(days=1)
    test_start = t0 + pd.Timedelta(days=train_days)
    days = pd.date_range(test_start, X.index[-1].normalize() - pd.Timedelta(days=1), freq="D")
    if max_days:
        days = days[:max_days]
    if len(days) == 0:
        raise ValueError("not enough data for the requested training window")

    def fit(name, start, end):
        m = (X.index >= start) & (X.index < end)
        kw = cfg.model_kwargs if name == "gbm_quantile" else {}
        return make_model(name, **kw).fit(X[m], y[m])

    naive = fit("seasonal_naive", t0, test_start)
    static = fit(cfg.model, t0, test_start)
    periodic = static
    n_periodic = 0
    healer = HealingController(X, y, cfg)
    healer.initial_fit(t0, test_start)

    rec = {k: {"loss": [], "q": {0.1: [], 0.5: [], 0.9: []}} for k in ("seasonal_naive", "static_gbm", "periodic_gbm", "healing_gbm")}
    ys, idx = [], []
    for i, day in enumerate(days):
        m = (X.index >= day) & (X.index < day + pd.Timedelta(days=1))
        Xd, yd = X[m], y[m]
        if len(Xd) == 0:
            continue
        if i > 0 and i % retrain_every == 0:
            periodic = fit(cfg.model, day - pd.Timedelta(days=train_days), day)
            n_periodic += 1
        hp = healer.predict(Xd)
        preds = {"seasonal_naive": naive.predict_quantiles(Xd), "static_gbm": static.predict_quantiles(Xd),
                 "periodic_gbm": periodic.predict_quantiles(Xd), "healing_gbm": hp}
        for k, p in preds.items():
            for q in rec[k]["q"]:
                rec[k]["q"][q].append(p[q])
        ys.append(yd.to_numpy())
        idx.append(Xd.index)
        healer.observe(Xd, yd, hp)

    yall = np.concatenate(ys)
    results = {}
    losses = {}
    for k, r in rec.items():
        P = {q: np.concatenate(v) for q, v in r["q"].items()}
        losses[k] = mean_pinball(yall, P)
        results[k] = {
            "mean_pinball": float(losses[k].mean()),
            "crps_approx": float(crps_from_quantiles(yall, P).mean()),
            "mae_p50": mae(yall, P[0.5]),
            "rmse_p50": rmse(yall, P[0.5]),
            "coverage_p10_p90": interval_coverage(yall, P[0.1], P[0.9]),
        }
    results["periodic_gbm"]["retrains"] = n_periodic
    results["healing_gbm"]["retrains"] = healer.n_promotions
    results["healing_gbm"]["heal_attempts"] = len(healer.events)
    dm = {}
    for other in ("static_gbm", "periodic_gbm", "seasonal_naive"):
        r = diebold_mariano(losses[other], losses["healing_gbm"], h=24)
        dm[f"healing_vs_{other}"] = {"dm_stat": r.stat, "p_value": r.p_value, "mean_loss_diff": r.mean_diff}
    events = [{**e.__dict__, "time": str(e.time)} for e in healer.events]
    return {"test_start": str(days[0].date()), "test_end": str(days[-1].date()), "n_days": len(days),
            "n_hours": int(len(yall)), "metrics": results, "dm_tests": dm, "healing_events": events}


def format_report(rep: dict, meta: dict) -> str:
    lines = [f"grid-drift-healing {__version__} backtest",
             f"data source: {meta['source']}{'  <<< SYNTHETIC DATA, NOT REAL >>>' if meta['source'] == 'SYNTHETIC' else ''}",
             f"location: lat={meta['lat']} lon={meta['lon']}  raw rows={meta['rows']}  range={meta['range']}",
             f"features: lags>=24h, calendar, clear-sky geometry, cloud/temp/wind/RH (perfect-prognosis); irradiance excluded",
             f"target: virtual {meta['capacity_kw']:.0f} kW PV plant (PVWatts-style proxy from irradiance+temperature)",
             f"injected drift: {meta['inject_drift'] or 'none'}",
             f"test window: {rep['test_start']} .. {rep['test_end']}  ({rep['n_days']} days, {rep['n_hours']} hourly points)",
             "",
             f"{'strategy':<16}{'pinball':>10}{'CRPS~':>10}{'MAE_p50':>10}{'RMSE_p50':>10}{'cov80':>8}{'retrains':>10}"]
    for k, r in rep["metrics"].items():
        lines.append(f"{k:<16}{r['mean_pinball']:>10.3f}{r['crps_approx']:>10.3f}{r['mae_p50']:>10.3f}"
                     f"{r['rmse_p50']:>10.3f}{r['coverage_p10_p90']:>8.3f}{str(r.get('retrains', '-')):>10}")
    lines.append("")
    lines.append("Diebold-Mariano (one-sided, H1: healing has lower pinball loss; HAC lag 24, HLN-corrected):")
    for k, v in rep["dm_tests"].items():
        lines.append(f"  {k:<30} DM={v['dm_stat']:+.3f}  p={v['p_value']:.4g}  mean_diff={v['mean_loss_diff']:+.4f}")
    lines.append("")
    lines.append(f"healing attempts: {len(rep['healing_events'])}  promotions: {rep['metrics']['healing_gbm']['retrains']}")
    for e in rep["healing_events"]:
        lines.append(f"  {e['time'][:10]}  trigger={e['trigger']:<12} psi_max={e['psi_max']:.3f} "
                     f"rel_loss={e['rolling_loss']:.3f} base={e['baseline_loss']:.3f} DM={e['dm_stat']:+.2f} "
                     f"p={e['p_value']:.4f} -> {'PROMOTED' if e['promoted'] else 'rejected'}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Rolling-origin backtest of drift-healing PV forecasting")
    p.add_argument("--lat", type=float, default=31.82, help="latitude (default: Hefei)")
    p.add_argument("--lon", type=float, default=117.23, help="longitude (default: Hefei)")
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default="2026-10-31")
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--retrain-every", type=int, default=30)
    p.add_argument("--max-days", type=int, default=None, help="limit number of test days")
    p.add_argument("--capacity-kw", type=float, default=1000.0)
    p.add_argument("--inject-drift", default=None,
                   help="SIMULATED target drift 'YYYY-MM-DD:factor', e.g. 2025-07-01:0.8 (soiling/derate)")
    p.add_argument("--offline", action="store_true", help="skip the network and use SYNTHETIC data")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--out", default="reports/backtest.json")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t = time.time()
    wf = load_weather(a.lat, a.lon, a.start, a.end, cache_dir=a.cache_dir, offline=a.offline)
    X, y = prepare(wf.df, a.capacity_kw, a.inject_drift, lat=wf.lat, lon=wf.lon)
    rep = run_backtest(X, y, train_days=a.train_days, retrain_every=a.retrain_every, max_days=a.max_days)
    meta = {"source": wf.source, "lat": wf.lat, "lon": wf.lon, "rows": len(wf.df),
            "range": f"{wf.df.index[0]} .. {wf.df.index[-1]}", "capacity_kw": a.capacity_kw,
            "inject_drift": a.inject_drift, "runtime_s": round(time.time() - t, 1)}
    print(format_report(rep, meta))
    print(f"\nruntime: {meta['runtime_s']} s")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, **rep}, indent=2, default=str))
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
