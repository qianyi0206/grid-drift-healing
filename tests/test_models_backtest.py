import numpy as np
import pandas as pd

from grid_drift_healing.backtest import main, prepare, run_backtest
from grid_drift_healing.features import build_features, pv_power
from grid_drift_healing.healing import HealingConfig
from grid_drift_healing.models import GBMQuantile, SeasonalNaive

FAST = {"max_iter": 40}


def test_pv_power_bounds(syn_weather):
    p = pv_power(syn_weather, capacity_kw=1000)
    assert p.min() >= 0 and p.max() <= 1000
    night = syn_weather["shortwave_radiation"] == 0
    assert (p[night] == 0).all()


def test_features_no_lookahead(syn_weather):
    df = syn_weather.assign(target=pv_power(syn_weather))
    X = build_features(df)
    t = df.index[500]
    assert X.loc[t, "lag_24"] == df["target"].iloc[476]
    assert X["lag_168"].iloc[:168].isna().all()


def test_models_quantiles_ordered(syn_weather):
    X, y = prepare(syn_weather)
    tr, te = slice(0, 3000), slice(3000, 3500)
    for m in (SeasonalNaive(), GBMQuantile(**FAST)):
        P = m.fit(X.iloc[tr], y.iloc[tr]).predict_quantiles(X.iloc[te])
        assert (P[0.1] <= P[0.5] + 1e-9).all() and (P[0.5] <= P[0.9] + 1e-9).all()


def test_gbm_beats_naive(syn_weather):
    X, y = prepare(syn_weather)
    rep = run_backtest(X, y, train_days=200, max_days=40, cfg=HealingConfig(model_kwargs=FAST))
    m = rep["metrics"]
    assert m["static_gbm"]["mean_pinball"] < m["seasonal_naive"]["mean_pinball"]


def test_healing_recovers_from_injected_drift():
    """SYNTHETIC scenario: 25% derate after day 230. Healing should promote at least once
    and beat the never-retrained static model."""
    from grid_drift_healing.ingestion import synthetic_weather
    w = synthetic_weather("2024-01-01", "2024-12-31", seed=5).df
    X, y = prepare(w, inject_drift="2024-08-18:0.75")
    cfg = HealingConfig(model_kwargs=FAST, recent_days=60, holdout_days=10, cooldown_days=5)
    rep = run_backtest(X, y, train_days=180, retrain_every=10_000, cfg=cfg)
    m = rep["metrics"]
    assert m["healing_gbm"]["retrains"] >= 1
    assert m["healing_gbm"]["mean_pinball"] < m["static_gbm"]["mean_pinball"]


def test_cli_offline(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = main(["--offline", "--start", "2024-01-01", "--end", "2024-09-30", "--train-days", "200",
               "--max-days", "20", "--out", str(out)])
    assert rc == 0 and out.exists()
    txt = capsys.readouterr().out
    assert "SYNTHETIC DATA, NOT REAL" in txt


def test_irradiance_not_a_feature(syn_weather):
    X, _ = prepare(syn_weather)
    assert "shortwave_radiation" not in X.columns
    assert "clear_sky_ghi" in X.columns
