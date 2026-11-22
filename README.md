# grid-drift-healing

**Early-stage MVP.** This is an autonomous pipeline for probabilistic renewable-energy forecasting. It detects drift and *self-heals*: when the incumbent model degrades, it retrains a candidate. The candidate is promoted only if a Diebold–Mariano test says it is significantly better on the most recent live data.

Default site: Hefei, China (31.82 N, 117.23 E). Data comes from the free [Open-Meteo](https://open-meteo.com) archive API, which needs no API key.

## What works today

| Module | What it does |
|---|---|
| `ingestion.py` | Pulls hourly data from the Open-Meteo archive and validates the schema (required variables, lengths, monotonic time, physical bounds, NaN budget). It caches responses as local CSV. If the network or the schema fails, it falls back to a synthetic generator, labeled `SYNTHETIC` everywhere. |
| `features.py` | Builds the target, a **virtual 1 MW PV plant** computed with a PVWatts-style proxy (irradiance + cell temperature). Features: lags of at least 24 h (valid for day-ahead), calendar terms, clear-sky geometry, and cloud, temperature, wind and RH covariates. Irradiance is *excluded* from the features because the target is derived from it. |
| `models.py` | A seasonal-naive (24 h persistence) model with empirical quantiles, and a gradient-boosting quantile model (P10/P50/P90, crossing-free). |
| `metrics.py` | Pinball loss, a quantile-based CRPS approximation, MAE, RMSE and interval coverage. |
| `drift.py` | PSI covariate drift with a seasonally matched reference, plus a rolling pinball-degradation detector in *relative* mode (model loss / persistence loss). The relative mode keeps seasonal changes in difficulty from triggering it. |
| `gate.py` | A one-sided Diebold–Mariano test with Newey–West HAC variance and the Harvey–Leybourne–Newbold small-sample correction. |
| `healing.py` | The loop: monitor → trigger → retrain a candidate on the recent window → DM gate on the incumbent's genuine live losses → promote or reject, with a cooldown. The PSI threshold self-calibrates (q95 of the site's own PSI history, floor 0.25). |
| `backtest.py` | A rolling-origin, day-ahead backtest CLI comparing seasonal-naive, static GBM, periodic retraining (every 30 days) and DM-gated healing. |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
python -m grid_drift_healing.backtest --lat 31.82 --lon 117.23 --start 2024-01-01 --end 2026-10-31
# simulated 20% derate (e.g. soiling) from 2025-09-01:
python -m grid_drift_healing.backtest --inject-drift 2025-09-01:0.8
# fully offline (synthetic data, clearly labelled):
python -m grid_drift_healing.backtest --offline --start 2024-01-01 --end 2024-12-31 --train-days 200
```

## Results so far (real Open-Meteo data, Hefei)

These are the raw outputs from the `logs/` directory. The table below is copied from `logs/backtest.log`: 661 test days, 15,864 hourly points, 1000 kW virtual plant, losses in kW.

```
strategy           pinball     CRPS~   MAE_p50  RMSE_p50   cov80  retrains
seasonal_naive      18.312    36.623    42.540    95.870   0.791         -
static_gbm           9.477    18.954    21.041    45.146   0.884         -
periodic_gbm         8.092    16.183    20.744    44.499   0.859        22
healing_gbm          7.922    15.844    21.109    45.407   0.841         8

  healing_vs_periodic_gbm        DM=+2.213  p=0.01347  mean_diff=+0.1698
```

What the backtest shows:

- Gated healing used **8 promotions out of 34 attempts**, compared with 22 blind periodic retrains. Its pinball loss was slightly but significantly lower than periodic retraining (p≈0.013) and much lower than the static model.
- The P50 point error (MAE/RMSE) is marginally *worse* than periodic retraining. The gate optimizes the average pinball loss across quantiles, not the median.
- With a simulated 20 % derate (`logs/backtest_injected_drift.log`), healing **ties** periodic retraining (DM p=0.55) and beats the static model. Detection of the step change is slow, because the lag features partly absorb it. This needs work.

## Honest limitations

- The target is a **physics-derived proxy**, not metered plant output. There is no public PV telemetry for Hefei in this MVP.
- The weather covariates are reanalysis values used as **perfect-prognosis** stand-ins for NWP forecasts. Real day-ahead errors will be larger.
- The PSI trigger is still noisy: many attempts are rejected by the gate. That is the gate doing its job, but it wastes compute.
- The backtest covers a single site with a single model family.

## Roadmap

- [ ] ENTSO-E Transparency adapter (load and generation actuals; needs an API token). **TODO, not implemented**
- [ ] EIA Open Data adapter (US balancing-authority demand). **TODO, not implemented**
- [ ] Open-Meteo *historical forecast* API, so the backtest uses real NWP forecasts instead of perfect prognosis
- [ ] Real plant or SCADA target via the same `target` column
- [ ] Recency-weighted candidates and faster detection of step changes (CUSUM / Page–Hinkley)
- [ ] Model registry, plus a scheduled job for live operation

## License

MIT © 2026 Qian Yi
