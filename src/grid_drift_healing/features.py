"""Target construction and feature engineering.

Target: hourly AC output (kW) of a *virtual* PV plant computed from irradiance and
temperature with a PVWatts-style model. Open-Meteo does not publish plant telemetry, so
this physics-derived proxy is the forecasting target in the MVP; real SCADA/ENTSO-E/EIA
series can be dropped in via the same ``target`` column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# shortwave_radiation is deliberately NOT a feature: the target is derived from it (leakage).
COVARIATES = ["temperature_2m", "cloud_cover", "wind_speed_10m", "relative_humidity_2m"]
LAGS = (24, 48, 168)


def pv_power(weather: pd.DataFrame, capacity_kw: float = 1000.0, gamma: float = -0.004,
             noct: float = 45.0, system_losses: float = 0.14) -> pd.Series:
    """PVWatts-like DC->AC proxy: P = C * G/1000 * (1 + gamma*(Tcell-25)) * (1-losses)."""
    g = weather["shortwave_radiation"].clip(lower=0)
    t_cell = weather["temperature_2m"] + g / 800.0 * (noct - 20.0)
    p = capacity_kw * g / 1000.0 * (1 + gamma * (t_cell - 25.0)) * (1 - system_losses)
    return p.clip(lower=0, upper=capacity_kw).rename("target")


def clear_sky_ghi(index: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Deterministic clear-sky GHI (W/m2) from solar geometry only (no measured data)."""
    doy = index.day_of_year.values
    solar_hour = (index.hour.values + 0.5 + lon / 15.0) % 24  # hour-ending -> mid-hour
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    latr = np.deg2rad(lat)
    hra = np.deg2rad(15 * (solar_hour - 12))
    cos_z = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(hra)
    return 1000 * np.clip(cos_z, 0, None) ** 1.15


def build_features(df: pd.DataFrame, target: str = "target", lat: float = 31.82,
                   lon: float = 117.23) -> pd.DataFrame:
    """Lagged target (>=24h, so valid for day-ahead issue), calendar + clear-sky geometry,
    and weather covariates (cloud, temperature, wind, humidity).

    Covariates at time t are used as a *perfect-prognosis* stand-in for NWP forecasts.
    """
    out = pd.DataFrame(index=df.index)
    out["clear_sky_ghi"] = clear_sky_ghi(df.index, lat, lon)
    for lag in LAGS:
        out[f"lag_{lag}"] = df[target].shift(lag)
    out["roll24_mean_lag24"] = df[target].shift(24).rolling(24).mean()
    hour = df.index.hour
    doy = df.index.day_of_year
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in COVARIATES:
        if c in df:
            out[c] = df[c]
    return out


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return list(feats.columns)
