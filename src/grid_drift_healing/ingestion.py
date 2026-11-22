"""Data ingestion from the Open-Meteo archive API with schema validation, local caching
and a clearly-labelled offline fallback (synthetic generator).

Open-Meteo is free and needs no API key: https://open-meteo.com/en/docs/historical-weather-api
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_VARS = (
    "temperature_2m",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "relative_humidity_2m",
)

# Physical plausibility bounds used by the schema check.
BOUNDS = {
    "temperature_2m": (-60.0, 60.0),
    "cloud_cover": (0.0, 100.0),
    "shortwave_radiation": (0.0, 1500.0),
    "wind_speed_10m": (0.0, 200.0),
    "relative_humidity_2m": (0.0, 100.0),
}


class SchemaError(ValueError):
    """Raised when an upstream payload does not match the expected schema."""


@dataclass
class WeatherFrame:
    """Hourly weather data plus provenance. ``source`` is one of
    ``open-meteo``, ``open-meteo-cache`` or ``SYNTHETIC``."""

    df: pd.DataFrame
    source: str
    lat: float
    lon: float

    @property
    def is_synthetic(self) -> bool:
        return self.source == "SYNTHETIC"


def validate_payload(payload: dict, variables: tuple[str, ...] = DEFAULT_VARS) -> pd.DataFrame:
    """Validate a raw Open-Meteo JSON payload and return a tidy hourly DataFrame (UTC index)."""
    if not isinstance(payload, dict):
        raise SchemaError("payload is not a JSON object")
    if payload.get("error"):
        raise SchemaError(f"API error: {payload.get('reason')}")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise SchemaError("missing 'hourly.time'")
    missing = [v for v in variables if v not in hourly]
    if missing:
        raise SchemaError(f"missing hourly variables: {missing}")
    n = len(hourly["time"])
    for v in variables:
        if len(hourly[v]) != n:
            raise SchemaError(f"length mismatch for {v}: {len(hourly[v])} != {n}")
    idx = pd.to_datetime(hourly["time"], utc=True)
    df = pd.DataFrame({v: pd.to_numeric(pd.Series(hourly[v]), errors="coerce").values for v in variables},
                      index=idx)
    df.index.name = "time"
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise SchemaError("time index not strictly increasing")
    return clean_frame(df)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Clip to physical bounds (out-of-range -> NaN) and interpolate short gaps (<=3h)."""
    df = df.copy()
    for col, (lo, hi) in BOUNDS.items():
        if col in df:
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.any():
                log.warning("%s: %d out-of-range values set to NaN", col, int(bad.sum()))
            df.loc[bad, col] = np.nan
    df = df.interpolate(limit=3, limit_direction="both")
    nan_frac = float(df.isna().mean().max()) if len(df) else 1.0
    if nan_frac > 0.05:
        raise SchemaError(f"too many missing values after cleaning ({nan_frac:.1%})")
    return df.dropna()


def _cache_path(cache_dir: Path, lat, lon, start, end, variables) -> Path:
    key = json.dumps([round(lat, 4), round(lon, 4), start, end, list(variables)])
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    return cache_dir / f"openmeteo_{lat:.2f}_{lon:.2f}_{start}_{end}_{h}.csv"


def fetch_open_meteo(lat: float, lon: float, start: str, end: str,
                     variables: tuple[str, ...] = DEFAULT_VARS,
                     cache_dir: str | Path = "data/cache", timeout: float = 60.0) -> WeatherFrame:
    """Fetch hourly archive data from Open-Meteo, using a local CSV cache if present."""
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, lat, lon, start, end, variables)
    if path.exists():
        df = pd.read_csv(path, index_col="time", parse_dates=["time"])
        df.index = pd.to_datetime(df.index, utc=True)
        log.info("loaded %d rows from cache %s", len(df), path)
        return WeatherFrame(clean_frame(df), "open-meteo-cache", lat, lon)

    import requests

    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(variables), "timezone": "GMT"}
    r = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    df = validate_payload(r.json(), variables)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    log.info("fetched %d rows from Open-Meteo, cached to %s", len(df), path)
    return WeatherFrame(df, "open-meteo", lat, lon)


def synthetic_weather(start: str, end: str, lat: float = 31.82, lon: float = 117.23,
                      seed: int = 0) -> WeatherFrame:
    """Generate SYNTHETIC hourly weather (clear-sky geometry + AR(1) clouds).

    This is NOT real data. It exists so tests and demos run offline.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", tz="UTC")
    n = len(idx)
    doy = idx.day_of_year.values
    solar_hour = (idx.hour.values + lon / 15.0) % 24
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    latr = np.deg2rad(lat)
    hra = np.deg2rad(15 * (solar_hour - 12))
    cos_z = np.sin(latr) * np.sin(decl) + np.cos(latr) * np.cos(decl) * np.cos(hra)
    clear = 1000 * np.clip(cos_z, 0, None) ** 1.15
    cloud = np.empty(n)
    c = 50.0
    for i in range(n):
        c = 0.97 * c + 0.03 * 50 + rng.normal(0, 6)
        cloud[i] = c
    cloud = np.clip(cloud, 0, 100)
    ghi = clear * (1 - 0.75 * (cloud / 100) ** 3.4)
    temp = 16 + 11 * np.sin(2 * np.pi * (doy - 105) / 365) + 4 * np.clip(cos_z, 0, None) + rng.normal(0, 1, n)
    wind = np.abs(3 + rng.normal(0, 1.5, n))
    rh = np.clip(70 + 0.2 * cloud - 1.0 * (temp - 16) + rng.normal(0, 5, n), 0, 100)
    df = pd.DataFrame({"temperature_2m": temp, "cloud_cover": cloud, "shortwave_radiation": ghi,
                       "wind_speed_10m": wind, "relative_humidity_2m": rh}, index=idx)
    df.index.name = "time"
    return WeatherFrame(df, "SYNTHETIC", lat, lon)


def load_weather(lat: float, lon: float, start: str, end: str, cache_dir: str | Path = "data/cache",
                 offline: bool = False) -> WeatherFrame:
    """Try Open-Meteo (or cache); on any failure fall back to SYNTHETIC data with a loud warning."""
    if not offline:
        try:
            return fetch_open_meteo(lat, lon, start, end, cache_dir=cache_dir)
        except Exception as exc:  # network down, schema error, HTTP error...
            log.warning("Open-Meteo unavailable (%s); FALLING BACK TO SYNTHETIC DATA", exc)
    return synthetic_weather(start, end, lat, lon)
