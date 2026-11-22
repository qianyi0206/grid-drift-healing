import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_drift_healing import ingestion as ing

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "open_meteo_hefei_sample.json"


def _payload(n=48):
    t = pd.date_range("2025-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {"hourly": {"time": t, **{v: [10.0] * n for v in ing.DEFAULT_VARS}}}


def test_validate_ok():
    df = ing.validate_payload(_payload())
    assert list(df.columns) == list(ing.DEFAULT_VARS)
    assert len(df) == 48 and str(df.index.tz) == "UTC"


def test_validate_missing_var():
    p = _payload()
    del p["hourly"]["cloud_cover"]
    with pytest.raises(ing.SchemaError, match="cloud_cover"):
        ing.validate_payload(p)


def test_validate_length_mismatch():
    p = _payload()
    p["hourly"]["temperature_2m"] = p["hourly"]["temperature_2m"][:-1]
    with pytest.raises(ing.SchemaError, match="length mismatch"):
        ing.validate_payload(p)


def test_api_error_payload():
    with pytest.raises(ing.SchemaError, match="API error"):
        ing.validate_payload({"error": True, "reason": "bad"})


def test_out_of_range_interpolated():
    p = _payload()
    p["hourly"]["cloud_cover"][5] = 250.0
    df = ing.validate_payload(p)
    assert df["cloud_cover"].iloc[5] == pytest.approx(10.0)


def test_too_many_nans_rejected():
    p = _payload()
    p["hourly"]["temperature_2m"] = [None] * 48
    with pytest.raises(ing.SchemaError):
        ing.validate_payload(p)


def test_bundled_real_sample_validates():
    """Bundled sample is a genuine Open-Meteo archive response (Hefei, 7 days)."""
    df = ing.validate_payload(json.loads(SAMPLE.read_text()))
    assert len(df) == 7 * 24
    assert df["shortwave_radiation"].max() > 100


def test_cache_roundtrip(tmp_path, monkeypatch):
    calls = {"n": 0}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return _payload()

    def fake_get(*a, **k):
        calls["n"] += 1
        return R()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    a = ing.fetch_open_meteo(31.82, 117.23, "2025-01-01", "2025-01-02", cache_dir=tmp_path)
    b = ing.fetch_open_meteo(31.82, 117.23, "2025-01-01", "2025-01-02", cache_dir=tmp_path)
    assert calls["n"] == 1
    assert a.source == "open-meteo" and b.source == "open-meteo-cache"
    np.testing.assert_allclose(a.df.values, b.df.values)


def test_offline_fallback_is_labelled_synthetic(tmp_path, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", boom)
    wf = ing.load_weather(31.82, 117.23, "2025-01-01", "2025-01-10", cache_dir=tmp_path)
    assert wf.is_synthetic and wf.source == "SYNTHETIC"


def test_synthetic_shape(syn_weather):
    assert len(syn_weather) == 366 * 24
    assert (syn_weather["shortwave_radiation"] >= 0).all()
