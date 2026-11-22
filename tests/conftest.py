import pytest

from grid_drift_healing.ingestion import synthetic_weather


@pytest.fixture(scope="session")
def syn_weather():
    return synthetic_weather("2024-01-01", "2024-12-31", seed=1).df
