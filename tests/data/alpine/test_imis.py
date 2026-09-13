"""Tests for Task 2.2: Swiss IMIS ingestion.

Verifies:
- API client with mocked responses returns expected station count (180)
- All expected variables present
- EnviDat fallback triggers when API returns 404
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from terrain_weather_ml.data.alpine.imis import (
    HTTPNotFoundError,
    IMISClient,
)
from terrain_weather_ml.data.alpine.schema import Network


def _make_synthetic_imis_stations(n: int = 180) -> list[dict[str, Any]]:
    """Create synthetic IMIS station metadata."""
    return [
        {
            "code": f"IMIS{i:04d}",
            "latitude": 46.0 + i * 0.01,
            "longitude": 8.0 + i * 0.005,
            "elevation": 2000.0 + i * 10,
        }
        for i in range(n)
    ]


def _make_synthetic_imis_observations(
    n_stations: int = 3,
    n_timesteps: int = 4,
) -> list[dict[str, Any]]:
    """Create synthetic IMIS observation data."""
    obs = []
    for i in range(n_stations):
        for t in range(n_timesteps):
            obs.append({
                "station_code": f"IMIS{i:04d}",
                "timestamp": datetime(2024, 1, 1, t // 2, (t % 2) * 30).isoformat(),
                "wind_speed": 5.0 + t * 0.5,
                "wind_direction": 270.0 + t,
                "air_temperature": -5.0 + t * 0.1,
                "snow_surface_temperature": -8.0 + t * 0.1,
                "snow_depth": 150.0 + t,
                "relative_humidity": 80.0 + t,
            })
    return obs


class MockHTTPClient:
    """Mock HTTP client that returns synthetic data."""

    def __init__(
        self,
        stations: list[dict[str, Any]] | None = None,
        observations: list[dict[str, Any]] | None = None,
        fail_api: bool = False,
    ):
        self.stations = stations or _make_synthetic_imis_stations()
        self.observations = observations or _make_synthetic_imis_observations()
        self.fail_api = fail_api
        self.get_calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        self.get_calls.append((url, params))

        if "stations" in url:
            return self.stations

        if "data" in url or "bulk" in url:
            if self.fail_api and "bulk" not in url:
                raise HTTPNotFoundError("404 Not Found")
            return self.observations

        return []


class TestIMISClient:
    """Tests for IMIS data ingestion."""

    def test_fetch_stations_returns_expected_count(self, tmp_path: Path):
        """Mocked API returns 180 stations."""
        mock = MockHTTPClient(stations=_make_synthetic_imis_stations(180))
        client = IMISClient(cache_dir=tmp_path, http_client=mock)
        stations = client.fetch_stations()
        assert len(stations) == 180

    def test_fetch_stations_all_fields_present(self, tmp_path: Path):
        """Every IMIS station has all required metadata fields."""
        mock = MockHTTPClient(stations=_make_synthetic_imis_stations(5))
        client = IMISClient(cache_dir=tmp_path, http_client=mock)
        stations = client.fetch_stations()

        for sid, meta in stations.items():
            assert meta.station_id == sid
            assert meta.network == Network.IMIS
            assert meta.latitude is not None
            assert meta.longitude is not None
            assert meta.elevation is not None
            assert meta.terrain_ref_lat is not None
            assert meta.terrain_ref_lon is not None

    def test_fetch_observations_returns_all_variables(self, tmp_path: Path):
        """Observations contain all expected IMIS variables."""
        mock = MockHTTPClient()
        client = IMISClient(cache_dir=tmp_path, http_client=mock)

        records = client.fetch_observations(
            datetime(2024, 1, 1), datetime(2024, 1, 2)
        )
        assert len(records) > 0

        rec = records[0]
        assert rec.network == Network.IMIS
        assert rec.wind_speed == pytest.approx(5.0, abs=1)
        assert rec.wind_direction == pytest.approx(270.0, abs=5)
        assert rec.temperature == pytest.approx(-5.0, abs=1)
        assert rec.snow_surface_temp == pytest.approx(-8.0, abs=1)
        assert rec.snow_depth == pytest.approx(150.0, abs=5)
        assert rec.humidity == pytest.approx(80.0, abs=5)

    def test_envidat_fallback_on_api_404(self, tmp_path: Path):
        """EnviDat fallback triggers when API returns 404."""
        mock = MockHTTPClient(fail_api=True)
        client = IMISClient(cache_dir=tmp_path, http_client=mock)

        records = client.fetch_observations(
            datetime(2020, 1, 1), datetime(2020, 1, 2)
        )

        # Should have called both API (failed) and EnviDat (succeeded)
        api_calls = [c for c in mock.get_calls if "bulk" not in c[0]]
        envidat_calls = [c for c in mock.get_calls if "bulk" in c[0]]
        assert len(api_calls) >= 1
        assert len(envidat_calls) == 1
        assert len(records) > 0

    def test_station_network_is_imis(self, tmp_path: Path):
        """All returned stations have IMIS network."""
        mock = MockHTTPClient()
        client = IMISClient(cache_dir=tmp_path, http_client=mock)
        stations = client.fetch_stations()
        for meta in stations.values():
            assert meta.network == Network.IMIS
