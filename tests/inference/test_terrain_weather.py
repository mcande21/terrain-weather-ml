"""Tests for Task 9.5 — colorado-avalanche-ml integration contract.

Covers:
- Request with 5 points and 24-hour horizon returns 5 forecast arrays,
  each with 24 hourly entries containing all required fields
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch
from fastapi.testclient import TestClient

from terrain_weather_ml.inference.app import create_app
from terrain_weather_ml.inference.service import InferenceConfig


def _mock_model_fn(lat, lon, dt):
    return torch.tensor([3.0, 4.0, 268.5, 0.5])


def _mock_terrain_lookup(lat, lon):
    return {"elevation": 3200.0, "aspect": 180.0}


@pytest.fixture()
def config():
    return InferenceConfig(
        model_version="test-v1",
        checkpoint_hash="abc123",
    )


@pytest.fixture()
def client(config):
    app = create_app(
        config=config,
        model_fn=_mock_model_fn,
        terrain_lookup_fn=_mock_terrain_lookup,
    )
    with TestClient(app) as c:
        yield c


class TestTerrainWeatherEndpoint:
    """Task 9.5: POST /terrain-weather (colorado-avalanche-ml contract)."""

    def test_5_points_24h_returns_complete_forecasts(self, client):
        """5 points with 24h horizon returns 5 forecast arrays with 24 entries."""
        now = datetime.now(UTC)
        points = [
            {
                "latitude": 39.0 + i * 0.5,
                "longitude": -106.0,
                "elevation": 3000.0 + i * 100,
                "aspect": 180.0,
            }
            for i in range(5)
        ]

        resp = client.post("/terrain-weather", json={
            "points": points,
            "datetime": now.isoformat(),
            "horizon_hours": 24,
        })

        assert resp.status_code == 200
        data = resp.json()

        assert len(data["forecasts"]) == 5
        assert data["model_version"] == "test-v1"
        assert "source_hrrr_cycle" in data

        for i, forecast in enumerate(data["forecasts"]):
            assert forecast["latitude"] == points[i]["latitude"]
            assert forecast["longitude"] == points[i]["longitude"]
            assert forecast["elevation"] == points[i]["elevation"]
            assert forecast["aspect"] == points[i]["aspect"]
            assert len(forecast["hourly"]) == 24

            for entry in forecast["hourly"]:
                assert "datetime" in entry
                assert "temperature" in entry
                assert "wind_speed" in entry
                assert "wind_direction" in entry
                assert "precipitation_rate" in entry
                assert "surface_pressure" in entry

                # Values should be physically plausible (from mock)
                assert entry["wind_speed"] > 0
                assert 0 <= entry["wind_direction"] < 360
                assert entry["temperature"] > 0
                assert entry["surface_pressure"] > 0

    def test_single_point_short_horizon(self, client):
        """Single point with 1h horizon works."""
        now = datetime.now(UTC)
        resp = client.post("/terrain-weather", json={
            "points": [{
                "latitude": 39.5,
                "longitude": -106.0,
                "elevation": 3200.0,
                "aspect": 180.0,
            }],
            "datetime": now.isoformat(),
            "horizon_hours": 1,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["forecasts"]) == 1
        assert len(data["forecasts"][0]["hourly"]) == 1

    def test_response_includes_elevation_and_aspect(self, client):
        """Response includes elevation and aspect for downstream completeness."""
        now = datetime.now(UTC)
        resp = client.post("/terrain-weather", json={
            "points": [{
                "latitude": 39.5,
                "longitude": -106.0,
                "elevation": 3500.0,
                "aspect": 225.0,
            }],
            "datetime": now.isoformat(),
            "horizon_hours": 1,
        })

        assert resp.status_code == 200
        forecast = resp.json()["forecasts"][0]
        assert forecast["elevation"] == 3500.0
        assert forecast["aspect"] == 225.0
