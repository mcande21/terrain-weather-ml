"""Tests for Task 9.1 — POST /predict endpoint.

Covers:
- Valid request returns all response fields
- Out-of-domain coordinates return 422 with bounds info
- Future datetime beyond 18h returns 422 with horizon limit
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch
from fastapi.testclient import TestClient

from terrain_weather_ml.inference.app import create_app
from terrain_weather_ml.inference.service import InferenceConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_model_fn(lat, lon, dt):
    """Return a shaped tensor: (u, v, temperature, precipitation)."""
    return torch.tensor([3.0, 4.0, 268.5, 0.5])


def _mock_terrain_lookup(lat, lon):
    """Return terrain features for any point."""
    return {"elevation": 3200.0, "aspect": 180.0}


@pytest.fixture()
def config():
    return InferenceConfig(
        model_version="test-v1",
        checkpoint_hash="abc123",
        domain_bounds={
            "lat_min": 36.5,
            "lat_max": 41.5,
            "lon_min": -109.5,
            "lon_max": -103.5,
        },
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPredictEndpoint:
    """Task 9.1: POST /predict."""

    def test_valid_request_returns_all_fields(self, client):
        """Valid (lat, lon, datetime) returns complete prediction."""
        now = datetime.now(UTC)
        resp = client.post("/predict", json={
            "latitude": 39.5,
            "longitude": -106.0,
            "datetime": now.isoformat(),
        })
        assert resp.status_code == 200
        data = resp.json()

        # All required fields present
        assert "wind_speed" in data
        assert "wind_direction" in data
        assert "temperature" in data
        assert "precipitation_rate" in data
        assert "elevation" in data
        assert "aspect" in data
        assert "model_version" in data
        assert "source_hrrr_cycle" in data

        # Values from mock model (u=3, v=4 -> speed=5)
        assert pytest.approx(data["wind_speed"], rel=1e-3) == 5.0
        assert data["temperature"] == 268.5
        assert data["precipitation_rate"] == 0.5
        assert data["elevation"] == 3200.0
        assert data["aspect"] == 180.0
        assert data["model_version"] == "test-v1"

    def test_out_of_domain_latitude(self, client):
        """Latitude outside domain returns 422 with bounds."""
        now = datetime.now(UTC)
        resp = client.post("/predict", json={
            "latitude": 50.0,
            "longitude": -106.0,
            "datetime": now.isoformat(),
        })
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "50.0" in detail
        assert "36.5" in detail
        assert "41.5" in detail

    def test_out_of_domain_longitude(self, client):
        """Longitude outside domain returns 422 with bounds."""
        now = datetime.now(UTC)
        resp = client.post("/predict", json={
            "latitude": 39.5,
            "longitude": -120.0,
            "datetime": now.isoformat(),
        })
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "-120.0" in detail

    def test_future_datetime_beyond_horizon(self, client):
        """Datetime >18h in future returns 422 with horizon."""
        future = datetime.now(UTC) + timedelta(hours=20)
        resp = client.post("/predict", json={
            "latitude": 39.5,
            "longitude": -106.0,
            "datetime": future.isoformat(),
        })
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "18" in detail

    def test_past_datetime_is_valid(self, client):
        """A past datetime should be accepted (hindcast)."""
        past = datetime.now(UTC) - timedelta(hours=6)
        resp = client.post("/predict", json={
            "latitude": 39.5,
            "longitude": -106.0,
            "datetime": past.isoformat(),
        })
        assert resp.status_code == 200
