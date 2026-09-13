"""Tests for Task 9.3 — Model loading, /health, and /metadata endpoints.

Covers:
- First request triggers model load (lazy initialization)
- /health returns 503 before load, 200 after
- /metadata returns all required fields
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch
from fastapi.testclient import TestClient

from terrain_weather_ml.inference.app import create_app
from terrain_weather_ml.inference.service import InferenceConfig, InferenceService


def _mock_model_fn(lat, lon, dt):
    return torch.tensor([1.0, 1.0, 268.0, 0.001])


def _mock_terrain_lookup(lat, lon):
    return {"elevation": 3000.0, "aspect": 90.0}


@pytest.fixture()
def config():
    return InferenceConfig(
        model_version="test-v1",
        checkpoint_hash="abc123def456",
        domain_bounds={
            "lat_min": 36.5,
            "lat_max": 41.5,
            "lon_min": -109.5,
            "lon_max": -103.5,
        },
    )


class TestHealthAndMetadata:
    """Task 9.3: Model loading, /health, /metadata."""

    def test_health_returns_503_before_model_load(self, config):
        """Before any prediction, model is not loaded -> 503."""
        service = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        # Model not loaded yet
        assert not service.is_ready

        # Build app manually to test pre-load state
        app = create_app(
            config=config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        with TestClient(app) as client:
            # Before any prediction, model not loaded
            resp = client.get("/health")
            assert resp.status_code == 503

    def test_health_returns_200_after_prediction(self, config):
        """After a prediction triggers model load, /health returns 200."""
        app = create_app(
            config=config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        now = datetime.now(UTC)

        with TestClient(app) as client:
            # Trigger model load with a prediction
            client.post("/predict", json={
                "latitude": 39.5,
                "longitude": -106.0,
                "datetime": now.isoformat(),
            })

            # Now health should be 200
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"

    def test_first_request_triggers_model_load(self, config):
        """Lazy init: model loads on first prediction request."""
        service = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        assert not service.is_ready

        # First prediction triggers load
        now = datetime.now(UTC)
        service.predict_single(39.5, -106.0, now)
        assert service.is_ready

    def test_metadata_returns_all_fields(self, config):
        """GET /metadata returns all required metadata fields."""
        app = create_app(
            config=config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )

        with TestClient(app) as client:
            resp = client.get("/metadata")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] == "test-v1"
        assert data["checkpoint_hash"] == "abc123def456"
        assert "domain_bounds" in data
        assert data["domain_bounds"]["lat_min"] == 36.5
        assert data["max_forecast_hours"] == 18
        assert isinstance(data["output_variables"], list)
        assert len(data["output_variables"]) >= 4

        # Each variable has name and unit
        for var in data["output_variables"]:
            assert "name" in var
            assert "unit" in var
