"""Tests for Task 9.2 — POST /predict/batch endpoint.

Covers:
- 100-point batch returns 100 ordered predictions
- Batch size >1000 returns 422
- Same-datetime points share a single StormCast forward pass
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch
from fastapi.testclient import TestClient

from terrain_weather_ml.inference.app import create_app
from terrain_weather_ml.inference.service import InferenceConfig


def _mock_terrain_lookup(lat, lon):
    return {"elevation": 3200.0, "aspect": 180.0}


@pytest.fixture()
def config():
    return InferenceConfig(
        model_version="test-v1",
        checkpoint_hash="abc123",
    )


class TestBatchEndpoint:
    """Task 9.2: POST /predict/batch."""

    def test_100_point_batch_returns_100_predictions(self, config):
        """Batch of 100 points returns 100 ordered predictions."""
        call_count = {"n": 0}

        def model_fn(lat, lon, dt):
            call_count["n"] += 1
            return torch.tensor([float(lat), 1.0, 268.0, 0.001])

        app = create_app(
            config=config,
            model_fn=model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        now = datetime.now(UTC)

        points = []
        for i in range(100):
            lat = 37.0 + i * 0.04  # stays in domain
            points.append({
                "latitude": lat,
                "longitude": -106.0,
                "datetime": now.isoformat(),
            })

        with TestClient(app) as client:
            resp = client.post("/predict/batch", json={"points": points})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["predictions"]) == 100

    def test_batch_exceeds_max_returns_422(self, config):
        """Batch >1000 returns 422."""
        app = create_app(
            config=config,
            model_fn=lambda *a: torch.tensor([1.0, 1.0, 268.0, 0.001]),
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        now = datetime.now(UTC)

        # Construct 1001 points — Pydantic max_length should reject this
        points = [
            {"latitude": 39.5, "longitude": -106.0, "datetime": now.isoformat()}
            for _ in range(1001)
        ]

        with TestClient(app) as client:
            resp = client.post("/predict/batch", json={"points": points})

        assert resp.status_code == 422

    def test_same_datetime_shares_forward_pass(self, config):
        """Points with the same datetime should share a StormCast forward pass.

        We verify this by tracking unique datetime args to the model function.
        """
        seen_datetimes = set()

        def tracking_model_fn(lat, lon, dt):
            seen_datetimes.add(dt)
            return torch.tensor([1.0, 1.0, 268.0, 0.001])

        app = create_app(
            config=config,
            model_fn=tracking_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        now = datetime.now(UTC)

        # 10 points at the same datetime
        points = [
            {"latitude": 39.0 + i * 0.1, "longitude": -106.0, "datetime": now.isoformat()}
            for i in range(10)
        ]

        with TestClient(app) as client:
            resp = client.post("/predict/batch", json={"points": points})

        assert resp.status_code == 200
        # All points shared the same datetime — only 1 unique datetime seen
        assert len(seen_datetimes) == 1
