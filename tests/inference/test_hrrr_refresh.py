"""Tests for Task 9.4 — HRRR input refresh.

Covers:
- Refresh picks up a new HRRR cycle
- Fetch failure logs warning, predictions continue with stale cycle
- HRRR cycle identifier appears in prediction response
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch

from terrain_weather_ml.inference.service import InferenceConfig, InferenceService


def _mock_model_fn(lat, lon, dt):
    return torch.tensor([1.0, 1.0, 268.0, 0.001])


def _mock_terrain_lookup(lat, lon):
    return {"elevation": 3000.0, "aspect": 90.0}


@pytest.fixture()
def config():
    return InferenceConfig(
        model_version="test-v1",
        checkpoint_hash="abc123",
        hrrr_refresh_seconds=1,
    )


class TestHRRRRefresh:
    """Task 9.4: HRRR input refresh."""

    @pytest.mark.asyncio
    async def test_refresh_picks_up_new_cycle(self, config):
        """Successful HRRR fetch updates the active cycle."""
        def hrrr_fetcher():
            return "2024-01-15T12:00:00Z"

        service = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
            hrrr_fetch_fn=hrrr_fetcher,
        )

        assert service.current_hrrr_cycle == "unknown"
        result = await service.refresh_hrrr()
        assert result is True
        assert service.current_hrrr_cycle == "2024-01-15T12:00:00Z"

    @pytest.mark.asyncio
    async def test_fetch_failure_keeps_stale_cycle(self, config):
        """Failed HRRR fetch keeps the previous cycle and logs warning."""
        def failing_fetcher():
            raise ConnectionError("S3 unreachable")

        service = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
            hrrr_fetch_fn=failing_fetcher,
        )
        service.set_hrrr_cycle("2024-01-15T06:00:00Z")

        result = await service.refresh_hrrr()
        assert result is False
        # Stale cycle preserved
        assert service.current_hrrr_cycle == "2024-01-15T06:00:00Z"

    def test_hrrr_cycle_in_prediction_response(self, config):
        """Prediction response includes the active HRRR cycle."""
        service = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        service.set_hrrr_cycle("2024-01-15T12:00:00Z")

        now = datetime.now(UTC)
        pred = service.predict_single(39.5, -106.0, now)
        assert pred.source_hrrr_cycle == "2024-01-15T12:00:00Z"

    def test_predictions_continue_after_failed_refresh(self, config):
        """Service can still serve predictions after HRRR fetch failure."""
        service = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
            hrrr_fetch_fn=lambda: (_ for _ in ()).throw(OSError("Network down")),
        )
        service.set_hrrr_cycle("2024-01-15T06:00:00Z")

        # Prediction should work fine
        now = datetime.now(UTC)
        pred = service.predict_single(39.5, -106.0, now)
        assert pred.wind_speed > 0
        assert pred.source_hrrr_cycle == "2024-01-15T06:00:00Z"
