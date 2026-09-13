"""Tests for Tasks 5.1-5.3 -- Inference NWP backbone refactor.

Covers:
- 5.1: InferenceConfig.backbone defaults to "nwp", service uses
       NWPPassthroughAdapter by default, imports from backbone package.
- 5.2: InferenceConfig(backbone="stormcast") selects StormCast adapter
       with experimental flag; "nwp" selects passthrough.
- 5.3: Default mode requires only HRRR (no ERA5 dependency).
       /health reports active backbone and data source.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

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
        checkpoint_hash="abc123",
    )


# ---------------------------------------------------------------------------
# 5.1 -- Default to NWPPassthroughAdapter
# ---------------------------------------------------------------------------

class TestDefaultBackbone:
    """Task 5.1: Service defaults to NWP passthrough adapter."""

    def test_config_defaults_to_nwp(self):
        """InferenceConfig.backbone defaults to 'nwp'."""
        cfg = InferenceConfig()
        assert cfg.backbone == "nwp"

    def test_service_uses_passthrough_adapter_by_default(self, config):
        """Service instantiated with default config uses NWP passthrough."""
        svc = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        assert svc.backbone_name == "nwp"

    def test_service_adapter_is_passthrough_instance(self, config):
        """The adapter on the service is an NWPPassthroughAdapter."""
        from terrain_weather_ml.backbone import NWPPassthroughAdapter

        svc = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        assert isinstance(svc.adapter, NWPPassthroughAdapter)


# ---------------------------------------------------------------------------
# 5.2 -- StormCast experimental option
# ---------------------------------------------------------------------------

class _MockStormCastAdapter:
    """Lightweight stand-in for StormCast when no checkpoint is available."""
    experimental = True

    def extract(self, nwp_data):
        return torch.zeros(1, 6, 4, 4)


def _mock_create_adapter(backbone, **kwargs):
    """Mock factory that returns a stub for stormcast."""
    from terrain_weather_ml.backbone import NWPPassthroughAdapter

    if backbone == "passthrough":
        return NWPPassthroughAdapter()
    if backbone == "stormcast":
        return _MockStormCastAdapter()
    raise ValueError(f"Unknown backbone: {backbone!r}")


class TestStormCastExperimentalOption:
    """Task 5.2: backbone='stormcast' selects StormCast adapter."""

    def test_stormcast_backbone_selection(self):
        """InferenceConfig(backbone='stormcast') accepted."""
        cfg = InferenceConfig(backbone="stormcast")
        assert cfg.backbone == "stormcast"

    @patch(
        "terrain_weather_ml.inference.service.create_adapter",
        side_effect=_mock_create_adapter,
    )
    def test_stormcast_service_adapter(self, _mock):
        """Service with backbone='stormcast' uses StormCastAdapter."""
        cfg = InferenceConfig(backbone="stormcast")
        svc = InferenceService(
            cfg,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        assert svc.backbone_name == "stormcast"

    def test_nwp_explicit_selection(self):
        """InferenceConfig(backbone='nwp') explicitly selects passthrough."""
        cfg = InferenceConfig(backbone="nwp")
        svc = InferenceService(
            cfg,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        assert svc.backbone_name == "nwp"

    def test_invalid_backbone_raises(self):
        """Unknown backbone name raises ValueError."""
        cfg = InferenceConfig(backbone="invalid")
        with pytest.raises(ValueError, match="Unknown backbone"):
            InferenceService(
                cfg,
                model_fn=_mock_model_fn,
                terrain_lookup_fn=_mock_terrain_lookup,
            )


# ---------------------------------------------------------------------------
# 5.3 -- Health reports backbone and data source
# ---------------------------------------------------------------------------

class TestHealthBackboneReporting:
    """Task 5.3: Health endpoint reports active backbone and data source."""

    def test_health_reports_backbone_nwp(self, config):
        """Health response includes backbone='nwp' by default."""
        app = create_app(
            config=config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        with TestClient(app) as client:
            # Trigger model load first
            now = datetime.now(UTC)
            client.post("/predict", json={
                "latitude": 39.5,
                "longitude": -106.0,
                "datetime": now.isoformat(),
            })
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["backbone"] == "nwp"
            assert data["data_source"] == "hrrr"

    @patch(
        "terrain_weather_ml.inference.service.create_adapter",
        side_effect=_mock_create_adapter,
    )
    def test_health_reports_backbone_stormcast(self, _mock):
        """Health with backbone='stormcast' reports stormcast + hrrr+era5."""
        cfg = InferenceConfig(backbone="stormcast")
        app = create_app(
            config=cfg,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        with TestClient(app) as client:
            now = datetime.now(UTC)
            client.post("/predict", json={
                "latitude": 39.5,
                "longitude": -106.0,
                "datetime": now.isoformat(),
            })
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["backbone"] == "stormcast"
            assert "era5" in data["data_source"]

    def test_default_mode_no_era5(self, config):
        """Default (nwp) mode data_source is 'hrrr' only, no ERA5."""
        svc = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        assert svc.data_source == "hrrr"

    def test_metadata_includes_backbone(self, config):
        """Metadata endpoint includes backbone info."""
        svc = InferenceService(
            config,
            model_fn=_mock_model_fn,
            terrain_lookup_fn=_mock_terrain_lookup,
        )
        meta = svc.get_metadata()
        assert meta["backbone"] == "nwp"
        assert meta["data_source"] == "hrrr"
