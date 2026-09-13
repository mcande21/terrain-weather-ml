"""Task 1.6: Tensor assembly and caching tests.

Verify: full pipeline on a small DEM produces shape (17, H, W),
second call returns identical tensor from cache without recomputation.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from terrain_weather_ml.terrain.encoder import (
    TerrainFeatureEncoder,
    CHANNEL_NAMES,
    C_TERRAIN,
)


@pytest.fixture
def synthetic_dem_path(tmp_path):
    """Create a small synthetic GeoTIFF DEM in UTM coordinates."""
    filepath = tmp_path / "test_dem.tif"
    height, width = 40, 50
    res = 30.0  # 30m resolution
    # UTM zone 32N coordinates (Swiss Alps area)
    west = 345000.0
    south = 5107000.0
    east = west + width * res
    north = south + height * res

    y = np.linspace(0, 1, height)
    x = np.linspace(0, 1, width)
    xx, yy = np.meshgrid(x, y)
    elevation = 1000.0 + 500.0 * yy + 200.0 * np.exp(
        -((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.05
    )
    elevation = elevation.astype(np.float32)

    transform = from_bounds(west, south, east, north, width, height)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(32632),
        transform=transform,
    ) as dst:
        dst.write(elevation, 1)

    return filepath


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


class TestTerrainFeatureEncoder:
    def test_channel_count(self):
        """C_TERRAIN should be 17."""
        assert C_TERRAIN == 17

    def test_channel_names_match_count(self):
        """Channel names list should have C_TERRAIN entries."""
        assert len(CHANNEL_NAMES) == C_TERRAIN

    def test_channel_names_order(self):
        """Channel names should follow the specified order."""
        assert CHANNEL_NAMES[0] == "elevation"
        assert CHANNEL_NAMES[1] == "slope"
        assert CHANNEL_NAMES[2] == "aspect_sin"
        assert CHANNEL_NAMES[3] == "aspect_cos"
        assert CHANNEL_NAMES[4] == "plan_curvature"
        assert CHANNEL_NAMES[5] == "profile_curvature"
        assert CHANNEL_NAMES[6] == "svf"
        assert CHANNEL_NAMES[7] == "sx_n"
        assert CHANNEL_NAMES[14] == "sx_nw"
        assert CHANNEL_NAMES[15] == "tpi"
        assert CHANNEL_NAMES[16] == "roughness"

    def _make_encoder(self, cache_dir, **kwargs):
        """Create an encoder with small radii for fast test execution."""
        defaults = dict(
            cache_dir=cache_dir,
            svf_radius=300.0,
            svf_azimuths=8,
            sx_radius=300.0,
            tpi_radius=150.0,
        )
        defaults.update(kwargs)
        return TerrainFeatureEncoder(**defaults)

    def test_encode_produces_correct_shape(self, synthetic_dem_path, cache_dir):
        """Full pipeline should produce shape (C_TERRAIN, H, W)."""
        encoder = self._make_encoder(cache_dir)
        tensor = encoder.encode(synthetic_dem_path)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == C_TERRAIN
        assert tensor.ndim == 3

    def test_encode_all_channels_finite(self, synthetic_dem_path, cache_dir):
        """All channels should contain finite values (no NaN or Inf)."""
        encoder = self._make_encoder(cache_dir)
        tensor = encoder.encode(synthetic_dem_path)
        assert torch.isfinite(tensor).all()

    def test_encode_device_cpu(self, synthetic_dem_path, cache_dir):
        """Output tensor should be on CPU by default."""
        encoder = self._make_encoder(cache_dir, device="cpu")
        tensor = encoder.encode(synthetic_dem_path)
        assert tensor.device.type == "cpu"

    def test_cache_hit_no_recomputation(self, synthetic_dem_path, cache_dir):
        """Second call should return cached tensor without recomputation."""
        encoder = self._make_encoder(cache_dir)

        # First call: computes features
        tensor1 = encoder.encode(synthetic_dem_path)

        # Patch compute functions to detect if they're called
        with patch(
            "terrain_weather_ml.terrain.encoder.compute_slope",
            side_effect=AssertionError("Should not recompute"),
        ):
            # Second call: should hit cache
            tensor2 = encoder.encode(synthetic_dem_path)

        torch.testing.assert_close(tensor1, tensor2)

    def test_cache_persists_to_disk(self, synthetic_dem_path, cache_dir):
        """Cached tensor should be retrievable from a new encoder instance."""
        encoder1 = self._make_encoder(cache_dir)
        tensor1 = encoder1.encode(synthetic_dem_path)

        # New encoder instance should find the cache on disk
        encoder2 = self._make_encoder(cache_dir)
        with patch(
            "terrain_weather_ml.terrain.encoder.compute_slope",
            side_effect=AssertionError("Should not recompute"),
        ):
            tensor2 = encoder2.encode(synthetic_dem_path)

        torch.testing.assert_close(tensor1, tensor2)

    def test_no_nlcd_uses_default_roughness(self, synthetic_dem_path, cache_dir):
        """Without NLCD input, roughness channel uses default value."""
        encoder = self._make_encoder(cache_dir)
        tensor = encoder.encode(synthetic_dem_path)
        # Roughness channel (index 16) should be uniform default
        roughness = tensor[16]
        assert roughness.std() < 1e-6  # all same value
