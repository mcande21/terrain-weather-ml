"""Task 1.5: TPI and surface roughness tests.

Verify: TPI on synthetic peak/valley DEM, NLCD mapping produces valid z0 values,
missing-NLCD fallback triggers warning and uses default.
"""

import logging

import numpy as np
import pytest

from terrain_weather_ml.terrain.tpi import compute_tpi
from terrain_weather_ml.terrain.roughness import (
    compute_roughness,
    NLCD_ROUGHNESS_TABLE,
)


class TestTPI:
    def test_peak_positive_tpi(self):
        """A peak pixel should have positive TPI (higher than surroundings)."""
        size = 101
        center = size // 2
        dem = np.zeros((size, size), dtype=np.float32)
        # Gaussian peak at center
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                dem[i, j] = 500.0 * np.exp(-(dist**2) / (2 * 10**2))
        tpi = compute_tpi(dem, resolution=30.0, radius=500.0)
        assert tpi[center, center] > 0

    def test_valley_negative_tpi(self):
        """A valley-bottom pixel should have negative TPI."""
        size = 101
        center = size // 2
        dem = np.ones((size, size), dtype=np.float32) * 500.0
        # Valley at center
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                dem[i, j] = 500.0 - 300.0 * np.exp(-(dist**2) / (2 * 10**2))
        tpi = compute_tpi(dem, resolution=30.0, radius=500.0)
        assert tpi[center, center] < 0

    def test_flat_surface_zero_tpi(self):
        """A flat surface should have TPI approximately 0."""
        dem = np.ones((80, 80), dtype=np.float32) * 500.0
        tpi = compute_tpi(dem, resolution=30.0, radius=500.0)
        interior = tpi[10:-10, 10:-10]
        np.testing.assert_allclose(interior, 0.0, atol=1e-3)

    def test_tpi_output_shape(self):
        """Output shape should match input shape."""
        dem = np.random.rand(30, 40).astype(np.float32) * 1000
        tpi = compute_tpi(dem, resolution=30.0, radius=500.0)
        assert tpi.shape == dem.shape

    def test_tpi_units_meters(self):
        """TPI should be in meters (elevation minus mean within radius)."""
        size = 81
        center = size // 2
        dem = np.ones((size, size), dtype=np.float32) * 100.0
        # Raise center pixel by 50m
        dem[center, center] = 150.0
        tpi = compute_tpi(dem, resolution=30.0, radius=500.0)
        # TPI at center should be close to 50 (but mean includes neighbors)
        assert tpi[center, center] > 40


class TestRoughness:
    def test_nlcd_roughness_table_exists(self):
        """NLCD roughness lookup table should map class IDs to z0 values."""
        assert len(NLCD_ROUGHNESS_TABLE) > 0
        for class_id, z0 in NLCD_ROUGHNESS_TABLE.items():
            assert isinstance(class_id, int)
            assert z0 >= 0

    def test_nlcd_mapping(self):
        """NLCD raster should be mapped to roughness z0 values."""
        nlcd = np.array([[11, 21, 41, 52, 71]], dtype=np.int32)
        roughness = compute_roughness(nlcd)
        assert roughness.shape == nlcd.shape
        assert roughness.dtype == np.float32
        # Each value should be a valid z0
        for val in roughness.flat:
            assert val >= 0

    def test_missing_nlcd_fallback(self, caplog):
        """When NLCD is None, should use default roughness and log warning."""
        with caplog.at_level(logging.WARNING):
            roughness = compute_roughness(None, shape=(50, 60), default_z0=0.1)
        assert roughness.shape == (50, 60)
        np.testing.assert_allclose(roughness, 0.1)
        assert any("roughness" in msg.lower() or "nlcd" in msg.lower()
                    for msg in caplog.messages)

    def test_unknown_nlcd_class_uses_default(self, caplog):
        """Unknown NLCD class IDs should fall back to default z0."""
        nlcd = np.array([[999]], dtype=np.int32)
        with caplog.at_level(logging.WARNING):
            roughness = compute_roughness(nlcd, default_z0=0.5)
        assert roughness[0, 0] == pytest.approx(0.5, abs=1e-6)
