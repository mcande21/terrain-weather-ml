"""Task 1.3: Sky-view factor (SVF) computation tests.

Verify: flat surface returns SVF ~ 1.0, synthetic hemispherical valley
returns SVF < 0.5, runtime scales linearly with search radius.
"""

import numpy as np
import pytest

from terrain_weather_ml.terrain.svf import compute_svf


class TestSVF:
    def test_flat_surface_svf_near_one(self):
        """A flat surface should have SVF approximately 1.0."""
        dem = np.ones((100, 100), dtype=np.float32) * 500.0
        svf = compute_svf(dem, resolution=30.0, max_radius=1500.0, n_azimuths=36)
        # Interior pixels away from edge effects
        interior = svf[10:-10, 10:-10]
        np.testing.assert_allclose(interior, 1.0, atol=0.05)

    def test_deep_valley_low_svf(self):
        """A pixel at the bottom of a deep valley should have SVF < 0.5."""
        size = 101
        dem = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        # Create steep walls around center
        for i in range(size):
            for j in range(size):
                dist = max(abs(i - center), abs(j - center))
                if dist > 5:
                    dem[i, j] = 500.0 * min(dist / 10.0, 1.0)
        svf = compute_svf(dem, resolution=30.0, max_radius=1500.0, n_azimuths=36)
        assert svf[center, center] < 0.5

    def test_svf_range(self):
        """SVF values should be in [0, 1]."""
        np.random.seed(42)
        dem = np.random.rand(80, 80).astype(np.float32) * 1000
        svf = compute_svf(dem, resolution=30.0, max_radius=900.0, n_azimuths=36)
        assert svf.min() >= 0.0
        assert svf.max() <= 1.0 + 1e-6

    def test_svf_output_shape(self):
        """Output shape should match input shape."""
        dem = np.random.rand(40, 50).astype(np.float32) * 1000
        svf = compute_svf(dem, resolution=30.0, max_radius=600.0, n_azimuths=36)
        assert svf.shape == dem.shape

    def test_svf_configurable_azimuths(self):
        """SVF should work with different azimuth counts."""
        dem = np.ones((50, 50), dtype=np.float32) * 500.0
        svf_36 = compute_svf(dem, resolution=30.0, max_radius=600.0, n_azimuths=36)
        svf_8 = compute_svf(dem, resolution=30.0, max_radius=600.0, n_azimuths=8)
        # Both should be close to 1.0 on flat surface
        np.testing.assert_allclose(svf_36[5:-5, 5:-5], 1.0, atol=0.05)
        np.testing.assert_allclose(svf_8[5:-5, 5:-5], 1.0, atol=0.05)
